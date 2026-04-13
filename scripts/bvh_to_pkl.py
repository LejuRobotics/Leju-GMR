# Standard library
import argparse
import json
import os
import pathlib
import pickle
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# Ensure the repository root is available for local imports.
CURRENT_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Third-party
import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local modules
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.human_key_normalizer import (
    ensure_required_human_keys_kuavo_s52,
    ensure_required_human_keys_kuavo_s54,
    ensure_required_human_keys_roban_s14,
    ensure_required_human_keys_roban_s17,
    normalize_human_frame_keys_leju_common,
)
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, qmai_read_bvh

HERE = pathlib.Path(__file__).parent

# ============================================================
# Robot Adapter Architecture (for maintainability)
#
# How to add a new robot model (example: kuavo_s52):
# 1) Implement build_preprocessor_kuavo_s52(...)
# 2) Implement load_motion_data_kuavo_s52(...)
# 3) Implement validate_qpos_kuavo_s52(...)
# 4) (Optional) Implement postprocess_qpos_kuavo_s52(...)
# 5) Register it in ROBOT_ADAPTERS
# 6) Add robot name to argparse choices
#
# Design rule:
# - Keep visualization flow generic in main loop.
# - Put all robot-specific logic into robot-specific functions.
# ============================================================


@dataclass(frozen=True)
class RobotAdapter:
    """Container for robot-specific retargeting behaviors."""
    build_preprocessor: Callable[[str, pathlib.Path, float], Any]
    load_motion_data: Callable[[str, str, float], tuple[list[dict], float]]
    validate_qpos: Callable[[np.ndarray], None]
    postprocess_qpos: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FpsPlan:
    """Resolved FPS policy for loading, playback pacing, and output metadata."""
    source_bvh_fps: float
    target_motion_fps: float
    output_pkl_fps: float
    viewer_rate_limit_enabled: bool


@dataclass(frozen=True)
class BvhUnitPlan:
    """Resolved BVH length unit and scale factor that converts to meters."""
    unit_name: str
    position_scale_to_meter: float


class PlaybackController:
    """Keyboard-driven playback state for pause/resume/step/quit."""

    def __init__(self) -> None:
        # Start in paused state for detailed visual inspection.
        self.is_paused = True
        self.step_once_requested = False
        self.stop_requested = False

    def handle_action(self, action: str, source: str = "viewer") -> None:
        """Handle normalized playback actions from viewer or terminal."""
        if action == "SPACE":
            self.is_paused = not self.is_paused
            state_text = "paused" if self.is_paused else "running"
            print(f"[Control] Playback {state_text} ({source}).")
            if self.is_paused:
                print_keyboard_help()
            return

        if action == "N":
            self.step_once_requested = True
            print(f"[Control] Step once requested ({source}).")
            return

        if action == "Q":
            self.stop_requested = True
            print(f"[Control] Stop requested ({source}).")
            return

    def handle_key(self, keycode: int) -> None:
        """Handle keyboard shortcuts from the MuJoCo viewer callback."""
        if keycode == ord(" "):
            self.handle_action("SPACE", source="viewer")
            return

        if keycode in (ord("n"), ord("N")):
            self.handle_action("N", source="viewer")
            return

        if keycode in (ord("q"), ord("Q")):
            self.handle_action("Q", source="viewer")
            return

    def consume_step_once(self) -> bool:
        """Consume single-step request and return whether to advance one frame."""
        if self.step_once_requested:
            self.step_once_requested = False
            return True
        return False


def print_keyboard_help() -> None:
    """Print runtime keyboard controls for playback debugging."""
    print("[Control] Keyboard shortcuts (viewer + terminal):")
    print("[Control]   Space: pause/resume")
    print("[Control]   N:     step one frame (when paused)")
    print("[Control]   Q:     quit")
    print("[Control] Initial state: paused")


class TerminalKeyReader:
    """Non-blocking terminal key reader with stable cbreak lifecycle."""

    def __init__(self) -> None:
        self._enabled = False
        self._stdin_file_descriptor: Optional[int] = None
        self._terminal_original_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self

        self._stdin_file_descriptor = sys.stdin.fileno()
        self._terminal_original_settings = termios.tcgetattr(self._stdin_file_descriptor)
        tty.setcbreak(self._stdin_file_descriptor)
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if (
            self._enabled
            and self._stdin_file_descriptor is not None
            and self._terminal_original_settings is not None
        ):
            termios.tcsetattr(
                self._stdin_file_descriptor,
                termios.TCSADRAIN,
                self._terminal_original_settings,
            )

    def read_action_nonblocking(self) -> Optional[str]:
        """Read one terminal action key without blocking."""
        if not self._enabled:
            return None

        ready_inputs, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready_inputs:
            return None

        key_char = sys.stdin.read(1)
        if key_char == " ":
            return "SPACE"
        if key_char in ("n", "N"):
            return "N"
        if key_char in ("q", "Q"):
            return "Q"
        return None


def resolve_fps_plan(
    bvh_fps: float,
    motion_fps: Optional[float],
    rate_limit_flag: Optional[bool],
) -> FpsPlan:
    """Resolve FPS policy from user arguments with deterministic precedence."""
    if bvh_fps <= 0:
        raise ValueError(f"Invalid --bvh_fps: {bvh_fps}. Must be > 0.")
    if motion_fps is not None and motion_fps <= 0:
        raise ValueError(f"Invalid --motion_fps: {motion_fps}. Must be > 0.")

    # If motion_fps is set, playback should follow motion_fps cadence regardless of rate_limit flag.
    if motion_fps is not None:
        return FpsPlan(
            source_bvh_fps=bvh_fps,
            target_motion_fps=motion_fps,
            output_pkl_fps=motion_fps,
            viewer_rate_limit_enabled=True,
        )

    # motion_fps not set -> use bvh fps as target.
    target_motion_fps = bvh_fps
    if rate_limit_flag is None:
        # Neither motion_fps nor rate_limit specified: pace by bvh fps.
        viewer_rate_limit_enabled = True
    else:
        # motion_fps not specified, explicit --rate_limit/--no-rate-limit should take effect.
        viewer_rate_limit_enabled = rate_limit_flag

    return FpsPlan(
        source_bvh_fps=bvh_fps,
        target_motion_fps=target_motion_fps,
        output_pkl_fps=target_motion_fps,
        viewer_rate_limit_enabled=viewer_rate_limit_enabled,
    )


def unit_name_to_scale_to_meter(unit_name: str) -> float:
    """Map BVH unit name to a factor that converts position values to meters."""
    unit_to_scale = {
        "mm": 1.0 / 1000.0,
        "cm": 1.0 / 100.0,
        "m": 1.0,
    }
    if unit_name not in unit_to_scale:
        raise ValueError(f"Unsupported bvh unit: {unit_name}")
    return unit_to_scale[unit_name]


HAND_END_EFFECTOR_KEY_CANDIDATES_COMMON = (
    "LeftHand",
    "RightHand",
    "LHand",
    "RHand",
)

FOOT_END_EFFECTOR_KEY_CANDIDATES_COMMON = (
    "LeftToeBase",
    "RightToeBase",
    "LeftToe",
    "RightToe",
    "LToe",
    "RToe",
    "LeftFoot",
    "RightFoot",
    "LFoot",
    "RFoot",
    "LeftFootMod",
    "RightFootMod",
    "LFootMod",
    "RFootMod",
)


def estimate_bvh_unit_from_first_frame_hand_foot_distance(
    motion_frames: list[dict],
) -> str:
    """
    Estimate BVH unit from first-frame hand-to-foot end-effector distances.

    Method:
    - Select the first frame.
    - Collect available hand end-effectors and foot end-effectors.
    - Compute pairwise distances between hands and feet.
    - Use the maximum distance as body-scale proxy.
    """
    if not motion_frames:
        return "cm"

    first_frame = motion_frames[0]
    hand_positions = [
        np.asarray(first_frame[key][0], dtype=np.float64)
        for key in HAND_END_EFFECTOR_KEY_CANDIDATES_COMMON
        if key in first_frame
    ]
    foot_positions = [
        np.asarray(first_frame[key][0], dtype=np.float64)
        for key in FOOT_END_EFFECTOR_KEY_CANDIDATES_COMMON
        if key in first_frame
    ]

    if not hand_positions or not foot_positions:
        return "cm"

    hand_foot_distances = [
        float(np.linalg.norm(hand_position - foot_position))
        for hand_position in hand_positions
        for foot_position in foot_positions
    ]
    max_hand_foot_distance = max(hand_foot_distances) if hand_foot_distances else 0.0

    # Thresholds separate common human-scale ranges:
    # m:   ~1-2.5
    # cm:  ~100-250
    # mm:  ~1000-2500
    if max_hand_foot_distance > 500.0:
        return "mm"
    if max_hand_foot_distance > 5.0:
        return "cm"
    return "m"


def resolve_bvh_unit_plan(
    bvh_unit_arg: str,
    bvh_format: str,
    motion_frames_for_estimation: list[dict],
) -> BvhUnitPlan:
    """Resolve final BVH unit setting from CLI argument and auto-estimation."""
    if bvh_unit_arg not in {"auto", "mm", "cm", "m"}:
        raise ValueError(f"Invalid --bvh_unit: {bvh_unit_arg}")

    if bvh_unit_arg == "auto":
        _ = bvh_format
        resolved_unit_name = estimate_bvh_unit_from_first_frame_hand_foot_distance(
            motion_frames_for_estimation
        )
    else:
        resolved_unit_name = bvh_unit_arg

    return BvhUnitPlan(
        unit_name=resolved_unit_name,
        position_scale_to_meter=unit_name_to_scale_to_meter(resolved_unit_name),
    )


def probe_motion_frames_for_unit_estimation(
    bvh_format: str,
    bvh_file_path: str,
) -> list[dict]:
    """Build raw-unit motion frames for BVH unit auto-estimation."""
    if bvh_format == "qmai":
        bvh_data = qmai_read_bvh(bvh_file_path)
    elif bvh_format in {"lafan1", "leju"}:
        bvh_data = read_bvh(bvh_file_path)
    else:
        return []

    global_data = utils.quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    rotation_quat_wxyz = np.array(
        [
            rotation_quat_xyzw[3],
            rotation_quat_xyzw[0],
            rotation_quat_xyzw[1],
            rotation_quat_xyzw[2],
        ]
    )

    probed_frames: list[dict] = []
    for frame_index in range(bvh_data.pos.shape[0]):
        frame_result: dict = {}
        for bone_index, bone_name in enumerate(bvh_data.bones):
            orientation = utils.quat_mul(rotation_quat_wxyz, global_data[0][frame_index, bone_index])
            position = global_data[1][frame_index, bone_index] @ rotation_matrix.T
            frame_result[bone_name] = (position, orientation)

        if "LFoot" in frame_result and "LToe" in frame_result:
            frame_result["LFootMod"] = (frame_result["LFoot"][0], frame_result["LToe"][1])
        if "RFoot" in frame_result and "RToe" in frame_result:
            frame_result["RFootMod"] = (frame_result["RFoot"][0], frame_result["RToe"][1])

        frame_result = normalize_human_frame_keys_leju_common(frame_result)
        probed_frames.append(frame_result)

    return probed_frames


def build_resampled_frame_indices(
    num_source_frames: int,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Build source-frame indices for time-scale-consistent resampling."""
    if num_source_frames <= 0:
        raise ValueError("No source frames available for resampling.")

    if abs(source_fps - target_fps) < 1e-9:
        return np.arange(num_source_frames, dtype=np.int64)

    duration_seconds = (num_source_frames - 1) / source_fps
    target_frame_count = int(round(duration_seconds * target_fps)) + 1
    target_frame_count = max(1, target_frame_count)

    raw_indices = np.linspace(0, num_source_frames - 1, target_frame_count)
    frame_indices = np.rint(raw_indices).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, num_source_frames - 1)
    return frame_indices


def fps_to_tag(fps_value: float) -> str:
    """Create a compact filename tag for fps."""
    rounded_fps = int(round(fps_value))
    return f"{rounded_fps}fps"


def build_output_pkl_path(
    save_path_arg: Optional[str],
    robot_name: str,
    bvh_file_path: str,
    output_fps: float,
) -> pathlib.Path:
    """Resolve the output pkl path from save_path argument."""
    bvh_stem_name = pathlib.Path(bvh_file_path).stem
    fps_tag = fps_to_tag(output_fps)

    # If save_path is not provided, use default output/<robot>/pkl/<bvh>.pkl.
    if save_path_arg is None or save_path_arg.strip() == "":
        return pathlib.Path("output") / robot_name / "pkl" / f"{bvh_stem_name}_{fps_tag}.pkl"

    save_path_obj = pathlib.Path(save_path_arg)
    if save_path_obj.suffix.lower() == ".pkl":
        return save_path_obj

    # If save_path is a directory, append <robot>/pkl/<bvh>.pkl.
    return save_path_obj / robot_name / "pkl" / f"{bvh_stem_name}_{fps_tag}.pkl"


def build_output_video_path(
    record_video_path_arg: Optional[str],
    robot_name: str,
    output_pkl_path: pathlib.Path,
) -> Optional[pathlib.Path]:
    """
    Resolve optional video output path.

    Rules:
    - If --record_video_path is not provided: no video is recorded.
    - If --record_video_path is provided without a value: use default
      output/<robot>/videos/<pkl_stem>.mp4.
    - If --record_video_path is provided with a value: use the given path.
    """
    if record_video_path_arg is None:
        return None

    if record_video_path_arg.strip() == "":
        return pathlib.Path("output") / robot_name / "videos" / f"{output_pkl_path.stem}.mp4"

    return pathlib.Path(record_video_path_arg)


REQUIRED_HUMAN_KEYS_ROBAN_S14 = (
    "Hips",
    "Spine",
    "LeftShoulder",
    "RightShoulder",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFootMod",
    "RightFootMod",
    "LeftToe",
    "RightToe",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)


def assert_required_human_keys_common(
    frame_data: Dict[str, tuple],
    required_keys: tuple[str, ...],
    context_name: str,
) -> None:
    """Validate required human keys and raise a clear error when keys are missing."""
    missing_keys = [key for key in required_keys if key not in frame_data]
    if missing_keys:
        available_keys_preview = list(frame_data.keys())[:30]
        raise ValueError(
            f"[{context_name}] Missing required human keys: {missing_keys}. "
            f"Available keys preview: {available_keys_preview}"
        )


def load_bvh_lafan1_common(bvh_file: str, position_scale_to_meter: float):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    # Convert from [x, y, z, w] to [w, x, y, z] format for scalar_first=True
    rotation_quat = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T * position_scale_to_meter
            result[bone] = (position, orientation)

        # Add modified foot pose
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])

        frames.append(result)
    
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height

def load_bvh_qmai_common(bvh_file: str, position_scale_to_meter: float):
    """Load qmai BVH file with format-specific common conversion."""
    data = qmai_read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    # Convert from [x, y, z, w] to [w, x, y, z] format for scalar_first=True
    rotation_quat = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T * position_scale_to_meter
            result[bone] = (position, orientation)

        # Add modified foot pose for dance_bj format
        if "LFoot" in result and "LToe" in result:
            result["LFootMod"] = (result["LFoot"][0], result["LToe"][1])
        if "RFoot" in result and "RToe" in result:
            result["RFootMod"] = (result["RFoot"][0], result["RToe"][1])
        frames.append(result)
    
    # Calculate human height
    if "Head" in result and "LFootMod" in result and "RFootMod" in result:
        human_height = result["Head"][0][2] - min(result["LFootMod"][0][2], result["RFootMod"][0][2])
    else:
        human_height = 1.75  # default value
    
    return frames, human_height


def convert_qpos_roban_s17_old_to_new_urdf(qpos):
    """
    将基于旧URDF（base_link=waist）的qpos转换为新URDF（base_link=torso）的qpos。
    
    旧URDF结构: base_link(waist) -> waist_yaw_joint(z轴) -> waist_yaw_link(torso)
    新URDF结构: base_link(torso) -> waist_yaw_joint(z轴) -> waist_yaw_link(waist)
    
    XML中的qpos布局:
      qpos[0:3]  = root position
      qpos[3:7]  = root rotation (wxyz)
      qpos[7:19] = 12个腿关节 (leg_l1~l6, leg_r1~r6)
      qpos[19]   = waist_yaw_joint
      qpos[20:28]= 8个手臂关节 (zarm_l1~l4, zarm_r1~r4)
      qpos[28:30]= 2个头部关节 (zhead_1, zhead_2)
    
    转换关系:
      新base_link_rot(torso) = 旧base_link_rot(waist) * Rz(waist_yaw_angle)
      新waist_yaw_angle = -旧waist_yaw_angle
    """
    WAIST_YAW_IDX = 19  # waist_yaw_joint 在 qpos 中的索引
    
    new_qpos = qpos.copy()
    
    # 提取旧的root rotation (wxyz格式) 和 waist_yaw角度
    old_root_rot_wxyz = qpos[3:7]  # [w, x, y, z]
    old_waist_yaw_angle = qpos[WAIST_YAW_IDX]
    
    # 将root rot从wxyz转为scipy的xyzw格式
    old_root_rot_xyzw = [old_root_rot_wxyz[1], old_root_rot_wxyz[2], old_root_rot_wxyz[3], old_root_rot_wxyz[0]]
    old_root_rot = R.from_quat(old_root_rot_xyzw)
    
    # 构造waist_yaw的旋转 Rz(theta)
    waist_yaw_rot = R.from_rotvec([0, 0, old_waist_yaw_angle])
    
    # 新的root_rot(torso) = 旧的root_rot(waist) * Rz(waist_yaw_angle)
    new_root_rot = old_root_rot * waist_yaw_rot
    
    # 转回wxyz格式
    new_root_rot_xyzw = new_root_rot.as_quat()  # [x, y, z, w]
    new_qpos[3] = new_root_rot_xyzw[3]  # w
    new_qpos[4] = new_root_rot_xyzw[0]  # x
    new_qpos[5] = new_root_rot_xyzw[1]  # y
    new_qpos[6] = new_root_rot_xyzw[2]  # z
    
    # 新的waist_yaw角度取反
    new_qpos[WAIST_YAW_IDX] = -old_waist_yaw_angle
    
    return new_qpos

def load_bvh_leju_common(bvh_file: str, position_scale_to_meter: float):
    """Load leju BVH file and normalize naming variants to canonical keys."""
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    # Convert from [x, y, z, w] to [w, x, y, z] format for scalar_first=True
    rotation_quat = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T * position_scale_to_meter
            result[bone] = (position, orientation)

        # Keep backward-compatible short-name FootMod generation.
        if "LFoot" in result and "LToe" in result:
            result["LFootMod"] = (result["LFoot"][0], result["LToe"][1])
        if "RFoot" in result and "RToe" in result:
            result["RFootMod"] = (result["RFoot"][0], result["RToe"][1])

        normalized_result = normalize_human_frame_keys_leju_common(result)
        frames.append(normalized_result)

    last_frame = frames[-1] if frames else {}
    # Calculate human height
    if "Head" in last_frame and "LFootMod" in last_frame and "RFootMod" in last_frame:
        human_height = last_frame["Head"][0][2] - min(
            last_frame["LFootMod"][0][2],
            last_frame["RFootMod"][0][2],
        )
    else:
        human_height = 1.75  # default value

    return frames, human_height


class RobanS17GMR(GeneralMotionRetargeting):
    """Robot-specific GMR implementation for roban_s17."""
    def __init__(
        self,
        actual_human_height: float = None,
        solver: str="daqp",
        damping: float=5e-1,
        verbose: bool=True,
        use_velocity_limit: bool=False,
        contact_sequence: Dict[str, np.ndarray] = None,
        ik_config_file: str = None,  # Optional external IK config path.
    ) -> None:
        # used for contact offset
        self.contact_sequence = contact_sequence
        self.previous_human_data = None

        # Load roban_s17 model.
        self.xml_file = str(HERE / ".." / "assets" / "biped_s17" / "xml" / "biped_s17.xml")

        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load roban_s17 IK config.
        if ik_config_file is None:
            ik_config_file = HERE / "biped_s17_qmai_retarget.json"
        with open(ik_config_file) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", ik_config_file)
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            # Use joint names (not motor names) for velocity limits
            # Exclude the dummy_to_base_link joint (free joint)
            VELOCITY_LIMITS = {}
            for joint_name in self.robot_dof_names.keys():
                if joint_name != 'dummy_to_base_link':
                    VELOCITY_LIMITS[joint_name] = 3*np.pi
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global


def _load_motion_data_common_by_format(
    bvh_format: str,
    bvh_file_path: str,
    position_scale_to_meter: float,
):
    """Load BVH motion by source format using current shared converters."""
    if bvh_format == "qmai":
        return load_bvh_qmai_common(
            bvh_file=bvh_file_path,
            position_scale_to_meter=position_scale_to_meter,
        )
    if bvh_format == "leju":
        return load_bvh_leju_common(
            bvh_file=bvh_file_path,
            position_scale_to_meter=position_scale_to_meter,
        )
    if bvh_format == "lafan1":
        return load_bvh_lafan1_common(
            bvh_file=bvh_file_path,
            position_scale_to_meter=position_scale_to_meter,
        )
    raise ValueError(f"Unsupported BVH format: {bvh_format}")


def load_motion_data_roban_s17(
    bvh_format: str,
    bvh_file_path: str,
    position_scale_to_meter: float,
):
    """
    roban_s17 specific motion-data loader.
    Template for new robot model:
    - keep function name as load_motion_data_<robot_name>
    - implement robot-specific data conversion if needed
    """
    motion_frames, actual_human_height = _load_motion_data_common_by_format(
        bvh_format,
        bvh_file_path,
        position_scale_to_meter,
    )
    normalized_motion_frames = [
        ensure_required_human_keys_roban_s17(frame_data)
        for frame_data in motion_frames
    ]
    return normalized_motion_frames, actual_human_height


def load_motion_data_kuavo_s54(
    bvh_format: str,
    bvh_file_path: str,
    position_scale_to_meter: float,
):
    """
    kuavo_s54 specific motion-data loader.
    Template for new robot model:
    - keep function name as load_motion_data_<robot_name>
    - implement robot-specific data conversion if needed
    """
    motion_frames, actual_human_height = _load_motion_data_common_by_format(
        bvh_format,
        bvh_file_path,
        position_scale_to_meter,
    )
    normalized_motion_frames = [
        ensure_required_human_keys_kuavo_s54(frame_data)
        for frame_data in motion_frames
    ]
    return normalized_motion_frames, actual_human_height


def load_motion_data_roban_s14(
    bvh_format: str,
    bvh_file_path: str,
    position_scale_to_meter: float,
):
    """
    roban_s14 specific motion-data loader.
    Template for new robot model:
    - keep function name as load_motion_data_<robot_name>
    - implement robot-specific data conversion if needed
    """
    motion_frames, actual_human_height = _load_motion_data_common_by_format(
        bvh_format,
        bvh_file_path,
        position_scale_to_meter,
    )

    normalized_motion_frames = [
        ensure_required_human_keys_roban_s14(frame_data)
        for frame_data in motion_frames
    ]

    if normalized_motion_frames:
        assert_required_human_keys_common(
            frame_data=normalized_motion_frames[0],
            required_keys=REQUIRED_HUMAN_KEYS_ROBAN_S14,
            context_name="roban_s14",
        )

    return normalized_motion_frames, actual_human_height


def load_motion_data_kuavo_s52(
    bvh_format: str,
    bvh_file_path: str,
    position_scale_to_meter: float,
):
    """
    kuavo_s52 specific motion-data loader.
    Template for new robot model:
    - keep function name as load_motion_data_<robot_name>
    - implement robot-specific data conversion if needed
    """
    motion_frames, actual_human_height = _load_motion_data_common_by_format(
        bvh_format,
        bvh_file_path,
        position_scale_to_meter,
    )

    normalized_motion_frames = [
        ensure_required_human_keys_kuavo_s52(frame_data)
        for frame_data in motion_frames
    ]
    return normalized_motion_frames, actual_human_height


def build_preprocessor_roban_s17(
    bvh_format: str,
    ik_config_file: pathlib.Path,
    actual_human_height: float,
):
    """
    roban_s17 specific preprocessor builder.
    Keep existing roban_s17 behavior unchanged to avoid regressions.
    """
    _ = bvh_format
    _ = actual_human_height
    return RobanS17GMR(
        actual_human_height=1.57,
        solver="daqp",
        damping=5e-1,
        verbose=True,
        use_velocity_limit=True,
        ik_config_file=str(ik_config_file),
    )


def build_preprocessor_kuavo_s54(
    bvh_format: str,
    ik_config_file: pathlib.Path,
    actual_human_height: float,
):
    """
    kuavo_s54 specific preprocessor builder.
    Template for new robot model:
    - keep function name as build_preprocessor_<robot_name>
    - configure solver/limits by robot model characteristics
    """
    _ = ik_config_file
    return GeneralMotionRetargeting(
        src_human=f"bvh_{bvh_format}",
        tgt_robot="kuavo_s54",
        actual_human_height=actual_human_height,
        solver="daqp",
        damping=5e-1,
        verbose=True,
        use_velocity_limit=False,
    )


def build_preprocessor_roban_s14(
    bvh_format: str,
    ik_config_file: pathlib.Path,
    actual_human_height: float,
):
    """
    roban_s14 specific preprocessor builder.
    """
    _ = ik_config_file
    return GeneralMotionRetargeting(
        src_human=f"bvh_{bvh_format}",
        tgt_robot="roban_s14",
        actual_human_height=actual_human_height,
        solver="daqp",
        damping=5e-1,
        verbose=True,
        use_velocity_limit=False,
    )


def build_preprocessor_kuavo_s52(
    bvh_format: str,
    ik_config_file: pathlib.Path,
    actual_human_height: float,
):
    """
    kuavo_s52 specific preprocessor builder.
    """
    _ = ik_config_file
    return GeneralMotionRetargeting(
        src_human=f"bvh_{bvh_format}",
        tgt_robot="kuavo_s52",
        actual_human_height=actual_human_height,
        solver="daqp",
        damping=5e-1,
        verbose=True,
        use_velocity_limit=False,
    )


def validate_qpos_roban_s17(qpos: np.ndarray) -> None:
    """roban_s17 specific qpos validation."""
    if qpos.ndim != 1:
        raise ValueError(f"[roban_s17] qpos must be 1D, got shape {qpos.shape}")
    if qpos.shape[0] < 30:
        raise ValueError(f"[roban_s17] qpos length is too small: {qpos.shape[0]}")


def validate_qpos_kuavo_s54(qpos: np.ndarray) -> None:
    """kuavo_s54 specific qpos validation."""
    if qpos.ndim != 1:
        raise ValueError(f"[kuavo_s54] qpos must be 1D, got shape {qpos.shape}")
    if qpos.shape[0] < 36:
        raise ValueError(f"[kuavo_s54] qpos length is too small: {qpos.shape[0]}")


def validate_qpos_roban_s14(qpos: np.ndarray) -> None:
    """roban_s14 specific qpos validation."""
    if qpos.ndim != 1:
        raise ValueError(f"[roban_s14] qpos must be 1D, got shape {qpos.shape}")
    # TODO: replace with exact expected dimension after confirming model.nq.
    if qpos.shape[0] < 20:
        raise ValueError(f"[roban_s14] qpos length is too small: {qpos.shape[0]}")


def validate_qpos_kuavo_s52(qpos: np.ndarray) -> None:
    """kuavo_s52 specific qpos validation."""
    if qpos.ndim != 1:
        raise ValueError(f"[kuavo_s52] qpos must be 1D, got shape {qpos.shape}")
    # TODO: replace with exact expected dimension after confirming model.nq.
    if qpos.shape[0] < 30:
        raise ValueError(f"[kuavo_s52] qpos length is too small: {qpos.shape[0]}")


def postprocess_qpos_roban_s17(qpos: np.ndarray) -> np.ndarray:
    """roban_s17 specific qpos postprocess hook."""
    return qpos


def postprocess_qpos_kuavo_s54(qpos: np.ndarray) -> np.ndarray:
    """kuavo_s54 specific qpos postprocess hook."""
    return qpos


def postprocess_qpos_roban_s14(qpos: np.ndarray) -> np.ndarray:
    """roban_s14 specific qpos postprocess hook."""
    return qpos


def postprocess_qpos_kuavo_s52(qpos: np.ndarray) -> np.ndarray:
    """kuavo_s52 specific qpos postprocess hook."""
    return qpos


ROBOT_ADAPTERS: Dict[str, RobotAdapter] = {
    "roban_s17": RobotAdapter(
        build_preprocessor=build_preprocessor_roban_s17,
        load_motion_data=load_motion_data_roban_s17,
        validate_qpos=validate_qpos_roban_s17,
        postprocess_qpos=postprocess_qpos_roban_s17,
    ),
    "kuavo_s54": RobotAdapter(
        build_preprocessor=build_preprocessor_kuavo_s54,
        load_motion_data=load_motion_data_kuavo_s54,
        validate_qpos=validate_qpos_kuavo_s54,
        postprocess_qpos=postprocess_qpos_kuavo_s54,
    ),
    "roban_s14": RobotAdapter(
        build_preprocessor=build_preprocessor_roban_s14,
        load_motion_data=load_motion_data_roban_s14,
        validate_qpos=validate_qpos_roban_s14,
        postprocess_qpos=postprocess_qpos_roban_s14,
    ),
    "kuavo_s52": RobotAdapter(
        build_preprocessor=build_preprocessor_kuavo_s52,
        load_motion_data=load_motion_data_kuavo_s52,
        validate_qpos=validate_qpos_kuavo_s52,
        postprocess_qpos=postprocess_qpos_kuavo_s52,
    ),
}


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from general_motion_retargeting import RobotMotionViewer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--format",
        choices=["lafan1", "qmai", "leju", "nokov"],
        default="qmai",
    )

    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_ADAPTERS.keys()),
        default="roban_s17",
    )

    parser.add_argument(
        "--record_video_path",
        nargs="?",
        const="",
        default=None,
        type=str,
        help=(
            "Enable video recording. "
            "If omitted: no video is generated. "
            "If provided without a value: default is output/<robot>/videos/<pkl_stem>.mp4. "
            "If provided with a value: use that path."
        ),
    )

    parser.add_argument(
        "--rate_limit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--bvh_fps",
        required=True,
        type=float,
        help="Source BVH frame rate in Hz.",
    )

    parser.add_argument(
        "--bvh_unit",
        choices=["auto", "mm", "cm", "m"],
        default="auto",
        help=(
            "BVH position unit. "
            "'auto' estimates from root-translation magnitude; "
            "'mm'/'cm'/'m' uses explicit scale."
        ),
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Output path. Accepts either a .pkl file path or a directory path. "
            "If omitted, default is output/<robot>/pkl/<bvh_stem>_<fps>.pkl."
        ),
    )
    
    parser.add_argument(
        "--motion_fps",
        default=None,
        type=float,
        help="Target output frame rate in Hz. If omitted, uses --bvh_fps.",
    )

    args = parser.parse_args()
    selected_robot_adapter = ROBOT_ADAPTERS[args.robot]

    bvh_file_path = args.bvh_file
    motion_frames_for_unit_estimation = probe_motion_frames_for_unit_estimation(
        bvh_format=args.format,
        bvh_file_path=bvh_file_path,
    )
    bvh_unit_plan = resolve_bvh_unit_plan(
        bvh_unit_arg=args.bvh_unit,
        bvh_format=args.format,
        motion_frames_for_estimation=motion_frames_for_unit_estimation,
    )

    bvh_source_key = f"bvh_{args.format}"
    if bvh_source_key not in IK_CONFIG_DICT or args.robot not in IK_CONFIG_DICT[bvh_source_key]:
        available_robot_names = list(IK_CONFIG_DICT.get(bvh_source_key, {}).keys())
        raise ValueError(
            f"No IK config for format='{args.format}', robot='{args.robot}'. "
            f"Available robots: {available_robot_names}"
        )
    json_file_path = IK_CONFIG_DICT[bvh_source_key][args.robot]
    fps_plan = resolve_fps_plan(
        bvh_fps=args.bvh_fps,
        motion_fps=args.motion_fps,
        rate_limit_flag=args.rate_limit,
    )
    output_pkl_path = build_output_pkl_path(
        save_path_arg=args.save_path,
        robot_name=args.robot,
        bvh_file_path=bvh_file_path,
        output_fps=fps_plan.output_pkl_fps,
    )
    video_output_path = build_output_video_path(
        record_video_path_arg=args.record_video_path,
        robot_name=args.robot,
        output_pkl_path=output_pkl_path,
    )
    record_video_enabled = video_output_path is not None
    
    print(f"[Config] BVH file: {bvh_file_path}")
    print(f"[Config] IK config: {json_file_path}")
    print(f"[Config] Video recording enabled: {record_video_enabled}")
    if record_video_enabled:
        print(f"[Config] Video output: {video_output_path}")
    print(f"[Config] PKL output: {output_pkl_path}")
    print(f"[Config] Source BVH FPS: {fps_plan.source_bvh_fps}")
    print(f"[Config] Target motion FPS: {fps_plan.target_motion_fps}")
    print(f"[Config] Viewer rate limit enabled: {fps_plan.viewer_rate_limit_enabled}")
    print(f"[Config] BVH unit arg: {args.bvh_unit}")
    print(f"[Config] Resolved BVH unit: {bvh_unit_plan.unit_name}")
    print(f"[Config] Position scale to meter: {bvh_unit_plan.position_scale_to_meter}")
    
    # Load motion data through robot-specific adapter entrypoint.
    mocap_data, actual_human_height = selected_robot_adapter.load_motion_data(
        bvh_format=args.format,
        bvh_file_path=bvh_file_path,
        position_scale_to_meter=bvh_unit_plan.position_scale_to_meter,
    )
    frame_indices = build_resampled_frame_indices(
        num_source_frames=len(mocap_data),
        source_fps=fps_plan.source_bvh_fps,
        target_fps=fps_plan.target_motion_fps,
    )

    qpos_list = []

    # Build robot-specific preprocessor through adapter.
    preprocessor = selected_robot_adapter.build_preprocessor(
        bvh_format=args.format,
        ik_config_file=pathlib.Path(json_file_path),
        actual_human_height=actual_human_height,
    )

    playback_controller = PlaybackController()

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps_plan.target_motion_fps,
        transparent_robot=0,
        record_video=record_video_enabled,
        video_path=str(video_output_path) if video_output_path is not None else None,
        keyboard_callback=playback_controller.handle_key,
    )
    print_keyboard_help()

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    pbar = tqdm(total=len(frame_indices), desc="Retargeting")
    frame_cursor = 0
    should_finish = False
    last_qpos = None
    last_scaled_human_data = None

    with TerminalKeyReader() as terminal_key_reader:
        while True:
            terminal_action = terminal_key_reader.read_action_nonblocking()
            if terminal_action is not None:
                playback_controller.handle_action(terminal_action, source="terminal")

            if playback_controller.stop_requested:
                should_finish = True

            should_advance = (
                (not playback_controller.is_paused) or playback_controller.consume_step_once()
            )

            if should_advance and not should_finish:
                # FPS measurement for frame advancement path.
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_start_time >= fps_display_interval:
                    actual_fps = fps_counter / (current_time - fps_start_time)
                    print(f"Actual rendering FPS: {actual_fps:.2f}")
                    fps_counter = 0
                    fps_start_time = current_time

                # Update task targets.
                source_frame_index = int(frame_indices[frame_cursor])
                qpos = preprocessor.retarget(mocap_data[source_frame_index], offset_to_ground=False)
                selected_robot_adapter.validate_qpos(qpos)
                qpos = selected_robot_adapter.postprocess_qpos(qpos)
                scaled_human_data = preprocessor.scaled_human_data

                qpos_list.append(qpos)
                last_qpos = qpos
                last_scaled_human_data = scaled_human_data
                pbar.update(1)

                frame_cursor += 1
                if frame_cursor >= len(frame_indices):
                    should_finish = True

            # Keep viewer responsive even when paused.
            if last_qpos is not None:
                robot_motion_viewer.step(
                    root_pos=last_qpos[:3] + np.array([0.0, 0.0, 0.0]),
                    root_rot=last_qpos[3:7],
                    dof_pos=last_qpos[7:],
                    human_motion_data=last_scaled_human_data,
                    rate_limit=(fps_plan.viewer_rate_limit_enabled or playback_controller.is_paused),
                    follow_camera=True,
                )

            if should_finish:
                break

    if len(qpos_list) == 0:
        pbar.close()
        robot_motion_viewer.close()
        tqdm.write("[Done] No frames processed; no PKL file generated.")
        raise RuntimeError("No frame was processed before stopping.")

    # Dump processed sequence to pkl file.
    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1,2,3,0]] for q in qpos_list])
    dof_pos = np.array([q[7:] for q in qpos_list])
    local_body_pos = None
    body_names = None
    motion_data = {
        "fps": fps_plan.output_pkl_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }
    os.makedirs(output_pkl_path.parent, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(motion_data, f)

    # Close progress bar and viewer before final logs.
    pbar.close()
    robot_motion_viewer.close()
    if record_video_enabled:
        tqdm.write(f"[Done] Output video path: {video_output_path}")
    tqdm.write(f"[Done] Output PKL path: {output_pkl_path}")
