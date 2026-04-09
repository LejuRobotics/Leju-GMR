import argparse
import csv
import os
import pathlib
import pickle
import select
import sys
import time

import numpy as np
import mujoco as mj
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import IK_CONFIG_DICT
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file, load_leju_bvh_file
import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, read_bvh_leju
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).parent

ROBOT_DEFAULT_HUMAN_HEIGHT = {
    "kuavo_s54": 1.57,
    "kuavo_s52": 1.57,
    "roban_s14": 1.80,
}


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _format_dof_by_robot(dof_pos: np.ndarray, robot: str) -> np.ndarray:
    # Keep consistent with bvh_to_robot.py export logic for roban_s14.
    if robot != "roban_s14":
        return dof_pos

    reduced_dof = np.zeros((dof_pos.shape[0], dof_pos.shape[1] - 2), dtype=dof_pos.dtype)
    reduced_dof[:, :13] = dof_pos[:, :13]  # legs
    reduced_dof[:, 13:17] = dof_pos[:, 13:17]  # left arms
    reduced_dof[:, 17:21] = dof_pos[:, 18:22]  # right arms
    return reduced_dof


def _scale_human_frames(frames, scale_factor):
    if abs(scale_factor - 1.0) < 1e-8:
        return
    for frame in frames:
        for body_name, (pos, rot) in frame.items():
            frame[body_name] = (np.asarray(pos) * scale_factor, rot)


def _get_joint_qpos_addr(model, joint_name: str):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid]), int(jid)


def _set_hinge_qpos_with_limit(model, qpos: np.ndarray, joint_name: str, value: float) -> bool:
    joint_info = _get_joint_qpos_addr(model, joint_name)
    if joint_info is None:
        return False
    qaddr, jid = joint_info
    # only scalar joints should be set by this helper
    if model.jnt_type[jid] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
        return False
    if model.jnt_limited[jid]:
        low, high = model.jnt_range[jid]
        value = float(np.clip(value, low, high))
    qpos[qaddr] = value
    return True


def _supports_knee_flex_guard(retargeter) -> bool:
    return (
        _get_joint_qpos_addr(retargeter.model, "leg_l4_joint") is not None
        and _get_joint_qpos_addr(retargeter.model, "leg_r4_joint") is not None
    )


def _seed_symmetric_knee_flex(retargeter, knee_flex_rad: float = 0.06) -> None:
    """Set a symmetric small knee bend before processing frame 0."""
    qpos = retargeter.configuration.data.qpos
    _set_hinge_qpos_with_limit(retargeter.model, qpos, "leg_l4_joint", knee_flex_rad)
    _set_hinge_qpos_with_limit(retargeter.model, qpos, "leg_r4_joint", knee_flex_rad)


def _enforce_knee_floor(retargeter, qpos: np.ndarray, knee_floor_rad: float = 0.04) -> np.ndarray:
    """Clamp both knees to at least knee_floor_rad and keep solver state in sync."""
    for knee_joint in ("leg_l4_joint", "leg_r4_joint"):
        joint_info = _get_joint_qpos_addr(retargeter.model, knee_joint)
        if joint_info is None:
            continue
        qaddr, _ = joint_info
        if qpos[qaddr] < knee_floor_rad:
            _set_hinge_qpos_with_limit(retargeter.model, qpos, knee_joint, knee_floor_rad)
    return qpos


def _load_leju_bvh_reference(bvh_file):
    """Loader aligned with kuavo_s54_retarget script behavior."""
    try:
        data = read_bvh(bvh_file)
    except ValueError as exc:
        # Some Leju BVH files use spacing that read_bvh cannot parse.
        # Fallback to read_bvh_leju while keeping the same transform/unit logic.
        if "could not convert string to float" not in str(exc):
            raise
        print("[Info] read_bvh parse failed, fallback to read_bvh_leju for compatibility.")
        data = read_bvh_leju(bvh_file)
    global_data = lafan_utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    rotation_quat = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])

    frames = []
    alias_to_canonical = {
        "Hips": "Hips",
        "Skeleton": "Hips",
        "Skeleton_002": "Hips",
        "Spine": "Spine",
        "Chest": "Spine",
        "Spine1": "Spine",
        "Neck": "Neck",
        "Head": "Head",
        "LeftShoulder": "LeftShoulder",
        "LShoulder": "LeftShoulder",
        "LeftArm": "LeftArm",
        "LUArm": "LeftArm",
        "LeftForeArm": "LeftForeArm",
        "LFArm": "LeftForeArm",
        "LeftHand": "LeftHand",
        "LHand": "LeftHand",
        "RightShoulder": "RightShoulder",
        "RShoulder": "RightShoulder",
        "RightArm": "RightArm",
        "RUArm": "RightArm",
        "RightForeArm": "RightForeArm",
        "RFArm": "RightForeArm",
        "RightHand": "RightHand",
        "RHand": "RightHand",
        "LeftUpLeg": "LeftUpLeg",
        "LThigh": "LeftUpLeg",
        "LeftLeg": "LeftLeg",
        "LShin": "LeftLeg",
        "LeftFoot": "LeftFoot",
        "LFoot": "LeftFoot",
        "LeftToe": "LeftToe",
        "LeftToeBase": "LeftToe",
        "LToe": "LeftToe",
        "RightUpLeg": "RightUpLeg",
        "RThigh": "RightUpLeg",
        "RightLeg": "RightLeg",
        "RShin": "RightLeg",
        "RightFoot": "RightFoot",
        "RFoot": "RightFoot",
        "RightToe": "RightToe",
        "RightToeBase": "RightToe",
        "RToe": "RightToe",
    }
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = lafan_utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm -> m
            result[bone] = (position, orientation)

        # Normalize mixed naming styles into canonical keys.
        normalized = {}
        for src_name, dst_name in alias_to_canonical.items():
            if src_name in result and dst_name not in normalized:
                normalized[dst_name] = result[src_name]
        result.update(normalized)

        # canonical aliases for cross-robot IK compatibility
        if "LeftFoot" in result and "LeftToe" in result:
            result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
            result["LeftToeBase"] = result["LeftToe"]
            result["LFootMod"] = result["LeftFootMod"]
        if "RightFoot" in result and "RightToe" in result:
            result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])
            result["RightToeBase"] = result["RightToe"]
            result["RFootMod"] = result["RightFootMod"]
        if "Hips" in result:
            result["Skeleton_002"] = result["Hips"]
            result["Skeleton"] = result["Hips"]
        if "Spine" in result:
            result["Chest"] = result["Spine"]

        frames.append(result)

    if "Head" in result and "LeftFootMod" in result and "RightFootMod" in result:
        human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    else:
        human_height = 1.75
    return frames, human_height


def save_csv(qpos_list: list[np.ndarray], csv_path: str, robot: str) -> None:
    if not qpos_list:
        raise ValueError("qpos_list is empty, nothing to save.")

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot_xyzw = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    dof_pos = _format_dof_by_robot(dof_pos, robot)

    _ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(root_pos.shape[0]):
            row = np.concatenate((root_pos[i], root_rot_xyzw[i], dof_pos[i]), axis=0)
            writer.writerow(row.tolist())

    print(f"[Saved] CSV: {csv_path} ({len(qpos_list)} frames)")


def save_pkl(qpos_list: list[np.ndarray], pkl_path: str, fps: int, robot: str) -> None:
    if not qpos_list:
        raise ValueError("qpos_list is empty, nothing to save.")

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])  # xyzw
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    dof_pos = _format_dof_by_robot(dof_pos, robot)

    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    _ensure_parent_dir(pkl_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"[Saved] PKL: {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file path.")
    parser.add_argument("--format", choices=["leju", "lafan1", "nokov"], default="leju")
    parser.add_argument(
        "--robot",
        choices=["roban_s14", "kuavo_s52", "kuavo_s54"],
        default="roban_s14",
        help="Target robot type.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Output CSV path. Defaults to output/csv/{bvh_name}_{robot}.csv",
    )
    parser.add_argument(
        "--save_pkl",
        action="store_true",
        default=False,
        help="Also save a PKL file with the same naming convention.",
    )
    parser.add_argument("--motion_fps", type=int, default=30, help="Metadata fps for saved PKL.")
    parser.add_argument(
        "--ik_config_file",
        type=str,
        default=None,
        help="Custom IK config path. Highest priority if provided.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Record visualization to video.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Video output path. Defaults to output/videos/{bvh_name}_{robot}.mp4",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Enable playback rate limiting to motion_fps.",
    )
    parser.add_argument(
        "--disable_velocity_limit",
        action="store_true",
        default=False,
        help="Disable IK velocity limit (enabled by default).",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=None,
        help="Override detected human height (meters) for retarget scaling.",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        default=False,
        help="Disable visualization window (enabled by default).",
    )
    parser.add_argument(
        "--use_scene_xml",
        action="store_true",
        default=False,
        help="Use scene.xml for viewer background (ground/skybox), if available.",
    )
    parser.add_argument(
        "--no_interactive",
        action="store_false",
        dest="interactive",
        default=True,
        help="Disable Enter pause/resume and n single-step control (enabled by default).",
    )
    parser.add_argument(
        "--no_start_paused",
        action="store_false",
        dest="start_paused",
        default=True,
        help="Disable start-paused mode when interactive is enabled (enabled by default).",
    )
    parser.add_argument(
        "--leju_scale_mode",
        choices=["auto", "none"],
        default="auto",
        help="Auto-fix Leju BVH unit scale when detected height is abnormal.",
    )
    parser.add_argument(
        "--leju_loader",
        choices=["reference", "legacy"],
        default="reference",
        help="Leju BVH parser: reference matches kuavo_s54_retarget behavior.",
    )
    args = parser.parse_args()

    src_human = f"bvh_{args.format}"
    ik_config_file = None
    if args.ik_config_file is not None:
        ik_config_file = pathlib.Path(args.ik_config_file)

    if ik_config_file is not None:
        ik_config_file = ik_config_file.resolve()
        if not ik_config_file.exists():
            raise FileNotFoundError(f"IK config file not found: {ik_config_file}")
        print(f"[Info] Using IK config file: {ik_config_file}")
    else:
        supported_robots = IK_CONFIG_DICT.get(src_human, {})
        if args.robot not in supported_robots:
            raise ValueError(
                f"Robot '{args.robot}' is not configured for source '{src_human}'. "
                f"Supported robots: {sorted(supported_robots.keys())}"
            )
        print(f"[Info] Using default IK mapping for {src_human}/{args.robot}")

    if args.format == "leju":
        if args.leju_loader == "reference":
            bvh_data_frames, actual_human_height = _load_leju_bvh_reference(args.bvh_file)
        else:
            bvh_data_frames, actual_human_height = load_leju_bvh_file(args.bvh_file)
        if args.leju_scale_mode == "auto":
            if actual_human_height < 0.6:
                _scale_human_frames(bvh_data_frames, 10.0)
                actual_human_height *= 10.0
                print(f"[Info] Auto-scaled Leju frames by x10 (detected height now {actual_human_height:.3f} m)")
            elif actual_human_height > 2.5:
                _scale_human_frames(bvh_data_frames, 0.1)
                actual_human_height *= 0.1
                print(f"[Info] Auto-scaled Leju frames by x0.1 (detected height now {actual_human_height:.3f} m)")
    else:
        bvh_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

    if args.actual_human_height is not None:
        actual_human_height = args.actual_human_height
    elif args.format == "leju" and args.robot in ROBOT_DEFAULT_HUMAN_HEIGHT:
        actual_human_height = ROBOT_DEFAULT_HUMAN_HEIGHT[args.robot]
        print(f"[Info] Use robot default human height: {actual_human_height:.3f} m")
    print(f"[Info] actual_human_height used for scaling: {actual_human_height:.3f} m")

    retargeter = GMR(
        src_human=src_human,
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        ik_config_file=str(ik_config_file) if ik_config_file is not None else None,
        use_velocity_limit=not args.disable_velocity_limit,
    )
    apply_knee_flex_guard = _supports_knee_flex_guard(retargeter)
    if apply_knee_flex_guard:
        _seed_symmetric_knee_flex(retargeter)

    robot_motion_viewer = None
    bvh_name = pathlib.Path(args.bvh_file).stem
    default_output_root = HERE / ".." / "output" / args.robot
    video_path = args.video_path or str(default_output_root / "videos" / f"{bvh_name}.mp4")
    if not args.no_visualize:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
            use_scene_xml=args.use_scene_xml,
            record_video=args.record_video,
            video_path=video_path,
        )

    qpos_list = []
    is_paused = args.interactive and args.start_paused
    step_once = False
    if args.interactive and not args.no_visualize:
        print("\n[交互控制] Enter: 暂停/继续, n: 单步, Ctrl+C: 退出并保存\n")
    try:
        for frame_idx, frame_data in enumerate(tqdm(bvh_data_frames, desc="Retargeting to CSV")):
            if args.interactive and not args.no_visualize:
                if select.select([sys.stdin], [], [], 0)[0]:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd == "":
                        is_paused = not is_paused
                        state = "暂停" if is_paused else "继续"
                        print(f"[交互] {state}，当前帧 {frame_idx}/{len(bvh_data_frames) - 1}")
                    elif cmd in ("n", "s", "step"):
                        step_once = True
                while is_paused and not step_once:
                    time.sleep(0.01)
                    if select.select([sys.stdin], [], [], 0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == "":
                            is_paused = False
                            print("[交互] 继续")
                        elif cmd in ("n", "s", "step"):
                            step_once = True

            qpos = retargeter.retarget(frame_data)
            if apply_knee_flex_guard and frame_idx == 0:
                qpos = _enforce_knee_floor(retargeter, qpos)
                retargeter.configuration.data.qpos[:] = qpos
            qpos_list.append(qpos)
            if robot_motion_viewer is not None:
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=True,
                )
            if step_once:
                step_once = False
                is_paused = True
    finally:
        if robot_motion_viewer is not None:
            robot_motion_viewer.close()

    default_csv_path = str(default_output_root / "csv" / f"{bvh_name}.csv")
    csv_path = args.csv_path or default_csv_path
    save_csv(qpos_list, csv_path, args.robot)

    if args.save_pkl:
        if args.csv_path is None:
            pkl_path = str(default_output_root / "pkl" / f"{bvh_name}.pkl")
        else:
            pkl_path = str(pathlib.Path(csv_path).with_suffix(".pkl"))
        save_pkl(qpos_list, pkl_path, args.motion_fps, args.robot)
