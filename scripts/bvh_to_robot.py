import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, "/home/leju/workspace/motion-transformer/")


import mink
import mujoco as mj
import numpy as np
import json
import pickle
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
from typing import Dict
import pathlib

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, qmai_read_bvh

import argparse
import time
from tqdm import tqdm
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file, load_leju_bvh_file
from general_motion_retargeting.params import IK_CONFIG_DICT

HERE = pathlib.Path(__file__).parent

def load_lafan1_file_for_biped_s17(bvh_file):
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
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        # Add modified foot pose
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])

        frames.append(result)
    
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height

def load_qmai_bvh_for_biped_s17(bvh_file):
    """Load qmai BVH file for biped_s17."""
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
            # dance_bj_01 uses cm, convert to m: cm -> m = / 100
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
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


def convert_qpos_old_to_new_urdf(qpos):
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


def save_qpos_s17_dance_no_head_joint(qpos_list, filepath):
    """Save qpos_list to CSV file for biped_s17.
    
    注意：CSV中保存的是基于XML的原始qpos（base_link=waist），不做URDF转换。
    URDF转换（waist→torso）在csv_to_npz_s22.py中插值完成后进行，
    避免slerp/lerp分别插值root_rot和waist_yaw导致的不一致问题。
    """
    import csv
    if not qpos_list:
        print(f"Warning: qpos_list is empty, nothing to save to {filepath}")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write data
        targetMaxFrame = 8633
        for frame_idx, qpos in enumerate(qpos_list):
            if frame_idx > targetMaxFrame:
                continue
            row = []
            row.extend(qpos[:3])  # Root position
            row.extend([qpos[4], qpos[5], qpos[6], qpos[3]])  # Root rotation (wxyz to xyzw)
            row.extend(qpos[7:28])  # Joint positions (21 joints, no head: 12 leg + 1 waist + 8 arm)
            writer.writerow(row)
    print(f"Saved s17 dance data to {filepath} with {min(len(qpos_list), targetMaxFrame)} frames (21 joints: 12 leg + 1 waist + 8 arm, no head)")


def load_leju_bvh_file_s17(bvh_file):
    """Load leju BVH file for biped_s17."""
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
            # dance_bj_01 uses mm, convert to m: mm -> m = / 1000
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # mm to m
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


class BipedS17GMR(GeneralMotionRetargeting):
    """General Motion Retargeting (GMR) for Biped S17 robot.
    Adapted from KuavoGMR for the S17 robot platform.
    """
    def __init__(
        self,
        actual_human_height: float = None,
        solver: str="daqp",
        damping: float=5e-1,
        verbose: bool=True,
        use_velocity_limit: bool=False,
        contact_sequence: Dict[str, np.ndarray] = None,
        ik_config_file: str = None,  # 新增参数：允许指定json配置文件
    ) -> None:
        # used for contact offset
        self.contact_sequence = contact_sequence
        self.previous_human_data = None

        # load the robot model for biped_s17
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

        # Load the IK config for biped_s17
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

ROBOT_DEFAULT_HUMAN_HEIGHT = {
    "kuavo_s54": 1.57,
    "kuavo_s52": 1.57,
    "roban_s14": 1.80,
}


def _scale_human_frames(frames, scale_factor):
    if abs(scale_factor - 1.0) < 1e-8:
        return
    for frame in frames:
        for body_name, (pos, rot) in frame.items():
            frame[body_name] = (np.asarray(pos) * scale_factor, rot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", help="BVH motion file to load.", required=True, type=str)
    parser.add_argument("--format", choices=["leju", "lafan1", "nokov", "qmai"], default="leju")
    parser.add_argument("--loop", default=False, action="store_true", help="Loop the motion.")
    parser.add_argument(
        "--robot",
        choices=["roban_s14", "kuavo_s52", "kuavo_s54", "biped_s17"],
        default="roban_s14",
    )
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="output/videos/example.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the robot motion.")
    parser.add_argument("--motion_fps", default=30, type=int)
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=None,
        help="Override detected human height (meters) for retarget scaling.",
    )
    parser.add_argument(
        "--ik_config_file",
        type=str,
        default=None,
        help="Custom IK config path. Highest priority if provided.",
    )
    parser.add_argument(
        "--leju_scale_mode",
        choices=["auto", "none"],
        default="auto",
        help="Auto-fix Leju BVH unit scale when detected height is abnormal.",
    )
    parser.add_argument(
        "--use_scene_xml",
        action="store_true",
        default=False,
        help="Use scene.xml for viewer background (ground/skybox), if available.",
    )
    args = parser.parse_args()

    if args.robot == "biped_s17":
        if args.format not in ["lafan1", "qmai", "leju"]:
            raise ValueError("biped_s17 only supports formats: lafan1, qmai, leju")

        bvh_filename = os.path.basename(args.bvh_file)
        base_name = os.path.splitext(bvh_filename)[0]
        json_file_path = IK_CONFIG_DICT[f"bvh_{args.format}"][args.robot]
        save_dir = args.save_path or "output/csv"
        os.makedirs(save_dir, exist_ok=True)
        csv_output_path = os.path.join(save_dir, f"{base_name}.csv")

        if args.format == "qmai":
            mocap_data, actual_human_height = load_qmai_bvh_for_biped_s17(args.bvh_file)
        elif args.format == "lafan1":
            mocap_data, actual_human_height = load_lafan1_file_for_biped_s17(args.bvh_file)
        else:
            mocap_data, actual_human_height = load_leju_bvh_file_s17(args.bvh_file)

        if args.actual_human_height is not None:
            actual_human_height = args.actual_human_height

        preprocessor = BipedS17GMR(
            actual_human_height=actual_human_height,
            solver="daqp",
            damping=5e-1,
            verbose=True,
            use_velocity_limit=True,
            ik_config_file=json_file_path,
        )

        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
            use_scene_xml=args.use_scene_xml,
            record_video=args.record_video,
            video_path=args.video_path,
        )

        qpos_list = []
        pbar = tqdm(total=len(mocap_data), desc="Retargeting")
        i = 0
        while True:
            pbar.update(1)
            qpos = preprocessor.retarget(mocap_data[i], offset_to_ground=False)
            qpos_list.append(qpos)
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=preprocessor.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if args.loop:
                i = (i + 1) % len(mocap_data)
            else:
                i += 1
                if i >= len(mocap_data):
                    break

        motion_data = {
            "fps": args.motion_fps,
            "root_pos": np.array([q[:3] for q in qpos_list]),
            "root_rot": np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list]),
            "dof_pos": np.array([q[7:] for q in qpos_list]),
            "local_body_pos": None,
            "link_body_list": None,
        }
        os.makedirs("output", exist_ok=True)
        with open("output/qpos_list.pkl", "wb") as f:
            pickle.dump(motion_data, f)
        print("Saved to output/qpos_list.pkl")
        save_qpos_s17_dance_no_head_joint(qpos_list, csv_output_path)
        pbar.close()
        robot_motion_viewer.close()
    else:
        if args.format == "qmai":
            raise ValueError("qmai format is currently only supported for biped_s17 in this script.")

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
            lafan1_data_frames, actual_human_height = load_leju_bvh_file(args.bvh_file)
            if args.leju_scale_mode == "auto":
                if actual_human_height < 0.6:
                    _scale_human_frames(lafan1_data_frames, 10.0)
                    actual_human_height *= 10.0
                    print(f"[Info] Auto-scaled Leju frames by x10 (detected height now {actual_human_height:.3f} m)")
                elif actual_human_height > 2.5:
                    _scale_human_frames(lafan1_data_frames, 0.1)
                    actual_human_height *= 0.1
                    print(f"[Info] Auto-scaled Leju frames by x0.1 (detected height now {actual_human_height:.3f} m)")
        else:
            lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

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
        )
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
            use_scene_xml=args.use_scene_xml,
            record_video=args.record_video,
            video_path=args.video_path,
        )

        qpos_list = []
        pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
        i = 0
        while True:
            pbar.update(1)
            qpos = retargeter.retarget(lafan1_data_frames[i])
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
            if args.save_path is not None:
                qpos_list.append(qpos)

            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    break

        if args.save_path is not None:
            root_pos = np.array([q[:3] for q in qpos_list])
            root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])
            dof_pos = np.array([q[7:] for q in qpos_list])
            if args.robot == "roban_s14":
                reduced_dof_pos = np.zeros((dof_pos.shape[0], dof_pos.shape[1] - 2), dtype=dof_pos.dtype)
                reduced_dof_pos[:, :13] = dof_pos[:, :13]
                reduced_dof_pos[:, 13:17] = dof_pos[:, 13:17]
                reduced_dof_pos[:, 17:21] = dof_pos[:, 18:22]
                dof_pos = reduced_dof_pos

            motion_data = {
                "fps": args.motion_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            }
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {args.save_path}")

        pbar.close()
        robot_motion_viewer.close()
