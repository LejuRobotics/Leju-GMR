import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smpl_file_full, get_smpl_data_offline

from rich import print

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="SMPL motion retargeting to robot.")
    parser.add_argument(
        "--smpl_file",
        help="SMPL .npz motion file to load.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--robot",
        choices=["kuavo_s54"],
        default="kuavo_s54",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion (.pkl).",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--video_path",
        default=None,
        help="Path to save the video (.mp4).",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--tgt_fps",
        default=30,
        type=int,
        help="Target FPS for the retargeted motion.",
    )

    args = parser.parse_args()

    BODY_MODELS_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPL data and process through body model
    smpl_data, body_model, smplx_output, actual_human_height = load_smpl_file_full(
        args.smpl_file, BODY_MODELS_FOLDER
    )

    # Extract motion frames with SMPL joint names
    smpl_frames, aligned_fps = get_smpl_data_offline(
        smpl_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )
    print(f"Loaded {len(smpl_frames)} frames at {aligned_fps:.1f} FPS, human height: {actual_human_height:.2f}m")

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smpl",
        tgt_robot=args.robot,
    )

    # Video path
    video_path = args.video_path
    if video_path is None:
        smpl_stem = pathlib.Path(args.smpl_file).stem
        video_path = f"videos/{args.robot}_{smpl_stem}.mp4"

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=video_path,
    )

    # FPS measurement
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    i = 0
    while True:
        if args.loop:
            i = (i + 1) % len(smpl_frames)
        else:
            i += 1
            if i >= len(smpl_frames):
                break

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Retarget
        qpos = retarget.retarget(smpl_frames[i])

        # Visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    robot_motion_viewer.close()
