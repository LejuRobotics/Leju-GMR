import argparse
import os
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="roban_s14")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    if robot_type == "roban_s14":
        extend_dof_pos = np.zeros((motion_dof_pos.shape[0], motion_dof_pos.shape[1] + 2), dtype=motion_dof_pos.dtype)
        extend_dof_pos[:, :13] = motion_dof_pos[:, :13] # legs
        extend_dof_pos[:, 13:17] = motion_dof_pos[:, 13:17] # left arms
        extend_dof_pos[:, 18:22] = motion_dof_pos[:, 17:21] # right arms
        motion_dof_pos = extend_dof_pos

    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()