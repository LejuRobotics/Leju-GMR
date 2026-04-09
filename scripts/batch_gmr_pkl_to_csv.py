import argparse
import pickle
import os
import csv
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    args = parser.parse_args()

    for i, file in enumerate(os.listdir(args.folder)):
        if file.endswith(".pkl"):
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
        else:
            continue

        frame_rate = motion_data["fps"]            
        root_pos = motion_data["root_pos"]
        root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz align with beyondmimic
        dof_pos = motion_data["dof_pos"]

        qpos_list = []
        for j in range(len(root_pos)):
            # 拼接: [pos(3), rot(4), dof(N)]
            q = np.concatenate([root_pos[j], root_rot[j], dof_pos[j]])
            qpos_list.append(q)
        
        filepath = os.path.join(args.folder, "csv", file.replace(".pkl", ".csv"))
        if not qpos_list:
            print(f"Warning: qpos_list is empty, nothing to save to {filepath}")
            break

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
                row.extend(qpos[7:])  # Joint positions (21 joints, no head: 12 leg + 1 waist + 8 arm)
                writer.writerow(row)
        print(f"Saved to {filepath}")
