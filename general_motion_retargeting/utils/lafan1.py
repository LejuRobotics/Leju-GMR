import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, read_bvh_leju


def load_bvh_file(bvh_file, format="lafan1"):
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
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
            
        if format == "lafan1":
            # Add modified foot pose
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif format == "nokov":
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        else:
            raise ValueError(f"Invalid format: {format}")
            
        frames.append(result)

    # Estimate height from head and feet in meters.
    try:
        human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    except KeyError:
        human_height = 1.75
    if not np.isfinite(human_height) or human_height < 0.5 or human_height > 2.5:
        human_height = 1.75

    return frames, human_height


def load_leju_bvh_file(bvh_file, format="leju"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    # from mocap format to kuavo format; support both old and newer BVH naming
    name_mapping = {
        # root / spine
        "Hips": "Hips",
        "Skeleton": "Hips",
        "Spine": "Spine",
        "Chest": "Spine",
        "Spine1": "Spine",
        "Neck": "Neck",
        "Head": "Head",
        # left arm
        "LeftShoulder": "LeftShoulder",
        "LShoulder": "LeftShoulder",
        "LeftArm": "LeftArm",
        "LUArm": "LeftArm",
        "LeftForeArm": "LeftForeArm",
        "LFArm": "LeftForeArm",
        "LeftHand": "LeftHand",
        "LHand": "LeftHand",
        # right arm
        "RightShoulder": "RightShoulder",
        "RShoulder": "RightShoulder",
        "RightArm": "RightArm",
        "RUArm": "RightArm",
        "RightForeArm": "RightForeArm",
        "RFArm": "RightForeArm",
        "RightHand": "RightHand",
        "RHand": "RightHand",
        # left leg
        "LeftUpLeg": "LeftUpLeg",
        "LThigh": "LeftUpLeg",
        "LeftLeg": "LeftLeg",
        "LShin": "LeftLeg",
        "LeftFoot": "LeftFoot",
        "LFoot": "LeftFoot",
        "LeftToeBase": "LeftToe",
        "LToe": "LeftToe",
        # right leg
        "RightUpLeg": "RightUpLeg",
        "RThigh": "RightUpLeg",
        "RightLeg": "RightLeg",
        "RShin": "RightLeg",
        "RightFoot": "RightFoot",
        "RFoot": "RightFoot",
        "RightToeBase": "RightToe",
        "RToe": "RightToe",
    }

    data = read_bvh_leju(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # rotation_matrix = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    # Convert from [x, y, z, w] to [w, x, y, z] format for scalar_first=True
    rotation_quat = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T  / 1000 # mm to m
            result[bone] = (position, orientation)

        # name mapping with alias fallback
        modified_result = {}
        for src_name, dst_name in name_mapping.items():
            if src_name in result and dst_name not in modified_result:
                modified_result[dst_name] = result[src_name]

        required_names = [
            "Hips", "Spine", "Neck", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
        ]
        missing = [name for name in required_names if name not in modified_result]
        if missing:
            raise KeyError(
                f"Missing required joints in BVH after name mapping: {missing}. "
                f"Available source joints: {list(result.keys())}"
            )

        # Keep both toe aliases to be compatible with different IK configs.
        modified_result["LeftToeBase"] = modified_result["LeftToe"]
        modified_result["RightToeBase"] = modified_result["RightToe"]
        modified_result["LeftFootMod"] = (modified_result["LeftFoot"][0], modified_result["LeftToe"][1])
        modified_result["RightFootMod"] = (modified_result["RightFoot"][0], modified_result["RightToe"][1])

        # Backward-compatible aliases used by older retarget configs.
        modified_result["Skeleton_002"] = modified_result["Hips"]
        modified_result["Chest"] = modified_result["Spine"]
        modified_result["LShoulder"] = modified_result["LeftShoulder"]
        modified_result["LUArm"] = modified_result["LeftArm"]
        modified_result["LFArm"] = modified_result["LeftForeArm"]
        modified_result["LHand"] = modified_result["LeftHand"]
        modified_result["RShoulder"] = modified_result["RightShoulder"]
        modified_result["RUArm"] = modified_result["RightArm"]
        modified_result["RFArm"] = modified_result["RightForeArm"]
        modified_result["RHand"] = modified_result["RightHand"]
        modified_result["LThigh"] = modified_result["LeftUpLeg"]
        modified_result["LShin"] = modified_result["LeftLeg"]
        modified_result["LFoot"] = modified_result["LeftFoot"]
        modified_result["LToe"] = modified_result["LeftToe"]
        modified_result["LFootMod"] = modified_result["LeftFootMod"]
        modified_result["RThigh"] = modified_result["RightUpLeg"]
        modified_result["RShin"] = modified_result["RightLeg"]
        modified_result["RFoot"] = modified_result["RightFoot"]
        modified_result["RToe"] = modified_result["RightToe"]
        modified_result["RFootMod"] = modified_result["RightFootMod"]
        frames.append(modified_result)
    
    human_height = modified_result["Head"][0][2] - min(modified_result["LeftFootMod"][0][2], modified_result["RightFootMod"][0][2])
    if not np.isfinite(human_height) or human_height < 0.5 or human_height > 2.5:
        human_height = 1.75
    return frames, human_height
