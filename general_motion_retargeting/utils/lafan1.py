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

    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

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
    # from mocap format to kuavo format
    name_mapping = {
        "Skeleton": "Hips",
        "Chest": "Spine",

        "Neck": "Neck",
        "Head": "Head",
        
        "LShoulder": "LeftShoulder",
        "LUArm": "LeftArm",
        "LFArm": "LeftForeArm",
        "LHand": "LeftHand",
        
        "RShoulder": "RightShoulder",
        "RUArm": "RightArm",
        "RFArm": "RightForeArm",
        "RHand": "RightHand",
        
        "LThigh": "LeftUpLeg",
        "LShin": "LeftLeg",
        "LFoot": "LeftFoot",
        "LToe": "LeftToe",

        "RThigh": "RightUpLeg",
        "RShin": "RightLeg",
        "RFoot": "RightFoot",
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

        # name mapping
        modified_result = {}
        for key in name_mapping.keys():
            modified_result[name_mapping[key]] = result[key]
        modified_result["LeftFootMod"] = (modified_result["LeftFoot"][0], modified_result["LeftToe"][1])
        modified_result["RightFootMod"] = (modified_result["RightFoot"][0], modified_result["RightToe"][1])
        frames.append(modified_result)
    
    human_height = modified_result["Head"][0][2] - min(modified_result["LeftFootMod"][0][2], modified_result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m
    print(modified_result.keys())
    return frames, human_height
