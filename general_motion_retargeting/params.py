import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "roban_s14": ASSET_ROOT / "roban_s14" / "biped_s14.xml",
    "biped_s17": ASSET_ROOT / "biped_s17" / "xml" / "biped_s17.xml",
    "kuavo_s52": ASSET_ROOT / "kuavo_s52" / "biped_s52.xml",
    "kuavo_s45": ASSET_ROOT / "kuavo_s45" / "biped_s45_collision.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "kuavo_s45": IK_CONFIG_ROOT / "smplx_to_kuavo.json",
    },
    "bvh_lafan1":{
        "roban_s14": IK_CONFIG_ROOT / "bvh_lafan1_to_robans14.json",
        "kuavo_s52": IK_CONFIG_ROOT / "bvh_lafan1_to_kuavos52.json",
    },
    "bvh_nokov":{
        #
    },
    "bvh_leju":{
        "roban_s14": IK_CONFIG_ROOT / "bvh_leju_to_robans14.json",
        "kuavo_s52": IK_CONFIG_ROOT / "bvh_leju_to_kuavos52.json",
    },
    "fbx":{
        #
    },
    "fbx_offline":{
        #
    },
    
    "xrobot":{
        #
    },
    "bvh_qmai":{
        "biped_s17": IK_CONFIG_ROOT / "bvh_qmai_to_s17.json",
    },
}

ROBOT_BASE_DICT = {
    "roban_s14": "waist_yaw_link",
    "kuavo_s52": "base_link",
    "kuavo_s45": "base_link",
    "biped_s17": "base_link",
}

VIEWER_CAM_DISTANCE_DICT = {
    "roban_s14": 3.0,
    "kuavo_s52": 3.0,
    "kuavo_s45": 3.0,
    "biped_s17": 2.0,
}
