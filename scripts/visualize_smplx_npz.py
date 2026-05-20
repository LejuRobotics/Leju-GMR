"""可视化 AMASS SMPL-X .npz 动作数据（用 aitviewer）。

依赖:
- aitviewer (pip install aitviewer)
- SMPL-X 模型文件需放在 assets/body_models/smplx/ 下:
  SMPLX_NEUTRAL.npz / SMPLX_MALE.npz / SMPLX_FEMALE.npz

用法:
  python scripts/visualize_smplx_npz.py
  python scripts/visualize_smplx_npz.py --npz output/CMU/05/05_01_stageii.npz
  python scripts/visualize_smplx_npz.py --npz <path> --fps 60

注：避开了 aitviewer.SMPLSequence.from_amass()，因为它期望旧版 AMASS 的
'mocap_framerate' key，而较新的 AMASS 文件用 'mocap_frame_rate'。
"""

import argparse
import pathlib

import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def load_amass_smplx(npz_path):
    """读取 AMASS SMPL-X .npz，返回 SMPLSequence 构造器需要的字段。"""
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    gender = str(data["gender"]) if "gender" in keys else "neutral"

    # 帧率（兼容两种 key）
    if "mocap_frame_rate" in keys:
        src_fps = float(data["mocap_frame_rate"])
    elif "mocap_framerate" in keys:
        src_fps = float(data["mocap_framerate"])
    else:
        src_fps = 120.0

    # 取出 SMPL-X 参数
    if "pose_body" in keys:
        # 新版 AMASS 拆分好的字段
        root_orient = data["root_orient"]      # (N, 3)
        pose_body = data["pose_body"]          # (N, 63)
        pose_hand = data.get("pose_hand", None)  # (N, 90) 或缺失
        pose_jaw = data.get("pose_jaw", None)    # (N, 3)
        pose_eye = data.get("pose_eye", None)    # (N, 6)
    else:
        # 老版 AMASS：从 165 维 poses 切片
        poses = data["poses"]                  # (N, 165)
        root_orient = poses[:, 0:3]
        pose_body = poses[:, 3:66]
        pose_jaw = poses[:, 66:69]
        pose_eye = poses[:, 69:75]
        pose_hand = poses[:, 75:165]

    trans = data["trans"]                      # (N, 3)
    betas = data["betas"]                      # (16,) 或 (1, 16)
    if betas.ndim == 1:
        betas = betas[None, :]                  # (1, 16)

    return {
        "gender": gender,
        "src_fps": src_fps,
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "pose_hand": pose_hand.astype(np.float32) if pose_hand is not None else None,
        "pose_jaw": pose_jaw.astype(np.float32) if pose_jaw is not None else None,
        "pose_eye": pose_eye.astype(np.float32) if pose_eye is not None else None,
        "trans": trans.astype(np.float32),
        "betas": betas.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="可视化 AMASS SMPL-X .npz")
    parser.add_argument(
        "--npz",
        default="output/CMU/01/01_01_stageii.npz",
        help="AMASS .npz 文件路径",
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="目标播放帧率（aitviewer 会按这个速度回放，不重采样数据）",
    )
    parser.add_argument(
        "--body_models",
        default=str(pathlib.Path(__file__).parent.parent / "assets" / "body_models"),
        help="SMPL-X 模型根目录（其下应有 smplx/SMPLX_*.npz）",
    )
    args = parser.parse_args()

    C.update_conf({
        "smplx_models": args.body_models,
        # aitviewer 默认 pyqt6 后端跟 moderngl 创建 GL 上下文有兼容问题（"Cannot detect window with OpenGL support"），
        # 切到 glfw 后端绕开。
        "window_type": "glfw",
        # AMASS 经 root_orient 处理后数据是 Z-up（MuJoCo 风格），aitviewer 默认 Y-up，所以要切到 Z-up
        "z_up": True,
    })

    info = load_amass_smplx(args.npz)
    n_frames = info["pose_body"].shape[0]
    print(f"加载 {args.npz}")
    print(f"  gender = {info['gender']}")
    print(f"  frames = {n_frames}")
    print(f"  src fps = {info['src_fps']:.1f}")
    print(f"  播放 fps = {args.fps}")

    smpl_layer = SMPLLayer(model_type="smplx", gender=info["gender"], device="cpu")

    # 把 numpy 转 torch 给 SMPLSequence
    def t(x):
        return torch.from_numpy(x) if x is not None else None

    # 左右手分开（aitviewer SMPL-X 接 poses_left/right_hand 各 (N, 45)）
    if info["pose_hand"] is not None:
        poses_left_hand = info["pose_hand"][:, :45]
        poses_right_hand = info["pose_hand"][:, 45:90]
    else:
        poses_left_hand = poses_right_hand = None

    seq = SMPLSequence(
        poses_body=t(info["pose_body"]),
        smpl_layer=smpl_layer,
        poses_root=t(info["root_orient"]),
        betas=t(info["betas"]),
        trans=t(info["trans"]),
        poses_left_hand=t(poses_left_hand),
        poses_right_hand=t(poses_right_hand),
        name=pathlib.Path(args.npz).stem,
    )

    v = Viewer()
    v.playback_fps = args.fps
    v.scene.add(seq)
    v.run()


if __name__ == "__main__":
    main()
