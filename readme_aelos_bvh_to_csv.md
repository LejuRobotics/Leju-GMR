# AELOS BVH 到 CSV 使用说明

这份文档只覆盖 AELOS 的当前可用链路：

```text
BVH(qmai) -> AELOS PKL -> AELOS CSV
```

AELOS 目前已有的 IK 配置是 `general_motion_retargeting/ik_configs/bvh_qmai_to_aelos.json`，因此 BVH 输入请使用 qmai 格式。若从千面视频接口下载 BVH，默认就是本链路需要的输入类型。

## 1. 环境准备

先按主 README 安装环境：

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
```

后续命令默认在仓库根目录执行。如果当前环境没有 `python` 命令，可用 `python3` 替换。

## 2. 文件路径约定

建议按下面的目录放置输入和输出：

```text
output/BVH/qmai/<motion_name>.bvh
output/aelos/pkl/<motion_name>.pkl
output/aelos/pkl/csv/<motion_name>.csv
output/aelos/video/<motion_name>.mp4
```

仓库里已有一个示例：

```text
output/BVH/qmai/apple_T.bvh
output/aelos/pkl/apple_T.pkl
output/aelos/pkl/csv/apple_T.csv
output/aelos/video/apple_T.mp4
```

## 3. BVH 转 AELOS PKL

当前 AELOS 使用 `scripts/bvh_to_robot.py` 生成 PKL，不使用 `scripts/bvh_to_pkl.py`。原因是新版 `bvh_to_pkl.py` 的 robot adapter 目前还没有注册 `aelos`。

最小示例：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file output/BVH/qmai/apple_T.bvh \
  --format qmai \
  --robot aelos \
  --motion_fps 30 \
  --rate_limit \
  --save_path output/aelos/pkl/apple_T.pkl
```

同时录制可视化视频：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file output/BVH/qmai/apple_T.bvh \
  --format qmai \
  --robot aelos \
  --motion_fps 30 \
  --rate_limit \
  --save_path output/aelos/pkl/apple_T.pkl \
  --record_video \
  --video_path output/aelos/video/apple_T.mp4
```

参数说明：

- `--bvh_file`：输入 BVH 文件。
- `--format qmai`：AELOS 当前只配置了 qmai BVH 到 AELOS 的 IK。
- `--robot aelos`：目标机器人。
- `--motion_fps`：输出动作帧率。qmai 默认常用 30，如果下载 BVH 时选择 60 FPS，这里也设为 60。
- `--rate_limit`：可视化时按动作帧率播放。
- `--save_path`：输出 PKL 文件路径。务必给到 `.pkl` 文件名。
- `--record_video` / `--video_path`：可选，开启并指定录制视频路径。

执行时会打开 MuJoCo 可视化窗口。脚本跑完整段 BVH 后会自动写出 PKL。

## 4. AELOS PKL 转 CSV

单个 PKL 转 CSV：

```bash
python scripts/pkl_to_csv.py \
  --robot aelos \
  --pkl_file output/aelos/pkl/apple_T.pkl \
  --output output/aelos/pkl/csv/apple_T.csv
```

批量转换一个目录下的 PKL：

```bash
python scripts/pkl_to_csv.py \
  --robot aelos \
  --folder output/aelos/pkl \
  --output output/aelos/pkl/csv
```

不要用 `scripts/batch_gmr_pkl_to_csv.py` 处理 AELOS。`scripts/pkl_to_csv.py` 会按 AELOS 的关节布局做严格校验和重排。

## 5. CSV 格式

CSV 无表头，每一行是一帧：

```text
root_pos_x, root_pos_y, root_pos_z,
root_rot_x, root_rot_y, root_rot_z, root_rot_w,
dof_0 ... dof_15
```

总列数为 23：

- `root_pos`：3 列，单位为米。
- `root_rot`：4 列，四元数顺序为 `xyzw`。
- `dof_pos`：16 列，单位为弧度。

AELOS CSV 的 16 个关节列顺序为：

| CSV 关节列 | AELOS joint |
| --- | --- |
| 0 | `jointRleg1` |
| 1 | `jointRleg2` |
| 2 | `jointRleg3` |
| 3 | `jointRleg4` |
| 4 | `jointRleg5` |
| 5 | `jointLleg1` |
| 6 | `jointLleg2` |
| 7 | `jointLleg3` |
| 8 | `jointLleg4` |
| 9 | `jointLleg5` |
| 10 | `jointRarm1` |
| 11 | `jointRarm2` |
| 12 | `jointRarm3` |
| 13 | `jointLarm1` |
| 14 | `jointLarm2` |
| 15 | `jointLarm3` |

这里的顺序是 `Leg -> Arm`。原始 AELOS XML 中 qpos 顺序是手臂在前、腿在后，`scripts/pkl_to_csv.py` 会自动把它重排成上表顺序。

快速检查 CSV 列数：

```bash
awk -F, 'NR==1 {print NF}' output/aelos/pkl/csv/apple_T.csv
```

期望输出：

```text
23
```

## 6. 可视化检查

检查 PKL：

```bash
python scripts/vis_robot_motion.py \
  --robot aelos \
  --robot_motion_path output/aelos/pkl/apple_T.pkl
```

检查 CSV：

```bash
python scripts/vis_robot_motion_csv.py \
  --robot aelos \
  --csv_path output/aelos/pkl/csv/apple_T.csv \
  --motion_fps 30
```

## 7. 常见问题

`No IK config for format ... robot='aelos'` 或 `KeyError: 'aelos'`：

AELOS 目前只配置了 `bvh_qmai_to_aelos.json`，命令中要使用 `--format qmai`。

`dof_pos has ... joints, expected 16 for aelos`：

输入 PKL 不是 AELOS 的 16 自由度数据，或不是通过当前 AELOS 链路生成的 PKL。

`ModuleNotFoundError: No module named 'numpy._core'`：

通常是生成 PKL 和读取 PKL 的 numpy 环境不一致。激活生成该文件时使用的 conda 环境后再转换。

MuJoCo 窗口打不开：

先确认已在带图形界面的机器上运行，并且 `pip install -e .` 已完成依赖安装。服务器或容器环境需要额外配置 MuJoCo 渲染后端。
