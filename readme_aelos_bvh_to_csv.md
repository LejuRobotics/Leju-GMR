# AELOS BVH 到 CSV / SRC 使用说明

这份文档只覆盖 AELOS 的当前可用链路：

```text
BVH(qmai 或 qingtong) -> AELOS PKL -|-> AELOS CSV
                                    |-> AELOS SRC
```

AELOS 使用 `scripts/bvh_to_robot.py`，不使用 `scripts/bvh_to_pkl.py`（后者尚未注册 `aelos`）。IK 配置：

| `--format` | IK 配置 |
| --- | --- |
| `qmai` | `general_motion_retargeting/ik_configs/bvh_qmai_to_aelos.json` |
| `qingtong` | `general_motion_retargeting/ik_configs/bvh_qingtong_to_aelos.json` |

千面接口下载的 BVH 用 `--format qmai`；青瞳 / SOMA（`*_to_soma_ZYX.bvh`）用 `--format qingtong`。

## 1. 环境准备

先按主 README 安装环境：

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
```

后续命令默认在仓库根目录执行。如果当前环境没有 `python` 命令，可用 `python3` 替换。

## 2. 文件路径约定

pkl、csv、src、video 彼此平级，都在 `output/aelos/` 下：

```text
output/BVH/qmai/<motion_name>.bvh
output/BVH/qingtong/<motion_name>.bvh
output/aelos/pkl/<motion_name>.pkl
output/aelos/csv/<motion_name>.csv
output/aelos/src/<motion_name>.src
output/aelos/video/<motion_name>.mp4
```

仓库里已有一个 qmai 示例：

```text
output/BVH/qmai/apple_T.bvh
output/aelos/pkl/apple_T.pkl
output/aelos/csv/apple_T.csv
output/aelos/src/apple_T.src
output/aelos/video/apple_T.mp4
```

不传 `--save_path` / `--video_path` 时，`bvh_to_robot.py` 按 BVH 文件名写到上述 `pkl/`、`video/`。不传 `--output` 时，`pkl_to_csv.py` 写到平级 `csv/`，`pkl_to_src.py` 写到平级 `src/`。

## 3. BVH 转 AELOS PKL

qmai 最小示例（apple_T 约 30fps，不传保存路径也会落到 `output/aelos/pkl/`）：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file output/BVH/qmai/apple_T.bvh \
  --format qmai \
  --robot aelos \
  --motion_fps 30 \
  --rate_limit
```

qingtong 示例（源 BVH 为 120fps）：

```bash
python scripts/bvh_to_robot.py \
  --bvh_file output/BVH/qingtong/311_猪猪侠_Take_094_zhozho_to_soma_ZYX.bvh \
  --format qingtong \
  --robot aelos \
  --motion_fps 120 \
  --record_video
```

同一目录批量转 pkl：

```bash
for bvh in output/BVH/qingtong/*.bvh; do
  python scripts/bvh_to_robot.py \
    --bvh_file "$bvh" \
    --format qingtong \
    --robot aelos \
    --motion_fps 120
done
```

参数说明：

- `--bvh_file`：输入 BVH 文件。
- `--format`：`qmai` 或 `qingtong`，必须和 IK 配置对应。
- `--robot aelos`：目标机器人。
- `--save_path`：PKL 路径。不传时默认 `output/aelos/pkl/<bvh文件名>.pkl`。也可传目录，脚本会在该目录下用 BVH 文件名生成 `.pkl`。
- `--record_video`：加上才会录视频。
- `--video_path`：视频路径。不传时默认 `output/aelos/video/<bvh文件名>.mp4`。也可传目录。
- `--motion_fps`：只影响 MuJoCo 预览和 mp4 封装帧率，**不改 PKL 动作帧率**。PKL 的 `fps` / `frame_time` 来自 BVH 的 `Frame Time`。默认 30；qingtong 源是 120，若要录像请传 `--motion_fps 120`，否则 mp4 会按 30fps 播得更慢。无法从 BVH 读到 `Frame Time` 时，才会把该值当作 PKL 帧率回退。
- `--rate_limit`：可视化时按 `--motion_fps` 限速播放；不加则尽快跑完。

执行时会打开 MuJoCo 可视化窗口。脚本跑完整段 BVH 后会自动写出 PKL。PKL 中同时保存 BVH 原始 `fps`、`frame_time` 和动作时长，后续生成 SRC 时会使用这些数据计算播放延时。

## 4. AELOS PKL 转 CSV

单个 PKL（可省略 `--output`，默认写到 `output/aelos/csv/apple_T.csv`）：

```bash
python scripts/pkl_to_csv.py \
  --robot aelos \
  --pkl_file output/aelos/pkl/apple_T.pkl
```

批量转换一个目录下的 PKL：

```bash
python scripts/pkl_to_csv.py \
  --robot aelos \
  --folder output/aelos/pkl
```

不要用 `scripts/batch_gmr_pkl_to_csv.py` 处理 AELOS。`scripts/pkl_to_csv.py` 会按 AELOS 的关节布局做严格校验和重排。

## 5. AELOS PKL 转 SRC

生成可下发给 AELOS 的 SRC 动作文件（可省略 `--output`，默认写到 `output/aelos/src/apple_T.src`）：

```bash
python scripts/pkl_to_src.py \
  --pkl_file output/aelos/pkl/apple_T.pkl \
  --print_frames 0
```

同一目录批量：

```bash
for pkl in output/aelos/pkl/*.pkl; do
  python scripts/pkl_to_src.py --pkl_file "$pkl" --print_frames 0
done
```

默认逐帧转换，并根据相邻帧最终 SRC 舵机值的最大位移自动生成每帧 `SPEED`：非零位移的最小值、中位数和最大值分别映射为 `30`、`75` 和 `120`，其余帧分段线性插值；首帧和静止帧使用 `30`。脚本会优先读取 PKL 中的 `frame_time`，将帧周期换算为毫秒并向下取整，自动为每帧生成 `DELAY`，因此使用新版 `scripts/bvh_to_robot.py` 生成的 PKL 时通常不需要手动设置播放延时。

常用可选参数：

- `--frame_stride N`：每隔 `N` 个 PKL 帧取一帧，默认值为 1；自动生成的 `DELAY` 会同步乘以 `N`，保持动作总时长基本不变。
- `--speed N`：关闭自适应速度并将每帧 SRC `SPEED` 固定为 `N`，允许范围为 `5～150`；不传时默认使用 `30/75/120` 自适应映射。
- `--delay_ms N`：手动指定每个输出帧后的 `DELAY` 毫秒数，覆盖 PKL 中的时序；传 0 可完全禁用 `DELAY`。
- `--keep_wait`：每个动作帧后增加 `WAIT`。这通常会让连续动作不够流畅，默认不启用。
- `--print_frames N`：在终端打印前 `N` 个转换结果，默认打印 1 帧；传 0 可关闭。
- `--output_csv <path>`：额外输出舵机原始值 CSV，便于检查 1～19 号舵机数据。
- `--leg_hip_min N` / `--leg_hip_max N`：分别限制左右髋关节的最终 SRC 值，对所有动作统一生效；默认范围为 `100～115`，不会再把一侧角度复制给另一侧。全局 `10～190` 限幅仍作为最后保护。

例如，每 3 帧采样一次，同时输出舵机值检查文件：

```bash
python scripts/pkl_to_src.py \
  --pkl_file output/aelos/pkl/apple_T.pkl \
  --output output/aelos/src/apple_T.src \
  --output_csv output/aelos/src/apple_T_servo.csv \
  --frame_stride 3
```

注意：这里通过 `--output_csv` 生成的是 19 路舵机原始值检查文件，与第 4 节中用于动作可视化的 23 列 AELOS CSV 格式不同。

转换时，脚本会把 PKL 中的 AELOS 关节弧度映射为舵机值，再按 AELOS 的 `MOTORA`～`MOTORE` 指令顺序写入 SRC。左右腿 2～4 号关节保持经过实物测试的固定站姿；左右腿 1 号关节分别使用各自的 PKL 重定向角度，并保留默认 `100～115` 的安全限幅；左右腿 5 号关节不使用 PKL 轨迹，而是根据同侧已经限幅的 1 号关节按 `q5=-q1` 联动，使最终 SRC 满足 `腿5=200-腿1`。17、18 号夹爪和 19 号头部不由当前 PKL 关节数据驱动，使用脚本内配置的默认值。

## 6. CSV 格式

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
awk -F, 'NR==1 {print NF}' output/aelos/csv/apple_T.csv
```

期望输出：

```text
23
```

## 7. 可视化检查

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
  --csv_path output/aelos/csv/apple_T.csv \
  --motion_fps 30
```

## 8. 常见问题

`No IK config for format ... robot='aelos'` 或 `KeyError: 'aelos'`：

确认 `--format` 是 `qmai` 或 `qingtong`，并且 `params.py` 里已注册对应 IK JSON。qmai 用 `bvh_qmai_to_aelos.json`，qingtong 用 `bvh_qingtong_to_aelos.json`。

`dof_pos has ... joints, expected 16 for aelos`：

输入 PKL 不是 AELOS 的 16 自由度数据，或不是通过当前 AELOS 链路生成的 PKL。

`ModuleNotFoundError: No module named 'numpy._core'`：

通常是生成 PKL 和读取 PKL 的 numpy 环境不一致。`scripts/pkl_to_src.py` 已包含常见 numpy 1.x/2.x PKL 的兼容处理；如果其他脚本仍然报错，请激活生成该文件时使用的 conda 环境后再转换。

生成 SRC 时显示 `DELAY = disabled`：

输入 PKL 中没有可用的 `frame_time`。建议用当前版本的 `scripts/bvh_to_robot.py` 重新生成 PKL，或者通过 `--delay_ms` 手动指定延时。

MuJoCo 窗口打不开：

先确认已在带图形界面的机器上运行，并且 `pip install -e .` 已完成依赖安装。服务器或容器环境需要额外配置 MuJoCo 渲染后端。
