# LEJU GMR：重定向工具链

## GMR效果示例

Cute roban is dancing.

<video src="./assets/materials/roban_dance.mp4" width="500" controls></video>

[Watch roban dancing](./assets/materials/roban_dance.mp4)

## 1、概述

GMR(General Motion Retargeting)是人形机器人运动重定向技术框架。此仓库是通用运动重定向项目，用于实现人体运动数据重定向到人形机器人运动数据的高效转换。

## 2、安装GMR

拉取本仓库代码。

```bash
git clone https://gitcode.com/OpenLET/Leju-GMR.git
```

进入仓库目录，创建conda环境。

```bash
cd Leju-GMR/
conda create -n gmr python=3.10 -y  
conda activate gmr
```

然后安装GMR：

```bash
pip install -e .
```

若使用的是.pkl格式的SMPL-X模型文件，需将`smplx/body_models.py`文件中的npz改为pkl格式。
当遇到一些渲染问题时，可尝试如下命令。

```bash
conda install -c conda-forge libstdcxx-ng -y
```

## 3、快速使用

在`output/BVH/leju`目录下已预置一个BVH文件，可以使用如下命令，运行这个demo。

```bash
# single motion
python scripts/bvh_to_pkl.py \
   --bvh_file output/BVH/leju/dance_bj_01_Skeleton_002.bvh \
   --bvh_fps 100  \
   --robot roban_s17  \
   --motion_fps 50  \
   --rate_limit   \
   --format leju \
   --record_video_path
```

## 4、数据准备

* 下载LAFAN1 bvh文件。

从官方 lafan1 仓库下载原始 LAFAN1 BVH 文件，并存放在`assets`文件夹下，后续需要加载文件路径时指定`assets`下对应的文件地址即可 。

* 下载SMPL-X 身体模型

从官网注册并下载SMPL-X 身体模型到`assets/body_models`路径下。

> https://smpl-x.is.tue.mpg.de/

文件结构需按以下方式整理：

```bash
- assets/body_models/smplx/
-- SMPLX_NEUTRAL.pkl
-- SMPLX_FEMALE.pkl
-- SMPLX_MALE.pkl
```

* AMASS官网下载 SMPL-X 数据

从 AMASS官网注册并下载 SMPL-X 数据到任意文件夹。注：请勿选择 SMPL+H 数据。

> https://amass.is.tue.mpg.de/login.php



## 5、人体 / 机器人运动数据形式化定义

为更好地使用本库，推荐了解本框架所采用的人体运动数据及所获取的机器人运动数据。

人体运动数据的每一帧均被形式化定义为一个字典，键值对为（人体部位名称，三维全局平移信息 + 全局旋转信息）。旋转信息通常以四元数表示（默认采用 wxyz 顺序，与 MuJoCo 仿真器保持一致）。

机器人运动数据的每一帧可理解为一个元组，包含（机器人基座平移信息、机器人基座旋转信息、机器人关节位置信息）。



## 6、使用

### 6.1、从BVH重定向至机器人

```bash
python scripts/bvh_to_pkl.py \
--bvh_file <path_to_bvh_data> \
--bvh_fps <source_bvh_fps> \
--robot roban_s17 \
--save_path <path_or_dir_to_save_robot_data.pkl> \
--rate_limit \
--format <format> \
--motion_fps <target_motion_fps> \
--bvh_unit <auto|mm|cm|m> \
--record_video_path [video_path]
```

默认情况下，MuJoCo 窗口中展示重定向后机器人动作的可视化效果。

参数说明：

* `--bvh_file`:指定要处理的 BVH 文件路径，此参数为必填参数。

* `--bvh_fps`:源 BVH 帧率（Hz），必填参数。

* `--robot`:指定运动重定向的目标机器人型号。

* `--save_path`:输出路径（可选）。支持以下模式：
  - 传入`.pkl`文件路径：按指定文件保存；
  - 传入目录路径：保存到`<dir>/<robot>/pkl/<bvh_stem>_<帧率>fps.pkl`；
  - 不传：默认保存到`output/roban_s17/pkl/<bvh_stem>_<帧率>fps.pkl`，例如`dance_bj_01_Skeleton_002_50fps.pkl`。

* `--rate_limit`:限制 MuJoCo 窗口按目标帧率播放。不传时默认开启。只要指定了`--motion_fps`，播放也会按该帧率限速，此时`--rate_limit`不再单独生效。若希望窗口尽快播完，需要同时去掉`--motion_fps`，并传入`--no-rate-limit`。

* `--motion_fps`:目标输出帧率（Hz，可选）。不指定时默认使用`--bvh_fps`。

* `--bvh_unit`:BVH 位置单位，支持`auto`、`mm`、`cm`、`m`，默认`auto`。若已知单位，建议显式指定（如`--bvh_unit cm`）。

* `--record_video_path [video_path]`:可视化录制参数（可选）：
  - 不传：不录制视频；
  - 仅传`--record_video_path`：默认保存到`output/roban_s17/videos/<pkl_stem>.mp4`；
  - 传具体路径：按指定路径保存视频。

* `--format`:指定人体动画数据格式。支持`leju`、`lafan1`、`qmai`。


### 6.2、可视化已保存的机器人动作

(1)可视化单个动作：

```bash
python scripts/vis_robot_motion.py \
--robot roban_s17 \
--robot_motion_path <path_to_save_robot_data.pkl>
```

若需录制视频，需添加参数`--record_video`和`--video_path <your_video_path.mp4>`。

(2)可视化 CSV 动作：

```bash
python scripts/vis_robot_motion_csv.py \
--robot roban_s17 \
--csv_path <path_to_robot_data.csv> \
--motion_fps 50
```

(3)可视化文件夹中的批量动作：

```bash
python scripts/vis_robot_motion_dataset.py \
--robot roban_s17 \
--robot_motion_folder <path_to_save_robot_data_folder>
```


### 6.3、预览 SMPL-X `.npz` 原始动作（不经过重定向）

调试重定向效果时，常需要先看一下"标准答案"（人体动作本身长什么样）。本框架通过 [aitviewer](https://github.com/eth-ait/aitviewer) 提供独立的 SMPL-X 数据预览工具：

```bash
# 默认查看 output/CMU/01/01_01_stageii.npz
python scripts/visualize_smplx_npz.py

# 指定文件 + 帧率
python scripts/visualize_smplx_npz.py \
    --npz output/CMU/05/05_01_stageii.npz \
    --fps 60
```

依赖：

```bash
pip install aitviewer
```

参数说明：

* `--npz`：AMASS `.npz` 文件路径
* `--fps`：播放帧率（aitviewer 默认按这个速度回放）
* `--body_models`：SMPL-X 模型根目录（默认 `assets/body_models/`）

注意事项：

* 脚本绕开了 aitviewer 自带的 `from_amass()`，因为它要求 `mocap_framerate` 键而新版 AMASS 用 `mocap_frame_rate`，会冲突。
* aitviewer 默认 Y-up 坐标系，本脚本已通过 `z_up: True` 切换为 Z-up，与 MuJoCo 风格一致。
* viewer 启动时如果遇到 `Cannot detect window with OpenGL support`，可能是 PyQt6 后端冲突，脚本已默认切到 `glfw` 后端。仍不行可尝试 `LIBGL_ALWAYS_SOFTWARE=1 python ...` 用软件渲染。

## 7、数据转换（pkl_to_csv）

> CSV 文件本身不存帧率，要查可以读对应 PKL 的 `motion_data["fps"]` 字段：
>
> ```bash
> python -c "import pickle; d=pickle.load(open('output/roban_s17/pkl/0211_re_005_Skeleton_50fps.pkl','rb')); print(d['fps'])"
> ```

框架提供数据转换功能，可将机器人运动 `.pkl` 转换为 `.csv`，并按机器人型号进行严格 DoF 校验与重排（`Leg + Waist + Arm`）。

- 基本用法

```bash
# 单文件转换
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--pkl_file <path_to_file.pkl>

# 文件夹批量转换
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--folder <path_to_pkl_folder>
```

- 参数说明
  - `--robot`（必需）：机器人型号
  - `--pkl_file` / `--folder`（二选一，必需）：
    - `--pkl_file`：转换单个 `.pkl` 文件；
    - `--folder`：扫描目录下所有 `.pkl` 文件并批量转换。
  - `--output`（可选）：
    - 单文件模式：可传目标 `.csv` 文件路径，或输出目录；
    - 批量模式：作为输出目录使用；
    - 不传时默认输出到与 `pkl/` 目录平级的 `csv/`。若输入不在名为 `pkl` 的目录中，则写到文件同级 `csv/`。
  - `--max_frames`（可选）：限制导出最大帧数（从第 0 帧开始）。

- 输出格式
  - 每行数据格式为：`[root_pos(x,y,z), root_rot(x,y,z,w), dof_pos(...)]`。

- 使用示例

```bash
# 示例1：转换单个 pkl 文件
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--pkl_file output/roban_s17/pkl/0211_re_005_Skeleton_50fps.pkl

# 示例2：批量转换目录中的 pkl 文件
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--folder output/roban_s17/pkl

# 示例3：指定输出目录并限制帧数
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--folder output/roban_s17/pkl \
--output output/roban_s17/csv \
--max_frames 2000
```
