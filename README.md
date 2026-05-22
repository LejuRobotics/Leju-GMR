# LEJU GMR：重定向工具链

## GMR效果示例
<table>
  <tr>
    <td align="center">
      <b>Demo 1</b><br>
      Cute roban is dancing.<br>
      <video src="./assets/materials/roban_dance.mp4" width="500" controls></video>
      <a href="./assets/materials/roban_dance.mp4">Watch roban dancing</a>
    </td>
    <td align="center">
      <b>Demo 2</b><br>
      Hansome kuavo is dancing.<br>
      <video src="./assets/materials/kuavo_dance.mp4" width="500" controls></video>
      <a href="./assets/materials/kuavo_dance.mp4">Watch kuavo dancing</a>
    </td>
  </tr>
</table>




## 1、概述

GMR(General Motion Retargeting)是人形机器人运动重定向技术框架。此仓库是通用运动重定向项目，用于实现人体运动数据重定向到人形机器人运动数据的高效转换。

## 2、安装GMR

拉取本仓库代码。

```bash
git clone https://gitee.com/leju-robot/leju-gmr.git
```

进入仓库目录，创建conda环境。

```bash
cd leju-gmr/
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

(1)重定向单个动作

```bash
python scripts/bvh_to_pkl.py \
--bvh_file <path_to_bvh_data> \
--bvh_fps <source_bvh_fps> \
--robot <robot_name> \
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

* `--robot`:指定运动重定向的目标机器人型号，当前支持`roban_s17`、`roban_s14`、`kuavo_s52`、`kuavo_s54`。

* `--save_path`:输出路径（可选）。支持以下模式：
  - 传入`.pkl`文件路径：按指定文件保存；
  - 传入目录路径：保存到`<dir>/<robot>/pkl/<bvh_stem>_<fps>.pkl`；
  - 不传：默认保存到`output/<robot>/pkl/<bvh_stem>_<fps>.pkl`。

* `--rate_limit`:参数用于限制机器人的运动重定向速率，使其与人体运动速率保持一致。若希望机器人以最快速率运动，移除该参数即可。

* `--motion_fps`:目标输出帧率（Hz，可选）。不指定时默认使用`--bvh_fps`。

* `--bvh_unit`:BVH 位置单位，支持`auto`、`mm`、`cm`、`m`，默认`auto`。若已知单位，建议显式指定（如`--bvh_unit cm`）。

* `--record_video_path [video_path]`:可视化录制参数（可选）：
  - 不传：不录制视频；
  - 仅传`--record_video_path`：默认保存到`output/<robot>/videos/<pkl_stem>.mp4`；
  - 传具体路径：按指定路径保存视频。

* `--format`:指定人体动画数据格式。支持`"leju", "lafan1", "qmai", "nokov"`。

(2)重定向文件夹中的批量动作：

```bash
python scripts/bvh_to_robot_dataset.py \
--src_folder <path_to_dir_of_bvh_data> \
--tgt_folder <path_to_dir_to_save_robot_data> \
--robot <robot_name>
```

批量重定向模式下默认不显示可视化效果。

参数说明：

* `--src_folder`:指定待处理文件的源根文件夹，此参数为必填项。

* `--tgt_folder`:指定重定向文件的保存目&#x5F55;**。**

* `--robot`:指定运动重定向的目标机器人型号。



### 6.2、从SMPL-X到机器人的运动重定向

适用于 AMASS 的 SMPL-X 拟合数据（`*_stageii.npz`，`surface_model_type: smplx`），文件包含 `pose_body` / `pose_hand` / `pose_jaw` / `pose_eye` / `root_orient` / `betas` / `trans` 等字段。

当前已支持的机器人型号：`roban_s17`、`kuavo_s45`、`kuavo_s54`。

注意事项: 安装 SMPL-X 后，若使用 SMPL-X pkl 格式文件，需将 `smplx/body_models.py` 文件中的文件扩展名 ext 从 npz 修改为 pkl。

(1)单段运动重定向

```bash
python scripts/smplx_to_robot.py \
--smplx_file <path_to_smplx_data.npz> \
--robot <robot_name> \
--save_path <path_to_save_robot_data.pkl> \
--rate_limit
```

默认情况下，MuJoCo 窗口将展示运动重定向后的机器人运动可视化效果。

参数说明：

* `--smplx_file`:指定要处理的 SMPL-X `.npz` 文件路径，此参数为必填参数。

* `--robot`:指定运动重定向的目标机器人型号，支持 `roban_s17` / `kuavo_s45` / `kuavo_s54`。

* `--save_path`:指定运动重定向文件保存路径，默认不保存。

* `--rate_limit`:参数用于限制机器人的运动重定向速率，使其与人体运动速率保持一致。若希望机器人以最快速率运动，移除该参数即可。

(2)文件夹批量运动重定向。

```bash
python scripts/smplx_to_robot_dataset.py \
--src_folder <path_to_dir_of_smplx_data> \
--tgt_folder <path_to_dir_to_save_robot_data> \
--robot <robot_name>
```

批量重定向模式下，默认不开启运动可视化功能。

参数说明：

* `--src_folder`:指定待处理文件的源根文件夹，此参数为必填项。

* `--tgt_folder`:指定重定向文件的保存目录。

* `--robot`:指定运动重定向的目标机器人型号。



### 6.3、从SMPL到机器人的运动重定向

适用于纯 SMPL 数据（`.npz` 的 `poses` 是 72 维，或 SMPL-H 的 156 维 — 内部会截断到前 22 个 body 关节，再通过 SMPL-X body model 计算 FK）。

当前已支持：`roban_s17`、`kuavo_s45`、`kuavo_s54`。

> 注意：如果你下载的 AMASS 文件是 `_stageii.npz` 后缀，通常已经是 SMPL-X 格式，应该用 `smplx_to_robot.py`；只有当 `npz` 的 `surface_model_type` 是 `smpl` 或 `smplh`（或者你拿到的是纯 SMPL 拟合的数据集如 HumanML3D），才走 SMPL 管道。

```bash
python scripts/smpl_to_robot.py \
--smpl_file <path_to_smpl_data.npz> \
--robot <robot_name> \
--save_path <path_to_save_robot_data.pkl> \
--rate_limit
```

参数语义同 `smplx_to_robot.py`。




### 6.4、可视化已保存的机器人动作

(1)可视化单个动作：

```bash
python scripts/vis_robot_motion.py \
--robot <robot_name> \
--robot_motion_path <path_to_save_robot_data.pkl>
```

若需录制视频，需添加参数 `--record_video `和` --video_path <your_video_path,mp4>`。

(2)可视化文件夹中的批量动作：

```bash
python scripts/vis_robot_motion_dataset.py \
--robot <robot_name> \
--robot_motion_folder <path_to_save_robot_data_folder>
```


### 6.5、预览 SMPL-X `.npz` 原始动作（不经过重定向）

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
> python -c "import pickle; d=pickle.load(open('output/<robot>/cmu/85_01.pkl','rb')); print(d['fps'])"
> ```

框架提供数据转换功能，可将机器人运动 `.pkl` 转换为 `.csv`，并按机器人型号进行严格 DoF 校验与重排（`Leg + Waist + Arm`）。

- 基本用法

```bash
# 单文件转换
python scripts/pkl_to_csv.py \
--robot <robot_name> \
--pkl_file <path_to_file.pkl>

# 文件夹批量转换
python scripts/pkl_to_csv.py \
--robot <robot_name> \
--folder <path_to_pkl_folder>
```

- 参数说明
  - `--robot`（必需）：机器人型号。当前支持 `roban_s14`、`roban_s17`、`kuavo_s52`、`kuavo_s54`。
  - `--pkl_file` / `--folder`（二选一，必需）：
    - `--pkl_file`：转换单个 `.pkl` 文件；
    - `--folder`：扫描目录下所有 `.pkl` 文件并批量转换。
  - `--output`（可选）：
    - 单文件模式：可传目标 `.csv` 文件路径，或输出目录；
    - 批量模式：作为输出目录使用；
    - 不传时默认输出到输入同级 `csv/` 子目录。
  - `--max_frames`（可选）：限制导出最大帧数（从第 0 帧开始）。

- 输出格式
  - 每行数据格式为：`[root_pos(x,y,z), root_rot(x,y,z,w), dof_pos(...)]`。
  - 其中 `dof_pos` 会按机器人配置强制输出为 `Leg -> Waist -> Arm` 顺序。

- 使用示例

```bash
# 示例1：转换单个 pkl 文件
python scripts/pkl_to_csv.py \
--robot roban_s14 \
--pkl_file output/roban_s14/pkl/0211_re_005_Skeleton_50fps.pkl

# 示例2：批量转换目录中的 pkl 文件
python scripts/pkl_to_csv.py \
--robot kuavo_s54 \
--folder output/kuavo_s54/pkl

# 示例3：指定输出目录并限制帧数
python scripts/pkl_to_csv.py \
--robot roban_s17 \
--folder output/roban_s17/pkl \
--output output/roban_s17/pkl/csv \
--max_frames 2000
```
## 8、适配目录(详见`general_motion_retargeting/params.py`)

### 8.1 机器人 × 数据格式 支持矩阵

| 机器人 | 标识 | DoF | qpos 维度 | SMPL-X | SMPL | BVH LAFAN1 | BVH Leju |
|--------|------|-----|-----------|:------:|:----:|:----------:|:--------:|
| Roban S17 | `roban_s17` | 21 (Leg×12 + Waist + Arm×8) | 28 | ✅ | ✅ | ✅ | ✅ |
| Kuavo S45 | `kuavo_s45` | 26 (Leg×12 + Arm×14) | 33 | ✅ | ✅ | TBD | TBD |
| Kuavo S54 | `kuavo_s54` | 27 (Leg×12 + Waist + Arm×14) | 36 | ✅ | ✅ | ✅ | ✅ |


### 8.2 数据格式 × 入口脚本

| 源格式 | 入口脚本 | 说明 |
|--------|---------|------|
| SMPL-X (.npz) | `scripts/smplx_to_robot.py` | AMASS `_stageii.npz` |
| SMPL / SMPL-H (.npz) | `scripts/smpl_to_robot.py` | 内部转 SMPL-X body model |
| BVH | `scripts/bvh_to_pkl.py --format <leju\|lafan1\|qmai\|nokov>` | 支持四种 BVH 子格式 |

### 8.3 辅助工具

| 脚本 | 用途 |
|------|------|
| `scripts/visualize_smplx_npz.py` | 用 aitviewer 预览 SMPL-X 原始动作 |
| `scripts/vis_robot_motion.py` | 可视化已保存的机器人 PKL |
| `scripts/pkl_to_csv.py` | PKL → CSV，按机器人 DoF 严格校验 |
| `scripts/batch_gmr_pkl_to_csv.py` | 批量 PKL → CSV（兼容 beyondmimic 格式） |
