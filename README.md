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
python scripts/bvh_to_robot.py \
--bvh_file output/BVH/leju/dance_bj_01_Skeleton_002.bvh \
--robot roban_s14 \
--save_path output/PKL/roban_dance.pkl \
--rate_limit \
--format leju
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

### 6.1、从BVH重定向至机器人(推荐使用，适配kuavo_s52与roban_s14)

(1)重定向单个动作

```bash
# single motion
python scripts/bvh_to_robot.py \
--bvh_file <path_to_bvh_data> \
--robot <path_to_robot_data> \
--save_path <path_to_save_robot_data.pkl> \
--rate_limit \
--format <format>
```

默认情况下，MuJoCo 窗口中展示重定向后机器人动作的可视化效果。

参数说明：

* `--bvh_file`:指定要处理的 BVH 文件路径，此参数为必填参数。

* `--robot`:指定运动重定向的目标机器人型号。

* `--save_path`:指定运动重定向文件保存路径，默认值为不保存。

* `--rate_limit`:参数用于限制机器人的运动重定向速率，使其与人体运动速率保持一致。若希望机器人以最快速率运动，移除该参数即可。

* `--record_video`:默认`False`，为布尔值参数，可决定是否录制可视化过程为视频

* `--video_path`:默认`output/videos/example.mp4`录制视频的保存路径

* `--format`: 指定人体动画数据的格&#x5F0F;**。**&#x4EC5;支持`"leju", "lafan1", "nokov"`三种解析格式，默认值为`leju`。决定 BVH 文件的加载方式，`format=leju`时调用`load_leju_bvh_file`；否则调用`load_bvh_file`并传入`format`参数。

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



### 6.2、从SMPL-X到机器人的运动重定向(目前仅适配kuavo_s45)

注意事项: 安装 SMPL-X 后，若使用 SMPL-X pkl 格式文件，需将 `smplx/body_models.py `文件中的文件扩展名 ext 从 npz 修改为 pkl。

(1)单段运动重定向

```bash
python scripts/smplx_to_robot.py \
--smplx_file <path_to_smplx_data> \
--robot <path_to_robot_data> \
--save_path <path_to_save_robot_data.pkl> \
--rate_limit
```

默认情况下，MuJoCo 窗口将展示运动重定向后的机器人运动可视化效果。

参数说明：

* `--smplx_file`:指定要处理的 SMPLX文件路径，此参数为必填参数。

* `--robot`:指定运动重定向的目标机器人型号。

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




### 6.3、可视化已保存的机器人动作

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
## 7、数据转换（bvh_to_csv）
框架提供数据转换功能，可以选择将bvh数据格式转换为csv格式。
- 基本用法

```bash
python transfer/batch_gmr_pkl_to_csv.py --folder <包含pkl文件的文件夹路径>
```

-  参数说明
  `--folder` (必需): 包含pkl文件的文件夹路径
  - 脚本会扫描该文件夹下所有`.pkl`文件
  - CSV文件将保存在**该文件夹下的`csv`子文件夹**中

- 使用示例

```bash
# 示例1: 转换output文件夹中的所有pkl文件
python transfer/batch_gmr_pkl_to_csv.py --folder GMR/output

# 示例2: 转换指定文件夹中的pkl文件
python transfer/batch_gmr_pkl_to_csv.py --folder /path/to/pkl/files
```
## 8、适配目录(详见`general_motion_retargeting/params.py`)

| Assigned ID | Robot/Data Format | Robot DoF | SMPLX ([AMASS](https://amass.is.tue.mpg.de/), [OMOMO](https://github.com/lijiaman/omomo_release)) | BVH [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)| FBX ([OptiTrack](https://www.optitrack.com/)) |  BVH (Leju) | PICO ([XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)) | More formats coming soon | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Kuavo S52 `kuavo_s52` | Leg (2\*6) + Waist (1) + Arm (2\*7) = 27 | TBD | ✅ | TBD |  ✅ | TBD |
| 1 | Roban S14 `roban_s14` | Leg (2\*6) + Waist (1) + Arm (2\*5) = 23 | TBD | ✅ | TBD | ✅ | TBD |
| 2 | Kuavo S45 `kuavo_s45` | Leg (2\*6) + Arm (2\*7) = 26 | ✅ | TBD | TBD | TBD | TBD |