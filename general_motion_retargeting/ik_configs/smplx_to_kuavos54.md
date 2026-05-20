# SMPL-X → Kuavo S54 IK 配置说明

配套文件：[`smplx_to_kuavos54.json`](./smplx_to_kuavos54.json)

把 SMPL-X（AMASS `.npz`，含 `root_orient` / `pose_body` / `betas` / `trans`）重定向到 Kuavo S54 机器人。SMPL-X 关节名（`pelvis` / `left_hip` / `left_shoulder` …）→ S54 机器人 body 链（`base_link` / `leg_l3_link` / `zarm_l3_link` …）。

## 顶层字段

| 字段 | 含义 |
|------|------|
| `robot_root_name` | MuJoCo XML 中机器人的根 body 名称（IK 求解器以它为参考链根） |
| `human_root_name` | SMPL-X 数据中根关节名，用于计算其他关节的局部位置 |
| `ground_height` | 地面 z 轴偏移（米），用于 `offset_human_data_to_ground()`，正值 = 人体整体上抬 |
| `human_height_assumption` | 假设的人体身高（米）。实际身高（从 `betas` 计算）与此值的比例会作为缩放系数应用到 `human_scale_table` 上 |
| `use_ik_match_table1` | 是否启用 table1 阶段 IK |
| `use_ik_match_table2` | 是否启用 table2 阶段 IK（两阶段求解时先 1 后 2） |

## `human_scale_table`

人体每个关节相对根（`pelvis`）的位置缩放系数：

- `1.0` 表示不缩放
- `> 1.0` 表示拉长该肢段（如 `left_shoulder: 1.2` 让肩部位置外扩 20%，匹配机器人较宽的肩宽）

当前文件中只有 `left_shoulder` / `right_shoulder` 是 `1.2`，其他全部 `1.0`。

## `ik_match_table1` 条目格式

每条记录形如：

```jsonc
"机器人 body 名": [
    "人体关节名",
    pos_weight,             // 跟踪位置的代价权重，0 = 不约束
    rot_weight,             // 跟踪朝向的代价权重，0 = 不约束
    [px, py, pz],           // 局部坐标系下额外位置偏移（米），补偿骨长差
    [qw, qx, qy, qz]        // 旋转偏移四元数（wxyz）
                            // target = human_quat * rot_offset
                            // 把人体关节朝向映射到机器人 body 朝向
]
```

权重直觉：

- `pos_weight` 量级 **10²~10³**（米尺度误差）
- `rot_weight` 量级 **10¹**（弧度尺度误差）
- 两个量级需要可比，IK cost 不会被某一项压制
- 脚踝 / 根节点等关键 body → 大权重
- 手臂等末端 → 较小权重，避免过约束

## 当前文件每个映射的设计意图

### 根 / 躯干

| Body | 人体关节 | pos / rot | 作用 |
|------|---------|-----------|------|
| `base_link` | `pelvis` | 500 / 200 | 浮动基座，全 6 DoF 强约束，确保整体位置和朝向跟人体走 |
| `waist_yaw` | `spine1` | 0 / 100 | 腰部 yaw 关节，只跟旋转（位置已由 `base_link` 决定） |

### 肩部锁骨

| Body | 人体关节 | pos / rot | 作用 |
|------|---------|-----------|------|
| `zarm_l1_link` / `zarm_r1_link` | `left_collar` / `right_collar` | 120 / 10 | 锚定肩部位置，避免 S54 肩部 3 DoF 多解时 IK 乱选解 |

### 腿部

| Body | 人体关节 | pos / rot | 作用 |
|------|---------|-----------|------|
| `leg_l/r3_link`（大腿） | `left/right_hip` | 150 / 5 | 髋部位置锚定 |
| `leg_l/r4_link`（小腿） | `left/right_knee` | 150 / 5 | 膝盖位置锚定，防止左右膝姿态不对称 |
| `leg_l/r6_link`（脚） | `left/right_ankle` | 400 / 100 | 脚踝最强约束，保证脚部位置精确 |
| `leg_l/r5_link`（脚尖） | `left/right_foot` | 120 / 0 | 脚趾位置，配合脚踝维持正确的脚踝 pitch |

### 手臂

| Body | 人体关节 | pos / rot | 作用 |
|------|---------|-----------|------|
| `zarm_l/r3_link`（上臂末端） | `left/right_shoulder` | 150 / 10 | 位置 + 朝向跟踪上臂 |
| `zarm_l/r4_link`（前臂） | `left/right_elbow` | 150 / 5~10 | 位置 + 朝向跟踪前臂 |
| `zarm_l/r7_link`（手腕） | `left/right_wrist` | 150 / 8 | 位置 + 朝向跟踪手腕 |

> 手臂各关节都加了 `pos_w=150` 位置锚定，避免 S54 肩部 3 DoF 的 IK 多解空间过大导致手臂"乱甩"。

### 头部

| Body | 人体关节 | pos / rot | 作用 |
|------|---------|-----------|------|
| `zhead_1_link` / `zhead_2_link` | `neck` / `head` | 0 / 10 | 纯朝向约束 |

## 关于旋转偏移的特别说明

- 左臂三个关节（`zarm_l3/4/7_link`）当前用统一的 `[0.7071, 0, -0.7071, 0]`（纯 R_y(-90°)）
- 右臂的 `zarm_r4_link` 还保留 Leju S54 时期的 `[0.0923, 0.7011, 0.0923, 0.7011]`（带 ~7.5° 倾斜微调），那是给 BVH 数据补偿用的；SMPL-X 数据上不需要
- 左右肩部 / 手腕用 `[0.7071, 0, -0.7071, 0]` vs `[0, 0.7071, 0, 0.7071]`，不是简单 Y-mirror，而是各自针对人体到机器人坐标系的独立标定结果
- **左脚踝（`leg_l6_link`）用 `[0.5610, -0.4300, -0.4300, -0.5610]`**（标准值是 `[0.5, -0.5, -0.5, -0.5]`，绕 Y 轴加了 +13° 修正），用于补偿 AMASS CMU 01_01 数据中左踝相对右踝持续多 ~14% dorsiflexion 的系统性偏差。若换其他 AMASS 数据集，可能需要重新调或恢复成标准值

## 关于 IK 多解问题（已知遗留）

S54 肩部有 3 个连续 DoF（pitch + roll + yaw），同一个上臂目标朝向可由多组关节角度达成。当前 `zarm_l1_link → left_collar` 的 `pos_w=120` 锚定肩部位置缓解了一部分多解，但**左臂偶尔仍会出现 zarm_l3_joint（肩 yaw）选解异常**的情况。

彻底解决需要在 `motion_retarget.py` 加 `mink.PostureTask`，让 IK 优先选择靠近 rest pose 的关节配置。目前 JSON 调整已经到极限，进一步修复需要代码层面改动。

## `ik_match_table2`

空对象 `{}` 表示不启用两阶段 IK。如要启用，把 `use_ik_match_table2` 改为 `true` 并在此填入第二阶段映射（通常权重不同，用于精修）。
