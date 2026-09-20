
import mink
import mujoco as mj
import numpy as np
import json
import re
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        use_collision_avoidance: bool=False,
        collision_avoidance_config: dict=None,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.posture_weight = float(ik_config.get("posture_weight", 0.0) or 0.0)

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        # Self-collision avoidance (e.g. keep the arms away from the thighs).
        # 支持多个 CollisionAvoidanceLimit（不同碰撞对可用不同参数）。
        self.collision_avoidance_limits = []
        if use_collision_avoidance:
            self.collision_avoidance_limits = self.build_collision_avoidance_limit(
                collision_avoidance_config
            ) or []
            if self.collision_avoidance_limits:
                self.ik_limits.extend(self.collision_avoidance_limits)
                if verbose:
                    n_pairs = sum(len(l.geom_id_pairs) for l in self.collision_avoidance_limits)
                    print(f"[GMR] Collision avoidance enabled: {n_pairs} geom pairs "
                          f"in {len(self.collision_avoidance_limits)} limit(s)")
            elif verbose:
                print("[GMR] Collision avoidance requested but no geom pairs were built; skipping")
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def build_collision_avoidance_limit(self, config: dict = None):
        """Build a mink.CollisionAvoidanceLimit for self-collision avoidance.

        Collision geoms are grouped by the body they are attached to, and body
        groups are matched against `body_group_patterns` (regex, first match
        wins). Every group pair listed in `group_pairs` is then checked for
        collisions by the QP solver.

        Currently tuned for the aelos robot (arm geoms vs. leg/torso geoms).
        """
        config = config or {}
        # regex patterns mapping a body name to a collision group
        body_group_patterns = config.get("body_group_patterns", {
            "left_hand_end": r"^L_Hand_duoji__3$",
            "right_hand_end": r"^R_Hand_duoji__3_$",
            "left_leg": r"^L_Leg_1_$|^L_leg_duoji__2$",
            "right_leg": r"^R_Leg__1$|^R_Leg_duoji_2_2$",
            "torso": r"^robot_base$",
        })
        # which group pairs should avoid each other.
        # 每项可以是 (组A, 组B) 使用默认参数，或 (组A, 组B, {参数覆盖}) 单独
        # 指定 gain/minimum_distance/detection_distance（如手-躯干用更小的
        # 安全距离，避免胸前动作被推得太开）。
        group_pairs = config.get("group_pairs", [
            ("left_hand_end", "left_leg"),
            ("left_hand_end", "right_leg"),
            ("right_hand_end", "left_leg"),
            ("right_hand_end", "right_leg"),
            ("left_hand_end", "torso", {"minimum_distance": 0.01, "detection_distance": 0.03}),
            ("right_hand_end", "torso", {"minimum_distance": 0.01, "detection_distance": 0.03}),
        ])
        gain = config.get("gain", 0.5)
        minimum_distance = config.get("minimum_distance", 0.035)
        detection_distance = config.get("detection_distance", 0.05)

        # collect collision geoms (contype/conaffinity != 0) per group
        group_geoms = {group: [] for group in body_group_patterns}
        for geom_id in range(self.model.ngeom):
            if self.model.geom_contype[geom_id] == 0 and self.model.geom_conaffinity[geom_id] == 0:
                continue  # visual-only geom
            body_id = self.model.geom_bodyid[geom_id]
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            for group, pattern in body_group_patterns.items():
                if re.match(pattern, body_name):
                    group_geoms[group].append(geom_id)
                    break

        if any(len(geoms) == 0 for geoms in group_geoms.values()):
            print("[GMR] [warn] Some collision groups have no geoms:",
                  {k: len(v) for k, v in group_geoms.items()})
            return None

        # 按参数组合聚合碰撞对：参数相同的对共用一个 limit（减少 QP 约束
        # 行数），参数不同的对（如手-躯干）单独建 limit。
        pairs_by_params = {}  # (gain, min_dist, detect_dist) -> [geom_pair, ...]
        for entry in group_pairs:
            group_a, group_b = entry[0], entry[1]
            overrides = entry[2] if len(entry) > 2 else {}
            if group_a not in group_geoms or group_b not in group_geoms:
                continue
            params = (
                overrides.get("gain", gain),
                overrides.get("minimum_distance", minimum_distance),
                overrides.get("detection_distance", detection_distance),
            )
            pairs_by_params.setdefault(params, []).append(
                (group_geoms[group_a], group_geoms[group_b])
            )

        limits = []
        for (g, min_d, det_d), collision_pairs in pairs_by_params.items():
            limits.append(mink.CollisionAvoidanceLimit(
                self.model,
                collision_pairs,
                gain=g,
                minimum_distance_from_collisions=min_d,
                collision_detection_distance=det_d,
            ))
        return limits or None

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        # 将初始 qpos 夹紧到关节限位范围内。
        # 模型默认 qpos 全为 0，当 XML 中收紧了关节限位（如 aelos 的 arm2
        # 下限从 0 改为 0.1）后，默认姿态会落在限位之外；IK 从限位外的
        # 姿态开始求解时，ConfigurationLimit 要求的恢复速度会超过速度
        # 限制，导致 QP 无解（NoSolutionFound）而崩溃。
        for jnt in range(self.model.njnt):
            if self.model.jnt_type[jnt] == mj.mjtJoint.mjJNT_FREE:
                continue  # 自由关节（浮动基座）没有限位
            if not self.model.jnt_limited[jnt]:
                continue  # 未声明 limited 的关节不限位
            padr = self.model.jnt_qposadr[jnt]
            lo, hi = self.model.jnt_range[jnt]
            q = self.configuration.data.qpos[padr]
            self.configuration.data.qpos[padr] = min(max(q, lo), hi)
        self.configuration.update()
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # Weak regularizer toward the default pose. On aelos this stops sequential
        # IK from walking into joint limits when the human is nearly standing.
        posture_weight = float(getattr(self, "posture_weight", 0.0) or 0.0)
        if posture_weight > 0.0:
            posture_task = mink.PostureTask(self.model, cost=posture_weight)
            posture_task.set_target_from_configuration(self.configuration)
            if self.use_ik_match_table1:
                self.tasks1.append(posture_task)
            elif self.use_ik_match_table2:
                self.tasks2.append(posture_task)

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver,
                damping=self.damping, limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver,
                    damping=self.damping, limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver,
                damping=self.damping, limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver,
                    damping=self.damping, limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            # BVH/SMPL may include bones not referenced by the current IK table; those have no offset entry.
            if body_name not in rot_offsets or body_name not in pos_offsets:
                offset_human_data[body_name] = [pos, quat]
                continue
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)

            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
