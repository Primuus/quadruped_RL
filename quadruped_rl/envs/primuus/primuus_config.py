"""Primuus 独立任务配置；只继承通用 base，不依赖 Go2。"""

import json
from pathlib import Path

from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PrimuusCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # 沿用训练落地高度，不是部署初始高度或奖励目标
        dof_pos_scale_range = [0.9, 1.1]  # 缩小复位扰动，避免小腿初始角越出原始机械限位
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': -0.75,  # 大腿轴为 -Y，不能复制 Go2 正角度站姿
            'FL_calf_joint': -1.5,
            'FR_hip_joint': 0.0,
            'FR_thigh_joint': -0.75,
            'FR_calf_joint': -1.5,
            'RL_hip_joint': 0.0,
            'RL_thigh_joint': -0.75,
            'RL_calf_joint': -1.5,
            'RR_hip_joint': 0.0,
            'RR_thigh_joint': -0.75,
            'RR_calf_joint': -1.5,
        }

    class env(LeggedRobotCfg.env):
        num_one_step_observations = 45
        history_length = 6
        num_observations = num_one_step_observations * history_length
        num_privileged_external_force_obs = 3  # 沿用现有零占位契约
        num_privileged_height_scan = 187  # 当前平地训练仍为零占位
        num_one_step_privileged_obs = num_one_step_observations + 3 + num_privileged_external_force_obs + num_privileged_height_scan
        num_privileged_obs = num_one_step_privileged_obs
        obs_mode = 'quadruped_history_45x6'
        num_actions = 12

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.0
        resampling_time = 5.0  # [s]；所有运动状态统一保持5秒并直接切换
        heading_command = False
        mode_names = ('turn', 'moving_turn', 'translate', 'stand', 'forward')
        mode_probabilities = [0.25, 0.25, 0.15, 0.20, 0.15]  # 原地转向、边走边转、平移、站立、纯前进
        min_translation_speed = 0.12  # [m/s]
        min_turn_yaw = 0.12  # [rad/s]
        max_translation_speed = 1.0  # 平移和边走边转的 XY 合成速度上限；纯前进使用 lin_vel_forward
        low_speed_probability = 0.30
        low_speed_scale = 0.35

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_forward = [0.12, 2.0]  # 纯前进速度 [m/s]，vy 和 yaw 固定为零
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 60.0}  # 关节端 PD 比例增益，与 primuus.yaml 同步
        damping = {'joint': 1.5}
        action_scale = 0.25
        decimation = 10  # 10 * 2 ms = 20 ms，策略 50 Hz，PD 每个物理步更新，即 500 Hz

    class asset(LeggedRobotCfg.asset):
        file = '{QUADRUPED_RL_ROOT_DIR}/resources/robots/primuus/urdf/primuus.urdf'
        name = 'primuus'
        joint_dynamics_file = '{QUADRUPED_RL_ROOT_DIR}/resources/robots/primuus/joint_dynamics.json'
        foot_name = '_foot_link'  # 仅四个独立足端，不把小腿碰撞计入足端
        penalize_contacts_on = ['thigh', 'calf']  # calf 同时选中固定碰撞载体
        terminate_after_contacts_on = ['base_link']
        collapse_fixed_joints = False  # 保留足端和小腿碰撞载体的八个固定关节
        replace_cylinder_with_capsule = False  # 与 MuJoCo 保持相同圆柱半径和端面，不自动扩包
        flip_visual_attachments = False  # STL 不使用 Go2 DAE 的视觉翻转
        self_collisions = -1  # 保留命名过滤位，不关闭整机自碰撞
        with (Path(QUADRUPED_RL_ROOT_DIR) / 'resources/robots/primuus/collision/self_collision_exclusions.json').open(encoding='utf-8') as _handle:
            self_collision_exclusions = json.load(_handle)
        thickness = 0.001  # [m]
        contact_offset = 0.001  # 显式写入每个形状，不只设置 sim.physx
        rest_offset = 0.0005  # [m]；与模型 CPU PhysX 接触检查条件一致

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.25]
        randomize_base_mass = False
        push_robots = False

    class noise(LeggedRobotCfg.noise):
        add_noise = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.28  # [m]；奖励目标，不代表默认角纯 PD 承重后的实际高度
        base_height_tolerance = 0.03  # 0.25-0.31 m 内不惩罚正常高度浮动
        base_height_sigma = 0.05
        clearance_height_target = -0.20  # 足端相对机身的摆动目标高度 [m]
        clearance_height_sigma = 0.05
        turn_step_min_yaw = 0.12
        turn_step_reward_interval = 0.5
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            base_height = -0.4
            foot_clearance = -0.01
            foot_slip = -0.10
            turn_step = 0.1
            action_rate = -0.01
            smoothness = -0.01
            torques = -0.0002
            collision = -1.0
            stand_still = -0.1
            stand_contacts = -0.4
            dof_pos_limits = -10.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # [s]；部署仍为 0.5 ms，两侧策略周期均为 20 ms
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.001
            rest_offset = 0.0005


class PrimuusCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        runner_class_name = 'HIMOnPolicyRunner'
        max_iterations = 10000
        run_name = 'rough'
        experiment_name = 'primuus'  # checkpoint / 导出文件与 Go2 隔离，不恢复 Go2 权重
