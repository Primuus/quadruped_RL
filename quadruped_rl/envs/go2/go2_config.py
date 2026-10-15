from quadruped_rl.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Go2 机器人实例配置。通用四足逻辑在 base 中，新增机器人时复制本文件并替换资产、关节和任务参数。


class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]  # 初始机身位置 [m]
        default_joint_angles = {  # action=0 时的默认关节角目标 [rad]
            'FL_hip_joint': 0.1,  # 左前髋关节
            'RL_hip_joint': 0.1,  # 左后髋关节
            'FR_hip_joint': -0.1,  # 右前髋关节
            'RR_hip_joint': -0.1,  # 右后髋关节

            'FL_thigh_joint': 0.8,  # 左前大腿关节
            'RL_thigh_joint': 1.0,  # 左后大腿关节
            'FR_thigh_joint': 0.8,  # 右前大腿关节
            'RR_thigh_joint': 1.0,  # 右后大腿关节

            'FL_calf_joint': -1.5,  # 左前小腿关节
            'RL_calf_joint': -1.5,  # 左后小腿关节
            'FR_calf_joint': -1.5,  # 右前小腿关节
            'RR_calf_joint': -1.5,  # 右后小腿关节
        }

    class env(LeggedRobotCfg.env):
        num_one_step_observations = 45  # 单步观测维度，HIMLoco 风格，不直接输入机身线速度
        history_length = 6  # 历史观测帧数，当前 HIM policy 固定使用 6 帧
        num_observations = num_one_step_observations * history_length  # policy 输入维度：45*6=270
        num_privileged_external_force_obs = 3  # privileged obs 外力/扰动占位维度，未启用扰动时填 0
        num_privileged_height_scan = 187  # privileged obs 高度采样维度，平地或未启用高度采样时填 0
        num_one_step_privileged_obs = num_one_step_observations + 3 + num_privileged_external_force_obs + num_privileged_height_scan  # critic 特权观测：45 + base_lin_vel3 + 外力3 + 高度187 = 238
        num_privileged_obs = num_one_step_privileged_obs  # critic 输入维度；actor 仍只使用历史观测
        obs_mode = 'quadruped_history_45x6'  # 四足 HIM 观测模式：6 帧 45 维历史观测
        num_actions = 12  # 动作维度，当前 Go2 实例一共 12 个关节

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 当前关闭速度命令课程学习，稳定后再单独启用
        max_curriculum = 2.0  # 当前机器人 walk 阶段命令课程上限
        resampling_time = 5.0  # 命令重采样周期 [s]，增加站立、行走和转向之间的切换覆盖
        heading_command = False  # 直接训练持续偏航速度命令，与遥控器 yaw 输入一致
        mode_probabilities = [0.25, 0.25, 0.30, 0.20]  # 原地转向、边走边转、平移、站立
        min_translation_speed = 0.12  # 平移场景最小速度 [m/s]，覆盖低速且高于站立阈值
        min_turn_yaw = 0.12  # 原地转向最小偏航速度 [rad/s]，覆盖慢转且高于站立阈值
        max_translation_speed = 1.0  # 平面速度模长上限 [m/s]，同时限制斜向运动速度
        low_speed_probability = 0.30  # 30% 命令从低速子集采样，其余保留完整速度范围
        low_speed_scale = 0.35  # 低速子集将 XY/yaw 的采样范围缩放到完整范围的 35%

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # 前向速度范围 [m/s]
            lin_vel_y = [-1.0, 1.0]  # 侧向速度范围 [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # 偏航角速度范围 [rad/s]，参考 legged_gym 保持较稳训练
            heading = [-3.14, 3.14]  # 目标朝向范围 [rad]

    class control( LeggedRobotCfg.control ):
        # PD 参数与训练侧对齐
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad] 当前刚度
        damping = {'joint': 0.5}  # [N*m*s/rad] 当前阻尼
        # target angle = action_scale * action + default_angle
        action_scale = 0.25  # 策略动作到关节角偏移的缩放
        # 一个策略动作在仿真中重复的子步数
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{QUADRUPED_RL_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'  # Go2 URDF 路径
        name = "go2"  # 仿真中的机器人名称
        foot_name = "foot"  # 足端 body 名称
        penalize_contacts_on = ["thigh", "calf"]  # 大腿/小腿碰撞时施加惩罚
        terminate_after_contacts_on = ["base"]  # 机身触地时终止 episode
        self_collisions = 1  # Isaac Gym 正数过滤值会关闭全部自碰撞；保持现有 Go2 任务行为不变

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True  # 随机化地面摩擦，参考 legged_gym 的基础鲁棒性配置
        friction_range = [0.4, 1.25]  # 摩擦系数随机范围，包含当前 MuJoCo 足端的 0.4 配置值
        randomize_base_mass = False  # 暂不随机化机身质量，避免一次性引入过多变量
        push_robots = False  # 暂不启用随机外推，等基础步态稳定后再打开

    class noise(LeggedRobotCfg.noise):
        add_noise = True  # 启用观测噪声；具体噪声幅值沿用 base.noise.noise_scales

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9  # 关节软限制比例
        base_height_target = 0.28  # 站立和运动共用的目标机身高度 [m]，区别于初始落地高度
        base_height_tolerance = 0.03  # 高度 0.25-0.31 m 内不惩罚，允许正常步态上下浮动
        base_height_sigma = 0.05  # 超出高度容差后的误差尺度 [m]，采用连续平方惩罚
        clearance_height_target = -0.20  # 机身坐标系下足端目标高度，启用 foot_clearance 时使用
        clearance_height_sigma = 0.05  # 仅对摆动足计算归一化高度误差，支撑足不被要求抬升
        turn_step_min_yaw = 0.12  # 慢速原地转向也可获得有效换步奖励，和命令下限一致
        turn_step_reward_interval = 0.5  # 每足最多积累 0.5 s 预算，避免通过高频跺脚刷分
        only_positive_rewards = False  # 保留负总奖励，区分低趴、非足端支撑等不同程度的错误

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0  # 线速度跟踪奖励
            tracking_ang_vel = 0.5  # 角速度跟踪奖励
            lin_vel_z = -2.0  # 机身竖直速度惩罚
            ang_vel_xy = -0.05  # 横滚/俯仰角速度惩罚
            orientation = -0.2  # 姿态偏差惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚
            joint_power = -0.0  # 关节功率惩罚，当前关闭
            base_height = -0.4  # 站立、平移和转向共用的高度容差惩罚
            feet_air_time = 0.0  # 参考 HIMLoco walk，避免把步态推得过于跳跃
            foot_clearance = -0.01  # 摆动足抬脚惩罚，使用归一化高度误差且排除支撑足
            foot_slip = -0.10  # 惩罚接触地面时的足端水平滑移
            turn_step = 0.1  # 转向换步的长期平均奖励上限约 0.1/s，要求同时跟踪 XY/yaw
            action_rate = -0.01  # 动作一阶差分惩罚
            smoothness = -0.01  # 二阶动作平滑惩罚，参考 HIMLoco walk 配置
            torques = -0.0002  # 力矩惩罚，安全基线保留基础约束
            dof_vel = -0.0  # 关节速度惩罚
            collision = -1.0  # 加强持续大腿/小腿触地惩罚，但不直接终止回合
            feet_stumble = -0.0  # 足端绊倒惩罚
            stand_still = -0.1  # 零命令下的默认站姿偏差惩罚，不作用于原地转向
            stand_contacts = -0.4  # 站立时缺少足端支撑的比例惩罚，四足支撑时为 0
            stand_height = -0.0  # 高度统一由 base_height 约束，避免站立时重复惩罚
            feet_contact_forces = -0.0  # 足端接触力惩罚
            dof_pos_limits = -10.0  # 关节位置限制惩罚，参考 Unitree Go2 配置
            dof_vel_limits = -0.0  # 关节速度限制惩罚
            torque_limits = -0.0  # 力矩限制惩罚


class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 熵奖励系数，保持一定探索

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'  # HIM actor-critic：history -> estimator -> actor
        algorithm_class_name = 'HIMPPO'  # HIM-PPO，额外训练 estimator
        runner_class_name = 'HIMOnPolicyRunner'  # 保存 estimator 更新所需的 next privileged obs
        max_iterations = 10000  # 最大训练迭代次数
        run_name = 'rough'  # 这次训练的运行名
        experiment_name = 'go2'  # 实验名
