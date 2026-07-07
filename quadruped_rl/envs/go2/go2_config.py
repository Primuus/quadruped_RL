from quadruped_rl.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Go2 专用配置。新增参数时请同步补充中文注释。


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
        num_observations = 45  # 观测维度，HIMLoco 风格，不直接输入机身线速度
        obs_mode = 'go2_45'  # 45 维观测模式，部署侧必须保持一致
        num_actions = 12  # 动作维度，Go2 一共 12 个关节

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 安全基线关闭速度命令课程学习，稳定后再单独启用
        max_curriculum = 2.0  # Go2 walk 阶段命令课程上限
        resampling_time = 10.0  # 命令重采样周期 [s]
        heading_command = True  # 使用 heading 误差生成偏航速度命令

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # 前向速度范围 [m/s]
            lin_vel_y = [-1.0, 1.0]  # 侧向速度范围 [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # 偏航角速度范围 [rad/s]
            heading = [-3.14, 3.14]  # 目标朝向范围 [rad]

    class control( LeggedRobotCfg.control ):
        # PD 参数与训练侧对齐
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad] 安全基线使用保守刚度
        damping = {'joint': 0.5}  # [N*m*s/rad] 安全基线使用保守阻尼
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
        self_collisions = 1  # 是否启用自碰撞过滤

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9  # 关节软限制比例
        base_height_target = 0.25  # 期望机身相对地面高度，基础 reward 默认不启用
        clearance_height_target = -0.20  # 机身坐标系下足端目标高度，启用 foot_clearance 时使用
        only_positive_rewards = True  # 安全基线裁剪负总奖励，降低早期训练不稳定性

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0  # 线速度跟踪奖励
            tracking_ang_vel = 0.5  # 角速度跟踪奖励
            lin_vel_z = -2.0  # 机身竖直速度惩罚
            ang_vel_xy = -0.05  # 横滚/俯仰角速度惩罚
            orientation = -0.2  # 姿态偏差惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚
            joint_power = -0.0  # 关节功率惩罚，安全基线默认关闭
            base_height = -0.0  # 机身高度偏差惩罚，安全基线默认关闭
            feet_air_time = 0.0  # 参考 HIMLoco walk，避免把步态推得过于跳跃
            foot_clearance = -0.0  # 足端抬脚约束，安全基线默认关闭
            action_rate = -0.01  # 动作一阶差分惩罚
            smoothness = -0.0  # 二阶动作平滑惩罚，安全基线默认关闭
            torques = -0.0002  # 力矩惩罚，安全基线保留基础约束
            dof_vel = -0.0  # 关节速度惩罚
            collision = -0.0  # walk 阶段先关闭非足端碰撞硬惩罚
            feet_stumble = -0.0  # 足端绊倒惩罚
            stand_still = -0.0  # 静止姿态惩罚
            feet_contact_forces = -0.0  # 足端接触力惩罚
            dof_pos_limits = -0.0  # 关节位置限制惩罚
            dof_vel_limits = -0.0  # 关节速度限制惩罚
            torque_limits = -0.0  # 力矩限制惩罚


class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 熵奖励系数，保持一定探索

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 10000  # 最大训练迭代次数
        run_name = 'rough'  # 这次训练的运行名
        experiment_name = 'go2'  # 实验名
