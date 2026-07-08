from .base_config import BaseConfig

# 新增配置项时请同步补充中文注释，保持训练和部署配置可读。


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # 并行环境数量，越大采样越快，但显存占用也更高
        num_one_step_observations = 45  # 单步 actor 观测维度；HIMLoco 四足默认 45 维观测
        history_length = 6  # 历史观测帧数；阶段 6 使用 6 帧历史
        num_observations = num_one_step_observations * history_length  # policy 输入维度：45*6=270
        num_privileged_external_force_obs = 3  # privileged obs 中的外力/扰动占位维度，未启用扰动时填 0
        num_privileged_height_scan = 187  # privileged obs 中的高度采样维度，平地或未启用高度采样时填 0
        num_one_step_privileged_obs = num_one_step_observations + 3 + num_privileged_external_force_obs + num_privileged_height_scan  # critic 特权观测：当前 45 维 + 机身线速度 3 维 + 外力 3 维 + 高度 187 维
        num_privileged_obs = num_one_step_privileged_obs  # critic 输入维度；actor 仍只使用 num_observations
        num_actions = 12  # 当前 Go2 动作维度；算法应从配置读取，不在算法中硬编码
        obs_mode = 'go2_history_45x6'  # 当前 Go2 观测模式：6 帧 45 维历史观测，后续可扩展其他机器人模式
        env_spacing = 3.  # 环境之间的间隔，平地/网格地形时用于避免互相碰撞
        send_timeouts = True  # 是否把超时信息传给算法
        episode_length_s = 20  # 单个 episode 的最大时长 [s]
        test = False  # 是否进入测试模式

    class terrain:
        mesh_type = 'plane'  # 地形类型：none / plane / heightfield / trimesh
        horizontal_scale = 0.1  # 地形网格水平分辨率 [m]
        vertical_scale = 0.005  # 地形高度分辨率 [m]
        border_size = 25  # 地形边界缓冲区大小 [m]
        curriculum = False  # 是否启用地形课程学习；安全基线默认关闭
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.  # 弹性系数
        measure_heights = False  # 是否在观测中加入地形高度采样；阶段 3 历史观测不使用高度采样
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 足底前后方向采样点
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 足底左右方向采样点
        selected = False  # 是否固定训练某一种地形
        terrain_kwargs = None  # 选定地形时的额外参数字典
        max_init_terrain_level = 5  # 课程学习初始难度等级
        terrain_length = 8.  # 单块地形长度 [m]
        terrain_width = 8.  # 单块地形宽度 [m]
        num_rows = 10  # 地形行数，对应难度层数
        num_cols = 20  # 地形列数，对应地形类型数量
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]  # 各类地形采样比例
        slope_treshold = 0.75  # 斜率阈值，超过后会被修正为垂直表面

    class commands:
        curriculum = False  # 是否启用速度命令课程学习；安全基线默认关闭
        max_curriculum = 2.0  # 命令课程学习上限
        num_commands = 4  # 命令维度：前向速度、侧向速度、偏航角速度、heading
        resampling_time = 10.  # 命令重采样周期 [s]
        heading_command = True  # 是否根据朝向误差自动计算偏航速度命令

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # 前向速度范围 [m/s]
            lin_vel_y = [-1.0, 1.0]  # 侧向速度范围 [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # 偏航角速度范围 [rad/s]
            heading = [-3.14, 3.14]  # 朝向目标范围 [rad]

    class init_state:
        pos = [0.0, 0.0, 1.]  # 初始机身位置 [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始四元数姿态
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 [rad/s]
        default_joint_angles = {  # action=0 时的目标关节角
            "joint_a": 0.,
            "joint_b": 0.
        }

    class control:
        control_type = 'P'  # 控制类型：P 位置 / V 速度 / T 力矩
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # PD 比例增益 [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # PD 阻尼增益 [N*m*s/rad]
        action_scale = 0.5  # policy 动作到目标关节角偏移的缩放系数
        decimation = 4  # 一个策略动作重复多少个仿真子步

    class asset:
        file = ""  # 机器人 URDF 文件路径
        name = "legged_robot"  # 机器人在仿真中的 actor 名称
        foot_name = "None"  # 足端 body 名称，用于接触和状态索引
        penalize_contacts_on = []  # 需要惩罚接触的 body 列表
        terminate_after_contacts_on = []  # 触碰后直接终止的 body 列表
        disable_gravity = False  # 是否关闭重力
        collapse_fixed_joints = True  # 是否合并固定关节
        fix_base_link = False  # 是否固定机身底座
        default_dof_drive_mode = 3  # 默认驱动模式：3 表示力矩模式
        self_collisions = 0  # 是否启用自碰撞过滤
        replace_cylinder_with_capsule = True  # 是否用胶囊体替代圆柱体，提升稳定性
        flip_visual_attachments = True  # 是否翻转视觉贴图坐标系

        density = 0.001  # 机器人密度
        angular_damping = 0.  # 角阻尼
        linear_damping = 0.  # 线阻尼
        max_angular_velocity = 1000.  # 最大角速度
        max_linear_velocity = 1000.  # 最大线速度
        armature = 0.  # 转动惯量附加项
        thickness = 0.01  # 碰撞几何厚度

    class domain_rand:
        randomize_friction = False  # 是否随机化地面摩擦；安全基线默认关闭
        friction_range = [0.5, 1.25]  # 地面摩擦系数随机范围
        randomize_base_mass = False  # 是否随机化机身质量
        added_mass_range = [-1., 1.]  # 机身附加质量范围 [kg]
        push_robots = False  # 是否周期性施加外推扰动；安全基线默认关闭
        push_interval_s = 15  # 外推扰动周期 [s]
        max_push_vel_xy = 1.  # 单次外推对应的最大平面速度增量

    class safety:
        # 安全基线开关：默认关闭高风险或尚未验证的 HIMLoco 化功能。
        enable_foot_clearance = False  # 是否启用足端抬脚 reward
        enable_rigid_body_state_rewards = False  # 是否刷新 rigid body state 并使用足端位姿/速度 reward
        enable_joint_power = False  # 是否启用关节功率惩罚
        enable_smoothness = False  # 是否启用二阶动作平滑惩罚
        enable_noise = False  # 是否启用观测噪声
        enable_domain_rand = False  # 是否启用域随机化
        enable_push = False  # 是否启用随机外推
        enable_height_scan = False  # 是否启用地形高度采样
        enable_command_curriculum = False  # 是否启用速度命令课程学习
        enable_history_obs = True  # 是否启用 45*6 历史观测；阶段 3 开始启用
        enable_him_actor_critic = False  # 是否启用 HIMActorCritic；Go2 阶段 6 开始启用
        enable_himppo = False  # 是否启用 HIM-PPO；Go2 阶段 6 开始启用

    class rewards:
        class scales:
            termination = -0.0  # 回合提前终止的奖励项
            tracking_lin_vel = 1.0  # 线速度跟踪奖励
            tracking_ang_vel = 0.5  # 角速度跟踪奖励
            lin_vel_z = -2.0  # 机身竖直速度惩罚
            ang_vel_xy = -0.05  # 横滚/俯仰角速度惩罚
            orientation = -0.2  # 姿态偏差惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚
            joint_power = -0.0  # 关节功率惩罚，安全基线默认关闭
            base_height = -0.0  # 机身高度偏差惩罚，安全基线默认关闭
            feet_air_time = 0.0  # 足端腾空时长奖励，Go2 walk 默认不额外鼓励跳跃步态
            foot_clearance = -0.0  # 足端抬脚约束，安全基线默认关闭
            action_rate = -0.01  # 动作变化率惩罚
            smoothness = -0.0  # 二阶动作平滑惩罚，安全基线默认关闭
            torques = -0.0002  # 力矩惩罚，安全基线保留基础约束
            dof_vel = -0.0  # 关节速度惩罚
            collision = -0.0  # 非预期碰撞惩罚，walk 阶段先不让硬惩罚压制步态探索
            feet_stumble = -0.0  # 足端绊倒惩罚
            stand_still = -0.0  # 静止状态惩罚/奖励项
            feet_contact_forces = -0.0  # 足端接触力过大的惩罚
            dof_pos_limits = -0.0  # 关节位置接近极限的惩罚
            dof_vel_limits = -0.0  # 关节速度接近极限的惩罚
            torque_limits = -0.0  # 关节力矩接近极限的惩罚

        only_positive_rewards = True  # 是否将负总奖励截断为 0
        tracking_sigma = 0.25  # 跟踪奖励的高斯宽度
        soft_dof_pos_limit = 1.0  # 关节软限制比例
        soft_dof_vel_limit = 1.0  # 关节速度软限制比例
        soft_torque_limit = 1.0  # 关节扭矩软限制比例
        base_height_target = 1.0  # 期望机身高度
        clearance_height_target = -0.20  # 机身坐标系下的足端目标高度 [m]，启用 foot_clearance 时使用
        max_contact_force = 100.  # 超过该接触力会被惩罚

    class normalization:
        class obs_scales:
            lin_vel = 2.0  # 线速度观测缩放
            ang_vel = 0.25  # 角速度观测缩放
            dof_pos = 1.0  # 关节位置观测缩放
            dof_vel = 0.05  # 关节速度观测缩放
            height_measurements = 5.0  # 地形高度观测缩放

        clip_observations = 100.  # 观测裁剪上限
        clip_actions = 100.  # 动作裁剪上限

    class noise:
        add_noise = False  # 是否在观测中加入噪声；安全基线默认关闭
        noise_level = 1.0  # 噪声总倍率

        class noise_scales:
            dof_pos = 0.01  # 关节位置噪声
            dof_vel = 1.5  # 关节速度噪声
            lin_vel = 0.1  # 线速度噪声
            ang_vel = 0.2  # 角速度噪声
            gravity = 0.05  # 重力方向噪声
            height_measurements = 0.1  # 高度采样噪声

    class viewer:
        ref_env = 0  # 相机跟随的环境编号
        pos = [10, 0, 6]  # 相机位置 [m]
        lookat = [11., 5, 3.]  # 相机注视点 [m]

    class sim:
        dt = 0.005  # 仿真主步长 [s]
        substeps = 1  # 每个主步长内的子步数
        gravity = [0., 0., -9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 上方向轴：0 表示 y 轴，1 表示 z 轴

        class physx:
            num_threads = 10  # PhysX 线程数
            solver_type = 1  # 求解器类型：0=pgs, 1=tgs
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 接触偏移 [m]
            rest_offset = 0.0  # 静止接触偏移 [m]
            bounce_threshold_velocity = 0.5  # 低于该速度时不计算反弹 [m/s]
            max_depenetration_velocity = 1.0  # 最大去穿透速度
            max_gpu_contact_pairs = 2**23  # GPU 接触对上限
            default_buffer_size_multiplier = 5  # 默认缓冲区倍数
            contact_collection = 2  # 接触收集模式：0=不收集，1=最后子步，2=所有子步


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1  # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 训练 runner 类名

    class policy:
        init_noise_std = 1.0  # 策略初始输出噪声标准差
        actor_hidden_dims = [512, 256, 128]  # actor 隐藏层宽度
        critic_hidden_dims = [512, 256, 128]  # critic 隐藏层宽度
        activation = 'elu'  # 激活函数：elu / relu / selu / crelu / lrelu / tanh / sigmoid
        estimator_hidden_dims = [128, 64, 16]  # HIMEstimator encoder 隐藏层，最后一维为 latent 维度
        estimator_target_hidden_dims = [128, 64]  # HIMEstimator target 分支隐藏层
        estimator_learning_rate = 1.e-3  # HIMEstimator 独立学习率；阶段 6 的 HIMPPO 使用
        estimator_max_grad_norm = 10.0  # HIMEstimator 梯度裁剪上限
        estimator_num_prototype = 32  # HIM 对比学习原型数量
        estimator_temperature = 3.0  # HIM 对比学习温度系数
        # 仅在循环网络策略下使用
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        value_loss_coef = 1.0  # value loss 权重
        use_clipped_value_loss = True  # 是否启用 clipped value loss
        clip_param = 0.2  # PPO clip 系数
        entropy_coef = 0.01  # 熵奖励系数
        num_learning_epochs = 5  # 每轮数据训练的 epoch 数
        num_mini_batches = 4  # 每轮拆分的小批次数量
        learning_rate = 1.e-3  # 学习率
        schedule = 'adaptive'  # 学习率调度：adaptive / fixed
        gamma = 0.99  # 折扣因子
        lam = 0.95  # GAE lambda
        desired_kl = 0.01  # 目标 KL 散度
        max_grad_norm = 1.0  # 梯度裁剪上限

    class runner:
        policy_class_name = 'ActorCritic'  # 策略网络类名；Go2 阶段 6 使用 HIMActorCritic
        algorithm_class_name = 'PPO'  # 算法类名；Go2 阶段 6 使用 HIMPPO
        runner_class_name = 'OnPolicyRunner'  # runner 类名；Go2 阶段 6 使用 HIMOnPolicyRunner
        num_steps_per_env = 24  # 每次迭代每个环境采样步数
        max_iterations = 1500  # 最大训练迭代次数

        save_interval = 50  # 每隔多少轮保存一次模型
        experiment_name = 'test'  # 实验名
        run_name = ''  # 本次运行名
        resume = False  # 是否从历史检查点恢复
        load_run = -1  # 加载第几个 run，-1 表示最后一个
        checkpoint = -1  # 加载第几个 checkpoint，-1 表示最后一个
        resume_path = None  # 由 load_run 和 checkpoint 组合生成的恢复路径
