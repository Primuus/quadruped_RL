# Primuus 训练配置与基础奖励

Primuus在`envs/primuus/primuus_config.py`中定义任务参数，继承通用环境与HIM-PPO配置。
`PrimuusEnv`负责Jenkins关节损耗、复位、站立姿态关节权重、水平约束和足端横向目标，指令采样使用通用`envs/base/commands.py`。
本任务启用19项基础奖励，使用固定范围的五类运动指令。

## 当前有效奖励

| 项目 | 权重 | 机制 |
|---|---:|---|
| tracking_lin_vel | 1.0 | exp(-XY速度误差平方和 / 0.25) |
| tracking_ang_vel | 0.5 | exp(-偏航角速度误差平方 / 0.25) |
| lin_vel_z | -2.0 | 机身竖直速度平方 |
| ang_vel_xy | -0.05 | 横滚/俯仰角速度平方和 |
| orientation | -0.2 | 重力投影XY分量平方和 |
| dof_acc | -2.5e-7 | 关节加速度平方和 |
| base_height | -0.4 | 超出统一0.28±0.03m高度容差后的归一化平方误差 |
| foot_clearance | -0.01 | 摆动足机身系高度误差平方×水平相对速度，排除支撑足 |
| foot_slip | -0.10 | 触地足对地水平速度平方和 |
| turn_step | 0.1 | 合格转向换步，按每足时间预算及XY/yaw跟踪计分 |
| action_rate | -0.01 | 动作一阶差分平方和 |
| smoothness | -0.01 | 动作二阶差分平方和 |
| torques | -0.0002 | 电机力矩平方和 |
| collision | -1.0 | 持续非足端接触数量 |
| stand_still | -0.1 | 零参考指令时，关节相对本机默认角的加权绝对偏差之和；四髋3倍，其余1倍 |
| stand_contacts | -0.4 | 零参考指令时，缺少支撑的足端比例 |
| stand_orientation | -5.0 | 零参考指令时，额外计算重力投影XY分量平方和 |
| stand_foot_lateral | +0.5 | 零参考指令时，exp(-四足横向目标误差平方和 / 0.10²) × 直立缩放 |
| dof_pos_limits | -10.0 | 实际关节角超软限位的幅度之和 |

`tracking_sigma=0.25`是误差平方的指数分母，不是标准差。
前进指令1m/s时，实际速度为0、0.5、1m/s的线速度奖励分别约为0.0183、0.3679、1。
转向和转向换步使用同一速度跟踪公式。

抬脚使用通用 `LeggedRobot._reward_foot_clearance`：足端目标机身系高度-0.20m，
高度误差尺度0.05m。计算相对足速时扣除机身平移与 `omega × r` 再转到机身坐标系；
仅未触地且有水平相对运动的足端产生代价。
这仍是高度引导惩罚，不强制固定四足相位，也不额外奖励长时间腾空。

零参考指令掩码与通用基线一致：XY模长和yaw绝对值均不超过0.1。
站姿偏差直接使用本机默认角。`rewards.stand_still_joint_weights`按关节名称指定相对权重：
四个`hip_joint`均为3.0，未列出的大腿和小腿关节为1.0，总系数仍为`scales.stand_still=-0.1`。
因此站立时髋关节的有效系数为-0.3，其余关节为-0.1；行走、侧移和转向不启用此项。
该项是姿态软约束，不锁定关节，也不要求四条腿使用相同动作。
`stand_orientation`复用通用`_reward_orientation()`的重力投影XY平方和，仅在站立指令下
额外乘以-5.0；与全局`orientation=-0.2`相加后，站立水平约束的总系数为-5.2。
它同时约束roll和pitch，不约束yaw；运动指令仍只有全局-0.2系数。

`stand_foot_lateral`使用同一站立指令掩码，按足端名称匹配机身系横向目标：
`stand_foot_lateral_targets`中FL/RL为+0.15m，FR/RR为-0.15m，总站宽为0.30m。
该值接近本机标称站姿约±0.1495m的足端横向位置。令各足横向误差为e_i，原始奖励为：

`exp(-sum(e_i²) / stand_foot_lateral_sigma²) * clamp(-gravity_z, 0, 0.7) / 0.7`

`stand_foot_lateral_sigma=0.10`m；该尺度用于四足误差平方和，不是每只脚的硬容差。
指数项使用正权重+0.5，越贴近目标越大；直立缩放避免翻倒时得到足端对齐奖励。
该项需要刚体状态，采用完整机身旋转变换，计算全部四足，不区分支撑与摆动。
只约束Y，不约束X/Z，不锁死足端，也不要求四脚受力相等。
行走、侧移、转向指令超出站立掩码时，两项站立奖励均为零。
改变机械模型时应重新校准足端目标。

### 奖励来源与Primuus适配

- 水平公式来自[legged_gym的LeggedRobot._reward_orientation](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)，
  [AnymalCFlatCfg.rewards.scales](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py)
  以-5.0启用，原实现无站立门控。Primuus复用相同公式，增加站立命令门控，作为额外奖励项；全局水平项保持-0.2。
- 横向公式来自[robot_lab的feet_distance_y_exp](https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L452)。
  该版本[通用配置](https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L616)
  权重为0、std为0.5m；[Tita配置](https://github.com/fan-ziqi/robot_lab/blob/26acb225aad129e222da7a34ba11f39e29b73e39/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/ddtrobot_tita/rough_env_cfg.py#L243)
  以-1.0和0.6m站宽启用。该负权重与贴近目标的优化方向相反，不用于Primuus。
  Primuus保留指数公式和直立缩放，将索引奇偶映射改为足端名称映射，增加站立门控；
  使用本机±0.15m目标、0.10m误差尺度和+0.5权重。这些参数是Primuus适配值，不代表上游四足任务的已验证配置。

修改奖励需要重新训练或续训并重新导出策略，已有部署模型不会随配置自动改变。
支撑足使用5N阈值并保留一帧接触滤波。
缺足奖励按四足缺失比例计分。

表外奖励保持零权重，包括通用base中的`stand_height`和`feet_air_time`。
HIM估计器使用原始观测学习速度和隐变量，训练损失包含速度估计与表征学习。
`only_positive_rewards=False`保留负总奖励，终止奖励权重为0。

## Primuus 专有配置

任务参数由Primuus实例配置和通用默认配置共同决定：

- 默认角：四腿hip=0、thigh=-0.75、calf=-1.5rad。
- PD=60/1.5，action_scale=0.25；物理2ms、策略20ms。
- actor观测270、critic观测238、动作12；HIM-PPO优化参数保持原值。
- 所有运动状态的高度目标统一0.28m，容差±0.03m。
- 课程关闭；纯前进速度范围0.12–2m/s，vy和yaw固定为0。其他平移状态的vx、vy分量范围均为[-1,1]m/s，XY合速度上限1m/s，偏航范围[-1,1]rad/s。
- 纯前进20%、原地转向20%、边走边转15%、平移25%、站立20%。平移模式中60%等概率采样纯后退、纯左移和纯右移，其余40%保留任意XY方向；边走边转保留任意XY方向。
- 每类指令统一保持5秒，直接切换，不经过额外加减速或制动参考。
- 纯前进按[0.12,1.0]、[1.0,1.5]、[1.5,2.0]m/s三段采样，概率分别为60%、15%、25%，各段内部均匀采样，不叠加低速缩放。
- 其他运动命令仍有30%采用0.35倍低速采样范围；非零平移最低模长0.12m/s，原地转向最低角速度绝对值0.12rad/s。
- 4096环境、每环境每轮24个策略步；HIM-PPO为5个epoch、4个mini-batch，学习率0.001并自适应调整，熵系数0.01，seed=1。
- actor/critic隐藏层512/256/128、ELU；观测噪声开启，地面摩擦随机范围[0.4,1.25]，机身质量随机化开启、附加质量范围[-0.5,0.5]kg，Push关闭。创建各环境时分别均匀采样机身质量偏移，回合复位不重新采样；腿部质量保持标称值。设置刚体属性时沿用`recomputeInertia=True`。
- 初始及复位机身高度0.35m、关节角复位倍率[0.9,1.1]，回合上限20秒。
- Jenkins干摩擦屈服力矩0.05Nm、粘性阻尼0.02Nm·s/rad、预滑移刚度5Nm/rad，每个物理步计算被动损耗。

这些是采样器设定，不是实际训练时间占比；复位、保持时长和序列会改变实际占比。

## 采样与日志

`commands.curriculum=False`，`max_curriculum`不参与范围调整。纯前进范围由`ranges.lin_vel_forward`设置，上限2m/s；其他平移状态仍受`max_translation_speed=1.0`限制。
指令每五秒重采样并直接切换。
`commands.mode_names`与`mode_probabilities`顺序一一对应；新增纯前进使用`forward`名称。
`translation_axis_probability`控制平移模式内轴向命令比例；三个轴向分别为纯后退、纯左移和纯右移，等概率采样。
`forward_speed_bins`与`forward_speed_probabilities`定义纯前进速度分段及概率；分段必须连续覆盖`ranges.lin_vel_forward`，概率之和为1。修改前进范围时同步修改分段。
五类`Command/turn`、`moving_turn`、`translate`、`stand`和`forward`日志统计速度误差、
滑移、换步、机身高度、关节姿态偏差和支撑足数，以及`Episode/*`、PPO损失与性能日志。

训练分析同时看回合长度、按运动状态的速度误差及确定性部署结果。
`Train/mean_reward`是每回合累计值；`Episode/rew_*`按最大回合时长20秒归一，
回合缩短会改变两者，不能只凭回报上升或惩罚下降认定效果改善。

## Primuus 增量Push

默认`domain_rand.push_robots=False`；设为`True`可启用瞬时水平速度扰动。`max_push_vel_xy=0.25`表示
世界坐标系X/Y各轴独立均匀采样[-0.25,+0.25]m/s的速度增量并叠加到当前速度，
合成增量最大约0.354m/s。只改变机身水平线速度，不改变竖直速度、角速度或速度命令；
不按命令速度范围裁剪实际速度，电机仍使用原有限矩。

`push_interval_range_s=[12.0,18.0]`控制各环境独立的随机间隔，按策略周期离散为
600至900步（包含端点）。初始化、回合复位及每次Push后均重新采样倒计时。
刚复位的环境不触发Push，也不消耗倒计时；计时独立于回合长度计数，避免与5秒命令切换固定重合。
Primuus使用该范围参数，不使用通用基类的固定`push_interval_s`计时。
设置`push_robots=False`可禁用扰动；切换配置需重新创建训练环境。

实现位于`PrimuusEnv._push_robots()`，通用控制循环通过同名方法分派。
速度叠加方式参考[IsaacLab的push_by_setting_velocity](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/events.py)，
随机间隔机制参考其[速度任务配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py)。
上游通用配置使用10–15秒、每轴±0.5m/s；Primuus适配为12–18秒、每轴±0.25m/s，
并明确跳过复位步骤。此方式是速度冲击近似，不代表指定牛顿数、作用点或持续时间的外力。
