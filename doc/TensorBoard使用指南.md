# TensorBoard 使用指南

[文档导航](README.md) | [训练调参教程](训练调参教程.md)

核对日期：2026-10-15。指标名称和统计口径来自当前 `HIMOnPolicyRunner` 与 `LeggedRobot`，不套用其他框架的同名曲线解释。TensorBoard 用于定位训练现象，不能代替 Isaac Gym play、MuJoCo 和实机验收。

## 1. 日志和启动

训练侧默认日志位置：

```text
~/quadruped_train/logs/go2/
  Oct15_20-00-00_rough/
    events.out.tfevents...
    model_50.pt
    model_100.pt
  exported/policies/
    policy_1.pt
    policy_1.onnx
```

run 名是示例。事件文件记录标量，`model_*.pt` 是训练 checkpoint；导出的 `policy_1.*` 不能用于恢复训练优化器。

在服务器启动 TensorBoard：

```bash
ssh -p 2222 hurricane@10.109.69.166
cd ~/quadruped_train
conda activate isaacgym
tensorboard --logdir logs/go2 --host 127.0.0.1 --port 6006
```

另开本地终端建立隧道：

```bash
ssh -p 2222 -N -L 6006:127.0.0.1:6006 hurricane@10.109.69.166
```

浏览器访问 `http://127.0.0.1:6006`。隧道终端保持打开；本地6006被占用时，可改为 `-L 16006:127.0.0.1:6006`，浏览器对应改用16006。

只在持有事件文件的一侧安装 TensorBoard即可，浏览器所在电脑不需要安装训练依赖。若想离线查看，可把某个run的事件文件复制到本地，再在本地启动TensorBoard。绑定 `0.0.0.0` 会开放网络访问，默认不需要这样做。

## 2. 页面与横轴

1. 进入 Scalars，先只选一个完整run。
2. 横轴 Step 对应已完成的训练迭代，不是单个环境的物理步或episode。
3. 先看原始曲线，再适度平滑；平滑过强可能掩盖突发崩溃。
4. 比较多个run时固定横轴、平滑系数与评估条件，同时记录代码和奖励版本。
5. 新的resume进程有新run目录，checkpoint编号会继续累计。

当前4096环境、每环境24个控制步，每迭代采样98304个transition。100轮能跑通只代表短时稳定，不等于步态已经收敛。

日志需要有episode结束后才出现 `Train/*` 与 `Episode/*` 的部分指标。TensorBoard刷新延迟或训练正在采样，不等于程序卡死。

## 3. 两种奖励曲线的统计口径

### Train/mean_reward

最近最多100个已结束episode的总回报均值。它是累计回报，不是“某一轮PPO的即时得分”，也不保证每个episode都完整运行20秒。

### Episode/rew_<name>

环境初始化会将非零奖励权重乘以控制周期：

```text
每控制步贡献 = 配置权重 × control_dt × 原始奖励函数
Episode/rew_name = 已结束回合该项累计贡献 / 配置的episode_length_s
```

runner再对收集到的episode统计取均值。分母使用配置的最大回合时长，不是该回合实际存活时间，所以短回合会拉低日志值。

当前Go2控制周期 `0.005*4=0.02 s`，配置回合长20 s。完整回合中，理论上限为1的线速度跟踪函数，对应日志接近权重1.0；角速度跟踪对应权重0.5。但如果回合很短，不能仅用“日志值/权重”解释跟踪准确率。

因此：

- `Train/mean_reward` 与所有 `Episode/rew_*` 不能直接逐点相加对账。
- 负奖励项的数值表示已加权代价，不是原始米、牛米或接触次数。
- 改奖励公式、权重、回合长度或采样分布后，总reward不能直接横向比较。
- 本轮源码默认 `only_positive_rewards=False`；旧日志不一定使用相同定义。

## 4. 先看这六组指标

| 顺序 | 指标 | 要回答的问题 |
| --- | --- | --- |
| 1 | `Train/mean_episode_length` | 是否频繁提前终止？ |
| 2 | `Train/mean_reward` | 当前目标下是否在学习，是否突然退化？ |
| 3 | `Command/<mode>/fraction` 和误差 | 四类命令是否都有覆盖，哪类不会？ |
| 4 | 高度、支撑足和非足端接触 | 是否用趴行、膝部支撑或扭身取巧？ |
| 5 | 滑移、换步、抬脚与动作代价 | 是否真正迈步，代价是否合理？ |
| 6 | loss、学习率、探索标准差和性能 | 更新是否稳定，计算是否异常？ |

总reward高、episode长，仅说明能存活并获得当前目标的回报；不证明机器人使用足端正常走路。

## 5. 存活与跟踪

### Train/mean_episode_length

单位是控制步。当前20 s / 0.02 s约1000步，终止条件的比较可能带来一控制步差异。

接近1000说明大多未提前终止，不等于站姿或步态正确。低趴但未触发机身/姿态终止的策略仍可能活满一局。

### Episode/rew_tracking_lin_vel

原始函数为：

```text
exp(-||command_xy - base_lin_vel_xy||² / tracking_sigma)
```

比较的是机身坐标系XY速度，当前 `tracking_sigma=0.25`、权重1.0。不检查支撑部位，滑动或非足端支撑也可能取得高分。

### Episode/rew_tracking_ang_vel

原始函数为：

```text
exp(-(command_yaw - base_ang_vel_z)² / tracking_sigma)
```

当前权重0.5。它只衡量偏航速度跟踪，不保证转向时抬脚、不滑移或机身姿态自然。

两项都高仍需看四类命令的误差；平均值会混入更容易的站立或低速样本。

## 6. Command 分组指标

当前模式名固定为：

```text
Command/turn/*          原地转向
Command/moving_turn/*   边走边转
Command/translate/*     平移
Command/stand/*         零命令站立
```

Go2配置比例为25%、25%、30%、20%，但曲线是本迭代实际采样占比，不保证每次都恰好等于配置概率。

| 后缀 | 统计含义 |
| --- | --- |
| `fraction` | 本轮采样中该模式占比 |
| `yaw_abs_error` | 偏航命令与实际角速度的绝对误差，rad/s |
| `linear_error` | XY命令与实际线速度误差的向量模长，m/s |
| `valid_steps_per_second` | 按有效腾空、抬升与落地条件计数的足端落地频率，四足合计，不是reward |
| `foot_slip_speed` | 每环境接触足端水平速度模长之和再平均，m/s；不是单只足端平均速度 |
| `feet_in_contact` | 向上接触力超过1 N的足端数，未使用站立支撑滤波 |
| `tilt` | 重力投影XY分量模长，不是以度表示的倾角 |
| `base_height` | 当前平地机身根节点世界z，m |
| `joint_pose_error` | 所有关节相对默认角的绝对误差之和，rad |
| `supporting_feet` | 5 N阈值且允许一帧漏检的支撑足数 |

某模式本轮没有样本时不会写该模式曲线，不能把缺失值当成零误差。

转向故障的典型组合：

```text
turn/yaw_abs_error 低
turn/foot_slip_speed 高
turn/valid_steps_per_second 低
```

这支持“通过滑动扭身而非正常换步”的判断，但仍需play画面确认。不能只提高 `tracking_ang_vel` 权重。

## 7. 高度、足端与非足端支撑

### Episode/rew_base_height

当前目标0.28 m、容差0.03 m，0.25-0.31 m区间内代价为零：

```text
max(abs(base_z - target) - tolerance, 0)² / sigma²
```

配合各模式 `base_height` 判断是站立矮、行走矮还是仅转向时下沉。初始0.42 m不是行走高度目标。

### Episode/contact_<body>

某个非足端body超过5 N力阈值的回合时间占比，近似范围0到1。这不是加权reward，也不是足端接触率。

逐条比较四腿大腿/小腿；长期同侧偏高值得检查关节映射、碰撞形状和奖励取巧。配置Primuus固定碰撞载体时，需要看其 `calf_collision_link`，不能只看不承载形状的原 `calf_link`。

### Episode/rew_collision

当前权重-1.0，只惩罚持续约0.06 s的非足端接触。短碰撞可以出现在 `contact_<body>` 中而尚未产生明显collision代价。惩罚不是硬碰撞约束，也不会自动修复模型穿插。

### Episode/rew_stand_still 与 rew_stand_contacts

只作用于XY命令模长不超过0.1 m/s、偏航命令不超过0.1 rad/s的站立样本。

- `stand_still`：偏离默认关节角的绝对偏差之和。
- `stand_contacts`：缺少支撑足的比例，四足正常支撑时为零。

原地转向不应被四足同时支撑的站立目标压制。全局 `stand_contacts` 曲线还受站立采样占比影响，应结合 `Command/stand/supporting_feet`。

### Episode/rew_foot_clearance

当前仅约束非支撑足。使用机身系足端高度、扣除机身平移/旋转后的相对速度，高度误差除以0.05 m再平方。目标-0.20 m是相对机身的坐标，不是离地抬20 cm。

接近零可能是高度合适，也可能是足端没有摆动或一直支撑，不能据此断言抬腿成功。

### Episode/rew_foot_slip 与 rew_turn_step

`foot_slip` 是接触足端水平速度平方和的负权重贡献。数值更接近零可能是滑移变小，也可能是机器人不动。

`turn_step` 按每足可积累的时间预算奖励有效落地，还要求XY/yaw跟踪、同向转动和没有持续非足端支撑。高频跺脚不会无限增加单位时间回报；`valid_steps_per_second` 与它不是同一个统计量。

## 8. 动作代价

| 曲线 | 原始含义 | 不能直接推出 |
| --- | --- | --- |
| `Episode/rew_torques` | 关节力矩平方和 | 真实电功率、能耗或峰值力矩 |
| `Episode/rew_action_rate` | 相邻动作差平方和 | 机器人一定在抖；正常摆腿也会变化 |
| `Episode/rew_smoothness` | 二阶动作差平方和 | 越接近零步态越好；不动作也可以很小 |
| `Episode/rew_dof_acc` | 关节速度差除以控制周期后的平方和 | 某个电机的单独加速度 |
| `Episode/rew_dof_pos_limits` | 超出软位置范围的偏差 | 超出机械硬限位的次数 |

这些代价必须与跟踪、实际动作和接触一起解释。只在MuJoCo抖而Isaac Gym正常时，先查模型、关节顺序、默认角、裁剪、缩放、控制周期与逐物理步PD，不先盲目增加smoothness。

## 9. Loss 与探索

| 曲线 | 当前含义和阅读方法 |
| --- | --- |
| `Loss/value_function` | critic返回值预测损失；受reward量级和数据分布影响，不要求单调归零 |
| `Loss/surrogate` | clipped PPO目标的损失；符号和大小不能直接解释为步态好坏 |
| `Loss/estimation` | 预测速度与下一时刻critic缩放线速度的MSE；不是未经缩放的m/s误差 |
| `Loss/swap` | estimator的原型对比损失；归约含原型维度，不能套用普通分类交叉熵的量级 |
| `Loss/learning_rate` | KL自适应后的学习率，当前代码范围1e-5到1e-2 |
| `Policy/mean_noise_std` | 训练动作正态分布的平均探索标准差，不是传感器噪声或PD增益 |

速度估计loss降低通常是有用信号，但是否接近零取决于状态噪声、目标缩放和运动复杂度。探索标准差也不要求一直下降；例如稳定在0.5不能单独判为异常。play和导出的部署模型使用确定性动作，不把训练探索噪声原样带入部署。

遇到NaN、突然巨大变化或reward崩溃时，检查观测/动作有限性、更新幅度、命令分布和设备状态，不凭某一条loss直接断定根因。

## 10. 性能指标

```text
Perf/total_fps
Perf/collection time
Perf/learning_time
```

`total_fps` 是并行采样控制步数除以采样加优化时间，不是单台机器人实时频率。采样慢可能来自物理接触、刚体状态刷新或CPU/GPU资源争用；学习慢可能与网络、mini-batch和设备有关。

这些曲线不能诊断GPU硬件健康。若出现CUDA launch failure且 `nvidia-smi` 也失败，应停止继续启动任务并检查驱动/硬件状态。

## 11. 从曲线到参数决策

| 现象 | 先核实 | 再做单组对照实验 |
| --- | --- | --- |
| reward升、episode短 | 终止条件与命令难度 | 缩小过难命令或调整姿态代价 |
| 活满一局但趴着走 | 高度、非足端接触、支撑足 | 高度/站立/碰撞奖励组合 |
| 转向滑步、不换脚 | turn分组滑移、有效落地和非足端接触 | 转向采样比例、滑移和换步门槛 |
| 不抬腿 | 碰撞模型、限位、动作映射、摆动状态 | clearance与动作代价的平衡 |
| MuJoCo单独变差 | 训练与部署契约和模型接触 | 先修对齐，再讨论重新训练 |
| loss突然异常 | 数据与设备状态 | 最后才考虑学习率和随机化强度 |

本表是排查顺序，不是“看到曲线就自动改参数”的规则。详细数值及新增reward/随机化步骤见[训练调参教程](训练调参教程.md)。

比较多个run时，尽量保持源码、训练种子、训练预算、命令模式和评估动作一致。若reward定义已不同，使用共同的速度误差、接触率和评估画面，而不是比较谁的总reward更大。

## 12. 导出、复制与回放

在训练服务器指定实际run与checkpoint：

```bash
cd ~/quadruped_train
python quadruped_rl/scripts/play_go2.py \
  --headless \
  --load_run Oct15_20-00-00_rough \
  --checkpoint 1000
```

run名为示例，需要替换为真实目录。默认选择最新有效run和最大编号checkpoint，不会选择reward最高的模型。

每次成功双导出得到：

```text
logs/go2/exported/policies/policy_1.pt
logs/go2/exported/policies/policy_1.onnx
```

重复play会覆盖固定导出文件，不覆盖checkpoint；ONNX失败时不应混用新PT和旧ONNX。导出成功后play继续执行约10个配置回合长度的仿真，`--headless` 只关闭窗口，终端暂时无输出不一定是卡住。

从本地部署仓库根目录复制：

```bash
scp -P 2222 'hurricane@10.109.69.166:~/quadruped_train/logs/go2/exported/policies/policy_1.*' \
  src/controller/rl/models/
```

这会覆盖同名文件。需要留存时使用独立版本目录，并相应修改机器人YAML；模型不会因源码更新自动变成新训练结果。

先在相同输入上比较双模型，再用平地MuJoCo观察：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx

python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --xml-path src/controller/rl/resources/robots/go2/scene_flat.xml
```

Go2默认配置实际选择ROBOCON场地，不是平地；比较训练与部署时应显式选择上述平地场景。Primuus不能使用Go2权重。

## 13. 记录模板

```text
源码commit / 机器人 / URDF或XML版本：
完整run目录 / checkpoint / 是否resume：
追加迭代数 / 控制周期：
修改的单个参数组：
各模式fraction与速度误差：
各模式高度 / supporting_feet / 非足端接触率：
滑移 / 换步 / 抬脚与动作代价：
loss / 学习率 / 探索标准差：
Isaac Gym play现象：
MuJoCo现象与双模型一致性：
结论与下一项实验：
```

训练曲线只能说明当前日志对应的实验。当前源码的参数表、历史checkpoint的参数和硬件的真实行为必须分别记录，不相互代替。
