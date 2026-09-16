# TensorBoard 使用指南

这份文档用于查看和分析 `quadruped_train` 的训练日志。重点说明 TensorBoard 怎么启动、每条曲线代表什么、如何判断训练是否有效，以及如何根据曲线决定下一步调参方向。

## 日志位置

训练日志在服务器的 `quadruped_train` 工程中：

```text
~/quadruped_train/logs/go2/
```

每次训练会生成一个 run 目录，格式类似：

```text
logs/go2/Jul09_18-30-10_rough/
```

目录里一般包含：

```text
events.out.tfevents...   # TensorBoard 日志
model_*.pt               # checkpoint
```

## 启动 TensorBoard

在服务器上执行：

```bash
cd ~/quadruped_train
conda activate isaacgym
tensorboard --logdir logs/go2 --host 0.0.0.0 --port 6006
```

然后在本地浏览器打开：

```text
http://192.168.0.214:6006
```

如果浏览器不能直接访问端口，用 SSH 转发。在本地电脑终端执行：

```bash
ssh -L 6006:localhost:6006 hurricane@192.168.0.214
```

然后本地浏览器打开：

```text
http://localhost:6006
```

如果服务器没有 TensorBoard：

```bash
conda activate isaacgym
pip install tensorboard
```

## TensorBoard 页面怎么设置

进入网页后：

1. 打开 `Scalars` 页面。
2. 左侧只勾选当前要看的 run，例如 `Jul09_18-30-10_rough`。
3. 如果曲线波动太大，把 `Smoothing` 调到 `0.6 ~ 0.8`。
4. 对比不同参数时，同时勾选多个 run。
5. 不要只看最后一个点，要看趋势。

## 最先看哪些曲线

优先看：

```text
Train/mean_reward
Train/mean_episode_length
Episode/rew_tracking_lin_vel
Episode/rew_tracking_ang_vel
Episode/rew_torques
Episode/rew_action_rate
Episode/rew_smoothness
Episode/rew_foot_clearance
Episode/rew_dof_pos_limits
Loss/value_function
Loss/surrogate
Loss/estimation
Policy/mean_noise_std
```

判断顺序：

```text
1. mean_episode_length 是否接近最大 episode 长度
2. mean_reward 是否整体上升
3. tracking_lin_vel / tracking_ang_vel 是否上升
4. torques / action_rate / smoothness 是否代价过大
5. dof_pos_limits 是否接近 0
6. loss 是否稳定，没有爆炸或 NaN
7. mean_noise_std 是否逐渐下降
```

## Train 指标

### Train/mean_reward

平均 episode 总奖励，代表整体训练表现。

一般希望：

```text
长期上升
后期趋于稳定
没有突然大幅崩掉
```

注意：

- 它不是越高就一定越好。
- 有时 reward 上升，但动作很抖或部署效果差。
- 必须结合 tracking、torques、smoothness、episode length 一起看。

常见现象：

```text
持续上升：策略在学习
长期不动：可能 reward 设计不合适或训练太难
突然大跌：可能参数太激进、训练崩了、随机化太强
```

### Train/mean_episode_length

平均 episode 持续 step 数。当前配置：

```text
episode_length_s = 20
sim.dt = 0.005
control.decimation = 4
control_dt = 0.02
最大 episode length 约为 20 / 0.02 = 1000
```

判断：

```text
接近 1000：基本不摔，能活完整局
逐渐上升：稳定性变好
很低：频繁摔倒或 reset
```

如果 `mean_reward` 上升，但 `mean_episode_length` 很低，说明策略可能在短时间内刷到 reward，但仍然不稳定。

## 速度跟踪指标

### Episode/rew_tracking_lin_vel

线速度跟踪奖励。看机器人是否按命令前进、后退、侧移。

计算逻辑近似：

```python
lin_vel_error = sum((command_xy - base_lin_vel_xy) ** 2)
reward = exp(-lin_vel_error / tracking_sigma)
```

判断：

```text
接近 1.0：线速度跟踪很好
长期很低：机器人没有按命令走
上升后又下降：可能训练变难或动作代价压制了步态
```

当前 Go2 权重：

```python
tracking_lin_vel = 1.0
```

所以这项最高大约接近 1.0。

### Episode/rew_tracking_ang_vel

yaw 角速度跟踪奖励。看机器人是否会按命令转向。

计算逻辑近似：

```python
ang_vel_error = (command_yaw - base_ang_vel_yaw) ** 2
reward = exp(-ang_vel_error / tracking_sigma)
```

当前 Go2 权重：

```python
tracking_ang_vel = 0.5
```

所以这项最高大约接近 0.5。

如果 `tracking_lin_vel` 很高，但 `tracking_ang_vel` 很低，说明会走直线但转向能力差。

## 动作代价指标

这些指标用来判断机器人是不是用很粗暴的方式走路。

### Episode/rew_torques

力矩惩罚。

```python
torques = -0.0002
```

越负说明用力越大。

判断：

```text
轻微变负：正常，走路需要用力
持续很负：动作可能很暴力，部署可能抖或不稳
突然变得很负：某个 reward 可能逼机器人用大力矩
```

如果脚抬不起来，可以考虑减弱力矩惩罚：

```python
torques = -0.0001
```

如果动作太暴力，可以增强：

```python
torques = -0.0003
torques = -0.0004
```

### Episode/rew_action_rate

一阶动作变化惩罚。

```python
action_rate = -0.01
```

它反映相邻 action 的变化幅度。

判断：

```text
越负：动作变化越快
太负：可能抖动
太弱：动作可能不够平滑
```

如果动作抖：

```python
action_rate = -0.02
```

如果脚抬不起来或动作迟钝：

```python
action_rate = -0.005
```

### Episode/rew_smoothness

二阶动作变化惩罚，更关注高频抖动。

```python
smoothness = -0.01
```

判断：

```text
越负：动作二阶变化越大，可能高频抖
接近 0：动作更平滑
```

如果 MuJoCo 里抖动明显：

```python
smoothness = -0.02
```

如果机器人不愿迈腿：

```python
smoothness = -0.005
```

### Episode/rew_dof_pos_limits

关节位置限制惩罚。

当前：

```python
dof_pos_limits = -10.0
soft_dof_pos_limit = 0.9
```

判断：

```text
接近 0：基本没有接近关节极限
明显变负：关节经常打到极限附近
```

如果这项很负，说明策略可能通过极端关节角完成动作，需要重点检查 play 画面。

## 抬腿相关指标

### Episode/rew_foot_clearance

足端抬脚约束。当前已经打开：

```python
foot_clearance = -0.01
clearance_height_target = -0.20
```

`foot_clearance` 权重非零时，训练环境会自动获取并刷新所需的 rigid-body state。

这项是惩罚项，通常是负值。不要简单理解为越高越好，要结合实际画面看。

判断：

```text
数值很小：这个 reward 对总 reward 影响不大
越来越负：可能脚高度误差变大，或摆腿速度增大
突然很负：可能 foot_clearance 过强或动作过大
```

如果 play 发现仍然拖脚：

```python
foot_clearance = -0.02
```

如果脚抬得太夸张或动作变大：

```python
foot_clearance = -0.005
```

## Loss 指标

### Loss/value_function

critic/value 网络损失。critic 用来估计当前状态未来能获得多少 reward。

判断：

```text
稳定较低：正常
剧烈爆炸：训练不稳定
长期很高：critic 学不准，PPO 更新质量差
```

这项不需要单调下降到 0，只要不爆炸即可。

### Loss/surrogate

PPO actor 更新损失。

这不是普通意义上的“越小越好”。重点看是否稳定。

判断：

```text
小幅波动：正常
突然巨大波动：策略更新过猛
长期异常大：学习率、reward、KL 可能有问题
```

### Loss/estimation

HIM estimator 估计损失。当前 HIM-PPO 会从历史观测估计隐含信息，再给 actor 使用。

判断：

```text
逐渐下降：estimator 在学
很高不降：历史观测不足或 estimator 学不动
突然爆炸：数据或训练不稳定
```

这项是 HIMLoco 化后很重要的指标。正常情况下应该明显下降。

### Loss/swap

HIM 对比/原型相关损失。主要看是否稳定，不需要像 `estimation` 一样明显降到很低。

判断：

```text
小幅波动：正常
突然爆炸：HIM estimator 相关训练不稳定
```

### Loss/learning_rate

当前自适应学习率。配置：

```python
schedule = 'adaptive'
desired_kl = 0.01
learning_rate = 1e-3
```

如果 KL 偏大，学习率会下降；如果 KL 偏小，学习率可能上升。

判断：

```text
缓慢调整：正常
快速降到很低：策略更新可能过猛，adaptive schedule 在压制学习率
长期很低且 reward 不涨：可能学习停滞
```

## Policy 指标

### Policy/mean_noise_std

策略动作分布的平均标准差，也就是探索噪声大小。

判断：

```text
从 1.0 逐渐下降：策略逐渐确定，正常
一直很高：策略还没学稳
过早降得很低：可能探索不足
突然异常：训练可能不稳定
```

例如某次训练：

```text
1.00 -> 0.245
```

说明策略已经从随机探索逐渐变得稳定。

## 一次训练如何分析

以一个 run 为例：

```text
logs/go2/Jul09_18-30-10_rough
```

如果看到：

```text
Train/mean_reward: 0 -> 23
Train/mean_episode_length: 低 -> 1000
rew_tracking_lin_vel: 接近 1.0
rew_tracking_ang_vel: 接近 0.5
Loss/estimation: 明显下降
Policy/mean_noise_std: 从 1.0 降到 0.2~0.3
```

可以判断：

```text
训练是健康的
机器人大概率学到了速度跟踪
不是完全不会走的策略
```

再看：

```text
rew_torques
rew_action_rate
rew_smoothness
rew_dof_pos_limits
rew_foot_clearance
```

判断走得是否粗暴、是否抖动、是否打关节极限、是否拖脚。

## 根据曲线改参数

### 1. reward 上升，episode length 很低

现象：

```text
Train/mean_reward 上升
Train/mean_episode_length 很低
```

说明策略可能短时间拿到 reward，但仍然频繁摔倒。

优先考虑：

```python
orientation = -0.3
ang_vel_xy = -0.1
lin_vel_z = -3.0
```

也可以缩小命令范围：

```python
lin_vel_x = [-0.5, 0.8]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-0.5, 0.5]
```

### 2. tracking 很高，但 MuJoCo 里动作抖

现象：

```text
rew_tracking_lin_vel 高
rew_tracking_ang_vel 高
rew_action_rate 很负
rew_smoothness 很负
```

优先考虑：

```python
action_rate = -0.02
smoothness = -0.02
```

如果还抖，降低学习率：

```python
learning_rate = 5e-4
```

### 3. 机器人不抬腿或拖脚

现象：

```text
play 里拖脚
rew_foot_clearance 影响很小
tracking 还不错
```

优先考虑：

```python
foot_clearance = -0.02
```

如果动作被压得太保守：

```python
torques = -0.0001
action_rate = -0.005
smoothness = -0.005
```

### 4. 力矩太大，动作暴力

现象：

```text
rew_torques 很负
机器人动作大
MuJoCo 里冲击明显
```

优先考虑：

```python
torques = -0.0003
torques = -0.0004
```

如果同时脚抬太高：

```python
foot_clearance = -0.005
```

### 5. 关节接近极限

现象：

```text
rew_dof_pos_limits 明显变负
play 里关节姿态很极端
```

优先考虑：

```python
soft_dof_pos_limit = 0.85
dof_pos_limits = -10.0  # 保持或增强
```

如果已经很强还不行，检查 default joint angles 和 action_scale。

### 6. estimator 学不好

现象：

```text
Loss/estimation 长期很高
Train/mean_reward 不涨
```

优先考虑：

```text
确认 obs 维度和 history_length 没改错
确认 actor obs 顺序和部署侧一致
降低学习率到 5e-4
先关闭过强随机化
```

## 多个 run 怎么比较

每次改参数后建议用不同 `run_name`：

```bash
python quadruped_rl/scripts/train_go2.py \
  --headless \
  --max_iterations 1000 \
  --run_name foot_clearance_002
```

TensorBoard 里同时勾选多个 run，看：

```text
谁的 mean_episode_length 更快接近 1000
谁的 mean_reward 更稳定
谁的 tracking 更高
谁的 torques/action_rate/smoothness 代价更低
谁的 dof_pos_limits 更接近 0
```

不要只比较最终 reward。一个 run 最终 reward 高，但动作代价很大，不一定更好。

## 从 TensorBoard 到 play

TensorBoard 只能判断趋势，最终必须看 play。

推荐节奏：

```text
100 iteration：检查是否能跑、是否 NaN、是否频繁摔
500 iteration：看曲线趋势
1000 iteration：导出并 play
2000~5000 iteration：判断是否值得长训
```

导出指定 checkpoint：

```bash
cd ~/quadruped_train
python quadruped_rl/scripts/play_go2.py \
  --headless \
  --load_run Jul09_18-30-10_rough \
  --checkpoint 950
```

导出后会生成：

```text
logs/go2/exported/policies/policy_1.pt
```

复制到本地部署：

```bash
scp hurricane@192.168.0.214:~/quadruped_train/logs/go2/exported/policies/policy_1.pt \
  /home/ry/project/quadruped/src/controller/rl/models/policy_1.pt
```

本地 MuJoCo 检查：

```bash
cd /home/ry/project/quadruped
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py
```

## 常见误区

### 只看 mean_reward

错误。必须同时看：

```text
mean_episode_length
tracking_lin_vel
tracking_ang_vel
torques
action_rate
smoothness
dof_pos_limits
```

### 看到 reward 上升就继续加复杂随机化

不建议。先 play 看步态。如果基础步态还有拖脚、抖动、关节极限问题，不要急着开 push 和 rough terrain。

### 看到 foot_clearance 为负就认为不好

不对。它是惩罚项，本来就是负值。重点看它是否异常变大，以及实际 play 是否抬腿合适。

### TensorBoard 显示好，MuJoCo 就一定好

不一定。TensorBoard 是训练仿真的统计，MuJoCo 是另一套物理模型。最终要用 MuJoCo 验证。

## 调参记录建议

每次训练记录：

```text
run_name:
commit:
是否 resume:
checkpoint:
max_iterations:

改动参数:
- 

TensorBoard:
- mean_reward:
- mean_episode_length:
- tracking_lin_vel:
- tracking_ang_vel:
- foot_clearance:
- torques:
- action_rate:
- smoothness:
- dof_pos_limits:
- estimation loss:
- mean_noise_std:

play 现象:
- 是否抬腿:
- 是否拖脚:
- 是否抖动:
- 是否摔倒:
- 速度是否偏大:
- 转向是否稳定:

结论:
- 保留 / 回退 / 继续微调
```

## 当前建议

当前训练参数已经打开：

```text
foot_clearance
noise
friction randomization
dof_pos_limits
smoothness
```

下一步建议：

```text
1. 先用 TensorBoard 判断 1000 iteration 左右曲线是否健康
2. 导出 900~1000 左右 checkpoint
3. 用 play 和 MuJoCo 看是否抬腿、是否抖动
4. 再决定是否增强 foot_clearance 或调整 action_rate/smoothness/torques
```
