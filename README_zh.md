<div align="center">
  <h1 align="center">Go2 RL 训练</h1>
</div>

<p align="center">
  <strong>基于 Isaac Gym 和 quadruped_rl 的 Go2 训练工作区。</strong>
</p>

## 目录说明

 - 这里只保留 Go2。
 - 训练代码运行在服务器上的 `quadruped_train`。
 - 部署代码放在本地 `quadruped` 工作区。
 - 本仓库不包含 ROS2 部署链路。

## 训练前准备

1. 进入工作区：

   ```bash
   cd ~/quadruped_train
   ```

2. 激活已有的 Isaac Gym conda 环境。

3. 确认当前环境里可以使用 `isaacgym`。

`rsl_rl` 已经随本仓库同步为官方版本，不需要单独克隆；如需单独更新，可以直接从官方仓库拉取。

## Docker 运行

Docker 里的训练环境和本机 conda 环境是分开的，不会互相影响。

前提条件：

- 已安装 Docker 和 NVIDIA Container Toolkit
- 已能访问 `nvcr.io/nvidia/pytorch:22.12-py3`，首次构建前通常需要先 `docker login nvcr.io`
- 本机已经下载并解压 Isaac Gym，推荐放在仓库根目录下的 `isaacgym/`

如果 Isaac Gym 不在仓库根目录下，再设置 `ISAACGYM_ROOT`，或者复制 `.env.example` 作为参考。

如果要在桌面环境里看窗口，可先放开 X11：

```bash
xhost +local:root
```

构建并启动容器：

```bash
bash docker/build.sh
```

直接跑安全验证命令：

```bash
bash docker/build.sh python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2
```

进入容器：

```bash
bash docker/build.sh
```

确认安全验证通过后的长训示例：

```bash
python quadruped_rl/scripts/train_go2.py --headless --num_envs 4096 --seed 1 --max_iterations 10000
```

## 训练命令

首次测试或修改训练配置后，先运行安全验证命令：

```bash
CUDA_LAUNCH_BLOCKING=1 python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2
```

常规小步验证命令：

```bash
python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2
```

安全验证通过后，再运行基础训练命令：

```bash
python quadruped_rl/scripts/train_go2.py --headless
```

这条命令已经默认使用 Go2，并会写入：

```bash
python quadruped_rl/scripts/train_go2.py \
  --task go2 \
  --experiment_name go2 \
  --run_name rough \
  --max_iterations 10000
```

## 当前训练配置

当前处于 HIMLoco 化阶段 4：保留 4096 环境和普通 PPO，actor 输入为 `45*6=270` 维历史观测，critic 额外使用 48 维 privileged obs。默认仍关闭 foot clearance、rigid body state reward、joint power、smoothness、噪声、域随机化、随机外推和高度采样。单步观测使用 HIMLoco 风格的 45 维，不直接输入机身线速度：

```text
[0:3]   速度命令 vx, vy, yaw_rate
[3:6]   机身角速度
[6:9]   重力方向在机身坐标系下的投影
[9:21]  12 个关节位置偏差
[21:33] 12 个关节速度
[33:45] 上一次策略动作
```

历史观测顺序为 `[当前 45 维, 上一帧 45 维, ..., 更早第 5 帧 45 维]`。critic privileged obs 为 `[当前 45 维观测, base_lin_vel 3 维]`，只用于训练 value function，不进入导出的 actor 策略。因为 actor/critic 输入维度已经变化，旧的 48 维、45 维或阶段 3 checkpoint 都不要继续 `--resume`。当前阶段训练出的策略仍是 270 维普通 PPO actor，不是最终 HIM-PPO 策略。

## 参数写法

命令行参数分两种：

- 开关型参数：直接写参数名，不带值。
  - `--headless`
  - `--resume`
  - `--horovod`
- 带值参数：参数名后面空格接值。
  - `--num_envs 4096`
  - `--seed 1`
  - `--checkpoint 500`
  - `--rl_device cuda:0`

多个参数可以连写：

```bash
python quadruped_rl/scripts/train_go2.py --headless --num_envs 4096 --seed 1
```

## 可用参数

| 参数 | 类型 | 示例 | 说明 |
| --- | --- | --- | --- |
| `--task` | 字符串 | `--task go2` | 任务名。当前只保留 Go2，通常不用改。 |
| `--resume` | 开关 | `--resume` | 从已有 checkpoint 继续训练。 |
| `--experiment_name` | 字符串 | `--experiment_name go2` | 日志目录名，默认 `go2`。 |
| `--run_name` | 字符串 | `--run_name rough` | 本次实验名，默认 `rough`。 |
| `--load_run` | 字符串 | `--load_run Jan01_12-00-00_rough` | 继续训练时指定要加载的 run；不填时自动取最近一次。 |
| `--checkpoint` | 整数 | `--checkpoint 500` | 继续训练时指定模型编号；不填时自动取最新 checkpoint。 |
| `--headless` | 开关 | `--headless` | 不打开窗口，适合服务器训练。 |
| `--horovod` | 开关 | `--horovod` | 多 GPU 训练开关。 |
| `--rl_device` | 字符串 | `--rl_device cuda:0` | RL 算法使用的设备。 |
| `--num_envs` | 整数 | `--num_envs 4096` | 并行环境数量。 |
| `--seed` | 整数 | `--seed 1` | 随机种子。 |
| `--max_iterations` | 整数 | `--max_iterations 20000` | 最大训练迭代次数。 |

## 常用示例

```bash
# 服务器上无窗口训练
python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2

# 安全验证通过后，指定并行环境数和随机种子
python quadruped_rl/scripts/train_go2.py --headless --num_envs 4096 --seed 1

# 继续上一次训练
python quadruped_rl/scripts/train_go2.py --resume

# 继续指定 run 和 checkpoint
python quadruped_rl/scripts/train_go2.py --resume --load_run rough --checkpoint 500

# 改实验名和训练轮数
python quadruped_rl/scripts/train_go2.py --experiment_name go2_v2 --run_name rough2 --max_iterations 20000
```

## 导出策略

如果需要把训练好的策略导出为 JIT 模型，运行：

```bash
python quadruped_rl/scripts/play_go2.py
```

导出文件默认保存在 `logs/go2/exported/policies/policy_1.pt`。

## 说明

- 训练在服务器执行。
- 本地 `quadruped` 工作区只负责部署和加载策略。
- 本仓库只保留 Go2。
