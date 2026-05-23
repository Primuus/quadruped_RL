<div align="center">
  <h1 align="center">Go2 RL 训练</h1>
</div>

<p align="center">
  <strong>基于 Isaac Gym 和 legged_gym 的 Go2 训练工作区。</strong>
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

3. 确认当前环境里可以使用 `isaacgym` 和 `rsl-rl`。

## 训练命令

基础命令：

```bash
python legged_gym/scripts/train_go2.py
```

这条命令已经默认使用 Go2，并会写入：

```bash
python legged_gym/scripts/train_go2.py \
  --task go2 \
  --experiment_name go2 \
  --run_name rough \
  --max_iterations 10000
```

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
python legged_gym/scripts/train_go2.py --headless --num_envs 4096 --seed 1
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
python legged_gym/scripts/train_go2.py --headless

# 指定并行环境数和随机种子
python legged_gym/scripts/train_go2.py --headless --num_envs 4096 --seed 1

# 继续上一次训练
python legged_gym/scripts/train_go2.py --resume

# 继续指定 run 和 checkpoint
python legged_gym/scripts/train_go2.py --resume --load_run rough --checkpoint 500

# 改实验名和训练轮数
python legged_gym/scripts/train_go2.py --experiment_name go2_v2 --run_name rough2 --max_iterations 20000
```

## 导出策略

如果需要把训练好的策略导出为 JIT 模型，运行：

```bash
python legged_gym/scripts/play_go2.py
```

导出文件默认保存在 `logs/go2/exported/policies/policy_1.pt`。

## 说明

- 训练在服务器执行。
- 本地 `quadruped` 工作区只负责部署和加载策略。
- 本仓库只保留 Go2。
