# 四足机器人 HIMLoco 训练工作区

`quadruped_train` 是强化学习训练工作区，负责在 Isaac Gym 中训练四足机器人策略，并导出给本地 `quadruped` 工程使用的 TorchScript 策略。

当前已经接入的机器人实例是 Go2。框架设计不是 Go2 专用，后续新增其他四足机器人时，应新增机器人配置、资产和任务注册，而不是改 HIM-PPO 主算法。

## 工程定位

```text
quadruped_train
  -> Isaac Gym 并行训练
  -> HIMLoco 风格 actor / estimator / critic
  -> 导出 estimator + actor TorchScript 策略
  -> quadruped 本地 MuJoCo 加载策略
```

本仓库只负责训练和策略导出，不包含 ROS2 部署链路。部署侧在本地 `quadruped` 工作区。

## 当前训练框架

默认 Go2 训练链路：

```text
policy_class_name = HIMActorCritic
algorithm_class_name = HIMPPO
runner_class_name = HIMOnPolicyRunner
```

观测和网络结构：

```text
actor 单帧观测：45
actor 历史长度：6
actor 外部输入：45 * 6 = 270
HIMEstimator：obs_history 270 -> vel3 + latent16
actor 内部输入：当前45 + vel3 + latent16 = 64
action：12
critic privileged obs：45 + base_lin_vel3 + external_force3 + height_scan187 = 238
```

单帧 45 维 actor 观测顺序：

```text
[0:3]   速度命令 vx, vy, yaw_rate
[3:6]   机身角速度
[6:9]   重力方向在机身坐标系下的投影
[9:21]  12 个关节位置偏差
[21:33] 12 个关节速度
[33:45] 上一次策略动作
```

历史观测顺序为：

```text
[当前 45 维, 上一帧 45 维, ..., 更早第 5 帧 45 维]
```

critic privileged obs 只用于训练 value function 和 estimator，不进入部署侧 TorchScript 外部输入。

## 目录说明

- `quadruped_rl/envs/base/`：四足机器人通用环境基类、观测、reward、随机化接口。
- `quadruped_rl/envs/go2/`：Go2 配置、观测封装、命令、reward、终止条件和随机化逻辑。
- `quadruped_rl/scripts/train_go2.py`：Go2 训练入口。
- `quadruped_rl/scripts/play_go2.py`：Go2 play / 策略导出入口。
- `quadruped_rl/utils/`：任务注册、地形、日志和通用工具。
- `rsl_rl/`：随工程同步的 RL 算法库，当前包含 HIM-PPO 相关实现。
- `docker/`：Docker 训练环境构建脚本。
- `logs/`：训练日志和 checkpoint 输出目录，通常不提交。

## 环境准备

1. 进入工作区：

   ```bash
   cd ~/quadruped_train
   ```

2. 激活 Isaac Gym conda 环境。

3. 确认 Python 能导入 `isaacgym`。

4. 安装当前工程：

   ```bash
   pip install -e .
   ```

`rsl_rl` 已随本仓库同步，不需要单独克隆。只有确认要更新算法库时，才从官方仓库重新同步。

## Docker 运行

Docker 环境和本机 conda 环境互相独立。

前提条件：

- 已安装 Docker 和 NVIDIA Container Toolkit。
- 已能访问 `nvcr.io/nvidia/pytorch:22.12-py3`。
- 已下载并解压 Isaac Gym，推荐放在仓库根目录下的 `isaacgym/`。

如果 Isaac Gym 不在仓库根目录下，可以设置 `ISAACGYM_ROOT`，或参考 `.env.example`。

构建并进入容器：

```bash
bash docker/build.sh
```

直接在容器里执行训练检查：

```bash
bash docker/build.sh python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2
```

## 训练命令

修改训练代码或配置后，先做小步稳定性检查：

```bash
python quadruped_rl/scripts/train_go2.py --headless --max_iterations 2
python quadruped_rl/scripts/train_go2.py --headless --max_iterations 100
```

基础训练：

```bash
python quadruped_rl/scripts/train_go2.py --headless
```

常用长训示例：

```bash
python quadruped_rl/scripts/train_go2.py \
  --headless \
  --num_envs 4096 \
  --seed 1 \
  --max_iterations 10000 \
  --run_name rough
```

## 参数写法

开关型参数直接写参数名：

```bash
--headless
--resume
--horovod
```

带值参数在参数名后接值：

```bash
--num_envs 4096
--seed 1
--checkpoint 500
--rl_device cuda:0
```

常用参数：

| 参数 | 示例 | 说明 |
| --- | --- | --- |
| `--task` | `--task go2` | 任务名，当前默认 Go2。 |
| `--headless` | `--headless` | 无窗口运行，适合服务器。 |
| `--num_envs` | `--num_envs 4096` | 并行环境数量。 |
| `--seed` | `--seed 1` | 随机种子。 |
| `--max_iterations` | `--max_iterations 10000` | 最大训练迭代次数。 |
| `--run_name` | `--run_name rough` | 本次实验名。 |
| `--experiment_name` | `--experiment_name go2` | 日志目录名。 |
| `--resume` | `--resume` | 从 checkpoint 继续训练。 |
| `--load_run` | `--load_run Jul08_10-00-00_rough` | 指定要加载的 run。 |
| `--checkpoint` | `--checkpoint 500` | 指定 checkpoint 编号。 |
| `--rl_device` | `--rl_device cuda:0` | RL 算法设备。 |

## 续训规则

适合 `--resume` 的情况：

- 只微调 reward 权重。
- 只打开小范围噪声或小范围随机化。
- obs/action/network 结构没有变化。
- 想在已有步态上继续增强鲁棒性。

不建议 `--resume` 的情况：

- 改了 obs 维度、privileged obs 维度、history 长度或 action 维度。
- 改了 actor/critic/estimator 输入结构。
- 大改 reward 目标。
- 同时打开多个强随机化项。
- 需要公平比较两个参数版本的收敛速度。

## 策略导出

训练完成后，用 play 入口加载 checkpoint 并导出 TorchScript：

```bash
python quadruped_rl/scripts/play_go2.py --headless
```

默认导出路径：

```text
logs/go2/exported/policies/policy_1.pt
```

指定某个 run 或 checkpoint：

```bash
python quadruped_rl/scripts/play_go2.py \
  --headless \
  --load_run rough \
  --checkpoint 1000
```

导出的 TorchScript 内部包含 `estimator + actor`，外部输入保持 270 维历史观测。

## 调参流程

每次只调整一个参数组：

```text
1. 修改一个参数组
2. 本地 py_compile / import 检查
3. 同步到服务器
4. 服务器小步训练 2 或 100 iteration
5. 短训 500 到 1000 iteration
6. play 观察步态
7. 导出 JIT 后在 quadruped 中做 MuJoCo 检查
8. 记录结论并提交
```

推荐调参顺序：

```text
joint_power
foot_clearance
noise
friction / base mass randomization
push
motor strength randomization
kp/kd randomization
delay
command curriculum
terrain curriculum / height scan 内容启用
```

判断参数是否有效时，不只看总 reward，还要看速度跟踪、力矩、动作平滑、脚是否拖地、是否频繁摔倒，以及 MuJoCo 部署效果。

## 与 quadruped 的关系

```text
quadruped_train 训练
  -> play_go2.py 导出 logs/go2/exported/policies/policy_1.pt
  -> 复制到 quadruped/src/controller/rl/models/policy_1.pt
  -> quadruped 本地 MuJoCo 运行
```

`quadruped_train` 不负责 ROS2 控制和实机链路；`quadruped` 不负责训练。
