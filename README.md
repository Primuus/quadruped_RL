# 四足机器人 HIM-PPO 训练工程

文档核对日期：2026-10-15。以下“当前配置”均指本分支源码，不代表已有 checkpoint 使用了这些参数。

## 阅读入口

| 需求 | 文档 |
| --- | --- |
| 安装、服务器连接和常见导入错误 | [中文安装指南](doc/setup_zh.md) / [Installation Guide](doc/setup_en.md) |
| 算法、网络、存储和 runner 的职责 | [本地 rsl_rl 说明](rsl_rl/README.md) |
| 参数生效、奖励、随机化和新增功能 | [训练调参教程](https://github.com/Primuus/quadruped_RL/blob/main/doc/训练调参教程.md) |
| 日志、曲线和按命令模式评估 | [TensorBoard 使用指南](https://github.com/Primuus/quadruped_RL/blob/main/doc/TensorBoard使用指南.md) |
| MuJoCo、模型资源和实机软件链路 | [部署工程](https://github.com/Primuus/quadruped_RL/tree/main) |

两份训练教程维护在部署分支的 `doc/`，不在本仓库复制参数表。`README_zh.md` 仅保留导航，避免同一说明出现两个不同版本。

`quadruped_train` 是强化学习训练工作区，负责在 Isaac Gym 中训练四足机器人策略，并从同一个部署模型直接导出给本地 `quadruped` 工程使用的 TorchScript 和 ONNX 策略。

当前已经接入的机器人实例是 Go2。框架设计不是 Go2 专用，后续新增其他四足机器人时，应新增机器人配置、资产和任务注册，而不是改 HIM-PPO 主算法。

## 工程定位

```text
quadruped_train
  -> Isaac Gym 并行训练
  -> HIMLoco 风格 actor / estimator / critic
  -> 直接导出 estimator + actor TorchScript / ONNX 策略
  -> quadruped 本地 MuJoCo 切换推理后端加载策略
```

本仓库只负责训练和策略导出，不包含 ROS2 部署链路。部署侧在本地 `quadruped` 工作区。

## 当前训练框架

通用四足训练链路：

```text
policy_class_name = HIMActorCritic
algorithm_class_name = HIMPPO
runner_class_name = HIMOnPolicyRunner
```

观测和网络结构：

```text
actor 单帧观测：45
actor 历史长度：6
actor 外部输入：45 * 6 = 270，当前适配 12 关节四足机器人
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

critic privileged obs 只用于训练 value function 和 estimator，不进入部署模型的外部输入。

## 当前状态与边界

| 项目 | 当前状态 |
| --- | --- |
| 已注册训练任务 | 只有 `go2`，不是所有机器人都已有可运行任务 |
| 训练环境 | Isaac Gym Preview / PhysX；自定义 `quadruped_rl`，不是 Isaac Lab |
| 默认并行规模 | 4096 环境，20 s 回合，50 Hz 策略周期 |
| 训练地形 | 当前 `create_sim()` 只创建平地；配置保留地形字段不等于已接入粗糙地形 |
| critic 外力与高度 | 3 + 187 维目前为零占位；没有实际外力/高度数据回填 |
| 正式算法 | Go2 选择 HIM-PPO；普通 PPO 源码仍保留在算法包中，但不是当前任务所选链路 |
| Primuus | 模型在部署工程中；未注册训练任务，未提供匹配策略 |
| 实机验收 | 不在本仓库进行，部署软件存在不等于硬件已经验证 |

当前结构借鉴 HIMLoco 的历史观测、estimator 和非对称 actor-critic；不能把它理解为上游 HIMLoco 的所有地形、随机化和机器人能力均已复现。

## 目录说明

- `quadruped_rl/envs/base/`：四足机器人通用环境基类、观测、reward、随机化、命令和终止条件。
- `quadruped_rl/envs/go2/`：当前 Go2 机器人实例配置、资产适配和兼容封装。
- `quadruped_rl/scripts/train.py`：通用训练入口，通过 `--task <robot>` 选择机器人任务。
- `quadruped_rl/scripts/play.py`：通用 play / 策略导出入口，通过 `--task <robot>` 选择机器人任务。
- `quadruped_rl/scripts/train_go2.py`：Go2 快捷训练入口，会自动设置 `--task go2`。
- `quadruped_rl/scripts/play_go2.py`：Go2 快捷 play / 策略导出入口。
- `quadruped_rl/utils/`：任务注册、导出、日志和工具；`terrain.py` 尚未接入当前平地创建链路。
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

4. 先安装仓库自带的算法包，再安装训练工程：

   ```bash
   python -m pip install --no-deps -e ./rsl_rl
   python -m pip install --no-deps -e .
   ```

`rsl_rl` 已随本仓库同步，不需要单独克隆，但必须安装它。只把仓库根目录加入 `PYTHONPATH` 仍找不到 `rsl_rl.env`。只有确认要更新算法库时，才从官方仓库重新同步。

已有服务器环境保留原依赖，不在训练过程中升级或降级包。当前 `setup.py` 声明 `numpy==1.20`，而已工作的服务器环境为 `numpy==1.23.2`；`--no-deps` 安装只建立源码映射，不代表依赖已经全部安装或元数据约束已经消除。环境清单和新环境步骤见安装指南。

ONNX 导出依赖 `onnx`；若当前环境尚未安装，需要在停训后补充：

```bash
pip install "onnx>=1.14,<2"
```

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

通用训练入口：

```bash
python quadruped_rl/scripts/train.py --task go2 --headless
```

Go2 快捷训练入口：

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
| `--max_iterations` | `--max_iterations 10000` | 本次 `learn()` 执行的迭代数；resume 时是追加量，不是总迭代上限。 |
| `--run_name` | `--run_name rough` | 本次实验名。 |
| `--experiment_name` | `--experiment_name go2` | 日志目录名。 |
| `--resume` | `--resume` | 从 checkpoint 继续训练。 |
| `--load_run` | `--load_run Oct15_20-00-00_rough` | 完整 run 目录名，不是 `--run_name` 的后缀。 |
| `--checkpoint` | `--checkpoint 500` | 指定 checkpoint 编号。 |
| `--rl_device` | `--rl_device cuda:0` | RL 算法设备。 |

`--horovod` 虽仍在参数解析器中，但当前 runner 未实现分布式训练链路，不作为可用多 GPU 命令推荐。

## 自碰撞配置

显示网格和碰撞体分别负责外观与物理约束。复杂的中空装配不能直接用一个 mesh 凸包代替，否则会填满关节的活动间隙。建议像 Go2 一样使用简化几何，或在 URDF 中引用离线分解的多个凸网格，并与 MuJoCo 使用相同的几何和装配排除规则。

`asset.self_collisions` 是 Isaac Gym 的数值过滤参数，不是布尔开关：

| 数值 | 行为 |
| --- | --- |
| `0` | 开启 actor 内可用的自碰撞；PhysX 仍过滤相邻父子 link。 |
| 正数，例如 `1` | 关闭全部自碰撞；当前 Go2 任务维持这一已有设置。 |
| `-1` | 使用导入资产的过滤位，适用于下面的命名排除接口。 |

`asset.self_collision_exclusions` 接受 `[(collision_name_a, collision_name_b), ...]`，只排除正常装配接触，不应排除整条腿或所有相邻 link。启用时要求：

- `self_collisions=-1`，否则 `create_actor` 会覆盖形状过滤位，环境会拒绝这类配置。
- 保持独立 link（`collapse_fixed_joints=False`），每个 URDF collision 对应一个预分解凸形状；不要再对这些网格自动执行 VHACD。
- 所有 collision 使用唯一且非空的名称；接口检查导入形状数量和引用名称，不匹配时直接报错。
- 每个排除对占一个过滤位，当前最多支持 31 对；未列出的非相邻、非同体组合允许碰撞，环境间仍由独立 collision group 隔离。

PhysX 会始终过滤 articulation 的直接父子 link，命名接口不能解除这一限制。需要这类部件发生接触时，可将碰撞几何挂在同坐标、固定连接的碰撞载体子 link，使实际接触对不再是直接父子。必须保留固定 link，并正确分配原部件的质量与惯量，使复合动力学不变；不能把碰撞几何移到足端上，否则会污染足端接触力。对于 Primuus，载体命名为 `*_calf_collision_link`，非足端接触选择须包含 `calf`，足端仍单独使用 `*_foot_link`。

关节限位应来自机械设计。不能仅为消除穿插而收紧范围，也不能用 reward 替代实际碰撞约束。Primuus URDF/XML 已恢复原始小腿范围 `[-2.7227, -0.8378] rad`，通过分段凸几何、固定碰撞载体及接触求解处理穿插；不是依靠修改关节零位或添加新驱动关节。

Primuus 的碰撞模型及接入条件维护在部署工程 `src/controller/rl/resources/robots/primuus/README.md`。其 21 个 link 包含 4 个仿真碰撞载体，仍只有 12 个驱动关节、4 个独立足端，不改变 `45 * 6` 观测契约。接入时必须完整复制模型目录、加载 20 对命名排除规则，并复核物理子步、接触偏移、策略周期和 PD 更新频率；不要把当前 Go2 参数视为所有机器人的默认物理参数。

实现位于 `quadruped_rl/envs/base/self_collision.py`，在资产导入后、创建 actor 前应用。Primuus 模型提供 `collision/self_collision_exclusions.json` 作为列表来源，但本次没有新增 Primuus 训练任务，也没有更改 Go2 的奖励、PD、观测或训练开关。

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

例如从 `model_1000.pt` 恢复并设置 `--max_iterations 500`，正常完成后保存 `model_1500.pt`。新进程创建新的日志目录，不把新曲线直接写回旧 run。`Ctrl+C` 不会额外强制保存当前更新；恢复点以磁盘上已经完整写出的 checkpoint 为准。

## 策略导出

训练完成后，用 play 入口加载 checkpoint，并从同一个 `estimator + actor` 部署模块直接导出 TorchScript 和 ONNX：

```bash
python quadruped_rl/scripts/play.py --task go2 --headless
```

Go2 快捷入口等价于：

```bash
python quadruped_rl/scripts/play_go2.py --headless
```

默认同时生成：

```text
logs/go2/exported/policies/policy_1.pt
logs/go2/exported/policies/policy_1.onnx
```

指定某个 run 或 checkpoint：

```bash
python quadruped_rl/scripts/play_go2.py \
  --headless \
  --load_run Oct15_20-00-00_rough \
  --checkpoint 1000
```

两个模型都直接从当前 PyTorch `estimator + actor` 导出，ONNX 不经过 TorchScript 二次转换。两者外部输入、输出和权重来源一致：

```text
输入：obs_history，当前 Go2 为 [batch, 270] float32
输出：actions，当前 Go2 为 [batch, 12] float32
ONNX opset：17
```

部署模型外部输入维度必须和机器人任务配置一致。当前 Go2 实例保持 270 维历史观测。

上例 run 名必须替换为实际目录。未指定 `--load_run` 时，从含 checkpoint 的 run 中选择 checkpoint 文件修改时间最新的目录；未指定 `--checkpoint` 时，在该 run 中选编号最大的 `model_*.pt`。这不是按最高 reward 选“最好策略”。

每次导出会覆盖同一实验目录下的 `policy_1.pt` 和 `policy_1.onnx`，不会覆盖训练 checkpoint。两个文件不是原子打包发布：如果 ONNX 导出失败，应重新完成双导出，不能混用新 PT 和旧 ONNX。导出后 `play` 仍继续运行仿真；`--headless` 只是不显示窗口，不是仅导出然后退出。

## 新增四足机器人

新增机器人时不要改 HIM-PPO 主算法，按实例扩展：

1. 在 `resources/robots/<robot>/` 添加 URDF/MJCF 资产。
2. 在 `quadruped_rl/envs/<robot>/` 新增配置类，继承 `LeggedRobotCfg` 和 `LeggedRobotCfgPPO`。
3. 在配置中替换 `asset.file`、`asset.name`、默认关节角、PD、命令范围、reward 权重和实验名。
4. 在 `quadruped_rl/envs/__init__.py` 注册新的 task，例如 `task_registry.register("<robot>", RobotEnv, RobotCfg(), RobotCfgPPO())`。
5. 用 `python quadruped_rl/scripts/train.py --task <robot> --headless` 训练。
6. 用 `python quadruped_rl/scripts/play.py --task <robot> --headless` 导出策略。

## 调参流程

### 命令覆盖与转向换步

Go2 的 `commands.mode_probabilities=[0.25, 0.25, 0.30, 0.20]` 依次对应原地转向、边走边转、平移和站立，每 5 s 重采样一次。平移最小速度为 0.12 m/s，原地转向最小偏航速度为 0.12 rad/s，均高于站立判定阈值。`max_translation_speed=1.0` 限制 XY 合成后的速度模长，而非仅限制各分量。

`low_speed_probability=0.30` 将 30% 命令划入低速子集，`low_speed_scale=0.35` 缩小其 XY/yaw 采样范围，其余命令保留完整范围。这些值均在机器人配置中调整，不改变策略输入格式。

`turn_step=0.1` 不再按落地次数直接累加：每足最多积累 `turn_step_reward_interval=0.5` s 的时间预算，符合腾空时长和抬升条件的落地消费该预算，四足取平均后计分。长期平均贡献不超过约 0.1/s；提高落地频率不会提高单位时间预算。奖励同时乘以平移和偏航跟踪分数、同向转动进度，并在持续非足端接触时关闭。`Command/*/valid_steps_per_second` 仍统计原始有效落地次数，不能直接视为奖励大小。

### 站立和运动共用高度

Go2 使用 `base_height_target=0.28` m，`base_height_tolerance=0.03` m。因此站立、平移和转向时，高度在 0.25-0.31 m 内均不受高度惩罚；超出区间才产生连续平方惩罚，不强制固定高度，也不会因高度越界直接终止回合。

```text
excess = max(abs(base_z - base_height_target) - base_height_tolerance, 0)
raw_height_penalty = (excess / base_height_sigma)^2
weighted_height_reward = base_height * raw_height_penalty * control_dt
```

当前 `base_height=-0.4`、`base_height_sigma=0.05` m。这里的高度是平地上机身根节点的世界 z；0.42 m 仅为初始落地高度，不是运动目标。新增机器人时，应按其正常承重姿态设置目标和容差。Go2 关闭单独的 `stand_height`，避免站立时重复计罚。

### 摆动足和非足端支撑

`foot_clearance=-0.01` 只约束非支撑足。足端位置与相对速度均转入机身坐标系，速度扣除机身平移和旋转分量；高度误差使用 `clearance_height_sigma=0.05` m 归一化，再按水平摆动速度计罚。`clearance_height_target=-0.20` m 是相对机身的目标高度，不是离地抬脚 20 cm，也不是对支撑足的要求。

Go2 的 `collision=-1.0` 惩罚持续大腿/小腿接触；力阈值 5 N、持续时间 0.06 s 用于过滤低力噪声和短碰撞。此项仍是软惩罚，不直接终止回合。观察 `Episode/contact_<body>` 的触地时间占比，以及 `Episode/rew_collision`，不能仅凭总奖励认定足端支撑正确。

`only_positive_rewards=False` 保留负总奖励，避免不同程度的低趴、碰撞都被截成零；新版本的总奖励数值不能与旧版直接比较。摩擦随机化范围为 `[0.4, 1.25]`，包含当前 MuJoCo 足端配置值，但并不消除两种引擎的动力学差异。

### 零命令站立

站立奖励要求平移命令模长不超过 `rewards.stand_lin_vel_threshold=0.1` m/s，且偏航命令绝对值不超过 `stand_ang_vel_threshold=0.1` rad/s。原地转向不满足这一条件，因此不会被要求保持默认站姿或四足同时支撑。

Go2 启用以下两项站立专用约束，高度由全局 `base_height` 负责：

| 参数 | 当前权重 | 含义 |
| --- | --- | --- |
| `stand_still` | `-0.1` | 惩罚实际关节角相对默认站姿的绝对偏差之和 |
| `stand_contacts` | `-0.4` | 惩罚缺少支撑的足端比例；四足支撑时为 0 |
| `stand_height` | `0.0` | 当前关闭，避免与全局高度奖励重复 |

足端向上接触力超过 `stand_contact_force_threshold=5.0` N 才算支撑，允许一次控制步的接触力漏检，历史在回合重置时清空。原地转向和行走不会被要求四足同时支撑。回合重置后重新计算机身速度、姿态和重力投影，避免首帧观测混入上一回合状态。

观察 `Episode/rew_base_height`、`Episode/rew_stand_still`、`Episode/rew_stand_contacts`，以及各 `Command/<mode>/base_height`、`joint_pose_error`、`supporting_feet`。后者分别是平均机身高度、关节角绝对偏差之和、按上述力阈值与漏检过滤统计的支撑足数；不要把原有未经滤波的 `feet_in_contact` 与 `supporting_feet` 混为同一指标。

本轮不改变 HIM-PPO、4096 环境、270 维策略输入、238 维 critic、12 维动作、PD 和动作缩放。奖励定义及命令分布已改变，建议新建实验从头训练；旧 checkpoint 虽然维度兼容，但其奖励值和收敛曲线不能与新目标直接比较。已经导出的模型不会随源码修改而变化。

```bash
cd ~/quadruped_train
conda activate isaacgym
python quadruped_rl/scripts/train_go2.py --headless --max_iterations 1500 --run_name locomotion_fix
```


每次只调整一个参数组：

```text
1. 修改一个参数组
2. 核对修改范围并同步训练服务器
3. 在服务器完成导入或语法检查
4. 由用户在服务器做小步训练 2 或 100 iteration
5. 短训 500 到 1000 iteration
6. play 观察步态
7. 导出 TorchScript / ONNX 后在 quadruped 中做双后端 MuJoCo 检查
8. 记录结论并提交
```

推荐调参顺序，具体步骤见调参教程：

```text
1. 核对模型、接触、站姿和训练/部署契约
2. 按站立、平移、原地转向、边走边转分别评估
3. 微调已有高度、接触、抬脚、滑移与换步奖励
4. 调整现有噪声、摩擦、机身质量和 push
5. 其他电机随机化、延迟、粗糙地形和真实高度输入：先实现并验证，再调参数
```

判断参数是否有效时，不只看总 reward，还要看速度跟踪、力矩、动作平滑、脚是否拖地、是否频繁摔倒，以及 MuJoCo 部署效果。

## 与 quadruped 的关系

```text
quadruped_train 训练
  -> play_go2.py 同时导出 policy_1.pt / policy_1.onnx
  -> 复制到 quadruped/src/controller/rl/models/
  -> quadruped 本地 MuJoCo 分别运行 TorchScript / ONNX Runtime
```

`quadruped_train` 不负责 ROS2 控制和实机链路；`quadruped` 不负责训练。
