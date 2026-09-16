# RL 部署子工程说明

`src/controller/rl` 是纯 Python 四足机器人 RL 部署包，不依赖 ROS2、ament 或 `colcon`。它包含两条正式运行链路：

- MuJoCo：读取机器人模型和 TorchScript/ONNX 策略，在本地完成闭环仿真；
- 实机：读取真实机器人配置，通过电机和 IMU 构造 45×6 actor 观测，执行 ONNX 策略并发送目标关节角。

当前 MuJoCo 示例使用 Go2。实机配置模板面向后续自研四足机器人，当前电机和 IMU 实现分别为 Unitree GO-M8010-6 与 Yesense YIS130。公共运行器不包含 Go2 专有分支。

## 总体数据契约

单帧 actor 观测固定为 45 维：

| 切片 | 维度 | 内容 |
| --- | ---: | --- |
| `[0:3]` | 3 | 速度命令 `[vx, vy, yaw_rate]` 及缩放 |
| `[3:6]` | 3 | 机身坐标系角速度及缩放 |
| `[6:9]` | 3 | 重力方向在机身坐标系的投影 |
| `[9:21]` | 12 | `(joint_pos - default_angles) * dof_pos_scale` |
| `[21:33]` | 12 | `joint_vel * dof_vel_scale` |
| `[33:45]` | 12 | 上一帧策略动作 |

历史缓存采用最新帧在前的顺序：

```text
[obs_t, obs_t-1, obs_t-2, obs_t-3, obs_t-4, obs_t-5]
```

因此 actor 输入为 `45 * 6 = 270` 维，输出为 12 维 action：

```text
target_joint_position = default_angles + action_scale * action
```

训练 critic 使用的 privileged observations 不进入部署侧。

## 目录结构

```text
src/controller/rl/
├── config/
│   ├── go2.yaml
│   └── real/custom_quadruped.example.yaml
├── models/
├── resources/robots/go2/
├── rl_controller/
│   ├── deploy_mujoco_rl.py
│   ├── deploy_real_rl.py
│   ├── core/
│   ├── input/
│   └── real/
├── third_party/
├── tools/compare_policy_backends.py
├── pyproject.toml
└── setup.py
```

## 顶层文件

### `setup.py`

定义 Python 包、Python 版本和运行依赖。依赖包含 NumPy、PyYAML、pyserial、evdev、PyTorch、MuJoCo 和 ONNX Runtime，不包含 ROS2 或测试框架。

### `pyproject.toml`

指定 setuptools 构建后端，使本目录可以通过以下命令进行 editable 安装：

```bash
python3 -m pip install -e src/controller/rl
```

### `config/go2.yaml`

Go2 MuJoCo 示例配置，负责策略路径、XML、关节/执行器顺序、PD、动作缩放、观测缩放、仿真频率、跌倒阈值和键盘参数。它只服务 MuJoCo，不应被复制为自研实机参数。

### `config/real/custom_quadruped.example.yaml`

真实四足机器人配置格式模板。模板中的 `null` 表示必须通过机械设计、训练配置、硬件手册或实机标定确定；解析器会拒绝未填写模板，从而避免带着占位参数启动硬件。

### `models/`

存放本地策略文件。通常包括：

```text
policy_1.pt
policy_1.onnx
```

模型默认不提交到 Git。实机入口只接受 ONNX；MuJoCo 可接受 TorchScript 或 ONNX。

### `resources/robots/<robot>/`

存放 MuJoCo XML、URDF 和网格资源。当前 `go2/scene_flat.xml` 是默认平地场景，和训练侧平地环境对齐；`go2/scene_obstacles.xml` 是可选障碍场景，`go2/go2.xml` 定义机器人本体。

### `tools/compare_policy_backends.py`

模型发布工具。它把同一批 270 维输入分别送入 TorchScript 和 ONNX，报告平均误差、最大误差、最大误差位置及 `allclose` 结果：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx
```

该工具不参与 MuJoCo 或实机运行时导入。

## MuJoCo 模块

### `rl_controller/deploy_mujoco_rl.py`

MuJoCo 正式入口和主循环，负责：

- 加载 `RobotRLConfig`、策略和 XML；
- 建立 MuJoCo 关节、执行器和配置顺序映射；
- 读取仿真状态并构造 45×6 历史观测；
- 按训练契约裁剪完整的 270 维观测；
- 执行 TorchScript/ONNX 推理；
- 通过 PD 计算 actuator 力矩；
- 读取电脑键盘速度命令；
- 根据机身高度和重力投影检测跌倒，跌倒后停止力矩但保留仿真；
- 使用 `--debug-steps N` 输出前 N 次策略周期的状态、原始动作、裁剪动作、目标角和力矩。

主要命令：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py

python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --headless --no-keyboard --command 0.5 0.0 0.0 --duration 30

python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard --duration 5 --debug-steps 100
```

### `rl_controller/core/config.py`

定义 `RobotRLConfig` 和 MuJoCo 默认配置。加载 YAML 后检查观测维度、历史长度、关节/执行器数量、PD 数组、命令范围和模型/资源路径。`Go2Config` 仅作为兼容别名，主接口是 `RobotRLConfig`。

### `rl_controller/core/observer.py`

定义 MuJoCo 侧 `RobotState`、四元数旋转、45 维单帧观测和 6 帧历史缓存。这里的切片顺序必须与训练 actor 和实机 `real/observation.py` 完全一致。

### `rl_controller/core/policy.py`

包含 `TorchScriptPolicy`、`OnnxRuntimePolicy` 和后端自动选择逻辑。MuJoCo 根据模型后缀或 `--policy-backend` 选择后端，并根据 `--device` 使用 CPU 或可用的 CUDA Execution Provider。

### `rl_controller/core/pd.py`

执行仿真侧动作映射和 PD：

```text
target = default_angles + action * action_scale
torque = kp * (target - joint_pos) - kd * joint_vel
```

可按 MuJoCo actuator 能力进一步裁剪力矩。

## 键盘模块

### `rl_controller/input/keyboard_input.py`

使用 Linux evdev 枚举和读取实体键盘，维护按键快照与按下事件，并支持独占/释放设备。指定设备时应使用 `/dev/input/by-path/` 稳定路径。

### `rl_controller/input/keyboard_controller.py`

把 Linux `KEY_*` 映射成速度命令和控制事件。它负责三档速度、组合键、清零、使能、停机、重置和状态输出，不直接访问 MuJoCo 或电机。

默认按键：

| 按键 | 作用 |
| --- | --- |
| `F` | 使能并重置策略历史 |
| `I` | MuJoCo 中禁用控制；实机中请求正常安全停机 |
| `W/S` | 前进/后退 |
| `A/D` | 左移/右移 |
| `Q/E` | 左转/右转 |
| `Space` | 清零速度命令 |
| `1/2/3` | 低/中/高速档 |
| `R` | 仿真重置 |
| `P` | 输出状态 |

## 实机配置与状态

### `rl_controller/real/config.py`

定义全部实机配置 dataclass 和严格 YAML 解析器。启动时一次性检查：

- schema 和未知字段；
- 关节顺序、默认角、观测/动作维度和历史长度；
- ONNX 路径、后缀和文件存在性；
- 电机总线、ID、方向、减速比、限位、最大目标步长和 PD；
- YIS130 设备、波特率、四元数顺序、安装旋转和角速度单位；
- 状态、推理、温度、倾角和关闭超时；
- 键盘映射、命令范围和速度档位。

当前正式后端为 `onnx`、`unitree_m8010` 和 `yesense_yis130`。配置通过后，内部模块按可信强类型配置运行，不重复解析 YAML。

### `rl_controller/real/types.py`

定义不可变的数据契约：

- `JointState`：完整关节位置、速度、力矩、温度、错误码和时间戳；
- `ImuState`：机身坐标系四元数、角速度和时间戳；
- `JointCommand`：输出轴目标位置、速度、KP、KD、前馈力矩和时间戳；
- `RealRobotState`：同步后的关节与 IMU 快照。

数组在构造时复制并设为只读，防止线程之间修改同一状态对象。

### `rl_controller/real/orientation.py`

提供四元数归一化、乘法、逆旋转和 IMU 安装旋转。YIS130 原始姿态与角速度在硬件边界统一转换到机器人机身坐标系。

### `rl_controller/real/state.py`

`RealStateAggregator` 只负责状态聚合：

- 等待完整电机与 IMU 状态；
- 保证 12 关节完整；
- 检查状态年龄；
- 检查多电机总线与 IMU 的时间偏差；
- 发布一个不可变的同步快照。

协议、CRC、单位和有限值由硬件后端负责，聚合器不重复解析硬件数据。

### `rl_controller/real/observation.py`

根据同步状态构造与训练侧一致的 45 维单帧观测，并维护 6 帧最新帧优先历史。它负责历史预热、时间戳递增、帧间断重置和固定切片顺序；观测有限值与裁剪由策略边界统一处理。

## 实机硬件

### `rl_controller/real/hardware/base.py`

定义 `MotorBackend` 和 `ImuBackend` Protocol，以及设备不可用、通信失败等统一异常。运行器只依赖协议，不依赖具体机器人名称。

### `rl_controller/real/hardware/unitree_m8010.py`

GO-M8010-6 多总线后端，负责：

- 延迟加载固定版本 Unitree Actuator SDK；
- 检查 SDK API 和减速比；
- 并行执行多条 RS485 总线；
- 校验返回帧、CRC 和 motor ID；
- 把电机转子侧位置、速度、力矩转换为关节输出轴单位；
- 确保每次反馈包含全部配置关节；
- 记录各总线时间戳和错误；
- 使用 BRAKE 模式实现幂等 `safe_stop()`；
- 在关闭串口前再次执行安全停机。

### `rl_controller/real/hardware/yesense_yis130.py`

YIS130 私有 UART 后端，负责：

- 从含噪字节流中寻找帧头；
- 校验长度、校验和和 TLV 数据字段；
- 解析角速度和四元数；
- 将 `deg/s` 转换为 `rad/s`；
- 根据配置的安装四元数转换到机身坐标系；
- 拒绝不完整、非有限或超时数据；
- 发布带单调时间戳的 `ImuState`。

## 实机策略

### `rl_controller/real/policy/base.py`

定义策略 Protocol、加载/契约/推理/超时异常，以及严格的 float32 一维张量检查。

### `rl_controller/real/policy/onnx_policy.py`

加载 ONNX Runtime CPU session，并在启动时检查：

- 恰好一个输入和一个输出；
- 输入/输出名称有效；
- rank 为 2；
- batch 为 1 或动态；
- dtype 为 `tensor(float)`；
- 特征维分别为 270 和 12。

运行时检查 ONNX 输出 batch、维度和 dtype。

### `rl_controller/real/policy/executor.py`

策略调用的唯一动态边界：

- 检查 270 维 float32 观测及 NaN/Inf；
- 按训练契约裁剪观测；
- 测量 ONNX 推理耗时并作废迟到结果；
- 检查 12 维 float32 动作及 NaN/Inf；
- 按 `clip_actions` 饱和动作；
- 第一次发生动作饱和时输出明确警告，并记录原始动作和累计次数。

### `rl_controller/real/policy/factory.py`

根据已验证实机配置创建 `OnnxPolicy + PolicyExecutor`。当前实机链路不接受 TorchScript；TorchScript 只用于 MuJoCo 和导出对比。

## 状态机、安全和运行器

### `rl_controller/real/state_machine.py`

定义不可逆生命周期：

```text
INIT -> READY -> MOVE_TO_DEFAULT -> RL
  \       \             \          \
   +-------+-------------+----------> FAULT -> SHUTDOWN
```

FAULT 锁存第一个 `FaultRecord` 作为根因，后续停机错误进入 `secondary_faults`，不会覆盖根因。记录包含：

```text
code, source, message, timestamp, runtime_state,
state_age_s, command, raw_action, target_position, cycle_duration_s
```

故障分类包括配置、硬件、策略、实时性、安全、输入、内部错误和关闭错误。

### `rl_controller/real/safety.py`

`RealSafetyManager` 只负责动态健康与最终输出授权：

- 电机温度和错误码；
- 机身 roll/pitch 倾倒阈值；
- 当前状态机是否允许主动命令；
- 目标关节角有限值和机械硬限位；
- 主动命令时间戳；
- 相邻目标关节角最大变化。

静态 PD、方向、减速比和配置维度只在启动配置边界检查。协议和反馈有限值由硬件后端检查，避免同一数组在多层重复扫描。

### `rl_controller/real/runner.py`

`RealRLRunner` 组织四个线程：

- motor：按 `motors.io_frequency_hz` 执行最终安全门、电机发送和反馈聚合；
- imu：读取、转换和聚合 YIS130 状态；
- policy：按 `policy.frequency_hz` 完成站姿过渡、历史观测和 ONNX 推理；
- monitor：检测阻塞的硬件或策略调用。

runner 负责：

- 只有第一份完整、新鲜、同步状态到达后才进入 READY；
- 从实测姿态平滑过渡到 `default_angles`；
- 历史预热完成后才进入 RL；
- 任意线程异常直接锁存 FAULT；
- 故障后停止新命令并执行幂等 `safe_stop()`；
- 正常停止和故障停止都关闭线程、电机、IMU 和输入资源；
- FAULT 后禁止自动恢复，必须重启并从 INIT 初始化。

### `rl_controller/deploy_real_rl.py`

实机正式 CLI。它不会提供默认机器人配置，必须显式传入完成后的 YAML 和硬件许可：

```bash
python3 src/controller/rl/rl_controller/deploy_real_rl.py \
  --config src/controller/rl/config/real/<robot>.yaml \
  --allow-real-hardware
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--config` | 必填，真实机器人 YAML |
| `--allow-real-hardware` | 必填，确认允许打开电机与 IMU |
| `--duration` | 指定运行秒数，`0` 表示直到停止或故障 |
| `--keyboard-device` | 指定 evdev 键盘设备 |
| `--no-keyboard` | 不打开键盘，配合固定命令使用 |
| `--command VX VY YAW_RATE` | 无键盘时的固定命令 |
| `--auto-enable` | 启动后自动进入站姿过渡 |

正常的 `I`、`Ctrl+C` 和 duration 到期走正常安全停机。运行故障打印一次完整根因快照并返回非零退出码。

## 边界原则

正式链路按以下五个边界分工：

1. **启动配置边界**：一次性拒绝错误、缺失和占位参数；
2. **硬件输入边界**：拒绝坏帧、错 ID、错误单位、不完整反馈和非有限值；
3. **策略边界**：拒绝错误 ONNX、观测、动作和迟到结果；
4. **最终输出边界**：每次发送前检查运行状态、最新状态、目标限位和电机健康；
5. **故障退出边界**：锁存根因、停止输出、`safe_stop()`、关闭资源并要求重新初始化。

内部模块只消费经过边界校验的数据，不使用默认值、旧状态或上一帧动作掩盖外部故障。

## 新增四足机器人

### MuJoCo

1. 在 `resources/robots/<robot>/` 放入 XML、URDF 和网格；
2. 新建 `config/<robot>.yaml`；
3. 填写关节与 actuator 顺序、默认角、PD、动作缩放和观测缩放；
4. 放入与该配置严格匹配的策略；
5. 运行 MuJoCo 并使用 `--debug-steps` 核对状态、动作、目标角和力矩。

### 实机

1. 复制 `config/real/custom_quadruped.example.yaml` 为独立机器人配置；
2. 根据真实机械结构填写关节顺序、ID、方向、零位、减速比和硬限位；
3. 根据训练配置填写 ONNX、默认角、动作/观测缩放和命令范围；
4. 标定 YIS130 安装旋转；
5. 根据硬件能力确定 PD、温度、倾角、时限和目标步长；
6. 若更换电机或 IMU，在 `hardware/base.py` Protocol 下增加新后端，并在配置和工厂中显式注册；
7. 由用户依次执行单电机、单总线、悬空关节、IMU、默认站姿、零命令和低速落地验证。

实机前必须具备独立物理急停和驱动器底层通信超时。Python watchdog 与 `safe_stop()` 不能替代硬件急停。
