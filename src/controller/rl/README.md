# RL 本地部署子工程说明

`src/controller/rl` 是 `quadruped` 的纯 Python RL 部署包，不依赖 ROS2、ament 或 `colcon`。当前可运行入口负责在本地 MuJoCo 中加载 `quadruped_train` 从同一个部署模型直接导出的 TorchScript 或 ONNX 策略；同时已经建立未来自研四足实机部署的公共接口、配置格式和厂商 SDK 集成基础。

当前已经接入 MuJoCo 的机器人实例是 Go2；核心代码按四足机器人通用接口组织，新增其他四足机器人时应新增 YAML 配置、物理资源和策略文件，而不是复制部署主循环。本仓库不包含训练代码。实机侧目前完成方案阶段 0–8，具有 Mock 组件、GO-M8010-6 电机后端、YIS130 IMU 后端、状态聚合、45×6 历史观测和严格的 Mock/ONNX 策略执行层，但还没有实机 runner 或可执行入口，默认运行方式不能访问真实设备。

## MuJoCo 数据流

```text
MuJoCo state
  -> RobotState
  -> 45 维单帧观测
  -> history buffer，形成 num_obs 维 policy 输入
  -> policy backend：TorchScript / ONNX Runtime
  -> estimator + actor
  -> num_actions 维 action
  -> action_scale + default_angles，得到目标关节角
  -> PD 计算力矩
  -> MuJoCo actuator
```

训练侧 critic 使用的 privileged obs 不进入这里的策略输入。部署侧只负责按 YAML 配置构造历史观测并调用所选推理后端；更换 TorchScript / ONNX 后端不会改变观测、动作裁剪、PD、MuJoCo 或键盘输入逻辑。

## 实机链路当前状态

未来实机链路采用 Python、ONNX Runtime、YESENSE YIS130 IMU 和工程内固定版本的 Unitree Actuator SDK。目标数据流为：

```text
Linux evdev 键盘命令
  + 电机 q/dq 反馈
  + 机身 IMU
  -> 45 维单帧观测
  -> 6 帧历史，形成 270 维输入
  -> ONNX policy，输出 12 维 action
  -> 目标关节角与安全检查
  -> Unitree GO-M8010-6 电机后端
```

当前提供不接设备即可导入的公共状态类型、硬件协议、严格配置解析器、Mock 配置、三类 Mock 组件及故障注入、官方 Actuator SDK 的固定源码快照、两个真实硬件后端、完整状态聚合、实机历史观测和统一策略执行层。真实后端都延迟加载可选依赖，只有未来实机入口创建并调用 `open()` 时才打开设备；ONNX Runtime 也只在实际创建 ONNX 后端时导入。Mock runner、运行线程、状态机和实机入口属于后续阶段。

## 目录结构

```text
src/controller/rl/
  README.md
  pyproject.toml
  setup.py
  config/go2.yaml
  config/real/
    mock.yaml
    custom_quadruped.example.yaml
  models/
  resources/robots/go2/
  third_party/
    unitree_actuator_sdk/
    build_unitree_actuator_sdk.sh
  rl_controller/
    deploy_mujoco_rl.py
    core/
    input/
    real/
      config.py
      observation.py
      orientation.py
      state.py
      types.py
      hardware/
        base.py
        mock_motor.py
        mock_imu.py
        unitree_m8010.py
        yesense_yis130.py
      policy/
        base.py
        executor.py
        factory.py
        mock_policy.py
        onnx_policy.py
  test/
    compare_policy_backends.py
    deploy_mujoco_rl_debug.py
    deploy_mujoco_rl_keyboard_debug.py
    test_keyboard_controller.py
    test_mock_backends.py
    test_real_config.py
    test_real_interfaces.py
    test_real_policy.py
    test_real_state_observation.py
    test_unitree_m8010_backend.py
    test_yesense_yis130_backend.py
```

## 顶层文件

| 文件 | 作用 |
| --- | --- |
| `README.md` | 当前说明文档，解释 RL 子工程结构、文件职责和运行方式。 |
| `pyproject.toml` | 选择 setuptools 构建后端，使该目录可作为标准 Python 工程安装。 |
| `setup.py` | 纯 Python 包安装脚本，声明运行依赖和测试依赖。配置、模型和资源继续由仓库目录管理。 |

## 配置文件

### `config/go2.yaml`

当前 Go2 机器人实例的部署配置。它是部署侧最重要的配置文件，负责把训练侧和 MuJoCo 部署侧对齐。新增其他四足机器人时，建议复制一份新的 YAML，例如 `config/<robot>.yaml`，并替换 `robot_name`、`xml_path`、关节顺序、actuator 顺序、默认角、PD 和策略路径。

主要内容：

- 策略文件路径：`policy_path`
- MuJoCo 场景路径：`xml_path`
- 仿真时长和步长：`simulation_duration`、`simulation_dt`
- 控制降频：`control_decimation`
- 观测维度：`num_one_step_obs`、`history_length`、`num_obs`
- 训练侧 critic 对齐信息：`num_one_step_privileged_obs`、`num_privileged_obs`
- 关节顺序：`joint_names`
- actuator 顺序：`actuator_names`
- PD 参数：`kps`、`kds`
- 默认关节角：`default_angles`
- 动作缩放和动作裁剪：`action_scale`、`clip_actions`
- 初始速度命令和命令限制：`command_init`、`command_limits`
- Linux 键盘映射和三档速度：`keyboard.bindings`、`keyboard.speed_levels`
- 跌倒检测阈值：`fall_height_threshold`、`fall_gravity_z_threshold`。`fall_height_threshold <= 0` 表示禁用机身高度检查。
- 观测缩放：`obs_scale_*`
- 初始机身位姿：`initial_base_pos`、`initial_base_quat`

修改原则：

- 训练侧改了 PD、默认角、action scale、clip actions 或观测缩放时，这里要同步。
- 部署侧 policy 外部输入必须和训练导出的模型输入维度一致。
- privileged obs 只用于配置对齐和检查，不会输入部署 policy。

### `config/real/mock.yaml`

完整且可解析的实机链路离线配置。它使用 Mock 策略、Mock 电机和 Mock IMU，不加载模型、不导入 Unitree SDK、也不打开串口。配置中的关节限位、增益和安全阈值只用于后续故障测试，不能作为真实机器人的默认值。

### `config/real/custom_quadruped.example.yaml`

未来自研四足的真实配置模板。已知的接口契约保留为 12 维动作、45 维单帧观测、6 帧历史和 270 维输入；机械零位、方向、Motor ID、总线设备、关节限位、增益、IMU 安装旋转、安全阈值和 ONNX 路径等未知值保持 `null`。模板本身应当解析失败，目的是防止未标定参数被误用于实机。

## 策略文件

### `models/`

默认策略目录。仓库不内置训练好的策略，训练完成后把从同一个 `estimator + actor` 直接导出的两个模型放到：

```text
models/policy_1.pt
models/policy_1.onnx
```

相关文件：

- `models/.gitignore`：避免误提交模型权重。
- `models/README.md`：说明模型目录的用途。

`policy_1.pt` 是过渡期的 TorchScript 基准，`policy_1.onnx` 是 ONNX Runtime 模型。两者权重来源、外部输入和动作输出应一致；ONNX 必须从 PyTorch `estimator + actor` 直接导出，不经过 TorchScript 二次转换。当前 Go2 策略输入为 270 维历史观测，输出为 12 维动作。

## MuJoCo 资源

### `resources/robots/go2/scene.xml`

当前 Go2 实例的 MuJoCo 场景入口文件。通常包含地面、光照、相机和 robot include。

### `resources/robots/go2/go2.xml`

当前 Go2 实例的 MuJoCo robot 模型文件。部署代码会根据 YAML 中的关节名和 actuator 名称，从这个模型中解析 qpos/qvel 地址和 actuator id。

### `resources/robots/go2/urdf/`、`resources/robots/go2/obj/`

当前 Go2 实例的机器人资产文件。用于模型几何、mesh 或 URDF 资源保留。新增机器人时应放到 `resources/robots/<robot>/` 下。

## Python 包入口

### `rl_controller/deploy_mujoco_rl.py`

RL 本地部署主入口。

负责内容：

- 读取机器人 YAML 配置，默认是 `go2.yaml`。
- 加载 MuJoCo 模型。
- 根据模型扩展名或 `--policy-backend` 加载 TorchScript / ONNX policy。
- 解析关节和 actuator 索引。
- 初始化仿真状态。
- 每隔 `control_decimation` 个 MuJoCo step 计算一次 policy action。
- 把 action 转成 PD 力矩。
- 写入 MuJoCo actuator。
- 默认接入当前 Linux 主机可见的本地键盘，也可用 `--no-keyboard` 运行固定命令。
- 检测摔倒后停止输出力矩，但不退出仿真。

主要类：

- `MujocoRLRunner`：封装 MuJoCo 仿真、策略推理、PD 输出和键盘输入。
- `Go2MujocoRunner`：兼容旧导入的别名，新代码优先使用 `MujocoRLRunner`。

常用命令行参数：

| 参数 | 作用 |
| --- | --- |
| `--config` | 指定 YAML 配置文件。 |
| `--policy-path` | 覆盖策略文件路径。 |
| `--policy-backend` | 推理后端：`auto`、`torchscript` 或 `onnx`；默认按模型扩展名选择。 |
| `--xml-path` | 覆盖 MuJoCo XML 路径。 |
| `--duration` | 覆盖仿真时长。 |
| `--command VX VY WZ` | 指定固定速度命令，需要同时使用 `--no-keyboard`。 |
| `--device` | 指定推理设备，如 `cpu` 或 `cuda:0`。ONNX 使用 CUDA 时需安装匹配的 `onnxruntime-gpu`。 |
| `--headless` | 不打开 MuJoCo viewer。 |
| `--debug-steps N` | 打印前 `N` 次 policy 推理的状态、动作和力矩信息，默认不打印。 |
| `--keyboard-device` | 手动限制为一个 Linux 键盘 event 设备；默认自动识别并监听所有合格键盘。 |
| `--no-keyboard` | 禁用 evdev 输入，改用 YAML 或 `--command` 固定命令。 |

## core 模块

### `rl_controller/core/config.py`

配置加载和校验模块。

负责内容：

- 定义 `RobotRLConfig` dataclass。
- 从 YAML 读取部署配置。
- 解析相对路径为绝对路径。
- 提供默认关节顺序、默认 PD、默认观测维度等常量。
- 校验动作维度、观测维度、history 长度、关节数量、actuator 数量。

这个文件是部署侧配置一致性的第一道检查。训练侧如果改变和部署有关的配置，应同步更新对应机器人 YAML 和这里的默认值或校验逻辑。`Go2Config` 目前保留为 `RobotRLConfig` 的兼容别名。

### `rl_controller/core/observer.py`

观测构造模块。

负责内容：

- 定义 `RobotState`，保存 MuJoCo 中读出的机身和关节状态。
- 提供四元数旋转工具：`quat_rotate`、`quat_rotate_inverse`。
- 按配置构造单帧 actor 观测。
- 维护 history buffer，输出 policy 输入。

当前 45 维观测顺序必须和 `quadruped_train` 保持一致：

```text
[0:3]   速度命令 vx, vy, yaw_rate
[3:6]   机身角速度
[6:9]   重力方向在机身坐标系下的投影
[9:21]  12 个关节位置偏差
[21:33] 12 个关节速度
[33:45] 上一次策略动作
```

### `rl_controller/core/policy.py`

策略推理后端模块。MuJoCo runner 只依赖统一的 `infer(observation)` 接口，不直接依赖具体模型格式。

负责内容：

- `Policy`：统一推理协议。
- `TorchScriptPolicy`：加载 `.pt` 并执行 PyTorch/TorchScript 推理。
- `OnnxRuntimePolicy`：加载 `.onnx` 并执行 ONNX Runtime 推理。
- `resolve_policy_backend()`：在 `auto` 模式下根据 `.pt` / `.onnx` 扩展名选择后端。
- `load_policy()`：创建对应后端实例。
- 将输入统一转换为连续的单帧 `float32` numpy 向量，并将输出统一转换为一维 `float32` action。

TorchScript 输出如果是 tuple/list，会取第一个输出作为 action。ONNX 模型必须只有一个输入，部署侧使用第一个输出作为 action。

### `rl_controller/core/pd.py`

action 到力矩的转换模块。

负责内容：

- 将 policy 输出 action 按配置中的 `clip_actions` 裁剪后映射为目标关节角：

  ```text
  target_pos = default_angles + action * action_scale
  ```

- 用 PD 计算关节力矩：

  ```text
  torque = kp * (target_pos - joint_pos) - kd * joint_vel
  ```

- 按 MuJoCo actuator torque limit 裁剪输出。

这个文件里的控制逻辑必须和训练侧 action scale、clip actions、默认角、PD 设定保持一致。当前 Go2 部署配置中 `clip_actions: 100.0`，对应训练侧 `normalization.clip_actions = 100`。

### `rl_controller/core/__init__.py`

core 模块导出文件，方便外部直接导入 `RobotRLConfig`、`RobotState`、`ObservationHistory` 等核心类型。`Go2Config`、`Go2State`、`Go2ObservationHistory` 仍作为旧名称别名保留。

## input 模块

RL 仿真输入直接读取运行主机可见的 Linux 键盘 event 设备，不依赖终端焦点或 MuJoCo viewer。它支持笔记本内置键盘以及连接到当前主机的 USB、蓝牙键盘；SSH 客户端按键不会自动成为远端主机的 evdev 事件。

### `rl_controller/input/keyboard_input.py`

Linux 键盘设备读取层。

负责内容：

- 优先查找 `/dev/input/by-path/*-event-kbd`。
- 按配置中的 `KEY_*` 能力筛选键盘，自动模式同时监听所有合格设备。
- 支持 `--keyboard-device` 手动限制为一个设备。
- 后台线程读取 `EV_KEY` 的按下、松开和长按事件。
- 线程安全地保存当前按键集合，并单独记录功能键的按下边沿。
- `F` 使能时通过 `EVIOCGRAB` 独占键盘，避免控制键同时触发 MuJoCo Viewer 快捷键；`I` 禁用时释放。
- 设备断开或读取异常时清空按键，供主循环禁用控制并清零命令。
- 程序退出时停止线程并关闭输入设备。

用户必须拥有 event 设备读取权限。可执行 `sudo usermod -aG input "$USER"` 并注销后重新登录，或者配置 udev 规则；不要用 root 运行完整 RL 控制程序。

### `rl_controller/input/keyboard_controller.py`

与具体输入设备、MuJoCo 和策略完全解耦的命令映射层。

负责内容：

- 将按住的方向键转换为 `[vx, vy, yaw_rate]`。
- 支持多方向组合和同轴反向键抵消。
- 应用 YAML 中配置的 25%、50%、100% 三档速度。
- 根据使能状态输出命令，并按 `command_limits` 裁剪。
- 处理停止、重新使能、MuJoCo 重置和状态打印等边沿事件。

键盘控制启动时默认禁用，需要先按 `F`。默认绑定如下：

| 按键 | 作用 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 清零运动命令；已按住的方向键释放前保持停止 |
| `F` | 使能；从禁用状态恢复时重置 history 和 `last_action` |
| `I` | 禁用并清零命令 |
| `1` / `2` / `3` | 低速 / 中速 / 高速 |
| `R` | 重置 MuJoCo 仿真 |
| `P` | 打印按键、档位和当前命令 |

普通方向键的按下和松开只改变 command，不重置 history、`last_action` 或策略状态。

### `rl_controller/input/__init__.py`

input 模块导出文件，供 `deploy_mujoco_rl.py` 导入 `KeyboardInput` 和 `KeyboardCommandController`。

## real 模块

`rl_controller/real` 是未来实机运行器依赖的硬件无关层。阶段 0–7 已定义数据契约、配置、离线 Mock 组件、真实硬件后端、完整状态和 actor 历史观测；只有显式创建并打开真实后端时才会访问串口。

### `rl_controller/real/types.py`

定义 `JointState`、`ImuState`、`JointCommand` 和 `RealRobotState`。所有数值数组统一复制为只读 `float32` 快照，错误码使用 `int64`，时间戳使用非负的单调时钟秒；构造时会检查每组关节数组的维度一致性。

### `rl_controller/real/hardware/base.py`

定义 `MotorBackend` 和 `ImuBackend` 协议，以及后端不可用、通信失败等公共异常。上层运行器只依赖这些协议，不接触 Unitree、YIS130 或 MuJoCo 专有类型。

### `rl_controller/real/config.py`

读取并严格校验 `config/real/*.yaml`，主要检查：

- schema 版本、字段缺失和字段拼写；
- 45×6=270 观测契约及 12 维动作；
- 策略关节顺序与电机映射顺序；
- 总线名称和同一总线 Motor ID 的唯一性；
- 方向、减速比、零位、关节限位、目标角步长和 PD 增益；
- 默认角是否位于对应输出轴限位内；
- IMU 四元数顺序、安装旋转和角速度单位；
- 策略、电机、日志频率以及状态和推理超时；
- 真实电机后端与 Mock 策略等危险组合；
- 真实设备路径、ONNX 文件和待填写字段。

该模块只依赖 NumPy 和 PyYAML，可以在未构建 Unitree SDK、未安装 MuJoCo、未连接硬件时导入。

### `rl_controller/real/hardware/mock_motor.py`

实现 `MotorBackend` 协议的确定性关节空间模拟后端。调用 `open()` 后，每次 `cycle()` 根据经过的单调时钟时间，以受限速度从当前关节角跟随 `JointCommand.position`。默认速度上限由每个关节的 `max_target_step_rad * io_frequency_hz` 得到，也可以在测试中显式覆盖。

后端支持：

- 检查命令关节数量和所有命令数组的有限性；
- 将模拟状态约束在输出轴关节限位内；
- `safe_stop()` 锁存停止状态，保持位置并清零速度和力矩；
- 注入过期时间戳、NaN、高温、电机错误码和无效状态；
- `clear_faults()` 清除故障，但不会解除已经触发的 `safe_stop()`。

### `rl_controller/real/hardware/mock_imu.py`

实现 `ImuBackend` 协议，输出机身坐标系下的 `[w, x, y, z]` 四元数和 `rad/s` 角速度。可设置正常状态，也可注入过期时间戳、NaN、无效状态，以及由 roll/pitch/yaw 构造的倾倒姿态。

### `rl_controller/real/hardware/unitree_m8010.py`

封装官方 Unitree Actuator SDK 的 GO-M8010-6 后端。它按配置管理一条或多条 USB-RS485 总线，同一 Motor ID 可以在不同总线上重复，但每个 `(bus, motor_id)` 地址必须唯一。每次 `cycle()` 必须收到全部关节的有效响应才会发布一个 `JointState`；任一串口异常、错误 ID、无效帧、非有限反馈或跨总线完成过慢都会使整轮失败。

配置中的 `zero_offset_rad` 定义为电机输出轴零角对应的机器人关节角，因而统一关系为：

```text
joint_position = direction * motor_output_position + zero_offset_rad
motor_output_position = direction * (joint_position - zero_offset_rad)

rotor_position = motor_output_position * gear_ratio
rotor_velocity = motor_output_velocity * gear_ratio
rotor_kp / rotor_kd = output_kp / output_kd ÷ gear_ratio²
rotor_torque = output_torque ÷ gear_ratio
```

反馈执行逆变换并回到策略关节顺序。后端还提供最近完整周期的 `last_bus_timestamps`、`last_bus_time_skew_s`、`last_state_age_s` 和 `last_bus_errors`，供后续状态聚合与安全层使用。`safe_stop()` 会尝试向所有电机发送 BRAKE，即使中途某个电机通信失败也会继续处理其余地址，并锁存停止状态。

### `rl_controller/real/hardware/yesense_yis130.py`

实现 YESENSE YIS130 的 YIS 私有 UART 协议和 `ImuBackend`：

- 从带噪声或分段到达的字节流中查找 `0x59 0x53` 帧头；
- 校验 TID、payload 长度和 CK1/CK2 双累加校验；
- 按 TLV 读取 `0x20` 角速度和 `0x41` 四元数，拒绝缺失、重复、错误长度或非有限数据；
- 将原始小端有符号整数按 `1e-6` 缩放，并按配置将 `deg/s` 统一为 `rad/s`；
- 将原始四元数统一为 `[w, x, y, z]` 并归一化；
- 坏帧后自动重新查找帧头，报告校验错误数、丢弃字节数、最近 TID、状态年龄和错误原因；
- 在 `state_timeout_s` 内收不到同时包含姿态和角速度的完整帧时关闭本次读取。

标准 YIS UART 输出中 q0 为四元数实部，因此真实模板使用 `quaternion_order: wxyz`；角速度原始单位为 `deg/s`。若设备固件输出配置与此不同，必须以该设备实际协议为准更新 YAML，不能在后端外重复换算。

### `rl_controller/real/orientation.py`

提供实机链路共用的四元数归一化、乘法、共轭和向量旋转。配置中的 `mounting_quaternion_wxyz` 明确定义为“传感器坐标向量旋转到机身坐标”的 `q_body_from_sensor`。后端执行：

```text
q_world_from_body = q_world_from_sensor * conjugate(q_body_from_sensor)
angular_velocity_body = rotate(q_body_from_sensor, angular_velocity_sensor)
```

安装四元数必须通过实物安装方向验证；单位四元数只适用于 IMU 轴与机身轴完全同向的情况。

### `rl_controller/real/state.py`

`RealStateAggregator` 在线程之间汇合 `JointState` 和 `ImuState`，只发布符合以下条件的 `RealRobotState`：

- 电机反馈恰好包含配置要求的全部关节，当前四足接口为 12 个；
- 电机和 IMU 的 `valid` 均为真，数值中没有 NaN/Inf，IMU 四元数范数有效；
- 每条电机总线时间戳、IMU 时间戳都未超过 `safety.max_state_age_s`；
- 所有来源的最大时间差不超过电机与 IMU `state_timeout_s` 中的较小值；
- `JointState.timestamp` 等于本轮所有电机总线中最早的时间戳。

GO-M8010-6 后端接入时，应把同一轮的 `last_bus_timestamps` 一并传入：

```python
joint_state = motor_backend.cycle(command)
state_aggregator.update_joints(
    joint_state,
    source_timestamps=motor_backend.last_bus_timestamps,
)
state_aggregator.update_imu(imu_backend.read())
robot_state = state_aggregator.snapshot()
```

`snapshot()` 在状态不完整时抛出 `StateUnavailableError`，`try_snapshot()` 则返回 `None`。非零电机错误码会随快照保留，温度和错误码是否触发停机属于后续安全管理阶段，不由聚合器提前吞掉。

### `rl_controller/real/observation.py`

`build_real_one_step_observation()` 构造与训练侧完全同序的 45 维当前帧：

```text
[0:3]   command * command_scale
[3:6]   body angular velocity * angular_velocity_scale
[6:9]   projected_gravity
[9:21]  (joint_position - default_angles) * dof_position_scale
[21:33] joint_velocity * dof_velocity_scale
[33:45] previous clipped action
```

`RealObservationHistory` 按 `[当前帧, 前1帧, ..., 前5帧]` 堆叠。状态时间戳必须严格递增；重复、倒退、过期会失败，帧间隔过大时会清空旧历史并从当前帧重新预热。只有连续帧数达到 `runtime.history_warmup_frames` 后 `update()` 才返回输入，否则返回 `None`。当前配置要求至少 6 帧，最终输出为连续 `float32` 的 270 维向量，并按 `policy.clip_observations` 执行与训练侧相同的裁剪。

### `rl_controller/real/policy/mock_policy.py`

实现与现有 `Policy` 协议相同的 `infer(observation)` 结构接口。它检查输入维度和有限性，并返回可配置、确定性的 `float32` 动作副本；不加载 TorchScript、ONNX 或模型文件。动作 NaN 注入用于后续安全层测试。

### `rl_controller/real/policy/base.py`

定义实机策略后端的最小协议和统一异常层级。后端必须公开输入维度、动作维度、后端名称以及 `infer()`；`require_float32_vector()` 对 shape、`float32` dtype 和 NaN/Inf 做严格检查，并返回隔离的连续数组。任何策略契约或推理错误都会抛出异常，不允许上层继续使用无效或上一次的动作。

### `rl_controller/real/policy/onnx_policy.py`

实现实机使用的 CPU ONNX Runtime actor 后端。构造时要求模型恰好具有一个输入和一个输出，两者都必须是二维 `tensor(float)`；batch 维可以为 `1` 或动态值，特征维必须分别等于 YAML 中配置的 `num_observations` 和 `num_actions`。每次推理再次检查输入和运行时输出的 shape、dtype 与有限性。

该模块顶层不会导入 ONNX Runtime。只有 `policy.backend: onnx` 并实际构造 `OnnxPolicy` 时才加载依赖和模型，因此没有 ONNX 文件或运行库不会阻塞 Mock 链路。

### `rl_controller/real/policy/executor.py`

`PolicyExecutor` 是 Mock 和 ONNX 后端共用的安全执行边界。它在调用后端前检查并按 `clip_observations` 裁剪观测，在后端返回后检查并按 `clip_actions` 裁剪动作，同时记录 `last_inference_time_s`。推理超过 `safety.max_policy_inference_time_s` 时，本次动作立即作废并抛出 `PolicyInferenceTimeoutError`。

当前超时检查是同步调用返回后的结果验收，不会强制中断正在执行的 ONNX Runtime 调用。方案阶段 9 的策略线程负责周期调度，阶段 10 的安全管理负责接收该异常并触发锁存停机。

### `rl_controller/real/policy/factory.py`

`load_real_policy()` 根据 `RealDeploymentConfig.policy.backend` 创建 Mock 或 ONNX 后端，并统一包装为 `PolicyExecutor`。未来 runner 只依赖该工厂和执行器，不需要感知 ONNX Runtime session 或 Mock 实现细节。

三个 Mock 组件当前用于单元级和手工单周期集成验证，不代表完整实机运行器。线程、状态机和安全管理仍按方案后续阶段实现。

## third_party 目录

### `third_party/unitree_actuator_sdk/`

Unitree 官方 `unitree_actuator_sdk` 的固定源码快照，版本和上游提交号记录在 `third_party/unitree_actuator_sdk.version`。第三方源码保持原样，关节映射、机械零位、方向、输出轴换算和安全检查将在自己的硬件后端中实现。

无硬件构建当前主机对应的 Python 绑定和 GO-M8010-6 官方示例：

```bash
bash src/controller/rl/third_party/build_unitree_actuator_sdk.sh
```

脚本只编译并导入模块，不打开 RS485 串口，也不会发送电机命令。当前 Python ABI 的生成模块和 `build/` 目录均被 Git 忽略。

## test 目录

### `test/deploy_mujoco_rl_debug.py`

MuJoCo RL 部署调试脚本。

默认参数：

```text
--no-keyboard
--command 0.0 0.0 0.0
--duration 5
--debug-steps 100
```

它不会改变部署逻辑，只是调用 `deploy_mujoco_rl.py` 的正式入口并打开调试输出。适合排查以下问题：

- Isaac Gym play 可以走，但 MuJoCo 中不稳定或不动。
- 初始姿态后很快触发摔倒检测。
- policy 原始输出过大，或者裁剪后的 action 很快接近 `clip_actions` 上限。
- PD 力矩长期顶到 actuator 限幅。
- 关节角、目标角和训练侧默认角不一致。

调试输出包含：

```text
base_z
projected_gravity
command
joint_pos
raw_action
clipped_action
torques
target_joint_pos
```

其中 `raw_action` 是当前推理后端的原始输出，`clipped_action` 是部署侧按 YAML 中 `clip_actions` 裁剪后实际送入 PD 的动作。当前 Go2 为 `clip_actions: 100.0`，正常情况下二者应基本一致；如果训练侧修改了 `normalization.clip_actions`，部署侧也要同步。

### `test/deploy_mujoco_rl_keyboard_debug.py`

带 Linux 本地键盘输入的 MuJoCo RL 调试脚本。

默认参数：

```text
--duration 30
--debug-steps 200
```

脚本默认自动识别并监听当前主机上的所有合格键盘，适合观察按键命令、策略输入和动作是否连续变化。必要时可传入 `--keyboard-device /dev/input/by-path/...-event-kbd` 限制为单个设备。

### `test/test_keyboard_controller.py`

不访问实际键盘和 MuJoCo 的命令映射单元测试，覆盖初始禁用、使能、组合键、反向抵消、三档速度、Space 停止锁存、禁用清零和仿真重置事件。

### `test/test_real_interfaces.py`

不访问设备的实机公共接口测试，覆盖状态数组的类型、维度、只读快照、时间戳和硬件协议结构。

### `test/test_mock_backends.py`

Mock 组件测试，覆盖电机速度限幅、生命周期、`safe_stop`、IMU 状态、策略输入输出以及各类故障注入。测试还手工执行一次 `270 维输入 -> 12 维动作 -> JointCommand -> 电机/IMU反馈 -> RealRobotState` 周期，确认不连接设备、不构建 Unitree SDK、不提供 ONNX 也能完成离线链路。

### `test/test_real_config.py`

实机配置解析测试，覆盖完整 Mock 配置、真实模板待填写错误、schema 和观测维度、重复 Motor ID、非法方向、未知总线、越界默认角、危险后端组合，以及配置模块不导入 MuJoCo 或 Unitree SDK。

### `test/test_unitree_m8010_backend.py`

完全使用伪 SDK 验证 GO-M8010-6 后端，不打开实际串口。覆盖关节侧与转子侧位置、速度、增益和力矩换算，多总线及重复 ID、完整反馈时间戳、状态年龄、无响应/坏帧/错误 ID/NaN/串口异常、跨总线超时、减速比核对以及全电机 BRAKE。

### `test/test_yesense_yis130_backend.py`

使用内存串口和 YIS 协议样例帧验证 YIS130 后端，不打开设备。覆盖分段帧、流重同步、CK1/CK2、TLV 数据、四元数顺序、角速度单位、安装旋转、缺失数据、串口异常和状态超时。

### `test/test_real_state_observation.py`

验证完整 12 关节门槛、多总线与 IMU 时间同步、状态新鲜度、无效/非有限状态拒绝、45 维各切片顺序、重力投影、6 帧连续预热、最新帧优先、观测裁剪和时间中断后重置。最后使用 Mock 电机与 IMU 连续生成 6 帧状态，确认能够得到稳定的 270 维输入并通过 Mock actor 接口。

### `test/test_real_policy.py`

使用内存伪 ONNX Runtime 验证实机策略层，不需要真实 ONNX 文件或运行库。覆盖模型输入输出数量、名称、rank、dtype、动态 batch、270/12 特征维，运行时输入输出契约、NaN/Inf、观测与动作裁剪、后端异常、推理超时、Mock/ONNX 工厂选择和 Mock 链路的可选依赖隔离。

### `test/compare_policy_backends.py`

TorchScript / ONNX 离线输出一致性检查脚本。它向两个模型输入完全相同的 `float32` 历史观测，打印平均绝对误差、最大绝对误差和 `allclose` 结果；比较失败时退出码为 1。

默认随机生成 128 组 `num_obs` 维输入，也可以用 `--observations observations.npy` 输入真实 MuJoCo 观测。该脚本只检查模型推理，不运行 MuJoCo，也不改变部署配置。

常用命令：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_keyboard_debug.py
```

覆盖键盘设备或打印步数：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_keyboard_debug.py \
  --keyboard-device /dev/input/by-path/<设备名>-event-kbd \
  --debug-steps 300
```

运行命令映射测试：

```bash
python3 -m unittest src/controller/rl/test/test_keyboard_controller.py -v
```

## 运行方式

先安装部署依赖：

```bash
pip install -e src/controller/rl
```

CPU 版安装会包含 `onnxruntime`。如果只补装依赖：

```bash
pip install mujoco torch numpy pyyaml pyserial evdev "onnxruntime>=1.16,<2"
```

再把训练直接导出的两个策略放到：

```text
src/controller/rl/models/policy_1.pt
src/controller/rl/models/policy_1.onnx
```

从 `quadruped` 仓库根目录运行：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py --headless
```

打开 MuJoCo viewer：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py
```

指定策略和模型：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --policy-path src/controller/rl/models/policy_1.pt \
  --xml-path src/controller/rl/resources/robots/go2/scene.xml \
  --device cpu
```

运行 ONNX Runtime 后端：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --policy-path src/controller/rl/models/policy_1.onnx \
  --device cpu
```

默认 `--policy-backend auto`：`.pt` 选择 TorchScript，`.onnx` 选择 ONNX Runtime。需要排查模型命名问题时可显式传入 `--policy-backend torchscript` 或 `--policy-backend onnx`。

指定速度命令：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --command 0.5 0.0 0.0 \
  --duration 30
```

默认自动查找并监听所有合格的本地 Linux 键盘。限制为单个键盘设备：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --keyboard-device /dev/input/by-path/<设备名>-event-kbd
```

控制默认禁用。按 `F` 后程序独占已监听的键盘，MuJoCo Viewer 不再收到控制按键；按 `I` 后清零命令并释放键盘。独占期间需要先按 `I`，再在终端使用 `Ctrl+C`。

打印前 100 次 policy 推理的调试信息：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --command 0.0 0.0 0.0 \
  --duration 5 \
  --debug-steps 100
```

使用调试脚本：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_debug.py
```

使用键盘调试脚本：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_keyboard_debug.py
```

比较两个模型的离线动作输出：

```bash
python3 src/controller/rl/test/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx
```

用保存的真实观测比较：

```bash
python3 src/controller/rl/test/compare_policy_backends.py \
  --observations observations.npy
```

完成离线输出比较后，应使用相同 YAML、初始状态和速度命令分别运行 `.pt` 与 `.onnx`，确认 MuJoCo 闭环行为一致。未来实机链路使用 Python + ONNX Runtime，并通过工程内 Unitree Actuator SDK 直接控制 GO-M8010-6，不使用 Go2 SDK2。

## 和训练侧的同步关系

训练侧改动后，需要同步部署侧的情况：

- 改了 `num_one_step_obs`、`history_length` 或 `num_obs`。
- 改了 45 维观测顺序。
- 改了 action 维度或关节顺序。
- 改了 `default_angles`。
- 改了 `action_scale`。
- 改了 `normalization.clip_actions`。
- 改了 PD 参数。
- 改了命令缩放或观测缩放。
- 改了导出策略的外部输入格式。
- 改了 estimator 或 actor 的部署结构，必须重新直接导出 `.pt` 和 `.onnx`，不能由其中一种格式转换另一种。

训练侧只改 critic privileged obs 内容、reward 权重、噪声、域随机化时，通常不需要改部署推理逻辑；但配置说明应保持一致。

## 摔倒处理

部署主循环会根据机身高度和重力方向投影判断是否摔倒。当前 Go2 配置中 `fall_height_threshold: 0.0`，表示不再因为机身高度过低直接停力矩；重力方向检查仍保留。判定摔倒后：

- policy 不再输出有效 action。
- 当前力矩清零。
- history buffer 重置。
- MuJoCo 仿真继续运行。
- 键盘按 `R` 可以重置仿真。

## 常用检查

语法检查：

```bash
python3 -m py_compile src/controller/rl/rl_controller/deploy_mujoco_rl.py
python3 -m py_compile src/controller/rl/rl_controller/core/config.py
python3 -m py_compile src/controller/rl/rl_controller/core/observer.py
python3 -m py_compile src/controller/rl/rl_controller/core/pd.py
python3 -m py_compile src/controller/rl/rl_controller/core/policy.py
python3 -m py_compile src/controller/rl/rl_controller/input/keyboard_input.py
python3 -m py_compile src/controller/rl/rl_controller/input/keyboard_controller.py
python3 -m py_compile src/controller/rl/rl_controller/real/config.py
python3 -m compileall -q src/controller/rl/rl_controller/real
PYTHONPATH=src/controller/rl python3 -m unittest discover \
  -s src/controller/rl/test -p 'test_*.py' -v
```

配置读取检查：

```bash
PYTHONPATH=src/controller/rl python3 - <<'PY'
from rl_controller.core.config import RobotRLConfig
cfg = RobotRLConfig.from_yaml('src/controller/rl/config/go2.yaml')
print(cfg.robot_name, cfg.num_obs, cfg.num_one_step_privileged_obs, cfg.num_privileged_obs)
PY
```

期望输出：

```text
go2 270 238 238
```

实机 Mock 配置读取检查：

```bash
PYTHONPATH=src/controller/rl python3 - <<'PY'
from rl_controller.real import RealDeploymentConfig
cfg = RealDeploymentConfig.from_yaml('src/controller/rl/config/real/mock.yaml')
print(cfg.robot.name, cfg.robot.num_obs, cfg.policy.backend, cfg.motors.backend)
PY
```

期望输出：

```text
mock_quadruped 270 mock mock
```

`custom_quadruped.example.yaml` 保留了未标定的 `null`，因此当前读取失败属于预期行为。
