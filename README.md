# 四足机器人 RL 部署工程

`quadruped` 只负责四足机器人强化学习策略的本地仿真和部署，不包含传统 PD ROS2 控制链路，也不在本仓库训练策略。

相关工程：

- RL 训练：`quadruped_train`，负责 Isaac Gym、HIM-PPO、策略训练与导出。
- 传统 PD：[quadruped_ros2](https://github.com/Primuus/quadruped_ros2)，负责 ROS2、Gazebo、FSM 和传统关节 PD 控制。
- RL 部署：本仓库，负责 MuJoCo、TorchScript/ONNX 推理、键盘命令和未来实机链路。

三个工程独立安装、独立运行，不互相导入源码。它们只通过机器人配置、观测/动作约定和导出的策略文件衔接。

## 当前能力

- 读取 TorchScript 或 ONNX 策略并自动选择推理后端。
- 在 MuJoCo 中构造与训练侧一致的 45 维单帧观测和 6 帧历史输入。
- 将策略动作转换为目标关节角和 PD 力矩。
- 使用 Linux evdev 读取运行主机的实体键盘命令。
- 保留正式 debug 输出和 TorchScript/ONNX 双模型输出对比工具。
- 提供通用实机状态、配置、Mock 后端、GO-M8010-6 电机后端和 YIS130 IMU 后端。

当前接入 MuJoCo 的机器人实例是 Go2。框架按通用四足机器人接口组织；新增机器人需要增加独立 YAML、MuJoCo 资源和匹配策略，不应复制部署主循环。

## 数据流

```text
quadruped_train
  -> 导出 policy_1.pt / policy_1.onnx
  -> 复制到本仓库 models/
  -> MuJoCo 或未来实机状态
  -> 45 维单帧观测
  -> 6 帧历史，形成 270 维 actor 输入
  -> TorchScript / ONNX policy
  -> 12 维 action
  -> 目标关节角
  -> PD 力矩
  -> MuJoCo actuator 或实机电机后端
```

训练侧 critic 的 privileged observations 只用于训练，不进入部署 actor。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/controller/rl/rl_controller/core` | 配置、观测、策略后端和 PD 计算 |
| `src/controller/rl/rl_controller/input` | Linux 本地键盘读取与命令状态机 |
| `src/controller/rl/rl_controller/real` | 实机配置、状态类型、观测聚合和硬件后端 |
| `src/controller/rl/config` | MuJoCo 和实机部署 YAML |
| `src/controller/rl/models` | 本地策略文件目录 |
| `src/controller/rl/resources` | MuJoCo XML、URDF 和网格资源 |
| `src/controller/rl/third_party` | 固定版本 Unitree Actuator SDK |
| `src/controller/rl/test` | 单元测试、MuJoCo debug 和双模型验证 |

每个文件的详细职责见 [RL 子工程说明](src/controller/rl/README.md)。

## 安装

建议在独立 Python 环境中，从仓库根目录安装：

```bash
python3 -m pip install -e src/controller/rl
```

主要依赖为：

```text
mujoco
torch
numpy
pyyaml
pyserial
evdev
onnxruntime
```

RL 链路不要求 ROS2、ament 或 `colcon build`。

## 策略文件

训练侧执行 `play` 导出后，将策略复制到：

```text
src/controller/rl/models/policy_1.pt
src/controller/rl/models/policy_1.onnx
```

从训练服务器复制：

```bash
scp hurricane@192.168.0.214:~/quadruped_train/logs/go2/exported/policies/policy_1.{pt,onnx} \
  /home/ry/project/quadruped/src/controller/rl/models/
```

模型文件默认不提交到 Git。TorchScript 和 ONNX 必须来自同一个训练 checkpoint，并与部署 YAML 中的观测维度、关节顺序、默认角、动作缩放和裁剪保持一致。

## MuJoCo 运行

打开 Viewer：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py
```

无界面运行：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py --headless
```

指定 ONNX 策略：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --policy-path src/controller/rl/models/policy_1.onnx \
  --policy-backend onnx \
  --device cpu \
  --headless
```

固定速度命令并禁用键盘：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --command 0.5 0.0 0.0 \
  --duration 30
```

打印前 100 次控制周期：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --duration 5 \
  --debug-steps 100
```

## 键盘控制

默认自动寻找运行主机上的 Linux 键盘。SSH 客户端按键不会自动转发成远端 evdev 事件。

| 按键 | 作用 |
| --- | --- |
| `F` | 使能控制并重置策略历史 |
| `I` | 禁用控制、清零命令并释放键盘 |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 清零运动命令 |
| `1` / `2` / `3` | 低速 / 中速 / 高速 |
| `R` | 重置 MuJoCo |
| `P` | 打印控制状态 |

指定设备时优先使用稳定路径：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --keyboard-device /dev/input/by-path/<设备名>-event-kbd
```

## 调试与验证

MuJoCo 正式 debug：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_debug.py
```

带键盘的 debug：

```bash
python3 src/controller/rl/test/deploy_mujoco_rl_keyboard_debug.py
```

比较 TorchScript 和 ONNX 的离线输出：

```bash
python3 src/controller/rl/test/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx
```

运行全部不接硬件的单元测试：

```bash
PYTHONPATH=src/controller/rl python3 -m unittest discover \
  -s src/controller/rl/test -p 'test_*.py' -v
```

## 实机状态

实机侧已经具备配置格式、公共状态类型、Mock 组件、GO-M8010-6 电机通信、YIS130 IMU 解析、状态同步和 45x6 观测历史。当前没有可直接驱动实物的完整 runner；在安全状态机、控制周期、急停和生命周期完成前，现有真实硬件后端不能视为可上机程序。

## 补充文档

- [训练调参教程](训练调参教程.md)
- [TensorBoard 使用指南](TensorBoard使用指南.md)
