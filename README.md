# 四足机器人 RL 部署工程

`quadruped` 负责四足机器人强化学习策略的本地仿真和实机部署，不在本仓库训练策略。

相关工程：

- RL 训练：`quadruped_train`，负责 Isaac Gym、HIM-PPO、策略训练及 TorchScript/ONNX 导出；
- 传统 PD：[quadruped_ros2](https://github.com/Primuus/quadruped_ros2)，负责 ROS2、Gazebo、FSM 和传统关节 PD 控制；
- RL 部署：本仓库，负责 MuJoCo、策略推理、电脑键盘命令和实机控制链路。

三个工程独立安装、独立运行，不互相导入源码。训练侧和部署侧只通过机器人参数、观测/动作契约及导出的模型文件衔接。

## 当前能力

- 加载 TorchScript 或 ONNX 策略并在 MuJoCo 中运行；
- 构造与训练侧一致的 45 维单帧观测和 6 帧历史输入；
- 将 12 维策略动作转换为目标关节角和仿真 PD 力矩；
- 使用 Linux evdev 读取运行主机的实体键盘；
- 通过 `--debug-steps` 输出 MuJoCo 状态、动作、目标角和力矩；
- 比较同一策略的 TorchScript 与 ONNX 输出；
- 解析严格的通用四足实机配置；
- 接入 Unitree GO-M8010-6 电机和 Yesense YIS130 IMU；
- 提供状态同步、ONNX 推理、输出安全门、watchdog、FAULT 锁存和确定性安全停机。

当前已有 Go2 的 MuJoCo 示例。公共观测、策略、状态机和安全接口不绑定 Go2；新增四足机器人时，应增加独立配置、物理模型和匹配策略，而不是复制控制主循环。

实机链路尚未经过真实机器人验证。真实电机参数、机械零位、关节方向、限位、增益、温度阈值和急停系统必须在有实物后逐项确认。

## 数据流

```text
quadruped_train
  -> checkpoint
  -> Isaac Gym play
  -> policy_1.pt / policy_1.onnx

quadruped
  -> MuJoCo 或真实硬件状态
  -> 45维单帧观测
  -> 6帧历史，形成270维actor输入
  -> TorchScript / ONNX policy
  -> 12维action
  -> default_angles + action_scale * action
  -> MuJoCo PD力矩或实机目标关节角
```

训练侧 critic 的 privileged observations 只用于训练，不进入部署 actor。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/controller/rl/rl_controller/core` | MuJoCo 配置、观测、策略后端和 PD 计算 |
| `src/controller/rl/rl_controller/input` | Linux 键盘读取与速度命令状态机 |
| `src/controller/rl/rl_controller/real` | 实机配置、状态、观测、硬件、策略、安全和运行器 |
| `src/controller/rl/config` | MuJoCo 配置和实机配置模板 |
| `src/controller/rl/models` | 本地策略文件目录 |
| `src/controller/rl/resources` | MuJoCo XML、URDF 和网格资源 |
| `src/controller/rl/tools` | 模型导出/发布工具 |
| `src/controller/rl/third_party` | 固定版本 Unitree Actuator SDK |
| `doc` | TensorBoard 使用和训练调参文档 |

每个文件的详细职责见 [RL 子工程说明](src/controller/rl/README.md)。

## 安装

建议在独立 Python 环境中，从仓库根目录安装：

```bash
python3 -m pip install -e src/controller/rl
```

主要依赖：

```text
numpy
pyyaml
pyserial
evdev
torch
mujoco
onnxruntime
```

RL 链路不要求 ROS2、ament 或 `colcon build`。

## 策略文件

训练侧执行 `play` 后，将同一 checkpoint 导出的模型复制到：

```text
src/controller/rl/models/policy_1.pt
src/controller/rl/models/policy_1.onnx
```

从训练服务器复制示例：

```bash
scp hurricane@192.168.0.214:~/quadruped_train/logs/go2/exported/policies/policy_1.{pt,onnx} \
  /home/ry/project/quadruped/src/controller/rl/models/
```

模型文件默认不提交到 Git。策略必须与部署配置中的关节顺序、默认角、动作缩放、观测缩放、裁剪范围和历史顺序完全一致。

比较 TorchScript 与 ONNX 输出：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx
```

## MuJoCo 运行

打开 Viewer：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py
```

无界面运行：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py --headless
```

指定 ONNX：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --policy-path src/controller/rl/models/policy_1.onnx \
  --policy-backend onnx \
  --device cpu
```

使用固定命令：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --command 0.5 0.0 0.0 \
  --duration 30
```

输出前 100 次策略周期的调试快照：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --duration 5 \
  --debug-steps 100
```

## 键盘控制

程序默认读取运行主机的 Linux evdev 键盘。SSH 客户端按键不会自动转发为远端 evdev 事件。

| 按键 | 作用 |
| --- | --- |
| `F` | 使能控制并重置策略历史 |
| `I` | MuJoCo 中禁用控制；实机中请求正常安全停机 |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左移 / 右移 |
| `Q` / `E` | 左转 / 右转 |
| `Space` | 清零运动命令 |
| `1` / `2` / `3` | 低速 / 中速 / 高速 |
| `R` | 重置 MuJoCo 仿真 |
| `P` | 打印控制状态 |

指定键盘设备：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --keyboard-device /dev/input/by-path/<device>-event-kbd
```

## 实机部署入口

实机配置模板：

```text
src/controller/rl/config/real/custom_quadruped.example.yaml
```

模板故意包含 `null`，不能直接启动。准备真实配置时必须填写：

- 策略路径及训练契约；
- 机器人关节顺序和默认角；
- 每个电机的总线、ID、方向、零位、减速比、限位和 PD；
- YIS130 串口、安装旋转和单位；
- 状态超时、温度、倾角、推理时限和站姿过渡参数；
- 与训练命令范围一致的键盘速度上限。

构建固定版本的 Unitree SDK Python 模块：

```bash
bash src/controller/rl/third_party/build_unitree_actuator_sdk.sh
```

完成配置及硬件逐级检查后，实机入口格式为：

```bash
python3 src/controller/rl/rl_controller/deploy_real_rl.py \
  --config src/controller/rl/config/real/<robot>.yaml \
  --allow-real-hardware
```

`--allow-real-hardware` 是必须显式提供的启动许可。它不是安全认证，也不能替代物理急停、驱动器底层超时、吊架测试和单电机验证。

## 辅助文档

- [TensorBoard 使用指南](doc/TensorBoard使用指南.md)
- [训练调参教程](doc/训练调参教程.md)

## 故障行为

运行时异常统一锁存为 FAULT：

```text
FAULT -> safe_stop -> 停止线程 -> 关闭IMU/电机/键盘 -> 非零退出码
```

终端会输出故障类别、来源、状态、状态年龄、速度命令、原始策略动作、目标关节角和周期耗时。第一个故障作为根因保留，停机阶段错误单独列出。FAULT 后不会自动恢复策略输出；排除原因后必须重新启动并从 INIT 初始化。

正常的 `I`、`Ctrl+C` 或运行时长结束不记为 FAULT，但同样执行 `safe_stop()` 和资源关闭。
