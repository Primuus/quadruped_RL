# 四足机器人 RL 部署工程

`quadruped` 负责四足机器人强化学习策略的本地仿真和实机部署，不在本仓库训练策略。

运行命令均从本仓库根目录执行。机器人资产、策略文件和运行配置必须配套，仿真部署与实机部署使用不同的状态输入和执行后端。

## 文档导航

| 需求 | 入口 |
| --- | --- |
| 理解部署包每个文件负责什么 | [RL 子工程说明](src/controller/rl/README.md) |
| 训练、日志和调参文档的阅读顺序 | [doc 导航](doc/README.md) |
| 策略文件、版本和输入输出契约 | [模型目录](src/controller/rl/models/README.md) |
| Primuus 机械模型与碰撞定义 | [Primuus 模型说明](src/controller/rl/resources/robots/primuus/README.md) |
| Primuus 跨引擎关节摩擦与阻尼 | [关节动力学](doc/关节动力学.md) |
| 官方电机 SDK 的固定版本与构建 | [第三方依赖说明](src/controller/rl/third_party/README.md) |

相关工程：

- RL 训练：`quadruped_train`，负责 Isaac Gym、HIM-PPO、策略训练及 TorchScript/ONNX 导出；
- 传统 PD：[quadruped_ros2](https://github.com/Primuus/quadruped_ros2)，负责 ROS2、Gazebo、FSM 和传统关节 PD 控制；
- RL 部署：本仓库，负责 MuJoCo、策略推理、电脑键盘命令和实机控制链路。

三个工程独立安装、独立运行，不互相导入源码。训练侧和部署侧只通过机器人参数、观测/动作契约及导出的模型文件衔接。

## 功能与边界

- 加载 TorchScript 或 ONNX 策略并在 MuJoCo 中运行；
- 构造与训练侧一致的 45 维单帧观测和 6 帧历史输入；
- 将 12 维策略动作转换为目标关节角和仿真 PD 力矩；
- 使用 Linux evdev 读取运行主机的实体键盘；
- 通过 `--debug-steps` 输出 MuJoCo 状态、动作、目标角和力矩；
- 比较同一策略的 TorchScript 与 ONNX 输出；
- 解析严格的通用四足实机配置；
- 接入 Unitree GO-M8010-6 电机和 Yesense YIS130 IMU；
- 提供状态同步、ONNX 推理、输出安全门、watchdog、FAULT 锁存和显式停机/故障报告。

Go2 与 Primuus 共用 MuJoCo 运行器，分别使用独立 YAML、物理模型和策略。公共观测、策略、状态机和安全接口不绑定 Go2；新增四足机器人时，应增加实例配置，不复制控制主循环。

| 机器人 | 训练入口 / 实验目录 | 本地配置 / 默认策略 |
| --- | --- | --- |
| Go2 | `train_go2.py` / `logs/go2` | `config/go2.yaml` / `models/go2/policy_1.pt` |
| Primuus | `train_primuus.py` / `logs/primuus` | `config/primuus.yaml` / `models/primuus/policy.pt` |

通用接口当前针对12驱动关节四足：单帧45维、6帧历史、12维动作。通用化不代表任意关节数、任意驱动协议均能不改代码直接接入。

真实电机参数、机械零位、关节方向、限位、增益、温度阈值和急停系统必须根据实物逐项确认，不能直接把仿真配置当成实机标定值。

Python watchdog不能强制中断底层SDK或ONNX Runtime中的阻塞调用。线程永久阻塞时进程也可能无法自行退出；物理急停和驱动器自身超时不可省略。

## 数据流

```text
quadruped_train
  -> checkpoint
  -> Isaac Gym play
  -> policy_1.pt / policy_1.onnx

quadruped
  -> MuJoCo 或真实硬件状态
  -> 45维单帧观测
  -> 6帧历史，形成270维部署模型输入
  -> estimator估计速度3维与latent16维
  -> actor读取当前45维与估计结果，共64维
  -> 12维action
  -> default_angles + action_scale * action
  -> MuJoCo PD力矩或实机目标关节角
```

训练侧 critic 的 privileged observations 只用于训练，不进入部署 actor。

Primuus 的仿真力矩另分为限幅后的电机输出与被动关节损耗。后者由两端相同的 `joint_dynamics.json` 和算法定义，不是额外电机命令；实机不重复施加虚拟摩擦。

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

本工程只支持在完整 Git checkout 中进行 editable 安装。请在独立 Python 环境中，从仓库根目录执行：

```bash
python3 -m pip install -e src/controller/rl
```

不要使用 `pip install src/controller/rl`、`pip install .` 或构建出的 wheel 作为部署产物。运行链路会直接使用源码树中的 `config/`、`models/`、`resources/` 和 `third_party/`；这些目录包含需要现场检查、替换或按本机 Python ABI 编译的资产，不属于普通 Python wheel。运行命令也应在该 checkout 内执行。

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

训练侧 `train` 新建时间戳目录保存 checkpoint；`play` 加载 checkpoint 后导出部署模型。按机器人分别放置：

```text
src/controller/rl/models/go2/policy_1.pt
src/controller/rl/models/go2/policy_1.onnx
src/controller/rl/models/primuus/policy.pt
src/controller/rl/models/primuus/policy.onnx
```

从训练服务器覆盖 Primuus 双模型，复制时将固定导出名改为部署名：

```bash
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.pt \
  src/controller/rl/models/primuus/policy.pt
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.onnx \
  src/controller/rl/models/primuus/policy.onnx
```

模型文件默认不提交到 Git。策略必须与部署配置中的关节顺序、默认角、动作缩放、观测缩放、裁剪范围和历史顺序完全一致。

实机软件链路是 Python + ONNX Runtime。critic不导出，estimator与actor一起导出。重复play覆盖 `policy_1.pt / policy_1.onnx`，不会递增为 `policy_2`，也不覆盖checkpoint；服务器导出不会自动替换本地模型，只有复制到上述目标路径才会覆盖。双文件必须来自同一次成功导出，详情见[模型目录说明](src/controller/rl/models/README.md)。

比较 TorchScript 与 ONNX 输出：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --config src/controller/rl/config/primuus.yaml \
  --torchscript-path src/controller/rl/models/primuus/policy.pt \
  --onnx-path src/controller/rl/models/primuus/policy.onnx
```

## MuJoCo 运行

使用 Primuus 策略进行闭环仿真：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --config src/controller/rl/config/primuus.yaml
```

Primuus 使用自身平地场景、默认角和 `60 / 1.5` PD。训练物理步长2 ms、部署0.5 ms，两侧策略周期均为20 ms；统一关节损耗只在Python运行器中生效，不改变电机限矩或地面接触模型。

不指定配置时默认使用 Go2：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py
```

`config/go2.yaml` 选择包含 Go2 模型的 `ROBOCON2027.xml`。临时切换到平地对照场景：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --xml-path src/controller/rl/resources/robots/go2/scene_flat.xml
```

场景路径可以通过 YAML 的 `xml_path` 或 `--xml-path` 指定。`ROBOCON2027.xml` 使用平面和静态障碍物构建场地，不依赖外部高度图；场地欧拉角单位为弧度，与 `go2.xml` 一致。切换仿真场景不会改变训练侧地形，也不代表现有策略已经具备越障能力。

Go2当前物理步长为2 ms、每10步更新一次策略，策略频率50 Hz。默认高度跌倒检查关闭，但重力投影倾倒检查仍启用；不要把它理解成删除了全部输出保护。

无界面运行：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py --headless
```

指定 ONNX：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --policy-path src/controller/rl/models/go2/policy_1.onnx \
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

`--command` 必须与 `--no-keyboard` 同用。默认键盘模式启动时禁用控制，需要按F使能；固定命令模式不读取键盘，也不需要F。

输出前 100 次策略周期的调试快照：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --no-keyboard \
  --duration 5 \
  --debug-steps 100
```

debug是正式运行器的可选输出，不是另一套部署链路。它区分模型原始 `raw_action`、裁剪后的 `clipped_action`、目标关节角和最终力矩；默认不打印快照。详细字段和排查顺序见RL子工程说明。

### 只查看机械模型

不加载策略、不运行PD，只查看Primuus：

```bash
python3 -m mujoco.viewer --mjcf=src/controller/rl/resources/robots/primuus/scene_flat.xml
```

这个原生Viewer不读取机器人YAML、策略、统一关节损耗或evdev控制器，播放后无控制机器人会倒地。它使用XML原生关节损耗，快捷键与本工程RL入口的WASD控制不同。检查策略运动请使用上面的Primuus RL入口，不能拿Go2权重运行Primuus YAML。

Primuus小腿机械范围为 `[-2.7227, -0.8378] rad`，通过固定碰撞载体和凸分段处理接触，不以收紧关节范围代替碰撞约束。几何职责及数值接触边界见[模型说明](src/controller/rl/resources/robots/primuus/README.md)。

## 键盘控制

程序默认读取运行主机的 Linux evdev 键盘。SSH 客户端按键不会自动转发为远端 evdev 事件。

按住方向键持续给命令，释放方向键后相应命令归零；F使能时独占已选键盘，I禁用时释放。需要本机evdev读取权限，不能用“先sudo启动整个控制器”代替正确设备权限。

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

三档分别使用速度命令上限的25%、50%、100%，默认中档。`command_limits`限制各轴，
`max_translation_speed`限制XY合成速度，数值须对应训练的 `commands.max_translation_speed`。
Go2和Primuus当前均为1 m/s；键盘各档的XY模长上限分别为0.25、0.5、1 m/s。
例如高速同时按W+A时输出约 `[0.707, 0.707, 0]`，中档约 `[0.354, 0.354, 0]`；
平移方向保持不变，yaw独立限幅。单轴命令仍为0.25、0.5、1 m/s。
YAML初始命令、`--command`和实机速度接口也执行同一合成限幅，但固定命令不受键盘档位缩放。
这些值是策略目标速度，不改变仿真时间倍率，也不保证实际速度无跟踪误差。

指定键盘设备：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --keyboard-device '/dev/input/by-path/<device>-event-kbd'
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
  --config 'src/controller/rl/config/real/<robot>.yaml' \
  --allow-real-hardware
```

不使用键盘、自动进入站姿过渡并保持固定速度命令时：

```bash
python3 src/controller/rl/rl_controller/deploy_real_rl.py \
  --config 'src/controller/rl/config/real/<robot>.yaml' \
  --allow-real-hardware \
  --no-keyboard --auto-enable --command 0.2 0.0 0.0
```

固定命令在 `enable_rl()` 清空旧命令之后写入，站姿过渡期间不参与观测，进入 RL 后持续生效。

`--allow-real-hardware` 是必须显式提供的启动许可。它不是安全认证，也不能替代物理急停、驱动器底层超时、吊架测试和单电机验证。

本工程不再提供mock硬件、dry-run实机后端或独立的无实物测试链路。保留的正式诊断入口是MuJoCo debug和双模型输出对比；硬件到位后的检查由用户逐级执行。

## 辅助文档

- [TensorBoard 使用指南](doc/TensorBoard使用指南.md)
- [训练调参教程](doc/训练调参教程.md)

## 故障行为

运行时异常统一锁存为 FAULT：

```text
FAULT -> 禁止新命令 -> 电机线程 safe_stop -> 等待线程退出 -> 关闭设备 -> 非零退出码
```

终端会输出故障类别、来源、状态、状态年龄、速度命令、原始策略动作、目标关节角和周期耗时。第一个故障作为根因保留，停机阶段错误单独列出。FAULT 后不会自动恢复策略输出；排除原因后必须重新启动并从 INIT 初始化。

电机工作线程是运行期电机 I/O 的唯一所有者。若底层 SDK 调用超过关闭时限，程序会明确报告仍存活的线程，并保留对应设备不做并发关闭；这种情况必须依靠驱动器通信超时和物理急停，不能依赖 Python 强制终止原生调用。

正常的 `I`、`Ctrl+C` 或运行时长结束不记为 FAULT，但同样执行 `safe_stop()` 和资源关闭。
