# Primuus 仿真资源

本目录仅描述当前 Primuus 的简化关节力矩模型，用于 sim2sim 准备，不模拟齿轮和连杆传动。

## 文件职责

| 文件 / 目录 | 作用 |
| --- | --- |
| `urdf/primuus.urdf` | 机械结构来源：17 个 link、12 个转动关节、4 个固定足端关节及质量惯量 |
| `stl/` | 原始显示网格；机身、髋、小腿和足端同时将对应网格用于碰撞 |
| `collision/` | 大腿的电机和板件分离碰撞网格，避免整个大腿凸包填满装配间隙 |
| `primuus.xml` | MuJoCo 本体：保留 URDF 的关节坐标和完整惯量，增加浮动机身、执行器、虚拟 IMU 和传感器 |
| `scene_flat.xml` | 引用本体并加入地面、灯光和相机；不复制机器人结构 |
| `../../../config/primuus.yaml` | 机器人运行配置：模型和策略路径、关节顺序、PD、默认姿态及观测契约 |

XML 使用相对于本目录的网格路径，不依赖最初的 CAD 导出目录。修改 URDF 的结构、惯量、坐标或碰撞后，必须同步 XML 并复核；两者不会自动同步。

## 当前参数

- 12 个执行器均按关节端力矩解释，控制输入和执行器力矩限制均为 `[-23.7, 23.7] N m`；URDF 的 effort 同步为 23.7。
- 默认姿态按 `FL / FR / RL / RR` 排列，每腿为 `hip=0、thigh=-0.75、calf=-1.5 rad`。
- YAML 的 `kp=60、kd=1.5` 是此前平地站立测试基线，不是电机厂家推荐值，也不代表带载抬腿和行走已通过。
- 物理步长为 `0.001 s`，策略每 20 步更新一次，即 50 Hz；PD 由现有运行器逐物理步更新。
- 地面和碰撞网格摩擦采用当前测试假设 `0.8 0.005 0.0001`。不模拟电机热限制、转速-力矩曲线或额外传动。
- XML 的 `imu` 是放在 `base_link` 原点的虚拟传感器，不代表实机安装位置。

## 查看模型

在工程根目录执行：

```bash
python3 -m mujoco.viewer --mjcf=src/controller/rl/resources/robots/primuus/scene_flat.xml
```

`home` 关键帧保存与 YAML 一致的参考位姿。查看器不会读取 YAML 或自动运行 PD，加载关键帧也不会主动保持站姿；直接播放没有控制器的模型可能倒地，不能用它判断站立控制效果。

## 策略配置

当前没有 Primuus 专用策略。`config/primuus.yaml` 的 `policy_path` 预留为 `models/primuus/policy.pt`，可以改为真实 Primuus TorchScript / ONNX 文件的路径。不得用 Go2 策略代替。

实际策略到位，并确认默认姿态、PD、动作顺序、45 * 6 观测和缩放参数与训练侧一致后，使用现有入口：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --config src/controller/rl/config/primuus.yaml
```

策略缺失时配置检查会拒绝启动，这是当前预期行为。本次没有添加无策略 PD 模式，也没有修改 Go2 的默认入口或配置。
