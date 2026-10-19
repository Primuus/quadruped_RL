# Primuus 仿真资源

[RL子工程说明](../../../README.md) | [运行配置](../../../config/primuus.yaml)

Primuus资产由URDF、显示网格、碰撞几何、MuJoCo本体和场景组成。训练与部署共用机械定义，但分别维护各自的引擎配置，不依赖Go2模型或策略。

## 文件职责

| 文件 / 目录 | 作用 |
| --- | --- |
| `urdf/primuus.urdf` | 训练机械定义：21个link、12个转动关节、8个固定关节及显式质量惯量 |
| `stl/` | 17个显示网格；独立足端的碰撞也引用对应STL |
| `collision/*_thigh_motor.STL` | 大腿电机碰撞网格 |
| `collision/*_thigh_plate.STL` | 板件离线分解来源，不作为运行时单个凸包 |
| `collision/convex/` | 49个碰撞OBJ，由URDF/XML共同引用 |
| `collision/self_collision_exclusions.json` | 20对正常装配接触的命名排除规则 |
| `primuus.xml` | MuJoCo本体：浮动机身、执行器、虚拟IMU和机械/碰撞数据 |
| `scene_flat.xml` | 引用本体并加入地面、灯光和相机 |
| `joint_dynamics.json` | 仿真关节损耗的物理单位参数与关节顺序；由Python适配器加载 |
| `../../../config/primuus.yaml` | 策略路径、关节顺序、PD、默认姿态与观测契约 |

XML网格路径相对于本目录。URDF/XML不会自动同步；修改碰撞、关节坐标或惯量后须同步维护。YAML不定义碰撞几何。

## 碰撞几何

模型使用65个碰撞形状：8个基础几何（4圆柱、4盒体）和57个mesh。显示精细度与碰撞复杂度相互独立，不能把显示网格的孔洞或面数直接等同于物理接触精度。

| 部位 | 数量 | 表示方式 |
| --- | --- | --- |
| 机身 | 1 | 单个凸包，避免外接盒体侵占腿部活动空间 |
| 髋部 | 4 | 每髋1个凸包 |
| 大腿 | 44 | FL/RL各10个，FR/RR各12个 |
| 小腿 | 12 | 每腿3个凸块，挂在固定碰撞载体上 |
| 足端 | 4 | 独立足端STL，与小腿接触分开归属 |

大腿几何按接触作用划分：

- `thigh_motor`、`thigh_inner/outer`：电机与两侧板。
- `thigh_housing`：XZ平面半径40 mm、沿Y方向长25 mm的圆柱，中心Y为左右镜像的±51.2 mm，近似上部外壳。
- `thigh_transition`：厚3 mm、沿Y方向25 mm的斜盒体，近似上部到长槽的过渡壁。
- `thigh_channel_back/front/tip`：长槽主壁和末端凸块；局部 `Z=-130 mm` 代表截面的主槽净宽为31 mm。
- `thigh_fold_guard_*`：折膝挡壁，左侧每腿1块、右侧每腿3块；分块数量取决于几何形状。
- `thigh_lower_rim`：下部外缘的低面数凸包。

这些是接触代理，不逐点复现显示网格的所有孔洞、凸台和装配腔体。简化必须保留关键运动通道，不能用整根实心胶囊填满中空大腿。主槽某个截面净宽不变，也不等于所有姿态下的表面距离都相同。

## 机械契约

- 总质量为 `12.04293025733458 kg`。各link使用显式局部质心和惯量，不能按简化碰撞体体积重新计算。
- 四个大腿关节横向偏置为左右镜像的 `+/-0.0343 m`；关节轴向、坐标和惯量须在训练URDF与MuJoCo XML间一致。
- 12个理想关节执行器限矩为 `+/-23.7 N m`，不额外模拟齿轮或连杆传动。
- 小腿机械范围为 `[-2.7227, -0.8378] rad`。自碰撞应由接触处理，不以收紧关节范围代替。
- 关节顺序为FL/FR/RL/RR，每腿hip/thigh/calf。默认角为每腿 `[0, -0.75, -1.5] rad`。
- XML中的body姿态与joint.ref成对补偿，使默认加载、home关键帧和YAML采用相同关节角语义。
- 四个 `*_calf_collision_link` 与小腿刚性连接，不新增DOF；载体和原小腿各占对应质量/惯量的一半，局部质心相同。
- 足端只归属足端接触。训练非足端选择包含 `calf`，从而包含固定碰撞载体。

## 引擎与控制

| 参数 | 设置与含义 |
| --- | --- |
| 物理步长 | 训练 `sim.dt=0.002`、`substeps=1`；部署MuJoCo timestep与YAML `simulation_dt`均为0.0005 s |
| PD周期 | 每物理步更新，训练500 Hz、部署2 kHz |
| 策略周期 | 训练 `decimation=10`、部署 `control_decimation=40`，均为20 ms，即50 Hz |
| PD增益 | `kp=60 / kd=1.5`，关节端单位 |
| 动作映射 | `target = default_angles + 0.25 * action` |
| 机身高度目标 | 0.28 m，容差±0.03 m，即0.25-0.31 m内不受高度惩罚 |
| MuJoCo接触 | margin 1 mm，solref时间常数0.5 ms；refsafe下有效时间常数至少为1 ms |
| PhysX接触 | contact_offset 1 mm、rest_offset 0.5 mm，位置/速度迭代8/2 |

高度目标是奖励约束，不是独立的机身高度控制器；默认关节角也不等于承重后能无误差保持的姿态。纯PD需要关节位置误差产生承重力矩，站姿高度、机身姿态、足端支撑和非足端接触应分别理解。

Primuus的训练与策略部署使用[统一关节损耗](../../../../../../doc/关节动力学.md)：Jenkins预滑移干摩擦加粘性阻尼，运行时替换原生关节损耗，不改PD、限矩和接触。参数是待实物标定的仿真假设。裸Viewer不运行Python适配器，仍使用XML原生损耗。

训练与部署保持相同策略周期，不要求物理步长和PD更新频率相同。离散软接触不保证任意高速冲击下数学意义的零穿入；不能用增大碰撞包络、扩大静止间距或关闭自碰撞来隐藏问题。MuJoCo与PhysX的参数不一一等价，具体语义见[MuJoCo接触参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

训练导入约定：

- `collapse_fixed_joints=False`，保留足端与小腿碰撞载体。
- `replace_cylinder_with_capsule=False`，保持圆柱尺寸与平端面，不自动扩为胶囊。
- `self_collisions=-1`，使用命名过滤位，不关闭整机自碰撞。
- MuJoCo设置 `filterparent=disable`，仅用掩码排除正常装配接触。通用位1、机身2、FL/FR/RL/RR髋4/8/16/32、对应大腿电机64/128/256/512；双方均拒绝对方类型才排除碰撞。

`PrimuusEnv / PrimuusCfg / PrimuusCfgPPO` 直接继承通用base，不导入Go2。两侧URDF、引用网格及排除JSON应一致，XML由部署工程维护。

## 查看与部署

在部署工程根目录查看模型：

```bash
python3 -m mujoco.viewer --mjcf=src/controller/rl/resources/robots/primuus/scene_flat.xml
```

空格暂停。裸Viewer不读取YAML、不运行PD，直接播放后的自由下落不能代表受控站立。默认部署初始机身Z为0.31212 m，训练复位高度为0.42 m，两者不是高度奖励目标。

查看碰撞时关闭Geom group 2、开启3，并启用 **Convex Hull**；否则STL可能仍按显示网格渲染。外观材质为浅银灰。

使用匹配Primuus观测和动作契约的策略：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --config src/controller/rl/config/primuus.yaml
```

策略路径由YAML的 `policy_path` 指定，默认是 `models/primuus/policy.pt`，对应训练导出的 `policy_1.pt`；ONNX对应 `models/primuus/policy.onnx`。复制与覆盖规则见[模型目录](../../../models/README.md)。正式入口会拒绝不存在的文件或不符合契约的模型，不得用Go2策略替代Primuus策略，也不要通过占位文件绕过校验。
