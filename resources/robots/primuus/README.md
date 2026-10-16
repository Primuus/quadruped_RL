# Primuus 训练资产

[训练首页](../../../README.md) | [任务配置](../../../quadruped_rl/envs/primuus/primuus_config.py)

本目录与部署工程 `src/controller/rl/resources/robots/primuus/` 的URDF及引用资产采用相同机械定义。训练不依赖部署工作区或Go2目录；MuJoCo XML和YAML由部署工程维护。

## 模型组成

模型包含21个link、12个转动关节、8个固定关节，总质量为 `12.04293025733458 kg`。显示使用17个STL，碰撞使用65个形状：4圆柱、4盒体和57个mesh，凸OBJ共49个。

| 部位 | 碰撞表示 |
| --- | --- |
| 机身 | 1个凸包，避免外接盒体侵占腿部活动空间 |
| 髋部 | 每髋1个凸包，共4个 |
| 大腿 | FL/RL各10个、FR/RR各12个，共44个 |
| 小腿 | 每腿3个凸块，共12个，保留固定载体 |
| 足端 | 4个独立足端，碰撞引用对应STL |

大腿按电机、侧板、上部外壳、过渡壁、长槽和折膝挡壁划分。上部外壳以半径40 mm、长度25 mm的圆柱近似，过渡壁以斜盒体近似；`thigh_channel_*` 保留主运动通道，`thigh_fold_guard_*` 表示折膝接触面，`thigh_lower_rim` 表示下缘轮廓。

局部 `Z=-130 mm` 代表截面的主槽净宽为31 mm。接触代理不逐点复现全部孔洞和装配小特征，但不能填占小腿运动间隙，也不能将中空大腿整体替换为实心胶囊。显示精细度不决定碰撞形状数量。

## 导入契约

- `collapse_fixed_joints=False`，保留四足端和四个 `*_calf_collision_link`。
- `replace_cylinder_with_capsule=False`，保持圆柱尺寸和平端面，不自动扩包。
- `self_collisions=-1`，使用JSON中的20对命名装配排除，不关闭整机自碰撞。
- 每个URDF collision对应一个形状，不自动VHACD；形状数量或名称不符时显式报错。
- 足端选择 `_foot_link`；非足端选择 `thigh`、`calf`，包含小腿固定碰撞载体。
- 各link保留显式局部质心和惯量，不根据碰撞体体积重算。原小腿和固定载体各占对应质量/惯量的一半，质心相同，不新增DOF。
- 四个大腿关节横向偏置为左右镜像的 `+/-0.0343 m`。
- 小腿机械范围为 `[-2.7227, -0.8378] rad`，理想关节力矩上限为 `23.7 N m`；不收紧限位代替接触。

`PrimuusEnv / PrimuusCfg / PrimuusCfgPPO` 直接继承base，不导入Go2。修改机械定义须同步两侧URDF、引用网格、排除JSON及部署XML，不修改Go2资产来接入Primuus。

## 控制与奖励

| 参数 | 含义 |
| --- | --- |
| 顺序 | FL/FR/RL/RR，每腿hip/thigh/calf |
| 默认角 | 每腿 `[0, -0.75, -1.5] rad`，与部署YAML/XML一致 |
| PD增益 | `kp=40 / kd=1.0`，关节端单位 |
| 动作映射 | `target = default_angles + 0.25 * action` |
| 物理及控制 | `dt=0.0005 / substeps=1 / decimation=40`，PD2 kHz、策略50 Hz |
| 高度目标 | 0.28 m，容差±0.03 m，即0.25-0.31 m内不惩罚 |
| 复位条件 | 机身Z为0.42 m，关节角倍率范围 `[0.9, 1.1]` |
| PhysX接触 | contact_offset 1 mm、rest_offset 0.5 mm、位置/速度迭代8/2 |

高度目标是奖励约束，不是独立高度控制器；默认关节角不保证承重后的高度与姿态。纯PD依靠位置误差产生承重力矩，足端支撑与非足端支撑应分开判断。

0.5 ms用于正常PD运动，不保证任意高速冲击严格零穿入。碰撞包络、静止间距、机械限位与实际运动空间不能为了消除数值负距离而随意改变。MuJoCo和PhysX分别配置接触求解参数，不能将两者的参数值或行为直接等同。
