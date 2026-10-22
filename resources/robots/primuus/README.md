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
| PD增益 | `kp=60 / kd=1.5`，关节端单位 |
| 动作映射 | `target = default_angles + 0.25 * action` |
| 物理及控制 | 训练与play为 `dt=0.002 / substeps=1 / decimation=10`，PD 500 Hz、策略50 Hz；部署为 `0.0005 * 40`，PD 2 kHz、策略50 Hz |
| 高度目标 | 0.28 m，容差±0.03 m，即0.25-0.31 m内不惩罚 |
| 复位条件 | 机身Z为0.42 m，关节角倍率范围 `[0.9, 1.1]` |
| PhysX接触 | contact_offset 1 mm、rest_offset 0.5 mm、位置/速度迭代8/2 |

高度目标是奖励约束，不是独立高度控制器；默认关节角不保证承重后的高度与姿态。纯PD依靠位置误差产生承重力矩，足端支撑与非足端支撑应分开判断。

## 策略与配置

Primuus的训练和play分别使用自己的入口及 `logs/primuus/`，不读取Go2配置。play固定导出 `policy_1.pt / policy_1.onnx`，复制到部署工程时分别放为 `models/primuus/policy.pt / policy.onnx`；重复复制覆盖本地文件，不改变YAML。权重、机械定义、观测/动作顺序与关节损耗定义需配套，不能用同维度的Go2策略替换。

## 速度指令与采样

配置位于`PrimuusCfg.commands`，采样使用`quadruped_rl/envs/base/commands.py`。
每类指令统一保持5秒、直接切换，回合复位时重新采样。
下表为采样概率，实际训练时间占比还受复位及指令序列影响。

| 动作 | 采样概率 | 连续目标指令范围 |
| --- | ---: | --- |
| 纯前进 | 15% | vx为0.12–2 m/s，vy/yaw为0 |
| 原地转向 | 25% | vx/vy为0，yaw绝对值0.12–1 rad/s，左右均分 |
| 边走边转 | 25% | 任意XY方向，合速度0.12–1 m/s，yaw在[-1,1]rad/s内采样 |
| 平移 | 15% | 任意XY方向，合速度0.12–1 m/s，yaw为0 |
| 站立 | 20% | vx、vy、yaw均为0 |

内部五类依次为turn/moving_turn/translate/stand/forward。纯前进单独使用`ranges.lin_vel_forward=[0.12,2.0]`，不受其他平移状态的合速度限幅。
平移和边走边转允许前后、左右及斜向分量；vx、vy原始分量范围均为[-1,1]m/s，随后将XY合速度限制到0.12–1m/s。
30%命令使用0.35倍低速采样范围，纯前进低速子集为0.12–0.7m/s，原地转向的最低角速度仍为0.12rad/s。
课程关闭，`max_curriculum`参数不参与范围调整。日志包括五类`Command/*`和基础奖励、损失、性能统计。
观测、动作、机械、PD及17项奖励的参数说明见[运动训练说明](../../../doc/primuus_motion_training.md)。

## 关节损耗

`joint_dynamics.json`定义Primuus的非零关节损耗，和部署侧同名文件逐字节同步。参数为干摩擦屈服力矩0.05 N m、粘性阻尼0.02 N m s/rad、预滑移刚度5 N m/rad，均属于待实物辨识的仿真假设，不是PhysX原生摩擦系数的换算结果。

两侧`joint_dynamics.py`采用相同Jenkins弹簧—滑块公式，每个物理步根据实际关节位移更新预滑移状态；静止时可保留保持力矩，反向时释放已储存弹性能。不直接清零速度，复位时清除摩擦历史。

`PrimuusEnv`在运行时替换原生DOF friction/damping，保持URDF机械数据、惯量、地面摩擦、限矩和碰撞不变。`torques`仍是限幅后的电机力矩，`passive_joint_torques`为被动损耗，`net_joint_torques`为二者之和；力矩奖励不混入被动损耗。Go2不启用此模型。

部署端通过YAML的`simulation_joint_dynamics_path`显式加载，只在MuJoCo中施加，不向真实电机重复添加。统一公式不替换足地接触求解；相同状态历史下的损耗力矩一致，不代表整机轨迹完全相同。旧策略的训练损耗模型不同，不应直接视为兼容新模型。

训练与部署保持相同策略周期，不要求物理步长和PD更新频率相同。离散接触不保证任意高速冲击严格零穿入。碰撞包络、静止间距、机械限位与实际运动空间不能为了消除数值负距离而随意改变。MuJoCo和PhysX分别配置接触求解参数，不能将两者的参数值或行为直接等同。
