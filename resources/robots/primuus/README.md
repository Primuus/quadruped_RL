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

配置位于 `PrimuusCfg.commands`，采样与课程实现位于 `quadruped_rl/envs/primuus/commands.py`。每5秒以及回合复位时重新采样，比例表示命令抽样概率，不保证实际时间占比完全相同。

| 模式 | 配置顺序中的概率 | 连续指令范围 |
| --- | --- | --- |
| stand | 20% | vx、vy、yaw均为0 |
| forward | 35% | vx为0.12至当前前进上限，vy/yaw为0 |
| backward | 10% | vx为-1至-0.12 m/s，vy/yaw为0 |
| lateral | 10% | vy为±0.12至±1 m/s，vx/yaw为0 |
| diagonal | 5% | XY合成速度0.12至1 m/s，四象限均分，夹角15°至75°，yaw为0 |
| turn | 10% | yaw为±0.12至±1 rad/s，vx/vy为0 |
| moving_turn | 10% | XY合成速度0.12至1 m/s，方向均匀，yaw为±0.12至±1 rad/s |

所有左右方向均分；斜向和边走边转通过极坐标采样速度，不将XY各自限幅后产生超过合成速度上限的命令。高速范围仅扩展直线前进，后退、侧移和混合运动保持各自的1 m/s上限。

前进内部连续分段采样：0.12–1、1–2、2–4 m/s分别占30%、40%、30%。未开放的区间移除，其他区间按原权重重新归一化；例如上限为1.5时，前两段各占3/7、4/7，第二段截断至1.5。`low_speed_probability/low_speed_scale`是通用基类的旧四模式参数，不控制Primuus七模式采样。

课程初始前进上限为1 m/s，每次增加0.5，最终上限4 m/s。4 m/s是待验证的训练目标，不代表当前策略或实机已经具备该能力。`curriculum=False`时直接开放完整范围，常规训练建议保留课程。

晋级只使用当前阶段、速度不低于当前上限80%的直线前进指令段。每段前1秒不统计跟踪，随后至少观察1秒；过早发生的非超时终止仍计入失败。每20仿真秒检查一次，累计至少200个合格段后，要求段均跟踪分数不低于0.8、非超时终止率不超过5%，才提高上限；跟踪分数为 `exp(-XY速度误差平方 / tracking_sigma)`。每个有效评估窗口结束后清空统计，新阶段不复用旧阶段指令段。站立、低速和其他运动模式不参与晋级。

命令在本步奖励和统计完成后、下一次观测构造前切换，保证统计对应实际执行的指令。TensorBoard的 `Command/<mode>/` 记录七种动作；`Command/forward_low`、`forward_normal`、`forward_fast`分别记录前进三个区间，区间边界归入后一段。区间的fraction以所有环境步为分母。`Curriculum/forward_limit`、`high_speed_tracking`、`high_speed_failure_rate`、`evaluated_segments`记录课程状态与最近一个完整评估窗口。

课程进度随训练checkpoint保存；旧checkpoint从1 m/s上限开始，新checkpoint恢复已学阶段。play读取保存的上限，但不会自动晋级。此次配置不增加网络输入或高度指令，不修改现有高度奖励、机械参数、PD与部署限幅。

## 关节损耗

`joint_dynamics.json`定义Primuus的非零关节损耗，和部署侧同名文件逐字节同步。参数为干摩擦屈服力矩0.05 N m、粘性阻尼0.02 N m s/rad、预滑移刚度5 N m/rad，均属于待实物辨识的仿真假设，不是PhysX原生摩擦系数的换算结果。

两侧`joint_dynamics.py`采用相同Jenkins弹簧—滑块公式，每个物理步根据实际关节位移更新预滑移状态；静止时可保留保持力矩，反向时释放已储存弹性能。不直接清零速度，复位时清除摩擦历史。

`PrimuusEnv`在运行时替换原生DOF friction/damping，保持URDF机械数据、惯量、地面摩擦、限矩和碰撞不变。`torques`仍是限幅后的电机力矩，`passive_joint_torques`为被动损耗，`net_joint_torques`为二者之和；力矩奖励不混入被动损耗。Go2不启用此模型。

部署端通过YAML的`simulation_joint_dynamics_path`显式加载，只在MuJoCo中施加，不向真实电机重复添加。统一公式不替换足地接触求解；相同状态历史下的损耗力矩一致，不代表整机轨迹完全相同。旧策略的训练损耗模型不同，不应直接视为兼容新模型。

训练与部署保持相同策略周期，不要求物理步长和PD更新频率相同。离散接触不保证任意高速冲击严格零穿入。碰撞包络、静止间距、机械限位与实际运动空间不能为了消除数值负距离而随意改变。MuJoCo和PhysX分别配置接触求解参数，不能将两者的参数值或行为直接等同。
