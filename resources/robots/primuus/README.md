# Primuus 训练资产

[训练首页](../../../README.md) | [任务配置](../../../quadruped_rl/envs/primuus/primuus_config.py)

本目录来自部署工程 `src/controller/rl/resources/robots/primuus/`，保留相同相对路径：

- `urdf/primuus.urdf`：训练导入入口，12个转动关节和8个固定关节，21个link。
- `stl/`：17个原始显示网格，以及机身、髋和足端的碰撞来源。
- `collision/`：大腿电机、板件来源、156个凸分块OBJ和20对命名自碰撞排除规则。

训练只需URDF及其引用资产；MuJoCo本体XML、场景XML和运行YAML仍由部署工程维护。训练运行不读取部署工作区，也不依赖Go2目录。

## 导入约定

- `collapse_fixed_joints=False`，保留四个足端和四个 `*_calf_collision_link`。
- `self_collisions=-1`，通过JSON排除正常装配接触，不关闭全部自碰撞。
- 足端选择 `_foot_link`；非足端接触选择 `thigh`、`calf`，后者包含小腿碰撞载体。
- 每个URDF collision对应一个凸形状，不再自动VHACD。
- 原始小腿范围 `[-2.7227, -0.8378] rad`，每关节理想力矩上限 `23.7 N m`。
- 策略关节顺序为FL/FR/RL/RR，每腿hip/thigh/calf。默认角 `[0, -0.75, -1.5]`，不使用Go2姿态。

修改资产时，训练和部署两侧的URDF、引用网格、命名排除规则必须同步；URDF坐标、惯量和碰撞变动还须同步部署XML。不要用收紧机械限位代替碰撞修复。

Primuus任务的PD暂沿用Go2的 `20 / 0.5`，部署YAML已同步；此前纯模型接触检查使用 `60 / 1.5`，不能当作当前训练配置通过。物理步长0.25 ms，PD每步更新，策略50 Hz，4096环境；GPU性能和训练行为由用户后续验收，本次未启动训练或仿真。
