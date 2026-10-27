# Primuus obstacles 障碍训练

`primuus_obstacles`是独立的Isaac Gym训练任务，环境、配置和日志与`primuus`平地任务分别选择。
它继承Primuus的机械模型、45×6历史观测、12关节动作、HIM-PPO、运动采样、域随机化及19项奖励。
Actor不读取地形高度，部署模型仍输入270维；critic的238维输入中，末尾187维填入实际地形高度观测。

## 入口与文件

在训练工程根目录执行：

```bash
python quadruped_rl/scripts/train_primuus.py --task primuus_obstacles --headless
python quadruped_rl/scripts/play_primuus.py --task primuus_obstacles --load_run <run目录名> --checkpoint <轮数>
```

通用`train.py`和`play.py`也支持`--task primuus_obstacles`。不传`--task`时，Primuus专用入口继续选择平地任务。

| 文件 | 职责 |
| --- | --- |
| `quadruped_rl/envs/primuus/primuus_obstacles_config.py` | 地形尺寸、幅度、采样开关及独立日志目录 |
| `quadruped_rl/envs/primuus/obstacles_terrain.py` | 调用Isaac Gym标准生成器，拼接高度场并转换成碰撞三角网格 |
| `quadruped_rl/envs/primuus/primuus_obstacles_env.py` | 加载网格、分配复位原点、采样高度及处理地形边界 |

检查点保存在`logs/primuus_obstacles/<run>/`，play导出至`logs/primuus_obstacles/exported/policies/`。
恢复本任务时，`--max_iterations`表示本次追加轮数。
平地检查点可以通过绝对目录作为起点，输出仍进入障碍任务目录：

```bash
python quadruped_rl/scripts/train_primuus.py --task primuus_obstacles --headless --resume \
  --load_run /绝对路径/quadruped_train/logs/primuus/<平地run目录名> \
  --checkpoint <轮数> --max_iterations <追加轮数>
```

两任务网络形状相同，可加载平地权重；障碍任务的critic高度输入有实际数值，加载后仍需训练适应地形。

## 地形布局和幅度

地形是4行×5列的拼图，每块8m×8m；行索引决定幅度，列决定地形类型。
每种类型约占相同环境数，各环境复位时独立随机选一行，不根据表现晋级。
当前`terrain.curriculum=False`，障碍任务不执行地形课程；调整难度使用下表范围和`num_rows`。

| 类型 | 范围参数 | 默认值与含义 |
| --- | --- | --- |
| `blocks` | `block_height_range` | 最大绝对高度2～8cm；标准生成器包含凸块和负高度凹陷 |
| `blocks` | `block_size_range`、`num_blocks` | 矩形长宽0.4～1.2m，每块地形采样24个，允许重叠 |
| `stairs_up` / `stairs_down` | `step_height_range`、`step_width` | 每级2～8cm、踏面深0.30m；整片台阶总高由级数决定 |
| `slope_up` / `slope_down` | `slope_range` | rise/run为0.05～0.25，约2.9°～14.0° |

台阶和坡面使用金字塔/倒金字塔生成器，中央为复位平台；类型中的up/down指从中央平台向外走时的上行/下行。
机器人初始世界Z为地形平台高度加`init_state.pos[2]=0.35m`，不是固定世界Z=0.35m。
`platform_size=4.0m`为中央安全平台参数，实际台阶平台会受踏面网格取整影响；覆盖既有±1m的XY复位扰动及机器人站姿。
水平网格0.05m、竖直量化0.0025m，幅度按网格量化；`slope_treshold=0.35`用于修正台阶陡边。
整张地图外有5m平地缓冲，碰撞使用单一三角网格，不额外叠加全局平面，避免遮住凹陷。

距本块地形边界不足`edge_margin=0.5m`时截断回合并复位，按timeout处理；已有碰撞或姿态失败条件优先。
这避免机器人走出所属地形或有限地图。回合时间上限仍是20秒。

## 地形高度与现有奖励

高度采样使用17×11个点，范围为机身前后±0.8m、左右±0.5m，只跟随机身位置和偏航角。
世界地面高度保存在`measured_heights`；critic高度输入为
`clip(z_base - 0.5 - measured_heights, -1, 1) * height_measurements_scale`。
更新包含正常策略步和局部复位，终止状态的critic观测在复位前取得。
`terrain.measure_heights=False`可关闭critic高度输入；现有高度奖励仍需要地面采样。

高度参考来自开源[legged_gym/legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)的
`_get_heights()`、`_reward_base_height()`及`compute_observations()`。
上游`_reward_base_height()`在所有状态计算`(mean(z_base-ground)-target)^2`；
通用[legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)中`base_height=-0.0`，默认不启用。
Primuus只采用其相对地面高度参考，保留原有`base_height`公式：目标0.28m、容差±0.03m、超出部分除以0.05m后平方，启用权重仍为−0.4，无额外触发门槛。
现有换步统计的接地高度在复位时初始化为平台高度，避免把平台的世界高度计作抬脚距离。
其他奖励、权重和运动状态判定均继承平地任务；本任务不新增奖励。

网格地形与复位原点机制参考[legged_gym Terrain](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py)。
实际生成调用Isaac Gym的`discrete_obstacles_terrain()`、`pyramid_stairs_terrain()`、`pyramid_sloped_terrain()`和`convert_heightfield_to_trimesh()`。
Primuus适配使用较小的初始幅度、独立任务、无地形晋级、相对地形复位及边界timeout。
187维高度仅进入critic；不同于上游可感知配置直接把高度拼入actor，Primuus保留270维盲行走Actor契约。

## MuJoCo场景

部署工程的`src/controller/rl/resources/robots/primuus/scene_obstacles.xml`提供固定布置的低障碍、台阶、斜坡及矮块。
它与训练地形覆盖相近类型，用于加载策略观察运动；布局不等于Isaac Gym的随机地形拼图。
场景参数和运行命令见部署工程的`resources/site/README.md`。
