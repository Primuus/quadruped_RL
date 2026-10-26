# 场地模型

此目录集中存放场地与独立障碍资源。机器人机械定义仍在`../robots/`中维护。

| 文件 | 用途 |
| --- | --- |
| `ROBOCON2027.xml` | 原始完整比赛场景；保留原Go2引用、障碍布局和物理设置，仅调整迁移后的引用与网格路径 |
| `stairs.xml` | 独立台阶几何片段，在MJCF的`worldbody`或静态`body`内include，不包含机器人、地面或全局物理设置 |
| `scene_primuus_stairs.xml` | Primuus台阶审核场景，包含本体、台阶、地面、灯光和视角 |

完整场景用于保留场地来源，当前Primuus开发和审核使用独立场景。其余障碍尚未逐项审核，不能将整场文件视为全部满足规则的已核定模型。

## 台阶尺寸与坐标

来源为`ROBOCON2027.xml`的`stairs_07_00`、`stairs_07_01`、`platform_08`。
尺寸依据《2027年足式机器人挑战赛-V1.0》第3.1节、PDF第5页的“沟壑组合”。
提取范围为台阶侧的两级短踏面及顶部平台，不包括沟壑、另一侧平台和斜坡。

局部原点位于第一阶前缘中心的地面投影；+X为上阶方向，+Y为横向，+Z向上，单位为米。

| 几何名称 | X范围 | Y范围 | 顶面高度 | 踏面深度 |
| --- | --- | --- | --- | --- |
| `stairs_step_1` | 0～0.32 | -0.40～0.40 | 0.12 | 0.32 |
| `stairs_step_2` | 0.32～0.64 | -0.40～0.40 | 0.24 | 0.32 |
| `stairs_top_platform` | 0.64～1.44 | -0.40～0.40 | 0.36 | 0.80 |

共三次0.12m高度提升，总长1.44m、宽0.80m，顶部平台0.80m×0.80m。
MuJoCo的box `size`为半尺寸；三个箱体均从地面向上填实，接缝没有间隙。
按规则使用绿色外观；摩擦参数`0.8 0.1 0.1`沿用来源场景，属于仿真假设，并非比赛材料的实测值。

复用时可以整体平移或绕Z旋转父body，不应分别改变各踏面的相对位置：

```xml
<body name="stairs" pos="1 0 0">
  <include file="stairs.xml" />
</body>
```

审核场景将第一阶前缘放在世界坐标x=1m，顶部平台后缘为x=2.44m。
机器人位于原点并朝+X，使用本体默认姿态和`home`关键帧。地面为水平平面，场地无额外护栏。
独立模型可用于上台阶及反向下台阶检查；平台后缘是0.36m高的竖直边界，并非下行楼梯。
场景沿用Primuus本体的物理设置，不继承整场文件的RK4设置。

## 查看与加载策略

在`quadruped`工程根目录查看几何：

```bash
python3 -m mujoco.viewer \
  --mjcf=src/controller/rl/resources/site/scene_primuus_stairs.xml
```

裸Viewer不运行PD或策略，播放后的自由下落不能作为策略表现判断。加载现有Primuus策略：

```bash
python3 src/controller/rl/rl_controller/deploy_mujoco_rl.py \
  --config src/controller/rl/config/primuus.yaml \
  --xml-path src/controller/rl/resources/site/scene_primuus_stairs.xml
```

`--xml-path`只替换场景，策略路径、关节顺序、PD和观测仍由Primuus配置决定。
已有平地策略不保证具备台阶能力。默认`primuus.yaml`继续指向平地，审核场景通过命令显式选择。

## 后续训练接入

该XML用于MuJoCo场景加载，Isaac Gym训练需依据相同尺寸另外构建碰撞地形。
按规则第4.3节，台阶通过要求每一级顶面至少与足底接触一次；后续成功判定需记录各级接触。
此资源不包含奖励、成功判定、地形课程或策略切换逻辑。审核通过后，再逐项提取其他障碍。
