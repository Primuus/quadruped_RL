# 训练环境安装与服务器使用

[返回训练首页](../README.md) | [English](setup_en.md)

本指南只对应本仓库的 Isaac Gym + HIM-PPO 训练链路。部署工程使用独立环境；不需要安装 ROS2、Unitree SDK2、`unitree_rl_gym`，也不要从上游克隆另一个 `rsl_rl` 覆盖本仓库的修改。

## 训练服务器

优先连接：

```bash
ssh -p 2222 hurricane@10.109.69.166
```

服务器内：

```bash
cd ~/quadruped_train
conda activate isaacgym
```

不要在服务器home目录直接运行训练脚本，路径相对于仓库根目录。Primuus使用 `train_primuus.py / play_primuus.py`，Go2使用自己的专用入口；通用入口默认Primuus。

以下为复现环境的参考版本，不是所有主机的兼容保证；使用前以目标conda环境中的实际包版本为准：

| 组件 | 版本或路径 |
| --- | --- |
| Python | 3.8.10 |
| Isaac Gym | 1.0rc4，源码位于 `~/RL/isaacgym` |
| PyTorch | 2.4.1 |
| NumPy | 1.23.2 |
| ONNX | 1.17.0 |
| TensorBoard | 2.12.0 |

修改已工作的环境前先记录依赖，停训后再操作。源码和文档同步不等于修改服务器底层：不要自动升级驱动、CUDA、Isaac Gym或系统设置；遇到设备或SDK限制先报告原因，由使用者决定处理方式。

## 已有环境接入源码

在已经有上述依赖的环境中执行：

```bash
cd ~/quadruped_train
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
```

这两个 editable 安装分别暴露 `rsl_rl.env` 和 `quadruped_rl`。本仓库根目录下的 `rsl_rl/` 是一个独立 Python 项目，其真正包位于 `rsl_rl/rsl_rl/`，只设置根目录 `PYTHONPATH` 不够。

`setup.py` 目前仍声明 `numpy==1.20`，与现有服务器 NumPy 不一致。这里使用 `--no-deps` 是为了保留现有已工作的环境，不表示元数据约束已修正；缺少的运行依赖仍需显式安装，不能把该选项当成依赖检查通过。

## 新环境准备顺序

新主机需要 NVIDIA GPU、可用驱动、C/C++ 构建工具及 NVIDIA 官方提供的 Isaac Gym Preview 安装包。先确认驱动正常，再复现训练环境；训练 Python 3.8 与部署 Python 3.10+ 不共用环境。

```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
```

先安装与目标 GPU、驱动和 Isaac Gym 匹配的 PyTorch，再安装 NumPy、ONNX、TensorBoard、matplotlib、PyYAML、ninja。优先复现现有工作环境，不直接选择“最新版本”。

解压官方 Isaac Gym 后，按实际位置设置路径并安装：

```bash
export ISAACGYM_ROOT="$HOME/RL/isaacgym"
python -m pip install --no-deps -e "$ISAACGYM_ROOT/python"
cd ~/quadruped_train
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
```

Isaac Gym不随本仓库分发。额外系统依赖按安装包内的说明和`docs/`配置。

## 检查安装

先做只读检查：

```bash
which python
python -m pip show isaacgym rsl_rl quadruped_rl
nvidia-smi
```

需要导入检查时，保持 Isaac Gym 在 PyTorch 之前导入：

```bash
python -c "import isaacgym; import torch; import quadruped_rl; from rsl_rl.env import VecEnv; print(torch.__version__)"
```

训练入口和默认参数见[训练首页](../README.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到脚本 | 先 `cd ~/quadruped_train`，不要修改脚本里的相对路径来补救错误工作目录 |
| `No module named quadruped_rl` | 在当前 Python 环境重新 editable 安装仓库根目录 |
| `No module named rsl_rl.env` | 安装仓库自带的 `./rsl_rl`；核对是否误装其他版本 |
| 提示 ninja 缺失 | 检查当前环境是否安装 ninja，以及激活环境后的 `PATH` |
| `libpython3.8.so` 找不到 | 核对 conda 动态库路径；必要时给单次命令设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"` |
| CUDA launch failure | 先停止新任务并查看 GPU 状态，不能仅凭报错栈就断定 reward 或 HIM-PPO 有错 |
| `nvidia-smi` 也无法识别 GPU | 优先检查驱动、内核错误及硬件状态；不要继续反复启动训练或直接重装整个环境 |

CUDA 调试可在单次复现命令前使用 `CUDA_LAUNCH_BLOCKING=1`。这改变执行同步行为，不能当作正式性能设置，也不能修复设备掉线。

## Docker

Docker 与 conda 环境独立。现有 Dockerfile 基于 `nvcr.io/nvidia/pytorch:22.12-py3`，其依赖版本不等于上表服务器环境；先核对本机驱动、Docker 和 NVIDIA Container Toolkit。

```bash
cd ~/quadruped_train
export ISAACGYM_ROOT="$HOME/RL/isaacgym"
bash docker/build.sh
```

`docker/build.sh` 构建并启动容器，再进入 shell；容器把本仓库挂载到 `/workspace/quadruped_train`、Isaac Gym 挂载到 `/opt/isaacgym`。启动脚本会在缺少 Isaac Gym 安装映射时安装挂载目录，但不会自动训练。

用户确认资源可用后，才执行训练命令。无头训练不需要可视化窗口；带窗口的 play 还需要可用的显示会话，SSH 本身不保证窗口能显示。

## 同步和取回产物

同步前先核对服务器改动。只传本次修改文件，使用 `rsync -n` 预览，不向正在训练的目录执行覆盖式全量同步或 `--delete`。

```bash
rsync -anv -e "ssh -p 2222" README.md \
  hurricane@10.109.69.166:~/quadruped_train/README.md
```

确认后移除 `-n`。检查点和事件日志不是源码，不应在普通代码同步时覆盖。

在本地部署仓库根目录取回Primuus双模型，并改为YAML引用的名称：

```bash
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.pt \
  src/controller/rl/models/primuus/policy.pt
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.onnx \
  src/controller/rl/models/primuus/policy.onnx
```

这会覆盖同名本地文件。先保存需要保留的版本，并确认PT/ONNX来自同一次成功导出。Go2双模型保留 `policy_1.*`并放在 `src/controller/rl/models/go2/`，不能混用机器人策略。重复play覆盖固定导出文件，不生成 `policy_2`；服务器导出不会自动更新本地部署。完整训练、导出与复制规则见[训练首页](../README.md)。

Primuus还需同步 `resources/robots/primuus/joint_dynamics.json`与两端关节损耗算法；它们属于仿真动力学配置，不包含在策略权重里，也不是实机电机标定参数。

