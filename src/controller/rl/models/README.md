# RL 策略模型目录

[RL子工程说明](../README.md) | [文档导航](../../../../doc/README.md)

这里按机器人目录存放训练工程导出的部署模型，不存放用于resume的完整训练checkpoint。

## 文件与契约

默认路径：

```text
go2/policy_1.pt
go2/policy_1.onnx
primuus/policy.pt
primuus/policy.onnx
```

- `.pt`：TorchScript部署模型，MuJoCo可加载。
- `.onnx`：ONNX部署模型，MuJoCo和当前Python实机入口可加载。

实机运行器使用Python + ONNX Runtime。两个文件均直接从同一checkpoint的PyTorch `estimator + actor` 导出，ONNX不经过TorchScript二次转换。critic、target encoder、原型表及优化器不进入部署模型。

| 项目 | 当前12关节四足契约 |
| --- | --- |
| 外部输入 | float32，`[batch, 270]`；运行器逐次使用batch=1 |
| 历史顺序 | 最新帧在前，6帧各45维 |
| 外部输出 | float32，`[batch, 12]`原始动作，不是力矩 |
| 内部处理 | encoder读取270维历史，actor读取当前45维+估计速度3维+latent16维，共64维 |
| 部署侧处理 | 观测/动作裁剪、目标角映射、MuJoCo PD或实机目标命令 |

模型本身不维护历史缓存，不接收238维critic观测。维度相同不代表可以跨机器人复用；关节顺序、默认角、缩放、控制周期和物理模型均须匹配。

## 导出与版本管理

训练侧 `train` 在 `logs/<experiment>/<时间戳>_<run_name>/` 新建训练目录，保存 `model_<iteration>.pt`。只有执行 `play` 才导出部署模型：默认覆盖 `logs/<experiment>/exported/policies/policy_1.pt` 和 `.onnx`，不覆盖checkpoint。

`policy_1`是固定文件名，不是自动递增版本号；不会因为目录已有文件就生成 `policy_2`。默认从含checkpoint的run中选择checkpoint修改时间最新的目录，再选其中最大编号；不是选择reward最高的策略。需要确定来源时，显式指定 `--load_run` 和 `--checkpoint`。

两个导出文件不是原子发布。只有双导出都成功，且知道其run/checkpoint来源后，才复制到本目录；不能混用新PT和旧ONNX。

需要保留多个版本时，可使用 `models/<robot>/<run>_model_<iteration>/` 并修改机器人YAML中的 `policy_path`。至少记录源码commit、run、checkpoint、机器人模型、关节损耗定义、观测/动作顺序、默认角、PD、缩放、裁剪和策略周期。这些记录由使用者维护，导出文件不自动绑定完整训练配置。

`config/go2.yaml` 引用 `models/go2/policy_1.pt`，`config/primuus.yaml` 引用 `models/primuus/policy.pt`。Primuus目录中的 `policy.pt`、`policy.onnx` 分别对应训练导出的 `policy_1.pt`、`policy_1.onnx`。不能把Go2模型重命名为Primuus策略。

## 复制与覆盖

从本地部署仓库根目录执行，覆盖Primuus默认双模型：

```bash
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.pt \
  src/controller/rl/models/primuus/policy.pt
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.onnx \
  src/controller/rl/models/primuus/policy.onnx
```

Go2保持训练导出名，放在自己的目录：

```bash
scp -P 2222 'hurricane@10.109.69.166:~/quadruped_train/logs/go2/exported/policies/policy_1.*' \
  src/controller/rl/models/go2/
```

训练、导出和本地复制是三个独立操作。服务器重训或重新导出不会自动更新本地策略；只有复制时才覆盖目标文件。复制后分别对两端对应文件运行 `sha256sum`，确认PT和ONNX均完整传输，再进行数值对比。需要保留旧版本时先另存，不能依赖固定文件名找回旧权重。

Primuus的仿真损耗模型不打包进神经网络。`joint_dynamics.json`与两端适配算法必须另行保持一致；改变损耗模型后，不能仅凭权重维度兼容就认为旧策略已适配，见[关节动力学](../../../../doc/关节动力学.md)。

## 数值对比

从部署仓库根目录执行：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --config src/controller/rl/config/primuus.yaml \
  --torchscript-path src/controller/rl/models/primuus/policy.pt \
  --onnx-path src/controller/rl/models/primuus/policy.onnx
```

可加 `--observations observations.npy` 使用记录的270维输入。通过只表示这批输入的数值一致，不等于闭环步态、实机安全或训练质量。显式模型路径在校验前覆盖配置，最终配置和模型文件仍须有效。Go2对比时改用其配置及双模型路径。

权重默认不提交Git；目录里是否存在某个文件是本机产物状态，不是源码保证。需要恢复训练时使用训练侧完整checkpoint，不能用部署PT/ONNX替代。
