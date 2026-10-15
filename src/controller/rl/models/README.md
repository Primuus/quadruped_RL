# RL 策略模型目录

[RL子工程说明](../README.md) | [文档导航](../../../../doc/README.md)

这里存放训练工程导出的部署模型，不存放用于resume的完整训练checkpoint。核对日期：2026-10-15。

## 文件与契约

Go2默认文件名：

```text
policy_1.pt
policy_1.onnx
```

- `policy_1.pt`：TorchScript部署模型，MuJoCo可加载。
- `policy_1.onnx`：ONNX部署模型，MuJoCo和当前Python实机入口可加载。

当前不是C++实机运行器。两个文件均直接从同一checkpoint的PyTorch `estimator + actor` 导出，ONNX不经过TorchScript二次转换。critic、原型表及优化器不进入部署模型。

| 项目 | 当前12关节四足契约 |
| --- | --- |
| 外部输入 | float32，`[batch, 270]`；运行器逐次使用batch=1 |
| 历史顺序 | 最新帧在前，6帧各45维 |
| 外部输出 | float32，`[batch, 12]`原始动作，不是力矩 |
| 内部处理 | 历史encoder估计速度与latent，再输入actor |
| 部署侧处理 | 观测/动作裁剪、目标角映射、MuJoCo PD或实机目标命令 |

模型本身不维护历史缓存，不接收238维critic观测。维度相同不代表可以跨机器人复用；关节顺序、默认角、缩放、控制周期和物理模型均须匹配。

## 导出与版本管理

训练侧play默认覆盖 `logs/<experiment>/exported/policies/policy_1.pt` 和 `.onnx`，不覆盖 `model_<iteration>.pt`。它选择的是最新有效run和最大编号checkpoint，不是reward最好的策略。

两个导出文件不是原子发布。只有双导出都成功，且知道其run/checkpoint来源后，才复制到本目录；不能混用新PT和旧ONNX。

需要保留多个版本时，可使用 `models/<robot>/<run>_model_<iteration>/` 并修改机器人YAML中的 `policy_path`。至少记录源码commit、run、checkpoint、机器人模型、观测/动作顺序、默认角、PD、缩放、裁剪和策略周期。

`config/primuus.yaml` 预留 `models/primuus/policy.pt`，当前没有对应策略。不能把Go2模型重命名为Primuus策略。

## 数值对比

从部署仓库根目录执行：

```bash
python3 src/controller/rl/tools/compare_policy_backends.py \
  --torchscript-path src/controller/rl/models/policy_1.pt \
  --onnx-path src/controller/rl/models/policy_1.onnx
```

可加 `--observations observations.npy` 使用记录的270维输入。通过表示这批输入的数值一致，不等于闭环步态、实机安全或训练质量已验证。配置中的模型路径也必须先通过文件校验。

权重默认不提交Git；目录里是否存在某个文件是本机产物状态，不是源码保证。需要恢复训练时使用训练侧完整checkpoint，不能用部署PT/ONNX替代。
