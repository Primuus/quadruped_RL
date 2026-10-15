# 文档导航

[返回仓库首页](../README.md)

本目录维护训练操作教程；部署源码与训练源码独立，教程以两侧当前实现为准。核对日期：2026-10-15。

## 按任务阅读

| 任务 | 文档 |
| --- | --- |
| 快速运行Go2 MuJoCo或查看Primuus模型 | [仓库首页](../README.md) |
| 理解部署包每个文件的职责 | [RL子工程说明](../src/controller/rl/README.md) |
| 了解训练默认参数、如何修改和新增reward/随机化 | [训练调参教程](训练调参教程.md) |
| 使用TensorBoard并解释曲线 | [TensorBoard使用指南](TensorBoard使用指南.md) |
| 管理checkpoint对应的双模型产物 | [策略模型目录](../src/controller/rl/models/README.md) |
| 了解Primuus几何、碰撞载体与验证范围 | [Primuus模型说明](../src/controller/rl/resources/robots/primuus/README.md) |
| 构建固定版本电机SDK | [第三方依赖说明](../src/controller/rl/third_party/README.md) |
| 安装训练环境和使用服务器 | [训练分支安装指南](https://github.com/Primuus/quadruped_RL/blob/train/doc/setup_zh.md) |
| 理解HIM-PPO算法包 | [训练分支算法说明](https://github.com/Primuus/quadruped_RL/blob/train/rsl_rl/README.md) |

## 当前边界

- 只有Go2训练任务；Primuus具备模型与独立配置，但尚无任务和匹配策略。
- 当前训练为平地。critic的外力3维、高度187维是零占位，不是已经接入真实信号。
- 当前实机软件链路是Python + ONNX Runtime +电机/IMU后端，未完成实物验收。
- 不保留mock实机测试链路；保留MuJoCo debug与双模型数值对比。
- 参数表描述源码，不能替代某个checkpoint的训练记录；示例数值也不表示已在所有机器人上验证。

## 维护原则

训练参数以训练分支实例配置为准，部署参数以机器人YAML/XML为准。更新接口时同步入口说明和对应专题，不在多个README复制完整教程。

第三方SDK及其内部依赖的上游Markdown保留原貌，不把厂商说明改写成本工程已经验证的能力。工作区外层的方案和日志不属于本目录维护范围。
