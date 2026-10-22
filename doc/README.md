# 文档导航

[返回仓库首页](../README.md)

本目录维护训练操作与部署原理说明；部署源码与训练源码独立，参数以各自机器人配置为准。

## 按任务阅读

| 任务 | 文档 |
| --- | --- |
| 运行Go2 / Primuus策略或查看机械模型 | [仓库首页](../README.md) |
| 理解部署包每个文件的职责 | [RL子工程说明](../src/controller/rl/README.md) |
| 了解训练默认参数、如何修改和新增reward/随机化 | [训练调参教程](训练调参教程.md) |
| 使用TensorBoard并解释曲线 | [TensorBoard使用指南](TensorBoard使用指南.md) |
| 管理checkpoint对应的双模型产物 | [策略模型目录](../src/controller/rl/models/README.md) |
| 理解Primuus关节损耗及训练/部署接入 | [关节动力学](关节动力学.md) |
| 了解Primuus几何、碰撞载体与模型边界 | [Primuus模型说明](../src/controller/rl/resources/robots/primuus/README.md) |
| 构建固定版本电机SDK | [第三方依赖说明](../src/controller/rl/third_party/README.md) |
| 安装训练环境和使用服务器 | [训练分支安装指南](https://github.com/Primuus/quadruped_RL/blob/train/doc/setup_zh.md) |
| 理解HIM-PPO算法包 | [训练分支算法说明](https://github.com/Primuus/quadruped_RL/blob/train/rsl_rl/README.md) |

## 使用边界

- Go2与Primuus分别注册独立训练任务，各自维护模型和策略，不交叉加载。
- 默认任务使用平地；critic的外力3维、高度187维是零占位，不代表真实测量。
- 实机软件链路是Python + ONNX Runtime +电机/IMU后端；仿真关节损耗不重复施加到真实电机。
- MuJoCo debug用于查看运行状态，双模型工具用于对照TorchScript和ONNX输出。
- 参数表描述源码默认值；具体策略的参数由其训练来源决定。

## 维护原则

训练参数以训练分支实例配置为准，部署参数以机器人YAML/XML为准。更新接口时同步入口说明和对应专题，不在多个README复制完整教程。

文档维护工程结构、配置含义、接口和使用方法。第三方SDK及内部依赖的上游Markdown保留原貌。
