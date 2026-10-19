# 中文文档入口

训练工程的中文主文档统一维护在 [README.md](README.md)，不再保存第二份完整副本。

- [训练框架、续训与双模型导出](README.md)
- [中文安装与服务器使用](doc/setup_zh.md)
- [English Installation Guide](doc/setup_en.md)
- [本地算法库职责](rsl_rl/README.md)
- [训练调参教程](https://github.com/Primuus/quadruped_RL/blob/main/doc/训练调参教程.md)
- [TensorBoard 使用指南](https://github.com/Primuus/quadruped_RL/blob/main/doc/TensorBoard使用指南.md)

Go2与Primuus独立注册并维护各自资产、配置和策略，通用入口默认Primuus。训练创建新run，play覆盖固定双模型；Primuus与MuJoCo共享关节损耗定义，不影响Go2。参数及使用方式以主README和对应机器人源码为准。
