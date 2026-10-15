# 第三方依赖

[返回RL子工程说明](../README.md)

## Unitree Actuator SDK

`unitree_actuator_sdk/` 是 Unitree 官方仓库的固定源码快照，供当前已实现的
GO-M8010-6 电机后端。电机协议和核心库保持上游实现；本工程只在 Python
wrapper 中加入绑定 API 与上游提交标记，用于拒绝错误版本。机器人关节映射、
机械零位、方向、减速比换算和安全检查均由自己的硬件适配层负责。

上游信息记录在 `unitree_actuator_sdk.version`。

当前固定提交为 `5b79a42d81cd69adac1367db79efdb972a898fd1`、绑定API为1。后端源码存在不表示已经通过真实电机验收；不是Unitree SDK2整机接口，也不是训练依赖。

更新 SDK 时必须：

1. 记录新的完整提交号。
2. 保留上游 `LICENSE`。
3. 同步更新 version 文件、Python wrapper、硬件后端和构建脚本中的 commit 标记；绑定契约变化时递增 API 版本。
4. 重新执行本地构建和 Python 导入检查。
5. 重新验证 GO-M8010-6 的命令与反馈字段。

构建当前主机对应的 C++ 示例和 Python 绑定：

```bash
bash src/controller/rl/third_party/build_unitree_actuator_sdk.sh
```

脚本只编译、校验版本标记并导入模块，不打开串口，也不会向电机发送命令。
生成的当前 Python ABI 模块和 `build/` 目录不会提交到 Git；每台部署主机都
必须使用自己的 Python 环境重新构建。

真实硬件阶段应使用 udev 规则或专用用户组授予 USB-RS485 访问权限，不要
使用 `sudo` 启动完整 RL 控制器。

上游SDK及其内部pybind11文档保留厂商原貌。本目录README负责本工程的集成契约，不把上游示例和工具当成正式RL运行入口。
