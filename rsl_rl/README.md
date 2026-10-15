# 本地 RSL RL 算法库

[返回训练首页](../README.md)

这是随工程维护的 `rsl_rl` 分支，包版本元数据仍为 `1.0.2`，不能据此认为它与上游同版本源码完全相同。现有 Go2 任务使用 HIM-PPO；不要单独升级或用上游安装覆盖这些实现。

源码源自 RSL RL，保留 NVIDIA、ETH Zurich、Nikita Rudin 等原始文件声明，许可见 [LICENSE](LICENSE)。HIM 相关实现及当前工程的导出契约以本目录实际代码为准。

## 安装与生效

从训练仓库根目录安装：

```bash
python -m pip install --no-deps -e ./rsl_rl
```

实际 Python 包位于 `rsl_rl/rsl_rl/`。机器人任务在 `quadruped_rl/envs/` 配置以下三个名称，不能只更换算法名：

```text
HIMActorCritic -> HIMPPO -> HIMOnPolicyRunner
```

当前通用基类仍有普通 PPO 默认值，Go2 配置显式覆盖为上述 HIM 组合。新增任务必须做同样的显式选择。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `rsl_rl/env/vec_env.py` | 环境接口：obs、privileged obs、动作、reset 和 step |
| `rsl_rl/modules/him_actor_critic.py` | 组合历史编码器、actor、critic 和训练动作分布 |
| `rsl_rl/modules/him_estimator.py` | 由历史观测估计速度与 latent，训练速度 MSE 和原型 swap loss |
| `rsl_rl/algorithms/him_ppo.py` | PPO clipped objective、GAE 返回后的优化、KL 学习率调节和 estimator 更新 |
| `rsl_rl/storage/him_rollout_storage.py` | 存储 rollout、历史输入及下一时刻 critic 观测 |
| `rsl_rl/runners/him_on_policy_runner.py` | 采样、终止前状态接续、优化调度、TensorBoard 和 checkpoint |
| `rsl_rl/modules/actor_critic*.py` | 保留的普通 MLP / recurrent actor-critic 实现 |
| `rsl_rl/algorithms/ppo.py`、`rsl_rl/storage/rollout_storage.py`、`rsl_rl/runners/on_policy_runner.py` | 保留的普通 PPO 链路；当前 Go2 不使用 |
| 各目录 `__init__.py` | 暴露类名，供训练任务注册器选择 |

机器人资产、奖励、命令和域随机化不在算法库里实现。导出工具也不在 runner 中，而在训练工程的 `quadruped_rl/utils/helpers.py`。

## 输入与网络

当前 12 关节四足任务的结构为：

```text
270维历史 = 最新帧在前的6帧 × 45维
estimator encoder: 270 -> 128 -> 64 -> 19
19维输出 = 速度3 + latent16；latent按L2归一化
actor: 当前45 + 估计速度3 + latent16 = 64维 -> 512 -> 256 -> 128 -> 12
critic: 238维单帧privileged obs -> 512 -> 256 -> 128 -> 1
```

actor 不从传感器直接读取真实机身线速度；训练时速度监督来自下一时刻 critic 的 `[45:48]` 缩放线速度。critic 的外力3维和高度187维在当前环境中仍是零占位，不能把维度存在等同于真实信号已接入。

部署只导出 encoder + actor。critic、target encoder、原型表、优化器及随机采样分布不进入导出模型，历史缓存由部署运行器维护。

## 训练与保存

- 默认每环境采样24个控制步；4096环境每次迭代采样98304个 transition。
- HIM runner 的 `--max_iterations` 是本次追加迭代数，恢复后继续累计编号。
- 周期保存间隔由 `runner.save_interval` 控制，当前为50；正常完成还保存末次 checkpoint。
- `Ctrl+C` 不额外保证即时保存；已写出的 checkpoint 才是恢复依据。
- loss 是训练诊断量，不等同于步态质量；解释见部署分支的 TensorBoard 指南。

当前 CLI 虽保留 `--horovod`，HIM runner 未提供正式分布式训练实现。本次文档整理不删除普通 PPO 源码，也不修改算法行为。



