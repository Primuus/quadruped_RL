from quadruped_rl.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from quadruped_rl.envs.go2.go2_env import Go2Env

from quadruped_rl.utils.task_registry import task_registry

# 当前只注册 Go2 这个机器人实例。新增四足机器人时，在 envs/<robot>/ 中新增配置/适配器，
# 再在这里注册新的 task；HIM-PPO 算法和 base 环境逻辑保持复用。
task_registry.register("go2", Go2Env, GO2RoughCfg(), GO2RoughCfgPPO())
