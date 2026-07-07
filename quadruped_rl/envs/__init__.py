from quadruped_rl.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from quadruped_rl.envs.go2.go2_env import Go2Env

from quadruped_rl.utils.task_registry import task_registry

task_registry.register("go2", Go2Env, GO2RoughCfg(), GO2RoughCfgPPO())
