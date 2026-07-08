"""Go2 任务的 episode 终止条件兼容封装。

实际实现放在四足通用 base 模块中，Go2 这里只保留旧函数名，避免已有导入失效。
"""

from quadruped_rl.envs.base.terminations import update_quadruped_termination

update_go2_termination = update_quadruped_termination
