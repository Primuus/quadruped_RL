"""Go2 任务的命令逻辑兼容封装。

实际实现放在四足通用 base 模块中，Go2 这里只保留旧函数名，避免已有导入失效。
"""

from quadruped_rl.envs.base.commands import (
    post_quadruped_command_step,
    resample_quadruped_commands,
    update_quadruped_command_curriculum,
)

post_go2_command_step = post_quadruped_command_step
resample_go2_commands = resample_quadruped_commands
update_go2_command_curriculum = update_quadruped_command_curriculum
