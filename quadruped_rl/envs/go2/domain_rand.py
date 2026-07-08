"""Go2 任务的随机化逻辑兼容封装。

实际实现放在四足通用 base 模块中，Go2 这里只保留旧函数名，避免已有导入失效。
"""

from quadruped_rl.envs.base.domain_rand import (
    apply_quadruped_safety_cfg,
    process_quadruped_rigid_body_props,
    process_quadruped_rigid_shape_props,
    push_quadruped_robots,
)

apply_go2_safety_cfg = apply_quadruped_safety_cfg
process_go2_rigid_body_props = process_quadruped_rigid_body_props
process_go2_rigid_shape_props = process_quadruped_rigid_shape_props
push_go2_robots = push_quadruped_robots
