"""50 Hz 指令参考；与部署 command_reference.py 保持数值一致。"""

import torch


def advance_reference(current, target, dt, cfg):
    """先退出高速直行包络，再引入侧移/转向；停车使用独立减速度。"""
    if not cfg.enabled:
        return target.clone()
    xy, requested = current[:, :2], target[:, :2]
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    target_speed = torch.linalg.vector_norm(requested, dim=1, keepdim=True)
    limit = cfg.transition_speed
    stop = torch.linalg.vector_norm(target, dim=1, keepdim=True) < 1e-7
    forward_only = (target[:, :1] > 0) & (target[:, 1:2].abs() < 1e-7) & (target[:, 2:3].abs() < 1e-7)
    entering_mixed = (~forward_only) & (speed > limit + 1e-6) & ~stop
    leaving_mixed = forward_only & (target_speed > limit) & ((current[:, 1:2].abs() > 1e-6) | (current[:, 2:3].abs() > 1e-6))
    intermediate = torch.where(entering_mixed, xy * (limit / speed.clamp(min=1e-7)), requested)
    intermediate = torch.where(leaving_mixed, requested * (limit / target_speed.clamp(min=1e-7)), intermediate)
    yaw_target = torch.where(entering_mixed | leaving_mixed, torch.zeros_like(target[:, 2:3]), target[:, 2:3])
    delta = intermediate - xy
    distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    slowing = (delta * xy).sum(dim=1, keepdim=True) < 0
    rate = torch.where(slowing, torch.full_like(speed, cfg.deceleration), torch.full_like(speed, cfg.acceleration))
    rate = torch.where(stop, torch.full_like(rate, cfg.braking_deceleration), rate)
    next_xy = xy + delta * (rate * dt / distance.clamp(min=1e-7)).clamp(max=1.0)
    yaw_rate = torch.where(stop, torch.full_like(speed, cfg.yaw_braking), torch.full_like(speed, cfg.yaw_acceleration))
    next_yaw = current[:, 2:3] + (yaw_target - current[:, 2:3]).clamp(min=-yaw_rate * dt, max=yaw_rate * dt)
    result = torch.cat((next_xy, next_yaw), dim=1)
    return torch.where(result.abs() < 1e-7, torch.zeros_like(result), result)
