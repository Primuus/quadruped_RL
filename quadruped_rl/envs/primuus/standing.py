"""停止后的正常站姿：关节、足端、高度和直立状态共同验收。"""

import math

import torch


class StandingRecovery:
    def __init__(self, cfg, nominal_feet, num_envs, device, dt):
        self.cfg, self.dt = cfg, dt
        self.joint_tolerance = torch.tensor(cfg.stand_joint_tolerance * 4, device=device)
        self.foot_tolerance = torch.tensor(cfg.stand_foot_tolerance, device=device)
        for value, shape in ((self.joint_tolerance, (12,)), (self.foot_tolerance, (2,))):
            if value.shape != shape or not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError('Invalid Primuus standing pose tolerances')
        for name in ('base_height_tolerance', 'stand_recovery_tilt'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Invalid Primuus standing recovery parameter: ' + name)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid Primuus standing recovery timestep')
        self.nominal_feet = nominal_feet
        # 保留原有日志接口；基础四足支撑奖励不提供换脚宽限。
        self.support_grace = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        self.support_grace[env_ids] = False

    def update(self, joints, default_joints, feet_body, height, gravity_z):
        cfg = self.cfg
        joint_error = (joints - default_joints).abs()
        foot_error = (feet_body[:, :, :2] - self.nominal_feet[None, :, :2]).abs()
        height_error = (height - cfg.motion_height_targets[0]).abs()
        tilt = torch.acos((-gravity_z).clamp(-1, 1))
        self.normal_pose = ((joint_error <= self.joint_tolerance).all(dim=1) &
                            (foot_error <= self.foot_tolerance).all(dim=(1, 2)) &
                            (height_error <= cfg.base_height_tolerance) &
                            (tilt <= cfg.stand_recovery_tilt))
        self.joint_error = joint_error.square().mean(dim=1).sqrt()
        self.foot_error = foot_error.amax(dim=(1, 2))
        self.height_error = height_error
