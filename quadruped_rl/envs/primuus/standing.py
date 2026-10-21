"""停止后的正常站姿：关节、足端、高度和直立状态共同验收。"""

import math

import torch


class StandingRecovery:
    def __init__(self, cfg, nominal_feet, device):
        self.cfg = cfg
        self.joint_tolerance = torch.tensor(cfg.stand_joint_tolerance * 4, device=device)
        self.foot_tolerance = torch.tensor(cfg.stand_foot_tolerance, device=device)
        for value, shape in ((self.joint_tolerance, (12,)), (self.foot_tolerance, (2,))):
            if value.shape != shape or not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError('Invalid Primuus standing pose tolerances')
        for name in ('stand_pose_scale', 'stand_foot_scale', 'base_height_tolerance', 'stand_recovery_tilt'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Invalid Primuus standing recovery parameter: ' + name)
        self.nominal_feet = nominal_feet

    def update(self, joints, default_joints, feet_body, height, gravity_z):
        cfg = self.cfg
        joint_error = (joints - default_joints).abs()
        foot_error = (feet_body[:, :, :2] - self.nominal_feet[None, :, :2]).abs()
        joint_excess = (joint_error - self.joint_tolerance).clamp(min=0)
        foot_excess = (foot_error - self.foot_tolerance).clamp(min=0)
        height_error = (height - cfg.motion_height_targets[0]).abs()
        height_excess = (height_error - cfg.base_height_tolerance).clamp(min=0)
        self.pose_penalty = (torch.sqrt(1 + (joint_excess / cfg.stand_pose_scale).square()) - 1).mean(dim=1)
        self.foot_penalty = (foot_excess / cfg.stand_foot_scale).square().sum(dim=2).mean(dim=1)
        self.height_penalty = (height_excess / cfg.base_height_sigma).square()
        tilt = torch.acos((-gravity_z).clamp(-1, 1))
        self.normal_pose = ((joint_error <= self.joint_tolerance).all(dim=1) &
                            (foot_error <= self.foot_tolerance).all(dim=(1, 2)) &
                            (height_error <= cfg.base_height_tolerance) &
                            (tilt <= cfg.stand_recovery_tilt))
        # 仍需回正时弱化静止惩罚，允许腿继续调整；恢复后再逐渐要求安静。
        self.quiet_weight = torch.exp(-self.pose_penalty - self.foot_penalty - self.height_penalty)
        self.joint_error = joint_error.square().mean(dim=1).sqrt()
        self.foot_error = foot_error.amax(dim=(1, 2))
        self.height_error = height_error
