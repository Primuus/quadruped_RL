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
        for name in ('stand_pose_scale', 'stand_foot_scale', 'base_height_tolerance', 'stand_recovery_tilt',
                     'stand_recovery_step_time'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Invalid Primuus standing recovery parameter: ' + name)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid Primuus standing recovery timestep')
        self.nominal_feet = nominal_feet
        self.missing_time = torch.zeros(num_envs, 4, device=device)
        self.support_grace = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.last_contacts = torch.ones_like(self.support_grace)
        self.last_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.support_penalty = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.missing_time[env_ids] = 0
        self.support_grace[env_ids] = False
        self.last_contacts[env_ids] = True
        self.last_active[env_ids] = False
        self.support_penalty[env_ids] = 0

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

    def update_support(self, contacts, standing_weight):
        missing = ~contacts
        active = standing_weight > 0
        three_feet = contacts.sum(dim=1) == 3
        self.missing_time.copy_(torch.where(contacts, 0.0, self.missing_time + self.dt))
        # 只在起脚或刚进入低速回正时授予单足宽限，不能在持续悬脚中反复刷新。
        start = missing & (self.last_contacts | (active & ~self.last_active)[:, None])
        grant = start & (active & three_feet & ~self.normal_pose)[:, None]
        # 已回到姿态容差内的在途换脚可完成落地；落地、超时、少于三足或离开回正阶段即失效。
        self.support_grace.copy_((self.support_grace | grant) & missing &
                                 (active & three_feet)[:, None] &
                                 (self.missing_time <= self.cfg.stand_recovery_step_time + 1e-6))
        self.support_penalty = (missing & ~self.support_grace).float().mean(dim=1)
        self.last_contacts.copy_(contacts)
        self.last_active.copy_(active)
