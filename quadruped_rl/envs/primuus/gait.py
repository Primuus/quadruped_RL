"""Primuus 足端活动区域与步态测量；平地使用，不规定四腿的相位。"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import torch


LEGS = ("FL", "FR", "RL", "RR")
QUALITY_FIELDS = (
    "foot_region_violation", "foot_slip_speed", "tilt",
    "target_violation", "clearance_violation", "height_error",
    "contact_fl", "contact_fr", "contact_rl", "contact_rr",
)


def _smoothstep(value):
    weight = value.clamp(0, 1)
    return weight.square() * (3 - 2 * weight)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def nominal_geometry(urdf_path, joint_angles, joint_names):
    """从正式 URDF 的默认角计算足端原点及硬限位，包含固定关节变换。"""
    joints = ET.parse(urdf_path).getroot().findall('joint')
    transforms = {'base_link': np.eye(4)}
    limits = {}
    pending = list(joints)
    while pending:
        remaining = []
        for joint in pending:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in transforms:
                remaining.append(joint)
                continue
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (_rotation([0, 0, 1], rpy[2]) @
                                     _rotation([0, 1, 0], rpy[1]) @ _rotation([1, 0, 0], rpy[0]))
            if joint.get('type') != 'fixed':
                name = joint.get('name')
                if joint.get('type') != 'revolute' or name not in joint_angles:
                    raise ValueError('Unsupported or unspecified Primuus joint: ' + name)
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] = transform[:3, :3] @ _rotation(axis, joint_angles[name])
                limit = joint.find('limit')
                limits[name] = [float(limit.get('lower')), float(limit.get('upper'))]
            transforms[child] = transforms[parent] @ transform
        if len(remaining) == len(pending):
            raise ValueError('Primuus URDF must form a connected tree rooted at base_link')
        pending = remaining
    feet = np.stack([transforms[leg + '_foot_link'][:3, 3] for leg in LEGS])
    return feet, np.array([limits[name] for name in joint_names])


class GaitQuality:
    def __init__(self, cfg, feet, limits, num_envs, device, dt):
        self.cfg, self.dt = cfg, dt
        for name in vars(type(cfg)):
            value = getattr(cfg, name)
            if not name.startswith('_') and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError('gait.%s must be finite and positive' % name)
        if (not cfg.clearance_min <= cfg.clearance_motion_height < cfg.clearance_max or
                cfg.region_tolerance >= cfg.lateral_margin):
            raise ValueError('Invalid Primuus clearance interval or foot region tolerance')
        self.nominal = torch.as_tensor(feet, dtype=torch.float, device=device)
        self.limits = torch.as_tensor(limits, dtype=torch.float, device=device)
        if self.nominal.shape != (4, 3) or self.limits.shape != (12, 2):
            raise ValueError('Primuus gait geometry requires four feet and twelve joint limits')
        if not torch.isfinite(self.nominal).all() or not torch.isfinite(self.limits).all() or (self.limits[:, 0] >= self.limits[:, 1]).any():
            raise ValueError('Invalid Primuus gait geometry')
        self.center = self.nominal[None, :, :2].repeat(num_envs, 1, 1)
        self.ground = torch.zeros(num_envs, 4, device=device)
        self.ground_seen = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.air_time = torch.zeros_like(self.ground)
        self.swing_peak = torch.zeros_like(self.ground)
        self.placement_penalty = torch.zeros(num_envs, device=device)
        self.region_penalty = torch.zeros_like(self.placement_penalty)
        self.crossing_penalty = torch.zeros_like(self.placement_penalty)
        self.clearance_penalty = torch.zeros_like(self.placement_penalty)
        self.direction_penalty = torch.zeros_like(self.placement_penalty)
        self.target_penalty = torch.zeros_like(self.placement_penalty)

    def reset(self, env_ids):
        self.center[env_ids] = self.nominal[:, :2]
        self.ground[env_ids] = 0
        self.ground_seen[env_ids] = False
        self.air_time[env_ids] = 0
        self.swing_peak[env_ids] = 0

    def update(self, feet_body, feet_z, feet_velocity, contacts, commands, targets, gravity_z, height_error,
               feet_body_velocity, raw_contacts=None):
        cfg = self.cfg
        speed = torch.linalg.vector_norm(commands[:, :2], dim=1)
        mixed = torch.maximum(commands[:, 1].abs() / cfg.mixed_lateral_speed,
                              commands[:, 2].abs() / cfg.mixed_yaw_speed).clamp(max=1)
        self.center.lerp_(feet_body[:, :, :2], self.dt / (cfg.center_time_constant + self.dt))
        offset = feet_body[:, :, :2] - self.nominal[None, :, :2]
        # 高速前后步幅开放；侧移/转向放宽横向走廊。所有尺寸都使用米。
        x_margin = cfg.stance_x_margin + cfg.stride_time * speed[:, None] + (~contacts) * cfg.swing_x_margin
        y_margin = cfg.lateral_margin + cfg.mixed_lateral_margin * mixed[:, None]
        margins = torch.stack((x_margin, y_margin.expand_as(x_margin)), dim=2)
        excess = (offset.abs() - margins).clamp(min=0)
        side = self.nominal[:, 1].sign()
        crossing = (cfg.min_lateral_distance - feet_body[:, :, 1] * side).clamp(min=0)
        region = (excess.amax(dim=2) > cfg.region_tolerance) | (crossing > cfg.region_tolerance)
        center_margin = torch.stack((cfg.center_x_tolerance + cfg.center_speed_tolerance * speed,
                                     cfg.center_y_tolerance + cfg.mixed_lateral_margin * mixed), dim=1)
        center_excess = (self.center - self.nominal[None, :, :2]).abs().sub(center_margin[:, None]).clamp(min=0)
        # 活动区域/平均中心在停车时交接给站姿区域；跨中线约束始终保留。
        self.region_penalty = ((excess / cfg.placement_scale).square().sum(dim=2) +
                               (center_excess / cfg.placement_scale).square().sum(dim=2)).mean(dim=1)
        self.crossing_penalty = (crossing / cfg.placement_scale).square().mean(dim=1)
        self.placement_penalty = self.region_penalty + self.crossing_penalty

        # 对地高度配合对地水平速度；换脚时持续检查，收住水平速度后允许自然落地。
        raw_contacts = contacts if raw_contacts is None else raw_contacts
        self.ground.copy_(torch.where(raw_contacts, feet_z, self.ground))
        self.ground_seen |= raw_contacts
        self.air_time.copy_(torch.where(contacts, 0.0, self.air_time + self.dt))
        clearance = (feet_z - self.ground).clamp(min=0)
        self.swing_peak.copy_(torch.where(contacts, 0.0, torch.maximum(self.swing_peak, clearance)))
        maximum = cfg.clearance_max + cfg.clearance_speed_scale * speed[:, None]
        minimum = (self.air_time >= cfg.clearance_min_air_time) * cfg.clearance_min
        swing = ~contacts & self.ground_seen & ((speed + commands[:, 2].abs())[:, None] > 0.1)
        horizontal_speed = torch.linalg.vector_norm(feet_velocity[:, :, :2], dim=2)
        motion_height = (cfg.clearance_motion_height * _smoothstep(horizontal_speed / cfg.clearance_motion_speed) *
                         _smoothstep(self.air_time / cfg.clearance_min_air_time))
        motion_low = (motion_height - clearance).clamp(min=0)
        # 峰值合格后，低空快速换脚仍有代价；起脚平滑增强，水平减速时平滑放松。
        low = torch.maximum((minimum - self.swing_peak).clamp(min=0), motion_low)
        high = (clearance - maximum).clamp(min=0)
        self.clearance_penalty = (((low + high) / cfg.clearance_scale).square() * swing).mean(dim=1)

        # 每足的换脚方向随 v_xy + yaw * [-y, x] 变化，不给转向套用直行方向。
        tangent = torch.stack((-feet_body[:, :, 1], feet_body[:, :, 0]), dim=2)
        desired = commands[:, None, :2] + commands[:, None, 2:3] * tangent
        desired_speed = torch.linalg.vector_norm(desired, dim=2)
        direction = desired / desired_speed.clamp(min=1e-6)[:, :, None]
        perpendicular = (feet_body_velocity[:, :, 0] * direction[:, :, 1] -
                         feet_body_velocity[:, :, 1] * direction[:, :, 0]).abs()
        direction_excess = (perpendicular - cfg.swing_direction_tolerance).clamp(min=0)
        direction_weight = _smoothstep(desired_speed / cfg.swing_direction_speed)
        self.direction_penalty = ((direction_excess / cfg.swing_direction_scale).square() *
                                  direction_weight * swing).mean(dim=1)
        # 只惩罚不可达目标，不在训练侧加入部署中没有的目标硬裁剪。
        target_excess = (self.limits[:, 0] - targets).clamp(min=0) + (targets - self.limits[:, 1]).clamp(min=0)
        self.target_penalty = (target_excess / cfg.target_scale).square().mean(dim=1)
        slip = torch.linalg.vector_norm(feet_velocity[:, :, :2], dim=2)
        self.measurements = {
            'foot_region_violation': region.float().mean(dim=1),
            'center_error': (self.center - self.nominal[None, :, :2]).abs().mean(dim=(1, 2)),
            'foot_slip_speed': (slip * raw_contacts).sum(dim=1) / raw_contacts.sum(dim=1).clamp(min=1),
            'tilt': torch.acos((-gravity_z).clamp(-1, 1)),
            'target_violation': (target_excess > cfg.target_tolerance).float().mean(dim=1),
            'clearance_violation': (((low + high) > cfg.region_tolerance) & swing).float().mean(dim=1),
            'clearance_motion_violation': ((motion_low > cfg.region_tolerance) & swing).float().mean(dim=1),
            'height_error': height_error,
            'swing_clearance': (clearance * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1),
            'swing_height_target': (motion_height * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1),
            'swing_horizontal_speed': (horizontal_speed * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1),
            'swing_direction_error': (direction_excess * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1),
        }

    def curriculum_values(self, contacts):
        values = [self.measurements[name] for name in QUALITY_FIELDS[:-4]]
        return torch.cat((torch.stack(values, dim=1), contacts.float()), dim=1)
