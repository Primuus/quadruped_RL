"""指令过渡参考；与 Primuus 训练的 reference.py 使用同一算法和秒制参数。"""

from dataclasses import dataclass, fields
from collections.abc import Mapping
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class CommandTransitionConfig:
    enabled: bool = False
    acceleration: float = 3.0
    deceleration: float = 4.0
    braking_deceleration: float = 6.0
    yaw_acceleration: float = 3.0
    yaw_braking: float = 4.0
    transition_speed: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('command_transition.enabled must be boolean')
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != 'enabled' and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'command_transition.{field.name} must be finite and positive')

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError('Invalid command_transition mapping')
        return cls(**value)


class CommandReference:
    def __init__(self, config: CommandTransitionConfig):
        self.config = config
        self.value = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.value.fill(0.0)

    def advance(self, target, dt):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (3,) or not np.isfinite(target).all() or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Reference requires a finite three-axis target and positive dt')
        cfg = self.config
        if not cfg.enabled:
            self.value = target.copy()
            return self.value.copy()
        xy = self.value[:2]
        requested = target[:2]
        speed = float(np.linalg.norm(xy))
        target_speed = float(np.linalg.norm(requested))
        limit = cfg.transition_speed
        stop = float(np.linalg.norm(target)) < 1e-7
        forward_only = target[0] > 0 and abs(target[1]) < 1e-7 and abs(target[2]) < 1e-7
        entering = not forward_only and speed > limit + 1e-6 and not stop
        leaving = forward_only and target_speed > limit and (abs(self.value[1]) > 1e-6 or abs(self.value[2]) > 1e-6)
        intermediate = xy * (limit / max(speed, 1e-7)) if entering else requested
        if leaving:
            intermediate = requested * (limit / max(target_speed, 1e-7))
        yaw_target = 0.0 if entering or leaving else float(target[2])
        delta = intermediate - xy
        rate = cfg.deceleration if float(np.dot(delta, xy)) < 0 else cfg.acceleration
        if stop:
            rate = cfg.braking_deceleration
        next_xy = xy + delta * min(1.0, rate * dt / max(float(np.linalg.norm(delta)), 1e-7))
        yaw_rate = cfg.yaw_braking if stop else cfg.yaw_acceleration
        next_yaw = self.value[2] + np.clip(yaw_target - self.value[2], -yaw_rate * dt, yaw_rate * dt)
        self.value = np.array([*next_xy, next_yaw], dtype=np.float32)
        self.value[np.abs(self.value) < 1e-7] = 0.0
        return self.value.copy()
