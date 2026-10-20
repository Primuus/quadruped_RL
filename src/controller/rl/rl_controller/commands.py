"""Velocity command limits shared by simulation, keyboard and real deployment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionCommandLimits:
    """Limits for motions whose training range is smaller than straight forward."""

    backward: float
    diagonal: float
    moving_turn: float

    def __post_init__(self) -> None:
        for name in ('backward', 'diagonal', 'moving_turn'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f'motion_limits.{name} must be finite and positive')

    @classmethod
    def from_mapping(cls, value: object) -> 'MotionCommandLimits | None':
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {'backward', 'diagonal', 'moving_turn'}:
            raise ValueError('motion_limits must contain backward, diagonal, and moving_turn')
        return cls(**{name: float(number) for name, number in value.items()})

    def scaled(self, scale: float, *, backward: float | None = None) -> 'MotionCommandLimits':
        return MotionCommandLimits(
            self.backward * scale if backward is None else backward,
            self.diagonal * scale,
            self.moving_turn * scale,
        )


def limit_velocity_command(
    command: object,
    command_limits: np.ndarray,
    max_translation_speed: float,
    motion_limits: MotionCommandLimits | None = None,
) -> np.ndarray:
    """Limit XY without changing its direction; limit yaw independently.

    Axis limits and the translation limit must be finite and positive, as
    checked by the startup configuration or keyboard controller.
    """
    value = np.asarray(command, dtype=np.float32)
    if value.shape != (3,):
        raise ValueError(f'velocity command must have shape (3,), got {value.shape}')
    if not np.all(np.isfinite(value)):
        raise ValueError('velocity command must contain only finite values')

    vx, vy = float(value[0]), float(value[1])
    x_limit = float(command_limits[0])
    translation_limit = max_translation_speed
    if motion_limits is not None:
        if vx < 0.0:
            x_limit = min(x_limit, motion_limits.backward)
        if vx != 0.0 and vy != 0.0:
            translation_limit = min(translation_limit, motion_limits.diagonal)
        if value[2] != 0.0:
            translation_limit = min(translation_limit, motion_limits.moving_turn)
    scale = max(
        1.0,
        abs(vx) / x_limit,
        abs(vy) / float(command_limits[1]),
        math.hypot(vx, vy) / translation_limit,
    )
    limited = value.copy()
    limited[0] = vx / scale
    limited[1] = vy / scale
    limited[2] = np.clip(value[2], -command_limits[2], command_limits[2])
    return limited
