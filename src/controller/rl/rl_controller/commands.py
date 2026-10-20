"""Velocity command limits shared by simulation, keyboard and real deployment."""

from __future__ import annotations

import math

import numpy as np


def limit_velocity_command(
    command: object,
    command_limits: np.ndarray,
    max_translation_speed: float,
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
    scale = max(
        1.0,
        abs(vx) / float(command_limits[0]),
        abs(vy) / float(command_limits[1]),
        math.hypot(vx, vy) / max_translation_speed,
    )
    limited = value.copy()
    limited[0] = vx / scale
    limited[1] = vy / scale
    limited[2] = np.clip(value[2], -command_limits[2], command_limits[2])
    return limited
