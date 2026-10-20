from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ..commands import MotionCommandLimits, limit_velocity_command


REQUIRED_BINDING_ACTIONS = (
    'forward',
    'backward',
    'left',
    'right',
    'turn_left',
    'turn_right',
    'stop',
    'enable',
    'disable',
    'speed_low',
    'speed_medium',
    'speed_high',
    'reset_sim',
    'print_status',
)

DIRECTION_ACTIONS = (
    'forward',
    'backward',
    'left',
    'right',
    'turn_left',
    'turn_right',
)


def required_binding_actions(level_count: int) -> tuple[str, ...]:
    return REQUIRED_BINDING_ACTIONS + (('speed_max',) if level_count == 4 else ())


@dataclass(frozen=True, slots=True)
class KeyboardKeyEffect:
    reset_sim: bool = False
    reset_policy_state: bool = False
    print_status: bool = False
    exclusive_input: bool | None = None
    log_message: str | None = None


class KeyboardCommandController:
    """Convert a set of Linux key names into a three-axis velocity command."""

    def __init__(
        self,
        *,
        command_limits: np.ndarray,
        max_translation_speed: float,
        bindings: Mapping[str, str],
        speed_levels: np.ndarray,
        default_speed_level: int = 2,
        motion_limits: MotionCommandLimits | None = None,
        backward_speed_levels: np.ndarray | None = None,
    ) -> None:
        self.command_limits = np.asarray(command_limits, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(self.command_limits)) or np.any(self.command_limits <= 0.0):
            raise ValueError('command_limits must contain three finite positive values')
        self.max_translation_speed = float(max_translation_speed)
        if not np.isfinite(self.max_translation_speed) or self.max_translation_speed <= 0.0:
            raise ValueError('max_translation_speed must be finite and positive')

        self.speed_levels = np.asarray(speed_levels, dtype=np.float32)
        if self.speed_levels.ndim != 1 or self.speed_levels.size not in (3, 4):
            raise ValueError('speed_levels must contain three or four values')
        if not np.all(np.isfinite(self.speed_levels)) or np.any(self.speed_levels <= 0.0):
            raise ValueError('speed_levels must contain finite positive values')
        if np.any(self.speed_levels > 1.0) or np.any(np.diff(self.speed_levels) <= 0.0):
            raise ValueError('speed_levels must be strictly increasing and no greater than 1.0')
        if not 1 <= default_speed_level <= len(self.speed_levels):
            raise ValueError('default_speed_level must refer to a configured speed level')
        self.motion_limits = motion_limits
        if motion_limits is not None:
            if motion_limits.backward > self.command_limits[0] or max(motion_limits.diagonal, motion_limits.moving_turn) > self.max_translation_speed:
                raise ValueError('motion_limits must fit command_limits and max_translation_speed')
        self.backward_speed_levels = None
        if backward_speed_levels is not None:
            values = np.asarray(backward_speed_levels, dtype=np.float32)
            if motion_limits is None:
                raise ValueError('backward_speed_levels requires motion_limits')
            if values.shape != self.speed_levels.shape or not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(np.diff(values) < 0.0):
                raise ValueError('backward_speed_levels must be positive, nondecreasing, and match speed_levels')
            if np.any(values > motion_limits.backward) or np.any(values > self.command_limits[0] * self.speed_levels):
                raise ValueError('backward_speed_levels exceed the configured backward/gear limits')
            self.backward_speed_levels = values

        self.bindings = {str(action): str(key) for action, key in bindings.items()}
        actions = required_binding_actions(len(self.speed_levels))
        missing_actions = [action for action in actions if action not in self.bindings]
        if missing_actions:
            raise ValueError(f'Missing keyboard bindings: {", ".join(missing_actions)}')
        configured_keys = [self.bindings[action] for action in actions]
        if len(set(configured_keys)) != len(configured_keys):
            raise ValueError('Each keyboard action must use a unique key')
        if not all(key.startswith('KEY_') for key in configured_keys):
            raise ValueError('keyboard bindings must use Linux evdev KEY_* names')

        self._key_to_action = {
            self.bindings[action]: action
            for action in actions
        }
        self._direction_keys = frozenset(self.bindings[action] for action in DIRECTION_ACTIONS)
        self._speed_level = int(default_speed_level)
        self._stop_until_directions_released = False
        self.enabled = False
        self.current_command = np.zeros(3, dtype=np.float32)

    @property
    def tracked_keys(self) -> frozenset[str]:
        return frozenset(self._key_to_action)

    @property
    def speed_level(self) -> int:
        return self._speed_level

    @property
    def speed_scale(self) -> float:
        return float(self.speed_levels[self._speed_level - 1])

    def disable(self) -> None:
        self.enabled = False
        self._stop_until_directions_released = False
        self.current_command.fill(0.0)

    def apply_key(self, key: str) -> KeyboardKeyEffect:
        action = self._key_to_action.get(str(key))
        if action is None or action in DIRECTION_ACTIONS:
            return KeyboardKeyEffect()

        if action == 'enable':
            was_enabled = self.enabled
            self.enabled = True
            self._stop_until_directions_released = False
            self.current_command.fill(0.0)
            return KeyboardKeyEffect(
                reset_policy_state=not was_enabled,
                exclusive_input=True,
                log_message='keyboard control enabled',
            )
        if action == 'disable':
            self.disable()
            return KeyboardKeyEffect(
                exclusive_input=False,
                log_message='keyboard control disabled; command cleared',
            )
        if action == 'stop':
            self._stop_until_directions_released = True
            self.current_command.fill(0.0)
            return KeyboardKeyEffect(log_message='keyboard motion command stopped')
        if action == 'reset_sim':
            self._stop_until_directions_released = True
            self.current_command.fill(0.0)
            return KeyboardKeyEffect(reset_sim=True, log_message='keyboard requested simulation reset')
        if action == 'print_status':
            return KeyboardKeyEffect(print_status=True)

        speed_level_by_action = {
            'speed_low': 1,
            'speed_medium': 2,
            'speed_high': 3,
            'speed_max': 4,
        }
        self._speed_level = speed_level_by_action[action]
        return KeyboardKeyEffect(
            log_message=(
                f'keyboard speed level {self._speed_level}: '
                f'{self.speed_scale * 100.0:.0f}%'
            )
        )

    def update_pressed_keys(self, pressed_keys: Iterable[str]) -> np.ndarray:
        pressed = frozenset(str(key) for key in pressed_keys)

        if self.bindings['stop'] in pressed:
            self._stop_until_directions_released = True

        if self._stop_until_directions_released:
            self.current_command.fill(0.0)
            if not (pressed & self._direction_keys) and self.bindings['stop'] not in pressed:
                self._stop_until_directions_released = False
            return self.current_command.copy()

        if not self.enabled:
            self.current_command.fill(0.0)
            return self.current_command.copy()

        axis = np.array(
            [
                float(self.bindings['forward'] in pressed)
                - float(self.bindings['backward'] in pressed),
                float(self.bindings['left'] in pressed)
                - float(self.bindings['right'] in pressed),
                float(self.bindings['turn_left'] in pressed)
                - float(self.bindings['turn_right'] in pressed),
            ],
            dtype=np.float32,
        )
        command = axis * self.command_limits * self.speed_scale
        limits = None
        if self.motion_limits is not None:
            backward = (
                float(self.backward_speed_levels[self._speed_level - 1])
                if self.backward_speed_levels is not None else None
            )
            limits = self.motion_limits.scaled(self.speed_scale, backward=backward)
            if axis[0] < 0.0:
                command[0] = -limits.backward
            if axis[0] != 0.0 and axis[1] != 0.0:
                # Two direction keys denote a 45-degree bearing, even with a faster forward range.
                command[:2] = axis[:2] * (limits.diagonal / np.sqrt(2.0))
        self.current_command = limit_velocity_command(
            command,
            self.command_limits * self.speed_scale,
            self.max_translation_speed * self.speed_scale,
            limits,
        )
        return self.current_command.copy()

    def describe(self, pressed_keys: Iterable[str] = ()) -> str:
        pressed = sorted(str(key) for key in pressed_keys)
        command = ', '.join(f'{value:.3f}' for value in self.current_command)
        mode = 'enabled' if self.enabled else 'disabled'
        keys = ','.join(pressed) if pressed else 'none'
        return (
            f'keyboard {mode}: speed_level={self._speed_level} '
            f'({self.speed_scale * 100.0:.0f}%), keys=[{keys}], command=[{command}]'
        )
