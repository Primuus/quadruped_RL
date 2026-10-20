from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ..commands import limit_velocity_command


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
    ) -> None:
        self.command_limits = np.asarray(command_limits, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(self.command_limits)) or np.any(self.command_limits <= 0.0):
            raise ValueError('command_limits must contain three finite positive values')
        self.max_translation_speed = float(max_translation_speed)
        if not np.isfinite(self.max_translation_speed) or self.max_translation_speed <= 0.0:
            raise ValueError('max_translation_speed must be finite and positive')

        self.bindings = {str(action): str(key) for action, key in bindings.items()}
        missing_actions = [action for action in REQUIRED_BINDING_ACTIONS if action not in self.bindings]
        if missing_actions:
            raise ValueError(f'Missing keyboard bindings: {", ".join(missing_actions)}')
        configured_keys = [self.bindings[action] for action in REQUIRED_BINDING_ACTIONS]
        if len(set(configured_keys)) != len(configured_keys):
            raise ValueError('Each keyboard action must use a unique key')

        self.speed_levels = np.asarray(speed_levels, dtype=np.float32)
        if self.speed_levels.shape != (3,):
            raise ValueError(f'speed_levels must have shape (3,), got {self.speed_levels.shape}')
        if not np.all(np.isfinite(self.speed_levels)) or np.any(self.speed_levels <= 0.0):
            raise ValueError('speed_levels must contain three finite positive values')
        if np.any(self.speed_levels > 1.0) or np.any(np.diff(self.speed_levels) <= 0.0):
            raise ValueError('speed_levels must be strictly increasing and no greater than 1.0')
        if default_speed_level not in (1, 2, 3):
            raise ValueError('default_speed_level must be 1, 2, or 3')

        self._key_to_action = {
            self.bindings[action]: action
            for action in REQUIRED_BINDING_ACTIONS
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
        self.current_command = limit_velocity_command(
            command,
            self.command_limits * self.speed_scale,
            self.max_translation_speed * self.speed_scale,
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
