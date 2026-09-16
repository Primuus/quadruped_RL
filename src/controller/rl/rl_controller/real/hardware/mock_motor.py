from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

import numpy as np

from ..config import RealMotorConfig
from ..types import JointCommand, JointState
from .base import BackendCommunicationError, BackendError


class MockMotorBackend:
    """Deterministic joint-space motor plant for offline deployment tests."""

    def __init__(
        self,
        config: RealMotorConfig,
        *,
        initial_position: Any | None = None,
        max_velocity_rad_s: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if config.backend != 'mock':
            raise ValueError('MockMotorBackend requires motors.backend=mock')
        if not config.joints:
            raise ValueError('MockMotorBackend requires at least one joint')

        self.config = config
        self._num_joints = len(config.joints)
        self._clock = clock
        self._nominal_dt = 1.0 / config.io_frequency_hz
        self._position_limits = np.asarray(
            [joint.position_limits_rad for joint in config.joints],
            dtype=np.float32,
        )
        self._max_velocity = self._make_vector(
            max_velocity_rad_s
            if max_velocity_rad_s is not None
            else [
                joint.max_target_step_rad * config.io_frequency_hz
                for joint in config.joints
            ],
            'max_velocity_rad_s',
            positive=True,
        )

        if initial_position is None:
            initial_position = np.zeros(self._num_joints, dtype=np.float32)
        position = self._make_vector(initial_position, 'initial_position')
        if np.any(position < self._position_limits[:, 0]) or np.any(
            position > self._position_limits[:, 1]
        ):
            raise ValueError('initial_position must be inside every joint limit')

        self._position = position
        self._velocity = np.zeros(self._num_joints, dtype=np.float32)
        self._torque = np.zeros(self._num_joints, dtype=np.float32)
        self._temperature = np.full(self._num_joints, 25.0, dtype=np.float32)
        self._error = np.zeros(self._num_joints, dtype=np.int64)

        self._lock = threading.Lock()
        self._is_open = False
        self._safe_stopped = True
        self._last_update_time: float | None = None

        self._stale_age_s = 0.0
        self._nan_fields: set[tuple[str, int]] = set()
        self._temperature_fault: tuple[float, int | None] | None = None
        self._error_fault: tuple[int, int | None] | None = None
        self._invalid_state = False
        self._communication_error: str | None = None

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._is_open

    @property
    def safe_stopped(self) -> bool:
        with self._lock:
            return self._safe_stopped

    def open(self) -> None:
        now = self._read_clock()
        with self._lock:
            if self._is_open:
                return
            self._velocity.fill(0.0)
            self._torque.fill(0.0)
            self._last_update_time = now
            self._safe_stopped = False
            self._is_open = True

    def cycle(self, command: JointCommand) -> JointState:
        if not isinstance(command, JointCommand):
            raise TypeError('command must be a JointCommand')
        if command.num_joints != self._num_joints:
            raise ValueError(
                f'command has {command.num_joints} joints, expected {self._num_joints}'
            )
        if not all(
            np.all(np.isfinite(vector))
            for vector in (
                command.position,
                command.velocity,
                command.kp,
                command.kd,
                command.torque,
            )
        ):
            raise BackendError('Mock motor command must contain only finite values')

        now = self._read_clock()
        with self._lock:
            self._require_open()
            if self._communication_error is not None:
                raise BackendCommunicationError(self._communication_error)
            previous_time = self._last_update_time
            dt = self._nominal_dt if previous_time is None else now - previous_time
            if dt < 0.0:
                raise BackendCommunicationError('Mock motor clock moved backwards')

            if self._safe_stopped:
                self._velocity.fill(0.0)
                self._torque.fill(0.0)
            else:
                target = np.clip(
                    command.position,
                    self._position_limits[:, 0],
                    self._position_limits[:, 1],
                )
                if dt > 0.0:
                    max_step = self._max_velocity * dt
                    step = np.clip(target - self._position, -max_step, max_step)
                    self._position = (self._position + step).astype(np.float32)
                    self._velocity = (step / dt).astype(np.float32)
                else:
                    self._velocity.fill(0.0)

                self._torque = (
                    command.kp * (target - self._position)
                    + command.kd * (command.velocity - self._velocity)
                    + command.torque
                ).astype(np.float32)

            self._last_update_time = now
            return self._make_state(now)

    def safe_stop(self) -> None:
        with self._lock:
            self._velocity.fill(0.0)
            self._torque.fill(0.0)
            self._safe_stopped = True

    def close(self) -> None:
        with self._lock:
            self._velocity.fill(0.0)
            self._torque.fill(0.0)
            self._safe_stopped = True
            self._is_open = False
            self._last_update_time = None

    def inject_stale_state(self, age_s: float) -> None:
        age = float(age_s)
        if not np.isfinite(age) or age < 0.0:
            raise ValueError('age_s must be finite and non-negative')
        with self._lock:
            self._stale_age_s = age

    def inject_nan(self, field: str = 'position', joint_index: int = 0) -> None:
        if field not in {'position', 'velocity', 'torque', 'temperature'}:
            raise ValueError(
                'field must be position, velocity, torque, or temperature'
            )
        index = self._validate_joint_index(joint_index)
        with self._lock:
            self._nan_fields.add((field, index))

    def inject_temperature(
        self,
        temperature_c: float,
        joint_index: int | None = None,
    ) -> None:
        temperature = float(temperature_c)
        if not np.isfinite(temperature):
            raise ValueError('temperature_c must be finite')
        index = (
            None
            if joint_index is None
            else self._validate_joint_index(joint_index)
        )
        with self._lock:
            self._temperature_fault = (temperature, index)

    def inject_error(self, error_code: int, joint_index: int | None = None) -> None:
        if isinstance(error_code, bool) or not isinstance(error_code, int):
            raise ValueError('error_code must be an integer')
        if error_code == 0:
            raise ValueError('error_code must be non-zero; use clear_faults() to clear it')
        index = (
            None
            if joint_index is None
            else self._validate_joint_index(joint_index)
        )
        with self._lock:
            self._error_fault = (error_code, index)

    def inject_invalid_state(self) -> None:
        with self._lock:
            self._invalid_state = True

    def inject_communication_error(
        self,
        message: str = 'injected Mock motor communication failure',
    ) -> None:
        reason = str(message).strip()
        if not reason:
            raise ValueError('communication error message must not be empty')
        with self._lock:
            self._communication_error = reason

    def clear_faults(self) -> None:
        with self._lock:
            self._stale_age_s = 0.0
            self._nan_fields.clear()
            self._temperature_fault = None
            self._error_fault = None
            self._invalid_state = False
            self._communication_error = None

    def _make_state(self, now: float) -> JointState:
        position = self._position.copy()
        velocity = self._velocity.copy()
        torque = self._torque.copy()
        temperature = self._temperature.copy()
        error = self._error.copy()
        if self._temperature_fault is not None:
            value, index = self._temperature_fault
            if index is None:
                temperature.fill(value)
            else:
                temperature[index] = value
        if self._error_fault is not None:
            value, index = self._error_fault
            if index is None:
                error.fill(value)
            else:
                error[index] = value

        float_fields = {
            'position': position,
            'velocity': velocity,
            'torque': torque,
            'temperature': temperature,
        }
        for field, index in self._nan_fields:
            float_fields[field][index] = np.nan

        return JointState(
            position=position,
            velocity=velocity,
            torque=torque,
            temperature=temperature,
            error=error,
            valid=not self._invalid_state,
            timestamp=max(0.0, now - self._stale_age_s),
        )

    def _make_vector(
        self,
        value: Any,
        field_name: str,
        *,
        positive: bool = False,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full(self._num_joints, float(array), dtype=np.float32)
        if array.shape != (self._num_joints,):
            raise ValueError(
                f'{field_name} must have shape ({self._num_joints},), '
                f'got {array.shape}'
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{field_name} must contain only finite values')
        if positive and np.any(array <= 0.0):
            raise ValueError(f'{field_name} must contain only positive values')
        return np.array(array, dtype=np.float32, copy=True)

    def _validate_joint_index(self, joint_index: int) -> int:
        if isinstance(joint_index, bool) or not isinstance(joint_index, int):
            raise ValueError('joint_index must be an integer')
        if not 0 <= joint_index < self._num_joints:
            raise ValueError(
                f'joint_index must be in [0, {self._num_joints - 1}]'
            )
        return joint_index

    def _require_open(self) -> None:
        if not self._is_open:
            raise BackendError('Mock motor backend is not open')

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise BackendCommunicationError(
                'Mock motor clock must return a finite, non-negative value'
            )
        return timestamp
