from __future__ import annotations

from collections.abc import Mapping
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from .config import RealDeploymentConfig
from .state_machine import RuntimeState
from .types import ImuState, JointCommand, JointState, RealRobotState


class SafetyViolationError(RuntimeError):
    """Raised when a command, state or control deadline is unsafe."""


class RealSafetyManager:
    """Validate every real-robot input and output against one config."""

    _COMMAND_STATES = frozenset(
        {
            RuntimeState.INIT,
            RuntimeState.READY,
            RuntimeState.MOVE_TO_DEFAULT,
            RuntimeState.RL,
        }
    )
    _ACTIVE_COMMAND_STATES = frozenset(
        {RuntimeState.MOVE_TO_DEFAULT, RuntimeState.RL}
    )

    def __init__(
        self,
        config: RealDeploymentConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        self.config = config
        self._clock = clock
        self._num_joints = config.robot.num_actions
        self._position_min = np.asarray(
            [joint.position_min_rad for joint in config.motors.joints],
            dtype=np.float32,
        )
        self._position_max = np.asarray(
            [joint.position_max_rad for joint in config.motors.joints],
            dtype=np.float32,
        )
        self._max_target_step = np.asarray(
            [joint.max_target_step_rad for joint in config.motors.joints],
            dtype=np.float32,
        )
        self._lock = threading.Lock()
        self._last_active_target: np.ndarray | None = None

    @property
    def last_active_target(self) -> np.ndarray | None:
        with self._lock:
            if self._last_active_target is None:
                return None
            return self._last_active_target.copy()

    def reset(self) -> None:
        with self._lock:
            self._last_active_target = None

    def validate_action(self, action: Any) -> np.ndarray:
        value = self._finite_vector(action, self._num_joints, 'policy action')
        limit = self.config.policy.clip_actions
        violating = np.flatnonzero(np.abs(value) > limit)
        if violating.size:
            index = int(violating[0])
            raise SafetyViolationError(
                f'policy action[{index}]={value[index]:.6f} exceeds '
                f'configured limit +/-{limit:.6f}'
            )
        return value

    def validate_target_position(self, target: Any) -> np.ndarray:
        value = self._finite_vector(
            target,
            self._num_joints,
            'target position',
        )
        violating = np.flatnonzero(
            (value < self._position_min) | (value > self._position_max)
        )
        if violating.size:
            index = int(violating[0])
            joint_name = self.config.robot.joint_names[index]
            raise SafetyViolationError(
                f'target position for {joint_name} is {value[index]:.6f} rad; '
                f'limits are [{self._position_min[index]:.6f}, '
                f'{self._position_max[index]:.6f}] rad'
            )
        return value

    def validate_command_before_send(
        self,
        command: JointCommand,
        runtime_state: RuntimeState,
        robot_state: RealRobotState | None,
    ) -> None:
        if not isinstance(command, JointCommand):
            raise TypeError('command must be a JointCommand')
        if not isinstance(runtime_state, RuntimeState):
            raise TypeError('runtime_state must be a RuntimeState')
        if runtime_state not in self._COMMAND_STATES:
            raise SafetyViolationError(
                f'joint commands are forbidden in {runtime_state.value}'
            )
        if command.num_joints != self._num_joints:
            raise SafetyViolationError(
                f'command has {command.num_joints} joints, '
                f'expected {self._num_joints}'
            )

        target = self.validate_target_position(command.position)
        for field_name, value in (
            ('velocity', command.velocity),
            ('kp', command.kp),
            ('kd', command.kd),
            ('torque', command.torque),
        ):
            self._finite_vector(value, self._num_joints, field_name)
        if np.any(command.kp < 0.0) or np.any(command.kd < 0.0):
            raise SafetyViolationError('kp and kd must not be negative')

        active = any(
            np.any(value != 0.0)
            for value in (
                command.velocity,
                command.kp,
                command.kd,
                command.torque,
            )
        )
        if active and runtime_state not in self._ACTIVE_COMMAND_STATES:
            raise SafetyViolationError(
                f'active joint command is forbidden in {runtime_state.value}'
            )

        if runtime_state in self._ACTIVE_COMMAND_STATES:
            if robot_state is None:
                raise SafetyViolationError(
                    f'a fresh robot state is required in {runtime_state.value}'
                )
            self.validate_robot_state(robot_state)
            self._validate_timestamp_age(
                command.timestamp,
                self._read_clock(),
                self.config.safety.max_state_age_s,
                'active joint command',
            )

        with self._lock:
            if not active:
                self._last_active_target = None
                return
            reference = (
                robot_state.joints.position
                if self._last_active_target is None
                else self._last_active_target
            )
            delta = np.abs(target - reference)
            tolerance = 1e-6
            violating = np.flatnonzero(
                delta > self._max_target_step + tolerance
            )
            if violating.size:
                index = int(violating[0])
                joint_name = self.config.robot.joint_names[index]
                raise SafetyViolationError(
                    f'target step for {joint_name} is {delta[index]:.6f} rad; '
                    f'limit is {self._max_target_step[index]:.6f} rad'
                )
            self._last_active_target = target.copy()

    def validate_joint_feedback(
        self,
        state: JointState,
        *,
        source_timestamps: Mapping[str, float] | None = None,
    ) -> None:
        self._validate_joint_feedback_at(
            state,
            source_timestamps=source_timestamps,
            now=self._read_clock(),
        )

    def validate_imu_feedback(self, state: ImuState) -> None:
        self._validate_imu_feedback_at(state, now=self._read_clock())

    def validate_robot_state(self, state: RealRobotState) -> None:
        if not isinstance(state, RealRobotState):
            raise TypeError('state must be a RealRobotState')
        if not state.valid:
            raise SafetyViolationError('robot state is marked invalid')
        now = self._read_clock()
        self._validate_joint_feedback_at(
            state.joints,
            source_timestamps=None,
            now=now,
        )
        self._validate_imu_feedback_at(state.imu, now=now)

    def validate_motor_cycle_duration(self, duration_s: float) -> None:
        self._validate_duration(
            duration_s,
            self.config.motors.state_timeout_s,
            'motor I/O cycle',
        )

    def validate_imu_cycle_duration(self, duration_s: float) -> None:
        self._validate_duration(
            duration_s,
            self.config.imu.state_timeout_s,
            'IMU read cycle',
        )

    def validate_policy_cycle_duration(self, duration_s: float) -> None:
        self._validate_duration(
            duration_s,
            1.0 / self.config.policy.frequency_hz,
            'policy cycle',
        )

    def _validate_joint_feedback_at(
        self,
        state: JointState,
        *,
        source_timestamps: Mapping[str, float] | None,
        now: float,
    ) -> None:
        if not isinstance(state, JointState):
            raise TypeError('state must be a JointState')
        if state.num_joints != self._num_joints:
            raise SafetyViolationError(
                f'motor feedback has {state.num_joints} joints, '
                f'expected {self._num_joints}'
            )
        if not state.valid:
            raise SafetyViolationError('motor feedback is marked invalid')
        for field_name, value in (
            ('position', state.position),
            ('velocity', state.velocity),
            ('torque', state.torque),
            ('temperature', state.temperature),
        ):
            self._finite_vector(value, self._num_joints, f'motor {field_name}')

        self._validate_timestamp_age(
            state.timestamp,
            now,
            self.config.motors.state_timeout_s,
            'motor feedback',
        )
        if source_timestamps is not None:
            if not source_timestamps:
                raise SafetyViolationError(
                    'motor source timestamps must not be empty'
                )
            for source_name, timestamp in source_timestamps.items():
                self._validate_timestamp_age(
                    timestamp,
                    now,
                    self.config.motors.state_timeout_s,
                    f'motor source {source_name!r}',
                )

        hot = np.flatnonzero(
            state.temperature
            > self.config.safety.max_motor_temperature_c
        )
        if hot.size:
            index = int(hot[0])
            raise SafetyViolationError(
                f'motor {self.config.robot.joint_names[index]} temperature '
                f'is {state.temperature[index]:.2f} C; limit is '
                f'{self.config.safety.max_motor_temperature_c:.2f} C'
            )
        errors = np.flatnonzero(state.error != 0)
        if errors.size:
            index = int(errors[0])
            raise SafetyViolationError(
                f'motor {self.config.robot.joint_names[index]} reported '
                f'error code {int(state.error[index])}'
            )

    def _validate_imu_feedback_at(
        self,
        state: ImuState,
        *,
        now: float,
    ) -> None:
        if not isinstance(state, ImuState):
            raise TypeError('state must be an ImuState')
        if not state.valid:
            raise SafetyViolationError('IMU feedback is marked invalid')
        quaternion = self._finite_vector(
            state.quaternion_wxyz,
            4,
            'IMU quaternion',
        )
        self._finite_vector(
            state.angular_velocity,
            3,
            'IMU angular velocity',
        )
        self._validate_timestamp_age(
            state.timestamp,
            now,
            self.config.imu.state_timeout_s,
            'IMU feedback',
        )

        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=2e-2):
            raise SafetyViolationError(
                f'IMU quaternion norm {norm:.6f} is invalid'
            )
        roll, pitch = self._quaternion_to_roll_pitch(quaternion / norm)
        limit = self.config.safety.max_abs_roll_pitch_rad
        if abs(roll) > limit or abs(pitch) > limit:
            raise SafetyViolationError(
                f'body tilt exceeds limit: roll={roll:.6f} rad, '
                f'pitch={pitch:.6f} rad, limit=+/-{limit:.6f} rad'
            )

    @staticmethod
    def _validate_duration(
        duration_s: float,
        limit_s: float,
        label: str,
    ) -> None:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0.0:
            raise SafetyViolationError(
                f'{label} duration must be finite and non-negative'
            )
        if duration > limit_s:
            raise SafetyViolationError(
                f'{label} exceeded {limit_s:.6f}s deadline: '
                f'{duration:.6f}s'
            )

    @staticmethod
    def _validate_timestamp_age(
        timestamp: float,
        now: float,
        limit_s: float,
        label: str,
    ) -> None:
        value = float(timestamp)
        if not math.isfinite(value) or value < 0.0:
            raise SafetyViolationError(
                f'{label} timestamp must be finite and non-negative'
            )
        age = now - value
        if age < 0.0:
            raise SafetyViolationError(
                f'{label} timestamp is ahead of the safety clock'
            )
        if age > limit_s:
            raise SafetyViolationError(
                f'{label} is {age:.6f}s old; limit is {limit_s:.6f}s'
            )

    @staticmethod
    def _quaternion_to_roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
        w, x, y, z = (float(value) for value in quaternion)
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sin_roll, cos_roll)
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        return roll, pitch

    @staticmethod
    def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,):
            raise SafetyViolationError(
                f'{label} must have shape ({size},), got {vector.shape}'
            )
        if not np.all(np.isfinite(vector)):
            raise SafetyViolationError(
                f'{label} must contain only finite values'
            )
        return np.array(vector, dtype=np.float32, copy=True)

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise SafetyViolationError(
                'safety clock must return finite, non-negative time'
            )
        return timestamp
