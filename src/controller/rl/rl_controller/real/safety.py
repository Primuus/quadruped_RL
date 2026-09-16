from __future__ import annotations

import math
import time

import numpy as np

from .config import RealDeploymentConfig
from .state_machine import RuntimeState
from .types import ImuState, JointCommand, JointState, RealRobotState


class SafetyViolationError(RuntimeError):
    """Raised when live feedback or a final motor command is unsafe."""


class RealSafetyManager:
    """Own dynamic health checks and the final motor-output authorization."""

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

    def __init__(self, config: RealDeploymentConfig) -> None:
        self.config = config
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
        self._last_active_target: np.ndarray | None = None

    @property
    def last_active_target(self) -> np.ndarray | None:
        if self._last_active_target is None:
            return None
        return self._last_active_target.copy()

    def reset(self) -> None:
        self._last_active_target = None

    def validate_motor_health(self, state: JointState) -> None:
        """Check operational motor health after the backend accepts a frame."""
        if not isinstance(state, JointState):
            raise TypeError('state must be a JointState')
        if state.num_joints != self._num_joints:
            raise SafetyViolationError(
                f'motor feedback has {state.num_joints} joints, '
                f'expected {self._num_joints}'
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

    def validate_imu_health(self, state: ImuState) -> None:
        """Check the configured body-tilt limit after IMU normalization."""
        if not isinstance(state, ImuState):
            raise TypeError('state must be an ImuState')
        roll, pitch = self._quaternion_to_roll_pitch(
            state.quaternion_wxyz
        )
        limit = self.config.safety.max_abs_roll_pitch_rad
        if abs(roll) > limit or abs(pitch) > limit:
            raise SafetyViolationError(
                f'body tilt exceeds limit: roll={roll:.6f} rad, '
                f'pitch={pitch:.6f} rad, limit=+/-{limit:.6f} rad'
            )

    def authorize_command(
        self,
        command: JointCommand,
        runtime_state: RuntimeState,
        robot_state: RealRobotState | None,
    ) -> None:
        """Perform the single final authorization before motor transmission."""
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

        target = np.asarray(command.position, dtype=np.float32)
        if not np.all(np.isfinite(target)):
            raise SafetyViolationError(
                'target position must contain only finite values'
            )
        violating = np.flatnonzero(
            (target < self._position_min) | (target > self._position_max)
        )
        if violating.size:
            index = int(violating[0])
            joint_name = self.config.robot.joint_names[index]
            raise SafetyViolationError(
                f'target position for {joint_name} is {target[index]:.6f} rad; '
                f'limits are [{self._position_min[index]:.6f}, '
                f'{self._position_max[index]:.6f}] rad'
            )

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

        if not active:
            self._last_active_target = None
            return
        if robot_state is None:
            raise SafetyViolationError(
                f'a synchronized robot state is required in '
                f'{runtime_state.value}'
            )
        self._validate_command_age(command.timestamp)

        reference = (
            robot_state.joints.position
            if self._last_active_target is None
            else self._last_active_target
        )
        delta = np.abs(target - reference)
        violating = np.flatnonzero(delta > self._max_target_step + 1e-6)
        if violating.size:
            index = int(violating[0])
            joint_name = self.config.robot.joint_names[index]
            raise SafetyViolationError(
                f'target step for {joint_name} is {delta[index]:.6f} rad; '
                f'limit is {self._max_target_step[index]:.6f} rad'
            )
        self._last_active_target = target.copy()

    def _validate_command_age(self, timestamp: float) -> None:
        now = time.monotonic()
        age = now - float(timestamp)
        if not math.isfinite(age) or age < 0.0:
            raise SafetyViolationError(
                'active joint command timestamp is invalid'
            )
        limit = self.config.safety.max_state_age_s
        if age > limit:
            raise SafetyViolationError(
                f'active joint command is {age:.6f}s old; '
                f'limit is {limit:.6f}s'
            )

    @staticmethod
    def _quaternion_to_roll_pitch(
        quaternion: np.ndarray,
    ) -> tuple[float, float]:
        w, x, y, z = (float(value) for value in quaternion)
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sin_roll, cos_roll)
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        return roll, pitch
