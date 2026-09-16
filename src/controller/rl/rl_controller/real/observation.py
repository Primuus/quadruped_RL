from __future__ import annotations

from typing import Any

import numpy as np

from .config import RealDeploymentConfig
from .orientation import quaternion_rotate_inverse_wxyz
from .types import RealRobotState


GRAVITY_VECTOR = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


class ObservationContinuityError(RuntimeError):
    """Raised when history receives a duplicate or backwards state timestamp."""


def build_real_one_step_observation(
    state: RealRobotState,
    command: Any,
    last_action: Any,
    config: RealDeploymentConfig,
) -> np.ndarray:
    """Build the actor's current frame in the exact training-side order."""
    if not isinstance(state, RealRobotState):
        raise TypeError('state must be a RealRobotState')
    num_actions = config.robot.num_actions
    if state.num_joints != num_actions:
        raise ValueError(
            f'state has {state.num_joints} joints, expected {num_actions}'
        )

    command_vector = _vector(command, 3, 'command')
    previous_action = _vector(last_action, num_actions, 'last_action')

    scales = config.policy.observation_scales
    projected_gravity = quaternion_rotate_inverse_wxyz(
        state.imu.quaternion_wxyz,
        GRAVITY_VECTOR,
    )
    observation = np.concatenate(
        (
            command_vector * scales.command,
            state.imu.angular_velocity * scales.angular_velocity,
            projected_gravity,
            (state.joints.position - config.robot.default_angles)
            * scales.dof_position,
            state.joints.velocity * scales.dof_velocity,
            previous_action,
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (config.robot.num_one_step_obs,):
        raise ValueError(
            'one-step observation has shape '
            f'{observation.shape}, expected ({config.robot.num_one_step_obs},)'
        )
    return observation


class RealObservationHistory:
    """Warm up and maintain newest-first actor observation history."""

    def __init__(
        self,
        config: RealDeploymentConfig,
        *,
        max_frame_gap_s: float | None = None,
    ) -> None:
        self.config = config
        self._step_size = config.robot.num_one_step_obs
        self._history_length = config.robot.history_length
        self._warmup_frames = config.runtime.history_warmup_frames
        configured_gap = (
            config.safety.max_state_age_s
            if max_frame_gap_s is None
            else float(max_frame_gap_s)
        )
        if not np.isfinite(configured_gap) or configured_gap <= 0.0:
            raise ValueError('max_frame_gap_s must be finite and positive')
        self._max_frame_gap_s = configured_gap
        self._buffer = np.zeros(config.robot.num_obs, dtype=np.float32)
        self._frames_seen = 0
        self._last_state_timestamp: float | None = None
        self._last_reset_reason: str | None = None
        self._last_one_step = np.zeros(self._step_size, dtype=np.float32)

    @property
    def ready(self) -> bool:
        return self._frames_seen >= self._warmup_frames

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def last_reset_reason(self) -> str | None:
        return self._last_reset_reason

    @property
    def last_one_step_observation(self) -> np.ndarray:
        return self._last_one_step.copy()

    def reset(self, reason: str | None = None) -> None:
        self._buffer.fill(0.0)
        self._last_one_step.fill(0.0)
        self._frames_seen = 0
        self._last_state_timestamp = None
        self._last_reset_reason = reason

    def update(
        self,
        state: RealRobotState,
        command: Any,
        last_action: Any,
    ) -> np.ndarray | None:
        previous_timestamp = self._last_state_timestamp
        if previous_timestamp is not None:
            frame_gap = state.timestamp - previous_timestamp
            if frame_gap <= 0.0:
                self.reset('state timestamps must increase for every history frame')
                raise ObservationContinuityError(self._last_reset_reason)
            if frame_gap > self._max_frame_gap_s:
                self.reset(
                    f'state frame gap {frame_gap:.6f}s exceeds '
                    f'{self._max_frame_gap_s:.6f}s'
                )

        one_step = build_real_one_step_observation(
            state,
            command,
            last_action,
            self.config,
        )
        if self._history_length > 1:
            self._buffer[self._step_size :] = self._buffer[
                : -self._step_size
            ].copy()
        self._buffer[: self._step_size] = one_step
        self._last_one_step = one_step.copy()
        self._last_state_timestamp = state.timestamp
        self._frames_seen += 1

        if not self.ready:
            return None
        return self._buffer.copy()

def _vector(value: Any, size: int, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f'{field_name} must have shape ({size},), got {vector.shape}')
    return np.array(vector, dtype=np.float32, copy=True)
