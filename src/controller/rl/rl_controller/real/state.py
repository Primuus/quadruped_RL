from __future__ import annotations

from collections.abc import Mapping
import threading
import time
from types import MappingProxyType
from typing import Callable

import numpy as np

from .config import RealDeploymentConfig
from .types import ImuState, JointState, RealRobotState


class StateUnavailableError(RuntimeError):
    """Raised when no complete, fresh and synchronized robot state exists."""


class RealStateAggregator:
    """Merge complete motor and IMU feedback into synchronized snapshots."""

    def __init__(
        self,
        *,
        num_joints: int,
        max_state_age_s: float,
        max_source_skew_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(num_joints, bool) or not isinstance(num_joints, int):
            raise ValueError('num_joints must be an integer')
        if num_joints <= 0:
            raise ValueError('num_joints must be positive')
        self._num_joints = num_joints
        self._max_state_age_s = self._positive_finite(
            max_state_age_s,
            'max_state_age_s',
        )
        self._max_source_skew_s = self._positive_finite(
            max_source_skew_s,
            'max_source_skew_s',
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._joints: JointState | None = None
        self._imu: ImuState | None = None
        self._motor_source_timestamps: dict[str, float] = {}
        self._last_rejection: str | None = 'motor and IMU feedback have not arrived'

    @classmethod
    def from_config(
        cls,
        config: RealDeploymentConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> 'RealStateAggregator':
        return cls(
            num_joints=config.robot.num_actions,
            max_state_age_s=config.safety.max_state_age_s,
            max_source_skew_s=min(
                config.motors.state_timeout_s,
                config.imu.state_timeout_s,
            ),
            clock=clock,
        )

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def last_rejection(self) -> str | None:
        with self._lock:
            return self._last_rejection

    @property
    def motor_source_timestamps(self) -> Mapping[str, float]:
        with self._lock:
            return MappingProxyType(self._motor_source_timestamps.copy())

    def feedback_ages(self) -> Mapping[str, float | None]:
        """Return the latest feedback ages without requiring a valid snapshot."""

        with self._lock:
            joints = self._joints
            imu = self._imu
            source_timestamps = self._motor_source_timestamps.copy()
        now = self._read_clock()

        motor_age = None if joints is None else now - joints.timestamp
        imu_age = None if imu is None else now - imu.timestamp
        available_ages = [
            age for age in (motor_age, imu_age) if age is not None
        ]
        result: dict[str, float | None] = {
            'motor': motor_age,
            'imu': imu_age,
            'oldest': max(available_ages) if available_ages else None,
        }
        for source_name, timestamp in sorted(source_timestamps.items()):
            result[f'motor_source:{source_name}'] = now - timestamp
        return MappingProxyType(result)

    def clear(self) -> None:
        with self._lock:
            self._joints = None
            self._imu = None
            self._motor_source_timestamps.clear()
            self._last_rejection = 'motor and IMU feedback have not arrived'

    def update_joints(
        self,
        state: JointState,
        *,
        source_timestamps: Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(state, JointState):
            raise TypeError('state must be a JointState')
        if state.num_joints != self._num_joints:
            raise ValueError(
                f'JointState has {state.num_joints} joints, '
                f'expected {self._num_joints}'
            )
        timestamps = self._validated_source_timestamps(
            state.timestamp,
            source_timestamps,
        )
        with self._lock:
            self._joints = state
            self._motor_source_timestamps = timestamps

    def update_imu(self, state: ImuState) -> None:
        if not isinstance(state, ImuState):
            raise TypeError('state must be an ImuState')
        with self._lock:
            self._imu = state

    def try_snapshot(self) -> RealRobotState | None:
        try:
            return self.snapshot()
        except StateUnavailableError:
            return None

    def snapshot(self) -> RealRobotState:
        now = self._read_clock()
        with self._lock:
            joints = self._joints
            imu = self._imu
            source_timestamps = self._motor_source_timestamps.copy()

            rejection = self._validate_snapshot(
                joints,
                imu,
                source_timestamps,
                now,
            )
            if rejection is not None:
                self._last_rejection = rejection
                raise StateUnavailableError(rejection)

            assert joints is not None
            assert imu is not None
            timestamps = tuple(source_timestamps.values()) + (imu.timestamp,)
            snapshot = RealRobotState(
                joints=joints,
                imu=imu,
                timestamp=min(timestamps),
            )
            self._last_rejection = None
            return snapshot

    def _validate_snapshot(
        self,
        joints: JointState | None,
        imu: ImuState | None,
        source_timestamps: Mapping[str, float],
        now: float,
    ) -> str | None:
        if joints is None:
            return 'complete motor feedback is unavailable'
        if imu is None:
            return 'IMU feedback is unavailable'
        if joints.num_joints != self._num_joints:
            return (
                f'motor feedback has {joints.num_joints} joints, '
                f'expected {self._num_joints}'
            )
        if not source_timestamps:
            return 'motor source timestamps are unavailable'

        timestamps = tuple(source_timestamps.values()) + (imu.timestamp,)
        ages = tuple(now - timestamp for timestamp in timestamps)
        if any(age < 0.0 for age in ages):
            return 'a hardware timestamp is ahead of the monotonic clock'
        oldest_age = max(ages)
        if oldest_age > self._max_state_age_s:
            return (
                f'hardware state is {oldest_age:.6f}s old, '
                f'limit is {self._max_state_age_s:.6f}s'
            )
        source_skew = max(timestamps) - min(timestamps)
        if source_skew > self._max_source_skew_s:
            return (
                f'hardware source skew is {source_skew:.6f}s, '
                f'limit is {self._max_source_skew_s:.6f}s'
            )
        return None

    @staticmethod
    def _validated_source_timestamps(
        state_timestamp: float,
        source_timestamps: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if source_timestamps is None:
            return {'motor': float(state_timestamp)}
        if not source_timestamps:
            raise ValueError('source_timestamps must not be empty')

        result: dict[str, float] = {}
        for name, value in source_timestamps.items():
            source_name = str(name).strip()
            if not source_name:
                raise ValueError('source timestamp names must not be empty')
            if source_name in result:
                raise ValueError(f'duplicate source timestamp name {source_name!r}')
            timestamp = float(value)
            if not np.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(
                    f'source timestamp {source_name!r} must be finite and non-negative'
                )
            result[source_name] = timestamp

        oldest_source = min(result.values())
        if not np.isclose(
            state_timestamp,
            oldest_source,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                'JointState.timestamp must equal the oldest motor source timestamp'
            )
        return result

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise StateUnavailableError(
                'state clock must return finite, non-negative time'
            )
        return timestamp

    @staticmethod
    def _positive_finite(value: float, field_name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f'{field_name} must be finite and positive')
        return number
