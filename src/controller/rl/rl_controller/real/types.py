from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly_float_vector(value: Any, size: int, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f'{field_name} must have shape ({size},), got {vector.shape}')
    result = np.array(vector, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _readonly_int_vector(value: Any, size: int, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.int64)
    if vector.shape != (size,):
        raise ValueError(f'{field_name} must have shape ({size},), got {vector.shape}')
    result = np.array(vector, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _validate_timestamp(value: float, field_name: str = 'timestamp') -> float:
    timestamp = float(value)
    if not np.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f'{field_name} must be a finite, non-negative monotonic time')
    return timestamp


@dataclass(frozen=True, slots=True)
class JointState:
    """One complete joint feedback snapshot in policy joint order."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    temperature: np.ndarray
    error: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position)
        if position.ndim != 1 or position.size == 0:
            raise ValueError(f'position must be a non-empty vector, got {position.shape}')
        num_joints = int(position.size)
        object.__setattr__(
            self,
            'position',
            _readonly_float_vector(self.position, num_joints, 'position'),
        )
        object.__setattr__(
            self,
            'velocity',
            _readonly_float_vector(self.velocity, num_joints, 'velocity'),
        )
        object.__setattr__(
            self,
            'torque',
            _readonly_float_vector(self.torque, num_joints, 'torque'),
        )
        object.__setattr__(
            self,
            'temperature',
            _readonly_float_vector(self.temperature, num_joints, 'temperature'),
        )
        object.__setattr__(
            self,
            'error',
            _readonly_int_vector(self.error, num_joints, 'error'),
        )
        object.__setattr__(self, 'timestamp', _validate_timestamp(self.timestamp))

    @property
    def num_joints(self) -> int:
        return int(self.position.size)


@dataclass(frozen=True, slots=True)
class ImuState:
    """One IMU snapshot expressed in the robot body frame."""

    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'quaternion_wxyz',
            _readonly_float_vector(self.quaternion_wxyz, 4, 'quaternion_wxyz'),
        )
        object.__setattr__(
            self,
            'angular_velocity',
            _readonly_float_vector(self.angular_velocity, 3, 'angular_velocity'),
        )
        object.__setattr__(self, 'timestamp', _validate_timestamp(self.timestamp))


@dataclass(frozen=True, slots=True)
class JointCommand:
    """One output command in policy joint order and output-side units."""

    position: np.ndarray
    velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position)
        if position.ndim != 1 or position.size == 0:
            raise ValueError(f'position must be a non-empty vector, got {position.shape}')
        num_joints = int(position.size)
        object.__setattr__(
            self,
            'position',
            _readonly_float_vector(self.position, num_joints, 'position'),
        )
        object.__setattr__(
            self,
            'velocity',
            _readonly_float_vector(self.velocity, num_joints, 'velocity'),
        )
        object.__setattr__(
            self,
            'kp',
            _readonly_float_vector(self.kp, num_joints, 'kp'),
        )
        object.__setattr__(
            self,
            'kd',
            _readonly_float_vector(self.kd, num_joints, 'kd'),
        )
        object.__setattr__(
            self,
            'torque',
            _readonly_float_vector(self.torque, num_joints, 'torque'),
        )
        object.__setattr__(self, 'timestamp', _validate_timestamp(self.timestamp))

    @property
    def num_joints(self) -> int:
        return int(self.position.size)


@dataclass(frozen=True, slots=True)
class RealRobotState:
    """Synchronized motor and IMU state consumed by the real-policy runner."""

    joints: JointState
    imu: ImuState
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.joints, JointState):
            raise TypeError('joints must be a JointState')
        if not isinstance(self.imu, ImuState):
            raise TypeError('imu must be an ImuState')
        object.__setattr__(self, 'timestamp', _validate_timestamp(self.timestamp))

    @property
    def num_joints(self) -> int:
        return self.joints.num_joints
