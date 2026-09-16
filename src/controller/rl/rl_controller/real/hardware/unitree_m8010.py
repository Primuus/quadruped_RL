from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import importlib
import platform
from pathlib import Path
import sys
import threading
import time
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np

from ..config import MotorJointConfig, RealMotorConfig
from ..types import JointCommand, JointState
from .base import (
    BackendCommunicationError,
    BackendError,
    BackendUnavailableError,
)


@dataclass(frozen=True, slots=True)
class _BusCycleResult:
    feedback: tuple[tuple[int, float, float, float, float, int], ...]
    timestamp: float


class UnitreeM8010Backend:
    """GO-M8010-6 backend with all values exposed in joint output units."""

    def __init__(
        self,
        config: RealMotorConfig,
    ) -> None:
        if config.backend != 'unitree_m8010':
            raise ValueError(
                'UnitreeM8010Backend requires motors.backend=unitree_m8010'
            )
        if not config.joints:
            raise ValueError('UnitreeM8010Backend requires at least one joint')

        buses = {bus.name: bus for bus in config.buses}
        if len(buses) != len(config.buses):
            raise ValueError('Motor bus names must be unique')

        joints_by_bus: dict[str, list[tuple[int, MotorJointConfig]]] = {
            name: [] for name in buses
        }
        addresses: set[tuple[str, int]] = set()
        for index, joint in enumerate(config.joints):
            if joint.bus not in buses:
                raise ValueError(
                    f'Joint {joint.joint_name!r} references unknown bus {joint.bus!r}'
                )
            address = (joint.bus, joint.motor_id)
            if address in addresses:
                raise ValueError(
                    f'Duplicate motor ID {joint.motor_id} on bus {joint.bus!r}'
                )
            addresses.add(address)
            joints_by_bus[joint.bus].append((index, joint))

        unused_buses = [name for name, joints in joints_by_bus.items() if not joints]
        if unused_buses:
            raise ValueError(
                'Every configured motor bus must own at least one joint; unused: '
                + ', '.join(unused_buses)
            )
        for bus in config.buses:
            if bus.device is None or not bus.device.strip():
                raise ValueError(f'Motor bus {bus.name!r} requires a device path')

        self.config = config
        self._num_joints = len(config.joints)
        self._buses = buses
        self._joints_by_bus = joints_by_bus
        self._sdk: Any | None = None

        self._lock = threading.RLock()
        self._ports: dict[str, Any] = {}
        self._bus_executor: ThreadPoolExecutor | None = None
        self._motor_type: Any | None = None
        self._foc_mode: Any | None = None
        self._brake_mode: Any | None = None
        self._is_open = False
        self._safe_stopped = True
        self._last_complete_state_timestamp: float | None = None
        self._last_complete_bus_timestamps: dict[str, float] = {}
        self._bus_errors: dict[str, str | None] = {
            name: None for name in buses
        }

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

    @property
    def last_bus_timestamps(self) -> Mapping[str, float]:
        """Timestamps from the latest complete all-bus feedback cycle."""
        with self._lock:
            return MappingProxyType(self._last_complete_bus_timestamps.copy())

    @property
    def last_bus_errors(self) -> Mapping[str, str | None]:
        with self._lock:
            return MappingProxyType(self._bus_errors.copy())

    @property
    def last_state_age_s(self) -> float | None:
        with self._lock:
            timestamp = self._last_complete_state_timestamp
        if timestamp is None:
            return None
        return max(0.0, self._read_clock() - timestamp)

    @property
    def last_bus_time_skew_s(self) -> float | None:
        with self._lock:
            timestamps = tuple(self._last_complete_bus_timestamps.values())
        if len(timestamps) != len(self._buses):
            return None
        return max(timestamps) - min(timestamps)

    def open(self) -> None:
        with self._lock:
            if self._is_open:
                return

            sdk = _load_unitree_sdk()
            self._validate_sdk_api(sdk)
            motor_type = sdk.MotorType.GO_M8010_6
            sdk_ratio = float(sdk.queryGearRatio(motor_type))
            if not np.isfinite(sdk_ratio) or sdk_ratio <= 0.0:
                raise BackendUnavailableError(
                    f'Unitree SDK returned invalid GO-M8010-6 gear ratio {sdk_ratio}'
                )
            for joint in self.config.joints:
                if not np.isclose(
                    joint.gear_ratio,
                    sdk_ratio,
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise BackendUnavailableError(
                        f'{joint.joint_name} gear_ratio={joint.gear_ratio} does not '
                        f'match Unitree SDK GO-M8010-6 ratio {sdk_ratio}'
                    )

            ports: dict[str, Any] = {}
            try:
                # The official SerialPort API receives only the device path. The
                # motor-specific baud rate is selected internally by the SDK.
                for bus in self.config.buses:
                    ports[bus.name] = sdk.SerialPort(bus.device)
            except Exception as exc:
                self._close_ports(ports)
                raise BackendUnavailableError(
                    f'Failed to open Unitree motor bus: {exc}'
                ) from exc

            try:
                foc_mode = sdk.queryMotorMode(
                    motor_type,
                    sdk.MotorMode.FOC,
                )
                brake_mode = sdk.queryMotorMode(
                    motor_type,
                    sdk.MotorMode.BRAKE,
                )
                bus_executor = ThreadPoolExecutor(
                    max_workers=len(self._buses),
                    thread_name_prefix='unitree-rs485',
                )
            except Exception as exc:
                self._close_ports(ports)
                raise BackendUnavailableError(
                    f'Failed to initialize Unitree motor bus workers: {exc}'
                ) from exc

            self._sdk = sdk
            self._motor_type = motor_type
            self._foc_mode = foc_mode
            self._brake_mode = brake_mode
            self._ports = ports
            self._bus_executor = bus_executor
            self._is_open = True
            self._safe_stopped = False
            self._last_complete_state_timestamp = None
            self._last_complete_bus_timestamps.clear()
            self._bus_errors = {name: None for name in self._buses}

    def cycle(self, command: JointCommand) -> JointState:
        self._validate_command(command)
        with self._lock:
            self._require_open()
            if self._safe_stopped:
                raise BackendError(
                    'Unitree motor backend is safe-stopped; close and reopen it '
                    'before sending FOC commands'
                )

            executor = self._bus_executor
            if executor is None:
                raise BackendError('Unitree motor bus workers are unavailable')

            cycle_start = self._read_clock()
            futures = {
                bus_name: executor.submit(
                    self._exchange_bus,
                    bus_name,
                    indexed_joints,
                    command,
                )
                for bus_name, indexed_joints in self._joints_by_bus.items()
            }
            results: dict[str, _BusCycleResult] = {}
            failures: list[str] = []
            for bus_name, future in futures.items():
                try:
                    results[bus_name] = future.result()
                    self._bus_errors[bus_name] = None
                except BackendCommunicationError as exc:
                    message = f'Bus {bus_name!r} communication failed: {exc}'
                    self._bus_errors[bus_name] = message
                    failures.append(message)
                except Exception as exc:
                    message = f'Bus {bus_name!r} communication failed: {exc}'
                    self._bus_errors[bus_name] = message
                    failures.append(message)
            if failures:
                raise BackendCommunicationError(' | '.join(failures))

            position = np.empty(self._num_joints, dtype=np.float32)
            velocity = np.empty(self._num_joints, dtype=np.float32)
            torque = np.empty(self._num_joints, dtype=np.float32)
            temperature = np.empty(self._num_joints, dtype=np.float32)
            error = np.empty(self._num_joints, dtype=np.int64)
            cycle_bus_timestamps: dict[str, float] = {}
            for bus_name, result in results.items():
                cycle_bus_timestamps[bus_name] = result.timestamp
                for values in result.feedback:
                    index, q, dq, tau, temp, motor_error = values
                    position[index] = q
                    velocity[index] = dq
                    torque[index] = tau
                    temperature[index] = temp
                    error[index] = motor_error

            now = self._read_clock()
            oldest_timestamp = min(cycle_bus_timestamps.values())
            state_age = now - oldest_timestamp
            cycle_duration = now - cycle_start
            if state_age < 0.0 or cycle_duration < 0.0:
                raise BackendCommunicationError('Motor backend clock moved backwards')
            if cycle_duration > self.config.state_timeout_s:
                message = (
                    f'Complete motor cycle took {cycle_duration:.6f}s; '
                    f'limit is {self.config.state_timeout_s:.6f}s'
                )
                for bus_name in cycle_bus_timestamps:
                    self._bus_errors[bus_name] = message
                raise BackendCommunicationError(message)
            if state_age > self.config.state_timeout_s:
                message = (
                    f'Complete motor state is already {state_age:.6f}s old; '
                    f'limit is {self.config.state_timeout_s:.6f}s'
                )
                for bus_name in cycle_bus_timestamps:
                    self._bus_errors[bus_name] = message
                raise BackendCommunicationError(message)

            self._last_complete_bus_timestamps = cycle_bus_timestamps
            self._last_complete_state_timestamp = oldest_timestamp
            return JointState(
                position=position,
                velocity=velocity,
                torque=torque,
                temperature=temperature,
                error=error,
                timestamp=oldest_timestamp,
            )

    def safe_stop(self) -> None:
        with self._lock:
            if not self._is_open or self._safe_stopped:
                return
            self._safe_stopped = True
            executor = self._bus_executor
            if executor is None:
                raise BackendError('Unitree motor bus workers are unavailable')
            futures = {
                bus_name: executor.submit(
                    self._brake_bus,
                    bus_name,
                    indexed_joints,
                )
                for bus_name, indexed_joints in self._joints_by_bus.items()
            }
            failures: list[str] = []
            for bus_name, future in futures.items():
                try:
                    bus_failures = future.result()
                except Exception as exc:
                    bus_failures = [f'bus worker failed: {exc}']
                if bus_failures:
                    message = (
                        f'Bus {bus_name!r} safe-stop failed: '
                        + '; '.join(bus_failures)
                    )
                    self._bus_errors[bus_name] = message
                    failures.append(message)
                else:
                    self._bus_errors[bus_name] = None
            if failures:
                raise BackendCommunicationError(' | '.join(failures))

    def close(self) -> None:
        with self._lock:
            if not self._is_open:
                return
            stop_error: BackendCommunicationError | None = None
            try:
                self.safe_stop()
            except BackendCommunicationError as exc:
                stop_error = exc
            finally:
                executor = self._bus_executor
                self._bus_executor = None
                self._close_ports(self._ports)
                self._ports = {}
                self._is_open = False
                self._safe_stopped = True
                self._sdk = None
                self._motor_type = None
                self._foc_mode = None
                self._brake_mode = None
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            if stop_error is not None:
                raise stop_error

    def _exchange_bus(
        self,
        bus_name: str,
        indexed_joints: list[tuple[int, MotorJointConfig]],
        command: JointCommand,
    ) -> _BusCycleResult:
        port = self._ports[bus_name]
        feedback: list[tuple[int, float, float, float, float, int]] = []
        for index, joint in indexed_joints:
            data = self._exchange_joint(port, joint, command, index)
            q, dq, tau = self._feedback_to_joint(data, joint)
            feedback.append(
                (
                    index,
                    q,
                    dq,
                    tau,
                    float(data.temp),
                    int(data.merror),
                )
            )
        return _BusCycleResult(
            feedback=tuple(feedback),
            timestamp=self._read_clock(),
        )

    def _brake_bus(
        self,
        bus_name: str,
        indexed_joints: list[tuple[int, MotorJointConfig]],
    ) -> list[str]:
        failures: list[str] = []
        port = self._ports[bus_name]
        for _, joint in indexed_joints:
            try:
                self._exchange_brake(port, joint)
            except Exception as exc:
                failures.append(
                    f'motor {joint.motor_id} ({joint.joint_name}): {exc}'
                )
        return failures

    def _exchange_joint(
        self,
        port: Any,
        joint: MotorJointConfig,
        command: JointCommand,
        index: int,
    ) -> Any:
        assert self._sdk is not None
        cmd = self._sdk.MotorCmd()
        data = self._sdk.MotorData()
        cmd.motorType = self._motor_type
        data.motorType = self._motor_type
        cmd.mode = self._foc_mode
        cmd.id = joint.motor_id

        ratio = joint.gear_ratio
        direction = float(joint.direction)
        motor_output_position = direction * (
            float(command.position[index]) - joint.zero_offset_rad
        )
        cmd.q = motor_output_position * ratio
        cmd.dq = direction * float(command.velocity[index]) * ratio
        cmd.kp = float(command.kp[index]) / (ratio * ratio)
        cmd.kd = float(command.kd[index]) / (ratio * ratio)
        cmd.tau = direction * float(command.torque[index]) / ratio

        result = port.sendRecv(cmd, data)
        self._validate_response(result, data, joint)
        return data

    def _exchange_brake(self, port: Any, joint: MotorJointConfig) -> None:
        assert self._sdk is not None
        cmd = self._sdk.MotorCmd()
        data = self._sdk.MotorData()
        cmd.motorType = self._motor_type
        data.motorType = self._motor_type
        cmd.mode = self._brake_mode
        cmd.id = joint.motor_id
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.kp = 0.0
        cmd.kd = 0.0
        cmd.tau = 0.0
        result = port.sendRecv(cmd, data)
        self._validate_response(result, data, joint)

    @staticmethod
    def _validate_response(
        result: Any,
        data: Any,
        joint: MotorJointConfig,
    ) -> None:
        if result is not None and not bool(result):
            raise BackendCommunicationError(
                f'Motor {joint.motor_id} ({joint.joint_name}) returned no frame'
            )
        if not bool(data.correct):
            raise BackendCommunicationError(
                f'Motor {joint.motor_id} ({joint.joint_name}) returned invalid CRC/frame'
            )
        if int(data.motor_id) != joint.motor_id:
            raise BackendCommunicationError(
                f'Motor {joint.motor_id} ({joint.joint_name}) received response '
                f'from ID {int(data.motor_id)}'
            )

    @staticmethod
    def _feedback_to_joint(
        data: Any,
        joint: MotorJointConfig,
    ) -> tuple[float, float, float]:
        ratio = joint.gear_ratio
        direction = float(joint.direction)
        motor_output_position = float(data.q) / ratio
        joint_values = (
            direction * motor_output_position + joint.zero_offset_rad,
            direction * float(data.dq) / ratio,
            direction * float(data.tau) * ratio,
        )
        if not all(np.isfinite(value) for value in joint_values):
            raise BackendCommunicationError(
                f'Motor {joint.motor_id} ({joint.joint_name}) returned '
                'non-finite feedback'
            )
        if not np.isfinite(float(data.temp)):
            raise BackendCommunicationError(
                f'Motor {joint.motor_id} ({joint.joint_name}) returned '
                'non-finite temperature'
            )
        return joint_values

    def _validate_command(self, command: JointCommand) -> None:
        if not isinstance(command, JointCommand):
            raise TypeError('command must be a JointCommand')
        if command.num_joints != self._num_joints:
            raise ValueError(
                f'command has {command.num_joints} joints, expected {self._num_joints}'
            )

    def _require_open(self) -> None:
        if not self._is_open:
            raise BackendError('Unitree motor backend is not open')

    def _read_clock(self) -> float:
        value = float(time.monotonic())
        if not np.isfinite(value) or value < 0.0:
            raise BackendCommunicationError(
                'Unitree motor clock must return finite, non-negative time'
            )
        return value

    @staticmethod
    def _validate_sdk_api(sdk: Any) -> None:
        required = (
            'MotorType',
            'MotorMode',
            'MotorCmd',
            'MotorData',
            'SerialPort',
            'queryGearRatio',
            'queryMotorMode',
        )
        missing = [name for name in required if not hasattr(sdk, name)]
        if missing:
            raise BackendUnavailableError(
                'Unitree actuator SDK is missing API members: '
                + ', '.join(missing)
            )

    @staticmethod
    def _close_ports(ports: Mapping[str, Any]) -> None:
        for port in ports.values():
            close = getattr(port, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _load_unitree_sdk() -> ModuleType:
    """Load an installed SDK or the source-tree binding built by our script."""
    try:
        return importlib.import_module('unitree_actuator_sdk')
    except (ImportError, OSError):
        pass

    rl_package_root = Path(__file__).resolve().parents[3]
    sdk_lib = rl_package_root / 'third_party' / 'unitree_actuator_sdk' / 'lib'
    machine = platform.machine().lower()
    core_name = (
        'libUnitreeMotorSDK_Arm64.so'
        if machine in {'aarch64', 'arm64'}
        else 'libUnitreeMotorSDK_Linux64.so'
    )
    core_library = sdk_lib / core_name
    if not core_library.is_file():
        raise BackendUnavailableError(
            f'Unitree actuator SDK core library not found: {core_library}'
        )

    try:
        ctypes.CDLL(str(core_library), mode=ctypes.RTLD_GLOBAL)
        sdk_path = str(sdk_lib)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        return importlib.import_module('unitree_actuator_sdk')
    except (ImportError, OSError) as exc:
        raise BackendUnavailableError(
            'Unitree actuator SDK Python binding is unavailable for this Python '
            'ABI. Run third_party/build_unitree_actuator_sdk.sh first.'
        ) from exc
