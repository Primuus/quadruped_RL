from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import struct
import threading
import time
from typing import Any

import numpy as np

from ..config import RealImuConfig
from ..orientation import (
    normalize_quaternion_wxyz,
    transform_imu_sample_to_body,
)
from ..types import ImuState
from .base import (
    BackendCommunicationError,
    BackendError,
    BackendUnavailableError,
)


YESENSE_HEADER = b'\x59\x53'
ANGULAR_VELOCITY_DATA_ID = 0x20
QUATERNION_DATA_ID = 0x41


class YesenseProtocolError(ValueError):
    """Raised when a checksum-valid YIS frame has an invalid payload."""


@dataclass(frozen=True, slots=True)
class YesenseFrame:
    tid: int
    message: bytes


class YesenseFrameParser:
    """Incrementally parse checksum-valid frames from a noisy byte stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.checksum_error_count = 0
        self.discarded_byte_count = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.checksum_error_count = 0
        self.discarded_byte_count = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[YesenseFrame]:
        self._buffer.extend(bytes(chunk))
        frames: list[YesenseFrame] = []
        while True:
            header_index = self._buffer.find(YESENSE_HEADER)
            if header_index < 0:
                keep = 1 if self._buffer.endswith(YESENSE_HEADER[:1]) else 0
                discarded = len(self._buffer) - keep
                if discarded > 0:
                    del self._buffer[:discarded]
                    self.discarded_byte_count += discarded
                break
            if header_index > 0:
                del self._buffer[:header_index]
                self.discarded_byte_count += header_index
            if len(self._buffer) < 5:
                break

            message_length = int(self._buffer[4])
            frame_length = 7 + message_length
            if len(self._buffer) < frame_length:
                break

            candidate = bytes(self._buffer[:frame_length])
            expected_ck1, expected_ck2 = yesense_checksum(candidate[2:-2])
            if candidate[-2:] != bytes((expected_ck1, expected_ck2)):
                del self._buffer[0]
                self.checksum_error_count += 1
                self.discarded_byte_count += 1
                continue

            tid = int.from_bytes(candidate[2:4], byteorder='little', signed=False)
            frames.append(
                YesenseFrame(
                    tid=tid,
                    message=candidate[5:-2],
                )
            )
            del self._buffer[:frame_length]
        return frames


def yesense_checksum(data: bytes | bytearray | memoryview) -> tuple[int, int]:
    ck1 = 0
    ck2 = 0
    for value in bytes(data):
        ck1 = (ck1 + value) & 0xFF
        ck2 = (ck2 + ck1) & 0xFF
    return ck1, ck2


def decode_yesense_imu_frame(
    frame: YesenseFrame,
    *,
    quaternion_order: str,
    angular_velocity_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    packets = _parse_message_packets(frame.message)
    angular_payload = _required_packet(
        packets,
        ANGULAR_VELOCITY_DATA_ID,
        12,
        'angular velocity',
    )
    quaternion_payload = _required_packet(
        packets,
        QUATERNION_DATA_ID,
        16,
        'quaternion',
    )

    angular_velocity = (
        np.asarray(struct.unpack('<3i', angular_payload), dtype=np.float64)
        * 1e-6
    )
    if angular_velocity_unit == 'deg/s':
        angular_velocity = np.deg2rad(angular_velocity)
    elif angular_velocity_unit != 'rad/s':
        raise YesenseProtocolError(
            f'Unsupported angular velocity unit {angular_velocity_unit!r}'
        )

    raw_quaternion = (
        np.asarray(struct.unpack('<4i', quaternion_payload), dtype=np.float64)
        * 1e-6
    )
    if quaternion_order == 'wxyz':
        quaternion_wxyz = raw_quaternion
    elif quaternion_order == 'xyzw':
        quaternion_wxyz = raw_quaternion[[3, 0, 1, 2]]
    else:
        raise YesenseProtocolError(
            f'Unsupported quaternion order {quaternion_order!r}'
        )

    quaternion_norm = float(np.linalg.norm(quaternion_wxyz))
    if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=2e-2):
        raise YesenseProtocolError(
            f'YIS130 quaternion norm {quaternion_norm:.6f} is outside tolerance'
        )
    try:
        quaternion_wxyz = normalize_quaternion_wxyz(
            quaternion_wxyz,
            'YIS130 quaternion',
        )
    except ValueError as exc:
        raise YesenseProtocolError(str(exc)) from exc
    if not np.all(np.isfinite(angular_velocity)):
        raise YesenseProtocolError('YIS130 angular velocity is non-finite')
    return quaternion_wxyz, angular_velocity.astype(np.float32)


class YesenseYis130Backend:
    """Read YIS private UART frames and publish body-frame IMU samples."""

    def __init__(
        self,
        config: RealImuConfig,
    ) -> None:
        if config.backend != 'yesense_yis130':
            raise ValueError(
                'YesenseYis130Backend requires imu.backend=yesense_yis130'
            )
        if config.device is None or not config.device.strip():
            raise ValueError('YIS130 requires a serial device path')
        if config.baud_rate is None or config.baud_rate <= 0:
            raise ValueError('YIS130 requires a positive baud rate')
        if config.quaternion_order not in {'wxyz', 'xyzw'}:
            raise ValueError('YIS130 quaternion_order must be wxyz or xyzw')
        if config.angular_velocity_unit not in {'rad/s', 'deg/s'}:
            raise ValueError(
                'YIS130 angular_velocity_unit must be rad/s or deg/s'
            )
        if config.mounting_quaternion_wxyz is None:
            raise ValueError('YIS130 requires an IMU-to-body mounting quaternion')

        self.config = config
        self._parser = YesenseFrameParser()
        self._lock = threading.RLock()
        self._serial: Any | None = None
        self._is_open = False
        self._last_frame_tid: int | None = None
        self._last_state_timestamp: float | None = None
        self._last_error: str | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._is_open

    @property
    def last_frame_tid(self) -> int | None:
        with self._lock:
            return self._last_frame_tid

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def checksum_error_count(self) -> int:
        with self._lock:
            return self._parser.checksum_error_count

    @property
    def discarded_byte_count(self) -> int:
        with self._lock:
            return self._parser.discarded_byte_count

    @property
    def last_state_age_s(self) -> float | None:
        with self._lock:
            timestamp = self._last_state_timestamp
        if timestamp is None:
            return None
        return max(0.0, self._read_clock() - timestamp)

    def open(self) -> None:
        with self._lock:
            if self._is_open:
                return
            factory = _load_serial_factory()
            serial_timeout = min(self.config.state_timeout_s, 0.005)
            serial_port = None
            try:
                serial_port = factory(
                    port=self.config.device,
                    baudrate=self.config.baud_rate,
                    timeout=serial_timeout,
                )
                reset_input = getattr(serial_port, 'reset_input_buffer', None)
                if callable(reset_input):
                    reset_input()
            except Exception as exc:
                close_error: Exception | None = None
                if serial_port is not None:
                    close = getattr(serial_port, 'close', None)
                    if callable(close):
                        try:
                            close()
                        except Exception as cleanup_exc:
                            close_error = cleanup_exc
                detail = f': {exc}'
                if close_error is not None:
                    detail += f'; cleanup also failed: {close_error}'
                raise BackendUnavailableError(
                    f'Failed to open YIS130 serial port {self.config.device}{detail}'
                ) from exc

            self._serial = serial_port
            self._parser.reset()
            self._last_frame_tid = None
            self._last_state_timestamp = None
            self._last_error = None
            self._is_open = True

    def read(self) -> ImuState:
        with self._lock:
            self._require_open()
            assert self._serial is not None
            start = self._read_clock()
            deadline = start + self.config.state_timeout_s
            last_problem: str | None = None
            checksum_errors_at_start = self._parser.checksum_error_count

            while True:
                before_read = self._read_clock()
                try:
                    waiting = max(0, int(getattr(self._serial, 'in_waiting', 0)))
                    chunk = self._serial.read(min(max(waiting, 1), 512))
                except Exception as exc:
                    message = f'YIS130 serial read failed: {exc}'
                    self._last_error = message
                    raise BackendCommunicationError(message) from exc
                received_at = self._read_clock()

                latest_sample: tuple[YesenseFrame, np.ndarray, np.ndarray] | None = None
                for frame in self._parser.feed(chunk):
                    try:
                        quaternion, angular_velocity = decode_yesense_imu_frame(
                            frame,
                            quaternion_order=self.config.quaternion_order,
                            angular_velocity_unit=self.config.angular_velocity_unit,
                        )
                    except YesenseProtocolError as exc:
                        last_problem = f'YIS130 frame {frame.tid} rejected: {exc}'
                        continue
                    latest_sample = (frame, quaternion, angular_velocity)

                if latest_sample is not None:
                    frame, quaternion, angular_velocity = latest_sample
                    try:
                        body_quaternion, body_angular_velocity = (
                            transform_imu_sample_to_body(
                                quaternion,
                                angular_velocity,
                                self.config.mounting_quaternion_wxyz,
                            )
                        )
                    except ValueError as exc:
                        message = f'YIS130 mounting transform failed: {exc}'
                        self._last_error = message
                        raise BackendCommunicationError(message) from exc

                    self._last_frame_tid = frame.tid
                    self._last_state_timestamp = received_at
                    self._last_error = None
                    return ImuState(
                        quaternion_wxyz=body_quaternion,
                        angular_velocity=body_angular_velocity,
                        timestamp=received_at,
                    )

                if received_at >= deadline:
                    message = 'Timed out waiting for a complete YIS130 IMU frame'
                    if last_problem is not None:
                        message += f'; last protocol error: {last_problem}'
                    elif (
                        self._parser.checksum_error_count
                        > checksum_errors_at_start
                    ):
                        message += '; checksum validation failed'
                    self._last_error = message
                    raise BackendCommunicationError(message)
                if not chunk and received_at <= before_read:
                    message = (
                        'YIS130 serial read returned no data without advancing '
                        'the monotonic timeout clock'
                    )
                    self._last_error = message
                    raise BackendCommunicationError(message)

    def close(self) -> None:
        with self._lock:
            if not self._is_open:
                return
            serial_port = self._serial
            self._serial = None
            self._is_open = False
            if serial_port is not None:
                close = getattr(serial_port, 'close', None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        self._last_error = f'Failed to close YIS130 serial port: {exc}'
                        raise BackendCommunicationError(self._last_error) from exc

    def _require_open(self) -> None:
        if not self._is_open:
            raise BackendError('YIS130 IMU backend is not open')

    def _read_clock(self) -> float:
        timestamp = float(time.monotonic())
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise BackendCommunicationError(
                'YIS130 clock must return finite, non-negative time'
            )
        return timestamp


def _parse_message_packets(message: bytes) -> dict[int, bytes]:
    packets: dict[int, bytes] = {}
    offset = 0
    while offset < len(message):
        if len(message) - offset < 2:
            raise YesenseProtocolError('truncated packet header')
        data_id = int(message[offset])
        data_length = int(message[offset + 1])
        offset += 2
        end = offset + data_length
        if end > len(message):
            raise YesenseProtocolError(
                f'data ID 0x{data_id:02X} exceeds message length'
            )
        if data_id in packets:
            raise YesenseProtocolError(f'duplicate data ID 0x{data_id:02X}')
        packets[data_id] = message[offset:end]
        offset = end
    return packets


def _required_packet(
    packets: dict[int, bytes],
    data_id: int,
    expected_length: int,
    field_name: str,
) -> bytes:
    if data_id not in packets:
        raise YesenseProtocolError(
            f'missing {field_name} packet 0x{data_id:02X}'
        )
    payload = packets[data_id]
    if len(payload) != expected_length:
        raise YesenseProtocolError(
            f'{field_name} packet 0x{data_id:02X} has length {len(payload)}, '
            f'expected {expected_length}'
        )
    return payload


def _load_serial_factory() -> Callable[..., Any]:
    try:
        serial_module = importlib.import_module('serial')
    except ImportError as exc:
        raise BackendUnavailableError(
            'pyserial is required for the YIS130 backend; install pyserial>=3.5'
        ) from exc
    factory = getattr(serial_module, 'Serial', None)
    if not callable(factory):
        raise BackendUnavailableError('pyserial does not expose serial.Serial')
    return factory
