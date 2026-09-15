from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Any

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import JointCommand, RealDeploymentConfig
from rl_controller.real.hardware import (
    BackendCommunicationError,
    BackendError,
    BackendUnavailableError,
    MotorBackend,
    UnitreeM8010Backend,
)


class FakeClock:
    def __init__(self, initial_time: float = 100.0) -> None:
        self.current_time = float(initial_time)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.current_time

    def advance(self, duration_s: float) -> None:
        with self._lock:
            self.current_time += float(duration_s)


class FakeUnitreeSdk:
    class MotorType:
        GO_M8010_6 = 'go_m8010_6'

    class MotorMode:
        FOC = 'foc'
        BRAKE = 'brake'

    class MotorCmd:
        pass

    class MotorData:
        correct = False

    def __init__(
        self,
        *,
        clock: FakeClock,
        gear_ratio: float = 6.33,
        advance_per_send_s: float = 0.0,
        wall_delay_per_send_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.gear_ratio = float(gear_ratio)
        self.advance_per_send_s = float(advance_per_send_s)
        self.wall_delay_per_send_s = float(wall_delay_per_send_s)
        self.calls: list[dict[str, Any]] = []
        self.ports: dict[str, Any] = {}
        self.behavior: dict[tuple[str, int], str] = {}

        sdk = self

        class SerialPort:
            def __init__(self, path: str) -> None:
                self.path = path
                self.closed = False
                sdk.ports[path] = self

            def sendRecv(self, cmd: Any, data: Any) -> bool:
                if sdk.wall_delay_per_send_s > 0.0:
                    time.sleep(sdk.wall_delay_per_send_s)
                sdk.clock.advance(sdk.advance_per_send_s)
                call = {
                    'path': self.path,
                    'motorType': cmd.motorType,
                    'mode': cmd.mode,
                    'id': int(cmd.id),
                    'q': float(cmd.q),
                    'dq': float(cmd.dq),
                    'kp': float(cmd.kp),
                    'kd': float(cmd.kd),
                    'tau': float(cmd.tau),
                    'thread_name': threading.current_thread().name,
                }
                sdk.calls.append(call)
                behavior = sdk.behavior.get((self.path, int(cmd.id)), 'ok')
                if behavior == 'raise':
                    raise OSError('injected serial failure')
                if behavior == 'return_false':
                    return False

                data.motor_id = int(cmd.id) + (1 if behavior == 'wrong_id' else 0)
                data.correct = behavior != 'bad_frame'
                data.q = float(cmd.q)
                data.dq = float(cmd.dq)
                data.tau = float(cmd.tau)
                data.temp = 30.0 + int(cmd.id)
                data.merror = int(cmd.id)
                if behavior == 'nan_feedback':
                    data.q = np.nan
                return True

            def close(self) -> None:
                self.closed = True

        self.SerialPort = SerialPort

    def queryGearRatio(self, motor_type: Any) -> float:
        if motor_type != self.MotorType.GO_M8010_6:
            raise AssertionError('unexpected motor type')
        return self.gear_ratio

    @staticmethod
    def queryMotorMode(motor_type: Any, mode: Any) -> str:
        return f'{motor_type}:{mode}'


class UnitreeM8010BackendTest(unittest.TestCase):
    def setUp(self) -> None:
        base = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.clock = FakeClock()
        self.sdk = FakeUnitreeSdk(clock=self.clock)
        buses = (
            replace(
                base.motors.buses[0],
                name='left_bus',
                device='/dev/fake-left',
                baud_rate=4_000_000,
            ),
            replace(
                base.motors.buses[0],
                name='right_bus',
                device='/dev/fake-right',
                baud_rate=4_000_000,
            ),
        )
        joints = []
        for index, source in enumerate(base.motors.joints):
            joints.append(
                replace(
                    source,
                    bus='left_bus' if index < 6 else 'right_bus',
                    motor_id=index % 6,
                    direction=-1 if index == 0 else 1,
                    zero_offset_rad=0.25 if index == 0 else 0.0,
                    gear_ratio=self.sdk.gear_ratio,
                )
            )
        self.config = replace(
            base.motors,
            backend='unitree_m8010',
            buses=buses,
            joints=tuple(joints),
            state_timeout_s=0.03,
        )
        self.backends: list[UnitreeM8010Backend] = []

    def tearDown(self) -> None:
        for backend in reversed(self.backends):
            try:
                backend.close()
            except Exception:
                pass

    def make_command(self) -> JointCommand:
        position = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        position[0] = 0.1
        return JointCommand(
            position=position,
            velocity=np.linspace(0.2, 0.4, 12, dtype=np.float32),
            kp=np.linspace(20.0, 31.0, 12, dtype=np.float32),
            kd=np.linspace(0.5, 1.0, 12, dtype=np.float32),
            torque=np.linspace(1.0, 2.1, 12, dtype=np.float32),
            timestamp=self.clock(),
        )

    def make_backend(self) -> UnitreeM8010Backend:
        backend = UnitreeM8010Backend(
            self.config,
            sdk_module=self.sdk,
            clock=self.clock,
        )
        self.backends.append(backend)
        return backend

    def test_converts_joint_commands_and_feedback(self) -> None:
        self.sdk.advance_per_send_s = 0.001
        backend = self.make_backend()
        self.assertIsInstance(backend, MotorBackend)
        backend.open()
        command = self.make_command()

        state = backend.cycle(command)

        ratio = self.sdk.gear_ratio
        first = next(
            call
            for call in self.sdk.calls
            if call['path'] == '/dev/fake-left' and call['id'] == 0
        )
        self.assertEqual(first['path'], '/dev/fake-left')
        self.assertEqual(first['id'], 0)
        self.assertAlmostEqual(first['q'], -(0.1 - 0.25) * ratio, places=6)
        self.assertAlmostEqual(first['dq'], -0.2 * ratio, places=6)
        self.assertAlmostEqual(first['kp'], 20.0 / ratio**2, places=6)
        self.assertAlmostEqual(first['kd'], 0.5 / ratio**2, places=6)
        self.assertAlmostEqual(first['tau'], -1.0 / ratio, places=6)
        np.testing.assert_allclose(state.position, command.position, atol=1e-6)
        np.testing.assert_allclose(state.velocity, command.velocity, atol=1e-6)
        np.testing.assert_allclose(state.torque, command.torque, atol=1e-6)
        np.testing.assert_array_equal(
            state.error,
            np.array([0, 1, 2, 3, 4, 5] * 2),
        )
        self.assertTrue(state.valid)

        timestamps = backend.last_bus_timestamps
        self.assertEqual(set(timestamps), {'left_bus', 'right_bus'})
        self.assertGreaterEqual(state.timestamp, 100.006)
        self.assertLessEqual(state.timestamp, 100.012)
        self.assertGreaterEqual(backend.last_bus_time_skew_s, 0.0)
        self.assertLessEqual(backend.last_bus_time_skew_s, 0.006001)
        self.assertGreaterEqual(backend.last_state_age_s, 0.0)
        self.assertLessEqual(backend.last_state_age_s, 0.006001)
        self.assertTrue(
            all(value is None for value in backend.last_bus_errors.values())
        )

    def test_exchanges_independent_buses_on_worker_threads(self) -> None:
        self.sdk.wall_delay_per_send_s = 0.002
        backend = self.make_backend()
        backend.open()

        backend.cycle(self.make_command())

        worker_by_path = {
            path: {
                call['thread_name']
                for call in self.sdk.calls
                if call['path'] == path
            }
            for path in ('/dev/fake-left', '/dev/fake-right')
        }
        self.assertTrue(
            all(
                len(worker_names) == 1
                for worker_names in worker_by_path.values()
            )
        )
        worker_names = {
            next(iter(worker_names))
            for worker_names in worker_by_path.values()
        }
        self.assertEqual(len(worker_names), 2)
        self.assertTrue(
            all(name.startswith('unitree-rs485') for name in worker_names)
        )

    def test_uses_duplicate_motor_ids_on_independent_buses(self) -> None:
        backend = self.make_backend()
        backend.open()

        backend.cycle(self.make_command())

        left_ids = [
            call['id'] for call in self.sdk.calls if call['path'] == '/dev/fake-left'
        ]
        right_ids = [
            call['id'] for call in self.sdk.calls if call['path'] == '/dev/fake-right'
        ]
        self.assertEqual(left_ids, list(range(6)))
        self.assertEqual(right_ids, list(range(6)))

    def test_rejects_incomplete_or_wrong_motor_feedback(self) -> None:
        for behavior, message in (
            ('return_false', 'returned no frame'),
            ('bad_frame', 'invalid CRC/frame'),
            ('wrong_id', 'received response from ID'),
            ('nan_feedback', 'non-finite feedback'),
            ('raise', 'communication failed'),
        ):
            with self.subTest(behavior=behavior):
                self.sdk.calls.clear()
                self.sdk.behavior = {('/dev/fake-right', 2): behavior}
                backend = self.make_backend()
                backend.open()
                with self.assertRaisesRegex(BackendCommunicationError, message):
                    backend.cycle(self.make_command())
                self.assertIsNone(backend.last_state_age_s)
                self.assertIsNotNone(backend.last_bus_errors['right_bus'])

    def test_rejects_state_that_is_stale_before_all_buses_complete(self) -> None:
        self.sdk.advance_per_send_s = 0.01
        backend = self.make_backend()
        backend.open()

        with self.assertRaisesRegex(
            BackendCommunicationError,
            'Complete motor cycle took',
        ):
            backend.cycle(self.make_command())

        self.assertIn('cycle took', backend.last_bus_errors['left_bus'])
        self.assertIsNone(backend.last_state_age_s)

    def test_safe_stop_brakes_every_motor_and_latches(self) -> None:
        backend = self.make_backend()
        backend.open()

        backend.safe_stop()

        self.assertTrue(backend.safe_stopped)
        self.assertEqual(len(self.sdk.calls), 12)
        brake_mode = self.sdk.queryMotorMode(
            self.sdk.MotorType.GO_M8010_6,
            self.sdk.MotorMode.BRAKE,
        )
        self.assertTrue(all(call['mode'] == brake_mode for call in self.sdk.calls))
        self.assertTrue(
            all(
                call[field] == 0.0
                for call in self.sdk.calls
                for field in ('q', 'dq', 'kp', 'kd', 'tau')
            )
        )
        with self.assertRaisesRegex(BackendError, 'safe-stopped'):
            backend.cycle(self.make_command())

        ports = tuple(self.sdk.ports.values())
        backend.close()
        self.assertTrue(all(port.closed for port in ports))

    def test_safe_stop_attempts_remaining_motors_after_failure(self) -> None:
        self.sdk.behavior = {('/dev/fake-left', 1): 'raise'}
        backend = self.make_backend()
        backend.open()

        with self.assertRaisesRegex(BackendCommunicationError, 'safe-stop failed'):
            backend.safe_stop()

        self.assertEqual(len(self.sdk.calls), 12)
        self.assertTrue(backend.safe_stopped)

    def test_rejects_sdk_gear_ratio_mismatch(self) -> None:
        mismatched = replace(
            self.config,
            joints=tuple(
                replace(joint, gear_ratio=5.0) for joint in self.config.joints
            ),
        )
        backend = UnitreeM8010Backend(
            mismatched,
            sdk_module=self.sdk,
            clock=self.clock,
        )

        with self.assertRaisesRegex(BackendUnavailableError, 'does not match'):
            backend.open()
        self.assertFalse(backend.is_open)


if __name__ == '__main__':
    unittest.main()
