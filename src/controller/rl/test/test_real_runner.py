from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import RealDeploymentConfig
from rl_controller.real.hardware import MockMotorBackend
from rl_controller.real.policy import MockPolicy, PolicyExecutor
from rl_controller.real.runner import (
    RealRLRunner,
    RealRunnerStartupError,
)
from rl_controller.real.state_machine import RuntimeState, RuntimeTransitionError
from rl_controller.real.types import ImuState


class FailingImuBackend:
    def __init__(self) -> None:
        self.closed = False

    def open(self) -> None:
        raise RuntimeError('injected IMU open failure')

    def read(self) -> ImuState:
        raise RuntimeError('IMU is unavailable')

    def close(self) -> None:
        self.closed = True


class SlowMotorBackend(MockMotorBackend):
    def cycle(self, command):
        time.sleep(self.config.state_timeout_s + 0.01)
        return super().cycle(command)


class BlockingPolicy(MockPolicy):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.inference_started = threading.Event()
        self.release_inference = threading.Event()

    def infer(self, observation):
        self.inference_started.set()
        self.release_inference.wait(timeout=1.0)
        return super().infer(observation)


class RealRLRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        source = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.config = replace(
            source,
            runtime=replace(
                source.runtime,
                move_to_default_duration_s=0.01,
            ),
        )
        self.runners: list[RealRLRunner] = []

    def tearDown(self) -> None:
        for runner in reversed(self.runners):
            if runner.state is RuntimeState.SHUTDOWN:
                continue
            try:
                runner.shutdown('test cleanup')
            except Exception:
                pass

    def make_runner(
        self,
        *,
        mock_action: np.ndarray | None = None,
        motor_backend=None,
        imu_backend=None,
        policy=None,
    ) -> RealRLRunner:
        runner = RealRLRunner.from_config(
            self.config,
            mock_action=mock_action,
            motor_backend=motor_backend,
            imu_backend=imu_backend,
            policy=policy,
        )
        self.runners.append(runner)
        return runner

    @staticmethod
    def wait_until(predicate, timeout_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return bool(predicate())

    def test_mock_runner_enters_rl_and_shuts_down_cleanly(self) -> None:
        action = np.full(12, 0.2, dtype=np.float32)
        runner = self.make_runner(mock_action=action)

        runner.start(ready_timeout_s=1.0)
        self.assertIs(runner.state, RuntimeState.READY)
        self.assertEqual(
            set(runner.live_thread_names),
            {
                'real-runtime-monitor',
                'real-motor-io',
                'real-imu-io',
                'real-policy',
            },
        )
        np.testing.assert_array_equal(
            runner.latest_joint_command.kp,
            np.zeros(12, dtype=np.float32),
        )

        runner.enable_rl()
        self.assertIs(runner.state, RuntimeState.MOVE_TO_DEFAULT)
        reached_terminal_state = runner.wait_for_state(
            (RuntimeState.RL, RuntimeState.FAULT),
            timeout=1.0,
        )
        self.assertTrue(reached_terminal_state)
        self.assertIs(runner.state, RuntimeState.RL, runner.fault_reason)
        np.testing.assert_allclose(runner.last_action, action)
        np.testing.assert_allclose(
            runner.latest_joint_command.position,
            self.config.robot.default_angles
            + action * self.config.robot.action_scale,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            runner.latest_joint_command.kp,
            np.full(12, 20.0, dtype=np.float32),
        )

        previous_policy_iterations = runner.worker_iterations['policy']
        clipped_command = runner.set_velocity_command([2.0, -2.0, 0.5])
        np.testing.assert_allclose(clipped_command, [1.0, -1.0, 0.5])
        self.assertTrue(
            self.wait_until(
                lambda: runner.worker_iterations['policy']
                > previous_policy_iterations
            )
        )
        np.testing.assert_allclose(
            runner.observation_history.last_one_step_observation[:3],
            [2.0, -2.0, 0.125],
            atol=1e-6,
        )
        self.assertTrue(
            all(count > 0 for count in runner.worker_iterations.values())
        )

        runner.shutdown()

        self.assertIs(runner.state, RuntimeState.SHUTDOWN)
        self.assertEqual(runner.live_thread_names, ())
        self.assertTrue(runner.motor_backend.safe_stopped)
        self.assertFalse(runner.motor_backend.is_open)
        self.assertFalse(runner.imu_backend.is_open)
        self.assertEqual(
            [transition.current for transition in runner.state_machine.transitions],
            [
                RuntimeState.READY,
                RuntimeState.MOVE_TO_DEFAULT,
                RuntimeState.RL,
                RuntimeState.SHUTDOWN,
            ],
        )

    def test_policy_fault_stops_workers_and_accepts_repeated_faults(self) -> None:
        runner = self.make_runner()
        runner.start()
        self.assertIsInstance(runner.policy.backend, MockPolicy)
        runner.policy.backend.inject_nan(3)

        runner.enable_rl()
        self.assertTrue(runner.wait_for_state(RuntimeState.FAULT, timeout=1.0))

        self.assertIn('finite values', runner.fault_reason or '')
        self.assertTrue(runner.motor_backend.safe_stopped)
        self.assertTrue(
            self.wait_until(lambda: runner.live_thread_names == ())
        )
        self.assertEqual(len(runner.worker_failures), 1)
        self.assertEqual(
            runner.worker_failures[0].worker_name,
            'real-policy',
        )

        runner.report_fault('second injected fault')
        self.assertEqual(
            [fault.reason for fault in runner.state_machine.faults][-1],
            'second injected fault',
        )
        runner.shutdown()
        self.assertIs(runner.state, RuntimeState.SHUTDOWN)

    def test_startup_failure_closes_already_open_motor_backend(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=self.config.robot.default_angles,
        )
        imu = FailingImuBackend()
        runner = self.make_runner(motor_backend=motor, imu_backend=imu)

        with self.assertRaisesRegex(
            RealRunnerStartupError,
            'injected IMU open failure',
        ):
            runner.start()

        self.assertIs(runner.state, RuntimeState.FAULT)
        self.assertFalse(motor.is_open)
        self.assertTrue(motor.safe_stopped)
        self.assertTrue(imu.closed)

        runner.shutdown()
        self.assertTrue(imu.closed)
        self.assertIs(runner.state, RuntimeState.SHUTDOWN)

    def test_commands_and_lifecycle_calls_are_validated(self) -> None:
        runner = self.make_runner()
        with self.assertRaisesRegex(
            RuntimeTransitionError,
            'only be enabled from READY',
        ):
            runner.enable_rl()
        with self.assertRaisesRegex(ValueError, r'shape \(3,\)'):
            runner.set_velocity_command([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, 'finite'):
            runner.set_velocity_command([0.0, np.nan, 0.0])

        runner.start()
        with self.assertRaisesRegex(
            RealRunnerStartupError,
            'only start from INIT',
        ):
            runner.start()
        runner.shutdown()
        runner.shutdown()

    def test_motor_temperature_and_error_code_trigger_safe_stop(self) -> None:
        fault_cases = (
            ('temperature', lambda motor: motor.inject_temperature(81.0, 3)),
            ('error code 7', lambda motor: motor.inject_error(7, 5)),
        )
        for expected_message, inject in fault_cases:
            with self.subTest(expected_message=expected_message):
                runner = self.make_runner()
                runner.start()
                inject(runner.motor_backend)

                self.assertTrue(
                    runner.wait_for_state(RuntimeState.FAULT, timeout=1.0)
                )
                self.assertIn(expected_message, runner.fault_reason or '')
                self.assertTrue(runner.motor_backend.safe_stopped)
                runner.shutdown()

    def test_stale_state_and_excessive_tilt_trigger_safe_stop(self) -> None:
        runner = self.make_runner()
        runner.start()
        runner.motor_backend.inject_stale_state(0.04)

        self.assertTrue(runner.wait_for_state(RuntimeState.FAULT, timeout=1.0))
        self.assertIn('motor feedback is', runner.fault_reason or '')
        self.assertTrue(runner.motor_backend.safe_stopped)
        runner.shutdown()

        tilted_runner = self.make_runner()
        tilted_runner.start()
        tilted_runner.imu_backend.inject_fall(roll_rad=1.3)

        self.assertTrue(
            tilted_runner.wait_for_state(RuntimeState.FAULT, timeout=1.0)
        )
        self.assertIn('body tilt exceeds', tilted_runner.fault_reason or '')
        self.assertTrue(tilted_runner.motor_backend.safe_stopped)
        tilted_runner.shutdown()

    def test_large_policy_target_step_is_never_sent_repeatedly(self) -> None:
        runner = self.make_runner()
        runner.start()
        runner.enable_rl()
        self.assertTrue(
            runner.wait_for_state(RuntimeState.RL, timeout=1.0),
            runner.fault_reason,
        )
        runner.policy.backend.set_action(
            np.full(self.config.robot.num_actions, 0.4, dtype=np.float32)
        )

        self.assertTrue(runner.wait_for_state(RuntimeState.FAULT, timeout=1.0))
        self.assertIn('target step', runner.fault_reason or '')
        self.assertTrue(runner.motor_backend.safe_stopped)
        motor_iterations = runner.worker_iterations['motor']
        time.sleep(0.03)
        self.assertEqual(
            runner.worker_iterations['motor'],
            motor_iterations,
        )
        runner.shutdown()

    def test_motor_cycle_deadline_failure_prevents_startup(self) -> None:
        motor = SlowMotorBackend(
            self.config.motors,
            initial_position=self.config.robot.default_angles,
        )
        runner = self.make_runner(motor_backend=motor)

        with self.assertRaisesRegex(
            RealRunnerStartupError,
            'motor.*exceeded',
        ):
            runner.start(ready_timeout_s=1.0)

        self.assertIs(runner.state, RuntimeState.FAULT)
        self.assertTrue(self.wait_until(lambda: motor.safe_stopped))
        runner.shutdown()

    def test_policy_watchdog_stops_motors_before_inference_returns(self) -> None:
        backend = BlockingPolicy(self.config.policy)
        policy = PolicyExecutor(
            backend,
            self.config.policy,
            max_inference_time_s=1.0,
        )
        runner = self.make_runner(policy=policy)
        runner.start()
        runner.enable_rl()

        self.assertTrue(backend.inference_started.wait(timeout=1.0))
        self.assertTrue(runner.wait_for_state(RuntimeState.FAULT, timeout=0.2))
        self.assertIn('real-policy exceeded', runner.fault_reason or '')
        self.assertTrue(
            self.wait_until(lambda: runner.motor_backend.safe_stopped)
        )
        np.testing.assert_array_equal(
            runner.last_action,
            np.zeros(self.config.robot.num_actions, dtype=np.float32),
        )

        backend.release_inference.set()
        self.assertTrue(
            self.wait_until(
                lambda: 'real-policy' not in runner.live_thread_names
            )
        )
        np.testing.assert_array_equal(
            runner.last_action,
            np.zeros(self.config.robot.num_actions, dtype=np.float32),
        )
        runner.shutdown()


if __name__ == '__main__':
    unittest.main()
