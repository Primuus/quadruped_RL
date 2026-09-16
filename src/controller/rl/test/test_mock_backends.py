from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import (
    JointCommand,
    MockImuBackend,
    MockMotorBackend,
    MockPolicy,
    RealDeploymentConfig,
    RealRobotState,
)
from rl_controller.real.hardware import BackendError, ImuBackend, MotorBackend


class FakeClock:
    def __init__(self, initial_time: float = 100.0) -> None:
        self.current_time = float(initial_time)

    def __call__(self) -> float:
        return self.current_time

    def advance(self, duration_s: float) -> None:
        self.current_time += float(duration_s)


class MockComponentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.clock = FakeClock()

    def make_command(
        self,
        position: np.ndarray | None = None,
    ) -> JointCommand:
        num_joints = self.config.robot.num_actions
        if position is None:
            position = np.full(num_joints, 0.5, dtype=np.float32)
        return JointCommand(
            position=position,
            velocity=np.zeros(num_joints, dtype=np.float32),
            kp=np.asarray(
                [joint.kp for joint in self.config.motors.joints],
                dtype=np.float32,
            ),
            kd=np.asarray(
                [joint.kd for joint in self.config.motors.joints],
                dtype=np.float32,
            ),
            torque=np.zeros(num_joints, dtype=np.float32),
            timestamp=self.clock(),
        )


class MockMotorBackendTest(MockComponentTestCase):
    def test_requires_open_and_matches_protocol(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=self.config.robot.default_angles,
            clock=self.clock,
        )
        self.assertIsInstance(motor, MotorBackend)

        with self.assertRaisesRegex(BackendError, 'not open'):
            motor.cycle(self.make_command())

        motor.open()
        self.assertTrue(motor.is_open)
        motor.close()
        self.assertFalse(motor.is_open)

    def test_follows_target_at_limited_velocity(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            max_velocity_rad_s=1.0,
            clock=self.clock,
        )
        motor.open()
        self.clock.advance(0.02)

        state = motor.cycle(self.make_command())

        np.testing.assert_allclose(state.position, 0.02, atol=1e-6)
        np.testing.assert_allclose(state.velocity, 1.0, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(state.torque)))
        self.assertEqual(state.position.dtype, np.float32)

    def test_default_velocity_limit_matches_per_cycle_target_step(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            clock=self.clock,
        )
        motor.open()
        self.clock.advance(1.0 / self.config.motors.io_frequency_hz)

        state = motor.cycle(self.make_command())

        expected_step = np.asarray(
            [
                joint.max_target_step_rad
                for joint in self.config.motors.joints
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(state.position, expected_step, atol=1e-6)

    def test_safe_stop_holds_position_and_zeroes_output(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            max_velocity_rad_s=1.0,
            clock=self.clock,
        )
        motor.open()
        self.clock.advance(0.02)
        moving_state = motor.cycle(self.make_command())

        motor.safe_stop()
        self.clock.advance(0.10)
        stopped_state = motor.cycle(self.make_command())

        self.assertTrue(motor.safe_stopped)
        np.testing.assert_allclose(stopped_state.position, moving_state.position)
        np.testing.assert_array_equal(stopped_state.velocity, np.zeros(12))
        np.testing.assert_array_equal(stopped_state.torque, np.zeros(12))

    def test_injects_and_clears_motor_faults(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            clock=self.clock,
        )
        motor.open()
        motor.inject_stale_state(0.2)
        motor.inject_nan('position', 2)
        motor.inject_temperature(95.0, 4)
        motor.inject_error(17, 6)
        motor.inject_invalid_state()
        self.clock.advance(0.01)

        faulty = motor.cycle(self.make_command())

        self.assertAlmostEqual(self.clock() - faulty.timestamp, 0.2)
        self.assertTrue(np.isnan(faulty.position[2]))
        self.assertEqual(float(faulty.temperature[4]), 95.0)
        self.assertEqual(int(faulty.error[6]), 17)
        self.assertFalse(faulty.valid)

        motor.clear_faults()
        self.clock.advance(0.01)
        recovered = motor.cycle(self.make_command())
        self.assertTrue(np.all(np.isfinite(recovered.position)))
        self.assertTrue(np.all(recovered.temperature == 25.0))
        self.assertTrue(np.all(recovered.error == 0))
        self.assertTrue(recovered.valid)
        self.assertEqual(recovered.timestamp, self.clock())

    def test_rejects_wrong_command_shape_and_nonfinite_command(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            clock=self.clock,
        )
        motor.open()

        short_command = JointCommand(
            position=np.zeros(11),
            velocity=np.zeros(11),
            kp=np.zeros(11),
            kd=np.zeros(11),
            torque=np.zeros(11),
            timestamp=self.clock(),
        )
        with self.assertRaisesRegex(ValueError, '11 joints, expected 12'):
            motor.cycle(short_command)

        invalid_position = np.zeros(12)
        invalid_position[0] = np.nan
        with self.assertRaisesRegex(BackendError, 'only finite values'):
            motor.cycle(self.make_command(invalid_position))

    def test_injects_and_clears_communication_failure(self) -> None:
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=np.zeros(12),
            clock=self.clock,
        )
        motor.open()
        motor.inject_communication_error('injected link loss')

        with self.assertRaisesRegex(BackendError, 'injected link loss'):
            motor.cycle(self.make_command())

        motor.clear_faults()
        state = motor.cycle(self.make_command())
        self.assertTrue(state.valid)


class MockImuBackendTest(MockComponentTestCase):
    def test_returns_body_frame_state_and_matches_protocol(self) -> None:
        imu = MockImuBackend(self.config.imu, clock=self.clock)
        self.assertIsInstance(imu, ImuBackend)
        with self.assertRaisesRegex(BackendError, 'not open'):
            imu.read()

        imu.open()
        state = imu.read()
        np.testing.assert_array_equal(
            state.quaternion_wxyz,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state.angular_velocity, np.zeros(3))
        self.assertTrue(state.valid)

    def test_injects_fall_stale_nan_and_invalid_state(self) -> None:
        imu = MockImuBackend(self.config.imu, clock=self.clock)
        imu.open()
        imu.inject_fall(roll_rad=math.pi / 2.0)
        imu.inject_stale_state(0.3)
        imu.inject_nan('angular_velocity', 1)
        imu.inject_invalid_state()

        state = imu.read()

        np.testing.assert_allclose(
            state.quaternion_wxyz,
            np.array(
                [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                dtype=np.float32,
            ),
            atol=1e-6,
        )
        self.assertTrue(np.isnan(state.angular_velocity[1]))
        self.assertAlmostEqual(self.clock() - state.timestamp, 0.3)
        self.assertFalse(state.valid)

        imu.clear_faults()
        recovered = imu.read()
        np.testing.assert_array_equal(
            recovered.quaternion_wxyz,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertTrue(np.all(np.isfinite(recovered.angular_velocity)))
        self.assertTrue(recovered.valid)

    def test_set_state_copies_validated_inputs(self) -> None:
        imu = MockImuBackend(self.config.imu, clock=self.clock)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        velocity = np.array([0.1, 0.2, 0.3])
        imu.set_state(
            quaternion_wxyz=quaternion,
            angular_velocity=velocity,
        )
        quaternion[0] = 0.0
        velocity[0] = 10.0
        imu.open()

        state = imu.read()
        self.assertEqual(float(state.quaternion_wxyz[0]), 1.0)
        np.testing.assert_allclose(state.angular_velocity, [0.1, 0.2, 0.3])


class MockPolicyTest(MockComponentTestCase):
    def test_returns_configured_action_as_an_isolated_copy(self) -> None:
        source = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        policy = MockPolicy(self.config.policy, action=source)
        source.fill(5.0)

        first = policy.infer(np.zeros(270, dtype=np.float32))
        first.fill(10.0)
        second = policy.infer(np.zeros(270, dtype=np.float32))

        np.testing.assert_allclose(
            second,
            np.linspace(-0.2, 0.2, 12, dtype=np.float32),
        )
        self.assertEqual(second.dtype, np.float32)

    def test_validates_observation_and_supports_nan_injection(self) -> None:
        policy = MockPolicy(self.config.policy)
        with self.assertRaisesRegex(ValueError, r'shape \(270,\)'):
            policy.infer(np.zeros(269))

        invalid_observation = np.zeros(270)
        invalid_observation[1] = np.inf
        with self.assertRaisesRegex(ValueError, 'only finite values'):
            policy.infer(invalid_observation)

        policy.inject_nan(3)
        faulty_action = policy.infer(np.zeros(270))
        self.assertTrue(np.isnan(faulty_action[3]))
        policy.clear_faults()
        self.assertTrue(np.all(np.isfinite(policy.infer(np.zeros(270)))))


class MockOfflineCycleTest(MockComponentTestCase):
    def test_executes_one_complete_policy_and_hardware_cycle(self) -> None:
        policy = MockPolicy(
            self.config.policy,
            action=np.full(12, 0.2, dtype=np.float32),
        )
        motor = MockMotorBackend(
            self.config.motors,
            initial_position=self.config.robot.default_angles,
            clock=self.clock,
        )
        imu = MockImuBackend(self.config.imu, clock=self.clock)
        motor.open()
        imu.open()

        action = policy.infer(
            np.zeros(self.config.policy.num_observations, dtype=np.float32)
        )
        target_position = (
            self.config.robot.default_angles
            + action * self.config.robot.action_scale
        )
        command = self.make_command(target_position)
        self.clock.advance(1.0 / self.config.policy.frequency_hz)
        joints = motor.cycle(command)
        imu_state = imu.read()
        robot_state = RealRobotState(
            joints=joints,
            imu=imu_state,
            valid=joints.valid and imu_state.valid,
            timestamp=max(joints.timestamp, imu_state.timestamp),
        )

        self.assertEqual(action.shape, (12,))
        self.assertEqual(robot_state.num_joints, 12)
        self.assertTrue(robot_state.valid)
        self.assertTrue(np.all(np.isfinite(robot_state.joints.position)))
        self.assertTrue(np.all(np.isfinite(robot_state.imu.quaternion_wxyz)))

    def test_mock_imports_do_not_load_hardware_or_onnx_modules(self) -> None:
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(RL_PACKAGE_ROOT)
        script = (
            'import sys\n'
            'from rl_controller.real import MockImuBackend, MockMotorBackend, MockPolicy\n'
            "for name in ('mujoco', 'onnxruntime', 'unitree_actuator_sdk'):\n"
            '    assert name not in sys.modules, name\n'
        )
        result = subprocess.run(
            [sys.executable, '-c', script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
