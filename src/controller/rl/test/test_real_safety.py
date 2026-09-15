from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import (
    ImuState,
    JointCommand,
    JointState,
    RealDeploymentConfig,
    RealRobotState,
    RealSafetyManager,
    RuntimeState,
    SafetyViolationError,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class RealSafetyManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.clock = FakeClock()
        self.safety = RealSafetyManager(self.config, clock=self.clock)
        self.num_joints = self.config.robot.num_actions

    def make_joint_state(
        self,
        *,
        position=None,
        velocity=None,
        temperature=None,
        error=None,
        valid: bool = True,
        age_s: float = 0.0,
    ) -> JointState:
        zeros = np.zeros(self.num_joints, dtype=np.float32)
        return JointState(
            position=zeros if position is None else position,
            velocity=zeros if velocity is None else velocity,
            torque=zeros,
            temperature=(
                np.full(self.num_joints, 25.0, dtype=np.float32)
                if temperature is None
                else temperature
            ),
            error=(
                np.zeros(self.num_joints, dtype=np.int64)
                if error is None
                else error
            ),
            valid=valid,
            timestamp=self.clock.value - age_s,
        )

    def make_imu_state(
        self,
        *,
        quaternion=None,
        angular_velocity=None,
        valid: bool = True,
        age_s: float = 0.0,
    ) -> ImuState:
        return ImuState(
            quaternion_wxyz=(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                if quaternion is None
                else quaternion
            ),
            angular_velocity=(
                np.zeros(3, dtype=np.float32)
                if angular_velocity is None
                else angular_velocity
            ),
            valid=valid,
            timestamp=self.clock.value - age_s,
        )

    def make_robot_state(
        self,
        *,
        joints: JointState | None = None,
        imu: ImuState | None = None,
        valid: bool = True,
    ) -> RealRobotState:
        joints = self.make_joint_state() if joints is None else joints
        imu = self.make_imu_state() if imu is None else imu
        return RealRobotState(
            joints=joints,
            imu=imu,
            valid=valid,
            timestamp=min(joints.timestamp, imu.timestamp),
        )

    def make_command(
        self,
        position=None,
        *,
        active: bool = True,
        kp=None,
        age_s: float = 0.0,
    ) -> JointCommand:
        zeros = np.zeros(self.num_joints, dtype=np.float32)
        if position is None:
            position = zeros
        if kp is None:
            kp = (
                np.full(self.num_joints, 20.0, dtype=np.float32)
                if active
                else zeros
            )
        return JointCommand(
            position=position,
            velocity=zeros,
            kp=kp,
            kd=(
                np.full(self.num_joints, 0.5, dtype=np.float32)
                if active
                else zeros
            ),
            torque=zeros,
            timestamp=self.clock.value - age_s,
        )

    def test_accepts_nominal_states_actions_and_commands(self) -> None:
        joints = self.make_joint_state()
        imu = self.make_imu_state()
        state = self.make_robot_state(joints=joints, imu=imu)

        self.safety.validate_joint_feedback(
            joints,
            source_timestamps={'left': 100.0, 'right': 100.0},
        )
        self.safety.validate_imu_feedback(imu)
        self.safety.validate_robot_state(state)
        np.testing.assert_array_equal(
            self.safety.validate_action(np.zeros(self.num_joints)),
            np.zeros(self.num_joints, dtype=np.float32),
        )
        self.safety.validate_command_before_send(
            self.make_command(active=False),
            RuntimeState.INIT,
            None,
        )
        self.safety.validate_command_before_send(
            self.make_command(),
            RuntimeState.MOVE_TO_DEFAULT,
            state,
        )
        self.assertIsNotNone(self.safety.last_active_target)

    def test_rejects_nonfinite_or_excessive_action_and_target(self) -> None:
        action = np.zeros(self.num_joints, dtype=np.float32)
        action[2] = np.nan
        with self.assertRaisesRegex(SafetyViolationError, 'finite'):
            self.safety.validate_action(action)

        action[2] = self.config.policy.clip_actions + 1.0
        with self.assertRaisesRegex(SafetyViolationError, 'configured limit'):
            self.safety.validate_action(action)

        target = np.zeros(self.num_joints, dtype=np.float32)
        target[4] = 4.0
        with self.assertRaisesRegex(SafetyViolationError, 'limits are'):
            self.safety.validate_target_position(target)

    def test_rejects_invalid_command_state_and_negative_gains(self) -> None:
        state = self.make_robot_state()
        with self.assertRaisesRegex(SafetyViolationError, 'forbidden in READY'):
            self.safety.validate_command_before_send(
                self.make_command(),
                RuntimeState.READY,
                state,
            )
        with self.assertRaisesRegex(SafetyViolationError, 'forbidden in FAULT'):
            self.safety.validate_command_before_send(
                self.make_command(active=False),
                RuntimeState.FAULT,
                state,
            )

        kp = np.full(self.num_joints, 20.0, dtype=np.float32)
        kp[1] = -1.0
        with self.assertRaisesRegex(SafetyViolationError, 'must not be negative'):
            self.safety.validate_command_before_send(
                self.make_command(kp=kp),
                RuntimeState.RL,
                state,
            )

        command = self.make_command()
        velocity = np.array(command.velocity, copy=True)
        velocity[0] = np.nan
        nonfinite_command = JointCommand(
            position=command.position,
            velocity=velocity,
            kp=command.kp,
            kd=command.kd,
            torque=command.torque,
            timestamp=command.timestamp,
        )
        with self.assertRaisesRegex(SafetyViolationError, 'velocity.*finite'):
            self.safety.validate_command_before_send(
                nonfinite_command,
                RuntimeState.RL,
                state,
            )

    def test_rejects_large_target_step_and_resets_on_passive_command(self) -> None:
        state = self.make_robot_state()
        self.safety.validate_command_before_send(
            self.make_command(),
            RuntimeState.MOVE_TO_DEFAULT,
            state,
        )

        target = np.zeros(self.num_joints, dtype=np.float32)
        target[0] = 0.051
        with self.assertRaisesRegex(SafetyViolationError, 'target step'):
            self.safety.validate_command_before_send(
                self.make_command(target),
                RuntimeState.MOVE_TO_DEFAULT,
                state,
            )

        self.safety.validate_command_before_send(
            self.make_command(active=False),
            RuntimeState.READY,
            None,
        )
        self.assertIsNone(self.safety.last_active_target)

    def test_rejects_invalid_stale_hot_or_faulted_motor_feedback(self) -> None:
        cases = {
            'marked invalid': self.make_joint_state(valid=False),
            'old': self.make_joint_state(age_s=0.031),
            'temperature': self.make_joint_state(
                temperature=np.full(
                    self.num_joints,
                    self.config.safety.max_motor_temperature_c + 1.0,
                    dtype=np.float32,
                )
            ),
            'error code': self.make_joint_state(
                error=np.ones(self.num_joints, dtype=np.int64)
            ),
            'finite': self.make_joint_state(
                velocity=np.full(self.num_joints, np.nan, dtype=np.float32)
            ),
        }
        for expected_message, state in cases.items():
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(
                    SafetyViolationError,
                    expected_message,
                ):
                    self.safety.validate_joint_feedback(state)

        with self.assertRaisesRegex(SafetyViolationError, 'motor source'):
            self.safety.validate_joint_feedback(
                self.make_joint_state(),
                source_timestamps={'slow_bus': 99.96},
            )

    def test_rejects_invalid_stale_or_tilted_imu_feedback(self) -> None:
        roll = self.config.safety.max_abs_roll_pitch_rad + 0.1
        tilted = np.asarray(
            [np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0],
            dtype=np.float32,
        )
        cases = {
            'marked invalid': self.make_imu_state(valid=False),
            'old': self.make_imu_state(age_s=0.031),
            'finite': self.make_imu_state(
                angular_velocity=np.asarray([np.nan, 0.0, 0.0])
            ),
            'norm': self.make_imu_state(
                quaternion=np.asarray([0.5, 0.0, 0.0, 0.0])
            ),
            'tilt': self.make_imu_state(quaternion=tilted),
        }
        for expected_message, state in cases.items():
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(
                    SafetyViolationError,
                    expected_message,
                ):
                    self.safety.validate_imu_feedback(state)

    def test_requires_fresh_state_for_active_control(self) -> None:
        with self.assertRaisesRegex(SafetyViolationError, 'fresh robot state'):
            self.safety.validate_command_before_send(
                self.make_command(),
                RuntimeState.RL,
                None,
            )
        with self.assertRaisesRegex(SafetyViolationError, 'robot state'):
            self.safety.validate_robot_state(
                self.make_robot_state(valid=False)
            )
        with self.assertRaisesRegex(SafetyViolationError, 'active joint command'):
            self.safety.validate_command_before_send(
                self.make_command(age_s=0.051),
                RuntimeState.RL,
                self.make_robot_state(),
            )

    def test_rejects_motor_imu_and_policy_cycle_deadlines(self) -> None:
        self.safety.validate_motor_cycle_duration(0.01)
        self.safety.validate_imu_cycle_duration(0.01)
        self.safety.validate_policy_cycle_duration(0.01)

        cases = (
            (self.safety.validate_motor_cycle_duration, 0.031, 'motor'),
            (self.safety.validate_imu_cycle_duration, 0.031, 'IMU'),
            (self.safety.validate_policy_cycle_duration, 0.021, 'policy'),
        )
        for validate, duration, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(
                    SafetyViolationError,
                    expected_message,
                ):
                    validate(duration)


if __name__ == '__main__':
    unittest.main()
