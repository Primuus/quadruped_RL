from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from ..input.keyboard_controller import REQUIRED_BINDING_ACTIONS


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_POLICY_BACKENDS = frozenset({'onnx'})
SUPPORTED_MOTOR_BACKENDS = frozenset({'unitree_m8010'})
SUPPORTED_IMU_BACKENDS = frozenset({'yesense_yis130'})
SUPPORTED_QUATERNION_ORDERS = frozenset({'wxyz', 'xyzw'})
SUPPORTED_ANGULAR_VELOCITY_UNITS = frozenset({'rad/s', 'deg/s'})
SUPPORTED_FAULT_ACTIONS = frozenset({'safe_stop'})

REQUIRED_KEYBOARD_ACTIONS = REQUIRED_BINDING_ACTIONS

_PLACEHOLDER_VALUES = frozenset(
    {
        'change_me',
        'placeholder',
        'required',
        'todo',
        '<required>',
        '<device>',
        '<model>',
    }
)


class RealConfigError(ValueError):
    """Raised when a real-deployment configuration violates its schema."""


@dataclass(frozen=True, slots=True)
class RealRobotConfig:
    name: str
    joint_names: tuple[str, ...]
    default_angles: np.ndarray
    action_scale: float
    num_one_step_obs: int
    history_length: int
    num_obs: int

    @property
    def num_actions(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True, slots=True)
class RealObservationScales:
    angular_velocity: float
    dof_position: float
    dof_velocity: float
    command: np.ndarray


@dataclass(frozen=True, slots=True)
class RealPolicyConfig:
    backend: str
    model_path: Path | None
    num_observations: int
    num_actions: int
    frequency_hz: float
    clip_observations: float
    clip_actions: float
    observation_scales: RealObservationScales


@dataclass(frozen=True, slots=True)
class MotorBusConfig:
    name: str
    device: str | None
    baud_rate: int | None


@dataclass(frozen=True, slots=True)
class MotorJointConfig:
    joint_name: str
    bus: str
    motor_id: int
    direction: int
    zero_offset_rad: float
    gear_ratio: float
    position_limits_rad: np.ndarray
    max_target_step_rad: float
    kp: float
    kd: float

    @property
    def position_min_rad(self) -> float:
        return float(self.position_limits_rad[0])

    @property
    def position_max_rad(self) -> float:
        return float(self.position_limits_rad[1])


@dataclass(frozen=True, slots=True)
class RealMotorConfig:
    backend: str
    buses: tuple[MotorBusConfig, ...]
    joints: tuple[MotorJointConfig, ...]
    io_frequency_hz: float
    state_timeout_s: float


@dataclass(frozen=True, slots=True)
class RealImuConfig:
    backend: str
    device: str | None
    baud_rate: int | None
    quaternion_order: str | None
    mounting_quaternion_wxyz: np.ndarray | None
    angular_velocity_unit: str | None
    state_timeout_s: float


@dataclass(frozen=True, slots=True)
class RealSafetyConfig:
    max_state_age_s: float
    max_motor_temperature_c: float
    max_abs_roll_pitch_rad: float
    max_policy_inference_time_s: float
    fault_action: str


@dataclass(frozen=True, slots=True)
class RealRuntimeConfig:
    move_to_default_duration_s: float
    history_warmup_frames: int
    shutdown_timeout_s: float


@dataclass(frozen=True, slots=True)
class RealKeyboardConfig:
    bindings: Mapping[str, str]
    command_limits: np.ndarray
    speed_levels: np.ndarray
    default_speed_level: int


@dataclass(frozen=True, slots=True)
class RealDeploymentConfig:
    source_path: Path
    package_root: Path
    schema_version: int
    robot: RealRobotConfig
    policy: RealPolicyConfig
    motors: RealMotorConfig
    imu: RealImuConfig
    safety: RealSafetyConfig
    runtime: RealRuntimeConfig
    keyboard: RealKeyboardConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'RealDeploymentConfig':
        return load_real_config(yaml_path)

    def validate(self) -> None:
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise RealConfigError(
                f'Unsupported schema_version={self.schema_version}; '
                f'expected {SUPPORTED_SCHEMA_VERSION}'
            )

        self._validate_robot_and_policy()
        self._validate_motor_mapping()
        self._validate_imu()
        self._validate_safety_and_runtime()
        self._validate_keyboard()
        self._validate_backend_combinations()

    def _validate_robot_and_policy(self) -> None:
        robot = self.robot
        policy = self.policy

        if not robot.joint_names:
            raise RealConfigError('robot.joint_names must not be empty')
        if len(set(robot.joint_names)) != robot.num_actions:
            raise RealConfigError('robot.joint_names must contain unique names')
        if robot.default_angles.shape != (robot.num_actions,):
            raise RealConfigError(
                'robot.default_angles length must match robot.joint_names'
            )
        _require_positive(robot.action_scale, 'robot.action_scale')
        if robot.history_length <= 0:
            raise RealConfigError('robot.history_length must be positive')

        expected_one_step_obs = 9 + 3 * robot.num_actions
        if robot.num_one_step_obs != expected_one_step_obs:
            raise RealConfigError(
                'robot.num_one_step_obs must equal '
                f'9 + 3 * num_actions ({expected_one_step_obs}), '
                f'got {robot.num_one_step_obs}'
            )
        expected_num_obs = robot.num_one_step_obs * robot.history_length
        if robot.num_obs != expected_num_obs:
            raise RealConfigError(
                'robot.num_obs must equal '
                f'num_one_step_obs * history_length ({expected_num_obs}), '
                f'got {robot.num_obs}'
            )

        if policy.num_actions != robot.num_actions:
            raise RealConfigError(
                'policy.num_actions must match the number of robot.joint_names'
            )
        if policy.num_observations != robot.num_obs:
            raise RealConfigError(
                'policy.num_observations must match robot.num_obs'
            )
        _require_positive(policy.frequency_hz, 'policy.frequency_hz')
        _require_positive(
            policy.clip_observations,
            'policy.clip_observations',
        )
        _require_positive(policy.clip_actions, 'policy.clip_actions')

        scales = policy.observation_scales
        _require_positive(
            scales.angular_velocity,
            'policy.observation_scales.angular_velocity',
        )
        _require_positive(
            scales.dof_position,
            'policy.observation_scales.dof_position',
        )
        _require_positive(
            scales.dof_velocity,
            'policy.observation_scales.dof_velocity',
        )
        if scales.command.shape != (3,) or np.any(scales.command <= 0.0):
            raise RealConfigError(
                'policy.observation_scales.command must contain three '
                'finite positive values'
            )

        if policy.model_path is None:
            raise RealConfigError(
                'policy.model_path is required when policy.backend=onnx'
            )
        if policy.model_path.suffix.lower() != '.onnx':
            raise RealConfigError(
                'policy.model_path must point to an .onnx file'
            )
        if not policy.model_path.is_file():
            raise RealConfigError(
                f'policy.model_path does not exist: {policy.model_path}'
            )

    def _validate_motor_mapping(self) -> None:
        motors = self.motors
        robot = self.robot

        if not motors.buses:
            raise RealConfigError('motors.buses must not be empty')
        bus_names = tuple(bus.name for bus in motors.buses)
        if len(set(bus_names)) != len(bus_names):
            raise RealConfigError('motors.buses names must be unique')

        if len(motors.joints) != robot.num_actions:
            raise RealConfigError(
                'motors.joints length must match robot.joint_names'
            )
        motor_joint_names = tuple(joint.joint_name for joint in motors.joints)
        if motor_joint_names != robot.joint_names:
            raise RealConfigError(
                'motors.joints must use the exact robot.joint_names order'
            )

        known_buses = set(bus_names)
        motor_addresses: set[tuple[str, int]] = set()
        for index, joint in enumerate(motors.joints):
            path = f'motors.joints[{index}]'
            if joint.bus not in known_buses:
                raise RealConfigError(
                    f'{path}.bus references unknown bus {joint.bus!r}'
                )
            address = (joint.bus, joint.motor_id)
            if address in motor_addresses:
                raise RealConfigError(
                    f'{path} duplicates motor ID {joint.motor_id} '
                    f'on bus {joint.bus!r}'
                )
            motor_addresses.add(address)

            if not 0 <= joint.motor_id <= 255:
                raise RealConfigError(
                    f'{path}.motor_id must be in [0, 255]'
                )
            if joint.direction not in (-1, 1):
                raise RealConfigError(
                    f'{path}.direction must be -1 or 1'
                )
            _require_positive(joint.gear_ratio, f'{path}.gear_ratio')
            if joint.position_min_rad >= joint.position_max_rad:
                raise RealConfigError(
                    f'{path}.position_limits_rad must be [min, max] with min < max'
                )
            _require_positive(
                joint.max_target_step_rad,
                f'{path}.max_target_step_rad',
            )
            if (
                joint.max_target_step_rad
                > joint.position_max_rad - joint.position_min_rad
            ):
                raise RealConfigError(
                    f'{path}.max_target_step_rad cannot exceed the joint range'
                )
            _require_nonnegative(joint.kp, f'{path}.kp')
            _require_nonnegative(joint.kd, f'{path}.kd')

            default_angle = float(robot.default_angles[index])
            if not joint.position_min_rad <= default_angle <= joint.position_max_rad:
                raise RealConfigError(
                    f'robot.default_angles[{index}]={default_angle} is outside '
                    f'{path}.position_limits_rad'
                )

        _require_positive(
            motors.io_frequency_hz,
            'motors.io_frequency_hz',
        )
        _require_positive(motors.state_timeout_s, 'motors.state_timeout_s')
        if motors.io_frequency_hz < self.policy.frequency_hz:
            raise RealConfigError(
                'motors.io_frequency_hz must be greater than or equal to '
                'policy.frequency_hz'
            )

    def _validate_imu(self) -> None:
        imu = self.imu
        _require_positive(imu.state_timeout_s, 'imu.state_timeout_s')

        if imu.quaternion_order not in SUPPORTED_QUATERNION_ORDERS:
            raise RealConfigError(
                'imu.quaternion_order must be one of '
                f'{sorted(SUPPORTED_QUATERNION_ORDERS)}'
            )
        if imu.angular_velocity_unit not in SUPPORTED_ANGULAR_VELOCITY_UNITS:
            raise RealConfigError(
                'imu.angular_velocity_unit must be one of '
                f'{sorted(SUPPORTED_ANGULAR_VELOCITY_UNITS)}'
            )
        if (
            imu.mounting_quaternion_wxyz is None
            or imu.mounting_quaternion_wxyz.shape != (4,)
        ):
            raise RealConfigError(
                'imu.mounting_quaternion_wxyz must contain four values'
            )
        quaternion_norm = float(np.linalg.norm(imu.mounting_quaternion_wxyz))
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-3):
            raise RealConfigError(
                'imu.mounting_quaternion_wxyz must be normalized'
            )

    def _validate_safety_and_runtime(self) -> None:
        safety = self.safety
        runtime = self.runtime

        _require_positive(safety.max_state_age_s, 'safety.max_state_age_s')
        _require_positive(
            safety.max_motor_temperature_c,
            'safety.max_motor_temperature_c',
        )
        _require_positive(
            safety.max_abs_roll_pitch_rad,
            'safety.max_abs_roll_pitch_rad',
        )
        if safety.max_abs_roll_pitch_rad > np.pi:
            raise RealConfigError(
                'safety.max_abs_roll_pitch_rad cannot exceed pi'
            )
        _require_positive(
            safety.max_policy_inference_time_s,
            'safety.max_policy_inference_time_s',
        )
        if safety.max_policy_inference_time_s >= 1.0 / self.policy.frequency_hz:
            raise RealConfigError(
                'safety.max_policy_inference_time_s must be shorter than one '
                'policy period'
            )
        if safety.fault_action not in SUPPORTED_FAULT_ACTIONS:
            raise RealConfigError(
                f'safety.fault_action must be one of {sorted(SUPPORTED_FAULT_ACTIONS)}'
            )
        if safety.max_state_age_s < self.motors.state_timeout_s:
            raise RealConfigError(
                'safety.max_state_age_s cannot be shorter than '
                'motors.state_timeout_s'
            )
        if safety.max_state_age_s < self.imu.state_timeout_s:
            raise RealConfigError(
                'safety.max_state_age_s cannot be shorter than imu.state_timeout_s'
            )

        _require_positive(
            runtime.move_to_default_duration_s,
            'runtime.move_to_default_duration_s',
        )
        if runtime.history_warmup_frames < self.robot.history_length:
            raise RealConfigError(
                'runtime.history_warmup_frames must be at least '
                'robot.history_length'
            )
        _require_positive(
            runtime.shutdown_timeout_s,
            'runtime.shutdown_timeout_s',
        )

    def _validate_keyboard(self) -> None:
        keyboard = self.keyboard
        missing_actions = set(REQUIRED_KEYBOARD_ACTIONS) - set(keyboard.bindings)
        if missing_actions:
            raise RealConfigError(
                'keyboard.bindings is missing actions: '
                f'{", ".join(sorted(missing_actions))}'
            )

        configured_keys = [
            keyboard.bindings[action] for action in REQUIRED_KEYBOARD_ACTIONS
        ]
        if len(set(configured_keys)) != len(configured_keys):
            raise RealConfigError(
                'keyboard.bindings must assign a unique key to every action'
            )
        if not all(key.startswith('KEY_') for key in configured_keys):
            raise RealConfigError(
                'keyboard.bindings values must use Linux evdev KEY_* names'
            )

        if (
            keyboard.command_limits.shape != (3,)
            or np.any(keyboard.command_limits <= 0.0)
        ):
            raise RealConfigError(
                'keyboard.command_limits must contain three finite positive values'
            )
        if (
            keyboard.speed_levels.shape != (3,)
            or np.any(keyboard.speed_levels <= 0.0)
            or np.any(keyboard.speed_levels > 1.0)
            or np.any(np.diff(keyboard.speed_levels) <= 0.0)
        ):
            raise RealConfigError(
                'keyboard.speed_levels must be three strictly increasing '
                'values in (0, 1]'
            )
        if keyboard.default_speed_level not in (1, 2, 3):
            raise RealConfigError(
                'keyboard.default_speed_level must be 1, 2, or 3'
            )

    def _validate_backend_combinations(self) -> None:
        for index, bus in enumerate(self.motors.buses):
            _require_real_device(
                bus.device,
                f'motors.buses[{index}].device',
            )
            if bus.baud_rate is None:
                raise RealConfigError(
                    f'motors.buses[{index}].baud_rate is required when '
                    'motors.backend=unitree_m8010'
                )

        _require_real_device(self.imu.device, 'imu.device')
        if self.imu.baud_rate is None:
            raise RealConfigError(
                'imu.baud_rate is required when imu.backend=yesense_yis130'
            )


def load_real_config(yaml_path: str | Path) -> RealDeploymentConfig:
    config_path = Path(yaml_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Real deployment config not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    root = _as_mapping(raw, 'config')
    _require_fields(
        root,
        {
            'schema_version',
            'robot',
            'policy',
            'motors',
            'imu',
            'safety',
            'runtime',
            'keyboard',
        },
        'config',
    )

    package_root = _config_package_root(config_path)
    config = RealDeploymentConfig(
        source_path=config_path,
        package_root=package_root,
        schema_version=_as_int(root['schema_version'], 'schema_version'),
        robot=_parse_robot(root['robot']),
        policy=_parse_policy(root['policy'], package_root),
        motors=_parse_motors(root['motors']),
        imu=_parse_imu(root['imu']),
        safety=_parse_safety(root['safety']),
        runtime=_parse_runtime(root['runtime']),
        keyboard=_parse_keyboard(root['keyboard']),
    )
    config.validate()
    return config


def _parse_robot(value: Any) -> RealRobotConfig:
    raw = _as_mapping(value, 'robot')
    _require_fields(
        raw,
        {
            'name',
            'joint_names',
            'default_angles',
            'action_scale',
            'num_one_step_obs',
            'history_length',
            'num_obs',
        },
        'robot',
    )
    joint_names = _as_string_tuple(raw['joint_names'], 'robot.joint_names')
    return RealRobotConfig(
        name=_as_string(raw['name'], 'robot.name'),
        joint_names=joint_names,
        default_angles=_as_float_vector(
            raw['default_angles'],
            len(joint_names),
            'robot.default_angles',
        ),
        action_scale=_as_float(raw['action_scale'], 'robot.action_scale'),
        num_one_step_obs=_as_int(
            raw['num_one_step_obs'],
            'robot.num_one_step_obs',
        ),
        history_length=_as_int(
            raw['history_length'],
            'robot.history_length',
        ),
        num_obs=_as_int(raw['num_obs'], 'robot.num_obs'),
    )


def _parse_policy(value: Any, package_root: Path) -> RealPolicyConfig:
    raw = _as_mapping(value, 'policy')
    _require_fields(
        raw,
        {
            'backend',
            'model_path',
            'num_observations',
            'num_actions',
            'frequency_hz',
            'clip_observations',
            'clip_actions',
            'observation_scales',
        },
        'policy',
    )
    backend = _as_choice(
        raw['backend'],
        SUPPORTED_POLICY_BACKENDS,
        'policy.backend',
    )
    scales_raw = _as_mapping(
        raw['observation_scales'],
        'policy.observation_scales',
    )
    _require_fields(
        scales_raw,
        {'angular_velocity', 'dof_position', 'dof_velocity', 'command'},
        'policy.observation_scales',
    )

    return RealPolicyConfig(
        backend=backend,
        model_path=_as_optional_path(
            raw['model_path'],
            package_root,
            'policy.model_path',
        ),
        num_observations=_as_int(
            raw['num_observations'],
            'policy.num_observations',
        ),
        num_actions=_as_int(raw['num_actions'], 'policy.num_actions'),
        frequency_hz=_as_float(raw['frequency_hz'], 'policy.frequency_hz'),
        clip_observations=_as_float(
            raw['clip_observations'],
            'policy.clip_observations',
        ),
        clip_actions=_as_float(raw['clip_actions'], 'policy.clip_actions'),
        observation_scales=RealObservationScales(
            angular_velocity=_as_float(
                scales_raw['angular_velocity'],
                'policy.observation_scales.angular_velocity',
            ),
            dof_position=_as_float(
                scales_raw['dof_position'],
                'policy.observation_scales.dof_position',
            ),
            dof_velocity=_as_float(
                scales_raw['dof_velocity'],
                'policy.observation_scales.dof_velocity',
            ),
            command=_as_float_vector(
                scales_raw['command'],
                3,
                'policy.observation_scales.command',
            ),
        ),
    )


def _parse_motors(value: Any) -> RealMotorConfig:
    raw = _as_mapping(value, 'motors')
    _require_fields(
        raw,
        {'backend', 'buses', 'joints', 'io_frequency_hz', 'state_timeout_s'},
        'motors',
    )

    buses_raw = _as_list(raw['buses'], 'motors.buses')
    buses: list[MotorBusConfig] = []
    for index, item in enumerate(buses_raw):
        path = f'motors.buses[{index}]'
        bus_raw = _as_mapping(item, path)
        _require_fields(bus_raw, {'name', 'device', 'baud_rate'}, path)
        buses.append(
            MotorBusConfig(
                name=_as_string(bus_raw['name'], f'{path}.name'),
                device=_as_optional_string(
                    bus_raw['device'],
                    f'{path}.device',
                ),
                baud_rate=_as_optional_positive_int(
                    bus_raw['baud_rate'],
                    f'{path}.baud_rate',
                ),
            )
        )

    joints_raw = _as_list(raw['joints'], 'motors.joints')
    joints: list[MotorJointConfig] = []
    for index, item in enumerate(joints_raw):
        path = f'motors.joints[{index}]'
        joint_raw = _as_mapping(item, path)
        _require_fields(
            joint_raw,
            {
                'joint_name',
                'bus',
                'motor_id',
                'direction',
                'zero_offset_rad',
                'gear_ratio',
                'position_limits_rad',
                'max_target_step_rad',
                'kp',
                'kd',
            },
            path,
        )
        joints.append(
            MotorJointConfig(
                joint_name=_as_string(
                    joint_raw['joint_name'],
                    f'{path}.joint_name',
                ),
                bus=_as_string(joint_raw['bus'], f'{path}.bus'),
                motor_id=_as_int(joint_raw['motor_id'], f'{path}.motor_id'),
                direction=_as_int(
                    joint_raw['direction'],
                    f'{path}.direction',
                ),
                zero_offset_rad=_as_float(
                    joint_raw['zero_offset_rad'],
                    f'{path}.zero_offset_rad',
                ),
                gear_ratio=_as_float(
                    joint_raw['gear_ratio'],
                    f'{path}.gear_ratio',
                ),
                position_limits_rad=_as_float_vector(
                    joint_raw['position_limits_rad'],
                    2,
                    f'{path}.position_limits_rad',
                ),
                max_target_step_rad=_as_float(
                    joint_raw['max_target_step_rad'],
                    f'{path}.max_target_step_rad',
                ),
                kp=_as_float(joint_raw['kp'], f'{path}.kp'),
                kd=_as_float(joint_raw['kd'], f'{path}.kd'),
            )
        )

    return RealMotorConfig(
        backend=_as_choice(
            raw['backend'],
            SUPPORTED_MOTOR_BACKENDS,
            'motors.backend',
        ),
        buses=tuple(buses),
        joints=tuple(joints),
        io_frequency_hz=_as_float(
            raw['io_frequency_hz'],
            'motors.io_frequency_hz',
        ),
        state_timeout_s=_as_float(
            raw['state_timeout_s'],
            'motors.state_timeout_s',
        ),
    )


def _parse_imu(value: Any) -> RealImuConfig:
    raw = _as_mapping(value, 'imu')
    _require_fields(
        raw,
        {
            'backend',
            'device',
            'baud_rate',
            'quaternion_order',
            'mounting_quaternion_wxyz',
            'angular_velocity_unit',
            'state_timeout_s',
        },
        'imu',
    )
    return RealImuConfig(
        backend=_as_choice(
            raw['backend'],
            SUPPORTED_IMU_BACKENDS,
            'imu.backend',
        ),
        device=_as_optional_string(raw['device'], 'imu.device'),
        baud_rate=_as_optional_positive_int(
            raw['baud_rate'],
            'imu.baud_rate',
        ),
        quaternion_order=_as_optional_string(
            raw['quaternion_order'],
            'imu.quaternion_order',
        ),
        mounting_quaternion_wxyz=_as_optional_float_vector(
            raw['mounting_quaternion_wxyz'],
            4,
            'imu.mounting_quaternion_wxyz',
        ),
        angular_velocity_unit=_as_optional_string(
            raw['angular_velocity_unit'],
            'imu.angular_velocity_unit',
        ),
        state_timeout_s=_as_float(
            raw['state_timeout_s'],
            'imu.state_timeout_s',
        ),
    )


def _parse_safety(value: Any) -> RealSafetyConfig:
    raw = _as_mapping(value, 'safety')
    _require_fields(
        raw,
        {
            'max_state_age_s',
            'max_motor_temperature_c',
            'max_abs_roll_pitch_rad',
            'max_policy_inference_time_s',
            'fault_action',
        },
        'safety',
    )
    return RealSafetyConfig(
        max_state_age_s=_as_float(
            raw['max_state_age_s'],
            'safety.max_state_age_s',
        ),
        max_motor_temperature_c=_as_float(
            raw['max_motor_temperature_c'],
            'safety.max_motor_temperature_c',
        ),
        max_abs_roll_pitch_rad=_as_float(
            raw['max_abs_roll_pitch_rad'],
            'safety.max_abs_roll_pitch_rad',
        ),
        max_policy_inference_time_s=_as_float(
            raw['max_policy_inference_time_s'],
            'safety.max_policy_inference_time_s',
        ),
        fault_action=_as_choice(
            raw['fault_action'],
            SUPPORTED_FAULT_ACTIONS,
            'safety.fault_action',
        ),
    )


def _parse_runtime(value: Any) -> RealRuntimeConfig:
    raw = _as_mapping(value, 'runtime')
    _require_fields(
        raw,
        {
            'move_to_default_duration_s',
            'history_warmup_frames',
            'shutdown_timeout_s',
        },
        'runtime',
    )
    return RealRuntimeConfig(
        move_to_default_duration_s=_as_float(
            raw['move_to_default_duration_s'],
            'runtime.move_to_default_duration_s',
        ),
        history_warmup_frames=_as_int(
            raw['history_warmup_frames'],
            'runtime.history_warmup_frames',
        ),
        shutdown_timeout_s=_as_float(
            raw['shutdown_timeout_s'],
            'runtime.shutdown_timeout_s',
        ),
    )


def _parse_keyboard(value: Any) -> RealKeyboardConfig:
    raw = _as_mapping(value, 'keyboard')
    _require_fields(
        raw,
        {
            'bindings',
            'command_limits',
            'speed_levels',
            'default_speed_level',
        },
        'keyboard',
    )
    bindings_raw = _as_mapping(raw['bindings'], 'keyboard.bindings')
    _require_fields(
        bindings_raw,
        set(REQUIRED_KEYBOARD_ACTIONS),
        'keyboard.bindings',
    )
    bindings = {
        action: _as_string(
            bindings_raw[action],
            f'keyboard.bindings.{action}',
        )
        for action in REQUIRED_KEYBOARD_ACTIONS
    }
    return RealKeyboardConfig(
        bindings=MappingProxyType(bindings),
        command_limits=_as_float_vector(
            raw['command_limits'],
            3,
            'keyboard.command_limits',
        ),
        speed_levels=_as_float_vector(
            raw['speed_levels'],
            3,
            'keyboard.speed_levels',
        ),
        default_speed_level=_as_int(
            raw['default_speed_level'],
            'keyboard.default_speed_level',
        ),
    )


def _config_package_root(config_path: Path) -> Path:
    if (
        config_path.parent.name == 'real'
        and config_path.parent.parent.name == 'config'
    ):
        return config_path.parents[2]
    return config_path.parent


def _require_fields(
    raw: Mapping[str, Any],
    required: set[str],
    path: str,
) -> None:
    keys = set(raw)
    missing = sorted(required - keys)
    if missing:
        raise RealConfigError(
            f'{path} is missing required fields: {", ".join(missing)}'
        )
    unknown = sorted(keys - required)
    if unknown:
        raise RealConfigError(
            f'{path} contains unknown fields: {", ".join(unknown)}'
        )


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        raise RealConfigError(f'{path} is required and cannot be null')
    if not isinstance(value, Mapping):
        raise RealConfigError(f'{path} must be a mapping')
    if not all(isinstance(key, str) for key in value):
        raise RealConfigError(f'{path} keys must be strings')
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if value is None:
        raise RealConfigError(f'{path} is required and cannot be null')
    if not isinstance(value, list):
        raise RealConfigError(f'{path} must be a list')
    return value


def _as_string(value: Any, path: str) -> str:
    if value is None:
        raise RealConfigError(f'{path} is required and cannot be null')
    if not isinstance(value, str) or not value.strip():
        raise RealConfigError(f'{path} must be a non-empty string')
    result = value.strip()
    if _is_placeholder(result):
        raise RealConfigError(f'{path} still contains a placeholder value')
    return result


def _as_optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, path)


def _as_string_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = _as_list(value, path)
    if not items:
        raise RealConfigError(f'{path} must not be empty')
    return tuple(
        _as_string(item, f'{path}[{index}]')
        for index, item in enumerate(items)
    )


def _as_int(value: Any, path: str) -> int:
    if value is None:
        raise RealConfigError(f'{path} is required and cannot be null')
    if isinstance(value, bool) or not isinstance(value, int):
        raise RealConfigError(f'{path} must be an integer')
    return int(value)


def _as_optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    result = _as_int(value, path)
    if result <= 0:
        raise RealConfigError(f'{path} must be positive')
    return result


def _as_float(value: Any, path: str) -> float:
    if value is None:
        raise RealConfigError(f'{path} is required and cannot be null')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealConfigError(f'{path} must be a number')
    result = float(value)
    if not np.isfinite(result):
        raise RealConfigError(f'{path} must be finite')
    return result


def _as_float_vector(value: Any, size: int, path: str) -> np.ndarray:
    items = _as_list(value, path)
    if len(items) != size:
        raise RealConfigError(
            f'{path} must contain {size} values, got {len(items)}'
        )
    vector = np.asarray(
        [
            _as_float(item, f'{path}[{index}]')
            for index, item in enumerate(items)
        ],
        dtype=np.float32,
    )
    vector.setflags(write=False)
    return vector


def _as_optional_float_vector(
    value: Any,
    size: int,
    path: str,
) -> np.ndarray | None:
    if value is None:
        return None
    return _as_float_vector(value, size, path)


def _as_choice(value: Any, choices: frozenset[str], path: str) -> str:
    result = _as_string(value, path).lower()
    if result not in choices:
        raise RealConfigError(
            f'{path} must be one of {sorted(choices)}, got {result!r}'
        )
    return result


def _as_optional_path(
    value: Any,
    package_root: Path,
    path: str,
) -> Path | None:
    text = _as_optional_string(value, path)
    if text is None:
        return None
    model_path = Path(text).expanduser()
    if not model_path.is_absolute():
        model_path = package_root / model_path
    return model_path.resolve()


def _require_positive(value: float, path: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise RealConfigError(f'{path} must be finite and positive')


def _require_nonnegative(value: float, path: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise RealConfigError(f'{path} must be finite and non-negative')


def _require_real_device(value: str | None, path: str) -> None:
    if value is None:
        raise RealConfigError(f'{path} is required for a real hardware backend')
    if _is_placeholder(value):
        raise RealConfigError(f'{path} still contains a placeholder value')
    if not Path(value).is_absolute():
        raise RealConfigError(
            f'{path} must be an absolute device path such as /dev/ttyUSB0'
        )


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized in _PLACEHOLDER_VALUES
        or normalized.startswith('<')
        or normalized.endswith('>')
    )
