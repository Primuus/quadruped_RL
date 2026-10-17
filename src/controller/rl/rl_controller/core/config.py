from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# 新增配置字段时请同步补充中文说明，确保训练/部署配置一致可读。

DEFAULT_ROBOT_NAME = 'go2'  # 当前默认机器人实例；新增机器人时通过 YAML 覆盖
DEFAULT_JOINT_NAMES = (
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
)  # 默认关节顺序，当前对应 Go2
DEFAULT_ACTUATOR_NAMES = tuple(name.replace('_joint', '') for name in DEFAULT_JOINT_NAMES)  # 默认执行器顺序
DEFAULT_DEFAULT_ANGLES = np.array(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)  # action=0 时的默认关节角，当前对应 Go2
DEFAULT_POLICY_PATH = 'models/policy_1.pt'  # 默认策略文件名
DEFAULT_KPS = np.full(12, 20.0, dtype=np.float32)  # 默认比例增益，当前对应 Go2
DEFAULT_KDS = np.full(12, 0.5, dtype=np.float32)  # 默认微分增益，当前对应 Go2
DEFAULT_COMMAND_INIT = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # 默认初始速度命令 [vx, vy, yaw_rate]
DEFAULT_COMMAND_LIMITS = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # 命令上限 [vx, vy, yaw_rate]
DEFAULT_COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)  # 命令观测缩放
DEFAULT_KEYBOARD_BINDINGS = {
    'forward': 'KEY_W',
    'backward': 'KEY_S',
    'left': 'KEY_A',
    'right': 'KEY_D',
    'turn_left': 'KEY_Q',
    'turn_right': 'KEY_E',
    'stop': 'KEY_SPACE',
    'enable': 'KEY_F',
    'disable': 'KEY_I',
    'speed_low': 'KEY_1',
    'speed_medium': 'KEY_2',
    'speed_high': 'KEY_3',
    'reset_sim': 'KEY_R',
    'print_status': 'KEY_P',
}  # Linux evdev 键名；仿真键盘映射的唯一配置来源
DEFAULT_KEYBOARD_SPEED_LEVELS = np.array([0.25, 0.5, 1.0], dtype=np.float32)  # 低、中、高三档速度比例
DEFAULT_KEYBOARD_DEFAULT_SPEED_LEVEL = 2  # 默认使用中速档
DEFAULT_INITIAL_BASE_POS = np.array([0.0, 0.0, 0.42], dtype=np.float32)  # 初始机身位置 [m]
DEFAULT_INITIAL_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 初始机身姿态四元数
DEFAULT_FALL_HEIGHT_THRESHOLD = 0.18  # 跌倒判定高度阈值
DEFAULT_FALL_GRAVITY_Z_THRESHOLD = -0.2  # 跌倒判定时的重力方向阈值
DEFAULT_NUM_ONE_STEP_OBS = 45  # 单帧观测维度
DEFAULT_HISTORY_LENGTH = 6  # 历史观测帧数
DEFAULT_NUM_OBS = DEFAULT_NUM_ONE_STEP_OBS * DEFAULT_HISTORY_LENGTH  # policy 输入维度：45*6=270
DEFAULT_NUM_ONE_STEP_PRIVILEGED_OBS = DEFAULT_NUM_ONE_STEP_OBS + 3 + 3 + 187  # 训练 critic 特权观测维度：45+base_lin_vel3+外力3+高度187
DEFAULT_NUM_PRIVILEGED_OBS = DEFAULT_NUM_ONE_STEP_PRIVILEGED_OBS  # 部署不使用，仅用于和训练侧配置对齐
DEFAULT_OBS_MODE = 'quadruped_history_45x6'  # 默认使用 6 帧 45 维历史观测
DEFAULT_CLIP_OBSERVATIONS = 100.0  # 与训练侧 normalization.clip_observations 保持一致
DEFAULT_CLIP_ACTIONS = 100.0  # 与训练侧 normalization.clip_actions 保持一致


def _default_array_for_actions(default: np.ndarray, num_actions: int, fill_value: float = 0.0) -> np.ndarray:
    if default.shape == (num_actions,):
        return default
    return np.full(num_actions, fill_value, dtype=np.float32)


def _as_float_array(value: Any, expected_size: int, default: np.ndarray) -> np.ndarray:
    array = np.asarray(default if value is None else value, dtype=np.float32)
    if array.shape != (expected_size,):
        raise ValueError(f'Expected shape ({expected_size},), got {array.shape}')
    return array


def _require_finite_array(value: np.ndarray, field_name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{field_name} must contain only finite values')


def _require_positive_finite(value: float, field_name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{field_name} must be finite and positive')


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def discover_package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return discover_package_root() / 'config' / 'go2.yaml'


@dataclass(slots=True)
class RobotRLConfig:
    robot_name: str  # 机器人名称，用于区分 Go2 或后续其他四足机器人
    package_root: Path  # 包根目录
    policy_path: Path  # 策略权重文件路径
    xml_path: Path  # MuJoCo 场景文件路径
    simulation_duration: float  # 仿真总时长 [s]
    simulation_dt: float  # 仿真步长 [s]
    control_decimation: int  # 一个动作持续的仿真步数
    num_actions: int  # 动作维度
    num_one_step_obs: int  # 单帧观测维度
    history_length: int  # 历史观测帧数
    num_obs: int  # 观测维度
    num_one_step_privileged_obs: int  # 训练 critic 单帧特权观测维度，部署 actor 不使用
    num_privileged_obs: int  # 训练 critic 输入维度，部署 actor 不使用
    obs_mode: str  # 观测模式
    joint_names: tuple[str, ...]  # 关节顺序
    actuator_names: tuple[str, ...]  # 执行器顺序
    kps: np.ndarray  # 每个关节的比例增益
    kds: np.ndarray  # 每个关节的微分增益
    default_angles: np.ndarray  # 默认关节角
    action_scale: float  # 动作缩放系数
    clip_observations: float  # 观测裁剪上限，需和训练侧 normalization.clip_observations 一致
    clip_actions: float  # 动作裁剪上限，需和训练侧 normalization.clip_actions 一致
    command_init: np.ndarray  # 初始速度命令
    command_limits: np.ndarray  # 速度命令上限
    command_scale: np.ndarray  # 命令观测缩放
    keyboard_bindings: dict[str, str]  # Linux evdev 键名到控制语义的映射
    keyboard_speed_levels: np.ndarray  # 键盘低、中、高三档速度比例
    keyboard_default_speed_level: int  # 默认速度档位，取值 1/2/3
    fall_height_threshold: float  # 跌倒高度阈值
    fall_gravity_z_threshold: float  # 跌倒重力方向阈值
    obs_scale_lin_vel: float  # 线速度观测缩放
    obs_scale_ang_vel: float  # 角速度观测缩放
    obs_scale_dof_pos: float  # 关节位置观测缩放
    obs_scale_dof_vel: float  # 关节速度观测缩放
    initial_base_pos: np.ndarray  # 初始机身位置
    initial_base_quat: np.ndarray  # 初始机身姿态

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        *,
        policy_path: str | Path | None = None,
        xml_path: str | Path | None = None,
        simulation_duration: float | None = None,
        command_init: np.ndarray | list[float] | None = None,
    ) -> 'RobotRLConfig':
        """Apply CLI overrides before validating; relative CLI paths use the working directory."""
        config_path = Path(yaml_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with config_path.open('r', encoding='utf-8') as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError('Robot RL config must be a mapping.')

        package_root = config_path.parent.parent

        robot_name = str(raw.get('robot_name', DEFAULT_ROBOT_NAME))
        policy_path = (
            Path(policy_path).expanduser().resolve()
            if policy_path is not None
            else _resolve_path(package_root, raw.get('policy_path', DEFAULT_POLICY_PATH))
        )
        xml_path = (
            Path(xml_path).expanduser().resolve()
            if xml_path is not None
            else _resolve_path(package_root, raw.get('xml_path', 'resources/robots/go2/scene_flat.xml'))
        )

        num_actions = int(raw.get('num_actions', len(DEFAULT_JOINT_NAMES)))
        num_one_step_obs = int(raw.get('num_one_step_obs', DEFAULT_NUM_ONE_STEP_OBS))
        history_length = int(raw.get('history_length', DEFAULT_HISTORY_LENGTH))
        num_obs = int(raw.get('num_obs', num_one_step_obs * history_length))
        num_one_step_privileged_obs = int(
            raw.get('num_one_step_privileged_obs', num_one_step_obs + 3 + 3 + 187)
        )
        num_privileged_obs = int(raw.get('num_privileged_obs', num_one_step_privileged_obs))
        default_angles = _default_array_for_actions(DEFAULT_DEFAULT_ANGLES, num_actions, 0.0)
        default_kps = _default_array_for_actions(DEFAULT_KPS, num_actions, 20.0)
        default_kds = _default_array_for_actions(DEFAULT_KDS, num_actions, 0.5)
        keyboard_raw = raw.get('keyboard') or {}
        if not isinstance(keyboard_raw, dict):
            raise ValueError('keyboard config must be a mapping')
        keyboard_bindings_raw = keyboard_raw.get('bindings') or {}
        if not isinstance(keyboard_bindings_raw, dict):
            raise ValueError('keyboard.bindings must be a mapping')
        keyboard_bindings = DEFAULT_KEYBOARD_BINDINGS.copy()
        keyboard_bindings.update(
            {str(action): str(key) for action, key in keyboard_bindings_raw.items()}
        )
        command_limits = _as_float_array(raw.get('command_limits'), 3, DEFAULT_COMMAND_LIMITS)
        command = _as_float_array(
            raw.get('command_init') if command_init is None else command_init,
            3,
            DEFAULT_COMMAND_INIT,
        )
        if command_init is not None:
            command = np.clip(command, -command_limits, command_limits)

        cfg = cls(
            robot_name=robot_name,
            package_root=package_root,
            policy_path=policy_path,
            xml_path=xml_path,
            simulation_duration=float(
                raw.get('simulation_duration', 60.0)
                if simulation_duration is None else simulation_duration
            ),
            simulation_dt=float(raw.get('simulation_dt', 0.002)),
            control_decimation=int(raw.get('control_decimation', 10)),
            num_actions=num_actions,
            num_one_step_obs=num_one_step_obs,
            history_length=history_length,
            num_obs=num_obs,
            num_one_step_privileged_obs=num_one_step_privileged_obs,
            num_privileged_obs=num_privileged_obs,
            obs_mode=str(raw.get('obs_mode', DEFAULT_OBS_MODE)),
            joint_names=tuple(str(item) for item in (raw.get('joint_names') or DEFAULT_JOINT_NAMES)),
            actuator_names=tuple(str(item) for item in (raw.get('actuator_names') or DEFAULT_ACTUATOR_NAMES)),
            kps=_as_float_array(raw.get('kps'), num_actions, default_kps),
            kds=_as_float_array(raw.get('kds'), num_actions, default_kds),
            default_angles=_as_float_array(raw.get('default_angles'), num_actions, default_angles),
            action_scale=float(raw.get('action_scale', 0.25)),
            clip_observations=float(raw.get('clip_observations', DEFAULT_CLIP_OBSERVATIONS)),
            clip_actions=float(raw.get('clip_actions', DEFAULT_CLIP_ACTIONS)),
            command_init=command,
            command_limits=command_limits,
            command_scale=_as_float_array(raw.get('command_scale'), 3, DEFAULT_COMMAND_SCALE),
            keyboard_bindings=keyboard_bindings,
            keyboard_speed_levels=_as_float_array(
                keyboard_raw.get('speed_levels'),
                3,
                DEFAULT_KEYBOARD_SPEED_LEVELS,
            ),
            keyboard_default_speed_level=int(
                keyboard_raw.get('default_speed_level', DEFAULT_KEYBOARD_DEFAULT_SPEED_LEVEL)
            ),
            fall_height_threshold=float(raw.get('fall_height_threshold', DEFAULT_FALL_HEIGHT_THRESHOLD)),
            fall_gravity_z_threshold=float(
                raw.get('fall_gravity_z_threshold', DEFAULT_FALL_GRAVITY_Z_THRESHOLD)
            ),
            obs_scale_lin_vel=float(raw.get('obs_scale_lin_vel', 2.0)),
            obs_scale_ang_vel=float(raw.get('obs_scale_ang_vel', 0.25)),
            obs_scale_dof_pos=float(raw.get('obs_scale_dof_pos', 1.0)),
            obs_scale_dof_vel=float(raw.get('obs_scale_dof_vel', 0.05)),
            initial_base_pos=_as_float_array(raw.get('initial_base_pos'), 3, DEFAULT_INITIAL_BASE_POS),
            initial_base_quat=_as_float_array(raw.get('initial_base_quat'), 4, DEFAULT_INITIAL_BASE_QUAT),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.robot_name.strip():
            raise ValueError('robot_name must not be empty')
        if not self.policy_path.is_file():
            raise ValueError(f'policy_path does not exist: {self.policy_path}')
        if not self.xml_path.is_file():
            raise ValueError(f'xml_path does not exist: {self.xml_path}')
        _require_positive_finite(self.simulation_duration, 'simulation_duration')
        _require_positive_finite(self.simulation_dt, 'simulation_dt')
        if self.control_decimation <= 0:
            raise ValueError('control_decimation must be positive')
        if self.num_actions <= 0:
            raise ValueError('num_actions must be positive')
        if self.history_length <= 0:
            raise ValueError('history_length must be positive')
        expected_one_step_obs = 9 + 3 * self.num_actions
        if self.num_one_step_obs != expected_one_step_obs:
            raise ValueError(
                f'Quadruped HIM one-step observation expects {expected_one_step_obs} observations '
                f'for {self.num_actions} actions, got {self.num_one_step_obs}'
            )
        expected_obs_mode = f'quadruped_history_{self.num_one_step_obs}x{self.history_length}'
        legacy_obs_mode = f'go2_history_{self.num_one_step_obs}x{self.history_length}'
        if self.obs_mode not in {expected_obs_mode, legacy_obs_mode}:
            raise ValueError(
                f'Quadruped history policy expects obs_mode={expected_obs_mode}, got {self.obs_mode}'
            )
        expected_obs = self.num_one_step_obs * self.history_length
        if self.num_obs != expected_obs:
            raise ValueError(f'Quadruped history policy expects {expected_obs} observations, got {self.num_obs}')
        expected_privileged_obs = self.num_one_step_obs + 3 + 3 + 187
        if self.num_one_step_privileged_obs != expected_privileged_obs:
            raise ValueError(
                f'Quadruped training critic expects {expected_privileged_obs} privileged observations, '
                f'got {self.num_one_step_privileged_obs}'
            )
        if self.num_privileged_obs != self.num_one_step_privileged_obs:
            raise ValueError(
                f'Quadruped training critic expects num_privileged_obs={self.num_one_step_privileged_obs}, '
                f'got {self.num_privileged_obs}'
            )
        if len(self.joint_names) != self.num_actions:
            raise ValueError('joint_names length must match num_actions')
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('joint_names must contain unique names')
        if len(self.actuator_names) != self.num_actions:
            raise ValueError('actuator_names length must match num_actions')
        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError('actuator_names must contain unique names')

        for field_name, values in (
            ('kps', self.kps),
            ('kds', self.kds),
            ('default_angles', self.default_angles),
            ('command_init', self.command_init),
            ('command_limits', self.command_limits),
            ('command_scale', self.command_scale),
            ('initial_base_pos', self.initial_base_pos),
            ('initial_base_quat', self.initial_base_quat),
        ):
            _require_finite_array(values, field_name)
        if np.any(self.kps < 0.0) or np.any(self.kds < 0.0):
            raise ValueError('kps and kds must be non-negative')
        _require_positive_finite(self.action_scale, 'action_scale')
        _require_positive_finite(self.clip_observations, 'clip_observations')
        _require_positive_finite(self.clip_actions, 'clip_actions')
        if np.any(self.command_limits <= 0.0):
            raise ValueError('command_limits must be positive')
        if np.any(self.command_scale <= 0.0):
            raise ValueError('command_scale must be positive')
        if np.any(np.abs(self.command_init) > self.command_limits):
            raise ValueError('command_init must be within command_limits')
        for field_name, value in (
            ('obs_scale_lin_vel', self.obs_scale_lin_vel),
            ('obs_scale_ang_vel', self.obs_scale_ang_vel),
            ('obs_scale_dof_pos', self.obs_scale_dof_pos),
            ('obs_scale_dof_vel', self.obs_scale_dof_vel),
        ):
            _require_positive_finite(value, field_name)
        if not np.isfinite(self.fall_height_threshold):
            raise ValueError('fall_height_threshold must be finite')
        if (
            not np.isfinite(self.fall_gravity_z_threshold)
            or not -1.0 <= self.fall_gravity_z_threshold <= 1.0
        ):
            raise ValueError('fall_gravity_z_threshold must be in [-1, 1]')
        quaternion_norm = float(np.linalg.norm(self.initial_base_quat))
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-4):
            raise ValueError('initial_base_quat must be normalized')
        missing_keyboard_actions = set(DEFAULT_KEYBOARD_BINDINGS) - set(self.keyboard_bindings)
        if missing_keyboard_actions:
            raise ValueError(
                f'Missing keyboard bindings: {", ".join(sorted(missing_keyboard_actions))}'
            )
        configured_keys = [
            self.keyboard_bindings[action]
            for action in DEFAULT_KEYBOARD_BINDINGS
        ]
        if len(set(configured_keys)) != len(configured_keys):
            raise ValueError('Each keyboard action must use a unique key')
        if not all(key.startswith('KEY_') for key in configured_keys):
            raise ValueError('keyboard bindings must use Linux evdev KEY_* names')
        if (
            not np.all(np.isfinite(self.keyboard_speed_levels))
            or np.any(self.keyboard_speed_levels <= 0.0)
            or np.any(self.keyboard_speed_levels > 1.0)
            or np.any(np.diff(self.keyboard_speed_levels) <= 0.0)
        ):
            raise ValueError(
                'keyboard.speed_levels must be strictly increasing values in (0, 1]'
            )
        if self.keyboard_default_speed_level not in (1, 2, 3):
            raise ValueError('keyboard.default_speed_level must be 1, 2, or 3')
