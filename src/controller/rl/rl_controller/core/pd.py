from __future__ import annotations

import numpy as np

from .config import RobotRLConfig


def compute_target_positions(
    action: np.ndarray,
    cfg: RobotRLConfig,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(cfg.num_actions)
    return (cfg.default_angles + action * cfg.action_scale).astype(np.float32)


def compute_pd_torques(
    target_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cfg: RobotRLConfig,
    torque_limits: np.ndarray | None = None,
) -> np.ndarray:
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(cfg.num_actions)
    joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(cfg.num_actions)
    joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(cfg.num_actions)
    torques = (target_pos - joint_pos) * cfg.kps - joint_vel * cfg.kds

    if torque_limits is not None:
        torque_limits = np.asarray(torque_limits, dtype=np.float32).reshape(cfg.num_actions)
        torques = np.clip(torques, -torque_limits, torque_limits)

    return torques.astype(np.float32)


def compute_torques(
    action: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cfg: RobotRLConfig,
    torque_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_pos = compute_target_positions(action, cfg)
    torques = compute_pd_torques(
        target_pos,
        joint_pos,
        joint_vel,
        cfg,
        torque_limits,
    )
    return torques, target_pos
