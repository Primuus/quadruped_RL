"""Primuus 左右镜像：髋轴已由模型镜像，关节只换序，不套用其他机器人的符号。"""

import torch


def mirror_actions(actions):
    if actions.shape[-1] != 12:
        raise ValueError("Primuus mirror expects 12 joint actions")
    return actions.reshape(*actions.shape[:-1], 4, 3)[..., [1, 0, 3, 2], :].reshape_as(actions)


def mirror_history(observations):
    if observations.shape[-1] % 45:
        raise ValueError("Primuus mirror expects complete 45-dimensional frames")
    frames = observations.reshape(*observations.shape[:-1], -1, 45)
    mirrored = frames.clone()
    # Polar vectors reflect y; axial angular velocity reflects x and z.
    mirrored[..., [1, 2, 3, 5, 7]] *= -1
    for start in (9, 21, 33):
        mirrored[..., start:start + 12] = mirror_actions(frames[..., start:start + 12])
    return mirrored.reshape_as(observations)


def mirror_privileged(observations):
    if observations.shape[-1] != 238:
        raise ValueError("Primuus critic mirror expects 238 observations")
    mirrored = observations.clone()
    mirrored[..., :45] = mirror_history(observations[..., :45])
    mirrored[..., [46, 49]] *= -1  # true linear velocity y and external force y
    mirrored[..., 51:] = observations[..., 51:].reshape(*observations.shape[:-1], 17, 11).flip(-1).reshape(*observations.shape[:-1], 187)
    return mirrored
