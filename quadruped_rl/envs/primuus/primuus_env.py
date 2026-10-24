"""Primuus 环境入口；适配关节损耗和站立姿态，观测与控制循环使用通用 base。"""

import math
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs.base.legged_robot import LeggedRobot
from quadruped_rl.utils.joint_dynamics import JointDynamicsParameters, JenkinsJointFriction


class PrimuusEnv(LeggedRobot):
    """独立机器人适配器，不继承或导入 Go2 环境。"""

    def __init__(self, cfg, *args, **kwargs):
        path = cfg.asset.joint_dynamics_file.format(QUADRUPED_RL_ROOT_DIR=QUADRUPED_RL_ROOT_DIR)
        self.joint_dynamics = JointDynamicsParameters.from_json(path)
        super().__init__(cfg, *args, **kwargs)

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        # Replace engine-specific losses, not motor PD, inertia, or effort limits.
        props['friction'].fill(0.)
        props['damping'].fill(0.)
        return props

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        if self.reward_scales.get('stand_foot_lateral', 0.0) == 0.0:
            return
        sigma = cfg.rewards.stand_foot_lateral_sigma
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError('stand_foot_lateral_sigma must be finite and positive')
        if self.reward_scales['stand_foot_lateral'] < 0.0:
            raise ValueError('stand_foot_lateral is an exponential reward and requires a positive scale')
        targets = cfg.rewards.stand_foot_lateral_targets
        expected = {leg + '_foot_link' for leg in ('FL', 'FR', 'RL', 'RR')}
        if set(targets) != expected:
            raise ValueError('stand_foot_lateral_targets must name all four Primuus feet exactly')
        for name, target in targets.items():
            side = 1.0 if name.startswith(('FL_', 'RL_')) else -1.0
            if not math.isfinite(target) or side * target <= 0.0:
                raise ValueError(f'Invalid stand_foot_lateral_targets for {name}: {target}')

    def _init_buffers(self):
        super()._init_buffers()
        self.stand_still_joint_weights = torch.tensor(
            [self.cfg.rewards.stand_still_joint_weights.get(name, 1.0) for name in self.dof_names],
            dtype=torch.float, device=self.device,
        )
        if self.reward_scales.get('stand_foot_lateral', 0.0) != 0.0:
            targets_by_body = {}
            for name, target in self.cfg.rewards.stand_foot_lateral_targets.items():
                body = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                if body < 0 or body in targets_by_body:
                    raise ValueError(f'Invalid or duplicate standing foot body: {name}')
                targets_by_body[body] = target
            feet = self.feet_indices.tolist()
            if len(feet) != 4 or set(feet) != set(targets_by_body):
                raise ValueError('Standing foot targets do not match the four selected feet')
            self.stand_foot_lateral_targets = torch.tensor(
                [targets_by_body[body] for body in feet], dtype=torch.float, device=self.device,
            )
        self.joint_dynamics.validate_joint_names(self.dof_names)
        self.joint_friction = JenkinsJointFriction(self.joint_dynamics, self.dof_pos, torch.clamp)
        self.passive_joint_torques = torch.zeros_like(self.torques)
        self.net_joint_torques = torch.zeros_like(self.torques)

    def _apply_torques(self):
        self.passive_joint_torques[:] = self.joint_friction.compute(self.dof_pos, self.dof_vel)
        # self.torques remains the capped motor effort used by torque rewards.
        self.net_joint_torques[:] = self.torques + self.passive_joint_torques
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.net_joint_torques))

    def _reward_stand_still(self):
        joint_error = torch.abs(self.dof_pos - self.default_dof_pos)
        return torch.sum(joint_error * self.stand_still_joint_weights, dim=1) * self._standing_command_mask()

    def _reward_stand_orientation(self):
        # legged_gym LeggedRobot._reward_orientation；仅给站立命令增加水平约束。
        return self._reward_orientation() * self._standing_command_mask()

    def _reward_stand_foot_lateral(self):
        # Adapted from robot_lab feet_distance_y_exp (26acb225); source and config in
        # doc/primuus_motion_training.md. Add a standing mask and name-based targets.
        # 使用完整机身旋转变换；世界位置、朝向和足端索引顺序不改变约束语义。
        relative = self.feet_pos - self.root_states[:, None, :3]
        quats = self.base_quat[:, None, :].expand(-1, relative.shape[1], -1)
        feet_body = quat_rotate_inverse(quats.reshape(-1, 4), relative.reshape(-1, 3))
        lateral = feet_body.reshape(self.num_envs, -1, 3)[:, :, 1]
        error = lateral - self.stand_foot_lateral_targets
        reward = torch.exp(-torch.square(error).sum(dim=1) / self.cfg.rewards.stand_foot_lateral_sigma ** 2)
        upright = torch.clamp(-self.projected_gravity[:, 2], min=0.0, max=0.7) / 0.7
        return reward * upright * self._standing_command_mask()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.joint_friction.reset(self.dof_pos, env_ids)
        self.passive_joint_torques[env_ids] = 0.
        self.net_joint_torques[env_ids] = 0.
