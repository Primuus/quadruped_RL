"""Primuus 环境入口；观测、奖励和控制循环使用通用 base。"""

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs.base.legged_robot import LeggedRobot
from quadruped_rl.envs.primuus.commands import PrimuusCommandSampler
from quadruped_rl.envs.primuus.reference import advance_reference
from quadruped_rl.envs.primuus.motion import MotionTracker
from quadruped_rl.envs.primuus.gait import GaitQuality, LEGS, nominal_geometry
from quadruped_rl.envs.primuus.standing import StandingRecovery
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
        # Primuus 的课程也需要足端测量，不能随某个奖励权重关闭而停采。
        self.use_rigid_body_state = True
        heights = torch.tensor(cfg.rewards.motion_height_targets)
        if heights.shape != (7,) or not torch.isfinite(heights).all() or (heights <= 0).any():
            raise ValueError('motion_height_targets must contain seven finite positive heights')
        if cfg.control.control_type != 'P':
            raise ValueError('Primuus gait target constraints require position control')

    def _init_buffers(self):
        super()._init_buffers()
        self.command_sampler = PrimuusCommandSampler(
            self.cfg.commands, self.num_envs, self.device, self.dt,
            self.cfg.rewards.tracking_sigma, training=not self.cfg.env.test,
        )
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        self.joint_dynamics.validate_joint_names(self.dof_names)
        self.joint_friction = JenkinsJointFriction(self.joint_dynamics, self.dof_pos, torch.clamp)
        self.passive_joint_torques = torch.zeros_like(self.torques)
        self.net_joint_torques = torch.zeros_like(self.torques)
        self.requested_commands = torch.zeros_like(self.commands)
        self.next_command_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion = MotionTracker(self.num_envs, self.device, self.dt, self.cfg.commands)
        self.standing_weight = torch.zeros(self.num_envs, device=self.device)
        self.motion_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.symmetry_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self.straight_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        expected = [leg + '_' + part + '_joint' for leg in LEGS for part in ('hip', 'thigh', 'calf')]
        if self.dof_names != expected:
            raise ValueError('Primuus gait requires FL/FR/RL/RR hip/thigh/calf order')
        expected_feet = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], leg + '_foot_link') for leg in LEGS]
        if self.feet_indices.tolist() != expected_feet:
            raise ValueError('Primuus gait requires FL/FR/RL/RR foot order')
        feet, limits = nominal_geometry(
            self.cfg.asset.file.format(QUADRUPED_RL_ROOT_DIR=QUADRUPED_RL_ROOT_DIR),
            self.cfg.init_state.default_joint_angles, self.dof_names,
        )
        self.gait = GaitQuality(self.cfg.gait, feet, limits, self.num_envs, self.device, self.dt)
        self.motion_height_targets = torch.tensor(self.cfg.rewards.motion_height_targets, device=self.device)
        self.standing = StandingRecovery(self.cfg.rewards, self.gait.nominal, self.num_envs, self.device, self.dt)

    def _post_physics_step_callback(self):
        # Commands change after this step's reward/metrics, immediately before the next observation.
        # The generic callback would change them before evaluating the step that used the old command.
        pass

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        commands, modes = self.command_sampler.sample(env_ids)
        self.motion.begin(env_ids, self.requested_commands[env_ids, :3], commands[:, :3],
                          self.root_states[env_ids, :3], self.command_modes[env_ids])
        self.requested_commands[env_ids] = commands
        self.command_modes[env_ids] = modes
        cfg = self.cfg.commands
        low, high = int(round(cfg.min_resampling_time / self.dt)), int(round(cfg.resampling_time / self.dt))
        self.next_command_step[env_ids] = self.common_step_counter + torch.randint(low, high + 1, (len(env_ids),), device=self.device)
        stopping = env_ids[modes == 0]
        low, high = int(round(cfg.stand_min_resampling_time / self.dt)), int(round(cfg.stand_resampling_time / self.dt))
        self.next_command_step[stopping] = self.common_step_counter + torch.randint(low, high + 1, (len(stopping),), device=self.device)

    def update_command_curriculum(self, env_ids):
        # The Primuus sampler evaluates completed high-speed segments, not whole-episode reward.
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit

    def _collect_command_metrics(self):
        self.extras['motion_metrics'] = self.motion.observe(
            self.requested_commands[:, :3], self.commands[:, :3], self.base_lin_vel,
            self.base_ang_vel[:, 2], self.root_states[:, :3], self.projected_gravity[:, 2],
            self.dof_vel, self.stand_contacts, self.reset_buf, self.time_out_buf, self.motion_phase,
            self.standing.normal_pose,
        )
        self.extras['motion_metrics']['Symmetry/position_bias'] = torch.stack((self.straight_mask.sum(), torch.where(self.straight_mask[:, None], self.symmetry_bias.abs(), 0.0).sum() / 4)).float()
        for name, values in self.gait.measurements.items():
            self.extras['motion_metrics']['Gait/' + name] = torch.stack((values.new_tensor(self.num_envs), values.sum()))
        stop = torch.linalg.vector_norm(self.requested_commands[:, :3], dim=1) < 1e-7
        for name, values in (('pose_error', self.standing.joint_error), ('foot_error', self.standing.foot_error),
                             ('height_error', self.standing.height_error), ('normal_pose_fraction', self.standing.normal_pose.float()),
                             ('support_grace_fraction', self.standing.support_grace.any(dim=1).float())):
            self.extras['motion_metrics']['Standing/' + name] = torch.stack((stop.sum(), torch.where(stop, values, 0.0).sum())).float()
        raw_contacts = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.stand_contact_force_threshold
        gait_values = self.gait.curriculum_values(self.symmetry_bias.abs().mean(dim=1), raw_contacts)
        self.command_sampler.observe(self.commands, self.base_lin_vel, self.reset_buf, self.time_out_buf,
                                     self.base_ang_vel[:, 2], self.motion.quality_ready, gait_values)
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        return super()._collect_command_metrics()

    def _command_metric_masks(self):
        masks = super()._command_metric_masks()
        # Keep settled standing separate from braking and posture recovery.
        masks['stand'] &= self.motion_phase == 3
        masks['braking'] = self.motion_phase == 1
        masks['settling'] = self.motion_phase == 2
        for name, (low, high) in zip(("forward_low", "forward_normal", "forward_fast"), self.cfg.commands.forward_speed_bins):
            # The boundary belongs to the next bin; the final bin includes its upper endpoint.
            upper = self.commands[:, 0] <= high if high == self.cfg.commands.max_curriculum else self.commands[:, 0] < high
            masks[name] = (self.command_modes == 1) & (self.commands[:, 0] >= low) & upper
        return masks

    def compute_observations(self):
        env_ids = ((self.episode_length_buf > 0) & (self.common_step_counter >= self.next_command_step)).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self.commands[:, :3] = advance_reference(self.commands[:, :3], self.requested_commands[:, :3], self.dt, self.cfg.commands.transition)
        super().compute_observations()

    def get_command_curriculum_metrics(self):
        return self.command_sampler.metrics()

    def get_training_state(self):
        return {"primuus_commands": self.command_sampler.state_dict()}

    def load_training_state(self, state):
        self.command_sampler.load_state_dict(None if state is None else state.get("primuus_commands"))
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        # The simulator is newly created on resume; no partial old command segment is continued.
        self.commands.zero_()
        self.requested_commands.zero_()
        self.motion.reset(torch.arange(self.num_envs, device=self.device))
        self.gait.reset(torch.arange(self.num_envs, device=self.device))
        self.standing.reset(torch.arange(self.num_envs, device=self.device))
        self.symmetry_bias.zero_()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        super().compute_observations()

    def _apply_torques(self):
        self.passive_joint_torques[:] = self.joint_friction.compute(self.dof_pos, self.dof_vel)
        # self.torques remains the capped motor effort used by torque rewards.
        self.net_joint_torques[:] = self.torques + self.passive_joint_torques
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.net_joint_torques))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion.reset(env_ids)
        self.requested_commands[env_ids] = 0
        self.commands[env_ids] = 0
        self.command_sampler.sequence_target[env_ids] = False
        self.command_sampler.recovery_target[env_ids] = False
        self.symmetry_bias[env_ids] = 0
        self.gait.reset(env_ids)
        self.standing.reset(env_ids)
        super().reset_idx(env_ids)
        self.joint_friction.reset(self.dof_pos, env_ids)
        self.passive_joint_torques[env_ids] = 0.
        self.net_joint_torques[env_ids] = 0.

    def _update_stand_contact_state(self):
        super()._update_stand_contact_state()
        raw_stop = torch.linalg.vector_norm(self.requested_commands[:, :3], dim=1) < 1e-7
        reference_stop = torch.linalg.vector_norm(self.commands[:, :3], dim=1) < 1e-7
        cfg = self.cfg.rewards
        speed = torch.linalg.vector_norm(self.base_lin_vel[:, :2], dim=1)
        yaw = self.base_ang_vel[:, 2].abs()
        width = cfg.stand_speed_exit - cfg.stand_speed_enter
        self.standing_weight.copy_(raw_stop * reference_stop * ((cfg.stand_speed_exit - speed) / width).clamp(0, 1) * ((cfg.stand_speed_exit - yaw) / width).clamp(0, 1))
        self.straight_mask = (self.requested_commands[:, 1].abs() < 1e-7) & (self.requested_commands[:, 2].abs() < 1e-7) & (self.requested_commands[:, 0].abs() >= self.cfg.commands.min_translation_speed) & (torch.linalg.vector_norm(self.commands[:, :3] - self.requested_commands[:, :3], dim=1) < 0.03) & (self.command_sampler.segment_steps * self.dt > 1.0)
        feet = self.feet_pos - self.root_states[:, None, :3]
        quat = self.base_quat[:, None].expand(-1, 4, -1).reshape(-1, 4)
        angular_velocity = self.root_states[:, None, 10:13].expand_as(feet)
        feet_velocity = self.feet_vel - self.root_states[:, None, 7:10] - torch.cross(angular_velocity, feet, dim=2)
        feet_velocity = quat_rotate_inverse(quat, feet_velocity.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        feet = quat_rotate_inverse(quat, feet.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        bias = torch.stack((feet[:, 0, 0] - feet[:, 1, 0], feet[:, 2, 0] - feet[:, 3, 0], feet[:, 0, 1] + feet[:, 1, 1], feet[:, 2, 1] + feet[:, 3, 1]), dim=1)
        updated = torch.lerp(self.symmetry_bias, bias, self.dt / (cfg.symmetry_time_constant + self.dt))
        self.symmetry_bias.copy_(torch.where(self.straight_mask[:, None], updated, 0.0))
        targets = self.default_dof_pos + self.cfg.control.action_scale * self.actions
        raw_contacts = self.contact_forces[:, self.feet_indices, 2] > cfg.stand_contact_force_threshold
        self.gait.update(feet, self.feet_pos[:, :, 2], self.feet_vel, self.stand_contacts,
                         self.commands[:, :3], targets, self.projected_gravity[:, 2],
                         (self._get_base_heights() - self._motion_height_target()).abs(), feet_velocity, raw_contacts)
        self.standing.update(self.dof_pos, self.default_dof_pos, feet,
                             self._get_base_heights(), self.projected_gravity[:, 2])
        self.standing.update_support(self.stand_contacts, self.standing_weight)
        quiet = ((speed < 0.05) & (yaw < 0.05) & (self.dof_vel.square().mean(dim=1).sqrt() < 0.25) &
                 self.stand_contacts.all(dim=1) & self.standing.normal_pose)
        self.motion_phase.copy_(torch.where(raw_stop, torch.where(reference_stop & (speed < cfg.stand_speed_exit) &
                               (yaw < cfg.stand_speed_exit), torch.where(quiet, 3, 2), 1), 0))

    def _motion_height_target(self):
        # 使用实际执行的参考判断运动状态，高速减速/停车阶段不提前压低机身。
        x, y, yaw = self.commands[:, 0].abs(), self.commands[:, 1].abs(), self.commands[:, 2].abs()
        moving, turning = torch.maximum(x, y) > 0.1, yaw > 0.1
        mode = torch.where(moving, torch.where(y > 0.1, torch.where(x > 0.1, 4, 3),
                           torch.where(self.commands[:, 0] < 0, 2, 1)), 0)
        mode = torch.where(turning, torch.where(moving, 6, 5), mode)
        return self.motion_height_targets[mode]

    def _reward_base_height(self):
        excess = ((self._get_base_heights() - self._motion_height_target()).abs() -
                  self.cfg.rewards.base_height_tolerance).clamp(min=0)
        return (excess / self.cfg.rewards.base_height_sigma).square()

    def _reward_foot_placement(self):
        return self.gait.crossing_penalty + self.gait.region_penalty * (1 - self.standing_weight)

    def _reward_foot_clearance(self):
        return self.gait.clearance_penalty

    def _reward_foot_swing_direction(self):
        return self.gait.direction_penalty

    def _reward_joint_target_limits(self):
        return self.gait.target_penalty

    def _reward_tracking_lin_vel(self):
        cfg = self.cfg.rewards
        scale = cfg.tracking_absolute_tolerance + cfg.tracking_relative_tolerance * torch.linalg.vector_norm(self.commands[:, :2], dim=1)
        return torch.exp(-(self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1) / scale.square())

    def _reward_tracking_ang_vel(self):
        cfg = self.cfg.rewards
        scale = cfg.yaw_absolute_tolerance + cfg.yaw_relative_tolerance * self.commands[:, 2].abs()
        return torch.exp(-(self.commands[:, 2] - self.base_ang_vel[:, 2]).square() / scale.square())

    def _reward_stand_still(self):
        return self.standing.pose_penalty * self.standing_weight

    def _reward_stand_foot_placement(self):
        return self.standing.foot_penalty * self.standing_weight

    def _reward_stand_height(self):
        return self.standing.height_penalty * self.standing_weight

    def _reward_stand_contacts(self):
        return self.standing.support_penalty * self.standing_weight

    def _reward_stand_joint_motion(self):
        return self.dof_vel.square().mean(dim=1) * self.standing_weight * self.standing.quiet_weight

    def _reward_braking(self):
        raw_stop = torch.linalg.vector_norm(self.requested_commands[:, :3], dim=1) < 1e-7
        error = (self.base_lin_vel[:, :2] - self.commands[:, :2]).square().sum(dim=1) + 0.25 * (self.base_ang_vel[:, 2] - self.commands[:, 2]).square()
        return (torch.sqrt(error + 0.01) - 0.1) * raw_stop

    def _reward_straight_symmetry(self):
        return (self.symmetry_bias / self.cfg.rewards.symmetry_position_scale).square().mean(dim=1) * self.straight_mask
