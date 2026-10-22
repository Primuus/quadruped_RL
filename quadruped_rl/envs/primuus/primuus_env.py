"""Primuus 环境入口；观测、奖励和控制循环使用通用 base。"""

from isaacgym import gymtorch
import torch

from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs.base.legged_robot import LeggedRobot
from quadruped_rl.envs.primuus.commands import PrimuusCommandSampler
from quadruped_rl.envs.primuus.reference import advance_reference
from quadruped_rl.envs.primuus.motion import MotionTracker
from quadruped_rl.utils.joint_dynamics import JointDynamicsParameters, JenkinsJointFriction


STABILITY_FIELDS = (
    "foot_slip_speed", "tilt", "height_error",
    "contact_fl", "contact_fr", "contact_rl", "contact_rr",
)


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
        self.motion_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _post_physics_step_callback(self):
        # Commands change after this step's reward/metrics, immediately before the next observation.
        # The generic callback would change them before evaluating the step that used the old command.
        pass

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        commands, modes = self.command_sampler.sample(env_ids)
        self.motion.begin(env_ids, self.requested_commands[env_ids, :3], commands[:, :3],
                          self.root_states[env_ids, :3])
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
            self.stand_contacts, self.reset_buf, self.time_out_buf, self.motion_phase,
        )
        raw_contacts = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.stand_contact_force_threshold
        slip = torch.linalg.vector_norm(self.feet_vel[:, :, :2], dim=2)
        stability_values = torch.cat((torch.stack((
            (slip * raw_contacts).sum(dim=1) / raw_contacts.sum(dim=1).clamp(min=1),
            torch.acos((-self.projected_gravity[:, 2]).clamp(-1, 1)),
            (self._get_base_heights() - self.cfg.rewards.base_height_target).abs(),
        ), dim=1), raw_contacts.float()), dim=1)
        count = stability_values.new_tensor(self.num_envs)
        totals = stability_values.sum(dim=0)
        for index, name in enumerate(STABILITY_FIELDS):
            self.extras['motion_metrics']['Stability/' + name] = torch.stack((count, totals[index]))
        self.command_sampler.observe(self.commands, self.base_lin_vel, self.reset_buf, self.time_out_buf)
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        return super()._collect_command_metrics()

    def _command_metric_masks(self):
        masks = super()._command_metric_masks()
        # stand按零目标指令统计；制动和收稳是其中的辅助分类。
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
        quiet = (speed < 0.05) & (yaw < 0.05) & self.stand_contacts.all(dim=1)
        self.motion_phase.copy_(torch.where(raw_stop, torch.where(reference_stop & (speed < cfg.stand_speed_exit) &
                               (yaw < cfg.stand_speed_exit), torch.where(quiet, 3, 2), 1), 0))
