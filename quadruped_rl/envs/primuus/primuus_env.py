"""Primuus 环境入口；观测、奖励和控制循环使用通用 base。"""

from isaacgym import gymtorch
import torch

from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs.base.legged_robot import LeggedRobot
from quadruped_rl.envs.primuus.commands import PrimuusCommandSampler
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

    def _post_physics_step_callback(self):
        # Commands change after this step's reward/metrics, immediately before the next observation.
        # The generic callback would change them before evaluating the step that used the old command.
        pass

    def _resample_commands(self, env_ids):
        commands, modes = self.command_sampler.sample(env_ids)
        self.commands[env_ids] = commands
        self.command_modes[env_ids] = modes

    def update_command_curriculum(self, env_ids):
        # The Primuus sampler evaluates completed high-speed segments, not whole-episode reward.
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit

    def _collect_command_metrics(self):
        self.command_sampler.observe(self.commands, self.base_lin_vel, self.reset_buf, self.time_out_buf)
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        return super()._collect_command_metrics()

    def _command_metric_masks(self):
        masks = super()._command_metric_masks()
        for name, (low, high) in zip(("forward_low", "forward_normal", "forward_fast"), self.cfg.commands.forward_speed_bins):
            # The boundary belongs to the next bin; the final bin includes its upper endpoint.
            upper = self.commands[:, 0] <= high if high == self.cfg.commands.max_curriculum else self.commands[:, 0] < high
            masks[name] = (self.command_modes == 1) & (self.commands[:, 0] >= low) & upper
        return masks

    def compute_observations(self):
        interval = max(1, int(self.cfg.commands.resampling_time / self.dt))
        env_ids = ((self.episode_length_buf > 0) & (self.episode_length_buf % interval == 0)).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        super().compute_observations()

    def get_command_curriculum_metrics(self):
        return self.command_sampler.metrics()

    def get_training_state(self):
        return {"primuus_commands": self.command_sampler.state_dict()}

    def load_training_state(self, state):
        self.command_sampler.load_state_dict(None if state is None else state.get("primuus_commands"))
        self.command_ranges["lin_vel_x"][1] = self.command_sampler.forward_limit
        # The simulator is newly created on resume; no partial old command segment is continued.
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        super().compute_observations()

    def _apply_torques(self):
        self.passive_joint_torques[:] = self.joint_friction.compute(self.dof_pos, self.dof_vel)
        # self.torques remains the capped motor effort used by torque rewards.
        self.net_joint_torques[:] = self.torques + self.passive_joint_torques
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.net_joint_torques))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.joint_friction.reset(self.dof_pos, env_ids)
        self.passive_joint_torques[env_ids] = 0.
        self.net_joint_torques[env_ids] = 0.
