"""Primuus actuator and inertial randomization; switches are fixed at creation.

Gain/COM/delay semantics follow IsaacLab events and DelayedPDActuator;
pre-limit motor scaling follows Walk These Ways. See the training guide.
"""

import math

import numpy as np
import torch


def _bounds(values, name, minimum=None):
    if (len(values) != 2 or not all(math.isfinite(v) for v in values)
            or values[0] > values[1]
            or (minimum is not None and values[0] < minimum)):
        raise ValueError(f'{name} requires finite ordered bounds >= {minimum}')
    return values


class PrimuusDomainRandomization:
    """Mixin for PrimuusEnv; no changes to policy observations or rewards."""

    def _parse_domain_randomization(self, cfg):
        dr = cfg.domain_rand
        self._randomized_control = any((dr.randomize_pd_gains,
                                       dr.randomize_motor_strength,
                                       dr.randomize_action_latency))
        if self._randomized_control and cfg.control.control_type != 'P':
            raise ValueError('Primuus actuator randomization requires position control')
        for enabled, names in (
            (dr.randomize_pd_gains, ('kp_scale_range', 'kd_scale_range')),
            (dr.randomize_motor_strength, ('motor_strength_range',)),
            (dr.randomize_joint_friction,
             ('joint_coulomb_scale_range', 'joint_viscous_scale_range')),
        ):
            if enabled:
                for name in names:
                    low, _ = _bounds(getattr(dr, name), name, minimum=0.0)
                    if low <= 0.0:
                        raise ValueError(f'{name} must be strictly positive')
        if dr.randomize_base_com:
            if set(dr.base_com_offset_range) != {'x', 'y', 'z'}:
                raise ValueError('base_com_offset_range must contain x, y, z')
            for axis, values in dr.base_com_offset_range.items():
                _bounds(values, f'base_com_offset_range[{axis}]')
        if dr.randomize_action_latency:
            bounds = _bounds(dr.action_latency_range_ms, 'action_latency_range_ms', 0.0)
            ratios = [v / (1000.0 * self.sim_params.dt) for v in bounds]
            # PhysX returns float32 dt; avoid dropping exact 2 ms multiples.
            ratios = [round(v) if math.isclose(v, round(v), rel_tol=1e-7)
                      else v for v in ratios]
            self.min_delay_steps = math.ceil(ratios[0])
            self.max_delay_steps = math.floor(ratios[1])
            if self.min_delay_steps > self.max_delay_steps:
                raise ValueError('action_latency_range_ms contains no physics timestep')

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if self.cfg.domain_rand.randomize_base_com:
            base_id = self.body_names.index('base_link')
            if env_id == 0:
                self.base_com_offsets = np.zeros((self.num_envs, 3))
                self.base_com_nominal = np.zeros((self.num_envs, 3))
            com = props[base_id].com
            self.base_com_nominal[env_id] = (com.x, com.y, com.z)
            offset = np.array([np.random.uniform(*self.cfg.domain_rand.base_com_offset_range[a])
                               for a in ('x', 'y', 'z')])
            self.base_com_offsets[env_id] = offset
            com.x += float(offset[0])
            com.y += float(offset[1])
            com.z += float(offset[2])
            props[base_id].com = com
        return props

    def _init_domain_rand_buffers(self):
        dr = self.cfg.domain_rand
        if self._randomized_control:
            self.kp_scale = torch.ones_like(self.actions)
            self.kd_scale = torch.ones_like(self.actions)
            self.motor_strength = torch.ones_like(self.actions)
        if dr.randomize_joint_friction:
            self.joint_coulomb_scale = torch.ones_like(self.actions)
            self.joint_viscous_scale = torch.ones_like(self.actions)
        if dr.randomize_action_latency:
            self.action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.action_delay_history = torch.zeros(
                self.max_delay_steps + 1, self.num_envs, self.num_actions, device=self.device,
            )
            self.action_delay_write_index = 0
            self.action_delay_env_ids = torch.arange(self.num_envs, device=self.device)
            self.applied_actions = torch.zeros_like(self.actions)
        self._resample_episode_domain_rand(torch.arange(self.num_envs, device=self.device))

    def _sample_scale(self, env_ids, bounds, per_joint=True):
        # Equal bounds deliberately consume no RNG, facilitating nominal comparisons.
        shape = (len(env_ids), self.num_actions if per_joint else 1)
        if bounds[0] == bounds[1]:
            return torch.full(shape, bounds[0], device=self.device)
        return torch.empty(shape, device=self.device).uniform_(*bounds)

    def _resample_episode_domain_rand(self, env_ids):
        if len(env_ids) == 0:
            return
        dr = self.cfg.domain_rand
        if dr.randomize_pd_gains:
            self.kp_scale[env_ids] = self._sample_scale(env_ids, dr.kp_scale_range, dr.pd_gains_per_joint)
            self.kd_scale[env_ids] = self._sample_scale(env_ids, dr.kd_scale_range, dr.pd_gains_per_joint)
        if dr.randomize_motor_strength:
            self.motor_strength[env_ids] = self._sample_scale(env_ids, dr.motor_strength_range)
        if dr.randomize_joint_friction:
            self.joint_coulomb_scale[env_ids] = self._sample_scale(env_ids, dr.joint_coulomb_scale_range)
            self.joint_viscous_scale[env_ids] = self._sample_scale(env_ids, dr.joint_viscous_scale_range)
            self.joint_friction.set_parameters(
                env_ids,
                self.joint_dynamics.coulomb_torque * self.joint_coulomb_scale[env_ids],
                self.joint_dynamics.viscous_damping * self.joint_viscous_scale[env_ids],
            )
            self.joint_friction.reset(self.dof_pos, env_ids)
        if dr.randomize_action_latency:
            if self.min_delay_steps == self.max_delay_steps:
                self.action_delay_steps[env_ids] = self.min_delay_steps
            else:
                self.action_delay_steps[env_ids] = torch.randint(
                    self.min_delay_steps, self.max_delay_steps + 1,
                    (len(env_ids),), device=self.device,
                )
            self.action_delay_history[:, env_ids] = 0.0
            self.applied_actions[env_ids] = 0.0

    def _get_delayed_actions(self, actions):
        if not self.cfg.domain_rand.randomize_action_latency:
            return actions
        index = self.action_delay_write_index
        self.action_delay_history[index] = actions
        read_index = (index - self.action_delay_steps) % len(self.action_delay_history)
        self.applied_actions[:] = self.action_delay_history[read_index, self.action_delay_env_ids]
        self.action_delay_write_index = (index + 1) % len(self.action_delay_history)
        return self.applied_actions

    def _compute_torques(self, actions):
        if not self._randomized_control:
            return super()._compute_torques(actions)
        applied = self._get_delayed_actions(actions)
        target = applied * self.cfg.control.action_scale + self.default_dof_pos
        # Scale motor output before clipping; passive Jenkins torque is added separately.
        torques = (self.p_gains * self.kp_scale * (target - self.dof_pos)
                   - self.d_gains * self.kd_scale * self.dof_vel) * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
