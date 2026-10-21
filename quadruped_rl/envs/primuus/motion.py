"""Primuus 运动事件统计；完整观察停车后的保护窗口，避免漏记迟发翻倒。"""

import torch


class MotionTracker:
    def __init__(self, num_envs, device, dt, cfg):
        self.dt, self.cfg, self.device = dt, cfg, device
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stopping = self.active.clone()
        self.source_modes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.body_done = self.active.clone()
        self.quiet_done = self.active.clone()
        self.elapsed = torch.zeros(num_envs, device=device)
        self.body_hold = self.elapsed.clone()
        self.quiet_hold = self.elapsed.clone()
        self.target_hold = self.elapsed.clone()
        self.candidate_distance = self.elapsed.clone()
        self.body_time = self.elapsed.clone()
        self.quiet_time = self.elapsed.clone()
        self.body_distance = self.elapsed.clone()
        self.origin = torch.zeros(num_envs, 2, device=device)
        self.velocity_mean = torch.zeros(num_envs, 3, device=device)
        self.pending = {}
        self.quality = torch.zeros(8, device=device)
        self.direction_quality = torch.zeros(26, 4, device=device)
        self.stopping_quality = torch.zeros(6, 2, device=device)
        self.source_ids = torch.arange(1, 7, device=device)
        self.quality_elapsed = 0.0
        self.quality_ready = False

    def _add(self, name, mask, values=None):
        count = mask.sum()
        value = count if values is None else torch.where(mask, values, 0.0).sum()
        pair = torch.stack((count, value)).float()
        if name not in self.pending:
            self.pending[name] = torch.zeros(2, device=self.device)
        self.pending[name] += pair

    def _finish(self, mask, failed):
        for stopping, prefix, offset in ((True, 'Stopping', 0), (False, 'Transition', 2)):
            selected = mask & (self.stopping == stopping)
            self._add(prefix + '/failure_rate', selected, failed.float())
            if stopping:
                success = selected & ~failed
                self._add('Stopping/body_time', success, self.body_time)
                self._add('Stopping/distance', success, self.body_distance)
                self._add('Stopping/quiet_time', success, self.quiet_time)
                sources = selected[:, None] & (self.source_modes[:, None] == self.source_ids)
                pairs = torch.stack((sources.sum(dim=0), (sources & failed[:, None]).sum(dim=0)), dim=1).float()
                self.stopping_quality += pairs
                for index, name in enumerate(('forward', 'backward', 'lateral', 'diagonal', 'turn', 'moving_turn')):
                    key = 'Stopping/from_' + name + '_failure_rate'
                    if key not in self.pending:
                        self.pending[key] = torch.zeros(2, device=self.device)
                    self.pending[key] += pairs[index]
            self.quality[offset] += selected.sum()
            self.quality[offset + 1] += (selected & failed).sum()
        self.active[mask] = False

    def begin(self, env_ids, previous, requested, position, source_modes):
        changed = torch.linalg.vector_norm(requested - previous, dim=1) > 1e-6
        interrupted = torch.zeros_like(self.active)
        interrupted[env_ids] = changed & self.active[env_ids]
        self._add('Transition/interrupted', interrupted)
        self._finish(interrupted, interrupted)
        selected = env_ids[changed]
        self.active[selected] = True
        self.stopping[selected] = torch.linalg.vector_norm(requested[changed], dim=1) < 1e-7
        self.source_modes[selected] = source_modes[changed]
        self.elapsed[selected] = 0
        self.body_hold[selected] = 0
        self.quiet_hold[selected] = 0
        self.target_hold[selected] = 0
        self.body_done[selected] = False
        self.quiet_done[selected] = False
        self.origin[selected] = position[changed, :2]

    def reset(self, env_ids):
        self.active[env_ids] = False
        self.body_done[env_ids] = False
        self.quiet_done[env_ids] = False
        self.elapsed[env_ids] = 0
        self.velocity_mean[env_ids] = 0

    def observe(self, requested, reference, velocity, yaw, position, gravity_z,
                joint_velocity, contacts, done, time_out, phase, normal_stance):
        self.elapsed += self.active * self.dt
        actual = torch.cat((velocity[:, :2], yaw[:, None]), dim=1)
        self.velocity_mean.lerp_(actual, self.dt / (0.2 + self.dt))
        upright = (position[:, 2] > 0.18) & (gravity_z < -0.866)
        body = (torch.linalg.vector_norm(velocity[:, :2], dim=1) < 0.1) & (yaw.abs() < 0.1) & upright
        quiet = ((torch.linalg.vector_norm(velocity[:, :2], dim=1) < 0.05) & (yaw.abs() < 0.05) &
                 (joint_velocity.square().mean(dim=1).sqrt() < 0.25) & contacts.all(dim=1) & upright & normal_stance &
                 (torch.linalg.vector_norm(reference, dim=1) < 1e-7))
        stopping = self.active & self.stopping
        starting_body = stopping & body & (self.body_hold == 0)
        distance = torch.linalg.vector_norm(position[:, :2] - self.origin, dim=1)
        self.candidate_distance[starting_body] = distance[starting_body]
        self.body_hold.copy_(torch.where(stopping & body, self.body_hold + self.dt, 0.0))
        self.quiet_hold.copy_(torch.where(stopping & quiet, self.quiet_hold + self.dt, 0.0))
        # A lost stable interval must be acquired again before the event can succeed.
        self.body_done &= stopping & body
        self.quiet_done &= stopping & quiet
        body_finished = stopping & ~self.body_done & (self.body_hold + 1e-5 >= 1.0)
        quiet_finished = stopping & ~self.quiet_done & (self.quiet_hold + 1e-5 >= 0.5)
        self.body_time[body_finished] = (self.elapsed - self.body_hold + self.dt).clamp(min=0)[body_finished]
        self.body_distance[body_finished] = self.candidate_distance[body_finished]
        self.quiet_time[quiet_finished] = (self.elapsed - self.quiet_hold + self.dt).clamp(min=0)[quiet_finished]
        self.body_done |= body_finished
        self.quiet_done |= quiet_finished
        xy_error = torch.linalg.vector_norm(self.velocity_mean[:, :2] - requested[:, :2], dim=1)
        yaw_error = (self.velocity_mean[:, 2] - requested[:, 2]).abs()
        target_speed = torch.linalg.vector_norm(requested[:, :2], dim=1)
        acquired = (xy_error < (0.12 + 0.10 * target_speed)) & (yaw_error < (0.12 + 0.10 * requested[:, 2].abs())) & upright
        self.target_hold.copy_(torch.where(self.active & ~self.stopping & acquired, self.target_hold + self.dt, 0.0))
        # 停车完整观察五秒，防止先短暂回正、随后塌姿仍被当作成功。
        settled = torch.where(self.stopping, self.body_done & self.quiet_done & body & quiet, self.target_hold >= 0.5)
        failed = self.active & done.bool() & ~time_out.bool()
        deadline = torch.where(self.stopping, self.cfg.stop_observation_time, 5.0)
        required = torch.where(self.stopping, self.cfg.stop_observation_time, 2.0)
        expired = self.active & (self.elapsed + 1e-5 >= deadline) & ~settled
        completed = self.active & (self.elapsed + 1e-5 >= required) & settled
        self._add('Transition/acquisition_time', completed & ~self.stopping & ~failed, (self.elapsed - self.target_hold + self.dt).clamp(min=0))
        self._finish(failed | expired | completed, failed | expired)
        censored = self.active & done.bool() & time_out.bool()
        self._add('Transition/censored', censored)
        self.active[censored] = False
        ready = torch.linalg.vector_norm(reference - requested, dim=1) < 0.03
        low_xy = ready & (target_speed >= self.cfg.min_translation_speed) & (target_speed <= self.cfg.low_speed_boundary)
        low_yaw = ready & (requested[:, 2].abs() >= self.cfg.min_turn_yaw) & (requested[:, 2].abs() <= self.cfg.low_speed_boundary)
        raw_xy_error = torch.linalg.vector_norm(velocity[:, :2] - requested[:, :2], dim=1)
        raw_yaw_error = (yaw - requested[:, 2]).abs()
        self._add('Tracking/low_xy_error', low_xy, raw_xy_error)
        self._add('Tracking/low_yaw_error', low_yaw, raw_yaw_error)
        self.quality[4] += low_xy.sum()
        self.quality[5] += torch.where(low_xy, raw_xy_error, 0.0).sum()
        self.quality[6] += low_yaw.sum()
        self.quality[7] += torch.where(low_yaw, raw_yaw_error, 0.0).sum()
        x, y, w = requested[:, 0], requested[:, 1], requested[:, 2]
        has_x, has_y, turning = x.abs() > 1e-6, y.abs() > 1e-6, w.abs() > 1e-6
        group = torch.where(has_x, (x < 0).long(), 2 + (y < 0).long())
        group = torch.where(has_x & has_y, 4 + 2 * (x < 0).long() + (y < 0).long(), group)
        bearing = torch.remainder(torch.round(torch.atan2(y, x) * (4 / 3.141592653589793)).long(), 8)
        group = torch.where(turning, torch.where(has_x | has_y, 10 + 2 * bearing + (w < 0).long(), 8 + (w < 0).long()), group)
        directional = torch.zeros_like(self.direction_quality)
        directional.index_add_(0, group, torch.stack((low_xy.float(), torch.where(low_xy, raw_xy_error, 0.0), low_yaw.float(), torch.where(low_yaw, raw_yaw_error, 0.0)), dim=1))
        self.direction_quality += directional
        self.pending['Tracking/directional_low_xy_error'] = directional[:, :2]
        self.pending['Tracking/directional_low_yaw_error'] = directional[:, 2:]
        for index, name in enumerate(('moving', 'braking', 'settling', 'standing')):
            self._add('Phase/' + name + '_fraction', torch.ones_like(self.active), (phase == index).float())
        self.quality_elapsed += self.dt
        if self.quality_elapsed + 1e-6 >= self.cfg.curriculum_min_time:
            q = self.quality
            enough = (q[0] >= self.cfg.quality_min_events) & (q[2] >= self.cfg.quality_min_events) & (q[4] > 0) & (q[6] > 0)
            d = self.direction_quality
            xy_required = torch.ones(26, device=self.device, dtype=torch.bool)
            xy_required[8:10] = False
            enough_directions = (d[xy_required, 0] >= self.cfg.quality_min_direction_samples).all() & (d[8:, 2] >= self.cfg.quality_min_direction_samples).all()
            directions_pass = (d[xy_required, 1] / d[xy_required, 0].clamp(min=1) <= self.cfg.quality_low_xy_error).all() & (d[8:, 3] / d[8:, 2].clamp(min=1) <= self.cfg.quality_low_yaw_error).all()
            s = self.stopping_quality
            enough_sources = (s[:, 0] >= self.cfg.quality_min_stopping_per_mode).all()
            sources_pass = (s[:, 1] / s[:, 0].clamp(min=1) <= self.cfg.quality_max_failure_rate).all()
            self.quality_ready = bool(enough & enough_directions & directions_pass & enough_sources & sources_pass &
                                      (q[1] / q[0].clamp(min=1) <= self.cfg.quality_max_failure_rate) &
                                      (q[3] / q[2].clamp(min=1) <= self.cfg.quality_max_failure_rate))
            self.quality_elapsed = 0.0
            if enough and enough_directions and enough_sources:
                q.zero_()
                d.zero_()
                s.zero_()
        metrics = self.pending
        self.pending = {}
        return metrics
