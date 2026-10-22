"""Primuus 八种运动及站立采样；仅直线前进使用速度课程。"""

import math

import torch


MODE_NAMES = ("stand", "forward", "backward", "lateral", "turn", "moving_turn")


class PrimuusCommandSampler:
    def __init__(self, cfg, num_envs, device, dt, tracking_sigma, training=True):
        self.cfg = cfg
        self.device = device
        self.dt = dt
        self.tracking_sigma = tracking_sigma
        self._validate()
        self.curriculum_enabled = bool(cfg.curriculum and training)
        self.forward_limit = cfg.initial_forward_speed if cfg.curriculum else cfg.max_curriculum
        self.probabilities = torch.tensor(cfg.mode_probabilities, dtype=torch.float, device=device)
        self.bin_probabilities = torch.tensor(cfg.forward_speed_probabilities, dtype=torch.float, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.high_speed = torch.zeros_like(self.active)
        self.segment_limit = torch.zeros(num_envs, device=device)
        self.segment_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.valid_steps = torch.zeros_like(self.segment_steps)
        self.tracking_sum = torch.zeros(num_envs, device=device)
        # Completed high-speed forward segments: count, mean tracking sum, failures.
        self.window = torch.zeros(3, device=device)
        self.check_elapsed = 0.0
        self.last_tracking = 0.0
        self.last_failure_rate = 0.0
        self.last_segments = 0

    def _validate(self):
        cfg = self.cfg
        if tuple(cfg.mode_names) != MODE_NAMES:
            raise ValueError("Primuus commands.mode_names must match the six supported modes")
        for name, length in (("mode_probabilities", 6), ("forward_speed_probabilities", 3)):
            values = getattr(cfg, name)
            if len(values) != length or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError("commands.%s must contain %d nonnegative probabilities summing to 1" % (name, length))
        positive = (
            "min_translation_speed", "min_turn_yaw", "max_translation_speed", "max_curriculum",
            "initial_forward_speed", "max_backward_speed", "max_lateral_speed", "max_moving_turn_speed",
            "max_turn_speed", "curriculum_step", "curriculum_min_time", "curriculum_min_segment_time",
        )
        if any(not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0 for name in positive):
            raise ValueError("Primuus command speeds and curriculum durations must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.tracking_sigma) or self.tracking_sigma <= 0:
            raise ValueError("dt and tracking_sigma must be finite and positive")
        if cfg.heading_command:
            raise ValueError("Primuus commands require heading_command=False")
        if not cfg.min_translation_speed <= cfg.initial_forward_speed <= cfg.max_curriculum <= cfg.max_translation_speed:
            raise ValueError("Primuus forward speed limits must be ordered")
        for name in ("max_backward_speed", "max_lateral_speed", "max_moving_turn_speed"):
            if not cfg.min_translation_speed <= getattr(cfg, name) <= cfg.max_translation_speed:
                raise ValueError("commands.%s must fit the translation speed limits" % name)
        if not cfg.min_turn_yaw <= cfg.max_turn_speed <= min(-cfg.ranges.ang_vel_yaw[0], cfg.ranges.ang_vel_yaw[1]):
            raise ValueError("Primuus turn speed must fit both yaw directions")
        if cfg.ranges.lin_vel_x[0] > -cfg.max_backward_speed or cfg.ranges.lin_vel_x[1] < max(cfg.max_curriculum, cfg.max_moving_turn_speed):
            raise ValueError("Primuus X range must cover forward, backward and moving turns")
        if cfg.ranges.lin_vel_y[0] > -cfg.max_lateral_speed or cfg.ranges.lin_vel_y[1] < cfg.max_lateral_speed:
            raise ValueError("Primuus Y range must cover both lateral directions")
        bins = cfg.forward_speed_bins
        if len(bins) != 3 or any(len(b) != 2 or any(not math.isfinite(v) for v in b) or b[0] >= b[1] for b in bins):
            raise ValueError("forward_speed_bins must contain three ordered finite intervals")
        if bins[0][0] != cfg.min_translation_speed or bins[-1][1] != cfg.max_curriculum or any(bins[i][1] != bins[i + 1][0] for i in range(2)):
            raise ValueError("forward_speed_bins must continuously cover the full forward range")
        if cfg.forward_speed_probabilities[0] <= 0 or not 0 < cfg.low_speed_probability < 1:
            raise ValueError("Primuus low-speed sampling probabilities must be positive")
        if not cfg.min_translation_speed <= cfg.low_speed_boundary <= min(cfg.max_backward_speed, cfg.max_lateral_speed, cfg.max_moving_turn_speed):
            raise ValueError("Primuus low-speed boundary must fit the translation ranges")
        if not 0 < cfg.curriculum_high_speed_fraction < 1 or not 0 < cfg.curriculum_tracking_threshold <= 1 or not 0 <= cfg.curriculum_max_failure_rate < 1:
            raise ValueError("Primuus curriculum fractions are outside their valid ranges")
        if not math.isfinite(cfg.curriculum_settle_time) or cfg.curriculum_settle_time < 0:
            raise ValueError("Primuus curriculum settling time must be finite and nonnegative")
        if not isinstance(cfg.curriculum_min_segments, int) or cfg.curriculum_min_segments < 1:
            raise ValueError("curriculum_min_segments must be a positive integer")
        if not cfg.curriculum_settle_time + cfg.curriculum_min_segment_time <= cfg.min_resampling_time <= cfg.resampling_time:
            raise ValueError("Command durations must allow settling and evaluation")
        if not 2 <= cfg.stop_observation_time < cfg.stand_min_resampling_time <= cfg.stand_resampling_time:
            raise ValueError("Standing durations must allow the parking observation window")

    def _uniform(self, count, low, high):
        return low + (high - low) * torch.rand(count, device=self.device)

    def _sign(self, count):
        return torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)

    def _translation(self, count, maximum):
        boundary = min(self.cfg.low_speed_boundary, maximum)
        low = self._uniform(count, self.cfg.min_translation_speed, boundary)
        high = self._uniform(count, boundary, maximum)
        return torch.where(torch.rand(count, device=self.device) < self.cfg.low_speed_probability, low, high)

    def _forward(self, count):
        if self.forward_limit == self.cfg.min_translation_speed:
            return torch.full((count,), self.forward_limit, device=self.device)
        weights = self.bin_probabilities.clone()
        for index, (low, _) in enumerate(self.cfg.forward_speed_bins):
            if low >= self.forward_limit:
                weights[index] = 0
        bins = torch.multinomial(weights, count, replacement=True)
        intervals = torch.tensor(self.cfg.forward_speed_bins, device=self.device)
        low = intervals[bins, 0]
        high = intervals[bins, 1].clamp(max=self.forward_limit)
        return low + (high - low) * torch.rand(count, device=self.device)

    def sample(self, env_ids):
        self._finish(env_ids)
        count = len(env_ids)
        commands = torch.zeros(count, 4, device=self.device)
        if count == 0:
            return commands, torch.empty(0, dtype=torch.long, device=self.device)
        cfg = self.cfg
        modes = torch.multinomial(self.probabilities, count, replacement=True)
        forward = self._forward(count)
        backward = -self._translation(count, cfg.max_backward_speed)
        lateral = self._translation(count, cfg.max_lateral_speed) * self._sign(count)
        moving_forward = self._translation(count, cfg.max_moving_turn_speed)
        yaw = self._uniform(count, cfg.min_turn_yaw, cfg.max_turn_speed) * self._sign(count)
        commands[:, 0] = torch.where(modes == 1, forward, torch.where(modes == 2, backward, 0.0))
        commands[:, 0] += torch.where(modes == 5, moving_forward, 0.0)
        commands[:, 1] = torch.where(modes == 3, lateral, 0.0)
        commands[:, 2] = torch.where((modes == 4) | (modes == 5), yaw, 0.0)
        self.active[env_ids] = True
        self.high_speed[env_ids] = (modes == 1) & (commands[:, 0] >= self.forward_limit * cfg.curriculum_high_speed_fraction)
        self.segment_limit[env_ids] = self.forward_limit
        self.segment_steps[env_ids] = 0
        self.valid_steps[env_ids] = 0
        self.tracking_sum[env_ids] = 0
        return commands, modes

    def observe(self, commands, actual_velocity, done, time_out):
        if not self.curriculum_enabled:
            return
        self.segment_steps += self.active.long()
        valid = (self.active & self.high_speed &
                 (commands[:, 0] >= self.segment_limit * self.cfg.curriculum_high_speed_fraction) &
                 (commands[:, 1].abs() < 1e-6) & (commands[:, 2].abs() < 1e-6) &
                 (self.segment_steps * self.dt > self.cfg.curriculum_settle_time))
        error = torch.sum(torch.square(commands[:, :2] - actual_velocity[:, :2]), dim=1)
        self.tracking_sum += torch.where(valid, torch.exp(-error / self.tracking_sigma), 0.0)
        self.valid_steps += valid.long()
        done_ids = done.nonzero(as_tuple=False).flatten()
        self._finish(done_ids, (done.bool() & ~time_out.bool())[done_ids])
        self.check_elapsed += self.dt
        if self.check_elapsed + 1e-9 < self.cfg.curriculum_min_time:
            return
        self.check_elapsed = 0.0
        count, score_sum, failures = self.window.tolist()
        if count < self.cfg.curriculum_min_segments:
            return
        self.last_segments = int(count)
        self.last_tracking = score_sum / count
        self.last_failure_rate = failures / count
        if self.last_tracking >= self.cfg.curriculum_tracking_threshold and self.last_failure_rate <= self.cfg.curriculum_max_failure_rate:
            self.forward_limit = min(self.forward_limit + self.cfg.curriculum_step, self.cfg.max_curriculum)
        self.window.zero_()

    def _finish(self, env_ids, failed=None):
        if len(env_ids) == 0:
            return
        if self.curriculum_enabled:
            if failed is None:
                failed = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            enough_time = self.valid_steps[env_ids] * self.dt + 1e-6 >= self.cfg.curriculum_min_segment_time
            eligible = self.active[env_ids] & self.high_speed[env_ids] & (self.segment_limit[env_ids] == self.forward_limit) & (enough_time | failed)
            mean_score = self.tracking_sum[env_ids] / self.valid_steps[env_ids].clamp(min=1)
            self.window += torch.stack((eligible.sum(), torch.where(eligible, mean_score, 0.0).sum(), (eligible & failed).sum()))
        self.active[env_ids] = False

    def state_dict(self):
        return {
            "version": 6,
            "forward_limit": float(self.forward_limit),
            "window": self.window.tolist(),
            "check_elapsed": self.check_elapsed,
            "last_tracking": self.last_tracking,
            "last_failure_rate": self.last_failure_rate,
            "last_segments": self.last_segments,
        }

    def load_state_dict(self, state):
        self.active.zero_()
        self.window.zero_()
        self.check_elapsed = 0.0
        self.forward_limit = self.cfg.initial_forward_speed if self.cfg.curriculum else self.cfg.max_curriculum
        if state is None:
            return
        if state.get("version") not in (1, 2, 3, 4, 5, 6) or not math.isfinite(state["forward_limit"]):
            raise ValueError("unsupported or invalid Primuus command curriculum checkpoint")
        saved_limit = state["forward_limit"]
        self.forward_limit = min(max(saved_limit, self.cfg.initial_forward_speed), self.cfg.max_curriculum) if self.cfg.curriculum else self.cfg.max_curriculum
        if state['version'] == 6 and self.forward_limit == saved_limit:
            window = state.get("window", [0.0, 0.0, 0.0])
            elapsed = state.get("check_elapsed", 0.0)
            if len(window) != 3 or any(not math.isfinite(v) or v < 0 for v in window) or not 0 <= window[1] <= window[0] or not 0 <= window[2] <= window[0] or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid Primuus curriculum evaluation window")
            self.window[:] = torch.tensor(window, device=self.device)
            self.check_elapsed = elapsed
        # 旧版本只保留前进上限；已删除的动作分布和质量门槛窗口不再恢复。
        self.last_tracking = state.get("last_tracking", 0.0)
        self.last_failure_rate = state.get("last_failure_rate", 0.0)
        self.last_segments = state.get("last_segments", 0)

    def metrics(self):
        return {
            "forward_limit": self.forward_limit,
            "high_speed_tracking": self.last_tracking,
            "high_speed_failure_rate": self.last_failure_rate,
            "evaluated_segments": self.last_segments,
            "turn_limit": self.cfg.max_turn_speed,
            "moving_turn_limit": self.cfg.max_turn_speed,
        }
