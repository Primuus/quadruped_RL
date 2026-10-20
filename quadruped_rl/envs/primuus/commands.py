"""Primuus 连续速度、过渡序列及分方向转向课程；不依赖 Isaac Gym。"""

import math

import torch


MODE_NAMES = ("stand", "forward", "backward", "lateral", "diagonal", "turn", "moving_turn")


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
        # Completed high-speed command segments: count, mean tracking sum, failures.
        self.window = torch.zeros(3, device=device)
        self.check_elapsed = 0.0
        self.last_tracking = 0.0
        self.last_failure_rate = 0.0
        self.last_segments = 0
        self.turn_limit = cfg.initial_turn_speed
        self.moving_turn_limit = cfg.initial_turn_speed
        self.transition_limit = min(cfg.transition_initial_speed, self.forward_limit)
        self.repair_elapsed = 0.0
        self.sequence_env = torch.arange(num_envs, device=device) < int(num_envs * cfg.transition_fraction) if training else torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.sequence_target = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.segment_modes = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.segment_yaw = torch.zeros(num_envs, device=device)
        self.segment_yaw_limit = torch.zeros(num_envs, device=device)
        self.yaw_error_sum = torch.zeros(num_envs, device=device)
        self.yaw_valid_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.turn_window = torch.zeros(2, 2, 3, device=device)  # mode, direction, count/error/failure
        self.last_quality_ready = False

    def _validate(self):
        cfg = self.cfg
        if tuple(cfg.mode_names) != MODE_NAMES:
            raise ValueError("Primuus commands.mode_names must match the seven supported modes")
        for name, length in (("mode_probabilities", 7), ("forward_speed_probabilities", 3)):
            values = getattr(cfg, name)
            if len(values) != length or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError("commands.%s must contain %d nonnegative probabilities summing to 1" % (name, length))
        positive = (
            "min_translation_speed", "min_turn_yaw", "max_translation_speed", "max_curriculum",
            "initial_forward_speed", "max_backward_speed", "max_lateral_speed", "max_diagonal_speed",
            "max_moving_turn_speed", "curriculum_step", "curriculum_min_time",
            "curriculum_min_segment_time", "resampling_time",
        )
        if any(not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0 for name in positive):
            raise ValueError("Primuus command speeds and curriculum durations must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.tracking_sigma) or self.tracking_sigma <= 0:
            raise ValueError("dt and tracking_sigma must be finite and positive")
        if cfg.heading_command:
            raise ValueError("Primuus directional commands require heading_command=False")
        if not cfg.min_translation_speed <= cfg.initial_forward_speed <= cfg.max_curriculum <= cfg.max_translation_speed:
            raise ValueError("Primuus forward speed limits must be ordered")
        for name in ("max_backward_speed", "max_lateral_speed", "max_diagonal_speed", "max_moving_turn_speed"):
            if not cfg.min_translation_speed <= getattr(cfg, name) <= cfg.max_translation_speed:
                raise ValueError("commands.%s must fit the translation speed limits" % name)
        x_range, y_range, yaw_range = cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_yaw
        if any(len(r) != 2 or any(not math.isfinite(v) for v in r) or r[0] >= r[1] for r in (x_range, y_range, yaw_range)):
            raise ValueError("Primuus command ranges must contain two ordered finite values")
        if x_range[0] > -cfg.max_backward_speed or x_range[1] < cfg.max_curriculum:
            raise ValueError("commands.ranges.lin_vel_x must cover backward and final forward speeds")
        mixed_max = max(cfg.max_lateral_speed, cfg.max_diagonal_speed, cfg.max_moving_turn_speed)
        if y_range[0] > -mixed_max or y_range[1] < mixed_max or x_range[0] > -max(cfg.max_diagonal_speed, cfg.max_moving_turn_speed):
            raise ValueError("Primuus XY ranges must cover all lateral and mixed commands")
        if not cfg.min_turn_yaw <= min(-yaw_range[0], yaw_range[1]):
            raise ValueError("min_turn_yaw must fit both yaw directions")
        bins = cfg.forward_speed_bins
        if len(bins) != 3 or any(len(b) != 2 or any(not math.isfinite(v) for v in b) or b[0] >= b[1] for b in bins):
            raise ValueError("forward_speed_bins must contain three ordered finite intervals")
        if bins[0][0] != cfg.min_translation_speed or bins[-1][1] != cfg.max_curriculum or any(bins[i][1] != bins[i + 1][0] for i in range(2)):
            raise ValueError("forward_speed_bins must continuously cover the full forward range")
        if cfg.forward_speed_probabilities[0] <= 0:
            raise ValueError("the low-speed forward bin must have nonzero probability")
        if not 0 < cfg.curriculum_high_speed_fraction < 1 or not 0 < cfg.curriculum_tracking_threshold <= 1 or not 0 <= cfg.curriculum_max_failure_rate < 1:
            raise ValueError("Primuus curriculum fractions are outside their valid ranges")
        if not math.isfinite(cfg.curriculum_settle_time) or cfg.curriculum_settle_time < 0 or cfg.curriculum_settle_time + cfg.curriculum_min_segment_time > cfg.resampling_time:
            raise ValueError("command segments must allow settling and a tracking evaluation interval")
        if not isinstance(cfg.curriculum_min_segments, int) or cfg.curriculum_min_segments < 1:
            raise ValueError("curriculum_min_segments must be a positive integer")
        if not 0 < cfg.min_resampling_time <= cfg.resampling_time or cfg.min_resampling_time < cfg.curriculum_settle_time + cfg.curriculum_min_segment_time:
            raise ValueError("Random command durations must permit settling and evaluation")
        if not 0 <= cfg.transition_fraction <= 1 or not 0 < cfg.low_speed_probability < 1:
            raise ValueError("Invalid command mixture probabilities")
        if not minimum_valid(cfg):
            raise ValueError("Invalid Primuus turn/transition curriculum or repair settings")

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
        # Closed bins are removed and the remaining probabilities are renormalized.
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
        modes = torch.multinomial(self.probabilities, count, replacement=True)
        cfg = self.cfg
        minimum = cfg.min_translation_speed
        # Sample all tensors at once; no per-environment Python loop or bearing-changing clamp.
        forward = self._forward(count)
        backward = -self._translation(count, cfg.max_backward_speed)
        lateral = self._translation(count, cfg.max_lateral_speed) * self._sign(count)
        # Keep each diagonal component nonzero and cover all four quadrants equally.
        angle = self._uniform(count, math.pi / 12, 5 * math.pi / 12)
        angle = torch.where(torch.rand(count, device=self.device) < cfg.keyboard_bearing_probability, torch.full_like(angle, math.pi / 4), angle)
        diagonal_speed = self._translation(count, cfg.max_diagonal_speed)
        diagonal_x = diagonal_speed * torch.cos(angle) * self._sign(count)
        diagonal_y = diagonal_speed * torch.sin(angle) * self._sign(count)
        moving_angle = self._uniform(count, -math.pi, math.pi)
        moving_speed = self._translation(count, cfg.max_moving_turn_speed)
        unit = 2 ** -0.5
        bearings = torch.tensor([[1, 0], [unit, unit], [0, 1], [-unit, unit], [-1, 0], [-unit, -unit], [0, -1], [unit, -unit]], device=self.device)
        anchors = bearings[torch.randint(8, (count,), device=self.device)]
        moving_direction = torch.where((torch.rand(count, device=self.device) < cfg.keyboard_bearing_probability)[:, None], anchors, torch.stack((torch.cos(moving_angle), torch.sin(moving_angle)), dim=1))
        # Dedicated environments alternate a straight source with a difficult destination.
        sequence = self.sequence_env[env_ids]
        destination = self.sequence_target[env_ids]
        choices = torch.tensor([0, 2, 6, 4, 5], device=self.device)
        weights = torch.tensor([0.35, 0.15, 0.20, 0.20, 0.10], device=self.device)
        sequence_modes = choices[torch.multinomial(weights, count, replacement=True)]
        modes = torch.where(sequence, torch.where(destination, sequence_modes, torch.ones_like(modes)), modes)
        source_speed = self._uniform(count, max(minimum, 0.8 * self.transition_limit), self.transition_limit)
        forward = torch.where(sequence & ~destination, source_speed, forward)
        self.sequence_target[env_ids] = torch.where(sequence, ~destination, destination)
        commands[:, 0] = torch.where(modes == 1, forward, torch.where(modes == 2, backward, 0.0))
        commands[:, 1] = torch.where(modes == 3, lateral, 0.0)
        commands[:, 0] += torch.where(modes == 4, diagonal_x, torch.where(modes == 6, moving_speed * moving_direction[:, 0], 0.0))
        commands[:, 1] += torch.where(modes == 4, diagonal_y, torch.where(modes == 6, moving_speed * moving_direction[:, 1], 0.0))
        yaw_max = torch.where(modes == 5, torch.full_like(forward, self.turn_limit), torch.full_like(forward, self.moving_turn_limit))
        yaw = self._uniform(count, cfg.min_turn_yaw, yaw_max) * self._sign(count)
        commands[:, 2] = torch.where((modes == 5) | (modes == 6), yaw, 0.0)
        self.active[env_ids] = True
        self.high_speed[env_ids] = (modes == 1) & (commands[:, 0] >= self.forward_limit * cfg.curriculum_high_speed_fraction)
        self.segment_limit[env_ids] = self.forward_limit
        self.segment_steps[env_ids] = 0
        self.valid_steps[env_ids] = 0
        self.tracking_sum[env_ids] = 0
        self.segment_modes[env_ids] = modes
        self.segment_yaw[env_ids] = commands[:, 2]
        self.segment_yaw_limit[env_ids] = yaw_max
        self.yaw_error_sum[env_ids] = 0
        self.yaw_valid_steps[env_ids] = 0
        return commands, modes

    def observe(self, commands, actual_velocity, done, time_out, actual_yaw=None, quality_ready=False):
        """Observe the command actually used by this physics step, before reset/resampling."""
        if not self.curriculum_enabled:
            return
        self.segment_steps += self.active.long()
        self.repair_elapsed += self.dt
        valid = self.active & self.high_speed & (commands[:, 0] >= self.segment_limit * self.cfg.curriculum_high_speed_fraction) & (commands[:, 1].abs() < 1e-6) & (commands[:, 2].abs() < 1e-6) & (self.segment_steps * self.dt > self.cfg.curriculum_settle_time)
        error = torch.sum(torch.square(commands[:, :2] - actual_velocity[:, :2]), dim=1)
        score = torch.exp(-error / self.tracking_sigma)
        self.tracking_sum += torch.where(valid, score, 0.0)
        self.valid_steps += valid.long()
        if actual_yaw is not None:
            yaw_valid = self.active & (self.segment_steps * self.dt > self.cfg.curriculum_settle_time)
            self.yaw_error_sum += torch.where(yaw_valid, (self.segment_yaw - actual_yaw).abs(), 0.0)
            self.yaw_valid_steps += yaw_valid.long()
        done_ids = done.nonzero(as_tuple=False).flatten()
        self._finish(done_ids, (done.bool() & ~time_out.bool())[done_ids])
        self.check_elapsed += self.dt
        if self.check_elapsed + 1e-9 < self.cfg.curriculum_min_time:
            return
        self.check_elapsed = 0.0
        count, score_sum, failures = self.window.tolist()
        if count >= self.cfg.curriculum_min_segments:
            self.last_segments = int(count)
            self.last_tracking = score_sum / count
            self.last_failure_rate = failures / count
            if self.last_tracking >= self.cfg.curriculum_tracking_threshold and self.last_failure_rate <= self.cfg.curriculum_max_failure_rate:
                self.forward_limit = min(self.forward_limit + self.cfg.curriculum_step, self.cfg.max_curriculum)
            self.window.zero_()
        self.last_quality_ready = bool(quality_ready)
        if quality_ready:
            self.transition_limit = min(self.transition_limit + self.cfg.transition_step, self.forward_limit)
        repair_ready = quality_ready and self.repair_elapsed >= self.cfg.repair_duration and self.transition_limit >= self.cfg.max_curriculum
        for index, name in enumerate(("turn_limit", "moving_turn_limit")):
            window = self.turn_window[index]
            if torch.all(window[:, 0] >= self.cfg.curriculum_min_segments):
                errors = window[:, 1] / window[:, 0]
                failures = window[:, 2] / window[:, 0]
                if repair_ready and torch.all(errors <= self.cfg.turn_error_threshold) and torch.all(failures <= self.cfg.quality_max_failure_rate):
                    setattr(self, name, min(getattr(self, name) + self.cfg.turn_curriculum_step, self.cfg.max_turn_speed))
                window.zero_()

    def _finish(self, env_ids, failed=None):
        if len(env_ids) == 0:
            return
        if self.curriculum_enabled:
            if failed is None:
                failed = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            # Early falls count even before the tracking warmup; they cannot be silently discarded.
            enough_time = self.valid_steps[env_ids] * self.dt + 1e-6 >= self.cfg.curriculum_min_segment_time
            eligible = self.active[env_ids] & self.high_speed[env_ids] & (self.segment_limit[env_ids] == self.forward_limit) & (enough_time | failed)
            mean_score = self.tracking_sum[env_ids] / self.valid_steps[env_ids].clamp(min=1)
            self.window += torch.stack((eligible.sum(), torch.where(eligible, mean_score, 0.0).sum(), (eligible & failed).sum()))
            yaw_steps = self.yaw_valid_steps[env_ids]
            for index, mode in enumerate((5, 6)):
                current_limit = self.turn_limit if index == 0 else self.moving_turn_limit
                for direction in range(2):
                    selected = self.active[env_ids] & (self.segment_modes[env_ids] == mode) & ((self.segment_yaw[env_ids] > 0) == (direction == 1)) & (self.segment_yaw[env_ids].abs() >= current_limit * self.cfg.curriculum_high_speed_fraction) & (self.segment_yaw_limit[env_ids] == current_limit) & ((yaw_steps * self.dt >= self.cfg.curriculum_min_segment_time) | failed)
                    yaw_error = self.yaw_error_sum[env_ids] / yaw_steps.clamp(min=1)
                    self.turn_window[index, direction] += torch.stack((selected.sum(), torch.where(selected, yaw_error, 0.0).sum(), (selected & failed).sum()))
        self.active[env_ids] = False

    def state_dict(self):
        return {
            "version": 2,
            "forward_limit": float(self.forward_limit),
            "window": self.window.tolist(),
            "check_elapsed": self.check_elapsed,
            "last_tracking": self.last_tracking,
            "last_failure_rate": self.last_failure_rate,
            "last_segments": self.last_segments,
            "turn_limit": self.turn_limit,
            "moving_turn_limit": self.moving_turn_limit,
            "transition_limit": self.transition_limit,
            "repair_elapsed": self.repair_elapsed,
            "turn_window": self.turn_window.tolist(),
        }

    def load_state_dict(self, state):
        # Old checkpoints have no curriculum state: start at the configured initial speed.
        self.active.zero_()
        self.window.zero_()
        self.check_elapsed = 0.0
        self.turn_limit = self.cfg.initial_turn_speed
        self.moving_turn_limit = self.cfg.initial_turn_speed
        self.transition_limit = self.cfg.transition_initial_speed
        self.repair_elapsed = 0.0
        self.turn_window.zero_()
        self.sequence_target.zero_()
        self.last_quality_ready = False
        if state is None:
            self.forward_limit = self.cfg.initial_forward_speed if self.cfg.curriculum else self.cfg.max_curriculum
            return
        if state.get("version") not in (1, 2) or not math.isfinite(state["forward_limit"]):
            raise ValueError("unsupported or invalid Primuus command curriculum checkpoint")
        saved_limit = state["forward_limit"]
        self.forward_limit = min(max(saved_limit, self.cfg.initial_forward_speed), self.cfg.max_curriculum) if self.cfg.curriculum else self.cfg.max_curriculum
        if self.forward_limit == saved_limit:
            window = state.get("window", [0.0, 0.0, 0.0])
            elapsed = state.get("check_elapsed", 0.0)
            if len(window) != 3 or any(not math.isfinite(v) or v < 0 for v in window) or not 0 <= window[1] <= window[0] or not 0 <= window[2] <= window[0] or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid Primuus curriculum evaluation window")
            self.window[:] = torch.tensor(window, device=self.device)
            self.check_elapsed = elapsed
        self.last_tracking = state.get("last_tracking", 0.0)
        self.last_failure_rate = state.get("last_failure_rate", 0.0)
        self.last_segments = state.get("last_segments", 0)
        for name, low, high in (("turn_limit", self.cfg.initial_turn_speed, self.cfg.max_turn_speed), ("moving_turn_limit", self.cfg.initial_turn_speed, self.cfg.max_turn_speed), ("transition_limit", self.cfg.transition_initial_speed, self.forward_limit)):
            value = state.get(name, low)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid saved %s" % name)
            setattr(self, name, value)
        self.repair_elapsed = state.get("repair_elapsed", 0.0)
        if not math.isfinite(self.repair_elapsed) or self.repair_elapsed < 0:
            raise ValueError("Invalid repair elapsed time")
        turn_window = torch.tensor(state.get('turn_window', torch.zeros(2, 2, 3).tolist()), device=self.device)
        if turn_window.shape != (2, 2, 3) or not torch.isfinite(turn_window).all() or (turn_window < 0).any() or (turn_window[..., 2] > turn_window[..., 0]).any():
            raise ValueError("Invalid saved turn evaluation window")
        self.turn_window.copy_(turn_window)

    def metrics(self):
        return {
            "forward_limit": self.forward_limit,
            "high_speed_tracking": self.last_tracking,
            "high_speed_failure_rate": self.last_failure_rate,
            "evaluated_segments": self.last_segments,
            "turn_limit": self.turn_limit,
            "moving_turn_limit": self.moving_turn_limit,
            "transition_limit": self.transition_limit,
            "repair_elapsed": self.repair_elapsed,
            "quality_ready": float(self.last_quality_ready),
        }


def minimum_valid(cfg):
    values = (cfg.transition_initial_speed, cfg.transition_step, cfg.repair_duration, cfg.initial_turn_speed, cfg.max_turn_speed, cfg.turn_curriculum_step, cfg.turn_error_threshold, cfg.quality_low_xy_error, cfg.quality_low_yaw_error)
    return (all(math.isfinite(v) and v > 0 for v in values)
            and cfg.min_translation_speed <= cfg.low_speed_boundary <= min(cfg.max_backward_speed, cfg.max_lateral_speed, cfg.max_diagonal_speed, cfg.max_moving_turn_speed)
            and cfg.transition_initial_speed <= cfg.initial_forward_speed
            and cfg.min_turn_yaw <= cfg.initial_turn_speed <= cfg.max_turn_speed <= min(-cfg.ranges.ang_vel_yaw[0], cfg.ranges.ang_vel_yaw[1])
            and 0 <= cfg.quality_max_failure_rate < 1 and cfg.quality_min_events > 0 and cfg.quality_min_direction_samples > 0
            and 0 <= cfg.keyboard_bearing_probability <= 1
            and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for k, v in vars(cfg.transition).items() if not k.startswith('_') and k != 'enabled')
            and isinstance(cfg.transition.enabled, bool)
            and cfg.transition.transition_speed <= min(cfg.max_backward_speed, cfg.max_lateral_speed, cfg.max_diagonal_speed, cfg.max_moving_turn_speed))
