"""CPU regression checks: python -m unittest discover -s tests -v (no Isaac Gym needed)."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load actual config files without executing envs/__init__.py (which requires Isaac Gym).
base_config = load_source("test_base_config", "quadruped_rl/envs/base/base_config.py")
with patch.dict(sys.modules, {"quadruped_rl.envs.base.base_config": base_config}):
    robot_config = load_source("quadruped_rl.envs.base.legged_robot_config", "quadruped_rl/envs/base/legged_robot_config.py")
with patch.dict(sys.modules, {"quadruped_rl.envs.base.legged_robot_config": robot_config}):
    primuus_config = load_source("test_primuus_config", "quadruped_rl/envs/primuus/primuus_config.py")
commands_module = load_source("test_primuus_commands_source", "quadruped_rl/envs/primuus/commands.py")
Sampler = commands_module.PrimuusCommandSampler


def config(**overrides):
    cfg = copy.deepcopy(primuus_config.PrimuusCfg().commands)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def sampler(count=4096, dt=0.02, **overrides):
    return Sampler(config(**overrides), count, "cpu", dt, 0.25)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_mode_probabilities_and_direction_bounds(self):
        count = 160000
        sample = sampler(count, curriculum=False)
        command, modes = sample.sample(torch.arange(count))
        frequencies = torch.bincount(modes, minlength=7) / count
        torch.testing.assert_close(frequencies, torch.tensor([.20, .35, .10, .10, .05, .10, .10]), atol=.004, rtol=0)
        self.assertTrue(torch.isfinite(command).all())
        self.assertTrue((command[:, 3] == 0).all())
        self.assertTrue((command[modes == 0] == 0).all())
        for mode, axis, sign, maximum in ((1, 0, 1, 4), (2, 0, -1, 1)):
            rows = command[modes == mode]
            self.assertTrue((rows[:, axis] * sign >= .12 - 1e-6).all())
            self.assertTrue((rows[:, axis] * sign <= maximum + 1e-6).all())
            self.assertTrue((rows[:, 1:] == 0).all())
        side = command[modes == 3]
        self.assertTrue((side[:, [0, 2]] == 0).all())
        self.assertTrue((side[:, 1].abs() >= .12 - 1e-6).all())
        self.assertTrue((side[:, 1].abs() <= 1).all())
        for mode in (4, 6):
            rows = command[modes == mode]
            speed = torch.linalg.vector_norm(rows[:, :2], dim=1)
            self.assertTrue((speed >= .12 - 1e-6).all())
            self.assertTrue((speed <= 1 + 1e-6).all())
            for axis in (0, 1):
                self.assertAlmostEqual((rows[:, axis] > 0).float().mean().item(), .5, delta=.02)
        self.assertTrue((command[modes == 4, 2] == 0).all())
        turn = command[modes == 5]
        self.assertTrue((turn[:, :2] == 0).all())
        for rows in (turn, command[modes == 6]):
            self.assertTrue((rows[:, 2].abs() >= .12 - 1e-6).all())
            self.assertTrue((rows[:, 2].abs() <= 1).all())
            self.assertAlmostEqual((rows[:, 2] > 0).float().mean().item(), .5, delta=.02)
        self.assertAlmostEqual((side[:, 1] > 0).float().mean().item(), .5, delta=.02)

    def test_continuous_forward_bins_and_progressive_renormalization(self):
        count = 80000
        sample = sampler(count, mode_probabilities=[0, 1, 0, 0, 0, 0, 0])
        ids = torch.arange(count)
        for limit, expected in ((1., [1, 0, 0]), (1.5, [3/7, 4/7, 0]), (2.5, [.3, .4, .3]), (4., [.3, .4, .3])):
            sample.forward_limit = limit
            command, _ = sample.sample(ids)
            x = command[:, 0]
            self.assertGreaterEqual(x.min().item(), .12 - 1e-6)
            self.assertLessEqual(x.max().item(), limit)
            self.assertGreater(x.max().item(), limit - .005)
            observed = torch.tensor([(x < 1).float().mean(), ((x >= 1) & (x < 2)).float().mean(), (x >= 2).float().mean()])
            torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.float), atol=.009, rtol=0)
            # Sampling covers the interval, not only the future deployment gear values.
            self.assertGreater(torch.unique((x * 1000).round()).numel(), 500)

    def test_partial_and_empty_resampling(self):
        sample = sampler(10)
        command, mode = sample.sample(torch.tensor([2, 8]))
        self.assertEqual(command.shape, (2, 4))
        self.assertEqual(mode.shape, (2,))
        self.assertEqual(sample.active.nonzero().flatten().tolist(), [2, 8])
        command, mode = sample.sample(torch.empty(0, dtype=torch.long))
        self.assertEqual(command.shape, (0, 4))
        self.assertEqual(mode.numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_sampling_and_curriculum_state(self):
        sample = Sampler(config(curriculum=False), 4096, "cuda:0", .02, .25)
        command, modes = sample.sample(torch.arange(4096, device="cuda:0"))
        self.assertEqual(command.device.type, "cuda")
        self.assertTrue(torch.isfinite(command).all())
        self.assertLessEqual(command[modes == 1, 0].max().item(), 4)
        self.assertLessEqual(torch.linalg.vector_norm(command[modes != 1, :2], dim=1).max().item(), 1 + 1e-6)
        state = sample.state_dict()
        sample.load_state_dict(state)
        self.assertEqual(sample.forward_limit, 4)

    def test_minimum_forward_limit_and_configured_ranges(self):
        sample = sampler(100, initial_forward_speed=.12)
        command, modes = sample.sample(torch.arange(100))
        self.assertTrue(torch.allclose(command[modes == 1, 0], torch.tensor(.12)))
        cfg = config()
        cfg.ranges.lin_vel_x = [-.5, 4]
        with self.assertRaises(ValueError):
            Sampler(cfg, 10, "cpu", .02, .25)

    def test_invalid_configuration_fails_before_simulation(self):
        for changes in (
            {"mode_probabilities": [0.25] * 4},
            {"mode_probabilities": [float("nan")] * 7},
            {"forward_speed_probabilities": [1, 0, -1]},
            {"max_backward_speed": 0},
            {"initial_forward_speed": 5},
            {"max_translation_speed": 1},
            {"heading_command": True},
            {"forward_speed_bins": [[.12, 1], [1.1, 2], [2, 4]]},
            {"curriculum_min_segments": 0},
            {"curriculum_settle_time": 5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                sampler(**changes)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        self.ids = torch.arange(2000)
        self.sample = sampler(2000, dt=.5, mode_probabilities=[0, 1, 0, 0, 0, 0, 0], curriculum_min_segments=20)
        self.zero = torch.zeros(2000, dtype=torch.bool)

    def segment(self, error=0, failed=None, timeout=False):
        command, _ = self.sample.sample(self.ids)
        actual = command[:, :3].clone()
        actual[:, 0] += error
        for _ in range(4):
            self.sample.observe(command, actual, self.zero, self.zero)
        done = self.zero if failed is None else failed
        self.sample.observe(command, actual, done, done if timeout else self.zero)
        self.sample.sample(self.ids)
        return command, actual

    def evaluate(self):
        self.sample.check_elapsed = self.sample.cfg.curriculum_min_time - self.sample.dt
        command = torch.zeros(2000, 4)
        self.sample.observe(command, command[:, :3], self.zero, self.zero)

    def test_only_good_high_speed_forward_segments_advance(self):
        self.segment()
        self.assertGreaterEqual(self.sample.window[0].item(), 20)
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 1.5)
        self.assertEqual(self.sample.last_failure_rate, 0)
        self.assertAlmostEqual(self.sample.last_tracking, 1)

    def test_tracking_failure_keeps_limit(self):
        self.segment(error=.6)
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 1)
        self.assertLess(self.sample.last_tracking, .8)

    def test_early_failures_count_before_settling(self):
        command, _ = self.sample.sample(self.ids)
        high_count = self.sample.high_speed.sum().item()
        self.sample.observe(command, command[:, :3], torch.ones_like(self.zero), self.zero)
        self.assertEqual(self.sample.window[0].item(), high_count)
        self.assertEqual(self.sample.window[2].item(), high_count)
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 1)
        self.assertEqual(self.sample.last_failure_rate, 1)

    def test_failures_block_advancement_even_with_good_tracking(self):
        self.segment(failed=torch.ones_like(self.zero))
        self.evaluate()
        self.assertAlmostEqual(self.sample.last_tracking, 1)
        self.assertEqual(self.sample.last_failure_rate, 1)
        self.assertEqual(self.sample.forward_limit, 1)

    def test_timeouts_are_not_counted_as_failures(self):
        self.segment(failed=torch.ones_like(self.zero), timeout=True)
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 1.5)

    def test_standing_and_low_commands_cannot_advance_curriculum(self):
        for probabilities in ([1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0]):
            self.sample = sampler(2000, dt=.5, mode_probabilities=probabilities, curriculum_min_segments=20)
            self.segment()
            self.evaluate()
            self.assertEqual(self.sample.forward_limit, 1)
            self.assertEqual(self.sample.window[0].item(), 0)
        self.sample = sampler(2000, dt=.5, curriculum_min_segments=20)
        command, _ = self.sample.sample(self.ids)
        self.sample.high_speed.zero_()
        for _ in range(4):
            self.sample.observe(command, command[:, :3], self.zero, self.zero)
        self.sample.sample(self.ids)
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 1)

    def test_stage_cap_stale_segments_and_evaluation_mode(self):
        self.segment()
        self.sample.forward_limit = 1.5
        self.sample.window.zero_()
        self.sample.sample(self.ids)  # segments collected at 1 m/s are discarded at the new stage
        self.assertEqual(self.sample.window[0].item(), 0)
        self.sample.forward_limit = 4
        self.segment()
        self.evaluate()
        self.assertEqual(self.sample.forward_limit, 4)
        frozen = Sampler(config(), 2000, "cpu", .5, .25, training=False)
        command, _ = frozen.sample(self.ids)
        for _ in range(50):
            frozen.observe(command, command[:, :3], self.zero, self.zero)
        self.assertEqual(frozen.window[0].item(), 0)
        self.assertEqual(frozen.forward_limit, 1)

    def test_state_roundtrip_and_legacy_checkpoint(self):
        self.sample.forward_limit = 2.5
        self.sample.window[:] = torch.tensor([30., 27., 1.])
        self.sample.check_elapsed = 7.5
        restored = sampler(2000)
        restored.load_state_dict(self.sample.state_dict())
        self.assertEqual(restored.forward_limit, 2.5)
        self.assertEqual(restored.check_elapsed, 7.5)
        torch.testing.assert_close(restored.window, self.sample.window)
        self.assertFalse(restored.active.any())
        restored.load_state_dict(None)
        self.assertEqual(restored.forward_limit, 1)
        self.assertEqual(restored.window.sum().item(), 0)

    def test_invalid_or_reconfigured_curriculum_checkpoint(self):
        state = self.sample.state_dict()
        for changes in ({"version": 2}, {"forward_limit": float("nan")}, {"window": [10, 11, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.sample.load_state_dict(dict(state, **changes))
        cfg = config(max_curriculum=3, forward_speed_bins=[[.12, 1], [1, 2], [2, 3]])
        restored = Sampler(cfg, 2000, "cpu", .02, .25)
        state.update(forward_limit=4, window=[30, 29, 0])
        restored.load_state_dict(state)
        self.assertEqual(restored.forward_limit, 3)
        self.assertEqual(restored.window.sum().item(), 0)


class CheckpointTests(unittest.TestCase):
    def test_real_runner_saves_restores_and_loads_old_checkpoint(self):
        stubs = {}
        for suffix, name, value in (("algorithms", "HIMPPO", object), ("env", "VecEnv", object), ("modules", "HIMActorCritic", object)):
            module = ModuleType("rsl_rl." + suffix)
            setattr(module, name, value)
            stubs[module.__name__] = module
        with patch.dict(sys.modules, stubs):
            runner_class = load_source("test_him_runner", "rsl_rl/rsl_rl/runners/him_on_policy_runner.py").HIMOnPolicyRunner
        command_sampler = sampler(10)
        command_sampler.forward_limit = 2.5
        actor = torch.nn.Linear(2, 1)
        actor.estimator = SimpleNamespace(optimizer=torch.optim.Adam(actor.parameters()))
        runner = runner_class.__new__(runner_class)
        runner.alg = SimpleNamespace(actor_critic=actor, optimizer=torch.optim.Adam(actor.parameters()))
        runner.current_learning_iteration = 1500
        runner.env = SimpleNamespace(
            get_training_state=lambda: {"primuus_commands": command_sampler.state_dict()},
            load_training_state=lambda state: command_sampler.load_state_dict(None if state is None else state["primuus_commands"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1500.pt"
            runner.save(path)
            expected = actor.weight.detach().clone()
            command_sampler.forward_limit = 1
            with torch.no_grad():
                actor.weight.zero_()
            runner.load(path)
            torch.testing.assert_close(actor.weight, expected)
            self.assertEqual(runner.current_learning_iteration, 1500)
            self.assertEqual(command_sampler.forward_limit, 2.5)
            old = torch.load(path)
            del old["env_state"]
            torch.save(old, path)
            runner.load(path)
            self.assertEqual(command_sampler.forward_limit, 1)


if __name__ == "__main__":
    unittest.main()
