from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import (
    MockAcceptanceSuite,
    RealDeploymentConfig,
    RealRLRunner,
    RuntimeState,
    write_acceptance_report,
)


class RealMockAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        source = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.config = replace(
            source,
            runtime=replace(
                source.runtime,
                move_to_default_duration_s=0.01,
            ),
        )

    def test_runner_telemetry_records_cycles_state_age_and_outputs(self) -> None:
        runner = RealRLRunner.from_config(self.config)
        try:
            runner.start(ready_timeout_s=1.0)
            runner.enable_rl()
            self.assertTrue(
                runner.wait_for_state(RuntimeState.RL, timeout=1.0),
                runner.fault_reason,
            )
            time.sleep(0.03)

            telemetry = runner.telemetry()
            self.assertIs(telemetry.state, RuntimeState.RL)
            self.assertTrue(
                all(
                    duration is not None and duration >= 0.0
                    for duration in telemetry.last_cycle_duration_s.values()
                )
            )
            self.assertTrue(
                all(
                    telemetry.max_cycle_duration_s[name]
                    >= telemetry.last_cycle_duration_s[name]
                    for name in telemetry.last_cycle_duration_s
                )
            )
            self.assertIsNotNone(telemetry.feedback_age_s['motor'])
            self.assertIsNotNone(telemetry.feedback_age_s['imu'])
            self.assertEqual(telemetry.action.shape, (12,))
            self.assertEqual(telemetry.target_position.shape, (12,))
            self.assertFalse(telemetry.action.flags.writeable)
            json.dumps(telemetry.as_dict(), allow_nan=False)
        finally:
            runner.shutdown('telemetry test complete')

    def test_short_soak_and_fault_matrix_pass_with_clean_resources(self) -> None:
        suite = MockAcceptanceSuite(
            self.config,
            soak_duration_s=0.03,
            sample_interval_s=0.01,
            fault_timeout_s=1.0,
        )

        report = suite.run()

        self.assertTrue(report['passed'])
        self.assertEqual(
            [scenario['category'] for scenario in report['scenarios']],
            ['nominal', 'communication', 'imu', 'policy', 'target'],
        )
        for scenario in report['scenarios']:
            with self.subTest(scenario=scenario['name']):
                self.assertTrue(scenario['behavior_passed'])
                self.assertTrue(scenario['cleanup_passed'])
                self.assertEqual(scenario['final_state'], 'SHUTDOWN')
                self.assertEqual(scenario['resources']['live_threads'], [])
                self.assertFalse(scenario['resources']['motor_open'])
                self.assertFalse(scenario['resources']['imu_open'])
                self.assertTrue(scenario['resources']['motor_safe_stopped'])
                self.assertGreaterEqual(len(scenario['samples']), 3)
                terminal = scenario['samples'][-2]
                self.assertIn('last_cycle_duration_s', terminal)
                self.assertIn('feedback_age_s', terminal)
                self.assertEqual(len(terminal['action']), 12)

        with tempfile.TemporaryDirectory() as directory:
            path = write_acceptance_report(
                report,
                Path(directory) / 'acceptance.json',
            )
            loaded = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(loaded['passed'])
        self.assertEqual(len(loaded['scenarios']), 5)

    def test_acceptance_import_is_isolated_from_simulation_and_hardware(self) -> None:
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(RL_PACKAGE_ROOT)
        script = (
            'import sys\n'
            'import rl_controller.real.acceptance\n'
            "for name in ('mujoco', 'onnxruntime', 'serial', "
            "'unitree_actuator_sdk', 'quadruped_rl'):\n"
            '    assert name not in sys.modules, name\n'
        )
        result = subprocess.run(
            [sys.executable, '-c', script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_interrupted_soak_still_shuts_down_runner(self) -> None:
        created_runners: list[RealRLRunner] = []
        original_factory = RealRLRunner.from_config

        def capture_runner(config):
            runner = original_factory(config)
            created_runners.append(runner)
            return runner

        suite = MockAcceptanceSuite(
            self.config,
            soak_duration_s=1.0,
            sample_interval_s=0.01,
            fault_timeout_s=1.0,
        )
        with patch.object(
            RealRLRunner,
            'from_config',
            side_effect=capture_runner,
        ), patch(
            'rl_controller.real.acceptance.time.sleep',
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                suite.run()

        self.assertEqual(len(created_runners), 1)
        runner = created_runners[0]
        self.assertIs(runner.state, RuntimeState.SHUTDOWN)
        self.assertEqual(runner.live_thread_names, ())
        self.assertFalse(runner.motor_backend.is_open)
        self.assertFalse(runner.imu_backend.is_open)
        self.assertTrue(runner.motor_backend.safe_stopped)


if __name__ == '__main__':
    unittest.main()
