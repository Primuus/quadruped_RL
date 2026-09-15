from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import unittest


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.deploy_real_rl import (
    _require_hardware_authorization,
    build_argument_parser,
    main,
)
from rl_controller.real import RealDeploymentConfig, RealRunnerError


class RealEntrypointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)

    def test_parser_defaults_to_mock_config(self) -> None:
        args = build_argument_parser().parse_args([])
        self.assertEqual(args.config, MOCK_CONFIG_PATH)
        self.assertFalse(args.no_keyboard)
        self.assertFalse(args.auto_enable)
        self.assertFalse(args.allow_real_hardware)

    def test_short_headless_mock_run_exits_cleanly(self) -> None:
        return_code = main(
            [
                '--config',
                str(MOCK_CONFIG_PATH),
                '--no-keyboard',
                '--auto-enable',
                '--duration',
                '0.05',
            ]
        )
        self.assertEqual(return_code, 0)

    def test_real_hardware_requires_explicit_authorization(self) -> None:
        real_imu_config = replace(
            self.config,
            imu=replace(self.config.imu, backend='yesense_yis130'),
        )
        with self.assertRaisesRegex(
            RealRunnerError,
            '--allow-real-hardware',
        ):
            _require_hardware_authorization(real_imu_config, False)
        _require_hardware_authorization(real_imu_config, True)

    def test_import_does_not_load_mujoco_or_optional_backends(self) -> None:
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(RL_PACKAGE_ROOT)
        script = (
            'import sys\n'
            'import rl_controller.deploy_real_rl\n'
            "for name in ('mujoco', 'onnxruntime', 'serial', "
            "'unitree_actuator_sdk'):\n"
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


if __name__ == '__main__':
    unittest.main()
