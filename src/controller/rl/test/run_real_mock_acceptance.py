from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import sys


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(RL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import RealDeploymentConfig
from rl_controller.real.acceptance import (
    MockAcceptanceSuite,
    write_acceptance_report,
)


DEFAULT_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the real-RL offline Mock soak and fault matrix.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help='Mock real-deployment YAML config.',
    )
    parser.add_argument(
        '--soak-duration',
        type=_positive_float,
        default=60.0,
        help='Nominal RL soak duration in seconds. Default: 60.',
    )
    parser.add_argument(
        '--sample-interval',
        type=_positive_float,
        default=None,
        help='Telemetry interval in seconds. Default: runtime log frequency.',
    )
    parser.add_argument(
        '--fault-timeout',
        type=_positive_float,
        default=2.0,
        help='Maximum wait for each injected fault. Default: 2.',
    )
    parser.add_argument(
        '--move-to-default-duration',
        type=_positive_float,
        default=None,
        help='Override the Mock default-pose transition duration.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='JSON report path. Default: src/controller/rl/log/<timestamp>.json.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = RealDeploymentConfig.from_yaml(args.config)
    if args.move_to_default_duration is not None:
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                move_to_default_duration_s=args.move_to_default_duration,
            ),
        )

    suite = MockAcceptanceSuite(
        config,
        soak_duration_s=args.soak_duration,
        sample_interval_s=args.sample_interval,
        fault_timeout_s=args.fault_timeout,
    )
    report = suite.run()
    output_path = args.output or _default_output_path()
    written_path = write_acceptance_report(report, output_path)

    for scenario in report['scenarios']:
        result = 'PASS' if scenario['passed'] else 'FAIL'
        reason = scenario['fault_reason'] or scenario['execution_error'] or '-'
        print(f"[{result}] {scenario['name']}: {reason}")
    print(f"Report: {written_path}")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report['passed'] else 1


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return RL_PACKAGE_ROOT / 'log' / f'real_mock_acceptance_{timestamp}.json'


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError('value must be finite and positive')
    return number


if __name__ == '__main__':
    raise SystemExit(main())
