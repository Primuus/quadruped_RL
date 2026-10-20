from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Sequence

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_controller.input import KeyboardCommandController, KeyboardInput
from rl_controller.real import (
    FaultCode,
    FaultRecord,
    PolicyError,
    RealConfigError,
    RealDeploymentConfig,
    RealRLRunner,
    RealRunnerError,
    RealRunnerShutdownError,
    RuntimeState,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run an ONNX policy on the configured quadruped hardware.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Completed real-deployment YAML configuration.',
    )
    parser.add_argument(
        '--duration',
        type=_nonnegative_float,
        default=0.0,
        help='Run time in seconds; 0 runs until Ctrl+C, operator stop, or FAULT.',
    )
    parser.add_argument(
        '--no-keyboard',
        action='store_true',
        help='Do not open Linux evdev keyboard devices.',
    )
    parser.add_argument(
        '--keyboard-device',
        type=Path,
        help='Explicit /dev/input event device; default discovers keyboards.',
    )
    parser.add_argument(
        '--auto-enable',
        action='store_true',
        help='Enter MOVE_TO_DEFAULT without waiting for the keyboard F key.',
    )
    parser.add_argument(
        '--command',
        type=float,
        nargs=3,
        metavar=('VX', 'VY', 'YAW_RATE'),
        default=None,
        help='Fixed command used with --no-keyboard.',
    )
    parser.add_argument(
        '--allow-real-hardware',
        action='store_true',
        help='Required acknowledgement before opening motors and the IMU.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.keyboard_device is not None and args.no_keyboard:
        print(
            'Real RL startup failed: --keyboard-device cannot be used with '
            '--no-keyboard',
            file=sys.stderr,
        )
        return 2
    if args.command is not None and not args.no_keyboard:
        print(
            'Real RL startup failed: --command requires --no-keyboard',
            file=sys.stderr,
        )
        return 2

    try:
        _require_hardware_authorization(args.allow_real_hardware)
        config = RealDeploymentConfig.from_yaml(args.config)
        runner = RealRLRunner.from_config(config)
    except Exception as exc:
        code = _startup_error_code(exc)
        print(
            f'Real RL startup failed [{code.value}]: {exc}',
            file=sys.stderr,
        )
        return 2

    keyboard: KeyboardInput | None = None
    keyboard_controller: KeyboardCommandController | None = None
    return_code = 0
    try:
        if not args.no_keyboard:
            keyboard = KeyboardInput(
                args.keyboard_device,
                tracked_keys=config.keyboard.bindings.values(),
            )
            keyboard_controller = KeyboardCommandController(
                command_limits=config.keyboard.command_limits,
                max_translation_speed=config.keyboard.max_translation_speed,
                bindings=config.keyboard.bindings,
                speed_levels=config.keyboard.speed_levels,
                default_speed_level=config.keyboard.default_speed_level,
                motion_limits=config.keyboard.motion_limits,
                backward_speed_levels=config.keyboard.backward_speed_levels,
            )

        runner.start()
        if args.auto_enable:
            if keyboard is not None and keyboard_controller is not None:
                enable_effect = keyboard_controller.apply_key(
                    config.keyboard.bindings['enable']
                )
                if enable_effect.exclusive_input:
                    keyboard.set_exclusive(True)
            runner.enable_rl()
        fixed_command = (
            (0.0, 0.0, 0.0)
            if args.command is None
            else args.command
        )
        runner.set_velocity_command(fixed_command)

        print(
            f'Real RL runner started: robot={config.robot.name}, '
            f'policy={config.policy.backend}, state={runner.state.value}'
        )
        started_at = time.monotonic()
        while (
            runner.state not in {RuntimeState.FAULT, RuntimeState.SHUTDOWN}
            and not runner.stop_requested
        ):
            if args.duration > 0.0 and time.monotonic() - started_at >= args.duration:
                runner.request_stop('configured duration elapsed')
                break
            if keyboard is not None and keyboard_controller is not None:
                _update_keyboard(
                    keyboard,
                    keyboard_controller,
                    runner,
                    config,
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('Shutdown requested by Ctrl+C')
        runner.request_stop('Ctrl+C')
    except Exception as exc:
        return_code = 2
        if runner.state not in {RuntimeState.FAULT, RuntimeState.SHUTDOWN}:
            runner.report_exception('main-loop', exc)
    finally:
        if keyboard is not None:
            try:
                keyboard.set_exclusive(False)
            except Exception as exc:
                return_code = 2
                if runner.state is not RuntimeState.SHUTDOWN:
                    runner.report_fault(
                        FaultCode.INPUT,
                        'keyboard-release',
                        str(exc),
                    )
            try:
                keyboard.close()
            except Exception as exc:
                return_code = 2
                if runner.state is not RuntimeState.SHUTDOWN:
                    runner.report_fault(
                        FaultCode.INPUT,
                        'keyboard-close',
                        str(exc),
                    )

        try:
            runner.shutdown(runner.stop_reason or 'shutdown requested')
        except RealRunnerShutdownError as exc:
            return_code = 3
            print(f'Real RL runner shutdown failed: {exc}', file=sys.stderr)

        if runner.fault_record is not None:
            return_code = return_code or 2
            _print_fault_record(runner.fault_record)
            for secondary in runner.secondary_faults:
                print(
                    f'Secondary failure [{secondary.code.value}] '
                    f'{secondary.source}: {secondary.message}',
                    file=sys.stderr,
                )
        else:
            print(
                'Real RL runner stopped safely: '
                f'{runner.stop_reason or "shutdown requested"}'
            )

    return return_code


def _update_keyboard(
    keyboard: KeyboardInput,
    controller: KeyboardCommandController,
    runner: RealRLRunner,
    config: RealDeploymentConfig,
) -> None:
    keyboard_error = keyboard.last_error
    if keyboard_error is not None:
        runner.report_fault(
            FaultCode.INPUT,
            'keyboard',
            f'keyboard input failed: {keyboard_error}',
        )
        return

    for key_name in keyboard.drain_key_events():
        effect = controller.apply_key(key_name)
        if effect.exclusive_input is not None:
            keyboard.set_exclusive(effect.exclusive_input)
        if effect.log_message:
            print(effect.log_message)
        if effect.print_status:
            print(controller.describe(keyboard.latest_snapshot().pressed_keys))
        if effect.reset_policy_state:
            runner.enable_rl()
        if key_name == config.keyboard.bindings['disable']:
            runner.request_stop('keyboard disable requested')
            return

    command = controller.update_pressed_keys(
        keyboard.latest_snapshot().pressed_keys
    )
    runner.set_velocity_command(command)


def _print_fault_record(record: FaultRecord) -> None:
    print(
        f'FAULT [{record.code.value}] {record.source}: {record.message}',
        file=sys.stderr,
    )
    print(
        '  '
        f'state={record.runtime_state.value} '
        f'timestamp={record.timestamp:.6f} '
        f'state_age_s={record.state_age_s} '
        f'cycle_duration_s={record.cycle_duration_s}',
        file=sys.stderr,
    )
    print(f'  command={record.command}', file=sys.stderr)
    print(f'  raw_action={record.raw_action}', file=sys.stderr)
    print(f'  target_position={record.target_position}', file=sys.stderr)


def _require_hardware_authorization(allowed: bool) -> None:
    if not allowed:
        raise RealRunnerError(
            'Real deployment requires the explicit --allow-real-hardware flag'
        )


def _startup_error_code(exception: Exception) -> FaultCode:
    if isinstance(
        exception,
        (FileNotFoundError, RealConfigError, RealRunnerError),
    ):
        return FaultCode.CONFIGURATION
    if isinstance(exception, PolicyError):
        return FaultCode.POLICY
    return FaultCode.HARDWARE


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            'value must be finite and non-negative'
        )
    return number


if __name__ == '__main__':
    raise SystemExit(main())
