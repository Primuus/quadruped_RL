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
    RealDeploymentConfig,
    RealRLRunner,
    RealRunnerError,
    RealRunnerShutdownError,
    RuntimeState,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the hardware-independent real quadruped RL chain.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help='Real deployment YAML; defaults to the offline Mock config.',
    )
    parser.add_argument(
        '--duration',
        type=_nonnegative_float,
        default=0.0,
        help='Run time in seconds; 0 runs until Ctrl+C or a fault.',
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
        default=(0.0, 0.0, 0.0),
        help='Fixed command used with --no-keyboard.',
    )
    parser.add_argument(
        '--allow-real-hardware',
        action='store_true',
        help='Explicitly allow non-Mock motor or IMU backends to open.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = RealDeploymentConfig.from_yaml(args.config)
    _require_hardware_authorization(config, args.allow_real_hardware)

    runner = RealRLRunner.from_config(config)
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
                bindings=config.keyboard.bindings,
                speed_levels=config.keyboard.speed_levels,
                default_speed_level=config.keyboard.default_speed_level,
            )

        runner.start()
        runner.set_velocity_command(args.command)
        if args.auto_enable:
            if keyboard is not None and keyboard_controller is not None:
                enable_effect = keyboard_controller.apply_key(
                    config.keyboard.bindings['enable']
                )
                if enable_effect.exclusive_input:
                    keyboard.set_exclusive(True)
            runner.enable_rl()

        print(
            f'Real RL runner started: robot={config.robot.name}, '
            f'policy={config.policy.backend}, state={runner.state.value}'
        )
        started_at = time.monotonic()
        while runner.state not in {RuntimeState.FAULT, RuntimeState.SHUTDOWN}:
            if args.duration > 0.0 and time.monotonic() - started_at >= args.duration:
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
    except Exception as exc:
        return_code = 2
        if runner.state not in {RuntimeState.FAULT, RuntimeState.SHUTDOWN}:
            try:
                runner.report_fault(f'main loop failed: {exc}')
            except Exception:
                pass
        print(f'Real RL runner failed: {exc}')
    finally:
        if keyboard is not None:
            try:
                keyboard.set_exclusive(False)
            except Exception:
                pass
            keyboard.close()

        fault_reason = runner.fault_reason
        try:
            runner.shutdown()
        except RealRunnerShutdownError as exc:
            return_code = 3
            print(f'Real RL runner shutdown failed: {exc}')
        if fault_reason is not None and return_code == 0:
            return_code = 2
            print(f'Real RL runner entered FAULT: {fault_reason}')

    return return_code


def _update_keyboard(
    keyboard: KeyboardInput,
    controller: KeyboardCommandController,
    runner: RealRLRunner,
    config: RealDeploymentConfig,
) -> None:
    keyboard_error = keyboard.last_error
    if keyboard_error is not None:
        raise RealRunnerError(f'keyboard input failed: {keyboard_error}')

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
            runner.report_fault('keyboard control disabled')
            return

    command = controller.update_pressed_keys(
        keyboard.latest_snapshot().pressed_keys
    )
    runner.set_velocity_command(command)


def _require_hardware_authorization(
    config: RealDeploymentConfig,
    allowed: bool,
) -> None:
    uses_real_hardware = (
        config.motors.backend != 'mock' or config.imu.backend != 'mock'
    )
    if uses_real_hardware and not allowed:
        raise RealRunnerError(
            'Real motor or IMU backend requires --allow-real-hardware'
        )


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError(
            'value must be finite and non-negative'
        )
    return number


if __name__ == '__main__':
    raise SystemExit(main())
