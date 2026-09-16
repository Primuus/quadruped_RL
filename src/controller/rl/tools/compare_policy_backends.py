from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.core.config import RobotRLConfig, default_config_path
from rl_controller.core.policy import OnnxRuntimePolicy, TorchScriptPolicy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compare TorchScript and ONNX outputs for identical policy observations.'
    )
    parser.add_argument('--config', type=Path, default=None, help='Path to the robot YAML config.')
    parser.add_argument('--torchscript-path', type=Path, default=None, help='TorchScript policy path.')
    parser.add_argument('--onnx-path', type=Path, default=None, help='ONNX policy path.')
    parser.add_argument(
        '--observations',
        type=Path,
        default=None,
        help='Optional .npy observations with shape (num_obs,) or (samples, num_obs).',
    )
    parser.add_argument('--samples', type=int, default=128, help='Random samples when --observations is omitted.')
    parser.add_argument('--seed', type=int, default=1, help='Random observation seed.')
    parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance.')
    parser.add_argument('--rtol', type=float, default=1e-4, help='Relative tolerance.')
    parser.add_argument('--torch-device', type=str, default='cpu', help='TorchScript device.')
    parser.add_argument('--onnx-device', type=str, default='cpu', help='ONNX Runtime device.')
    return parser


def load_observations(args: argparse.Namespace, num_obs: int) -> np.ndarray:
    if args.observations is not None:
        observations = np.load(Path(args.observations).expanduser().resolve())
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations[np.newaxis, :]
    else:
        if args.samples <= 0:
            raise ValueError('--samples must be positive')
        rng = np.random.default_rng(args.seed)
        observations = rng.standard_normal((args.samples, num_obs)).astype(np.float32)

    if observations.ndim != 2 or observations.shape[1] != num_obs:
        raise ValueError(
            f'Observations must have shape (samples, {num_obs}), got {observations.shape}'
        )
    return observations


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = RobotRLConfig.from_yaml(args.config or default_config_path())

    torchscript_path = args.torchscript_path or cfg.policy_path
    onnx_path = args.onnx_path or Path(torchscript_path).with_suffix('.onnx')
    observations = load_observations(args, cfg.num_obs)

    torchscript_policy = TorchScriptPolicy(torchscript_path, device=args.torch_device)
    onnx_policy = OnnxRuntimePolicy(onnx_path, device=args.onnx_device)

    torchscript_actions = np.stack(
        [torchscript_policy.infer(observation) for observation in observations]
    )
    onnx_actions = np.stack([onnx_policy.infer(observation) for observation in observations])
    if torchscript_actions.shape != onnx_actions.shape:
        raise ValueError(
            f'Policy output shapes differ: TorchScript {torchscript_actions.shape}, '
            f'ONNX {onnx_actions.shape}'
        )

    absolute_error = np.abs(torchscript_actions - onnx_actions)
    max_error_index = np.unravel_index(np.argmax(absolute_error), absolute_error.shape)
    is_close = np.allclose(
        torchscript_actions,
        onnx_actions,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(f'samples: {observations.shape[0]}')
    print(f'action_shape: {torchscript_actions.shape}')
    print(f'mean_abs_error: {float(np.mean(absolute_error)):.9e}')
    print(f'max_abs_error: {float(np.max(absolute_error)):.9e}')
    print(f'max_error_index: {max_error_index}')
    print(f'atol: {args.atol:.3e}')
    print(f'rtol: {args.rtol:.3e}')
    print(f'allclose: {is_close}')

    if not is_close:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
