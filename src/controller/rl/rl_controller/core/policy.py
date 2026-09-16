from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Policy(Protocol):
    backend_name: str

    def infer(self, observation: np.ndarray) -> np.ndarray:
        ...


def _resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Policy file not found: {path}')
    return path


def _prepare_observation(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 1:
        raise ValueError(f'Policy observation must be one-dimensional, got shape {obs.shape}')
    if not np.all(np.isfinite(obs)):
        raise ValueError('Policy observation must contain only finite values')
    return np.ascontiguousarray(obs)


def _validate_action(action: np.ndarray, backend_name: str) -> np.ndarray:
    array = np.asarray(action)
    if array.ndim != 1:
        raise ValueError(
            f'{backend_name} policy action must be one-dimensional, got shape {array.shape}'
        )
    if array.dtype != np.dtype(np.float32):
        raise ValueError(
            f'{backend_name} policy action must have dtype float32, got {array.dtype}'
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{backend_name} policy action must contain only finite values')
    return np.array(array, dtype=np.float32, order='C', copy=True)


class TorchScriptPolicy:
    backend_name = 'torchscript'

    def __init__(self, model_path: str | Path, device: str = 'cpu') -> None:
        self.model_path = _resolve_model_path(model_path)
        self.device = torch.device(device)
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

    def infer(self, observation: np.ndarray) -> np.ndarray:
        obs = _prepare_observation(observation)
        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(obs_tensor)

        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError(
                    'TorchScript policy must return one Tensor or a one-element tuple/list'
                )
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                'TorchScript policy output must be a Tensor, got '
                f'{type(output).__name__}'
            )
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError(
                'TorchScript policy output must have shape [1, num_actions], '
                f'got {tuple(output.shape)}'
            )
        if output.dtype != torch.float32:
            raise ValueError(
                f'TorchScript policy output must have dtype float32, got {output.dtype}'
            )

        action = output.detach().to('cpu').numpy()[0]
        return _validate_action(action, self.backend_name)


class OnnxRuntimePolicy:
    backend_name = 'onnxruntime'

    def __init__(self, model_path: str | Path, device: str = 'cpu') -> None:
        self.model_path = _resolve_model_path(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                'ONNX inference requires onnxruntime. Install the deployment package or run: '
                'pip install "onnxruntime>=1.16,<2"'
            ) from exc

        providers = self._providers_for_device(ort, device)
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f'ONNX policy must have exactly one input, got {len(inputs)}')
        if len(outputs) < 1:
            raise ValueError('ONNX policy must have at least one output')

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @staticmethod
    def _providers_for_device(ort, device: str):
        normalized = str(device).strip().lower()
        if normalized == 'cpu':
            return ['CPUExecutionProvider']

        if normalized == 'cuda' or normalized.startswith('cuda:'):
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available:
                raise RuntimeError(
                    'CUDAExecutionProvider is unavailable. Install a matching onnxruntime-gpu package '
                    'or use --device cpu.'
                )
            device_id = 0
            if ':' in normalized:
                device_id = int(normalized.split(':', maxsplit=1)[1])
            return [
                ('CUDAExecutionProvider', {'device_id': device_id}),
                'CPUExecutionProvider',
            ]

        raise ValueError(f'Unsupported ONNX Runtime device: {device}')

    def infer(self, observation: np.ndarray) -> np.ndarray:
        obs = _prepare_observation(observation)[np.newaxis, :]
        output = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = np.asarray(output)
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(f'ONNX policy output must have batch size 1, got shape {action.shape}')
        return _validate_action(action[0], self.backend_name)


def resolve_policy_backend(model_path: str | Path, backend: str = 'auto') -> str:
    normalized = backend.strip().lower()
    aliases = {
        'jit': 'torchscript',
        'pt': 'torchscript',
        'ort': 'onnx',
        'onnxruntime': 'onnx',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == 'auto':
        suffix = Path(model_path).suffix.lower()
        if suffix == '.onnx':
            return 'onnx'
        if suffix in {'.pt', '.jit', '.ts'}:
            return 'torchscript'
        raise ValueError(
            f'Cannot infer policy backend from extension {suffix!r}; '
            'use --policy-backend torchscript or --policy-backend onnx.'
        )
    if normalized not in {'torchscript', 'onnx'}:
        raise ValueError(f'Unsupported policy backend: {backend}')
    return normalized


def load_policy(model_path: str | Path, backend: str = 'auto', device: str = 'cpu') -> Policy:
    resolved_backend = resolve_policy_backend(model_path, backend)
    if resolved_backend == 'onnx':
        return OnnxRuntimePolicy(model_path, device=device)
    return TorchScriptPolicy(model_path, device=device)
