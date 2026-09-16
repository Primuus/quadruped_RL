from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RealPolicyConfig
from .base import (
    PolicyContractError,
    PolicyInferenceError,
    PolicyLoadError,
)


class OnnxPolicy:
    """Strict ONNX Runtime backend for a real-deployment actor."""

    backend_name = 'onnx'

    def __init__(
        self,
        config: RealPolicyConfig,
    ) -> None:
        if config.backend != 'onnx':
            raise PolicyLoadError('OnnxPolicy requires policy.backend=onnx')
        if config.model_path is None:
            raise PolicyLoadError('ONNX policy.model_path is required')

        self.model_path = Path(config.model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise PolicyLoadError(
                f'ONNX policy file does not exist: {self.model_path}'
            )

        self.num_observations = config.num_observations
        self.num_actions = config.num_actions
        ort = self._import_runtime()

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider'],
            )
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
        except Exception as exc:
            raise PolicyLoadError(
                f'Failed to load ONNX policy: {self.model_path}'
            ) from exc

        if len(inputs) != 1:
            raise PolicyContractError(
                'ONNX policy must have exactly one input, '
                f'got {len(inputs)}'
            )
        if len(outputs) != 1:
            raise PolicyContractError(
                'ONNX policy must have exactly one output, '
                f'got {len(outputs)}'
            )

        self._validate_tensor_metadata(
            inputs[0],
            tensor_name='input',
            feature_size=self.num_observations,
        )
        self._validate_tensor_metadata(
            outputs[0],
            tensor_name='output',
            feature_size=self.num_actions,
        )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @staticmethod
    def _import_runtime() -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PolicyLoadError(
                'ONNX inference requires onnxruntime>=1.16,<2'
            ) from exc
        return ort

    @staticmethod
    def _validate_tensor_metadata(
        metadata: Any,
        *,
        tensor_name: str,
        feature_size: int,
    ) -> None:
        name = getattr(metadata, 'name', None)
        if not isinstance(name, str) or not name:
            raise PolicyContractError(
                f'ONNX policy {tensor_name} must have a non-empty name'
            )

        data_type = getattr(metadata, 'type', None)
        if data_type != 'tensor(float)':
            raise PolicyContractError(
                f'ONNX policy {tensor_name} must use tensor(float), '
                f'got {data_type!r}'
            )

        shape = getattr(metadata, 'shape', None)
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise PolicyContractError(
                f'ONNX policy {tensor_name} must have rank 2, got {shape!r}'
            )

        batch_size = shape[0]
        dynamic_batch = batch_size is None or isinstance(batch_size, str)
        fixed_batch_one = (
            isinstance(batch_size, Integral)
            and not isinstance(batch_size, bool)
            and int(batch_size) == 1
        )
        if not dynamic_batch and not fixed_batch_one:
            raise PolicyContractError(
                f'ONNX policy {tensor_name} batch dimension must be 1 or '
                f'dynamic, got {batch_size!r}'
            )

        actual_feature_size = shape[1]
        if (
            isinstance(actual_feature_size, bool)
            or not isinstance(actual_feature_size, Integral)
            or int(actual_feature_size) != feature_size
        ):
            raise PolicyContractError(
                f'ONNX policy {tensor_name} feature dimension must be '
                f'{feature_size}, got {actual_feature_size!r}'
            )

    def infer(self, observation: np.ndarray) -> np.ndarray:
        model_input = observation[np.newaxis, :]

        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: model_input},
            )
        except Exception as exc:
            raise PolicyInferenceError('ONNX policy inference failed') from exc

        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise PolicyContractError(
                'ONNX Runtime must return exactly one policy output'
            )

        output = np.asarray(outputs[0])
        expected_shape = (1, self.num_actions)
        if output.shape != expected_shape:
            raise PolicyContractError(
                f'ONNX policy output must have shape {expected_shape}, '
                f'got {output.shape}'
            )
        if output.dtype != np.dtype(np.float32):
            raise PolicyContractError(
                f'ONNX policy output must have dtype float32, '
                f'got {output.dtype}'
            )
        return np.array(output[0], dtype=np.float32, order='C', copy=True)
