from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import RealDeploymentConfig
from .base import PolicyLoadError
from .executor import PolicyExecutor
from .mock_policy import MockPolicy
from .onnx_policy import OnnxPolicy


def load_real_policy(
    config: RealDeploymentConfig,
    *,
    mock_action: Any | None = None,
    ort_module: Any | None = None,
    clock: Callable[[], float] | None = None,
) -> PolicyExecutor:
    """Create a validated policy executor from a real deployment config."""

    if config.policy.backend == 'mock':
        backend = MockPolicy(config.policy, action=mock_action)
    elif config.policy.backend == 'onnx':
        backend = OnnxPolicy(config.policy, ort_module=ort_module)
    else:
        raise PolicyLoadError(
            f'Unsupported real policy backend: {config.policy.backend}'
        )

    executor_kwargs: dict[str, Any] = {
        'max_inference_time_s': (
            config.safety.max_policy_inference_time_s
        ),
    }
    if clock is not None:
        executor_kwargs['clock'] = clock
    return PolicyExecutor(backend, config.policy, **executor_kwargs)
