from __future__ import annotations

from ..config import RealDeploymentConfig
from .executor import PolicyExecutor
from .onnx_policy import OnnxPolicy


def load_real_policy(
    config: RealDeploymentConfig,
) -> PolicyExecutor:
    """Create a validated policy executor from a real deployment config."""

    backend = OnnxPolicy(config.policy)
    return PolicyExecutor(
        backend,
        config.policy,
        max_inference_time_s=config.safety.max_policy_inference_time_s,
    )
