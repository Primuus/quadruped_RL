from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


class PolicyError(RuntimeError):
    """Base exception for real-deployment policy failures."""


class PolicyLoadError(PolicyError):
    """Raised when a configured policy backend cannot be loaded."""


class PolicyContractError(PolicyError):
    """Raised when policy data violates the configured tensor contract."""


class PolicyInferenceError(PolicyError):
    """Raised when a loaded policy cannot produce a valid action."""


class PolicyInferenceTimeoutError(PolicyInferenceError):
    """Raised when a policy result arrives after its configured deadline."""


@runtime_checkable
class PolicyBackend(Protocol):
    """Minimal backend contract consumed by the real policy executor."""

    backend_name: str
    num_observations: int
    num_actions: int

    def infer(self, observation: np.ndarray) -> np.ndarray:
        ...


def require_float32_vector(
    value: Any,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    """Return an isolated contiguous vector after strict policy validation."""

    array = np.asarray(value)
    if array.shape != (size,):
        raise PolicyContractError(
            f'{name} must have shape ({size},), got {array.shape}'
        )
    if array.dtype != np.dtype(np.float32):
        raise PolicyContractError(
            f'{name} must have dtype float32, got {array.dtype}'
        )
    if not np.all(np.isfinite(array)):
        raise PolicyContractError(f'{name} must contain only finite values')
    return np.array(array, dtype=np.float32, order='C', copy=True)
