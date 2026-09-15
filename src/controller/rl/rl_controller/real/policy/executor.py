from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from ..config import RealPolicyConfig
from .base import (
    PolicyBackend,
    PolicyContractError,
    PolicyError,
    PolicyInferenceError,
    PolicyInferenceTimeoutError,
    require_float32_vector,
)


class PolicyExecutor:
    """Validate, time and clip every real-deployment policy call."""

    def __init__(
        self,
        backend: PolicyBackend,
        config: RealPolicyConfig,
        *,
        max_inference_time_s: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if backend.num_observations != config.num_observations:
            raise PolicyContractError(
                'Policy backend observation dimension does not match config: '
                f'{backend.num_observations} != {config.num_observations}'
            )
        if backend.num_actions != config.num_actions:
            raise PolicyContractError(
                'Policy backend action dimension does not match config: '
                f'{backend.num_actions} != {config.num_actions}'
            )
        if (
            not math.isfinite(max_inference_time_s)
            or max_inference_time_s <= 0.0
        ):
            raise ValueError('max_inference_time_s must be finite and positive')

        self.backend = backend
        self.config = config
        self.max_inference_time_s = float(max_inference_time_s)
        self._clock = clock
        self.last_inference_time_s: float | None = None

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        obs = require_float32_vector(
            observation,
            size=self.config.num_observations,
            name='Policy observation',
        )
        obs = np.clip(
            obs,
            -self.config.clip_observations,
            self.config.clip_observations,
        )
        obs = np.ascontiguousarray(obs, dtype=np.float32)

        start_time = self._read_clock()
        try:
            raw_action = self.backend.infer(obs)
        except PolicyError:
            raise
        except Exception as exc:
            raise PolicyInferenceError(
                f'{self.backend_name} policy inference failed'
            ) from exc
        end_time = self._read_clock()

        elapsed_s = end_time - start_time
        if elapsed_s < 0.0:
            raise PolicyInferenceError('Policy inference clock moved backwards')
        self.last_inference_time_s = elapsed_s
        if elapsed_s > self.max_inference_time_s:
            raise PolicyInferenceTimeoutError(
                f'Policy inference exceeded {self.max_inference_time_s:.6f}s '
                f'deadline: {elapsed_s:.6f}s'
            )

        action = require_float32_vector(
            raw_action,
            size=self.config.num_actions,
            name='Policy action',
        )
        action = np.clip(
            action,
            -self.config.clip_actions,
            self.config.clip_actions,
        )
        return np.array(action, dtype=np.float32, order='C', copy=True)

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not math.isfinite(timestamp):
            raise PolicyInferenceError(
                'Policy inference clock returned a non-finite timestamp'
            )
        return timestamp
