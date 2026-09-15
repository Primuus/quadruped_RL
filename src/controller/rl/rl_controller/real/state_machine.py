from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Callable

import numpy as np


class RuntimeState(str, Enum):
    INIT = 'INIT'
    READY = 'READY'
    MOVE_TO_DEFAULT = 'MOVE_TO_DEFAULT'
    RL = 'RL'
    FAULT = 'FAULT'
    SHUTDOWN = 'SHUTDOWN'


class RuntimeTransitionError(RuntimeError):
    """Raised when a runtime state transition is not permitted."""


@dataclass(frozen=True, slots=True)
class RuntimeTransition:
    previous: RuntimeState
    current: RuntimeState
    reason: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class RuntimeFault:
    reason: str
    timestamp: float


class RealRuntimeStateMachine:
    """Thread-safe, latched lifecycle for the real robot runner."""

    _ALLOWED_TRANSITIONS = {
        RuntimeState.INIT: {
            RuntimeState.READY,
            RuntimeState.FAULT,
            RuntimeState.SHUTDOWN,
        },
        RuntimeState.READY: {
            RuntimeState.MOVE_TO_DEFAULT,
            RuntimeState.FAULT,
            RuntimeState.SHUTDOWN,
        },
        RuntimeState.MOVE_TO_DEFAULT: {
            RuntimeState.RL,
            RuntimeState.FAULT,
            RuntimeState.SHUTDOWN,
        },
        RuntimeState.RL: {
            RuntimeState.FAULT,
            RuntimeState.SHUTDOWN,
        },
        RuntimeState.FAULT: {RuntimeState.SHUTDOWN},
        RuntimeState.SHUTDOWN: set(),
    }

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._state = RuntimeState.INIT
        self._transitions: list[RuntimeTransition] = []
        self._faults: list[RuntimeFault] = []

    @property
    def state(self) -> RuntimeState:
        with self._condition:
            return self._state

    @property
    def transitions(self) -> tuple[RuntimeTransition, ...]:
        with self._condition:
            return tuple(self._transitions)

    @property
    def faults(self) -> tuple[RuntimeFault, ...]:
        with self._condition:
            return tuple(self._faults)

    @property
    def fault_reason(self) -> str | None:
        with self._condition:
            return self._faults[0].reason if self._faults else None

    @property
    def latest_fault_reason(self) -> str | None:
        with self._condition:
            return self._faults[-1].reason if self._faults else None

    def mark_ready(self, reason: str = 'initial hardware state is ready') -> None:
        self._transition(RuntimeState.READY, reason)

    def begin_move_to_default(
        self,
        reason: str = 'RL control requested',
    ) -> None:
        self._transition(RuntimeState.MOVE_TO_DEFAULT, reason)

    def enter_rl(self, reason: str = 'observation history is ready') -> None:
        self._transition(RuntimeState.RL, reason)

    def fault(self, reason: str) -> None:
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError('fault reason must not be empty')

        with self._condition:
            if self._state is RuntimeState.SHUTDOWN:
                raise RuntimeTransitionError(
                    'Cannot report a fault after SHUTDOWN'
                )
            timestamp = self._read_clock()
            self._faults.append(
                RuntimeFault(
                    reason=normalized_reason,
                    timestamp=timestamp,
                )
            )
            if self._state is not RuntimeState.FAULT:
                self._transition_locked(
                    RuntimeState.FAULT,
                    normalized_reason,
                    timestamp,
                )
            self._condition.notify_all()

    def shutdown(self, reason: str = 'shutdown requested') -> None:
        with self._condition:
            if self._state is RuntimeState.SHUTDOWN:
                return
            self._transition_locked(
                RuntimeState.SHUTDOWN,
                str(reason).strip() or 'shutdown requested',
                self._read_clock(),
            )
            self._condition.notify_all()

    def wait_for(
        self,
        states: RuntimeState | Iterable[RuntimeState],
        timeout: float | None = None,
    ) -> bool:
        if isinstance(states, RuntimeState):
            expected_states = frozenset({states})
        else:
            expected_states = frozenset(states)
        if not expected_states or not all(
            isinstance(state, RuntimeState) for state in expected_states
        ):
            raise ValueError('states must contain at least one RuntimeState')
        if timeout is not None:
            timeout = float(timeout)
            if not np.isfinite(timeout) or timeout < 0.0:
                raise ValueError('timeout must be finite and non-negative')

        with self._condition:
            return self._condition.wait_for(
                lambda: self._state in expected_states,
                timeout=timeout,
            )

    def _transition(self, target: RuntimeState, reason: str) -> None:
        with self._condition:
            self._transition_locked(target, reason, self._read_clock())
            self._condition.notify_all()

    def _transition_locked(
        self,
        target: RuntimeState,
        reason: str,
        timestamp: float,
    ) -> None:
        previous = self._state
        if target not in self._ALLOWED_TRANSITIONS[previous]:
            raise RuntimeTransitionError(
                f'Invalid runtime transition: {previous.value} -> '
                f'{target.value}'
            )
        self._state = target
        self._transitions.append(
            RuntimeTransition(
                previous=previous,
                current=target,
                reason=str(reason),
                timestamp=timestamp,
            )
        )

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise RuntimeTransitionError(
                'runtime state clock must return finite, non-negative time'
            )
        return timestamp
