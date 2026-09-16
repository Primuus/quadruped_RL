from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable

import numpy as np


class RuntimeState(str, Enum):
    INIT = 'INIT'
    READY = 'READY'
    MOVE_TO_DEFAULT = 'MOVE_TO_DEFAULT'
    RL = 'RL'
    FAULT = 'FAULT'
    SHUTDOWN = 'SHUTDOWN'


class FaultCode(str, Enum):
    CONFIGURATION = 'CONFIGURATION'
    HARDWARE = 'HARDWARE'
    POLICY = 'POLICY'
    TIMING = 'TIMING'
    SAFETY = 'SAFETY'
    INPUT = 'INPUT'
    INTERNAL = 'INTERNAL'
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
class FaultRecord:
    code: FaultCode
    source: str
    message: str
    timestamp: float
    runtime_state: RuntimeState
    state_age_s: float | None = None
    command: tuple[float, ...] = ()
    raw_action: tuple[float, ...] = ()
    target_position: tuple[float, ...] = ()
    cycle_duration_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, FaultCode):
            raise TypeError('code must be a FaultCode')
        if not isinstance(self.runtime_state, RuntimeState):
            raise TypeError('runtime_state must be a RuntimeState')
        if not str(self.source).strip():
            raise ValueError('fault source must not be empty')
        if not str(self.message).strip():
            raise ValueError('fault message must not be empty')
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError('fault timestamp must be finite and non-negative')
        for field_name in ('state_age_s', 'cycle_duration_s'):
            value = getattr(self, field_name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    f'{field_name} must be finite and non-negative when set'
                )
        for field_name in ('command', 'raw_action', 'target_position'):
            values = tuple(float(value) for value in getattr(self, field_name))
            object.__setattr__(self, field_name, values)

    @property
    def reason(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        return {
            'code': self.code.value,
            'source': self.source,
            'message': self.message,
            'timestamp': self.timestamp,
            'runtime_state': self.runtime_state.value,
            'state_age_s': self.state_age_s,
            'command': list(self.command),
            'raw_action': list(self.raw_action),
            'target_position': list(self.target_position),
            'cycle_duration_s': self.cycle_duration_s,
        }


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
        self._root_fault: FaultRecord | None = None
        self._secondary_faults: list[FaultRecord] = []

    @property
    def state(self) -> RuntimeState:
        with self._condition:
            return self._state

    @property
    def transitions(self) -> tuple[RuntimeTransition, ...]:
        with self._condition:
            return tuple(self._transitions)

    @property
    def fault_record(self) -> FaultRecord | None:
        with self._condition:
            return self._root_fault

    @property
    def secondary_faults(self) -> tuple[FaultRecord, ...]:
        with self._condition:
            return tuple(self._secondary_faults)

    @property
    def fault_reason(self) -> str | None:
        with self._condition:
            return None if self._root_fault is None else self._root_fault.message

    @property
    def latest_fault_reason(self) -> str | None:
        with self._condition:
            if self._secondary_faults:
                return self._secondary_faults[-1].message
            return None if self._root_fault is None else self._root_fault.message

    def mark_ready(self, reason: str = 'initial hardware state is ready') -> None:
        self._transition(RuntimeState.READY, reason)

    def begin_move_to_default(
        self,
        reason: str = 'RL control requested',
    ) -> None:
        self._transition(RuntimeState.MOVE_TO_DEFAULT, reason)

    def enter_rl(self, reason: str = 'observation history is ready') -> None:
        self._transition(RuntimeState.RL, reason)

    def fault(self, record: FaultRecord) -> bool:
        if not isinstance(record, FaultRecord):
            raise TypeError('record must be a FaultRecord')
        with self._condition:
            if self._state is RuntimeState.SHUTDOWN:
                raise RuntimeTransitionError(
                    'Cannot report a fault after SHUTDOWN'
                )
            if self._root_fault is not None:
                self._secondary_faults.append(record)
                self._condition.notify_all()
                return False

            self._root_fault = record
            self._transition_locked(
                RuntimeState.FAULT,
                record.message,
                record.timestamp,
            )
            self._condition.notify_all()
            return True

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
