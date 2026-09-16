from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import threading
import time

import numpy as np

from .config import RealDeploymentConfig
from .hardware import (
    BackendError,
    ImuBackend,
    MotorBackend,
    UnitreeM8010Backend,
    YesenseYis130Backend,
)
from .observation import ObservationContinuityError, RealObservationHistory
from .policy import (
    PolicyError,
    PolicyExecutor,
    PolicyInferenceTimeoutError,
    load_real_policy,
)
from .safety import RealSafetyManager, SafetyViolationError
from .state import RealStateAggregator, StateUnavailableError
from .state_machine import (
    FaultCode,
    FaultRecord,
    RealRuntimeStateMachine,
    RuntimeState,
    RuntimeTransitionError,
)
from .types import JointCommand


class RealRunnerError(RuntimeError):
    """Base exception for real runner lifecycle failures."""


class RealRunnerStartupError(RealRunnerError):
    """Raised when startup cannot produce a synchronized hardware state."""


class RealRunnerShutdownError(RealRunnerError):
    """Raised after shutdown when one or more resources failed to close."""


class WorkerDeadlineError(RealRunnerError):
    """Raised when a completed or blocked worker exceeds its deadline."""


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    worker_name: str
    exception: Exception
    cycle_duration_s: float | None


class RealRLRunner:
    """Threaded real-hardware runtime with one latched fault boundary."""

    def __init__(
        self,
        config: RealDeploymentConfig,
        *,
        motor_backend: MotorBackend,
        imu_backend: ImuBackend,
        policy: PolicyExecutor,
    ) -> None:
        if not isinstance(motor_backend, MotorBackend):
            raise TypeError('motor_backend must implement MotorBackend')
        if not isinstance(imu_backend, ImuBackend):
            raise TypeError('imu_backend must implement ImuBackend')
        if not isinstance(policy, PolicyExecutor):
            raise TypeError('policy must be a PolicyExecutor')
        if motor_backend.num_joints != config.robot.num_actions:
            raise ValueError(
                'motor backend joint count does not match robot config: '
                f'{motor_backend.num_joints} != {config.robot.num_actions}'
            )
        if policy.config.num_observations != config.robot.num_obs:
            raise ValueError('policy observation count does not match robot config')
        if policy.config.num_actions != config.robot.num_actions:
            raise ValueError('policy action count does not match robot config')

        self.config = config
        self.motor_backend = motor_backend
        self.imu_backend = imu_backend
        self.policy = policy

        self.state_machine = RealRuntimeStateMachine()
        self.state_aggregator = RealStateAggregator.from_config(config)
        self.observation_history = RealObservationHistory(config)
        self.safety_manager = RealSafetyManager(config)

        num_actions = config.robot.num_actions
        self._kp = np.asarray(
            [joint.kp for joint in config.motors.joints],
            dtype=np.float32,
        )
        self._kd = np.asarray(
            [joint.kd for joint in config.motors.joints],
            dtype=np.float32,
        )
        self._zeros = np.zeros(num_actions, dtype=np.float32)
        self._velocity_command = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(num_actions, dtype=np.float32)
        self._move_start_position: np.ndarray | None = None
        self._move_start_time: float | None = None

        self._control_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._watchdog_lock = threading.Lock()
        self._motor_io_lock = threading.Lock()
        self._fault_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._initial_state_event = threading.Event()
        self._worker_cycle_started: dict[str, float | None] = {
            'motor': None,
            'imu': None,
            'policy': None,
        }
        self._last_cycle_duration_s: dict[str, float | None] = {
            'motor': None,
            'imu': None,
            'policy': None,
        }
        self._watchdog_deadlines = {
            'motor': config.motors.state_timeout_s,
            'imu': config.imu.state_timeout_s,
            'policy': 1.0 / config.policy.frequency_hz,
        }
        self._threads: list[threading.Thread] = []
        self._started = False
        self._shutdown_started = False
        self._stop_reason: str | None = None

        self._latest_joint_command = self._make_joint_command(
            config.robot.default_angles,
            use_gains=False,
        )

    @classmethod
    def from_config(cls, config: RealDeploymentConfig) -> 'RealRLRunner':
        """Construct the current production hardware and ONNX chain."""
        motor_backend = UnitreeM8010Backend(config.motors)
        imu_backend = YesenseYis130Backend(config.imu)
        policy = load_real_policy(config)
        return cls(
            config,
            motor_backend=motor_backend,
            imu_backend=imu_backend,
            policy=policy,
        )

    @property
    def state(self) -> RuntimeState:
        return self.state_machine.state

    @property
    def fault_record(self) -> FaultRecord | None:
        return self.state_machine.fault_record

    @property
    def secondary_faults(self) -> tuple[FaultRecord, ...]:
        return self.state_machine.secondary_faults

    @property
    def fault_reason(self) -> str | None:
        return self.state_machine.fault_reason

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def velocity_command(self) -> np.ndarray:
        with self._control_lock:
            return self._velocity_command.copy()

    @property
    def last_action(self) -> np.ndarray:
        with self._control_lock:
            return self._last_action.copy()

    @property
    def latest_joint_command(self) -> JointCommand:
        with self._control_lock:
            return self._latest_joint_command

    @property
    def live_thread_names(self) -> tuple[str, ...]:
        return tuple(thread.name for thread in self._threads if thread.is_alive())

    def start(self, *, ready_timeout_s: float | None = None) -> None:
        with self._lifecycle_lock:
            if self.state is not RuntimeState.INIT or self._started:
                raise RealRunnerStartupError(
                    f'Runner can only start from INIT, current state is '
                    f'{self.state.value}'
                )
            timeout_s = (
                self.config.runtime.shutdown_timeout_s
                if ready_timeout_s is None
                else self._positive_timeout(ready_timeout_s)
            )
            self.state_aggregator.clear()
            self.observation_history.reset('runner startup')
            self.safety_manager.reset()
            self._stop_event.clear()
            self._initial_state_event.clear()
            self._stop_reason = None

            try:
                self.motor_backend.open()
                self.imu_backend.open()
            except Exception as exc:
                reason = f'hardware startup failed: {exc}'
                self.report_fault(
                    FaultCode.HARDWARE,
                    'startup',
                    reason,
                )
                for source, cleanup_error in self._close_after_startup_failure():
                    self._record_secondary_fault(
                        FaultCode.SHUTDOWN,
                        source,
                        cleanup_error,
                    )
                raise RealRunnerStartupError(reason) from exc

            self._threads = self._create_threads()
            self._started = True
            for thread in self._threads:
                thread.start()

        if not self._wait_for_initial_state(timeout_s):
            if self.state is RuntimeState.FAULT:
                raise RealRunnerStartupError(
                    self.fault_reason or 'worker failed during startup'
                )
            reason = (
                f'initial synchronized state was unavailable after '
                f'{timeout_s:.3f}s'
            )
            self.report_fault(FaultCode.HARDWARE, 'startup', reason)
            raise RealRunnerStartupError(reason)

        try:
            self.state_machine.mark_ready()
        except RuntimeTransitionError as exc:
            raise RealRunnerStartupError(
                self.fault_reason or str(exc)
            ) from exc

    def enable_rl(self) -> None:
        if self.state is not RuntimeState.READY:
            raise RuntimeTransitionError(
                f'RL can only be enabled from READY, current state is '
                f'{self.state.value}'
            )
        try:
            state = self.state_aggregator.snapshot()
        except StateUnavailableError as exc:
            raise RealRunnerError(
                f'Cannot begin default-pose transition: {exc}'
            ) from exc

        with self._control_lock:
            self._velocity_command.fill(0.0)
            self._last_action.fill(0.0)
            self._move_start_position = np.array(
                state.joints.position,
                dtype=np.float32,
                copy=True,
            )
            self._move_start_time = time.monotonic()
            self.observation_history.reset('moving to default pose')
            self.state_machine.begin_move_to_default()
            self._latest_joint_command = self._make_joint_command(
                self._move_start_position,
                use_gains=True,
            )

    def set_velocity_command(self, command: object) -> np.ndarray:
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (3,):
            raise ValueError(
                f'velocity command must have shape (3,), got {value.shape}'
            )
        if not np.all(np.isfinite(value)):
            raise ValueError('velocity command must contain only finite values')
        clipped = np.clip(
            value,
            -self.config.keyboard.command_limits,
            self.config.keyboard.command_limits,
        ).astype(np.float32)
        with self._control_lock:
            self._velocity_command = clipped
            return self._velocity_command.copy()

    def clear_velocity_command(self) -> None:
        with self._control_lock:
            self._velocity_command.fill(0.0)

    def wait_for_state(
        self,
        states: RuntimeState | Iterable[RuntimeState],
        timeout: float | None = None,
    ) -> bool:
        return self.state_machine.wait_for(states, timeout)

    def request_stop(self, reason: str = 'operator stop requested') -> None:
        normalized_reason = str(reason).strip() or 'operator stop requested'
        self._stop_reason = normalized_reason
        self._stop_event.set()
        with self._control_lock:
            self._velocity_command.fill(0.0)
            self._last_action.fill(0.0)
        try:
            with self._motor_io_lock:
                self.motor_backend.safe_stop()
        except Exception as exc:
            self._latch_fault(
                FaultCode.HARDWARE,
                'operator-stop',
                f'motor safe_stop failed: {exc}',
                stop_motor=False,
            )

    def report_exception(
        self,
        source: str,
        exception: Exception,
        *,
        cycle_duration_s: float | None = None,
    ) -> FaultRecord:
        return self.report_fault(
            self._classify_exception(exception),
            source,
            f'{source} failed: {exception}',
            cycle_duration_s=cycle_duration_s,
        )

    def report_fault(
        self,
        code: FaultCode,
        source: str,
        message: str,
        *,
        cycle_duration_s: float | None = None,
    ) -> FaultRecord:
        return self._latch_fault(
            code,
            source,
            message,
            cycle_duration_s=cycle_duration_s,
            stop_motor=True,
        )

    def shutdown(self, reason: str = 'shutdown requested') -> None:
        with self._lifecycle_lock:
            if self.state is RuntimeState.SHUTDOWN:
                return
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._stop_event.set()

        errors: list[tuple[str, str]] = []
        if self._started:
            try:
                with self._motor_io_lock:
                    self.motor_backend.safe_stop()
            except Exception as exc:
                errors.append(('shutdown.safe_stop', str(exc)))

            deadline = time.monotonic() + self.config.runtime.shutdown_timeout_s
            current_thread = threading.current_thread()
            for thread in self._threads:
                if thread is current_thread:
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
            live_threads = self.live_thread_names
            if live_threads:
                errors.append(
                    (
                        'shutdown.threads',
                        'worker threads did not stop: '
                        + ', '.join(live_threads),
                    )
                )

        for source, close in (
            ('shutdown.imu', self.imu_backend.close),
            ('shutdown.motor', self.motor_backend.close),
        ):
            try:
                close()
            except Exception as exc:
                errors.append((source, str(exc)))

        for source, message in errors:
            self._latch_fault(
                FaultCode.SHUTDOWN,
                source,
                message,
                stop_motor=False,
            )

        self.state_machine.shutdown(reason)
        if errors:
            raise RealRunnerShutdownError(
                ' | '.join(f'{source}: {message}' for source, message in errors)
            )

    def __enter__(self) -> 'RealRLRunner':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is not None and self.state is not RuntimeState.SHUTDOWN:
            self.report_exception('runner-context', exc_value)
        self.shutdown()

    def _create_threads(self) -> list[threading.Thread]:
        workers: tuple[tuple[str, Callable[[], None]], ...] = (
            ('monitor', self._monitor_loop),
            (
                'motor',
                lambda: self._periodic_loop(
                    'motor',
                    self.config.motors.io_frequency_hz,
                    self._motor_step,
                ),
            ),
            (
                'imu',
                lambda: self._periodic_loop(
                    'imu',
                    self.config.motors.io_frequency_hz,
                    self._imu_step,
                ),
            ),
            (
                'policy',
                lambda: self._periodic_loop(
                    'policy',
                    self.config.policy.frequency_hz,
                    self._policy_step,
                ),
            ),
        )
        return [
            threading.Thread(
                target=self._worker_entry,
                args=(name, target),
                name=f'real-{name}',
                daemon=False,
            )
            for name, target in workers
        ]

    def _worker_entry(
        self,
        worker_name: str,
        target: Callable[[], None],
    ) -> None:
        try:
            target()
        except Exception as exc:
            if self._stop_event.is_set():
                return
            self.report_exception(
                worker_name,
                exc,
                cycle_duration_s=self._last_cycle_duration_s.get(worker_name),
            )

    def _monitor_loop(self) -> None:
        poll_interval_s = min(
            0.01,
            min(self._watchdog_deadlines.values()) / 4.0,
        )
        while not self._stop_event.wait(poll_interval_s):
            failure = self._find_watchdog_failure()
            if failure is None:
                continue
            self.report_exception(
                failure.worker_name,
                failure.exception,
                cycle_duration_s=failure.cycle_duration_s,
            )
            return

    def _find_watchdog_failure(self) -> _WorkerFailure | None:
        now = time.monotonic()
        with self._watchdog_lock:
            cycle_started = self._worker_cycle_started.copy()
        for worker_name, started_at in cycle_started.items():
            if started_at is None:
                continue
            deadline_s = self._watchdog_deadlines[worker_name]
            elapsed_s = now - started_at
            if elapsed_s <= deadline_s:
                continue
            return _WorkerFailure(
                worker_name=worker_name,
                exception=WorkerDeadlineError(
                    f'exceeded {deadline_s:.6f}s watchdog deadline: '
                    f'{elapsed_s:.6f}s'
                ),
                cycle_duration_s=elapsed_s,
            )
        return None

    def _periodic_loop(
        self,
        worker_name: str,
        frequency_hz: float,
        callback: Callable[[], None],
    ) -> None:
        period_s = 1.0 / frequency_hz
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            started_at = time.monotonic()
            with self._watchdog_lock:
                self._worker_cycle_started[worker_name] = started_at
            try:
                callback()
            finally:
                duration_s = time.monotonic() - started_at
                self._last_cycle_duration_s[worker_name] = duration_s
                with self._watchdog_lock:
                    self._worker_cycle_started[worker_name] = None

            deadline_s = self._watchdog_deadlines[worker_name]
            if duration_s > deadline_s:
                raise WorkerDeadlineError(
                    f'{worker_name} cycle exceeded {deadline_s:.6f}s '
                    f'deadline: {duration_s:.6f}s'
                )

            next_deadline += period_s
            now = time.monotonic()
            if next_deadline <= now:
                next_deadline = now
                continue
            self._stop_event.wait(next_deadline - now)

    def _motor_step(self) -> None:
        with self._motor_io_lock:
            if self._stop_event.is_set():
                return
            with self._control_lock:
                runtime_state = self.state
                command = self._latest_joint_command

            robot_state = None
            if runtime_state is not RuntimeState.INIT:
                robot_state = self.state_aggregator.snapshot()
            self.safety_manager.authorize_command(
                command,
                runtime_state,
                robot_state,
            )

            state = self.motor_backend.cycle(command)
            self.safety_manager.validate_motor_health(state)
            source_timestamps = getattr(
                self.motor_backend,
                'last_bus_timestamps',
                None,
            )
            self.state_aggregator.update_joints(
                state,
                source_timestamps=source_timestamps or None,
            )
            self._mark_initial_state_if_ready()

    def _imu_step(self) -> None:
        state = self.imu_backend.read()
        self.safety_manager.validate_imu_health(state)
        self.state_aggregator.update_imu(state)
        self._mark_initial_state_if_ready()

    def _mark_initial_state_if_ready(self) -> None:
        if self._initial_state_event.is_set():
            return
        if self.state_aggregator.try_snapshot() is not None:
            self._initial_state_event.set()

    def _policy_step(self) -> None:
        state = self.state
        if state is RuntimeState.MOVE_TO_DEFAULT:
            self._move_to_default_step()
        elif state is RuntimeState.RL:
            self._rl_step()

    def _move_to_default_step(self) -> None:
        state = self.state_aggregator.snapshot()
        now = time.monotonic()
        with self._control_lock:
            if (
                self._move_start_position is None
                or self._move_start_time is None
            ):
                raise RealRunnerError(
                    'MOVE_TO_DEFAULT started without an initial pose'
                )
            elapsed_s = max(0.0, now - self._move_start_time)
            alpha = min(
                1.0,
                elapsed_s / self.config.runtime.move_to_default_duration_s,
            )
            target = (
                self._move_start_position
                + alpha
                * (
                    self.config.robot.default_angles
                    - self._move_start_position
                )
            ).astype(np.float32)
            self._latest_joint_command = self._make_joint_command(
                target,
                use_gains=True,
            )
            last_action = self._last_action.copy()
        if alpha < 1.0:
            return

        observation = self.observation_history.update(
            state,
            np.zeros(3, dtype=np.float32),
            last_action,
        )
        if observation is None:
            return
        if not self._apply_policy(observation):
            return
        self.state_machine.enter_rl()

    def _rl_step(self) -> None:
        state = self.state_aggregator.snapshot()
        with self._control_lock:
            velocity_command = self._velocity_command.copy()
            last_action = self._last_action.copy()
        observation = self.observation_history.update(
            state,
            velocity_command,
            last_action,
        )
        if observation is None:
            raise RealRunnerError(
                'Observation history lost readiness during RL'
            )
        self._apply_policy(observation)

    def _apply_policy(self, observation: np.ndarray) -> bool:
        action = self.policy.infer(observation)
        if (
            self._stop_event.is_set()
            or self.state in {RuntimeState.FAULT, RuntimeState.SHUTDOWN}
        ):
            return False
        target = (
            self.config.robot.default_angles
            + action * self.config.robot.action_scale
        ).astype(np.float32)
        with self._control_lock:
            self._last_action = action.copy()
            self._latest_joint_command = self._make_joint_command(
                target,
                use_gains=True,
            )
        return True

    def _make_joint_command(
        self,
        position: object,
        *,
        use_gains: bool,
    ) -> JointCommand:
        return JointCommand(
            position=np.asarray(position, dtype=np.float32),
            velocity=self._zeros,
            kp=self._kp if use_gains else self._zeros,
            kd=self._kd if use_gains else self._zeros,
            torque=self._zeros,
            timestamp=time.monotonic(),
        )

    def _wait_for_initial_state(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self._initial_state_event.is_set():
            if self.state is RuntimeState.FAULT:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._initial_state_event.wait(timeout=min(remaining, 0.01))
        return self.state is not RuntimeState.FAULT

    def _latch_fault(
        self,
        code: FaultCode,
        source: str,
        message: str,
        *,
        cycle_duration_s: float | None = None,
        stop_motor: bool,
    ) -> FaultRecord:
        with self._fault_lock:
            record = self._make_fault_record(
                code,
                source,
                message,
                cycle_duration_s,
            )
            is_root = self.state_machine.fault(record)
            self._stop_event.set()
            with self._control_lock:
                self._velocity_command.fill(0.0)
                self._last_action.fill(0.0)

        if is_root and stop_motor:
            try:
                with self._motor_io_lock:
                    self.motor_backend.safe_stop()
            except Exception as exc:
                self._record_secondary_fault(
                    FaultCode.SHUTDOWN,
                    'fault.safe_stop',
                    f'motor safe_stop failed: {exc}',
                )
        return record

    def _record_secondary_fault(
        self,
        code: FaultCode,
        source: str,
        message: str,
    ) -> None:
        with self._fault_lock:
            record = self._make_fault_record(code, source, message, None)
            self.state_machine.fault(record)

    def _make_fault_record(
        self,
        code: FaultCode,
        source: str,
        message: str,
        cycle_duration_s: float | None,
    ) -> FaultRecord:
        state_age_s: float | None = None
        try:
            state_age_s = self.state_aggregator.feedback_ages().get('oldest')
            if state_age_s is not None:
                state_age_s = max(0.0, float(state_age_s))
        except Exception:
            state_age_s = None

        with self._control_lock:
            command = tuple(float(value) for value in self._velocity_command)
            target = tuple(
                float(value) for value in self._latest_joint_command.position
            )
            last_action = self._last_action.copy()
        raw_action = self.policy.last_raw_action
        if raw_action is None:
            raw_action = last_action

        return FaultRecord(
            code=code,
            source=str(source).strip(),
            message=str(message).strip(),
            timestamp=time.monotonic(),
            runtime_state=self.state,
            state_age_s=state_age_s,
            command=command,
            raw_action=tuple(float(value) for value in raw_action),
            target_position=target,
            cycle_duration_s=cycle_duration_s,
        )

    def _close_after_startup_failure(self) -> list[tuple[str, str]]:
        errors: list[tuple[str, str]] = []
        for source, close in (
            ('startup.imu_close', self.imu_backend.close),
            ('startup.motor_close', self.motor_backend.close),
        ):
            try:
                close()
            except Exception as exc:
                errors.append((source, str(exc)))
        return errors

    @staticmethod
    def _classify_exception(exception: Exception) -> FaultCode:
        if isinstance(exception, (WorkerDeadlineError, PolicyInferenceTimeoutError)):
            return FaultCode.TIMING
        if isinstance(exception, BackendError):
            return FaultCode.HARDWARE
        if isinstance(exception, PolicyError):
            return FaultCode.POLICY
        if isinstance(exception, SafetyViolationError):
            return FaultCode.SAFETY
        if isinstance(exception, (StateUnavailableError, ObservationContinuityError)):
            return FaultCode.INPUT
        return FaultCode.INTERNAL

    @staticmethod
    def _positive_timeout(value: float) -> float:
        timeout_s = float(value)
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError('ready_timeout_s must be finite and positive')
        return timeout_s
