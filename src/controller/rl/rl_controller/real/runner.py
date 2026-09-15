from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import numpy as np

from .config import RealDeploymentConfig
from .hardware import (
    ImuBackend,
    MockImuBackend,
    MockMotorBackend,
    MotorBackend,
    UnitreeM8010Backend,
    YesenseYis130Backend,
)
from .observation import RealObservationHistory
from .policy import PolicyExecutor, load_real_policy
from .state import RealStateAggregator, StateUnavailableError
from .state_machine import (
    RealRuntimeStateMachine,
    RuntimeState,
    RuntimeTransitionError,
)
from .types import JointCommand, RealRobotState


class RealRunnerError(RuntimeError):
    """Base exception for real runner lifecycle failures."""


class RealRunnerStartupError(RealRunnerError):
    """Raised when the runner cannot obtain its initial synchronized state."""


class RealRunnerShutdownError(RealRunnerError):
    """Raised after shutdown when one or more resources failed to close."""


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    worker_name: str
    message: str
    timestamp: float
    exception: Exception


class RealRLRunner:
    """Threaded hardware-to-policy runtime without a command-line frontend."""

    def __init__(
        self,
        config: RealDeploymentConfig,
        *,
        motor_backend: MotorBackend,
        imu_backend: ImuBackend,
        policy: PolicyExecutor,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
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
        self._clock = clock

        self.state_machine = RealRuntimeStateMachine(clock=clock)
        self.state_aggregator = RealStateAggregator.from_config(
            config,
            clock=clock,
        )
        self.observation_history = RealObservationHistory(
            config,
            clock=clock,
        )

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
        self._metrics_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._initial_state_event = threading.Event()
        self._worker_failures: queue.Queue[WorkerFailure] = queue.Queue()
        self._recorded_failures: list[WorkerFailure] = []
        self._worker_iterations = {'motor': 0, 'imu': 0, 'policy': 0}
        self._threads: list[threading.Thread] = []
        self._started = False
        self._shutdown_started = False

        self._latest_joint_command = self._make_joint_command(
            config.robot.default_angles,
            use_gains=False,
        )

    @classmethod
    def from_config(
        cls,
        config: RealDeploymentConfig,
        *,
        motor_backend: MotorBackend | None = None,
        imu_backend: ImuBackend | None = None,
        policy: PolicyExecutor | None = None,
        sdk_module: Any | None = None,
        serial_factory: Callable[..., Any] | None = None,
        ort_module: Any | None = None,
        mock_action: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        policy_clock: Callable[[], float] = time.perf_counter,
    ) -> 'RealRLRunner':
        if motor_backend is None:
            if config.motors.backend == 'mock':
                motor_backend = MockMotorBackend(
                    config.motors,
                    initial_position=config.robot.default_angles,
                    clock=clock,
                )
            elif config.motors.backend == 'unitree_m8010':
                motor_backend = UnitreeM8010Backend(
                    config.motors,
                    sdk_module=sdk_module,
                    clock=clock,
                )
            else:
                raise ValueError(
                    f'Unsupported motor backend: {config.motors.backend}'
                )

        if imu_backend is None:
            if config.imu.backend == 'mock':
                imu_backend = MockImuBackend(config.imu, clock=clock)
            elif config.imu.backend == 'yesense_yis130':
                imu_backend = YesenseYis130Backend(
                    config.imu,
                    serial_factory=serial_factory,
                    clock=clock,
                )
            else:
                raise ValueError(
                    f'Unsupported IMU backend: {config.imu.backend}'
                )

        if policy is None:
            policy = load_real_policy(
                config,
                mock_action=mock_action,
                ort_module=ort_module,
                clock=policy_clock,
            )

        return cls(
            config,
            motor_backend=motor_backend,
            imu_backend=imu_backend,
            policy=policy,
            clock=clock,
        )

    @property
    def state(self) -> RuntimeState:
        return self.state_machine.state

    @property
    def fault_reason(self) -> str | None:
        return self.state_machine.fault_reason

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
    def worker_failures(self) -> tuple[WorkerFailure, ...]:
        with self._metrics_lock:
            return tuple(self._recorded_failures)

    @property
    def worker_iterations(self) -> Mapping[str, int]:
        with self._metrics_lock:
            return self._worker_iterations.copy()

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
            self._stop_event.clear()
            self._initial_state_event.clear()

            try:
                self.motor_backend.open()
                self.imu_backend.open()
            except Exception as exc:
                self._close_after_startup_failure()
                reason = f'hardware startup failed: {exc}'
                self.state_machine.fault(reason)
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
            self.report_fault(reason)
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
            self._move_start_time = self._read_clock()
            self.observation_history.reset('moving to default pose')
            self._latest_joint_command = self._make_joint_command(
                self._move_start_position,
                use_gains=True,
            )
        self.state_machine.begin_move_to_default()

    def set_velocity_command(self, command: Any) -> np.ndarray:
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

    def report_fault(self, reason: str) -> None:
        self.state_machine.fault(reason)
        self._stop_event.set()
        with self._control_lock:
            self._velocity_command.fill(0.0)
            self._last_action.fill(0.0)
        try:
            self.motor_backend.safe_stop()
        except Exception as exc:
            self.state_machine.fault(f'motor safe_stop failed: {exc}')

    def shutdown(self, reason: str = 'shutdown requested') -> None:
        with self._lifecycle_lock:
            if self.state is RuntimeState.SHUTDOWN:
                return
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._stop_event.set()

        errors: list[str] = []
        if self._started:
            try:
                self.motor_backend.safe_stop()
            except Exception as exc:
                errors.append(f'motor safe_stop failed: {exc}')

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
                    'worker threads did not stop: ' + ', '.join(live_threads)
                )

        for name, close in (
            ('IMU', self.imu_backend.close),
            ('motor', self.motor_backend.close),
        ):
            try:
                close()
            except Exception as exc:
                errors.append(f'{name} close failed: {exc}')

        if errors and self.state is not RuntimeState.FAULT:
            self.state_machine.fault(' | '.join(errors))
        elif errors:
            for error in errors:
                self.state_machine.fault(error)
        self.state_machine.shutdown(reason)

        if errors:
            raise RealRunnerShutdownError(' | '.join(errors))

    def __enter__(self) -> 'RealRLRunner':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is not None and self.state is not RuntimeState.SHUTDOWN:
            self.report_fault(f'runner context failed: {exc_value}')
        self.shutdown()

    def _create_threads(self) -> list[threading.Thread]:
        workers = (
            ('real-runtime-monitor', self._monitor_loop),
            (
                'real-motor-io',
                lambda: self._periodic_loop(
                    'motor',
                    self.config.motors.io_frequency_hz,
                    self._motor_step,
                ),
            ),
            (
                'real-imu-io',
                lambda: self._periodic_loop(
                    'imu',
                    self.config.motors.io_frequency_hz,
                    self._imu_step,
                ),
            ),
            (
                'real-policy',
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
                name=name,
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
            if not self._stop_event.is_set():
                try:
                    timestamp = self._read_clock()
                except Exception:
                    timestamp = time.monotonic()
                self._worker_failures.put(
                    WorkerFailure(
                        worker_name=worker_name,
                        message=f'{worker_name} failed: {exc}',
                        timestamp=timestamp,
                        exception=exc,
                    )
                )

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                failure = self._worker_failures.get(timeout=0.01)
            except queue.Empty:
                continue
            with self._metrics_lock:
                self._recorded_failures.append(failure)
            self.report_fault(failure.message)
            return

    def _periodic_loop(
        self,
        worker_name: str,
        frequency_hz: float,
        callback: Callable[[], None],
    ) -> None:
        period_s = 1.0 / frequency_hz
        next_deadline = self._read_clock()
        while not self._stop_event.is_set():
            callback()
            with self._metrics_lock:
                self._worker_iterations[worker_name] += 1

            next_deadline += period_s
            now = self._read_clock()
            if next_deadline <= now:
                next_deadline = now
                continue
            self._stop_event.wait(next_deadline - now)

    def _motor_step(self) -> None:
        with self._control_lock:
            command = self._latest_joint_command
        state = self.motor_backend.cycle(command)

        source_timestamps = getattr(
            self.motor_backend,
            'last_bus_timestamps',
            None,
        )
        self.state_aggregator.update_joints(
            state,
            source_timestamps=(
                source_timestamps if source_timestamps else None
            ),
        )
        self._mark_initial_state_if_ready()

    def _imu_step(self) -> None:
        self.state_aggregator.update_imu(self.imu_backend.read())
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
        now = self._read_clock()
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
            if alpha < 1.0:
                return

            observation = self.observation_history.update(
                state,
                np.zeros(3, dtype=np.float32),
                self._last_action,
            )
            if observation is None:
                return
            self._apply_policy(observation)
            self.state_machine.enter_rl()

    def _rl_step(self) -> None:
        state = self.state_aggregator.snapshot()
        with self._control_lock:
            observation = self.observation_history.update(
                state,
                self._velocity_command,
                self._last_action,
            )
            if observation is None:
                raise RealRunnerError(
                    'Observation history lost readiness during RL'
                )
            self._apply_policy(observation)

    def _apply_policy(self, observation: np.ndarray) -> None:
        action = self.policy.infer(observation)
        target = (
            self.config.robot.default_angles
            + action * self.config.robot.action_scale
        ).astype(np.float32)
        self._last_action = np.array(action, dtype=np.float32, copy=True)
        self._latest_joint_command = self._make_joint_command(
            target,
            use_gains=True,
        )

    def _make_joint_command(
        self,
        position: Any,
        *,
        use_gains: bool,
    ) -> JointCommand:
        return JointCommand(
            position=np.asarray(position, dtype=np.float32),
            velocity=self._zeros,
            kp=self._kp if use_gains else self._zeros,
            kd=self._kd if use_gains else self._zeros,
            torque=self._zeros,
            timestamp=self._read_clock(),
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

    def _close_after_startup_failure(self) -> None:
        try:
            self.imu_backend.close()
        except Exception:
            pass
        try:
            self.motor_backend.safe_stop()
        except Exception:
            pass
        try:
            self.motor_backend.close()
        except Exception:
            pass

    def _read_clock(self) -> float:
        timestamp = float(self._clock())
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise RealRunnerError(
                'runner clock must return finite, non-negative time'
            )
        return timestamp

    @staticmethod
    def _positive_timeout(value: float) -> float:
        timeout_s = float(value)
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError('ready_timeout_s must be finite and positive')
        return timeout_s
