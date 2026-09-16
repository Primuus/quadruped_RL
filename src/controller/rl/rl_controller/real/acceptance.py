from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import RealDeploymentConfig
from .hardware import MockImuBackend, MockMotorBackend
from .policy import MockPolicy
from .runner import RealRLRunner
from .state_machine import RuntimeState


FaultInjector = Callable[[RealRLRunner], None]


class MockAcceptanceError(RuntimeError):
    """Raised when the offline acceptance suite cannot be executed."""


class MockAcceptanceSuite:
    """Run nominal soak and fault-matrix checks through the real runner."""

    def __init__(
        self,
        config: RealDeploymentConfig,
        *,
        soak_duration_s: float = 60.0,
        sample_interval_s: float | None = None,
        fault_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        if (
            config.motors.backend != 'mock'
            or config.imu.backend != 'mock'
            or config.policy.backend != 'mock'
        ):
            raise MockAcceptanceError(
                'Offline acceptance requires mock motor, IMU, and policy backends'
            )
        self.config = config
        self.soak_duration_s = self._positive_finite(
            soak_duration_s,
            'soak_duration_s',
        )
        interval = (
            1.0 / config.runtime.log_frequency_hz
            if sample_interval_s is None
            else sample_interval_s
        )
        self.sample_interval_s = self._positive_finite(
            interval,
            'sample_interval_s',
        )
        self.fault_timeout_s = self._positive_finite(
            fault_timeout_s,
            'fault_timeout_s',
        )
        self._clock = clock

    def run(self) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started = self._read_clock()
        scenarios = [
            self._run_scenario(
                name='nominal_soak',
                category='nominal',
                expected_state=RuntimeState.RL,
                injector=None,
                expected_fault_text=None,
            ),
            self._run_scenario(
                name='motor_communication_fault',
                category='communication',
                expected_state=RuntimeState.FAULT,
                injector=self._inject_motor_communication_fault,
                expected_fault_text='injected motor communication failure',
            ),
            self._run_scenario(
                name='imu_nonfinite_fault',
                category='imu',
                expected_state=RuntimeState.FAULT,
                injector=self._inject_imu_fault,
                expected_fault_text='IMU angular velocity',
            ),
            self._run_scenario(
                name='policy_nonfinite_fault',
                category='policy',
                expected_state=RuntimeState.FAULT,
                injector=self._inject_policy_fault,
                expected_fault_text='Policy action',
            ),
            self._run_scenario(
                name='target_step_fault',
                category='target',
                expected_state=RuntimeState.FAULT,
                injector=self._inject_target_fault,
                expected_fault_text='target step',
            ),
        ]
        completed_at = datetime.now(timezone.utc)
        return {
            'schema_version': 1,
            'suite': 'real_mock_offline_acceptance',
            'started_at_utc': self._format_utc(started_at),
            'completed_at_utc': self._format_utc(completed_at),
            'duration_s': self._read_clock() - started,
            'passed': all(scenario['passed'] for scenario in scenarios),
            'config': {
                'source_path': str(self.config.source_path),
                'robot': self.config.robot.name,
                'motor_backend': self.config.motors.backend,
                'imu_backend': self.config.imu.backend,
                'policy_backend': self.config.policy.backend,
            },
            'parameters': {
                'soak_duration_s': self.soak_duration_s,
                'sample_interval_s': self.sample_interval_s,
                'fault_timeout_s': self.fault_timeout_s,
            },
            'scenarios': scenarios,
        }

    def _run_scenario(
        self,
        *,
        name: str,
        category: str,
        expected_state: RuntimeState,
        injector: FaultInjector | None,
        expected_fault_text: str | None,
    ) -> dict[str, Any]:
        started = self._read_clock()
        runner = RealRLRunner.from_config(self.config)
        samples: list[dict[str, Any]] = []
        execution_error: str | None = None
        shutdown_error: str | None = None
        terminal_state = runner.state
        baseline_iterations = runner.worker_iterations
        active_started = started

        try:
            try:
                runner.start(ready_timeout_s=self._startup_timeout_s())
                runner.enable_rl()
                if not runner.wait_for_state(
                    (RuntimeState.RL, RuntimeState.FAULT),
                    timeout=self._rl_transition_timeout_s(),
                ):
                    raise MockAcceptanceError('timed out while entering RL')
                samples.append(self._sample(runner, 'rl_ready', started))
                if runner.state is not RuntimeState.RL:
                    raise MockAcceptanceError(
                        runner.fault_reason
                        or 'runner faulted while entering RL'
                    )

                baseline_iterations = runner.worker_iterations
                active_started = self._read_clock()
                if injector is None:
                    self._run_nominal_soak(runner, samples, started)
                else:
                    samples.append(
                        self._sample(runner, 'pre_injection', started)
                    )
                    injector(runner)
                    self._wait_for_fault(runner, samples, started)
                terminal_state = runner.state
                samples.append(self._sample(runner, 'terminal', started))
            except Exception as exc:
                terminal_state = runner.state
                execution_error = f'{type(exc).__name__}: {exc}'
                samples.append(
                    self._sample(runner, 'execution_error', started)
                )

            fault_reason = runner.fault_reason
            terminal_iterations = runner.worker_iterations
            active_duration_s = max(
                0.0,
                self._read_clock() - active_started,
            )
            worker_failures = [
                {
                    'worker_name': failure.worker_name,
                    'message': failure.message,
                    'timestamp': failure.timestamp,
                    'exception_type': type(failure.exception).__name__,
                }
                for failure in runner.worker_failures
            ]
        finally:
            try:
                runner.shutdown(
                    f'offline acceptance scenario {name} complete'
                )
            except Exception as exc:
                shutdown_error = f'{type(exc).__name__}: {exc}'

        samples.append(self._sample(runner, 'shutdown', started))

        resources = self._resource_status(runner)
        behavior_passed = (
            execution_error is None
            and terminal_state is expected_state
            and (
                expected_fault_text is None
                or expected_fault_text in (fault_reason or '')
            )
        )
        cleanup_passed = (
            shutdown_error is None
            and runner.state is RuntimeState.SHUTDOWN
            and not resources['live_threads']
            and resources['motor_open'] is False
            and resources['imu_open'] is False
            and resources['motor_safe_stopped'] is True
        )
        iteration_delta = {
            worker: terminal_iterations[worker]
            - baseline_iterations.get(worker, 0)
            for worker in terminal_iterations
        }
        observed_frequency_hz = {
            worker: (
                count / active_duration_s if active_duration_s > 0.0 else 0.0
            )
            for worker, count in iteration_delta.items()
        }
        return {
            'name': name,
            'category': category,
            'passed': behavior_passed and cleanup_passed,
            'behavior_passed': behavior_passed,
            'cleanup_passed': cleanup_passed,
            'expected_terminal_state': expected_state.value,
            'terminal_state': terminal_state.value,
            'final_state': runner.state.value,
            'fault_reason': fault_reason,
            'execution_error': execution_error,
            'shutdown_error': shutdown_error,
            'duration_s': self._read_clock() - started,
            'active_duration_s': active_duration_s,
            'worker_iteration_delta': iteration_delta,
            'observed_frequency_hz': observed_frequency_hz,
            'worker_failures': worker_failures,
            'resources': resources,
            'samples': samples,
        }

    def _run_nominal_soak(
        self,
        runner: RealRLRunner,
        samples: list[dict[str, Any]],
        scenario_started: float,
    ) -> None:
        deadline = self._read_clock() + self.soak_duration_s
        while True:
            now = self._read_clock()
            if runner.state is not RuntimeState.RL:
                return
            if now >= deadline:
                return
            time.sleep(min(self.sample_interval_s, deadline - now))
            samples.append(self._sample(runner, 'soak', scenario_started))

    def _wait_for_fault(
        self,
        runner: RealRLRunner,
        samples: list[dict[str, Any]],
        scenario_started: float,
    ) -> None:
        deadline = self._read_clock() + self.fault_timeout_s
        while runner.state is not RuntimeState.FAULT:
            remaining = deadline - self._read_clock()
            if remaining <= 0.0:
                raise MockAcceptanceError('fault injection did not reach FAULT')
            runner.wait_for_state(
                RuntimeState.FAULT,
                timeout=min(self.sample_interval_s, remaining),
            )
            samples.append(
                self._sample(runner, 'fault_wait', scenario_started)
            )

    def _sample(
        self,
        runner: RealRLRunner,
        label: str,
        scenario_started: float,
    ) -> dict[str, Any]:
        sample = runner.telemetry().as_dict()
        sample['label'] = label
        sample['elapsed_s'] = self._read_clock() - scenario_started
        return sample

    @staticmethod
    def _inject_motor_communication_fault(runner: RealRLRunner) -> None:
        backend = runner.motor_backend
        if not isinstance(backend, MockMotorBackend):
            raise MockAcceptanceError('motor backend is not MockMotorBackend')
        backend.inject_communication_error(
            'injected motor communication failure'
        )

    @staticmethod
    def _inject_imu_fault(runner: RealRLRunner) -> None:
        backend = runner.imu_backend
        if not isinstance(backend, MockImuBackend):
            raise MockAcceptanceError('IMU backend is not MockImuBackend')
        backend.inject_nan('angular_velocity', 0)

    @staticmethod
    def _inject_policy_fault(runner: RealRLRunner) -> None:
        backend = runner.policy.backend
        if not isinstance(backend, MockPolicy):
            raise MockAcceptanceError('policy backend is not MockPolicy')
        backend.inject_nan(0)

    def _inject_target_fault(self, runner: RealRLRunner) -> None:
        backend = runner.policy.backend
        if not isinstance(backend, MockPolicy):
            raise MockAcceptanceError('policy backend is not MockPolicy')

        joint_index = 0
        joint = self.config.motors.joints[joint_index]
        default = float(self.config.robot.default_angles[joint_index])
        positive_room = joint.position_max_rad - default
        negative_room = default - joint.position_min_rad
        direction = 1.0 if positive_room >= negative_room else -1.0
        available_room = max(positive_room, negative_room)
        target_delta = joint.max_target_step_rad * 1.5
        if target_delta >= available_room:
            raise MockAcceptanceError(
                'mock joint limits do not permit a target-step fault injection'
            )
        action_value = direction * target_delta / self.config.robot.action_scale
        if abs(action_value) > self.config.policy.clip_actions:
            raise MockAcceptanceError(
                'policy clip limit does not permit a target-step fault injection'
            )
        action = np.zeros(self.config.robot.num_actions, dtype=np.float32)
        action[joint_index] = action_value
        backend.set_action(action)

    def _resource_status(self, runner: RealRLRunner) -> dict[str, Any]:
        return {
            'live_threads': list(runner.live_thread_names),
            'motor_open': getattr(runner.motor_backend, 'is_open', None),
            'imu_open': getattr(runner.imu_backend, 'is_open', None),
            'motor_safe_stopped': getattr(
                runner.motor_backend,
                'safe_stopped',
                None,
            ),
        }

    def _startup_timeout_s(self) -> float:
        return max(1.0, self.config.runtime.shutdown_timeout_s)

    def _rl_transition_timeout_s(self) -> float:
        warmup_s = (
            self.config.runtime.history_warmup_frames
            / self.config.policy.frequency_hz
        )
        return max(
            self.fault_timeout_s,
            self.config.runtime.move_to_default_duration_s + warmup_s + 1.0,
        )

    def _read_clock(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise MockAcceptanceError('acceptance clock must be finite')
        return value

    @staticmethod
    def _positive_finite(value: float, field_name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f'{field_name} must be finite and positive')
        return number

    @staticmethod
    def _format_utc(value: datetime) -> str:
        return value.isoformat(timespec='seconds').replace('+00:00', 'Z')


def write_acceptance_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    return path
