from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG_PATH = RL_PACKAGE_ROOT / 'config' / 'real' / 'mock.yaml'
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real import (
    OnnxPolicy,
    PolicyContractError,
    PolicyExecutor,
    PolicyInferenceError,
    PolicyInferenceTimeoutError,
    PolicyLoadError,
    RealDeploymentConfig,
    load_real_policy,
)


class TensorMetadata:
    def __init__(
        self,
        name: str,
        shape: list[object],
        data_type: str = 'tensor(float)',
    ) -> None:
        self.name = name
        self.shape = shape
        self.type = data_type


class FakeSession:
    def __init__(
        self,
        *,
        inputs: list[TensorMetadata] | None = None,
        outputs: list[TensorMetadata] | None = None,
        action: np.ndarray | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.inputs = inputs or [TensorMetadata('observation', ['batch', 270])]
        self.outputs = outputs or [TensorMetadata('action', ['batch', 12])]
        self.action = (
            np.zeros((1, 12), dtype=np.float32)
            if action is None
            else action
        )
        self.run_error = run_error
        self.last_output_names: list[str] | None = None
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[TensorMetadata]:
        return self.inputs

    def get_outputs(self) -> list[TensorMetadata]:
        return self.outputs

    def run(
        self,
        output_names: list[str],
        feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        if self.run_error is not None:
            raise self.run_error
        self.last_output_names = output_names
        self.last_feed = {
            name: np.array(value, copy=True) for name, value in feed.items()
        }
        return [self.action]


class FakeOnnxRuntime:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.model_path: str | None = None
        self.providers: list[str] | None = None

    def InferenceSession(
        self,
        model_path: str,
        *,
        providers: list[str],
    ) -> FakeSession:
        self.model_path = model_path
        self.providers = providers
        return self.session


class RecordingPolicy:
    backend_name = 'recording'

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        action: np.ndarray,
        error: Exception | None = None,
    ) -> None:
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.action = action
        self.error = error
        self.last_observation: np.ndarray | None = None

    def infer(self, observation: np.ndarray) -> np.ndarray:
        self.last_observation = np.array(observation, copy=True)
        if self.error is not None:
            raise self.error
        return self.action


class SequenceClock:
    def __init__(self, *timestamps: float) -> None:
        self.timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self.timestamps)


class RealPolicyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RealDeploymentConfig.from_yaml(MOCK_CONFIG_PATH)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / 'policy.onnx'
        self.model_path.touch()
        self.onnx_policy_config = replace(
            self.config.policy,
            backend='onnx',
            model_path=self.model_path,
        )
        self.onnx_config = replace(
            self.config,
            policy=self.onnx_policy_config,
        )

    def make_onnx_policy(
        self,
        session: FakeSession,
    ) -> tuple[OnnxPolicy, FakeOnnxRuntime]:
        runtime = FakeOnnxRuntime(session)
        return (
            OnnxPolicy(self.onnx_policy_config, ort_module=runtime),
            runtime,
        )


class OnnxPolicyTest(RealPolicyTestCase):
    def test_loads_cpu_session_and_runs_float32_batch(self) -> None:
        expected_action = np.linspace(-0.4, 0.4, 12, dtype=np.float32)[None, :]
        session = FakeSession(action=expected_action)
        policy, runtime = self.make_onnx_policy(session)

        observation = np.linspace(-1.0, 1.0, 270, dtype=np.float32)
        action = policy.infer(observation)

        self.assertEqual(runtime.model_path, str(self.model_path))
        self.assertEqual(runtime.providers, ['CPUExecutionProvider'])
        self.assertEqual(session.last_output_names, ['action'])
        assert session.last_feed is not None
        self.assertEqual(session.last_feed['observation'].shape, (1, 270))
        self.assertEqual(session.last_feed['observation'].dtype, np.float32)
        np.testing.assert_array_equal(action, expected_action[0])
        self.assertTrue(action.flags.c_contiguous)

    def test_accepts_dynamic_or_fixed_batch_dimension(self) -> None:
        for batch_dimension in (None, 'batch', 1):
            with self.subTest(batch_dimension=batch_dimension):
                session = FakeSession(
                    inputs=[
                        TensorMetadata(
                            'observation',
                            [batch_dimension, 270],
                        )
                    ],
                    outputs=[
                        TensorMetadata('action', [batch_dimension, 12])
                    ],
                )
                policy, _ = self.make_onnx_policy(session)
                self.assertEqual(policy.num_observations, 270)

    def test_rejects_wrong_model_tensor_count(self) -> None:
        cases = (
            (
                FakeSession(
                    inputs=[
                        TensorMetadata('first', [1, 270]),
                        TensorMetadata('second', [1, 270]),
                    ]
                ),
                'exactly one input',
            ),
            (
                FakeSession(
                    outputs=[
                        TensorMetadata('first', [1, 12]),
                        TensorMetadata('second', [1, 12]),
                    ]
                ),
                'exactly one output',
            ),
        )
        for session, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyContractError, message):
                    self.make_onnx_policy(session)

    def test_rejects_wrong_model_dtype_rank_or_dimension(self) -> None:
        cases = (
            (
                [TensorMetadata('observation', [1, 270], 'tensor(double)')],
                [TensorMetadata('action', [1, 12])],
                r'input must use tensor\(float\)',
            ),
            (
                [TensorMetadata('observation', [270])],
                [TensorMetadata('action', [1, 12])],
                'input must have rank 2',
            ),
            (
                [TensorMetadata('observation', [2, 270])],
                [TensorMetadata('action', [1, 12])],
                'input batch dimension',
            ),
            (
                [TensorMetadata('observation', [1, 269])],
                [TensorMetadata('action', [1, 12])],
                'input feature dimension must be 270',
            ),
            (
                [TensorMetadata('observation', [1, 270])],
                [TensorMetadata('action', [1, 13])],
                'output feature dimension must be 12',
            ),
        )
        for inputs, outputs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyContractError, message):
                    self.make_onnx_policy(
                        FakeSession(inputs=inputs, outputs=outputs)
                    )

    def test_rejects_missing_model_before_loading_runtime(self) -> None:
        config = replace(
            self.onnx_policy_config,
            model_path=self.model_path.with_name('missing.onnx'),
        )
        with self.assertRaisesRegex(PolicyLoadError, 'does not exist'):
            OnnxPolicy(config, ort_module=object())

    def test_missing_onnx_runtime_fails_closed(self) -> None:
        original_import = builtins.__import__

        def import_without_onnxruntime(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ModuleNotFoundError('onnxruntime is unavailable')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=import_without_onnxruntime):
            with self.assertRaisesRegex(PolicyLoadError, 'requires onnxruntime'):
                OnnxPolicy(self.onnx_policy_config)

    def test_fails_closed_on_runtime_error_or_invalid_output(self) -> None:
        cases = (
            (
                FakeSession(run_error=RuntimeError('runtime failed')),
                PolicyInferenceError,
                'inference failed',
            ),
            (
                FakeSession(action=np.zeros(12, dtype=np.float32)),
                PolicyContractError,
                'output must have shape',
            ),
            (
                FakeSession(action=np.zeros((1, 12), dtype=np.float64)),
                PolicyContractError,
                'output must have dtype float32',
            ),
            (
                FakeSession(
                    action=np.full((1, 12), np.nan, dtype=np.float32)
                ),
                PolicyContractError,
                'only finite values',
            ),
        )
        observation = np.zeros(270, dtype=np.float32)
        for session, error_type, message in cases:
            with self.subTest(message=message):
                policy, _ = self.make_onnx_policy(session)
                with self.assertRaisesRegex(error_type, message):
                    policy.infer(observation)


class PolicyExecutorTest(RealPolicyTestCase):
    def make_backend(
        self,
        action: np.ndarray | None = None,
        error: Exception | None = None,
    ) -> RecordingPolicy:
        if action is None:
            action = np.zeros(12, dtype=np.float32)
        return RecordingPolicy(270, 12, action, error)

    def make_executor(
        self,
        backend: RecordingPolicy,
        *,
        policy_config=None,
        max_inference_time_s: float = 1.0,
        clock=None,
    ) -> PolicyExecutor:
        kwargs = {'max_inference_time_s': max_inference_time_s}
        if clock is not None:
            kwargs['clock'] = clock
        return PolicyExecutor(
            backend,
            policy_config or self.config.policy,
            **kwargs,
        )

    def test_clips_observation_and_action_without_mutating_inputs(self) -> None:
        policy_config = replace(
            self.config.policy,
            clip_observations=1.0,
            clip_actions=0.5,
        )
        raw_action = np.linspace(-2.0, 2.0, 12, dtype=np.float32)
        original_action = raw_action.copy()
        backend = self.make_backend(raw_action)
        executor = self.make_executor(backend, policy_config=policy_config)
        observation = np.linspace(-2.0, 2.0, 270, dtype=np.float32)
        original_observation = observation.copy()

        action = executor.infer(observation)

        assert backend.last_observation is not None
        self.assertLessEqual(float(np.max(np.abs(backend.last_observation))), 1.0)
        self.assertLessEqual(float(np.max(np.abs(action))), 0.5)
        np.testing.assert_array_equal(observation, original_observation)
        np.testing.assert_array_equal(raw_action, original_action)
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(action.flags.c_contiguous)

    def test_rejects_invalid_observations_before_backend_call(self) -> None:
        cases = (
            (np.zeros(269, dtype=np.float32), r'shape \(270,\)'),
            (np.zeros(270, dtype=np.float64), 'dtype float32'),
            (np.full(270, np.inf, dtype=np.float32), 'only finite values'),
        )
        for observation, message in cases:
            with self.subTest(message=message):
                backend = self.make_backend()
                executor = self.make_executor(backend)
                with self.assertRaisesRegex(PolicyContractError, message):
                    executor.infer(observation)
                self.assertIsNone(backend.last_observation)

    def test_rejects_invalid_actions(self) -> None:
        cases = (
            (np.zeros(11, dtype=np.float32), r'shape \(12,\)'),
            (np.zeros(12, dtype=np.float64), 'dtype float32'),
            (np.full(12, np.nan, dtype=np.float32), 'only finite values'),
        )
        for action, message in cases:
            with self.subTest(message=message):
                executor = self.make_executor(self.make_backend(action))
                with self.assertRaisesRegex(PolicyContractError, message):
                    executor.infer(np.zeros(270, dtype=np.float32))

    def test_rejects_late_result_and_records_inference_time(self) -> None:
        executor = self.make_executor(
            self.make_backend(),
            max_inference_time_s=0.01,
            clock=SequenceClock(10.0, 10.02),
        )

        with self.assertRaisesRegex(
            PolicyInferenceTimeoutError,
            'exceeded 0.010000s',
        ):
            executor.infer(np.zeros(270, dtype=np.float32))
        self.assertAlmostEqual(executor.last_inference_time_s, 0.02)

    def test_wraps_unexpected_backend_error(self) -> None:
        backend = self.make_backend(error=RuntimeError('backend failed'))
        executor = self.make_executor(backend)

        with self.assertRaisesRegex(
            PolicyInferenceError,
            'recording policy inference failed',
        ) as context:
            executor.infer(np.zeros(270, dtype=np.float32))
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_rejects_backend_dimension_mismatch(self) -> None:
        backend = RecordingPolicy(
            269,
            12,
            np.zeros(12, dtype=np.float32),
        )
        with self.assertRaisesRegex(
            PolicyContractError,
            'observation dimension does not match',
        ):
            self.make_executor(backend)


class RealPolicyFactoryTest(RealPolicyTestCase):
    def test_mock_factory_does_not_use_onnx_runtime(self) -> None:
        action = np.full(12, 0.25, dtype=np.float32)
        policy = load_real_policy(
            self.config,
            mock_action=action,
            ort_module=object(),
        )

        self.assertEqual(policy.backend_name, 'mock')
        np.testing.assert_array_equal(
            policy.infer(np.zeros(270, dtype=np.float32)),
            action,
        )

    def test_onnx_factory_returns_validated_executor(self) -> None:
        session = FakeSession(
            action=np.full((1, 12), 0.1, dtype=np.float32)
        )
        runtime = FakeOnnxRuntime(session)
        policy = load_real_policy(self.onnx_config, ort_module=runtime)

        self.assertEqual(policy.backend_name, 'onnx')
        np.testing.assert_allclose(
            policy.infer(np.zeros(270, dtype=np.float32)),
            0.1,
        )


if __name__ == '__main__':
    unittest.main()
