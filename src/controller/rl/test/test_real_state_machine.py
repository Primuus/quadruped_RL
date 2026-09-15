from __future__ import annotations

from pathlib import Path
import sys
import unittest


RL_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_PACKAGE_ROOT))

from rl_controller.real.state_machine import (
    RealRuntimeStateMachine,
    RuntimeState,
    RuntimeTransitionError,
)


class FakeClock:
    def __init__(self, timestamp: float = 100.0) -> None:
        self.timestamp = timestamp

    def __call__(self) -> float:
        result = self.timestamp
        self.timestamp += 0.1
        return result


class RealRuntimeStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = RealRuntimeStateMachine(clock=FakeClock())

    def test_runs_the_complete_nominal_lifecycle(self) -> None:
        self.assertIs(self.machine.state, RuntimeState.INIT)

        self.machine.mark_ready()
        self.machine.begin_move_to_default()
        self.machine.enter_rl()
        self.machine.shutdown()

        self.assertIs(self.machine.state, RuntimeState.SHUTDOWN)
        self.assertEqual(
            [transition.current for transition in self.machine.transitions],
            [
                RuntimeState.READY,
                RuntimeState.MOVE_TO_DEFAULT,
                RuntimeState.RL,
                RuntimeState.SHUTDOWN,
            ],
        )
        self.assertTrue(
            all(
                current.timestamp < following.timestamp
                for current, following in zip(
                    self.machine.transitions,
                    self.machine.transitions[1:],
                )
            )
        )

    def test_fault_is_latched_and_repeated_causes_are_recorded(self) -> None:
        self.machine.mark_ready()
        self.machine.fault('motor worker stopped')
        self.machine.fault('IMU worker stopped')

        self.assertIs(self.machine.state, RuntimeState.FAULT)
        self.assertEqual(self.machine.fault_reason, 'motor worker stopped')
        self.assertEqual(self.machine.latest_fault_reason, 'IMU worker stopped')
        self.assertEqual(
            [fault.reason for fault in self.machine.faults],
            ['motor worker stopped', 'IMU worker stopped'],
        )
        self.assertEqual(
            [transition.current for transition in self.machine.transitions],
            [RuntimeState.READY, RuntimeState.FAULT],
        )

        self.machine.shutdown('fault acknowledged')
        self.assertIs(self.machine.state, RuntimeState.SHUTDOWN)

    def test_rejects_skipped_or_reversed_transitions(self) -> None:
        with self.assertRaisesRegex(
            RuntimeTransitionError,
            'INIT -> MOVE_TO_DEFAULT',
        ):
            self.machine.begin_move_to_default()

        self.machine.mark_ready()
        with self.assertRaisesRegex(
            RuntimeTransitionError,
            'READY -> RL',
        ):
            self.machine.enter_rl()

        self.machine.shutdown()
        with self.assertRaisesRegex(
            RuntimeTransitionError,
            'after SHUTDOWN',
        ):
            self.machine.fault('late worker failure')

    def test_wait_for_reports_current_or_missing_state(self) -> None:
        self.assertTrue(self.machine.wait_for(RuntimeState.INIT, timeout=0.0))
        self.assertFalse(self.machine.wait_for(RuntimeState.READY, timeout=0.0))
        with self.assertRaisesRegex(ValueError, 'at least one RuntimeState'):
            self.machine.wait_for((), timeout=0.0)


if __name__ == '__main__':
    unittest.main()
