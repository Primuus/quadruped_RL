from __future__ import annotations

import argparse
import math
import sys
import time
import weakref
from itertools import count
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from rl_controller.commands import limit_velocity_command
from rl_controller.core.config import RobotRLConfig, default_config_path
from rl_controller.core.command_reference import CommandReference
from rl_controller.core.joint_dynamics import JointDynamicsParameters, JenkinsJointFriction
from rl_controller.core.observer import ObservationHistory, RobotState, quat_rotate_inverse
from rl_controller.core.pd import compute_pd_torques, compute_torques
from rl_controller.core.policy import load_policy
from rl_controller.input import KeyboardCommandController, KeyboardInput

GRAVITY_VECTOR = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _validate_runtime_options(render_fps: float, status_interval: float, cpu_threads: int = 1) -> None:
    if not math.isfinite(render_fps) or render_fps <= 0.0:
        raise ValueError('render_fps must be finite and positive')
    if not math.isfinite(status_interval) or status_interval < 0.0:
        raise ValueError('status_interval must be finite and nonnegative')
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError('cpu_threads must be a positive integer')


def _close_viewer(viewer: mujoco.viewer.Handle) -> None:
    # MuJoCo 3.2–3.8 close() only requests exit; is_running() is already false
    # while the daemon render thread may still be destroying its GLFW window.
    # Its weak Simulate reference expires after _launch_internal finishes.
    simulate_ref = getattr(viewer, '_sim', None)
    viewer.close()
    if isinstance(simulate_ref, weakref.ReferenceType):
        deadline = time.perf_counter() + 5.0
        while simulate_ref() is not None:
            if time.perf_counter() >= deadline:
                raise RuntimeError('MuJoCo viewer did not finish closing within 5 seconds')
            time.sleep(0.005)


class MujocoRLRunner:
    def __init__(
        self,
        cfg: RobotRLConfig,
        device: str = 'cpu',
        policy_backend: str = 'auto',
        keyboard_input: KeyboardInput | None = None,
        keyboard_controller: KeyboardCommandController | None = None,
        debug_steps: int = 0,
        render_fps: float = 60.0,
        status_interval: float = 1.0,
        cpu_threads: int = 1,
    ) -> None:
        _validate_runtime_options(render_fps, status_interval, cpu_threads)
        self.cfg = cfg
        self.render_fps = render_fps
        self.status_interval = status_interval
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.model.opt.timestep = cfg.simulation_dt
        self.data = mujoco.MjData(self.model)
        self.policy = load_policy(
            cfg.policy_path, backend=policy_backend, device=device, cpu_threads=cpu_threads,
        )
        self.keyboard_input = keyboard_input
        self.keyboard_controller = keyboard_controller
        self.command_reference = CommandReference(cfg.command_transition)
        self.policy_command = np.zeros(3, dtype=np.float32)
        print(f'Policy backend: {self.policy.backend_name}')
        print(f'Policy model: {cfg.policy_path}')
        print(
            f'MuJoCo {mujoco.__version__}: physics_dt={cfg.simulation_dt:g}s '
            f'policy_dt={cfg.simulation_dt * cfg.control_decimation:g}s '
            f'render_fps<={render_fps:g} cpu_threads={cpu_threads}'
        )

        self.joint_qpos_adr = self._resolve_joint_qpos_adr()
        self.joint_qvel_adr = self._resolve_joint_qvel_adr()
        self.actuator_ids = self._resolve_actuator_ids()
        self.actuator_torque_limits = self._resolve_actuator_torque_limits()
        self.joint_friction = None
        self.passive_joint_torque = np.zeros(cfg.num_actions, dtype=np.float64)
        if cfg.simulation_joint_dynamics_path is not None:
            parameters = JointDynamicsParameters.from_json(cfg.simulation_joint_dynamics_path)
            parameters.validate_joint_names(cfg.joint_names)
            self.joint_friction = JenkinsJointFriction(
                parameters, self.data.qpos[self.joint_qpos_adr], np.clip,
            )
            self.model.dof_frictionloss[self.joint_qvel_adr] = 0.0
            self.model.dof_damping[self.joint_qvel_adr] = 0.0
            print(f'Simulation joint dynamics: {cfg.simulation_joint_dynamics_path}')

        if self.keyboard_controller is not None:
            self.command = self.keyboard_controller.current_command.copy()
        else:
            self.command = limit_velocity_command(
                cfg.command_init, cfg.command_limits, cfg.max_translation_speed, cfg.motion_limits,
            )
        self.last_action = np.zeros(cfg.num_actions, dtype=np.float32)
        self.observation_history = ObservationHistory(cfg)
        self.current_torque = np.zeros(cfg.num_actions, dtype=np.float32)
        self.target_joint_pos = cfg.default_angles.copy()
        self.is_fallen = False
        self._fall_reason: str | None = None
        self.debug_steps = max(0, int(debug_steps))
        self._debug_policy_steps = 0

        self._reset_state()

    def _resolve_joint_qpos_adr(self) -> np.ndarray:
        indices = np.zeros(self.cfg.num_actions, dtype=np.int32)
        for i, joint_name in enumerate(self.cfg.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f'Joint not found in model: {joint_name}')
            indices[i] = self.model.jnt_qposadr[joint_id]
        return indices

    def _resolve_joint_qvel_adr(self) -> np.ndarray:
        indices = np.zeros(self.cfg.num_actions, dtype=np.int32)
        for i, joint_name in enumerate(self.cfg.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f'Joint not found in model: {joint_name}')
            indices[i] = self.model.jnt_dofadr[joint_id]
        return indices

    def _resolve_actuator_ids(self) -> np.ndarray:
        indices = np.zeros(self.cfg.num_actions, dtype=np.int32)
        for i, actuator_name in enumerate(self.cfg.actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f'Actuator not found in model: {actuator_name}')
            indices[i] = actuator_id
        return indices

    def _resolve_actuator_torque_limits(self) -> np.ndarray:
        limits = np.full(self.cfg.num_actions, np.inf, dtype=np.float32)
        for i, actuator_id in enumerate(self.actuator_ids):
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                low, high = self.model.actuator_ctrlrange[actuator_id]
                limits[i] = float(max(abs(low), abs(high)))
        return limits

    def _reset_state(self) -> None:
        self.command_reference.reset()
        self.policy_command.fill(0.0)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = self.cfg.initial_base_pos
        self.data.qpos[3:7] = self.cfg.initial_base_quat
        for index, qpos_addr in enumerate(self.joint_qpos_adr):
            self.data.qpos[qpos_addr] = self.cfg.default_angles[index]
        self.last_action.fill(0.0)
        self.observation_history.reset()
        self.current_torque.fill(0.0)
        self.target_joint_pos = self.cfg.default_angles.copy()
        if self.joint_friction is not None:
            self.joint_friction.reset(self.data.qpos[self.joint_qpos_adr])
            self.passive_joint_torque.fill(0.0)
            self.data.qfrc_applied[self.joint_qvel_adr] = 0.0
        self.is_fallen = False
        self._fall_reason = None
        self._debug_policy_steps = 0
        mujoco.mj_forward(self.model, self.data)

    def _read_state(self) -> RobotState:
        base_pos = self.data.qpos[0:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_lin_vel_world = self.data.qvel[0:3].copy()
        base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel_world)
        # MuJoCo free-joint angular qvel is already expressed in the body frame.
        base_ang_vel = self.data.qvel[3:6].copy()
        joint_pos = np.array([self.data.qpos[idx] for idx in self.joint_qpos_adr], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[idx] for idx in self.joint_qvel_adr], dtype=np.float32)
        return RobotState(
            base_pos=base_pos,
            base_quat=base_quat,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def _compute_policy_output(self) -> None:
        if self.is_fallen:
            self.last_action.fill(0.0)
            self.current_torque.fill(0.0)
            return

        state = self._read_state()
        fallen, reason = self._detect_fall(state)
        if fallen:
            self._mark_fallen(reason)
            return

        self.policy_command = self.command_reference.advance(
            self.command, self.cfg.simulation_dt * self.cfg.control_decimation,
        )
        observation = self.observation_history.update(state, self.policy_command, self.last_action)
        observation = np.clip(
            observation,
            -self.cfg.clip_observations,
            self.cfg.clip_observations,
        )
        raw_action = self.policy.infer(observation)
        clipped_action = np.clip(raw_action, -self.cfg.clip_actions, self.cfg.clip_actions)
        torques, target_joint_pos = compute_torques(
            clipped_action,
            state.joint_pos,
            state.joint_vel,
            self.cfg,
            self.actuator_torque_limits,
        )
        self._print_debug_snapshot(state, raw_action, clipped_action, torques, target_joint_pos)
        self.last_action = clipped_action.astype(np.float32, copy=False)
        self.current_torque = torques
        self.target_joint_pos = target_joint_pos

    def _refresh_pd_output(self) -> None:
        if self.is_fallen:
            self.current_torque.fill(0.0)
            return
        state = self._read_state()
        self.current_torque = compute_pd_torques(
            self.target_joint_pos,
            state.joint_pos,
            state.joint_vel,
            self.cfg,
            self.actuator_torque_limits,
        )

    def _print_debug_snapshot(
        self,
        state: RobotState,
        raw_action: np.ndarray,
        clipped_action: np.ndarray,
        torques: np.ndarray,
        target_joint_pos: np.ndarray,
    ) -> None:
        if self._debug_policy_steps >= self.debug_steps:
            return

        projected_gravity = quat_rotate_inverse(state.base_quat, GRAVITY_VECTOR)
        print(f'[mujoco-rl-debug] policy_step={self._debug_policy_steps}')
        print(f'  base_z: {float(state.base_pos[2]):.6f}')
        print(f'  projected_gravity: {np.array2string(projected_gravity, precision=5, suppress_small=True)}')
        print(f'  command: {np.array2string(self.command, precision=5, suppress_small=True)}')
        print(f'  policy_command: {np.array2string(self.policy_command, precision=5, suppress_small=True)}')
        print(f'  joint_pos: {np.array2string(state.joint_pos, precision=5, suppress_small=True)}')
        print(f'  raw_action: {np.array2string(raw_action, precision=5, suppress_small=True)}')
        print(f'  clipped_action: {np.array2string(clipped_action, precision=5, suppress_small=True)}')
        print(f'  torques: {np.array2string(torques, precision=5, suppress_small=True)}')
        print(f'  target_joint_pos: {np.array2string(target_joint_pos, precision=5, suppress_small=True)}')
        print('-' * 80)
        self._debug_policy_steps += 1

    def _detect_fall(self, state: RobotState) -> tuple[bool, str]:
        projected_gravity = quat_rotate_inverse(state.base_quat, GRAVITY_VECTOR)
        reasons: list[str] = []

        if self.cfg.fall_height_threshold > 0.0 and float(state.base_pos[2]) < self.cfg.fall_height_threshold:
            reasons.append(
                f'base_z={float(state.base_pos[2]):.3f} < {self.cfg.fall_height_threshold:.3f}'
            )
        if float(projected_gravity[2]) > self.cfg.fall_gravity_z_threshold:
            reasons.append(
                f'gravity_z={float(projected_gravity[2]):.3f} > {self.cfg.fall_gravity_z_threshold:.3f}'
            )

        return (len(reasons) > 0, '; '.join(reasons))

    def _mark_fallen(self, reason: str) -> None:
        if not self.is_fallen:
            self.is_fallen = True
            self._fall_reason = reason
            print(f'Fallen detected, torque output disabled: {reason}')
        self.last_action.fill(0.0)
        self.observation_history.reset()
        self.current_torque.fill(0.0)

    def _process_keyboard_input(self) -> None:
        if self.keyboard_input is None or self.keyboard_controller is None:
            return

        if self.keyboard_input.last_error is not None:
            print(
                'keyboard input stopped; control disabled and command cleared: '
                f'{self.keyboard_input.last_error}'
            )
            self.keyboard_input = None
            self.keyboard_controller.disable()
            self.command.fill(0.0)
            return

        print_status = False
        for key in self.keyboard_input.drain_key_events():
            effect = self.keyboard_controller.apply_key(key)
            if effect.exclusive_input is not None:
                try:
                    self.keyboard_input.set_exclusive(effect.exclusive_input)
                except RuntimeError as exc:
                    self.keyboard_controller.disable()
                    self.command.fill(0.0)
                    print(
                        'keyboard control remains disabled because exclusive input failed: '
                        f'{exc}'
                    )
                    continue
            if effect.log_message:
                print(effect.log_message)
            if effect.reset_policy_state:
                self.last_action.fill(0.0)
                self.observation_history.reset()
                self.command_reference.reset()
            if effect.reset_sim:
                self._reset_state()
            print_status = print_status or effect.print_status

        snapshot = self.keyboard_input.latest_snapshot()
        self.command = self.keyboard_controller.update_pressed_keys(snapshot.pressed_keys)
        if print_status:
            print(self.keyboard_controller.describe(snapshot.pressed_keys))

    def _write_control(self) -> None:
        self.data.ctrl[:] = 0.0
        for torque, actuator_id in zip(self.current_torque, self.actuator_ids):
            self.data.ctrl[actuator_id] = float(torque)
        if self.joint_friction is not None:
            self.passive_joint_torque[:] = self.joint_friction.compute(
                self.data.qpos[self.joint_qpos_adr], self.data.qvel[self.joint_qvel_adr],
            )
            # Joint entries belong to this loss model; base external forces remain untouched.
            self.data.qfrc_applied[self.joint_qvel_adr] = self.passive_joint_torque

    def run(self, headless: bool = False) -> None:
        total_steps = (
            None if self.cfg.simulation_duration == 0.0
            else max(1, int(np.ceil(self.cfg.simulation_duration / self.cfg.simulation_dt)))
        )
        decimation = max(1, int(self.cfg.control_decimation))

        if headless:
            self._run_loop(total_steps, None, decimation)
            return

        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try:
            self._run_loop(total_steps, viewer, decimation)
        finally:
            _close_viewer(viewer)

    def _print_runtime_status(
        self, sim_elapsed: float, wall_elapsed: float, real_time_factor: float,
        viewer_syncs: int, *, finished: bool = False,
        frame_sim_max_ms: float = 0.0, frame_wall_max_ms: float = 0.0,
    ) -> None:
        state = self._read_state()
        actual = (state.base_lin_vel[0], state.base_lin_vel[1], state.base_ang_vel[2])

        def velocity(values) -> str:
            return '(' + ', '.join(f'{value:.3f}' for value in values) + ')'

        label = 'mujoco-rl:done' if finished else 'mujoco-rl'
        print(
            f'[{label}] sim={sim_elapsed:.2f}s wall={wall_elapsed:.2f}s '
            f'RTF={real_time_factor:.3f} viewer_syncs={viewer_syncs} '
            f'frame_gap_max_ms(sim/wall)={frame_sim_max_ms:.1f}/{frame_wall_max_ms:.1f} '
            f'target={velocity(self.command)} reference={velocity(self.policy_command)} '
            f'actual={velocity(actual)} [vx, vy m/s; yaw rad/s]',
            flush=True,
        )

    def _run_loop(self, total_steps: int | None, viewer: mujoco.viewer.Handle | None, decimation: int) -> None:
        wall_start = time.perf_counter()
        next_render = wall_start
        render_interval = 1.0 / self.render_fps
        next_status = wall_start + self.status_interval
        last_status_wall = wall_start
        last_status_sim = 0.0
        sim_elapsed = 0.0
        viewer_syncs = 0
        last_render_wall = None
        last_render_sim = 0.0
        window_sim_gap_ms = window_wall_gap_ms = 0.0
        total_sim_gap_ms = total_wall_gap_ms = 0.0
        steps = count() if total_steps is None else range(total_steps)
        for step in steps:
            self._process_keyboard_input()
            if step % decimation == 0:
                self._compute_policy_output()
            else:
                self._refresh_pd_output()

            self._write_control()
            physics_dt = self.model.opt.timestep
            mujoco.mj_step(self.model, self.data)
            sim_elapsed += physics_dt

            now = time.perf_counter()
            if viewer is not None and now >= next_render:
                if not viewer.is_running():
                    break
                if last_render_wall is not None:
                    sim_gap_ms = (sim_elapsed - last_render_sim) * 1000.0
                    wall_gap_ms = (now - last_render_wall) * 1000.0
                    window_sim_gap_ms = max(window_sim_gap_ms, sim_gap_ms)
                    window_wall_gap_ms = max(window_wall_gap_ms, wall_gap_ms)
                    total_sim_gap_ms = max(total_sim_gap_ms, sim_gap_ms)
                    total_wall_gap_ms = max(total_wall_gap_ms, wall_gap_ms)
                last_render_wall, last_render_sim = now, sim_elapsed
                viewer.sync()
                viewer_syncs += 1
                now = time.perf_counter()
                # Drop overdue display frames; keep physics and PD at their own rate.
                periods = math.floor((now - next_render) / render_interval) + 1
                next_render += periods * render_interval

            if self.status_interval > 0.0 and now >= next_status:
                self._print_runtime_status(
                    sim_elapsed, now - wall_start,
                    (sim_elapsed - last_status_sim) / (now - last_status_wall),
                    viewer_syncs,
                    frame_sim_max_ms=window_sim_gap_ms,
                    frame_wall_max_ms=window_wall_gap_ms,
                )
                last_status_wall, last_status_sim = now, sim_elapsed
                next_status = now + self.status_interval
                window_sim_gap_ms = window_wall_gap_ms = 0.0

            # Absolute deadlines avoid accumulating one sleep overshoot at every 0.5ms step.
            sleep_time = wall_start + sim_elapsed - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

        wall_elapsed = time.perf_counter() - wall_start
        self._print_runtime_status(
            sim_elapsed, wall_elapsed, sim_elapsed / max(wall_elapsed, 1e-9),
            viewer_syncs, finished=True,
            frame_sim_max_ms=total_sim_gap_ms, frame_wall_max_ms=total_wall_gap_ms,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a quadruped RL policy in MuJoCo.')
    parser.add_argument('--config', type=Path, default=None, help='Path to the robot YAML config.')
    parser.add_argument('--policy-path', type=Path, default=None, help='Override policy path.')
    parser.add_argument(
        '--policy-backend',
        choices=('auto', 'torchscript', 'onnx'),
        default='auto',
        help='Policy inference backend. Default: infer from the model file extension.',
    )
    parser.add_argument('--xml-path', type=Path, default=None, help='Override MuJoCo XML path.')
    parser.add_argument(
        '--duration', type=float, default=None,
        help='Override simulation duration in seconds; 0 runs until stopped.',
    )
    parser.add_argument(
        '--command',
        type=float,
        nargs=3,
        metavar=('VX', 'VY', 'WZ'),
        default=None,
        help='Override command vector [m/s, m/s, rad/s].',
    )
    parser.add_argument('--device', type=str, default='cpu', help='Policy inference device, for example cpu or cuda:0.')
    parser.add_argument('--headless', action='store_true', help='Run without the MuJoCo viewer.')
    parser.add_argument(
        '--cpu-threads', type=int, default=1,
        help='CPU inference threads for TorchScript/ONNX; 1 avoids competing with rendering. Default: 1.',
    )
    parser.add_argument(
        '--render-fps', type=float, default=60.0,
        help='Maximum viewer synchronization rate in wall-clock Hz. Default: 60.',
    )
    parser.add_argument(
        '--status-interval', type=float, default=1.0,
        help='Runtime status interval in wall-clock seconds; 0 disables periodic status. Default: 1.',
    )
    parser.add_argument(
        '--debug-steps',
        type=int,
        default=0,
        help='Print state/action/torque details for the first N policy steps. Default: 0.',
    )
    parser.add_argument(
        '--keyboard-device',
        type=Path,
        default=None,
        help='Linux keyboard event device. Default: listen to all readable keyboard devices.',
    )
    parser.add_argument(
        '--no-keyboard',
        action='store_true',
        help='Disable evdev keyboard input and use the YAML/--command fixed command.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_options(args.render_fps, args.status_interval, args.cpu_threads)
    except ValueError as exc:
        parser.error(str(exc))
    if args.no_keyboard and args.keyboard_device is not None:
        parser.error('--keyboard-device cannot be combined with --no-keyboard')
    if args.command is not None and not args.no_keyboard:
        parser.error('--command is a fixed command and requires --no-keyboard')

    config_path = args.config or default_config_path()
    cfg = RobotRLConfig.from_yaml(
        config_path,
        policy_path=args.policy_path,
        xml_path=args.xml_path,
        simulation_duration=args.duration,
        command_init=args.command,
    )

    keyboard_input = None
    keyboard_controller = None
    if not args.no_keyboard:
        keyboard_controller = KeyboardCommandController(
            command_limits=cfg.command_limits,
            max_translation_speed=cfg.max_translation_speed,
            bindings=cfg.keyboard_bindings,
            speed_levels=cfg.keyboard_speed_levels,
            default_speed_level=cfg.keyboard_default_speed_level,
            motion_limits=cfg.motion_limits,
            backward_speed_levels=cfg.keyboard_backward_speed_levels,
        )
        keyboard_input = KeyboardInput(
            args.keyboard_device,
            tracked_keys=keyboard_controller.tracked_keys,
        )
        print(f'Keyboard inputs: {len(keyboard_input.device_paths)} device(s)')
        for device_name, device_path in zip(
            keyboard_input.device_names,
            keyboard_input.device_paths,
        ):
            print(f'  - {device_name} ({device_path})')
        print('Keyboard control starts disabled; press F to enable and grab the keyboards.')
        print('Press I to disable control and release the keyboards.')
    else:
        print(f'Keyboard input disabled; fixed command: {cfg.command_init}')

    try:
        runner = MujocoRLRunner(
            cfg,
            device=args.device,
            policy_backend=args.policy_backend,
            keyboard_input=keyboard_input,
            keyboard_controller=keyboard_controller,
            debug_steps=args.debug_steps,
            render_fps=args.render_fps,
            status_interval=args.status_interval,
            cpu_threads=args.cpu_threads,
        )
        runner.run(headless=args.headless)
    finally:
        if keyboard_input is not None:
            keyboard_input.close()

if __name__ == '__main__':
    main()
