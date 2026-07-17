import os

import isaacgym
from quadruped_rl import QUADRUPED_RL_ROOT_DIR
from quadruped_rl.envs import *
from quadruped_rl.utils import export_policy_as_jit_and_onnx, get_args, task_registry


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Export TorchScript as the reference and ONNX as the future C++ runtime model.
    if EXPORT_POLICY:
        path = os.path.join(QUADRUPED_RL_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        jit_path, onnx_path = export_policy_as_jit_and_onnx(ppo_runner.alg.actor_critic, path)
        print('Exported TorchScript policy to: ', jit_path)
        print('Exported ONNX policy to: ', onnx_path)

    for _ in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
