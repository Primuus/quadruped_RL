import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from isaacgym import gymapi
from isaacgym import gymutil

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return int(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    root_path = Path(root)
    if load_run in (-1, '-1'):
        if not root_path.is_dir():
            raise ValueError(f"No runs in this directory: {root_path}")
        runs = []
        for run_path in root_path.iterdir():
            if not run_path.is_dir() or run_path.name == 'exported':
                continue
            checkpoints = list(run_path.glob('model_*.pt'))
            if checkpoints:
                runs.append((max(path.stat().st_mtime_ns for path in checkpoints), run_path))
        if not runs:
            raise ValueError(f"No runs with checkpoints in this directory: {root_path}")
        run_path = max(runs, key=lambda run: (run[0], run[1].name))[1]
    else:
        run_path = root_path / str(load_run)
        if not run_path.is_dir():
            raise ValueError(f"Run directory does not exist: {run_path}")

    if checkpoint == -1:
        checkpoints = [
            path for path in run_path.glob('model_*.pt')
            if path.stem[len('model_'):].isdigit()
        ]
        if not checkpoints:
            raise ValueError(f"No checkpoints in run directory: {run_path}")
        model_path = max(checkpoints, key=lambda path: int(path.stem[len('model_'):]))
    else:
        model_path = run_path / f'model_{checkpoint}.pt'
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    return str(model_path)

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "primuus", "help": "Registered robot task (default: primuus). Robot-specific entry points select their own task."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'estimator'):
        exporter = PolicyExporterHIM(actor_critic)
        return exporter.export(path)
    elif hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        return exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
        return path


def export_policy_as_jit_and_onnx(actor_critic, path, opset_version=17):
    """Directly export the same HIM deployment module to TorchScript and ONNX."""
    if not hasattr(actor_critic, 'estimator'):
        raise ValueError('Dual TorchScript/ONNX export currently requires a HIM estimator + actor policy.')

    exporter = PolicyExporterHIM(actor_critic)
    jit_path = exporter.export(path)
    onnx_path = exporter.export_onnx(path, opset_version=opset_version)
    return jit_path, onnx_path


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        return path


class PolicyExporterHIM(torch.nn.Module):
    """导出 HIM 部署模型：输入历史观测，内部执行 estimator + actor。"""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_actor_obs = int(actor_critic.num_actor_obs)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.num_latent = int(actor_critic.estimator.num_latent)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:3 + self.num_latent]
        vel, latent = parts[..., :3], parts[..., 3:]
        latent = F.normalize(latent, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=1))

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        self.to('cpu').eval()
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        return path

    def export_onnx(self, path, opset_version=17):
        """Export ONNX directly from this estimator + actor module."""
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'ONNX export requires the onnx package. Install it with: pip install "onnx>=1.14,<2"'
            ) from exc

        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.onnx')
        self.to('cpu').eval()
        example_obs = torch.zeros(1, self.num_actor_obs, dtype=torch.float32)
        torch.onnx.export(
            self,
            example_obs,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['obs_history'],
            output_names=['actions'],
            dynamic_axes={
                'obs_history': {0: 'batch_size'},
                'actions': {0: 'batch_size'},
            },
            dynamo=False,
        )
        return path
    
