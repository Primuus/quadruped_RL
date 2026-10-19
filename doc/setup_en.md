# Training Environment And Server Guide

[Training Overview](../README.md) | [中文](setup_zh.md)

This guide covers this repository's Isaac Gym and HIM-PPO training stack. Deployment uses a separate environment. ROS2, Unitree SDK2 and `unitree_rl_gym` are not training dependencies. Install the bundled `rsl_rl`; do not replace it with an upstream checkout.

## Existing Server

Preferred connection:

```bash
ssh -p 2222 hurricane@10.109.69.166
```

On the server:

```bash
cd ~/quadruped_train
conda activate isaacgym
```

Run scripts from the repository root, not from the home directory.

Reference versions for reproducing the environment; inspect the active conda environment before use:

| Component | Version or location |
| --- | --- |
| Python | 3.8.10 |
| Isaac Gym | 1.0rc4, source at `~/RL/isaacgym` |
| PyTorch | 2.4.1 |
| NumPy | 1.23.2 |
| ONNX | 1.17.0 |
| TensorBoard | 2.12.0 |

These versions are not a compatibility guarantee for every machine. Source synchronization does not authorize driver, CUDA, SDK, or system-setting changes. Report device or SDK limitations before changing the underlying environment; do not upgrade packages while training is running.

## Attach The Local Packages

In an environment that already has the required dependencies:

```bash
cd ~/quadruped_train
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
```

The first command exposes `rsl_rl.env`; the second exposes `quadruped_rl`. Adding only the repository root to `PYTHONPATH` does not expose the nested `rsl_rl/rsl_rl` package correctly.

The root `setup.py` still declares `numpy==1.20`, whereas the existing server has NumPy 1.23.2. `--no-deps` preserves that environment; it neither fixes this metadata mismatch nor installs missing dependencies.

## A New Machine

Use an NVIDIA GPU with a working driver, a C/C++ toolchain and the official Isaac Gym Preview archive. Training uses Python 3.8; deployment requires Python 3.10 or later in a separate environment.

```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
```

Install PyTorch appropriate for the GPU, driver and Isaac Gym, then NumPy, ONNX, TensorBoard, matplotlib, PyYAML and ninja. Reproduce the working environment rather than selecting the latest package versions.

After extracting Isaac Gym:

```bash
export ISAACGYM_ROOT="$HOME/RL/isaacgym"
python -m pip install --no-deps -e "$ISAACGYM_ROOT/python"
cd ~/quadruped_train
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
```

Isaac Gym is not distributed with this repository. Follow the archive's local documentation for system dependencies. On a new machine, check package imports, physical responses, and training separately rather than assuming matching versions guarantee compatibility.

## Checks And Troubleshooting

Read-only checks:

```bash
which python
python -m pip show isaacgym rsl_rl quadruped_rl
nvidia-smi
```

Import Isaac Gym before PyTorch:

```bash
python -c "import isaacgym; import torch; import quadruped_rl; from rsl_rl.env import VecEnv; print(torch.__version__)"
```

Successful imports do not validate physics or training. The user can run a short training session after checking GPU availability:

```bash
python quadruped_rl/scripts/train_primuus.py --headless --max_iterations 100
```

This uses the default 4096 environments. Primuus and Go2 have independent train/play entry points; generic scripts default to Primuus. Use `train_go2.py` and `play_go2.py` for Go2.

| Symptom | Check |
| --- | --- |
| Script not found | Change to `~/quadruped_train` |
| Missing `quadruped_rl` | Editable-install the repository in the active environment |
| Missing `rsl_rl.env` | Editable-install the bundled `./rsl_rl`, not another version |
| Missing ninja | Check the active environment and `PATH` |
| Missing `libpython3.8.so` | Check conda library paths; use a command-scoped `LD_LIBRARY_PATH` if necessary |
| CUDA launch failure | Stop new launches and inspect GPU health before blaming rewards or the algorithm |
| GPU missing from `nvidia-smi` | Investigate driver, kernel and hardware state before relaunching training |

`CUDA_LAUNCH_BLOCKING=1` is a diagnostic setting for a single reproduction, not a fix for a lost GPU or a production performance setting.

## Docker

Docker and conda are independent. The Dockerfile uses `nvcr.io/nvidia/pytorch:22.12-py3`, not the package set listed above. Check Docker, the NVIDIA Container Toolkit and host drivers first.

```bash
cd ~/quadruped_train
export ISAACGYM_ROOT="$HOME/RL/isaacgym"
bash docker/build.sh
```

The repository is mounted at `/workspace/quadruped_train` and Isaac Gym at `/opt/isaacgym`. The startup script can attach the mounted Isaac Gym package but does not start training. Visual play additionally requires a working display session; SSH alone does not provide one.

## Synchronization And Artifacts

Inspect remote changes before syncing. Preview selected files with `rsync -n`; do not use blanket overwrite or `--delete` against a running training workspace.

```bash
rsync -anv -e "ssh -p 2222" README.md \
  hurricane@10.109.69.166:~/quadruped_train/README.md
```

Remove `-n` only after reviewing the preview. Ordinary source synchronization must not overwrite logs or checkpoints.

From the local deployment repository root, retrieve Primuus files under the names used by its YAML:

```bash
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.pt \
  src/controller/rl/models/primuus/policy.pt
scp -P 2222 hurricane@10.109.69.166:~/quadruped_train/logs/primuus/exported/policies/policy_1.onnx \
  src/controller/rl/models/primuus/policy.onnx
```

This overwrites the local pair. Preserve required versions and ensure both files came from the same successful export. Go2 keeps `policy_1.*` under `src/controller/rl/models/go2/`. Repeated play overwrites fixed export names, never automatically creates `policy_2`, and does not update local deployment files until they are copied. See the [training overview](../README.md) for resume and export behavior.

Primuus also mirrors `joint_dynamics.json` and the joint-loss algorithm between training and deployment. They are simulation definitions, not part of the neural-network export or hardware-identified motor parameters.

