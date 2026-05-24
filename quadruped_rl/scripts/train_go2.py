import isaacgym

from quadruped_rl.envs import *
from quadruped_rl.scripts.train import train
from quadruped_rl.utils import get_args


def main():
    args = get_args()
    args.task = "go2"
    if args.experiment_name is None:
        args.experiment_name = "go2"
    if args.run_name is None:
        args.run_name = "rough"
    if args.max_iterations is None:
        args.max_iterations = 10000
    train(args)


if __name__ == "__main__":
    main()
