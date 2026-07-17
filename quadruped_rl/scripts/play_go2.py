import isaacgym

import quadruped_rl.scripts.play as play_module
from quadruped_rl.envs import *
from quadruped_rl.utils import get_args


def main():
    play_module.EXPORT_POLICY = True

    args = get_args()
    args.task = "go2"
    if args.experiment_name is None:
        args.experiment_name = "go2"
    if args.run_name is None:
        args.run_name = "rough"
    play_module.play(args)


if __name__ == "__main__":
    main()
