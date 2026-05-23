import isaacgym

import legged_gym.scripts.play as play_module
from legged_gym.envs import *
from legged_gym.utils import get_args


def main():
    play_module.EXPORT_POLICY = True
    play_module.RECORD_FRAMES = False
    play_module.MOVE_CAMERA = False

    args = get_args()
    args.task = "go2"
    if args.experiment_name is None:
        args.experiment_name = "go2"
    if args.run_name is None:
        args.run_name = "rough"
    play_module.play(args)


if __name__ == "__main__":
    main()
