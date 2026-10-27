import isaacgym

import quadruped_rl.scripts.play as play_module
from quadruped_rl.envs import *
from quadruped_rl.utils import get_args


def main():
    play_module.EXPORT_POLICY = True

    args = get_args()
    if args.task not in ("primuus", "primuus_obstacles"):
        raise ValueError("Primuus entry point accepts --task primuus or primuus_obstacles")
    play_module.play(args)


if __name__ == "__main__":
    main()
