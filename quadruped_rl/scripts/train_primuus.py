import isaacgym

from quadruped_rl.envs import *
from quadruped_rl.scripts.train import train
from quadruped_rl.utils import get_args


def main():
    args = get_args()
    args.task = "primuus"
    train(args)


if __name__ == "__main__":
    main()
