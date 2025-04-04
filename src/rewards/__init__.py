from src.configs import config_flam
from src.pkgs.RoHM.utils import dist_util
from src.rewards.stabilize_reward import StabilizeReward


def get_stablilize_reward():
    args_env, args_rohm, args_kwargs = config_flam()
    dist_util.setup_dist(args_rohm.device)
    device = dist_util.dev()
    reward = StabilizeReward(args_env, args_rohm, device)
    return reward
