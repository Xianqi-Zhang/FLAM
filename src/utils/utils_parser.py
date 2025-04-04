"""
Code: https://github.com/nicklashansen/tdmpc2/blob/vectorized_env/tdmpc2/common/parser.py
"""
import os
import re
import hydra
from pathlib import Path
from termcolor import cprint
from omegaconf import OmegaConf
from gymnasium.spaces import Box, Dict
from gymnasium.vector import AsyncVectorEnv

MODEL_SIZE = {  # parameters (M)
    1: {'enc_dim': 256, 'mlp_dim': 384, 'latent_dim': 128, 'num_enc_layers': 2, 'num_q': 2},
    5: {'enc_dim': 256, 'mlp_dim': 512, 'latent_dim': 512, 'num_enc_layers': 2},
    19: {'enc_dim': 1024, 'mlp_dim': 1024, 'latent_dim': 768, 'num_enc_layers': 3},
    48: {'enc_dim': 1792, 'mlp_dim': 1792, 'latent_dim': 768, 'num_enc_layers': 4},
    317: {'enc_dim': 4096, 'mlp_dim': 4096, 'latent_dim': 1376, 'num_enc_layers': 5, 'num_q': 8},
}


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """
    cfg.work_dir = os.path.join(os.getcwd(), 'logs', cfg.task, str(cfg.seed), cfg.exp_name)
    cfg.task_title = cfg.task.replace('-', ' ').title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)  # Bin size for discrete regression
    cfg.multitask = False
    cfg.task_dim = 0
    return cfg


def update_cfg(cfg, env):
    """Update configuration with the env info."""
    if isinstance(env, AsyncVectorEnv):
        cfg.action_dim = env.single_action_space.shape[0]
        if isinstance(env.single_observation_space, Dict):  # Dict
            cfg.obs_shape = {k: v.shape for k, v in env.single_observation_space.items()}
        else:  # * Box.
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    else:
        cfg.action_dim = env.action_space.shape[0]
        try:  # * Dict.
            cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
        except:  # * Box.
            cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}

    if (
            ('obs' not in cfg.obs_shape)
            and ('state' not in cfg.obs_shape)
            and (cfg.env_obs_type in cfg.obs_shape)
    ):
        cfg.obs = cfg.env_obs_type
        cprint('Using obs: {}'.format(cfg.obs), 'green')
    try:
        cfg.episode_length = env.max_episode_steps
    except:
        cfg.episode_length = 1000
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    return cfg
