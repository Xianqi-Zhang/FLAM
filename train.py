import os
import warnings
from termcolor import cprint
from src.configs.config_env import config_env
from src.rewards import get_stablilize_reward
from src.policy.tdmpc2 import TDMPC2
from src.policy.trainer import Trainer
from src.policy.buffer import Buffer
from src.utils.utils import setup_seed
from src.utils.utils_env import make_env
from src.utils.utils_logger import Logger
from src.utils.utils_parser import parse_cfg, update_cfg
from src.configs.config_model import config_model

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_API_KEY'] = '0b25371ee8ecea1177437ad1abbe1b58e725abe5'


def train():
    args_env = config_env()
    cfg = config_model()
    cfg.task = args_env.env
    cfg = parse_cfg(cfg)
    setup_seed(cfg.seed)
    env = make_env(args_env)
    cfg = update_cfg(cfg, env)
    cprint('Work dir: {}'.format(cfg.work_dir), 'yellow')
    cprint('Seed steps: {}'.format(cfg.seed_steps), 'green')

    # * -----------------------------------------------------------
    s_reward = get_stablilize_reward()
    trainer = Trainer(cfg, env, TDMPC2(cfg), Buffer(cfg), Logger(cfg), s_reward)
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
