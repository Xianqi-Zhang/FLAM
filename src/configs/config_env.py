"""
@Project     ：FLAM
@File        ：config_env.py
@Author      ：Xianqi-Zhang
@Date        ：2024/7/26
@Last        : 2024/7/26
@Description : 
"""
import configargparse


def config_env():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'HumanoidBench environment configuration'
    group = configargparse.ArgParser(formatter_class=arg_formatter,
                                     config_file_parser_class=cfg_parser,
                                     description=description,
                                     prog='')
    group.add_argument('--cuda', default=True, type=bool,
                       help='If toggled, cuda will be enabled by default.')
    group.add_argument('--env', default='h1hand-walk-v0', help='e.g. h1hand-walk-v0',
                       choices=[
                           # * For PPO implementation test.
                           'HalfCheetah-v4'

                           # * Humanoid-Bench
                           # * https://humanoid-bench.github.io/
                           # * Locomotion.
                           'h1hand-walk-v0',
                           'h1hand-stand-v0',
                           'h1hand-run-v0',
                           'h1hand-reach-v0',
                           'h1hand-hurdle-v0',
                           'h1hand-crawl-v0',
                           'h1hand-maze-v0',
                           'h1hand-sit_simple-v0',
                           'h1hand-sit_hard-v0',
                           'h1hand-balance_simple-v0',
                           'h1hand-balance_hard-v0',
                           'h1hand-stair-v0',
                           'h1hand-slide-v0',
                           'h1hand-pole-v0',

                           # * Manipulation.
                           'h1hand-push-v0',
                           'h1hand-cabinet-v0',
                           'h1strong-highbar_hard-v0',
                           'h1hand-door-v0',
                           'h1hand-truck-v0',
                           'h1hand-cube-v0',
                           'h1hand-bookshelf_simple-v0',
                           'h1hand-bookshelf_hard-v0',
                           'h1hand-basketball-v0',
                           'h1hand-window-v0',
                           'h1hand-spoon-v0',
                           'h1hand-kitchen-v0',
                           'h1hand-package-v0',
                           'h1hand-powerlift-v0',
                           'h1hand-room-v0',
                           'h1hand-insert_small-v0',
                           'h1hand-insert_normal-v0',
                       ])

    group.add_argument('--max_episode_steps', default=1_000, type=int)
    group.add_argument('--keyframe', default=None)
    group.add_argument('--policy_path', default=None)
    group.add_argument('--mean_path', default=None)
    group.add_argument('--var_path', default=None)
    group.add_argument('--policy_type', default=None)
    group.add_argument('--small_obs', default=False)

    # * Settings:
    # * (1) Only proprio [robot pose + objects pose]:
    # *     - obs_wrapper = False, sensors = *
    # * (2) Proprio + robot global pose:
    # *     - obs_wrapper = True, sensors = ''
    # * (3) Proprio + robot global pose + images:
    # *     - obs_wrapper = True, sensors = 'image'
    group.add_argument('--obs_wrapper', default='True')
    group.add_argument('--sensors', default='privileged',
                       choices=['image', 'tactile', 'privileged'])

    # * NOTE: to get (nicer) 'human' rendering to work, you need to fix
    # * the compatibility issue between mujoco>3.0 and gymnasium:
    # * https://github.com/Farama-Foundation/Gymnasium/issues/749
    group.add_argument('--render_mode', default='rgb_array', choices=['human', 'rgb_array'])

    # * Misc
    group.add_argument('--wandb_project_name', default='FLAM', type=str)
    group.add_argument('--wandb_entity', default='xianqi_zhang', type=str)
    group.add_argument('--disable_wandb', default=True, type=bool)
    group.add_argument('--capture_video', default=False, type=bool,
                       help='Whether to capture videos, check out videos folder.')

    args, _ = group.parse_known_args()
    return args


def get_env_kwargs(args_env=None):
    args_env = config_env() if args_env is None else args_env
    kwargs = vars(args_env).copy()
    if 'env' in kwargs:
        kwargs.pop('env')
    if 'render_mode' in kwargs:
        kwargs.pop('render_mode')
    if ('keyframe' in kwargs) and (kwargs['keyframe'] is None):
        kwargs.pop('keyframe')
    return kwargs
