"""
@Project     ：FLAM
@File        ：utils_env.py
@Author      ：Xianqi-Zhang
@Date        ：2024/7/29
@Last        : 2024/7/29
@Description : 
"""
import os
import cv2
import sys
import time
import torch
import random
import warnings
import numpy as np
import humanoid_bench
from humanoid_bench.env import ROBOTS, TASKS
from typing import Literal, Tuple
from termcolor import cprint
from collections import defaultdict

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.utils.utils import save_img
from src.configs.config_env import config_env, get_env_kwargs
from src.utils.utils_output import convert_policy_outputs_to_actions
from src.utils.utils_matching import convert_robot_qpos_to_body_qpos, select_target_env_obs

warnings.filterwarnings('ignore')


# * ------------------------------------------------------ * #
# * -------------------- Env Wrapper --------------------- * #
# * ------------------------------------------------------ * #
class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    Code:
    - https://github.com/carlosferrazza/humanoid-bench/blob/main/tdmpc2/tdmpc2/envs/wrappers/tensor.py
    """

    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._obs_to_tensor(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, done, truncated, info = self.env.step(action)
        info = defaultdict(float, info)
        info['success'] = float(info['success'])
        return (
            self._obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            done,
            truncated,
            info,
        )


# * ------------------------------------------------------ * #
# * --------------------- Env Create --------------------- * #
# * ------------------------------------------------------ * #
def make_env(args_env=None, idx=0, capture_video=False, setup_seed=False):
    """Make Humanoid environment."""
    args_env = config_env() if args_env is None else args_env
    kwargs = get_env_kwargs(args_env)
    env = gym.make(
        args_env.env,
        **kwargs
    )
    env = TensorWrapper(env)
    env.max_episode_steps = env.get_wrapper_attr('_max_episode_steps')
    cprint('Env max episode steps: {}'.format(env.max_episode_steps), 'green')

    if capture_video and idx == 0:
        # * Possible error during recording videos: Must be a real number, not NoneType.
        # * Solution:
        # * pip uninstall moviepy decorator
        # * pip install moviepy
        # * Reference: https://github.com/PJLab-ADG/DriveLikeAHuman/issues/11
        env = gym.wrappers.RecordVideo(env, f'videos/{args_env.run_name}')
    if setup_seed:
        # seed = idx
        seed = random.randint(0, 1000)
        env.seed(seed)
        env.action_space.seed(seed)
    # env.reset()
    return env


def make_env_thunk(args_env=None, idx=0, capture_video=False, setup_seed=True):
    def thunk():
        env = make_env(args_env, idx, capture_video, setup_seed)
        return env

    return thunk


def make_envs(
        args_env=None,
        capture_video=False,
        setup_seed=True,
        num_envs=None,
):
    args_env = config_env() if args_env is None else args_env
    num_envs = args_env.num_envs if num_envs is None else num_envs
    cprint('Env type: {}'.format(args_env.multi_env_type), 'green')
    cprint('Num envs: {}'.format(args_env.num_envs), 'green')

    if args_env.multi_env_type == 'AsyncVectorEnv':
        envs = AsyncVectorEnv([  # * Do not use gym.vector.SyncVectorEnv.
            make_env_thunk(args_env, i, capture_video, setup_seed) for i in range(num_envs)
        ])
    elif args_env.multi_env_type == 'SubprocVecEnv':
        # * Bad performance with stable-baselines3 (v2.3.2).
        envs = SubprocVecEnv([  # * Do not use gym.vector.SyncVectorEnv.
            make_env_thunk(args_env, i, capture_video, setup_seed) for i in range(num_envs)
        ])
    else:
        raise NotImplementedError

    return envs


def get_env_space_shape(args_env, env, policy_name='stabilize_policy', target_obs_name='proprio'):
    if isinstance(env, AsyncVectorEnv):
        # * For env 'HalfCheetah-v4', ppo implementation test.
        obs_space_shape = env.single_observation_space.shape
        act_space_shape = env.single_action_space.shape
        args_env.num_envs = 1
    elif isinstance(env, SubprocVecEnv) or isinstance(env, Wrapper):
        # * For env 'HumanoidBench'.
        # if policy_name == 'stabilize_policy':
        if False:
            obs_space_shape = (19,)
            act_space_shape = (19,)
        else:
            obs_space = env.observation_space
            act_space = env.action_space
            if args_env.obs_wrapper:
                obs_space = obs_space[target_obs_name]
            obs_space_shape = obs_space.shape  # * (138,)
            act_space_shape = act_space.shape  # * (61,)
    else:
        raise NotImplementedError
    return obs_space_shape, act_space_shape


def check_env_type(envs):
    if isinstance(envs, AsyncVectorEnv):
        return 'GYMVecEnv'
    elif isinstance(envs, Wrapper):
        return 'Wrapper'
    elif isinstance(envs, SubprocVecEnv):
        return 'SBVecEnv'
    else:
        # raise NotImplementedError
        return None


# * ------------------------------------------------------ * #
# * ------------------- Action Execute ------------------- * #
# * ------------------------------------------------------ * #
def execute_actions_in_env(
        input_raw=None,
        input_spo=None,
        input_ipo=None,
        input_type: Literal['raw', 'spo', 'ipo', 'mix'] = 'raw',
        args_env=None,
        env=None,
        device=None,
        save_path='./outputs/humanoid_images',
        save_image=False,
        show_render=False,
        normalize=False,
        unnormalize=False,
        postprocess=False,
        ignore_wrist=True,
        debug=False
):
    """
    Params:
        - input_type:
            - raw: actions that can be executed directly.
            - spo: StabilizePolicy outputs, only body qpos,  need to be
                   converted to actions.
            - ipo: InteractPolicy outputs, body+wrists+hands, need to
                   combine with input_spo.
        - postprocess:
            - For StabilizePolicy training.
            - Select body qpos and convert to tensor.
    """
    # * Preprocess.
    actions, input_shape = convert_policy_outputs_to_actions(
        input_raw=input_raw, input_spo=input_spo, input_ipo=input_ipo, input_type=input_type
    )
    # actions = actions.reshape(-1, actions.shape[-1])
    args_env = config_env() if args_env is None else args_env
    env = make_env(args_env) if env is None else env
    env.reset()
    if normalize:
        actions = env.task.task.normalize_action(actions)
    if unnormalize:
        actions = env.task.task.unnormalize_action(actions)
    if save_image:
        os.makedirs(save_path, exist_ok=True)

    # * Execute.
    observations = []
    rewards = []
    for step in range(actions.shape[0]):
        ob, reward, terminated, truncated, info = env.step(actions[step])
        observations.append(ob)
        rewards.append(reward)
        if show_render or save_image:
            img = env.render()  # * Render.
            if show_render and (args_env.render_mode == 'rgb_array'):
                cv2.imshow('Step', img[:, :, ::-1])
                cv2.waitKey(1)
            if save_image:
                image_name = 'frame_{}.png'.format(format(step, '03d'))
                file_path = os.path.join(save_path, image_name)
                save_img(img, file_path)
        if terminated or truncated:
            env.reset()
            if debug:
                cprint('Reset', 'yellow')

    # * Postprocess.
    if postprocess:
        observations = [
            convert_robot_qpos_to_body_qpos(
                select_target_env_obs(args_env, obs), device=device, robot_type='h1hand',
                ignore_wrist=ignore_wrist
            ) for obs in observations
        ]
        observations = torch.cat(observations).reshape(input_shape)

    return observations, rewards
