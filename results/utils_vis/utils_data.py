"""
@Project     ：visualization
@File        ：utils_data.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/18
@Last        : 2025/6/5
@Description : 
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Literal
from collections import defaultdict
from utils_vis.utils_vis import smooth
from utils_vis.utils_method import SCALING_FACTOR_METHOD_INFO, SIMILARITY_THRESHOLD_METHOD_INFO, \
    AMP_METHOD_INFO, TEN_MILLION_METHOD_INFO
from utils_vis.utils_task import LOCOMOTION_TASK_INFO, MANIPULATION_TASK_INFO, \
    SCALING_FACTOR_TASK_INFO, SIMILARITY_THRESHOLD_TASK_INFO, AMP_TASK_INFO, TEN_MILLION_TASK_INFO


def load_reproduce_results(
        method_name: Literal['cqn', 'cqn_as'] = 'cqn_as',
        dir_path: str = './data_vis/cqn_as_reproduce/',
        reproduce_tasks: tuple = ('Balance-Hard', 'Pole', 'Sit-Hard', 'Slide', 'Stair'),
):
    task_results = defaultdict(dict)
    for task in reproduce_tasks:
        files = get_target_files(os.path.join(dir_path, method_name, task))
        if len(files) != 0:
            episode_returns = []
            for file in files:
                result = pd.read_csv(file, encoding='utf-8')
                step = np.array(result['step']) // 10000 * 10000
                episode_return = np.array(result['episode_reward'])
                episode_returns.append(episode_return)
            data_mean = np.mean(np.array(episode_returns), axis=0)
            data_std = np.std(np.array(episode_returns), axis=0)
            task_results[task]['step'] = step
            task_results[task]['mean'] = data_mean
            task_results[task]['std'] = data_std
    return task_results


def load_cqn_results(
        file_path: str = './data_vis/cqn_as/humanoidbench_results.pkl',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        with_reproduce_results: bool = True
):
    """
    Source: https://github.com/younggyoseo/CQN-AS
    Tasks: Only 8 locomotion tasks from HumanoidBench
        'h1hand_stand', 'h1hand_walk', 'h1hand_run', 'h1hand_reach',
        'h1hand_hurdle', 'h1hand_crawl', 'h1hand_maze', 'h1hand_sit_simple'
    Methods:
        'cqn_as', 'cqn', 'sac'
    """
    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    # * The reproduced results only train 2M steps.
    if with_reproduce_results:
        cqn_re_results = load_reproduce_results(method_name='cqn')
        cqn_as_re_results = load_reproduce_results(method_name='cqn_as')
        for task in cqn_re_results.keys():
            if task not in results:
                results[task] = {}
            results[task]['cqn'] = cqn_re_results[task]
        for task in cqn_as_re_results.keys():
            if task not in results:
                results[task] = {}
            results[task]['cqn_as'] = cqn_as_re_results[task]

    if smooth_data:
        for task in results.keys():
            for method in results[task].keys():
                step = results[task][method]['step']  # * (997,)
                mean = results[task][method]['mean']
                std = results[task][method]['std']
                step = np.array(step) / 1000000
                mean = smooth([mean], window_size=smooth_window_size)
                std = smooth([std], window_size=smooth_window_size)
                results[task][method]['step'] = np.array(step)
                results[task][method]['mean'] = np.array(mean)
                results[task][method]['std'] = np.array(std)
    return results


def load_humanoidbench_results(
        file_path: str = './data_vis/humanoid_bench/main_results.json',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    """
    Sources: https://github.com/carlosferrazza/humanoid-bench
    Tasks:
        'balance_simple', 'balance_hard', 'basketball', 'bookshelf_simple', 'bookshelf_hard',
        'cabinet', 'crawl', 'cube', 'door', 'highbar', 'hurdle', 'insert_small', 'insert_normal',
        'kitchen', 'maze', 'package', 'pole', 'powerlift', 'push', 'reach', 'room', 'run',
        'sit_simple', 'sit_hard', 'slide', 'spoon', 'stair', 'stand', 'truck', 'walk', 'window'
    Methods:
        'DreamerV3', 'PPO', 'SAC', 'TD-MPC2'
    Seeds:
        'seed_0', 'seed_1', 'seed_2'
    For each seed, two data:
        'million_steps', 'return'
    """
    with open(file_path, 'r') as f:
        humanoidbench_results = json.load(f)

    new_results = defaultdict(dict)
    seeds = ['seed_0', 'seed_1', 'seed_2']
    for task in humanoidbench_results.keys():
        for method in humanoidbench_results[task].keys():
            steps = []
            episode_returns = []
            for seed in seeds:
                steps.append(humanoidbench_results[task][method][seed]['million_steps'])
                episode_returns.append(humanoidbench_results[task][method][seed]['return'])
            if smooth_data:
                episode_returns = smooth(episode_returns, window_size=smooth_window_size)
            data = {
                xlabel: np.array(steps).flatten(),
                ylabel: np.array(episode_returns).flatten(),
            }
            data = pd.DataFrame(data)
            new_results[task][method] = data
    return new_results


def get_target_files(data_dir, target_types: tuple = ('csv',)):
    _is_target = lambda file_name: '.' in file_name and file_name.rsplit('.')[-1] in target_types
    files = []
    for file in os.listdir(data_dir):
        files.append(file)
    data_path = [os.path.join(data_dir, name) for name in files if _is_target(name)]
    return data_path


def load_flam_results(
        dir_path: str = './data_vis/flam',
        task_type: Literal['locomotion', 'manipulation'] = 'locomotion',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    task_results = {}
    if task_type == 'locomotion':
        task_infos = LOCOMOTION_TASK_INFO
    elif task_type == 'manipulation':
        task_infos = MANIPULATION_TASK_INFO
    else:
        raise NotImplementedError
    for task in task_infos.keys():
        file_dir = os.path.join(dir_path, task_type, task)
        files = get_target_files(file_dir)
        if len(files) != 0:
            steps = []
            episode_returns = []
            for file in files:
                result = pd.read_csv(file, encoding='utf-8')
                # * Due to the trajectory segment setting in FLAM, i.e., interact N steps, then
                # * update N steps, so the number of test steps may fluctuate up and down N.
                # * Here is the processing to facilitate drawing
                step = (np.array(result['step']) // 10000 * 10000) / 1000000  # * Convert to 1M.
                episode_return = np.array(result['episode_reward'])
                if smooth_data:
                    episode_return = smooth([episode_return], window_size=smooth_window_size)
                steps.append(step)
                episode_returns.append(episode_return)
            # * Task results.
            data = {
                xlabel: np.array(steps).flatten(),
                ylabel: np.array(episode_returns).flatten(),
            }
            data = pd.DataFrame(data)
        else:
            data = None
        task_results[task] = data
    return task_results


def load_flam_results_details_scaling_factor(
        dir_path: str = './data_vis/flam/details/scaling_factor',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    task_results = defaultdict(dict)
    for task in SCALING_FACTOR_TASK_INFO.keys():
        file_dir = os.path.join(dir_path, task)
        for method in SCALING_FACTOR_METHOD_INFO .keys():
            if method == 'lambda_0.0':
                # * lambda = 0.0, i.e, TD-MPC2, the results is taken from HumanoidBench.
                continue
            files = get_target_files(os.path.join(file_dir, method))
            if len(files) != 0:
                steps = []
                episode_returns = []
                for file in files:
                    result = pd.read_csv(file, encoding='utf-8')
                    step = (np.array(result['step']) // 10000 * 10000) / 1000000
                    episode_return = np.array(result['episode_reward'])
                    if smooth_data:
                        episode_return = smooth([episode_return], window_size=smooth_window_size)
                    steps.append(step)
                    episode_returns.append(episode_return)
                # * Task results.
                data = {
                    xlabel: np.array(steps).flatten(),
                    ylabel: np.array(episode_returns).flatten(),
                }
                data = pd.DataFrame(data)
            else:
                data = None
            task_results[task][method] = data
    return task_results


def load_flam_results_details_similarity_threshold(
        dir_path: str = './data_vis/flam/details/similarity_threshold',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    task_results = defaultdict(dict)
    for task in SIMILARITY_THRESHOLD_TASK_INFO.keys():
        file_dir = os.path.join(dir_path, task)
        for method in SIMILARITY_THRESHOLD_METHOD_INFO .keys():
            files = get_target_files(os.path.join(file_dir, method))
            if len(files) != 0:
                steps = []
                episode_returns = []
                for file in files:
                    result = pd.read_csv(file, encoding='utf-8')
                    step = (np.array(result['step']) // 10000 * 10000) / 1000000
                    episode_return = np.array(result['episode_reward'])
                    if smooth_data:
                        episode_return = smooth([episode_return], window_size=smooth_window_size)
                    steps.append(step)
                    episode_returns.append(episode_return)
                # * Task results.
                data = {
                    xlabel: np.array(steps).flatten(),
                    ylabel: np.array(episode_returns).flatten(),
                }
                data = pd.DataFrame(data)
            else:
                data = None
            task_results[task][method] = data
    return task_results

def load_flam_results_details_amp(
        dir_path: str = './data_vis/flam/details/amp',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    task_results = defaultdict(dict)
    for task in AMP_TASK_INFO.keys():
        file_dir = os.path.join(dir_path, task)
        for method in AMP_METHOD_INFO .keys():
            files = get_target_files(os.path.join(file_dir, method))
            if len(files) != 0:
                steps = []
                episode_returns = []
                for file in files:
                    result = pd.read_csv(file, encoding='utf-8')
                    step = (np.array(result['step']) // 10000 * 10000) / 1000000
                    episode_return = np.array(result['episode_reward'])
                    if smooth_data:
                        episode_return = smooth([episode_return], window_size=smooth_window_size)
                    steps.append(step)
                    episode_returns.append(episode_return)
                # * Task results.
                data = {
                    xlabel: np.array(steps).flatten(),
                    ylabel: np.array(episode_returns).flatten(),
                }
                data = pd.DataFrame(data)
            else:
                data = None
            task_results[task][method] = data
    return task_results

def load_flam_results_details_10m(
        dir_path: str = './data_vis/flam/details/10m',
        smooth_data: bool = True,
        smooth_window_size: int = 10,
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return'
):
    task_results = defaultdict(dict)
    for task in TEN_MILLION_TASK_INFO.keys():
        file_dir = os.path.join(dir_path, task)
        for method in TEN_MILLION_METHOD_INFO .keys():
            files = get_target_files(os.path.join(file_dir, method))
            if len(files) != 0:
                steps = []
                episode_returns = []
                for file in files:
                    result = pd.read_csv(file, encoding='utf-8')
                    step = (np.array(result['step']) // 10000 * 10000) / 1000000
                    episode_return = np.array(result['episode_reward'])
                    if smooth_data:
                        episode_return = smooth([episode_return], window_size=smooth_window_size)
                    steps.append(step)
                    episode_returns.append(episode_return)
                # * Task results.
                data = {
                    xlabel: np.array(steps).flatten(),
                    ylabel: np.array(episode_returns).flatten(),
                }
                data = pd.DataFrame(data)
            else:
                data = None
            task_results[task][method] = data
    return task_results