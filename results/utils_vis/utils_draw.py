"""
@Project     ：visualization
@File        ：utils_draw.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/20
@Last        : 2025/3/20
@Description : 
"""
from typing import Literal
from utils_vis.utils_method import get_method_info
from utils_vis.utils_task import get_task_info, convert_task_name
from utils_vis.utils_vis import setup_ax, draw_line_with_mean_std, draw_line_with_multi_seeds
from utils_vis.utils_data import load_cqn_results, load_humanoidbench_results, load_flam_results, \
    load_flam_ablation_results


def draw_cqn_results(ax, fig_row, fig_col, fontsize: int = 12):
    cqn_results = load_cqn_results()
    for task_name in cqn_results.keys():
        task_info = get_task_info(task_name)
        if task_info is None:
            continue
        id = task_info['ID']
        ax_tmp = ax[id // fig_col][id % fig_col]
        for method_name in cqn_results[task_name].keys():
            if method_name == 'sac':
                # * We use SAC results from HumanoidBench.
                continue
            step = cqn_results[task_name][method_name]['step']  # * (997,)
            mean = cqn_results[task_name][method_name]['mean']
            std = cqn_results[task_name][method_name]['std']
            label, color_line, color_fill = get_method_info(method_name)
            draw_line_with_mean_std(ax_tmp, step, mean, std, label=label,
                                    color_line=color_line, color_fill=color_fill)
        # * Setup ax.
        show_xlabel = True if (id // fig_col) == (fig_row - 1) else False  # * xlabel at last line.
        show_ylabel = True if (id % fig_col) == 0 else False  # * ylabel at left.
        setup_ax(ax_tmp, success_value=task_info['success'],
                 fontsize=fontsize, title=convert_task_name(task_name),
                 xticks_info=(0, 2.1, 0.5), yticks_info=task_info['yticks'],
                 xlim_info=(0, 2), ylim_info=task_info['ylim'],
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def draw_humanoidbench_results(
        ax,
        fig_row,
        fig_col,
        task_type: Literal['locomotion', 'manipulation'] = 'locomotion',
        fontsize: int = 12
):
    humanoidbench_results = load_humanoidbench_results()
    for task_name in humanoidbench_results.keys():
        task_info = get_task_info(task_name, task_type=task_type)
        if task_info is None:
            continue
        id = task_info['ID']
        if fig_row == 1 or fig_col == 1:
            ax_tmp = ax[id]
        else:
            ax_tmp = ax[id // fig_col][id % fig_col]
        for method_name in humanoidbench_results[task_name].keys():
            # * Since the results of PPO only cover a subset of tasks,
            # * we do not use it as a baseline.
            # * HumanoidBench paper: https://arxiv.org/pdf/2403.10506
            # * Section V.B
            # * "We only run PPO on a subset of tasks (walk, kitchen, door, package)..."
            if method_name == 'PPO':
                continue
            label, color_line, _ = get_method_info(method_name)
            draw_line_with_multi_seeds(ax_tmp, data=humanoidbench_results[task_name][method_name],
                                       label=label, color=color_line)
        # * Setup ax.
        show_xlabel = True if (id // fig_col) == (fig_row - 1) else False  # * xlabel at last line.
        show_ylabel = True if (id % fig_col) == 0 else False  # * ylabel at left.
        setup_ax(ax_tmp, success_value=task_info['success'],
                 fontsize=fontsize, title=convert_task_name(task_name),
                 xticks_info=(0, 2.1, 0.5), yticks_info=task_info['yticks'],
                 xlim_info=(0, 2), ylim_info=task_info['ylim'],
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def draw_flam_results(
        ax,
        fig_row,
        fig_col,
        dir_path: str = './data_vis/flam',
        task_type: Literal['locomotion', 'manipulation'] = 'locomotion',
        fontsize: int = 12
):
    flam_results = load_flam_results(dir_path=dir_path, task_type=task_type)
    for task_name in flam_results.keys():
        if flam_results[task_name] is None:
            continue
        task_info = get_task_info(task_name, task_type=task_type)
        if task_info is None:
            continue
        id = task_info['ID']
        if fig_row == 1 or fig_col == 1:
            ax_tmp = ax[id]
        else:
            ax_tmp = ax[id // fig_col][id % fig_col]
        label, color_line, _ = get_method_info('flam')
        draw_line_with_multi_seeds(ax_tmp, data=flam_results[task_name],
                                   label=label, color=color_line)
        # * Setup ax.
        show_xlabel = True if (id // fig_col) == (fig_row - 1) else False  # * xlabel at last line.
        show_ylabel = True if (id % fig_col) == 0 else False  # * ylabel at left.
        setup_ax(ax_tmp, success_value=task_info['success'],
                 fontsize=fontsize, title=task_name,
                 xticks_info=(0, 2.1, 0.5), yticks_info=task_info['yticks'],
                 xlim_info=(0, 2), ylim_info=task_info['ylim'],
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def draw_flam_ablation_results(
        ax,
        fig_row,
        fig_col,
        dir_path: str = './data_vis/flam/ablation',
        linewidth: float = 2.5,
        fontsize: int = 12,
        draw_lambda_0: bool = True
):
    flam_ablation_results = load_flam_ablation_results(dir_path=dir_path)
    if draw_lambda_0:  # * lambda = 0.0, i.e., TD-MPC2.
        humanoidbench_results = load_humanoidbench_results()
        for task in humanoidbench_results.keys():
            for method in humanoidbench_results[task].keys():
                if method == 'TD-MPC2':
                    tmp = convert_task_name(task)
                    flam_ablation_results[tmp]['lambda_0.0'] = humanoidbench_results[task][method]

    for task_name in flam_ablation_results.keys():
        if flam_ablation_results[task_name] is None:
            continue
        task_info = get_task_info(task_name, task_type='ablation')
        if task_info is None:
            continue
        id = task_info['ID']
        if fig_row == 1 or fig_col == 1:
            ax_tmp = ax[id]
        else:
            ax_tmp = ax[id // fig_col][id % fig_col]
        for method, data in flam_ablation_results[task_name].items():
            if data is not None:
                label, color_line, _ = get_method_info(method, ablation=True)
                draw_line_with_multi_seeds(ax_tmp, data=data, label=label, color=color_line,
                                           linewidth=linewidth)
        # * Setup ax.
        show_xlabel = True if (id // fig_col) == (fig_row - 1) else False  # * xlabel at last line.
        show_ylabel = True if (id % fig_col) == 0 else False  # * ylabel at left.
        setup_ax(ax_tmp, success_value=task_info['success'],
                 fontsize=fontsize, title=task_name,
                 xticks_info=(0, 2.1, 0.5), yticks_info=task_info['yticks'],
                 xlim_info=(0, 2), ylim_info=task_info['ylim'],
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel)
