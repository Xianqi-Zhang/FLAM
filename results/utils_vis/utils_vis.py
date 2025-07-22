"""
@Project     ：visualization
@File        ：utils_vis.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/17
@Last        : 2025/3/17
@Description : 
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from typing import Literal


def smooth(data: Union[list, np.ndarray], window_size=1):
    """
    Moving average smoothing.
    Param:
        - data: list of ndarray or list, [[], [], ...]
        - window_size: int, size of the moving average window
    Returns:
        - Smoothed data with the same shape as input
    """
    if window_size <= 1:
        return data
    smooth_data = []
    for d in data:
        # * Do not change mode 'valid' to ['same' or 'full'].
        # * ['same', 'full'] will result in a misleading downward
        # * trend in the curve.
        # * 'valid': reduce window_size - 1 dimensions.
        # *     - input.shape: (N, )
        # *     - output.shape: (N - (window_size - 1), )
        # * The data is padded to maintain input-output dimensions.
        y = np.ones(window_size) / window_size
        pad_left = (window_size - 1) // 2
        pad_right = (window_size - 1) - pad_left
        d_padded = np.pad(d, (pad_left, pad_right), mode='edge')
        d_padded = np.convolve(d_padded, y, mode='valid')
        smooth_data.append(d_padded)
    if len(data) == 1:
        return smooth_data[0]
    return smooth_data


def setup_ax(
        ax,
        success_value: int = None,
        success_color: str = '#A9A9A9',
        fontsize: int = 12,
        title: str = 'Task',
        xticks_info: tuple = (0, 11, 2),  # * (min, max, step_size) value range.
        yticks_info: tuple = (0, 31, 5),
        xlim_info: tuple = (0, 11),
        ylim_info: tuple = (0, 31),  # * (min, max) value range.
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return',
        show_xlabel: bool = False,
        show_ylabel: bool = True,
        legend_loc: str = 'lower right',
        show_legend: bool = False
):
    if success_value is not None:
        ax.axhline(y=success_value, xmin=0, xmax=1, color=success_color, ls='--', lw=1.0)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(linestyle=':', which='major', color='#D3D3D3')
    ax.set_xticks([i for i in np.arange(*xticks_info)])
    ax.set_yticks([i for i in np.arange(*yticks_info)])
    ax.set_xlim(xlim_info)
    ax.set_ylim(ylim_info)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xlabel('', fontsize=fontsize)  # * Remove labels.
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_ylabel('', fontsize=fontsize)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=fontsize)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()  # * Close subplot legend.


def draw_line_with_multi_seeds(
        ax,
        data,
        label: str = 'Method Name',
        xlabel: str = 'Environment Steps ($\\times 10^6$)',
        ylabel: str = 'Episode Return',
        color: str = '#ABCDEF',
        marker: str = None,  # * 'o'
        linewidth: int = 2.0,
        estimator_type: Literal['mean', 'median'] = 'mean'
):
    """
    Param:
        - data: [results_with_seed_0, results_with_seed_1, ...]
    """
    if estimator_type == 'mean':
        estimator = np.mean
    elif estimator_type == 'median':
        estimator = np.median
    else:
        raise NotImplementedError
    # * errorbar='sd' !!!
    sns.lineplot(ax=ax, data=data, label=label, x=xlabel, y=ylabel, color=color, marker=marker,
                 linewidth=linewidth, estimator=estimator, errorbar='sd')


def draw_lines_with_multi_seeds(ax, inputs: list, colors: list):
    for input, color in zip(inputs, colors):
        draw_line_with_multi_seeds(ax=ax, data=input, color=color)


def draw_line_with_mean_std(
        ax,
        step,
        data_mean,
        data_std,
        label: str = 'Method Name',
        color_line: str = '#abcdef',
        color_fill: str = 'violet',
        alpha: float = 0.3,
        linewidth: int = 2.0
):
    """Used for CQN-AS results."""
    ax.plot(step, data_mean, color=color_line, label=label, lw=linewidth)
    ax.fill_between(step, data_mean - data_std, data_mean + data_std, color=color_fill,
                    alpha=alpha)
