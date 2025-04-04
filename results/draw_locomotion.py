"""
@Project     ：visualization
@File        ：draw_locomotion.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/20
@Last        : 2025/3/20
@Description : 
"""
import matplotlib.pyplot as plt
from utils_vis.utils_draw import draw_cqn_results, draw_humanoidbench_results, draw_flam_results

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    fig_row = 3
    fig_col = 4
    fig, ax = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(13, 8))
    fig.patch.set_alpha(0)
    draw_cqn_results(ax, fig_row, fig_col)
    draw_humanoidbench_results(ax, fig_row, fig_col, task_type='locomotion')
    draw_flam_results(ax, fig_row, fig_col, task_type='locomotion')

    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(labels), fontsize=12)
    plt.tight_layout()
    fig.tight_layout(pad=4.0, w_pad=0.3, h_pad=0.3)
    plt.savefig('locomotion_results_600.png', dpi=600, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.show()
