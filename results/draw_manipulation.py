"""
@Project     ：visualization
@File        ：draw_manipulation.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/20
@Last        : 2025/3/20
@Description : 
"""
import matplotlib.pyplot as plt
from utils_vis.utils_draw import draw_humanoidbench_results, draw_flam_results

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    fig_row = 2
    fig_col = 4
    fig, ax = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(13, 6))
    fig.patch.set_alpha(0)
    draw_humanoidbench_results(ax, fig_row, fig_col, task_type='manipulation')
    draw_flam_results(ax, fig_row, fig_col, task_type='manipulation')

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(labels), fontsize=12)
    plt.tight_layout()
    fig.tight_layout(pad=4.0, w_pad=0.3, h_pad=0.3)
    plt.savefig('manipulation_results_600.png', dpi=600, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.show()
