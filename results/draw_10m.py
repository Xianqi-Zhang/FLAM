"""
@Project     ：visualization
@File        ：draw_10m.py
@Author      ：Xianqi-Zhang
@Date        ：2025/6/5
@Last        : 2025/6/7
@Description :
"""
import matplotlib.pyplot as plt
from utils_vis.utils_draw import draw_cqn_results, draw_humanoidbench_results, \
    draw_flam_details_10m

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    fig_row = 1
    fig_col = 1
    fig, ax = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(10, 6))
    fig.patch.set_alpha(0)
    draw_cqn_results(ax, fig_row, fig_col, target_task='run', xlim=10, xstep=2)
    draw_humanoidbench_results(ax, fig_row, fig_col, task_type='locomotion', target_task='run', xlim=10, xstep=2)
    draw_flam_details_10m(ax, fig_row, fig_col, linewidth=2.3, fontsize=14, xlim=10, xstep=2)

    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(labels), fontsize=14)
    plt.tight_layout()
    fig.tight_layout(pad=4.0, w_pad=0.3, h_pad=0.3)
    plt.savefig('10m_300.png', dpi=300, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.show()
