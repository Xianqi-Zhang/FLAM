"""
@Project     ：Visualization
@File        ：draw_scaling_factor.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/23
@Last        : 2025/3/23
@Description : 
"""

import matplotlib.pyplot as plt
from utils_vis.utils_draw import draw_flam_details_scaling_factor

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    fig_row = 1
    fig_col = 4
    fig, ax = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(13, 3.8))
    fig.patch.set_alpha(0)
    draw_flam_details_scaling_factor(ax, fig_row, fig_col, linewidth=2.3, fontsize=12)

    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(labels), fontsize=12)  # * Set legend.
    plt.tight_layout()
    fig.tight_layout(pad=4.0, w_pad=0.3, h_pad=0.3)
    plt.savefig('scaling_factor_300.png', dpi=300, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.show()
