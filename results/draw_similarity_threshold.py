"""
@Project     ：Visualization
@File        ：draw_similarity_threshold.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/23
@Last        : 2025/6/5
@Description : 
"""

import matplotlib.pyplot as plt
from utils_vis.utils_draw import draw_flam_details_similarity_threshold

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    fig_row = 2
    fig_col = 2
    fig, ax = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(8, 7))
    fig.patch.set_alpha(0)
    draw_flam_details_similarity_threshold(ax, fig_row, fig_col, linewidth=2.3, fontsize=15)

    lines, labels = fig.axes[0].get_legend_handles_labels()
    # * Set legend.
    fig.legend(lines, labels, loc='upper center', ncol=len(labels), fontsize=15)
    plt.tight_layout()
    fig.tight_layout(pad=4.0, w_pad=0.3, h_pad=0.3)
    plt.savefig('similarity_threshold_600.png', dpi=600, bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    plt.show()
