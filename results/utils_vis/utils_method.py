"""
@Project     ：visualization
@File        ：utils_method.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/18
@Last        : 2025/3/18
@Description : 
"""

METHOD_INFO = {
    'flam': {'name': 'FLAM (Ours)', 'color_line': '#f18b51', 'color_fill': '#ff7976'},
    'cqn': {'name': 'CQN', 'color_line': '#d3bff3', 'color_fill': '#ebe5f6'},
    'cqn_as': {'name': 'CQN-AS', 'color_line': '#c2c2c2', 'color_fill': '#d6d6d6'},
    'sac': {'name': 'SAC', 'color_line': '#c59a81', 'color_fill': '#e8d1a6'},
    'SAC': {'name': 'SAC', 'color_line': '#c59a81', 'color_fill': '#e8d1a6'},
    'TD-MPC2': {'name': 'TD-MPC2', 'color_line': '#7cc3c5', 'color_fill': '#70BAAE'},
    'DreamerV3': {'name': 'DreamerV3', 'color_line': '#f9bdd9', 'color_fill': '#eed8ea'},
}

METHOD_ABLATION_INFO = {
    # * lambda = 0.0, i.e.,  TD-MPC2.
    'lambda_0.0': {'name': '$\lambda$ = 0.0', 'color_line': '#b9dedf', 'color_fill': '#7cc3c5'},

    'lambda_0.5': {'name': '$\lambda$ = 0.5', 'color_line': '#cca38a', 'color_fill': '#b9dedf'},
    'lambda_1.0': {'name': '$\lambda$ = 1.0', 'color_line': '#ffa16b', 'color_fill': '#f5aa7f'},
    'lambda_10.0': {'name': '$\lambda$ = 10.0', 'color_line': '#e892b0', 'color_fill': '#70BAAE'},
}


def get_method_info(method_name, ablation: bool = False):
    info = METHOD_ABLATION_INFO[method_name] if ablation else METHOD_INFO[method_name]
    label = info['name']
    color_line = info['color_line']
    color_fill = info['color_fill']
    return label, color_line, color_fill
