"""
@Project     ：visualization
@File        ：utils_task.py
@Author      ：Xianqi-Zhang
@Date        ：2025/3/18
@Last        : 2025/3/18
@Description : 
"""
from typing import Literal

# * xticks and xlim are the same for all tasks.
# * 'xticks': (0, 2.1, 0.5), 'xlim': (0, 2)

LOCOMOTION_TASK_INFO = {
    'Walk': {'ID': 0, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Stand': {'ID': 1, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Run': {'ID': 2, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Reach': {'ID': 3, 'yticks': (0, 14100, 3500), 'ylim': (0, 14000), 'success': 12000},
    'Hurdle': {'ID': 4, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Crawl': {'ID': 5, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Maze': {'ID': 6, 'yticks': (0, 1600, 375), 'ylim': (0, 1500), 'success': 1200},
    'Sit-Hard': {'ID': 7, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Balance-Hard': {'ID': 8, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 800},
    'Stair': {'ID': 9, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
    'Slide': {'ID': 10, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
    'Pole': {'ID': 11, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
}

MANIPULATION_TASK_INFO = {
    'Door': {'ID': 0, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 375},
    'Bookshelf-Simple': {'ID': 1, 'yticks': (0, 2600, 625), 'ylim': (0, 2500), 'success': 1900},
    'Bookshelf-Hard': {'ID': 2, 'yticks': (0, 2600, 625), 'ylim': (0, 2500), 'success': 1900},
    'Spoon': {'ID': 3, 'yticks': (0, 900, 200), 'ylim': (0, 800), 'success': 650},
    'Powerlift': {'ID': 4, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 800},
    'Room': {'ID': 5, 'yticks': (0, 700, 150), 'ylim': (0, 600), 'success': 400},
    'Insert-Small': {'ID': 6, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 350},
    'Insert-Normal': {'ID': 7, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 350},
}

SCALING_FACTOR_TASK_INFO = {
    'Walk': {'ID': 0, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    # 'Stand': {'ID': 1, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    # 'Run': {'ID': 2, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Pole': {'ID': 1, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
    'Door': {'ID': 2, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 375},
    'Insert-Normal': {'ID': 3, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 350},
}

SIMILARITY_THRESHOLD_TASK_INFO = {
    'Walk': {'ID': 0, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Pole': {'ID': 1, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
    'Door': {'ID': 2, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 375},
    'Insert-Normal': {'ID': 3, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 350},
}

AMP_TASK_INFO = {
    'Walk': {'ID': 0, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
    'Pole': {'ID': 1, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 700},
    'Door': {'ID': 2, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 375},
    'Insert-Normal': {'ID': 3, 'yticks': (0, 600, 125), 'ylim': (0, 500), 'success': 350},
}

TEN_MILLION_TASK_INFO = {
    'Run': {'ID': 0, 'yticks': (0, 1100, 250), 'ylim': (0, 1000), 'success': 750},
}


def convert_task_name(task_name):
    task_name = task_name.replace('h1hand_', '')
    # * For task name with -simple/ -hard/ -normal.
    task_name = task_name.replace('_s', '-S').replace('_h', '-H').replace('_n', '-N')
    task_name = task_name[:1].upper() + task_name[1:]
    return task_name


def get_task_info(
        task_name,
        task_type: str = 'locomotion'
):
    """
    - task_type: ['locomotion', 'manipulation', 'scaling_factor', 'similarity_threshold']
    """
    task = convert_task_name(task_name)
    if task_type == 'locomotion' and task in LOCOMOTION_TASK_INFO:
        task_info = LOCOMOTION_TASK_INFO[task]
    elif task_type == 'manipulation' and task in MANIPULATION_TASK_INFO:
        task_info = MANIPULATION_TASK_INFO[task]
    elif task_type == 'scaling_factor' and task in SCALING_FACTOR_TASK_INFO:
        task_info = SCALING_FACTOR_TASK_INFO[task]
    elif task_type == 'similarity_threshold' and task in SIMILARITY_THRESHOLD_TASK_INFO:
        task_info = SIMILARITY_THRESHOLD_TASK_INFO[task]
    elif task_type == 'amp' and task in AMP_TASK_INFO:
        task_info = AMP_TASK_INFO[task]
    elif task_type == '10m' and task in TEN_MILLION_TASK_INFO:
        task_info = TEN_MILLION_TASK_INFO[task]
    else:
        task_info = None
    return task_info
