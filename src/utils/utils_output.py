"""
@Project     ：FLAM
@File        ：utils_output.py
@Author      ：Xianqi-Zhang
@Date        ：2024/8/23
@Last        : 2024/8/23
@Description : 
"""
import torch
from typing import Literal
from src.utils.utils_matching import convert_body_qpos_to_robot_qpos


def convert_policy_outputs_to_actions(
        input_raw=None,
        input_spo=None,
        input_ipo=None,
        input_type: Literal['raw', 'spo', 'ipo', 'mix'] = 'raw',
):
    assert (input_raw is not None) or (input_spo is not None) or (input_ipo is not None)
    if input_type == 'raw':
        actions = input_raw
        input_shape = input_raw.shape
    elif input_type == 'spo':
        actions = convert_body_qpos_to_robot_qpos(input_spo)
        input_shape = input_spo.shape
    elif input_type == 'ipo':
        # * TODO: InteractPolicy output postprocess.
        raise NotImplementedError
    elif input_type == 'mix':
        # * TODO: Mix of spo, and ipo.
        raise NotImplementedError
    else:
        raise ValueError('Invalid input type. Expected "raw", "spo", "ipo", or "mix".')

    actions = actions.reshape(-1, actions.shape[-1])
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    return actions, input_shape
