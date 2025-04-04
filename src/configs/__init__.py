"""
@Project     ：FLAM
@File        ：__init__.py
@Author      ：Xianqi-Zhang
@Date        ：2024/8/28
@Last        : 2024/8/28
@Description :
"""
from .config_rohm import config_rohm
from .config_env import config_env, get_env_kwargs


def config_flam():
    args_env = config_env()
    args_rohm = config_rohm()
    args_kwargs = get_env_kwargs(args_env)
    return args_env, args_rohm, args_kwargs
