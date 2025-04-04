"""
@Project     ：FLAM
@File        ：utils.py
@Author      ：Xianqi-Zhang
@Date        ：2024/5/20
@Last        : 2024/5/20
@Description : Utils functions for FLAM.
"""
import cv2
import torch
import random
import numpy as np


def setup_seed(seed, torch_deterministic=True):
    """Setup seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = torch_deterministic
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True


def save_img(img_data, img_name='img.png'):
    rgb_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name, rgb_img)
