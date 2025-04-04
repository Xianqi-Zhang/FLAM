"""
@Project     ：FLAM
@File        ：layers.py
@Author      ：Xianqi-Zhang
@Date        ：2024/10/25
@Last        : 2024/10/25
@Description :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble


class Ensemble(nn.Module):
    """
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, modules: list, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        # * https://pytorch.ac.cn/tutorials/intermediate/ensembling.html
        # * https://pytorch.org/functorch/stable/generated/functorch.combine_state_for_ensemble.html
        fmodel, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fmodel, in_dims=(0, 0, None), randomness='different', **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])

    def forward(self, *args, **kwargs):
        return self.vmap(list(self.params), (), *args, **kwargs)


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 - 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-eps, eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[: h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim  # * 8

    def forward(self, x):
        shp = x.shape  # * n, c, h, w
        x = x.view(*shp[:-1], -1, self.dim)  # * n, c, h*(w/self.dim), self.dim
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """
    # assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out: dict = None):
    """
    Returns a dictionary of encoders for each observation in the dict.
    https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
    """
    out = {} if out is None else out
    for k in cfg.obs_shape.keys():
        if k == 'rgb':
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))  # * !!!
        elif (k in ['state', cfg.obs, cfg.env_obs_type]):
            # print(cfg.obs_shape)
            # * {'camera_orient': [1, 9], 'camera_pos': [1, 3], 'proprio': [138],
            # * 'global_orient': [23, 4], 'transl': [23, 3], 'privileged': [151]}
            if (k not in cfg.obs_shape) or (cfg.obs_shape[k] is None):
                continue
            out[k] = mlp(
                cfg.obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        else:
            # raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
            pass  # * For other info, e.g., 'camera_orient'.
    return nn.ModuleDict(out)
