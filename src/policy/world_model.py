"""
@Project     ：FLAM
@File        ：world_model.py
@Author      ：Xianqi-Zhang
@Date        ：2024/10/28
@Last        : 2024/10/28
@Description : 
"""
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Literal
from src.policy import layers
import src.utils.utils_math as math
import src.utils.utils_init as init


class WorldModel(nn.Module):
    """
    https://github.com/nicklashansen/tdmpc2/blob/vectorized_env/tdmpc2/common/world_model.py
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            # * Expand the action dimensions of different tasks so that they have the same length.
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.tasks[i]] = 1.0

        # * TD-MPC2, arXiv:2310.16828, Eq(6).
        # * - Encoder, z = h(s, e)
        # * - Latent dynamics, z' = d(z, a, e)
        # * - Reward, r = R(z, a, e)
        # * - Terminal value, q = Q(z, a, e)
        # * - Policy prior, a = p(z, e)
        dim_zae = cfg.latent_dim + cfg.task_dim + cfg.action_dim  # * z + e + a
        dim_ze = cfg.latent_dim + cfg.task_dim  # * z + e
        dim_mlp = 2 * [cfg.mlp_dim]  # * MLP hidden dimensions.
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(dim_zae, dim_mlp, cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(dim_zae, dim_mlp, max(cfg.num_bins, 1))
        self._Qs = layers.Ensemble([
            layers.mlp(dim_zae, dim_mlp, max(cfg.num_bins, 1), dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])
        # * chunk(2, dim=-1) as (mu, log_std) in pi().
        self._pi = layers.mlp(dim_ze, dim_mlp, 2 * cfg.action_dim)

        # * Weight initalization.
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])

        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)  # * !!!
        self.log_std_min = torch.tensor(cfg.log_std_min)  # * -10, TD-MPC2, Table 8.
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min  # * 2 - (-10)

    def mix_parameters(self, *args, **kwargs) -> list:
        encoder_lr = self.cfg.lr * self.cfg.enc_lr_scale
        param = [
            {'params': self._encoder.parameters(*args, **kwargs), 'lr': encoder_lr},
            {'params': self._dynamics.parameters(*args, **kwargs)},
            {'params': self._reward.parameters(*args, **kwargs)},
            {'params': self._Qs.parameters(*args, **kwargs)},
            {'params': self._task_emb.parameters(*args, **kwargs) if self.cfg.multitask else []}
        ]
        return param

    def policy_parameters(self, *args, **kwargs):
        return self._pi.parameters(*args, **kwargs)

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """Soft-update target Q-networks using Polyak averaging."""
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                # * https://pytorch.org/docs/stable/generated/torch.lerp.html#torch.lerp
                p_target.data.lerp_(p.data, self.cfg.tau)  # * tau: 0.01

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encoder: z = h(s, e)
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Latent dynamics: z' = d(z, a, e)
        Predicts the next latent state given the current latent state
        and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Reward: r = R(z, a, e)
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Policy prior: a = p(z, e)
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # * Gaussian policy prior.
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # * Mask out unused action dimensions.
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # * No masking.
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type: Literal['min', 'avg', 'all'] = 'min', target=False):
        """
        Terminal value: q = Q(z, a, e)
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)
        if return_type == 'all':
            return out
        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]  # * self.cfg.num_q: 5
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
