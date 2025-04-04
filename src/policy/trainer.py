import gc
import math
import torch
import numpy as np
from time import time
from typing import Literal
from tensordict.tensordict import TensorDict


class Trainer():

    def __init__(self, cfg, env, agent, buffer, logger, stabilize_reward=None):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.stabilize_reward = stabilize_reward  # * StabilizeReward.
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # print('Architecture:', self.agent.model)
        # print('Learnable parameters: {:,}'.format(self.agent.model.total_params))
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            if self.cfg.save_video:
                self.logger.video.save(self._step, key='results/video')
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))

        _process_data = lambda x: obs[x].unsqueeze(0).cpu() if x in obs else None
        td = TensorDict(
            dict(
                obs=obs[self.cfg.obs].unsqueeze(0).cpu(),  # * 'privileged'
                proprio=_process_data('proprio'),
                camera_orient=_process_data('camera_orient'),
                camera_pos=_process_data('camera_pos'),
                transl=_process_data('transl'),
                global_orient=_process_data('global_orient'),
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def update_reward(
            self,
            tds,
            trajectory_segment_length,
            reward_update_type,
            reward_merge_type: Literal['add', 'replace', 'merge'] = 'add',
            episode_length=1000
    ):
        if self.stabilize_reward is not None:
            if reward_update_type == 'normal':
                segment_num = len(tds) // trajectory_segment_length
                if segment_num >= 1:
                    data_num = segment_num * trajectory_segment_length
                    input_obs = [data.squeeze(0) for data in tds[:data_num]]
                    s_rewards, _, obs_src, obs_rec = self.stabilize_reward(input_obs)
                    s_rewards = s_rewards.cpu().reshape(data_num)
                    for i in range(len(s_rewards)):
                        tds[i]['reward'] += s_rewards[i]
            elif reward_update_type == 'padding':
                # * Padding.
                input_obs = [data.squeeze(0) for data in tds]
                segment_num = math.ceil(len(tds) / trajectory_segment_length)
                data_num = segment_num * trajectory_segment_length
                diff_num = data_num - len(tds)
                padding_data = input_obs[-1]
                for _ in range(diff_num):
                    input_obs.append(padding_data)
                s_rewards, _, obs_src, obs_rec = self.stabilize_reward(input_obs)
                s_rewards = s_rewards.cpu().reshape(data_num)
                for i in range(len(tds)):
                    average_step_reward = self.cfg.expect_return / episode_length
                    r_s = torch.tensor([s_rewards[i] * average_step_reward * self.cfg.gamma])
                    if reward_merge_type == 'add':
                        tds[i]['reward'] += r_s
                    elif reward_merge_type == 'replace':
                        tds[i]['reward'] = r_s
                    elif reward_merge_type == 'merge':
                        if tds[i]['reward'] == 0:
                            tds[i]['reward'] = r_s
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        return tds

    def train(
            self,
            trajectory_segment_length=145,
            max_update_num=None,
            reward_update_type: Literal['normal', 'padding'] = 'padding',
            reward_merge_type: Literal['add', 'replace', 'merge'] = 'add',
    ):
        """
        Train a TD-MPC2 agent.
        Params:
        - trajectory_segment_length: Used for RoHM.
            - 144 (145 - 1) for env.reset()
        """
        if max_update_num is None:
            # max_update_num = trajectory_segment_length // 2
            max_update_num = trajectory_segment_length
        pretraining_phase = True
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:
            # * Reset environment.
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False
                if self._step > 0:
                    episode_return = torch.tensor([td['reward'] for td in self._tds[1:]]).sum()
                    train_metrics.update(
                        episode_reward=episode_return,
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')

                    # * ------------------------------------------------
                    # * Update rewards.
                    self._tds = self.update_reward(
                        self._tds, trajectory_segment_length, reward_update_type, reward_merge_type
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    # * ------------------------------------------------

                    self._ep_idx = self.buffer.add(torch.cat(self._tds))  # * Add to buffer.
                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            # * Collect experience.
            tmp_step = 0
            while tmp_step < trajectory_segment_length:
                tmp_step += 1
                if self._step > self.cfg.seed_steps:
                    action = self.agent.act(obs, t0=len(self._tds) == 1)
                else:
                    action = self.env.rand_act()
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                self._tds.append(self.to_td(obs, action, reward))
                if done:
                    break

            # * Update agent.
            if self._step >= self.cfg.seed_steps:
                if pretraining_phase:
                    pretraining_phase = False
                    update_num = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    # update_num = 1
                    update_num = min(tmp_step, max_update_num)
                for _ in range(update_num):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += tmp_step

            # * Evaluate agent periodically.
            _curr = self._step // self.cfg.eval_freq
            _prev = (self._step - tmp_step) // self.cfg.eval_freq
            if _curr > _prev:
                eval_next = True

        self.logger.finish(self.agent)
