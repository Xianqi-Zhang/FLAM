"""
@Project     ：FLAM
@File        ：stabilize_reward.py
@Author      ：Xianqi-Zhang
@Date        ：2024/12/24
@Last        : 2024/12/24
@Description : 
"""
import torch
from typing import Literal
from src.utils.utils_rohm import rohm_load_rohm_params
from src.utils.utils_matching import init_smplx_body_pose
from src.rewards.utils_stabilize_rewards import stabilize_env_obs_with_rohm, \
    process_env_obs_for_input, expand_obs


class StabilizeReward():
    """
    Generate actions to stabilize the state_i+1 based on state_i.
    This will be combined with TaskPolicy and InteractPolicy to
    generate the final execution action to complete a task.

    Input: State/States
        - Single: s_i.
        - Sequence: (..., s_i, s_i+1, ..., s_j, ...)
    Output: Action/Actions
        - Single: a_i, making state_i+1 more stable.
        - Sequence: (..., a_i, a_i+1, ..., a_j, ...)
    Training:
        - s_i+1 -> Motion Reconstruction Model -> s_rec_i+1
            - Motion Reconstruction Model (MRM, i.e., RoHM) only
              generates physical/visible plausible motions, w/o
              interaction.
        - s_i+1 -> DenoisePolicy -> a_rec
            - making a_rec -> env -> s_rec_i+1
    """

    def __init__(
            self,
            args_env,
            args_rohm,
            device,
            clip_len=None,  # * Default setting since currently do not use sequence input.
            robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
            ignore_wrist=True,
            only_body=True,
            joint_reward=0.1,
            joint_reward_threshold: int = 0.1,
            total_reward_threshold: int = 1.5,
            total_reward: int = 1.0,
            obs_rec_update_freq: int = 1,
    ):
        super().__init__()
        self.__name__ = 'stabilize_policy'
        self.args_env = args_env
        self.args_rohm = args_rohm
        self.device = device
        self.clip_len = clip_len
        self.robot_type = robot_type
        self.ignore_wrist = ignore_wrist
        self.only_body = only_body
        self.body_dim = 19 if self.ignore_wrist else 21

        if 'h1' not in self.robot_type:
            raise NotImplementedError
        self.body_dim = 19 if self.ignore_wrist else 21
        self.leg_dim = 5  # * 10 for left and right.
        self.arm_dim = 4 if self.ignore_wrist else 5  # * 10 for left and right.

        # * Parameters for reward calculation.
        self.joint_reward = joint_reward
        self.joint_reward_threshold = joint_reward_threshold
        self.total_reward_threshold = total_reward_threshold
        self.total_reward = total_reward
        self.obs_rec_update_freq = obs_rec_update_freq
        self.reward_step = 0
        self.rohm_params = rohm_load_rohm_params(self.args_rohm, self.device)
        self.joints_init_orient = init_smplx_body_pose(clip_len=self.args_rohm.clip_len - 2,
                                                       only_joints_orient=True)
        self.smplx_init_pose = init_smplx_body_pose(clip_len=self.args_rohm.clip_len,
                                                    device=device)

    def __call__(self, observations, obs_rec_update_freq: int = None):
        """
        Stabilize policy rewards, calculated by env_obs and stabilized pose
        (human motion reconstruction model outputs, i.e., rohm outputs).
        Since RoHM input must be a sequence, the reward should be
        calculated with sequence input.
        Reward: [0, 1.9] (joint_reward * joint_num, i.e., 0.1 * 19)
        """
        if obs_rec_update_freq is not None:
            self.obs_rec_update_freq = obs_rec_update_freq
        # * Reconstruction motions with RoHM models.
        if self.reward_step % self.obs_rec_update_freq == 0:
            self.obs_rec = stabilize_env_obs_with_rohm(
                observations,
                args_env=self.args_env,
                args_rohm=self.args_rohm,
                rohm_params=self.rohm_params,
                device=self.device,
                joints_init_orient=self.joints_init_orient,
                smplx_init_pose=self.smplx_init_pose,
                robot_type=self.robot_type,
                ignore_wrist=self.ignore_wrist,
                only_body=self.only_body
            )
            # * Not self.clip_len!!!
            self.obs_rec = expand_obs(
                self.obs_rec, target_dim=self.args_rohm.clip_len, clip_len=self.args_rohm.clip_len
            )  # * (bs * clip_len, 19)
            self.obs_rec = self.obs_rec.reshape((-1, self.args_rohm.clip_len, self.body_dim))
            self.reward_step = 0
        else:
            self.reward_step += 1

        # * Obs preprocessing.
        inputs, obs_src = process_env_obs_for_input(
            observations, self.args_env, self.device, self.only_body, self.robot_type,
            self.ignore_wrist, self.args_rohm.clip_len, self.body_dim
        )  # * (bs * clip_len, 19 * 9), (bs * clip_len, 19)
        # obs_src = obs_src.reshape((-1, self.args_rohm.clip_len, self.body_dim))

        # * Rewards.
        # rewards = F.cosine_similarity(obs_src, self.obs_rec)
        rewards = torch.square(obs_src - self.obs_rec)
        rewards[rewards < self.joint_reward_threshold] = -self.joint_reward  # * Single joint.
        rewards[rewards > 0.0] = 0.0
        rewards = torch.sum(-rewards, -1)
        # * The reward is set only when the number of matching joints exceeds a certain value,
        # * e.g., joint_reward=0.1, total_reward_threshold=1.0 means 10 joints matching is needed.
        rewards[rewards < self.total_reward_threshold] = 0.0
        # rewards[rewards >= self.total_reward_threshold] = self.total_reward

        return rewards, inputs, obs_src, self.obs_rec
