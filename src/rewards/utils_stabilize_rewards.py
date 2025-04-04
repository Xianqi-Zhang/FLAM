"""
@Project     ：FLAM
@File        ：utils_stabilize_rewards.py
@Author      ：Xianqi-Zhang
@Date        ：2024/8/26
@Last        : 2024/8/26
@Description : 
"""
import torch
from typing import Literal
from src.utils.utils_robot_data import stabilize_robot_data_with_rohm_for_pose
from src.utils.utils_matching import select_target_env_obs, convert_robot_qpos_to_body_qpos, \
    convert_smplx_joints_pose_to_robot, convert_env_obs_to_tensor


def process_env_obs_for_input(
        observations,
        args_env,
        device,
        only_body_part=True,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True,
        clip_len=145,
        body_dim=19,
        single_joint_dim=9
) -> [torch.Tensor, torch.Tensor]:
    """
    Returns:
        - inputs:  (num_envs * clip_len, 19 * 9), for StabilizePolicy training.
        - obs_src: (num_envs * clip_len, 19), for reward function.
    """
    inputs = [
        convert_env_obs_to_tensor(
            args_env, obs, device, only_body_part, robot_type, ignore_wrist
        ) for obs in observations
    ]
    inputs = torch.cat(inputs)
    inputs = inputs.reshape((-1, body_dim * single_joint_dim))
    obs_src = inputs.reshape(-1, clip_len, body_dim, single_joint_dim)[:, :, :, 0]  # * qpos part.
    return inputs, obs_src


def stabilize_env_obs_with_rohm(
        observations,
        args_env,
        args_rohm,
        rohm_params,
        device,
        joints_init_orient: torch.Tensor = None,
        smplx_init_pose: torch.Tensor = None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True,
        only_body=True
):
    rohm_smplx_pose = stabilize_robot_data_with_rohm_for_pose(
        args_env=args_env,
        args_rohm=args_rohm,
        h1_data=observations,
        smplx_model=rohm_params['smplx_model'],
        rohm_models=rohm_params['rohm_models'],
        traj_mean=rohm_params['traj_mean'],
        traj_std=rohm_params['traj_std'],
        pose_mean=rohm_params['pose_mean'],
        pose_std=rohm_params['pose_std'],
        device=device,
        smplx_init_pose=smplx_init_pose
    )  # * torch.Tensor, shape (bs * (clip_len-2), 21, 3).
    robot_pose = convert_smplx_joints_pose_to_robot(
        rohm_smplx_pose, joints_init_orient=joints_init_orient, robot_type=robot_type
    )  # * (bs * (clip_len-2), 61)
    if only_body:  # * Only select body qpos, w/o hands.
        # * (bs * (clip_len-2), 19)
        robot_pose = convert_robot_qpos_to_body_qpos(robot_pose, device, robot_type, ignore_wrist)
    return robot_pose


def expand_obs(
        input: torch.Tensor,
        target_dim=145,
        clip_len=145,
        expand_type: Literal['head', 'tail'] = 'tail',
) -> torch.Tensor:
    """
    Expand input tensor to target dimensions for RoHM outputs.
    Params:
        - input: shape (bs * (clip_len-2), 19)
    """
    batch_size = input.shape[0] // (clip_len - 2)  # * -2 is caused by RoHM inference.
    input = input.reshape(batch_size, clip_len - 2, -1)  # * (bs, clip_len-2, 19)
    if expand_type == 'head':
        output = torch.cat([input[:, :(target_dim - input.shape[1]), :], input], dim=1)
    elif expand_type == 'tail':
        output = torch.cat([input, input[:, -(target_dim - input.shape[1]):, :]], dim=1)
    else:
        raise NotImplementedError
    output = output.reshape(-1, input.shape[-1])
    return output
