"""
@Project     ：FLAM
@File        ：utils_rohm_inference.py
@Author      ：Xianqi-Zhang
@Date        ：2024/6/15
@Last        : 2024/7/23
@Description : High-level utils functions for RoHM inference.
"""
import torch
import numpy as np
from typing import Literal

from src.pkgs.RoHM.utils.other_utils import REPR_LIST
from src.pkgs.RoHM.utils.konia_transform import rotation_matrix_to_angle_axis
from src.pkgs.RoHM.data_loaders.common.quaternion import rot6d_to_rotmat
from src.pkgs.RoHM.data_loaders.motion_representation import recover_from_repr_smpl, \
    get_repr_smplx, convert_repr_dict_to_smplx_params_dict
from src.utils.utils_smplx import convert_smplx_clip_to_repr, convert_smplx_clip_to_params_dict
from src.utils.utils_rohm import rohm_load_mean_std, rohm_reconstruct_traj, \
    rohm_traj_to_motion_repr, rohm_filter_items, rohm_add_noise_to_smplx_params, \
    rohm_generate_joints_position


# * ------------------------------------------------------ * #
# * ------------------- RoHM Inference ------------------- * #
# * ------------------------------------------------------ * #

def rohm_preprocess_input(
        joints_clip,
        smplx_clip,
        smplx_model,
        device=None,
        mean=None,
        std=None,
        pose_feat_dim=272,
        noise=None
) -> dict:
    """
    Process joints_clip and smplx_clip for RoHM inference.
    Params:
        joint_clip: shape [clip_len, 22, 3]
        smplx_clip: shape [clip_len, 178]
    """
    repr_dict, _, cano_smplx_params = convert_smplx_clip_to_repr(joints_clip, smplx_clip,
                                                                 smplx_model, device)

    # * Organize data.
    data = {}
    data['motion_repr_clean'] = np.concatenate([repr_dict[key] for key in REPR_LIST], axis=-1)

    # * Add noise.
    if noise is not None:
        cano_smplx_params_noisy = rohm_add_noise_to_smplx_params(cano_smplx_params.copy(), noise)
        cano_positions_noisy = rohm_generate_joints_position(cano_smplx_params_noisy, smplx_model,
                                                             device)
        repr_dict_noisy = get_repr_smplx(positions=cano_positions_noisy,
                                         smplx_params_dict=cano_smplx_params_noisy,
                                         feet_vel_thre=5e-5)  # * A dict of reprs.
        data['motion_repr_noisy'] = np.concatenate([repr_dict_noisy[key] for key in REPR_LIST],
                                                   axis=-1)
        # * dataloader_amass.py: line-344
        # * if self.task == 'pose':  # PoseNet conditioned on clean traj input
        # data['motion_repr_noisy'][:, :traj_feat_dim] = data['motion_repr_clean'][:, :traj_feat_dim]
    else:
        data['motion_repr_noisy'] = data['motion_repr_clean'].copy()  # * [clip_len-1, 263]

    if mean is None or std is None:
        mean, std, _, _ = rohm_load_mean_std()  # * np.ndarray
    data['motion_repr_clean'] = (data['motion_repr_clean'] - mean) / std
    data['motion_repr_noisy'] = (data['motion_repr_noisy'] - mean) / std

    # * cond and traj_cond.
    temp = data['motion_repr_noisy']
    noisy_traj = np.concatenate([temp[..., [0]],
                                 temp[..., 2:4],
                                 temp[..., [6]],
                                 temp[..., 7:13],
                                 temp[..., 16:19]], axis=-1)  # * [144, 13]
    data['traj_cond'] = noisy_traj  # condition of TrajNet: noisy trajectory
    # * PoseControl signal: clean local pose features
    data['traj_control_cond'] = data['motion_repr_clean'][:, -pose_feat_dim:]
    data = {k: torch.from_numpy(v).to(device).unsqueeze(0).float() for k, v in data.items()}

    return data


def rohm_process_traj_cond(
        input_data,
        posenet_output,
        traj_pose_feat_dim=272,
        traj_traj_feat_dim=13,
        mask_traj=False
):
    """
    Process control_cond in traj data for iter > 0.
    """
    # * Process control_cond.
    # * Copy local pose from PoseNet to TrajControl condition.
    # * torch.Size([bs, 144, 272]) <- torch.cat(([bs, 143, 272], [bs, 1, 272]), dim=1)
    ctl_cond_1 = posenet_output[:, :, 0].permute(0, 2, 1)[:, :, -traj_pose_feat_dim:]
    ctl_cond_2 = input_data['traj_control_cond'][:, -2].clone().unsqueeze(1)
    input_data['traj_control_cond'] = torch.cat((ctl_cond_1, ctl_cond_2), dim=1)

    if mask_traj:  # * args.iter2_cond_noisy_traj and args.infill_traj
        # * For iter>0, TrajNet conditions on noisy visible
        # * input traj and predicted traj for occluded parts
        # * from last inference iteration.
        traj_vis = input_data['traj_cond'][:, :, :traj_traj_feat_dim] * mask_traj
        traj_occ = posenet_output * (1 - mask_traj)
        input_data['traj_cond'][:, :, :traj_traj_feat_dim] = traj_vis + traj_occ

    return input_data


def rohm_process_input(input_data, trajnet_output=None, posenet_output=None, motion_repr=None):
    """
    Process input_data for different args.sample_iter.
    The first iteration is different from others.
    Iteration > 0 based previous prediction.
    """
    if (trajnet_output is not None) and (posenet_output is not None) and (motion_repr is not None):
        input_data['motion_repr_noisy'] = input_data['traj_motion_repr_noisy']
        # * Update.
        input_data['traj_motion_repr_noisy'] = motion_repr
        input_data['traj_cond'] = trajnet_output
        input_data['pose_cond'] = posenet_output[:, :, 0].permute(0, 2, 1)
        input_data = rohm_process_traj_cond(input_data, posenet_output)
    else:
        # * For iter_idx == 0.
        input_data['traj_motion_repr_noisy'] = input_data['motion_repr_noisy'].clone()
        input_data['pose_motion_repr_noisy'] = input_data['motion_repr_noisy'][:, :-1].clone()
        input_data['pose_cond'] = input_data['pose_motion_repr_noisy'].clone()

    return input_data


def rohm_process_smplx_clip_for_inference(smplx_clip: torch.Tensor, smplx_model, device) -> dict:
    """
    Process smplx_clip for RoHM inference.

    Params:
        - smplx_clip: info with body_pose, torch.Tensor or np.ndarray
    Returns:
        - input_data: list
    """
    body_params = convert_smplx_clip_to_params_dict(smplx_clip, device)
    smplx_output = smplx_model(**body_params)
    joints_clip = smplx_output.joints[:, 0:22]  # Joint positions [bs, 22, 3].
    input_data = rohm_preprocess_input(joints_clip.cpu().numpy(), smplx_clip.cpu().numpy(),
                                       smplx_model, device)
    return input_data


def _rohm_inference(
        args,
        input_data,
        trajnet,
        traj_diffusion,
        posenet,
        pose_diffusion,
        traj_mean,
        traj_std,
        pose_mean,
        pose_std,
        smplx_model,
        device,
        traj_traj_feat_dim=13
):
    # * ----------- Trajectory network forward ----------- * #
    input_data['motion_repr_noisy'] = input_data['traj_motion_repr_noisy'].clone()
    _, trajnet_output = traj_diffusion.eval_losses(
        model=trajnet,
        batch=input_data,
        shape=list(input_data['motion_repr_noisy'][:, :, :traj_traj_feat_dim].shape),
        progress=False,
        clip_denoised=False,
        timestep_respacing=args.timestep_respacing_eval,
        cond_fn_with_grad=args.cond_fn_with_grad,
        compute_loss=False,
        smplx_model=smplx_model
    )

    motion_repr_root_rec = rohm_traj_to_motion_repr(trajnet_output,  # * [bs, 144, 294]
                                                    input_data['motion_repr_clean'].shape,
                                                    traj_traj_feat_dim,
                                                    device,
                                                    args.repr_abs_only)
    traj_rec_full = rohm_reconstruct_traj(device, smplx_model, motion_repr_root_rec,
                                          traj_mean, traj_std, pose_mean, pose_std)

    # * -------------------------------------------------------
    # * Replace condition traj with denoised output from traj network.
    input_data['pose_cond'][:, :, :22] = traj_rec_full
    # * Select one to execute. [torch.permute().unsqueeze() or rohm_mask_pose_cond]
    input_data['pose_cond'] = torch.permute(input_data['pose_cond'], (0, 2, 1)).unsqueeze(-2)
    # input_data = rohm_mask_pose_cond(args, input_data, bs, clip_len, pose_traj_feat_dim)
    input_data['motion_repr_noisy'] = input_data['pose_motion_repr_noisy'].clone()
    # * -------------------------------------------------------

    # * ----------------- PoseNet forward ----------------- * #
    bs, clip_len, body_feat_dim = input_data['pose_motion_repr_noisy'].shape  # * clip_len=143
    pose_shape = [bs, body_feat_dim, 1, clip_len]
    _, posenet_output = pose_diffusion.eval_losses(
        model=posenet,
        batch=input_data,
        shape=pose_shape,
        progress=False,  # * True for tqdm.
        clip_denoised=False,
        timestep_respacing=args.timestep_respacing_eval,
        cond_fn_with_grad=args.cond_fn_with_grad,
        early_stop=args.early_stop,
        compute_loss=False,
        grad_type='amass',
        smplx_model=smplx_model
    )

    return input_data, trajnet_output, posenet_output, motion_repr_root_rec


def rohm_inference(
        args,
        input_data,
        rohm_models,
        traj_mean,
        traj_std,
        pose_mean,
        pose_std,
        smplx_model,
        device,
        traj_traj_feat_dim=13
):
    trajnet_output = None
    posenet_output = None
    motion_repr = None
    for iter_idx in range(args.sample_iter):
        # * Select models.
        pose_diffusion = rohm_models['diffusion_posenet_eval']
        posenet = rohm_models['posenet']
        if iter_idx == 0:  # * Vanilla TrajNet.
            traj_diffusion = rohm_models['diffusion_trajnet_eval']
            trajnet = rohm_models['trajnet']
        else:  # * TrajNet with TrajControl.
            traj_diffusion = rohm_models['diffusion_trajnet_control_eval']
            trajnet = rohm_models['trajnet_control']

        # * Process input_data.
        input_data = rohm_process_input(input_data, trajnet_output, posenet_output, motion_repr)
        # * Inference.
        (input_data, trajnet_output, posenet_output, motion_repr) = _rohm_inference(
            args, input_data, trajnet, traj_diffusion, posenet, pose_diffusion,
            traj_mean, traj_std, pose_mean, pose_std, smplx_model, device, traj_traj_feat_dim
        )

    return posenet_output


def rohm_postprocess_output_to_motion_repr(rohm_output, pose_std, pose_mean, raw_output=False):
    if raw_output:
        motion_repr = rohm_output[:, :, 0].permute(0, 2, 1)
        motion_repr = motion_repr * pose_std + pose_mean
    else:
        motion_repr = rohm_output
    return motion_repr


def rohm_postprocess_output_to_joints_clip(
        rohm_output,
        smplx_model,
        pose_std,
        pose_mean,
        idx=None,
        device=None,
        recover_mode: Literal['joint_abs_traj', 'joint_rel_traj', 'smplx_params'] = 'smplx_params',
        raw_output=False
):
    motion_repr = rohm_postprocess_output_to_motion_repr(rohm_output, pose_std, pose_mean,
                                                         raw_output)
    repr_dict = rohm_filter_items(motion_repr, idx, device)
    joints_clip, smplx_verts = recover_from_repr_smpl(repr_dict, smplx_model, recover_mode, True,
                                                      to_numpy=True)
    joints_clip = joints_clip.squeeze(0)
    smplx_verts = smplx_verts.squeeze(0)

    return joints_clip, smplx_verts


def rohm_postprocess_output_to_smplx_params_dict(
        rohm_output,
        pose_std,
        pose_mean,
        idx=None,
        device=None,
        raw_output=False
):
    motion_repr = rohm_postprocess_output_to_motion_repr(rohm_output, pose_std, pose_mean,
                                                         raw_output)
    repr_dict = rohm_filter_items(motion_repr, idx, device)
    smplx_params_dict = convert_repr_dict_to_smplx_params_dict(repr_dict)
    return smplx_params_dict


def rohm_postprocess_output_to_body_pose(
        rohm_output,
        pose_std,
        pose_mean,
        idx=None,
        device=None,
        raw_output=False
):
    motion_repr = rohm_postprocess_output_to_motion_repr(rohm_output, pose_std, pose_mean,
                                                         raw_output)
    repr_dict = rohm_filter_items(motion_repr, idx, device)
    body_pose_mat = rot6d_to_rotmat(repr_dict['smplx_body_pose_6d'].reshape(-1, 6))
    body_pose = rotation_matrix_to_angle_axis(body_pose_mat).reshape(-1, 21, 3)  # * [bs*T, 21, 3]
    # body_pose = body_pose.reshape(-1, 63)
    return body_pose
