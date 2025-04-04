"""
@Project     ：FLAM
@File        ：utils_robot_data.py
@Author      ：Xianqi-Zhang
@Date        ：2024/7/20
@Last        : 2024/7/24
@Description : 
"""
import os
import math
import torch
import torch.multiprocessing as mp
# import multiprocessing
from tqdm import tqdm
from typing import Literal
from collections import defaultdict

from src.pkgs.RoHM.utils.render_util import material_body_rec_vis
from src.utils.utils_smplx import render_smplx_body_skeleton
from src.utils.utils_matching import convert_robot_camera_transform_to_smplx, \
    process_robot_data_for_smplx_render, process_robot_data_for_rohm_inference, \
    process_robot_data_for_rohm_inference_batch
from src.utils.utils_rohm_inference import rohm_inference, rohm_postprocess_output_to_joints_clip, \
    rohm_postprocess_output_to_body_pose


def render_robot_data_with_smplx(
        args_env, h1_data, smplx_model, device, frame_num=None,
        cam_x=1100, cam_y=540, H=1080, W=1920,
        material_body=material_body_rec_vis,
        save_path='./outputs/render_imgs/test',
        render_skeleton=False,
        render_ground=True
):
    """
    Render Humanoid-Bench robot data using SMPL-X model.
    Params:
        - (cam_x, cam_y):camera center pos.
            - Default: (1100, 540)
            - Only body area: (900, 300) (maybe)
        - (H, W): image size.
    """
    os.makedirs(save_path, exist_ok=True)
    print('[Info] Image save path: {}'.format(save_path))
    joints_clip, smplx_verts = process_robot_data_for_smplx_render(args_env, h1_data, smplx_model,
                                                                   device)
    frame_num = joints_clip.shape[0] if frame_num is None else frame_num  # * (clip_len,)
    for i in tqdm(range(frame_num)):
        img_name = 'frame_{}.png'.format(format(i, '03d'))
        # * Get camera pose to align the Humanoid-Bench rendered robot images.
        cam_orient = h1_data[i]['camera_orient']
        cam_pos = h1_data[i]['camera_pos']
        cam_trans = convert_robot_camera_transform_to_smplx(cam_orient, cam_pos)
        render_smplx_body_skeleton(smplx_verts[i], joints_clip[i], smplx_model,
                                   material_body=material_body,
                                   save_path=save_path, img_name=img_name,
                                   cam_x=cam_x, cam_y=cam_y, H=H, W=W, cam_trans=cam_trans,
                                   render_skeleton=render_skeleton, render_ground=render_ground)


def stabilize_robot_data_with_rohm(
        args_env, args_rohm, h1_data, smplx_model, rohm_models,
        traj_mean, traj_std, pose_mean, pose_std, device,
        smplx_init_pose=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        batch_process=False
) -> torch.Tensor:
    sequence_num = math.ceil(len(h1_data) / args_rohm.clip_len)
    if sequence_num > 1 and batch_process:
        input_data = process_robot_data_for_rohm_inference_batch(
            sequence_num, args_env, h1_data, smplx_model, device, smplx_init_pose, robot_type,
            args_rohm.clip_len
        )
    else:
        if sequence_num == 1:
            input_data = process_robot_data_for_rohm_inference(
                args_env, h1_data, smplx_model, device, smplx_init_pose, robot_type
            )
            # traj_noisy_full = input_data['motion_repr_noisy'][:, :, :rohm_info['joints_num']]
        else:
            processed_data = defaultdict(list)
            for i in range(sequence_num):
                data = h1_data[i * args_rohm.clip_len: (i + 1) * args_rohm.clip_len]
                tmp_data = process_robot_data_for_rohm_inference(
                    args_env, data, smplx_model, device, smplx_init_pose, robot_type
                )
                for key, value in tmp_data.items():
                    processed_data[key].append(value.clone())
                    del value  # * Avoid memory leak.
                del tmp_data
            input_data = {key: torch.cat(value, dim=0) for key, value in processed_data.items()}

    with torch.no_grad():
        rohm_output = rohm_inference(args_rohm, input_data, rohm_models,
                                     traj_mean, traj_std, pose_mean, pose_std,
                                     smplx_model, device)  # * [bs, 294, 1, 143]
    return rohm_output


def stabilize_robot_data_with_rohm_for_render(
        args_env, args_rohm, h1_data, smplx_model, rohm_models,
        traj_mean, traj_std, pose_mean, pose_std, device,
        smplx_init_pose=None
) -> list:
    rohm_output = stabilize_robot_data_with_rohm(
        args_env, args_rohm, h1_data, smplx_model, rohm_models,
        traj_mean, traj_std, pose_mean, pose_std, device,
        smplx_init_pose
    )
    rendered_data = []
    # * joints_clip.shape: (clip_len, 22, 3)
    # * smplx_verts.shape: (clip_len, 10475, 3)
    joints_clip, smplx_verts = rohm_postprocess_output_to_joints_clip(
        rohm_output, smplx_model, pose_std, pose_mean, device=device, raw_output=True
    )  # * Do not set idx.
    rendered_data.append((joints_clip, smplx_verts))
    return rendered_data


def stabilize_robot_data_with_rohm_for_pose(
        args_env, args_rohm, h1_data, smplx_model, rohm_models,
        traj_mean, traj_std, pose_mean, pose_std, device,
        smplx_init_pose=None
) -> torch.Tensor:
    rohm_output = stabilize_robot_data_with_rohm(
        args_env, args_rohm, h1_data, smplx_model, rohm_models,
        traj_mean, traj_std, pose_mean, pose_std, device,
        smplx_init_pose
    )
    body_pose = rohm_postprocess_output_to_body_pose(
        rohm_output, pose_std, pose_mean, device=device, raw_output=True
    )  # * [bs*T, 21, 3]
    return body_pose
