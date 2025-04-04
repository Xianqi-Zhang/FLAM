"""
@Project     ：FLAM
@File        ：utils_debug.py
@Author      ：Xianqi-Zhang
@Date        ：2024/6/14
@Last        : 2024/7/23
@Description : Utils functions for debugging.
"""
import os
import cv2
import glob
import torch
import pickle
import numpy as np
import gymnasium as gym
import humanoid_bench
from humanoid_bench.env import ROBOTS, TASKS
from typing import Literal
from termcolor import cprint

from src.configs import config_flam
from src.utils.utils import save_img
from src.pkgs.RoHM.utils import dist_util
from src.utils.utils_rohm import rohm_generate_dataloader, rohm_load_noise
from src.utils.utils_rohm_inference import rohm_preprocess_input
from src.utils.utils_smplx import render_smplx_body_skeleton, convert_smplx_clip_to_joints_clip, \
    cano_joints_and_smplx_clip
from src.utils.utils_matching import convert_robot_pose_to_smplx, \
    convert_smplx_joints_pose_to_robot, convert_robot_qpos_to_body_qpos
from src.utils.utils_matching import select_target_env_obs


def print_dict(data: dict, print_type: str = 'shape', color: str = 'green'):
    cprint('--' * 20, color)
    for key, value in data.items():
        v_string = value
        if print_type == 'shape':
            try:
                v_string = value.shape
            except:
                v_string = len(value)
        elif print_type == 'type':
            v_string = type(value)
        cprint('{}: {}'.format(key, v_string), color)
    cprint('--' * 20, color)


def load_test_data_amass(
        data_save_path='./outputs/test_amass.pkl',
        data_num=1,
        clip_len=145,
        joints_num=22,
        reload=False
) -> dict:
    """
    Load preprocessed AMASS data.
    (Only use one sequence for learning and testing.)

    Return:
        A dict with items:
            - 'data_num': data_num,
            - 'joints_clips': [joints_clip_0, joints_clip_1, ...],
            - 'smplx_clips': [smplx_clip_0, smplx_clip_1, ...]

    More info for dimensions:
        - joints_clip_i: shape (clip_len, 22, 3)
        - smplx_clip_i: shape (clip_len, 178)

    1.  For joint_clip_i:
        ====================== Dimension 22 ======================
        SMPL-X [https://arxiv.org/pdf/1904.05866]
        - Section 3.1
        - SMPL-X has N=10475 vertices  and K=54 joints, which includes
        - joints for neck, jaw, eyeballs and fingers.
        - The pose θ is R^{3(K+1)}, where K is the number of body joints
        - in addition to a joint for global rotation.
        - The fingers has 30 joints.
        -
        - --> So it has a total of 55 joints (54 + 1).
        -
        - RoHM [https://arxiv.org/pdf/2401.08570],
        - does not use hand pose and facial expression (Section 3).
        - Finally, without
                - 3 facial expression joints (jaw-1, leye-1, reye-1),
                - 30 hand pose joints (left_hand-15, right_hand-15),
        - it remains 22 joints (55 - 3 - 30).

        ======================= Dimension 3 =======================
        As described in SMPL-X Section 3.1,
        - 3 DoF per joint as axis-angle rotations.

    2.  For smplx_clip_i:
        ====================== Dimension 178 ======================
        According to preprocessing_amass.py in pkgs.RoHM,
        smplx params consist of 8 items [
            global_orient-3, transl-3, betas-10, body_pose-63,
            hand_pose-90, jaw_pose-3, leye_pose-3, reye_pose-3
        ]
        So, 3 + 3 + 10 + 63 + 90 + 3+ 3 + 3 =  178.
        ---
        In DatasetAMASS in dataloader_amass.py,
        - [global_orient-3, transl-3, betas-10, body_pose-63] are used,
        - [hand_pose-90, jaw_pose-3, leye_pose-3, reye_pose-3] are zero.
    """
    if reload or (not os.path.exists(data_save_path)):
        preprocessed_root = './src/pkgs/RoHM/datasets/AMASS_smplx_preprocessed'
        preprocessed_joints_dir = os.path.join(preprocessed_root, 'pose_data_fps_30')
        preprocessed_smpl_dir = os.path.join(preprocessed_root, 'smpl_data_fps_30')
        dataset_name = 'TotalCapture'
        seqs_path = glob.glob(os.path.join(preprocessed_joints_dir, dataset_name, '*/*.npy'))
        seqs_path = sorted(seqs_path)
        seqs_path = seqs_path[:data_num]

        joints_clips = []
        smplx_clips = []
        for path in seqs_path:
            seq_name = path.split('/')[-2]
            npy_name = path.split('/')[-1]
            path_joints = os.path.join(preprocessed_joints_dir, dataset_name, seq_name, npy_name)
            path_smplx = os.path.join(preprocessed_smpl_dir, dataset_name, seq_name, npy_name)
            seq_joints = np.load(path_joints)  # [seq_len, 25, 3]
            seq_smplx = np.load(path_smplx)  # [seq_len, 178]
            # # * For consistency with DatasetAMASS test split.
            # seq_joints = seq_joints[1:-1]
            # seq_smplx = seq_smplx[1:-1]

            # * Clip sequences.
            N = len(seq_joints)
            if N >= clip_len:
                num_valid_clip = int(N / clip_len)
                for i in range(num_valid_clip):
                    joints_clip = seq_joints[(clip_len * i):clip_len * (i + 1)]
                    smplx_clip = seq_smplx[(clip_len * i):clip_len * (i + 1)]
                    joints_clips.append(joints_clip[:, :joints_num, :])  # * Only use 22-dim.
                    smplx_clips.append(smplx_clip)

        save_data = {
            'data_num': data_num, 'joints_clips': joints_clips, 'smplx_clips': smplx_clips
        }
        with open(data_save_path, 'wb') as f:
            pickle.dump(save_data, f)

    # * Load data.
    with open(data_save_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_debug_data(
        args,
        smplx_model,
        device,
        add_noise=True,
        test_type: str = 'debug_inference',
        data_save_path='./outputs/test_amass.pkl',
        data_num=1,
        clip_len=145,
        joints_num=22,
        reload=False
):
    if test_type == 'debug_input':
        # * Do not process and directly return raw data.
        raw_data = load_test_data_amass(data_save_path, data_num, clip_len, joints_num, reload)
        return raw_data

    elif test_type == 'debug_inference':
        # * Only load small test data for debug.
        index = 0  # * Only select the first sequence.
        raw_data = load_test_data_amass(data_save_path, data_num, clip_len, joints_num, reload)
        joints_clip = raw_data['joints_clips'][index]  # * (clip_len, 22, 3)
        smplx_clip = raw_data['smplx_clips'][index]  # * (clip_len, 178)
        noise = rohm_load_noise(args) if add_noise else None
        input_data = rohm_preprocess_input(joints_clip, smplx_clip, smplx_model, device,
                                           noise=noise)
        return [input_data]

    elif test_type == 'debug_render':
        # * Only load small test data for debug.
        index = 0  # * Only select the first sequence.
        raw_data = load_test_data_amass(data_save_path, data_num, clip_len, joints_num, reload)
        joints_clip = raw_data['joints_clips'][index]  # * (clip_len, 22, 3)
        smplx_clip = raw_data['smplx_clips'][index]  # * (clip_len, 178)

        # * Change position and orientation.
        # * Perform canonicalization to the original motion sequence.
        # * IMPORTANT: if not canonicalize, the rendered body and
        # * skeleton will be 180 degrees different.
        cano_joints_clip, cano_smplx_verts = cano_joints_and_smplx_clip(joints_clip, smplx_clip,
                                                                        smplx_model, device)
        return cano_joints_clip, cano_smplx_verts

    elif test_type == 'all':
        # * Load all amass dataset for testing.
        _, _, _, _, traj_dataloader, _ = rohm_generate_dataloader(args, dist_util.dev())
        return traj_dataloader

    else:
        raise ValueError('Only support test_type: [test_inference, test_render, all].')


def test_smplx_body_pos(index, axis, bs, device, smplx_clip_dim=178):
    smplx_clip = torch.zeros(bs, smplx_clip_dim)
    global_orient = torch.tensor([1.4, 0, 0])
    transl = torch.tensor([1, 0.5154, 0.8858])

    # * 63: 21 joints * 3 dof
    orientations = torch.zeros(bs, 21, 3)
    orientations[:, index, axis] = 1.4
    orientations = orientations.reshape(-1, 63)

    smplx_clip[:, 0:3] = global_orient
    smplx_clip[:, 3:6] = transl
    smplx_clip[:, 16:(16 + 63)] = orientations

    return smplx_clip.to(device)


def test_smplx_joint_idx(smplx_model, device):
    """
    Render images with different joint settings for joints checking.
    """
    for axis in [0, 1, 2]:
        save_path = './outputs/render_imgs/{}'.format(axis)
        os.makedirs(save_path, exist_ok=True)
        for idx in range(21):
            smplx_clip = test_smplx_body_pos(idx, axis, 145, device)
            # * smplx_clip --> joints_clip: qpos --> xpos.
            joints_clip, smplx_verts = convert_smplx_clip_to_joints_clip(smplx_clip, smplx_model,
                                                                         device)
            img_name = '{}_{}.png'.format(axis, idx)
            render_smplx_body_skeleton(smplx_verts[0], joints_clip[0], smplx_model,
                                       save_path=save_path, img_name=img_name, debug=True)


# * ------------------------------------------------------ * #
# * ------------- Humanoid-Bench ENV related ------------- * #
# * ------------------------------------------------------ * #

def print_ob(env, ob):
    if isinstance(ob, dict):
        print(f'ob_space = {env.observation_space}')
        print(f'ob = ')
        for k, v in ob.items():
            print(f'  {k}: {v.shape}')
            assert v.shape == env.observation_space.spaces[k].shape
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f'ob_space = {env.observation_space}, ob = {ob.shape}')
        assert env.observation_space.shape == ob.shape
    print(f'ac_space = {env.action_space.shape}')


def test_select_action(
        args_env,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        index=None,
        val=None,
        obs=None
):
    if obs is not None:
        obs = select_target_env_obs(args_env, obs)
    if robot_type == 'h1':
        action_shape = (19,)
        action = np.zeros(action_shape)
        if obs is not None:
            action = obs[:19]
    elif robot_type in {'h1hand', 'h1touch', 'h1strong'}:
        action_shape = (61,)
        action = np.zeros(action_shape)
        if obs is not None:
            action[:16] = obs[:16]
            action[40:45] = obs[40:45]
    else:
        raise ValueError(f'Unknown robot type: {robot_type}')
    if (index is not None) and (val is not None):
        assert index < action_shape[0], f'index should be less than {action_shape[0]}'
        action[index] = val
    return action


def test_robot_qpos_rec(
        args_env,
        h1_data: list,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=False
):
    qpos_rec = []
    for obs in h1_data:
        obs = select_target_env_obs(args_env, obs)
        rec_item = convert_robot_qpos_to_body_qpos(obs, robot_type=robot_type,
                                                   ignore_wrist=ignore_wrist, cat_direction=1)
        qpos_rec.append(rec_item)
    qpos_rec = torch.stack(qpos_rec).squeeze(1).float()
    return qpos_rec


def test_robot_qpos_conversion(
        args_env,
        h1_data: list,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=False
):
    # * qpos_src -> smplx_pose -> qpos_rec
    smplx_clip = convert_robot_pose_to_smplx(args_env, h1_data, robot_type=robot_type,
                                             ignore_wrist=ignore_wrist)
    smplx_pose = smplx_clip[:, 16:(16 + 63)].reshape(-1, 21, 3)
    robot_actions = convert_smplx_joints_pose_to_robot(smplx_pose, robot_type=robot_type,
                                                       ignore_wrist=ignore_wrist)
    qpos_actions_rec = convert_robot_qpos_to_body_qpos(robot_actions, robot_type=robot_type,
                                                       ignore_wrist=ignore_wrist, cat_direction=1)
    return qpos_actions_rec


def test_robot_smplx_data_convert(
        args_env,
        h1_data: list,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=False
):
    """
    Compare robot qpos_src and qpos_rec after conversion.
    """
    # * Original src.
    qpos_src = test_robot_qpos_rec(args_env, h1_data, robot_type, ignore_wrist)
    # * Conversion: qpos_src -> smplx_pose -> qpos_rec
    qpos_rec = test_robot_qpos_conversion(args_env, h1_data, robot_type, ignore_wrist)
    assert qpos_src.shape == qpos_rec.shape, \
        cprint('src.shape: {}  | rec.shape: {}'.format(qpos_src.shape, qpos_rec.shape), 'red')
    return torch.allclose(qpos_src, qpos_rec, rtol=1e-3, atol=1e-3)


# * ------------------------------------------------------ * #
# * -------------------- Env Releated -------------------- * #
# * ------------------------------------------------------ * #

def test_generate_action(env):
    """
    Reference:
        - https://github.com/carlosferrazza/humanoid-bench/issues/21
    """
    action_raw = np.zeros(env.action_space.shape)
    joint_to_action = {}
    offset = 0
    for i in range(env.model.njnt):
        print(f'joint {i}: {env.model.joint(i).name}')
        jnt_name = env.model.joint(i).name
        if jnt_name.startswith('free'):
            joint_to_action[env.model.joint(i).name] = range(i + offset, i + offset + 7)
            offset += 6
        else:
            joint_to_action[env.model.joint(i).name] = i + offset

    for i in range(env.model.nu):
        print(f'actuator {i}: {env.model.actuator(i).name}')
        act_name = env.model.actuator(i).name
        if act_name.startswith('lh') or act_name.startswith('rh'):
            action_raw[i] = 0.0
        else:
            action_raw[i] = env.data.qpos[joint_to_action[act_name]]

    action = env.task.normalize_action(action_raw)
    return action


def test_generate_env_data(
        save_data=True,
        clip_len=145,
        random_action=False,
        zero_hand_pos=True,
        image_save_path='./humanoid_render_images',
        data_save_path='./h1hand_obs_act_data.pkl'
):
    args_env, _, args_kwargs = config_flam()
    env = gym.make(args_env.env, render_mode=args_env.render_mode, **args_kwargs)
    ob, _ = env.reset()
    if save_data:
        os.makedirs(image_save_path, exist_ok=True)

    observations = []
    actions = []
    step = 0
    while step < clip_len:
        if random_action:
            action = env.action_space.sample()  # * np.ndarray, (61,)
            if zero_hand_pos:
                action[21:] = 0
            # action = env.task.normalize_action(action)
        else:
            action = test_generate_action(env)
        ob, reward, terminated, truncated, info = env.step(action)
        img = env.render()  # * Render.
        if args_env.render_mode == 'rgb_array':
            cv2.imshow('test_env', img[:, :, ::-1])
            cv2.waitKey(1)
        if save_data:
            file_path = os.path.join(image_save_path, 'frame_{}.png'.format(format(step, '03d')))
            save_img(img, file_path)
        if terminated or truncated:
            env.reset()
            cprint('Reset', 'yellow')
        observations.append(ob)
        actions.append(action)
        step += 1

    h1_data = {'obs': observations, 'actions': actions}
    if save_data:
        with open(data_save_path, 'wb') as f:
            pickle.dump(h1_data, f)
        cprint('Save data.', 'green')

    env.close()
