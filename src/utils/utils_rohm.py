"""
@Project     ：FLAM
@File        ：utils_rohm.py
@Author      ：Xianqi-Zhang
@Date        ：2024/5/21
@Last        : 2024/6/20
@Description : Utils functions for RoHM.
               Code based on RoHM: https://github.com/sanweiliti/RoHM
"""
import os
import json
import smplx
import pickle
import random
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from typing import Final, Literal
from scipy.spatial.transform import Rotation as R

from src.pkgs.RoHM.model.posenet import PoseNet
from src.pkgs.RoHM.model.trajnet import TrajNet
from src.pkgs.RoHM.diffusion.respace import SpacedDiffusionPoseNet, SpacedDiffusionTrajNet
from src.pkgs.RoHM.diffusion import gaussian_diffusion_posenet, gaussian_diffusion_trajnet
from src.pkgs.RoHM.data_loaders.dataloader_amass import DatasetAMASS
from src.pkgs.RoHM.data_loaders.motion_representation import recover_from_repr_smpl, get_repr_smplx
from src.pkgs.RoHM.data_loaders.common.quaternion import rot6d_to_rotmat
from src.pkgs.RoHM.utils.model_util import create_gaussian_diffusion
from src.pkgs.RoHM.utils.other_utils import REPR_LIST, REPR_DIM_DICT
from src.pkgs.RoHM.utils.konia_transform import rotation_matrix_to_angle_axis
from src.pkgs.RoHM.utils.render_util import render_img

lower_body_names: Final = (
    'leftLeg', 'rightLeg', 'leftToeBase', 'rightToeBase',
    'leftFoot', 'rightFoot', 'leftUpLeg', 'rightUpLeg'
)
upper_body_names: Final = (
    'head', 'leftEye', 'rightEye', 'eyeballs',
    'neck', 'spine', 'spine1', 'spine2', 'hips',
    'leftShoulder', 'rightShoulder',
    'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm',
    'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1'
)
# mask_lower_body_names: Final = ('leftLeg', 'rightLeg')
mask_lower_body_names: Final = lower_body_names
mask_upper_body_names: Final = (
    'neck', 'spine', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder',
    'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm'
)

rohm_info: Final = {
    'pose_traj_feat_dim': 22,
    'pose_pose_feat_dim': 272,
    'traj_traj_feat_dim': 13,
    'traj_pose_feat_dim': 272,
    'joints_num': 22
}


class RoHMDataStorage():

    def __init__(self, args, record_clean=True, load_data=False, file_path=None):
        self.args = args
        self.record_clean = record_clean
        if load_data:
            self.load(file_path)
        else:
            self.LOAD_FLAG = False  # * Data from test or load from pre-saved.
            self.input_noise = self.args.input_noise
            self.mask_scheme = self.args.mask_scheme
            self.motion_repr_clean_list = []
            self.motion_repr_noisy_list = []
            self.motion_repr_rec_list = []

    def update(
            self,
            predicated_pose,
            motion_repr_clean,
            motion_repr_noisy,
            traj_noisy_full,
            pose_mean,
            pose_std
    ):
        if self.record_clean:
            # * [bs, clip_len, body_feat_dim]
            # motion_repr_clean = motion_repr_clean[:, :, 0].permute(0, 2, 1)
            motion_repr_clean = motion_repr_clean * pose_std + pose_mean

        motion_repr_rec = predicated_pose[:, :, 0].permute(0, 2, 1)
        motion_repr_rec = motion_repr_rec * pose_std + pose_mean

        if self.input_noise:
            motion_repr_noisy[:, :, :22] = traj_noisy_full[:, :-1, :]
            motion_repr_noisy = motion_repr_noisy * pose_std + pose_mean

        self.motion_repr_clean_list.append(motion_repr_clean)
        self.motion_repr_noisy_list.append(motion_repr_noisy)
        self.motion_repr_rec_list.append(motion_repr_rec)

    def load(self, file_path=None):
        self.LOAD_FLAG = True
        if file_path is None:
            if self.args.saved_data_path is None:
                raise ValueError('file_path is None')
            file_path = self.args.saved_data_path.format(self.args.mask_scheme)
        print('Storage path: {}'.format(file_path))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.input_noise = data['input_noise']
        self.mask_scheme = data['mask_scheme']
        self.motion_repr_clean_list = data['motion_repr_clean_list']
        self.motion_repr_noisy_list = data['motion_repr_noisy_list']
        self.motion_repr_rec_list = data['motion_repr_rec_list']

    def save(self, data_num=None, save_path=None):
        # * Save path.
        if save_path is None:
            os.makedirs(self.args.save_root, exist_ok=True)
            # file_name = '_iter_{}_iter2trajnoisy_{}_iter2posenoisy_{}_earlystop_{}_seed_{}.pkl'
            file_name = '_i{}_i2tn{}_i2pn{}_es{}_s{}.pkl'
            file_name = file_name.format(
                self.args.sample_iter, self.args.iter2_cond_noisy_traj,
                self.args.iter2_cond_noisy_pose, self.args.early_stop,
                self.args.seed
            )
            # save_dir = 'test_amass_full_grad_{}_mask_{}'.format(self.args.cond_fn_with_grad,
            #                                                     self.mask_scheme)
            save_dir = 'amass_fg{}_m{}'.format(self.args.cond_fn_with_grad, self.mask_scheme)

            if self.args.input_noise and self.args.load_noise:
                save_dir += '_n{}'.format(self.args.load_noise_level)
            if self.args.infill_traj:
                save_dir += '_infill_t{}'.format(self.args.traj_mask_ratio)
            save_dir += file_name
            save_path = os.path.join(self.args.save_root, save_dir)

        def _t2n(x):
            return torch.cat(x, dim=0).detach().cpu().numpy() if not self.LOAD_FLAG else x

        motion_repr_clean = self.motion_repr_clean_list
        motion_repr_noisy = self.motion_repr_noisy_list
        motion_repr_rec = self.motion_repr_rec_list
        if data_num is not None:
            motion_repr_clean = motion_repr_clean[:data_num]
            motion_repr_noisy = motion_repr_noisy[:data_num]
            motion_repr_rec = motion_repr_rec[:data_num]
        data = {}
        data['input_noise'] = self.input_noise
        data['mask_scheme'] = self.mask_scheme
        data['motion_repr_clean_list'] = _t2n(motion_repr_clean) if self.record_clean else None
        data['motion_repr_noisy_list'] = _t2n(motion_repr_noisy) if self.args.input_noise else None
        data['motion_repr_rec_list'] = _t2n(motion_repr_rec)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=2)


# * Models and data load related.
def rohm_load_rohm_params(args_rohm, device):
    pose_mean, pose_std, traj_mean, traj_std = rohm_load_mean_std(device)  # * torch.Tensor
    rohm_models = rohm_load_models(args_rohm, device)  # * RoHM models.
    smplx_neutral = rohm_load_smplx(device=device)  # * Smplx model.
    rohm_params = {
        'args_rohm': args_rohm,
        'smplx_model': smplx_neutral,
        'rohm_models': rohm_models,
        'traj_mean': traj_mean,
        'traj_std': traj_std,
        'pose_mean': pose_mean,
        'pose_std': pose_std,
    }
    return rohm_params


def rohm_load_models(args, device) -> dict:
    """Load PoseNet, TrajNet and Diffusions."""
    # * 01. PoseNet.
    posenet = PoseNet(body_model_path=args.body_model_path, device=device).to(device)
    print('[INFO] loaded PoseNet checkpoint path:', args.model_path_posenet)
    weights = torch.load(args.model_path_posenet, map_location=lambda storage, loc: storage)
    posenet.load_state_dict(weights)
    posenet.eval()

    # * 02. TrajNet.
    trajnet = TrajNet(repr_abs_only=args.repr_abs_only,
                      device=device,
                      trajcontrol=False).to(device)
    print('[INFO] Loaded TrajNet checkpoint:', args.model_path_trajnet)
    weights = torch.load(args.model_path_trajnet, map_location=lambda storage, loc: storage)
    trajnet.load_state_dict(weights)
    trajnet.eval()

    # * 03. TrajNet_Control
    trajnet_control = TrajNet(repr_abs_only=args.repr_abs_only,
                              device=device,
                              trajcontrol=True).to(device)
    print('[INFO] Loaded TrajNet TrajControl checkpoint:', args.model_path_trajnet_control)
    weights = torch.load(args.model_path_trajnet_control,
                         map_location=lambda storage, loc: storage)
    trajnet_control.load_state_dict(weights)
    trajnet_control.eval()

    # * 04. Diffusion.
    diffusion_posenet_eval = create_gaussian_diffusion(
        args,
        gd=gaussian_diffusion_posenet,
        return_class=SpacedDiffusionPoseNet,
        num_diffusion_timesteps=args.diffusion_steps_posenet,
        timestep_respacing=args.timestep_respacing_eval,
        device=device
    )
    diffusion_trajnet_eval = create_gaussian_diffusion(
        args,
        gd=gaussian_diffusion_trajnet,
        return_class=SpacedDiffusionTrajNet,
        num_diffusion_timesteps=args.diffusion_steps_trajnet,
        timestep_respacing=args.timestep_respacing_eval,
        device=device
    )
    diffusion_trajnet_control_eval = create_gaussian_diffusion(
        args,
        gd=gaussian_diffusion_trajnet,
        return_class=SpacedDiffusionTrajNet,
        num_diffusion_timesteps=args.diffusion_steps_trajnet,
        timestep_respacing=args.timestep_respacing_eval,
        device=device
    )
    rohm = {
        'posenet': posenet,
        'trajnet': trajnet,
        'trajnet_control': trajnet_control,
        'diffusion_posenet_eval': diffusion_posenet_eval,
        'diffusion_trajnet_eval': diffusion_trajnet_eval,
        'diffusion_trajnet_control_eval': diffusion_trajnet_control_eval
    }
    return rohm


def rohm_load_smplx(
        body_model_path='src/pkgs/RoHM/data/body_models/smplx_model',
        device=torch.device('cpu'),
        model_type='smplx',
        gender='neutral',
        flat_hand_mean=True,
        use_pca=False
):
    """Load smplx model: a joint 3D model of the human body."""
    smplx_model = smplx.create(
        model_path=body_model_path,
        model_type=model_type,
        gender=gender,
        flat_hand_mean=flat_hand_mean,
        use_pca=use_pca
    ).to(device)
    return smplx_model


def _rohm_load_std_mean(file_path, device=None):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    data = np.concatenate([data[key] for key in data.keys()], axis=-1)
    if device is not None:
        data = torch.from_numpy(data).to(device)
    return data


def rohm_load_mean_std(device=None):
    """
    Load mean and std data. (AMASS dataset.)
    Params:
        - device: if it is not None, return torch.Tensor, else np.ndarray.

    In fact,
        - the value of [pose_mean] is equal [traj_mean],
        - the value of [pose_std] is equal [traj_std].
    mean & std: shape [294]
    """
    pose_mean_path = './src/pkgs/RoHM/data/checkpoints/posenet_checkpoint/AMASS_mean.pkl'
    pose_std_path = './src/pkgs/RoHM/data/checkpoints/posenet_checkpoint/AMASS_std.pkl'
    traj_mean_path = './src/pkgs/RoHM/data/checkpoints/trajnet_checkpoint/AMASS_mean.pkl'
    traj_std_path = './src/pkgs/RoHM/data/checkpoints/trajnet_checkpoint/AMASS_std.pkl'

    pose_mean = _rohm_load_std_mean(pose_mean_path, device)
    pose_std = _rohm_load_std_mean(pose_std_path, device)
    traj_mean = _rohm_load_std_mean(traj_mean_path, device)
    traj_std = _rohm_load_std_mean(traj_std_path, device)

    return pose_mean, pose_std, traj_mean, traj_std


def rohm_load_presaved_noise(args) -> dict:
    """
    smplx_noise_dict: (key, value.shape)
        {
            'global_orient': (746, 145, 3),
            'transl': (746, 145, 3),
            'betas': (746, 145, 10),
            'body_pose': (746, 145, 21, 3)
        }
    """
    noise_pkl_path = args.noise_pkl_path.format(args.load_noise_level)
    presaved_noise_dict = None
    if os.path.exists(noise_pkl_path):
        with open(noise_pkl_path, 'rb') as f:
            presaved_noise_dict = pickle.load(f)
    return presaved_noise_dict


def rohm_load_noise(args) -> dict:
    """
    noise_dict: (key, value.shape)
        {
            'global_orient': (clip_len, 3)
            'transl': (clip_len, 3)  # * (145, 3)
            'betas': (clip_len, 10)
            'body_pose': (clip_len, 21, 3)
        }
    """
    if args.load_noise:
        presaved_noise = rohm_load_presaved_noise(args)
        i = random.randint(0, list(presaved_noise.values())[0].shape[0] - 1)
        noise = {key: value[i, :, :] for key, value in presaved_noise.items()}
    else:
        transl_scale = args.noise_std_smplx_trans  # * 0.03
        betas_scale = args.noise_std_smplx_betas  # * 0.1
        global_orient_scale = args.noise_std_smplx_global_rot  # * 3
        body_pose_scale = args.noise_std_smplx_body_rot  # * 3
        noise = {  # * key: np.random.normal(loc, scale, size)
            'global_orient': np.random.normal(0.0, global_orient_scale, (args.clip_len, 3)),
            'transl': np.random.normal(0.0, transl_scale, (args.clip_len, 3)),
            'betas': np.random.normal(0.0, betas_scale, (args.clip_len, 10)),
            'body_pose': np.random.normal(0.0, body_pose_scale, (args.clip_len, 21, 3))
        }
    return noise


def rohm_add_noise_to_smplx_params(data, noise):
    # * global_orient
    global_orient_mat = R.from_rotvec(data['global_orient'])  # [145, 3, 3]
    global_orient_angle = global_orient_mat.as_euler('zxy', degrees=True)
    global_orient_angle += noise['global_orient']
    n_global_orient = R.from_euler('zxy', global_orient_angle, degrees=True).as_rotvec()
    data['global_orient'] = n_global_orient

    # * transl & betas
    data['transl'] += noise['transl']
    data['betas'] += noise['betas']

    # * body_pose
    body_pose_mat = R.from_rotvec(data['body_pose'].reshape(-1, 3))
    body_pose_angle = body_pose_mat.as_euler('zxy', degrees=True)  # [145*21, 3]
    body_pose_angle += noise['body_pose'].reshape(-1, 3)
    n_body_pose = R.from_euler('zxy', body_pose_angle, degrees=True).as_rotvec()
    data['body_pose'] = n_body_pose.reshape(-1, 21, 3)

    return data


def rohm_filter_items(
        src_repr,
        idx=None,
        device=None,
        repr_name_list=None,
        repr_dim_dict=None
) -> dict:
    """
    Select items from src_repr according to REPR_LIST and REPR_DIM_DICT.
    """
    repr_name_list = REPR_LIST if repr_name_list is None else repr_name_list
    repr_dim_dict = REPR_DIM_DICT if repr_dim_dict is None else repr_dim_dict
    start = 0
    dst_repr = {}
    for repr_name in repr_name_list:
        if idx is None:
            repr = src_repr[..., start:start + repr_dim_dict[repr_name]]
        else:
            repr = src_repr[[idx], ..., start:start + repr_dim_dict[repr_name]]
        if (device is not None) and isinstance(repr, np.ndarray):
            repr = torch.from_numpy(repr).to(device)
        dst_repr[repr_name] = repr
        start += repr_dim_dict[repr_name]
    return dst_repr


def rohm_generate_dataloader(
        args,
        device,
        split: str = 'test',
        num_workers: int = 8,
        amass_test_datasets: dict = None
):
    """Generate Pose dataloader and Traj dataloader"""
    if amass_test_datasets is None:
        amass_test_datasets = ['TotalCapture']
    # * Load pre-computed body noise.
    if args.load_noise:
        noise_pkl_path = args.noise_pkl_path.format(args.load_noise_level)
        with open(noise_pkl_path, 'rb') as f:
            loaded_smplx_noise_dict = pickle.load(f)
    else:
        loaded_smplx_noise_dict = None

    # * 01. Pose dataloader.
    print('[INFO] Generating pose dataset...')
    log_dir_pose = args.model_path_posenet.split('/')[0:-1]
    log_dir_pose = '/'.join(log_dir_pose)
    pose_dataset = DatasetAMASS(preprocessed_amass_root=args.dataset_root,
                                split=split,
                                amass_datasets=amass_test_datasets,
                                body_model_path=args.body_model_path,
                                input_noise=args.input_noise,
                                noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                noise_std_smplx_trans=args.noise_std_smplx_trans,
                                noise_std_smplx_betas=args.noise_std_smplx_betas,
                                load_noise=args.load_noise,
                                loaded_smplx_noise_dict=loaded_smplx_noise_dict,
                                task='pose',  # * !
                                clip_len=args.clip_len,
                                logdir=log_dir_pose,  # * !
                                device=device)
    pose_dataloader = DataLoader(pose_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)
    pose_info = {
        'body_feat_dim': pose_dataset.body_feat_dim,
        'traj_feat_dim': pose_dataset.traj_feat_dim,
        'pose_feat_dim': pose_dataset.pose_feat_dim,
        'mean': torch.from_numpy(pose_dataset.Mean).to(device),
        'std': torch.from_numpy(pose_dataset.Std).to(device)
    }

    # * 02. Traj dataloader.
    print('[INFO] Generating traj dataset...')
    log_dir_traj = args.model_path_trajnet.split('/')[0:-1]
    log_dir_traj = '/'.join(log_dir_traj)
    traj_dataset = DatasetAMASS(preprocessed_amass_root=args.dataset_root,
                                split=split,
                                amass_datasets=amass_test_datasets,
                                body_model_path=args.body_model_path,
                                repr_abs_only=args.repr_abs_only,  # * !
                                input_noise=args.input_noise,
                                noise_std_smplx_global_rot=args.noise_std_smplx_global_rot,
                                noise_std_smplx_body_rot=args.noise_std_smplx_body_rot,
                                noise_std_smplx_trans=args.noise_std_smplx_trans,
                                noise_std_smplx_betas=args.noise_std_smplx_betas,
                                load_noise=args.load_noise,
                                loaded_smplx_noise_dict=loaded_smplx_noise_dict,
                                task='traj',  # * !
                                clip_len=args.clip_len,
                                logdir=log_dir_traj,  # * !
                                device=device)
    traj_dataloader = DataLoader(traj_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)
    traj_info = {
        'body_feat_dim': traj_dataset.body_feat_dim,
        'traj_feat_dim': traj_dataset.traj_feat_dim,
        'pose_feat_dim': traj_dataset.pose_feat_dim,
        'mean': torch.from_numpy(traj_dataset.Mean).to(device),
        'std': torch.from_numpy(traj_dataset.Std).to(device)
    }

    return pose_dataset, pose_dataloader, pose_info, traj_dataset, traj_dataloader, traj_info


def rohm_generate_joints_position(smplx_params, smplx_model, device):
    smplx_params = {key: torch.from_numpy(value).to(device).float() for key, value in
                    smplx_params.items()}
    bs = list(smplx_params.values())[0].shape[0]
    face_and_hand = {
        'jaw_pose': torch.zeros(bs, 3).to(device),
        'leye_pose': torch.zeros(bs, 3).to(device),
        'reye_pose': torch.zeros(bs, 3).to(device),
        'left_hand_pose': torch.zeros(bs, 45).to(device),
        'right_hand_pose': torch.zeros(bs, 45).to(device),
        'expression': torch.zeros(bs, 10).to(device),
    }
    smplx_params.update(face_and_hand)
    # * [clip_len, 22, 3]
    position = smplx_model(**smplx_params).joints[:, :22].detach().cpu().numpy()
    return position


def rohm_get_mask_vis_id(mask_scheme, traj_mask_ratio, clip_len):
    """
    Ids related to LIMBS_BODY_SMPL (defined in other_utils.py).
    Lower mask: left leg, right leg
    Upper mask: left arm, right arm, spline
    """
    mask_joint_id = None
    vis_joint_id = None
    mask_start = None
    mask_end = None
    if mask_scheme == 'lower':  # * Mask out lower body part.
        mask_joint_id = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])
        vis_joint_id = set(range(22)) - set(mask_joint_id)
    elif mask_scheme == 'upper':  # * Mask out upper body part.
        mask_joint_id = np.asarray([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        vis_joint_id = set(range(22)) - set(mask_joint_id)
    elif mask_scheme == 'full':
        mask_start = 65
        mask_len = int(traj_mask_ratio * clip_len)
        mask_end = mask_start + mask_len
    return mask_joint_id, vis_joint_id, mask_start, mask_end


def rohm_mask_pose_cond(args, data, batch_size, clip_len, pose_traj_feat_dim=22) -> dict:
    """
    Apply occlusion masks.
    For iter inference>0, do not use occlusion mask if iter2_cond_noisy_pose=False.
    Input dim:
        - iter 0: [bs, body_feat_dim, clip_len] ([bs, 143, 294])
        - iter>0: [bs, clip_len, 1, body_feat_dim] ([bs, 294, 1, 143])
    Output dim: [bs, clip_len, 1, body_feat_dim]
    """
    if len(data['pose_cond'].shape) == 4:
        data['pose_cond'] = data['pose_cond'].squeeze(-2).permute(0, 2, 1)

    # # * Replace condition traj with denoised output from traj network.
    # if not (args.mask_scheme == 'lower' and not args.input_noise):
    #     data['pose_cond'][:, :, :22] = traj_rec_full

    # * Add mask.
    if args.mask_scheme in ['lower', 'upper']:
        mask_joint_id, _, _, _ = rohm_get_mask_vis_id(args.mask_scheme, args.traj_mask_ratio,
                                                      args.clip_len)
        cursor_1 = pose_traj_feat_dim + mask_joint_id * 3
        cursor_2 = pose_traj_feat_dim + 22 * 3 + mask_joint_id * 3
        cursor_3 = pose_traj_feat_dim + 22 * 3 + 22 * 3 + (mask_joint_id - 1) * 6
        for k in range(3):
            data['pose_cond'][:, :, cursor_1 + k] = 0.
        for k in range(3):
            data['pose_cond'][:, :, cursor_2 + k] = 0.
        for k in range(6):
            data['pose_cond'][:, :, cursor_3 + k] = 0.
        data['pose_cond'][:, :, -4:] = 0.

    elif args.mask_scheme == 'full':
        # * Mask out full body pose (excluding traj) for some frames.
        if not args.infill_traj:
            mask_len = 30
            start = torch.FloatTensor(batch_size).uniform_(0, clip_len - 1).long()  # * [bs]
            end = start + mask_len
            end[end > clip_len] = clip_len
        else:
            mask_len = int(args.traj_mask_ratio * clip_len)
            # default setup for tab.1 in the paper [RoHM].
            start = torch.ones([batch_size]).long() * 65
            end = start + mask_len
        data['pose_cond'][:, :, -4:] = 0.
        for idx in range(batch_size):
            data['pose_cond'][idx, start[idx]:end[idx], 22:] = 0
    data['pose_cond'] = torch.permute(data['pose_cond'], (0, 2, 1)).unsqueeze(-2)

    return data


def rohm_traj_to_motion_repr(
        pred_traj,
        motion_repr,
        traj_traj_feat_dim,
        device,
        absolute_traj_only=True,
        zero_pose=True
) -> torch.Tensor:
    """
    Convert predicted/reconstructed traj repr to full motion repr.
    Pose part from gt (but unused) if not zero_pose.
    If zero_pose, motion_repr is motion_repr shape, otherwise it's
    motion_repr tensor.

    Return:
        - motion_repr_root_rec: shape [bs, 144, 294] (144, same as input.)
    """
    if zero_pose:
        pose_part_shape = [motion_repr[0], motion_repr[1], motion_repr[2] - traj_traj_feat_dim]
        pose_part = torch.zeros(pose_part_shape, device=device)
        motion_repr_full = torch.zeros(motion_repr, device=device)
    else:
        pose_part = motion_repr[:, :, traj_traj_feat_dim:]
        motion_repr_full = motion_repr

    if not absolute_traj_only:
        motion_repr_clean_root_rec = torch.cat([pred_traj, pose_part], dim=-1)  # * [bs, 144, 294]
    else:
        motion_repr_clean_root_rec = motion_repr_full
        motion_repr_clean_root_rec[..., 0] = pred_traj[..., 0]
        motion_repr_clean_root_rec[..., 2:4] = pred_traj[..., 1:3]
        motion_repr_clean_root_rec[..., 6] = pred_traj[..., 3]
        motion_repr_clean_root_rec[..., 7:13] = pred_traj[..., 4:10]
        motion_repr_clean_root_rec[..., 16:19] = pred_traj[..., 10:13]
    return motion_repr_clean_root_rec


def rohm_reconstruct_traj(
        device,
        smplx_model,
        motion_repr,
        traj_mean,
        traj_std,
        pose_mean,
        pose_std
) -> torch.Tensor:
    """
    Reconstruct full traj repr (including both absolute and relative repr).

    Params:
        - motion_repr: shape [bs, 144, 294]
    Returns:
        - traj_rec_full: shape [bs, 143, 22]
            - The dimension reduction is caused by get_repr_smplx(),
            - which calculates velocity related variables with 2 frames.
    """
    motion_repr = motion_repr * traj_std + traj_mean
    if isinstance(motion_repr, torch.Tensor):
        motion_repr = motion_repr.detach().cpu().numpy()

    cur_total_dim = 0
    repr_clean_root_rec = {}
    for repr_name in REPR_LIST:
        end_dim = cur_total_dim + REPR_DIM_DICT[repr_name]
        tmp_data = motion_repr[..., cur_total_dim:end_dim]
        repr_clean_root_rec[repr_name] = torch.from_numpy(tmp_data).to(device)
        cur_total_dim += REPR_DIM_DICT[repr_name]
    rec_ric_data_rec_from_smpl, _ = recover_from_repr_smpl(repr_clean_root_rec, smplx_model,
                                                           to_numpy=True)

    traj_rec_full = []
    for i in range(len(rec_ric_data_rec_from_smpl)):
        global_orient_mat = rot6d_to_rotmat(repr_clean_root_rec['smplx_rot_6d'][i])  # * [T, 3, 3]
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat)  # * [T, 3]
        body_pos_mat = rot6d_to_rotmat(repr_clean_root_rec['smplx_body_pose_6d'][i].reshape(-1, 6))
        body_pos_aa = rotation_matrix_to_angle_axis(body_pos_mat).reshape(-1, 21, 3)
        smplx_params_dict = {
            'transl': repr_clean_root_rec['smplx_trans'][i].detach().cpu().numpy(),
            'global_orient': global_orient_aa.detach().cpu().numpy(),
            'body_pose': body_pos_aa.reshape(-1, 63).detach().cpu().numpy(),
            'betas': repr_clean_root_rec['smplx_betas'][i].detach().cpu().numpy()
        }
        # * !!!
        # * (clip_len -> clip_len-1) in get_repr_smplx because of velocity related calculations.
        repr_dict = get_repr_smplx(positions=rec_ric_data_rec_from_smpl[i],
                                   smplx_params_dict=smplx_params_dict,
                                   feet_vel_thre=5e-5)
        # * new_motion_repr_clean_root_rec
        new_m_r_c_r_r = np.concatenate([repr_dict[key] for key in REPR_LIST], axis=-1)
        new_m_r_c_r_r = (torch.from_numpy(new_m_r_c_r_r).to(device) - pose_mean) / pose_std
        traj_rec_full.append(new_m_r_c_r_r[:, :22])
    traj_rec_full = torch.stack(traj_rec_full)  # * [bs, 143, 22]

    return traj_rec_full


def rohm_generate_mask_traj(device, batch_size, clip_len, feat_dim, mask_start=65, mask_len=14):
    """
    Generate mask for batch_traj['traj_cond'].
    Default setup for tab.1 in the paper.
    Mask part: start -> start+mask_len
    Return:
        torch.Tensor, size [batch_size, clip_len, feat_dim]
    """
    mask_traj = torch.ones(batch_size, clip_len).to(device)  # * [bs, T]
    end = mask_start + mask_len
    mask_traj[:, mask_start:end] = 0
    mask_traj = mask_traj.unsqueeze(-1).repeat(1, 1, feat_dim)  # * [bs, T, feat_dim]
    return mask_traj


# * ------------------------------------------------------ * #
# * ----------------- Visualize & Render ----------------- * #
# * ------------------------------------------------------ * #

def rohm_get_cam_trans(visualize=False, debug=False) -> np.ndarray:
    p = 2 if visualize else 1
    if debug:
        # return np.array([[0, 0, -1, 5],
        #                  [-1, 0, 0, p],
        #                  [0, -1, 0, p],
        #                  [0, 0, 0, 1]])
        return np.array([[-1, 0, 0, p],
                         [0, 0, -1, 5],
                         [0, -1, 0, p],
                         [0, 0, 0, 1]])
    else:
        return np.array([[0, 0, -1, 5],
                         [-1, 0, 0, p],
                         [0, -1, 0, p],
                         [0, 0, 0, 1]])


def rohm_create_o3d_visualizer(w_window=960, h_window=540):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    # vis.create_window()
    vis.create_window(width=w_window, height=h_window)
    vis.add_geometry(mesh_frame)

    ground = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=0.1)
    ground_trans = np.array([[1, 0, 0, -5],
                             [0, 1, 0, 0],
                             [0, 0, 1, -0.1],
                             [0, 0, 0, 1]])
    ground.transform(ground_trans)
    ground.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(ground)
    return vis


def rohm_create_o3d_body_mesh(smpl_verts, smplx_model, color, p_x=0.0, p_y=0.0, p_z=0.0):
    transformation = np.identity(4)
    transformation[:, 3] = [p_x, p_y, p_z, 1.0]
    body_mesh = o3d.geometry.TriangleMesh()
    body_mesh.vertices = o3d.utility.Vector3dVector(smpl_verts)
    body_mesh.triangles = o3d.utility.Vector3iVector(smplx_model)
    body_mesh.compute_vertex_normals()
    body_mesh.paint_uniform_color(color)
    body_mesh.transform(transformation)
    return body_mesh


def rohm_add_geometry(vis, geometry_lists: list):
    for geometry_list in geometry_lists:
        for geometry in geometry_list:
            vis.add_geometry(geometry)
    return vis


def rohm_remove_geometry(vis, geometry_lists: list):
    for geometry_list in geometry_lists:
        for geometry in geometry_list:
            vis.remove_geometry(geometry)
    return vis


def rohm_render_img(
        off_screen_render, body, skel, img_path, img_name, alpha=1.0, flip_image_horizon=True
):
    if body is None and skel is None:
        raise ValueError('Body and Skeleton cannot be None at the same time.')
    color_body = render_img(off_screen_render, body, alpha) if body is not None else None
    color_skel = render_img(off_screen_render, skel, alpha) if skel is not None else None
    if color_body is not None and color_skel is not None:
        color_body.paste(color_skel, (0, 0), color_skel)
    color_img = color_body if color_body is not None else color_skel
    if flip_image_horizon:
        color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)
    color_img.save(os.path.join(img_path, img_name))


def rohm_smplx_body_verts(
        part: Literal['lower', 'upper'] = 'lower',
        smplx_vert_path='src/pkgs/RoHM/data/smplx_vert_segmentation.json',
        vert_num=10475
) -> [list, list]:
    """
    Return:
        body_verts, mask_verts
    """
    smplx_segment = json.load(open(smplx_vert_path))
    if part == 'lower':  # * Mask lower part.
        lower_body_verts = []
        for x in mask_lower_body_names:
            lower_body_verts.extend(smplx_segment[x])
        return list(set(range(vert_num)) - set(lower_body_verts)), list(set(lower_body_verts))
    else:  # * Mask upper part.
        upper_body_verts = []
        for x in mask_upper_body_names:
            upper_body_verts.extend(smplx_segment[x])
        return list(set(range(vert_num)) - set(upper_body_verts)), list(set(upper_body_verts))


def rohm_set_vertex_color(
        body_verts,
        mask_verts,
        body_colors=None,
        mask_colors=None,
        body_alpha=1.0,
        mask_alpha=0.8,
        vert_num=10475
) -> np.ndarray:
    body_colors = np.array([240, 230, 140]) if body_colors is None else body_colors
    mask_colors = np.array([240, 230, 140]) if mask_colors is None else mask_colors
    vertex_colors = np.tile(body_colors, (vert_num, 1))
    vertex_colors[mask_verts] = mask_colors
    vertex_alpha = np.ones((vert_num, 1)) * 255
    vertex_alpha[body_verts] *= body_alpha
    vertex_alpha[mask_verts] *= mask_alpha
    vertex_colors_alpha = np.concatenate([vertex_colors, vertex_alpha], axis=-1)
    return vertex_colors_alpha


# * ------------------------------------------------------ * #
# * ---------------- Visualize and Render ---------------- * #
# * ------------------------------------------------------ * #
from src.pkgs.RoHM.utils.vis_util import *
from src.pkgs.RoHM.utils.render_util import *
from src.pkgs.RoHM.utils.other_utils import *
from src.utils.utils_smplx import render_smplx_body_skeleton
from src.utils.utils_rohm_inference import rohm_postprocess_output_to_joints_clip


def rohm_visualize(args, storage, device, record_clean=True):
    # * Smplx model and parameters initialization.
    pose_mean, pose_std, _, _ = rohm_load_mean_std(device)  # * torch.Tensor
    smplx_neutral = rohm_load_smplx(device=device)  # * Smplx model.
    faces = smplx_neutral.faces
    n_seq = storage.motion_repr_rec_list.shape[0]
    clip_len = storage.motion_repr_rec_list.shape[1]
    print('Sequence Num: ', n_seq)
    contact_lbl_rec_list = storage.motion_repr_rec_list[:, :, -4:]  # * [n_seq, clip_len, 4]
    contact_lbl_rec_list = np.where(contact_lbl_rec_list > 0.5, 1.0, 0.0)
    if record_clean:
        contact_lbl_clean_list = storage.motion_repr_clean_list[:, :, -4:]
    cam_trans = rohm_get_cam_trans(visualize=args.visualize)
    mask_joint_id, _, start, end = rohm_get_mask_vis_id(args.mask_scheme, args.traj_mask_ratio,
                                                        args.clip_len)
    mask_joint_id = mask_joint_id.tolist() if mask_joint_id is not None else None

    print('Visualizing...')
    print('[R][left - Reconstruction]: [blue] visible parts | [yellow] occluded parts')
    print('[N][middle - Noisy/occluded input]')
    print('[G][right - Ground truth]: [red]')
    print('[foot contact]: [red] not in contact with floor | [green] in contact with floor')
    vis = rohm_create_o3d_visualizer()
    for bs in tqdm(range(0, n_seq, args.vis_interval)):
        # * Get smplx vertices.
        joints_clip_rec, smpl_verts_rec = rohm_postprocess_output_to_joints_clip(
            storage.motion_repr_rec_list, smplx_neutral, pose_std, pose_mean, bs, device
        )
        joints_clip_noisy, smpl_verts_noisy = rohm_postprocess_output_to_joints_clip(
            storage.motion_repr_noisy_list, smplx_neutral, pose_std, pose_mean, bs, device
        )
        if record_clean:
            joints_clip_clean, smpl_verts_clean = rohm_postprocess_output_to_joints_clip(
                storage.motion_repr_clean_list, smplx_neutral, pose_std, pose_mean, bs, device
            )

        for t in range(clip_len):
            # * Body skeletons.
            skeleton_rec_list = vis_skeleton(joints=joints_clip_rec[t],
                                             limbs=LIMBS_BODY_SMPL,
                                             add_trans=np.array([0, 0.0, 2.5]),
                                             mask_scheme=args.mask_scheme,
                                             cur_mask_joint_id=mask_joint_id)
            if storage.input_noise:
                skeleton_noisy_list = vis_skeleton(joints=joints_clip_noisy[t],
                                                   limbs=LIMBS_BODY_SMPL,
                                                   add_trans=np.array([0, 1.0, 2.5]),
                                                   mask_scheme=args.mask_scheme,
                                                   cur_mask_joint_id=mask_joint_id)
            if record_clean:
                skeleton_gt_list = vis_skeleton(joints=joints_clip_clean[t],
                                                limbs=LIMBS_BODY_SMPL,
                                                add_trans=np.array([0, 2.0, 2.5]))
            # * Foot contact labels.
            foot_sphere_rec_list = vis_foot_contact(joints_clip_rec[t],
                                                    contact_lbl_rec_list[bs, t])
            if record_clean:
                foot_sphere_clean_list = vis_foot_contact(joints_clip_clean[t],
                                                          contact_lbl_clean_list[bs, t],
                                                          np.array([0, 2.0, 0.0]))

            # * Body mesh.
            body_mesh_rec = rohm_create_o3d_body_mesh(smpl_verts_rec[t], faces, COLOR_VIS_O3D)
            body_mesh_noisy = rohm_create_o3d_body_mesh(smpl_verts_noisy[t], faces, COLOR_OCC_O3D,
                                                        p_y=1.0)
            if record_clean:
                body_mesh_clean = rohm_create_o3d_body_mesh(smpl_verts_clean[t], faces,
                                                            COLOR_GT_O3D, p_y=2.0)

            if record_clean:
                body_mesh_list = [body_mesh_rec, body_mesh_noisy, body_mesh_clean]
                geometry_lists = [body_mesh_list, foot_sphere_clean_list, foot_sphere_rec_list,
                                  skeleton_gt_list, skeleton_rec_list]
            else:
                body_mesh_list = [body_mesh_rec, body_mesh_noisy]
                geometry_lists = [body_mesh_list, foot_sphere_rec_list, skeleton_rec_list]
            if storage.input_noise:
                geometry_lists.append(skeleton_noisy_list)
            vis = rohm_add_geometry(vis, geometry_lists)

            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param = update_cam(cam_param, cam_trans)
            # ctr.convert_from_pinhole_camera_parameters(cam_param)
            ctr.convert_from_pinhole_camera_parameters(cam_param, True)
            vis.poll_events()
            vis.update_renderer()
            vis = rohm_remove_geometry(vis, geometry_lists)


def rohm_render(
        args, storage, device, record_clean=True, H=1080, W=1920, cam_x=960, cam_y=540,
        material_body_gt=material_body_gt, render_skeleton=True, render_ground=True
):
    # * Smplx model and parameters initialization.
    pose_mean, pose_std, _, _ = rohm_load_mean_std(device)  # * torch.Tensor
    smplx_neutral = rohm_load_smplx(device=device)  # * Smplx model.
    n_seq = storage.motion_repr_rec_list.shape[0]
    clip_len = storage.motion_repr_rec_list.shape[1]
    print('Sequence Num: ', n_seq)
    contact_lbl_rec_list = storage.motion_repr_rec_list[:, :, -4:]  # * [n_seq, clip_len, 4]
    contact_lbl_rec_list = np.where(contact_lbl_rec_list > 0.5, 1.0, 0.0)
    # contact_lbl_clean_list = storage.motion_repr_clean_list[:, :, -4:]
    mask_joint_id, _, start, end = rohm_get_mask_vis_id(args.mask_scheme, args.traj_mask_ratio,
                                                        args.clip_len)
    mask_joint_id = mask_joint_id.tolist() if mask_joint_id is not None else None

    # * Render related parameters.
    file_path = 'occ_{}_noise_{}'.format(args.traj_mask_ratio, args.load_noise_level)
    args.render_save_path = os.path.join(args.render_save_dir, file_path)
    img_save_path_rec = os.path.join(args.render_save_path, 'pred')
    img_save_path_input = os.path.join(args.render_save_path, 'input')  # * Noisy
    img_save_path_gt = os.path.join(args.render_save_path, 'gt')
    os.makedirs(img_save_path_rec, exist_ok=True)
    os.makedirs(img_save_path_input, exist_ok=True)
    os.makedirs(img_save_path_gt, exist_ok=True)
    colors_noisy = None
    if args.mask_scheme in ['lower', 'upper']:
        body_verts_list, mask_verts_list = rohm_smplx_body_verts(part=args.mask_scheme)
        colors_noisy = rohm_set_vertex_color(body_verts_list, mask_verts_list)

    print('Rendering...')
    print('Render interval: {}'.format(args.render_interval))
    for bs in tqdm(range(0, n_seq, args.render_interval)):
        seq_path = 'seq_{}'.format(format(bs, '03d'))
        cur_rec_path = os.path.join(img_save_path_rec, seq_path)
        cur_input_path = os.path.join(img_save_path_input, seq_path)
        cur_gt_path = os.path.join(img_save_path_gt, seq_path)
        os.makedirs(cur_rec_path, exist_ok=True)
        os.makedirs(cur_input_path, exist_ok=True)
        os.makedirs(cur_gt_path, exist_ok=True)

        # * Get smplx vertices.
        joints_clip_rec, smpl_verts_rec = rohm_postprocess_output_to_joints_clip(
            storage.motion_repr_rec_list, smplx_neutral, pose_std, pose_mean, bs, device
        )
        joints_clip_noisy, smpl_verts_noisy = rohm_postprocess_output_to_joints_clip(
            storage.motion_repr_noisy_list, smplx_neutral, pose_std, pose_mean, bs, device
        )
        if record_clean:
            joints_clip_clean, smpl_verts_clean = rohm_postprocess_output_to_joints_clip(
                storage.motion_repr_clean_list, smplx_neutral, pose_std, pose_mean, bs, device
            )

        for t in range(clip_len):
            flag_lower_upper = args.mask_scheme in ['lower', 'upper']
            flag_full_mask_area = (args.mask_scheme == 'full') and (t < start or t >= end)
            flag_check = (flag_lower_upper or flag_full_mask_area)
            img_name = 'frame_{}.png'.format(format(t, '03d'))  # * Single image.
            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
            render_smplx_body_skeleton(
                smpl_verts_rec[t],
                joints_clip_rec[t],
                smplx_neutral,
                r,
                flag_full_mask_area,
                mask_scheme=args.mask_scheme,
                mask_joint_id=mask_joint_id,
                contact_lbl=contact_lbl_rec_list[bs, t],
                material_body=material_body_rec_vis if flag_check else material_body_rec_occ,
                save_path=cur_rec_path,
                img_name=img_name,
                sekl_name='pred_joint',
                alpha=1.0,
                add_occ_joints=True,
                render_skeleton=render_skeleton,  # * !!!
                render_ground=render_ground
            )
            render_smplx_body_skeleton(
                smpl_verts_noisy[t],
                joints_clip_noisy[t],
                smplx_neutral,
                r,
                flag_full_mask_area,
                mask_scheme=args.mask_scheme,
                mask_joint_id=mask_joint_id,
                contact_lbl=None,
                material_body=None if flag_lower_upper else material_body_noisy,
                save_path=cur_input_path,
                img_name=img_name,
                sekl_name='input_joint',
                alpha=0.5 if flag_full_mask_area else 1.0,
                add_occ_joints=False,
                colors_verts=colors_noisy,  # * !
                render_skeleton=render_skeleton,  # * !!!
                render_ground=render_ground
            )
            if record_clean:
                render_smplx_body_skeleton(
                    smpl_verts_clean[t],
                    None,
                    smplx_neutral,
                    r,
                    flag_full_mask_area,
                    mask_scheme=args.mask_scheme,
                    mask_joint_id=mask_joint_id,
                    contact_lbl=None,
                    material_body=material_body_gt,
                    save_path=cur_gt_path,
                    img_name=img_name,
                    sekl_name=None,
                    alpha=1.0,
                    add_occ_joints=False,
                    render_skeleton=render_skeleton,  # * !!!
                    render_ground=render_ground
                )
