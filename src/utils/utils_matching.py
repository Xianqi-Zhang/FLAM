"""
@Project     ：FLAM
@File        ：utils_matching.py
@Author      ：Xianqi-Zhang
@Date        ：2024/7/17
@Last        : 2024/9/26
@Description : Matching Humanoid-Bench data and SMPL-X.
"""
import torch
import numpy as np
import torch.multiprocessing as mp
from typing import Final, Literal
from collections import defaultdict
from tensordict.tensordict import TensorDict
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion, \
    matrix_to_quaternion, quaternion_multiply
from src.utils.utils_smplx import convert_smplx_clip_to_joints_clip
from src.utils.utils_rohm_inference import rohm_process_smplx_clip_for_inference

XYZ_INDEX: Final = {
    'x': 0,
    'y': 1,
    'z': 2,
}

H1_BODY_INFO: Final = {
    # * Relative to h1_pos.xml in Humanoid-Benchmark.
    # * Lower-Body: y <-> x
    0: {'name': 'left_hip_yaw', 'axis': 'z'},
    1: {'name': 'left_hip_roll', 'axis': 'y'},
    2: {'name': 'left_hip_pitch', 'axis': 'x'},
    3: {'name': 'left_knee', 'axis': 'x'},
    4: {'name': 'left_ankle', 'axis': 'x'},

    5: {'name': 'right_hip_yaw', 'axis': 'z'},
    6: {'name': 'right_hip_roll', 'axis': 'y'},
    7: {'name': 'right_hip_pitch', 'axis': 'x'},
    8: {'name': 'right_knee', 'axis': 'x'},
    9: {'name': 'right_ankle', 'axis': 'x'},

    10: {'name': 'torso', 'axis': 'y'},  # * z -> y: not consistent with h1_pos.xml

    # * Upper-Body: z <-> x
    11: {'name': 'left_shoulder_pitch', 'axis': 'y'},
    12: {'name': 'left_shoulder_roll', 'axis': 'z'},
    13: {'name': 'left_shoulder_yaw', 'axis': 'x'},
    14: {'name': 'left_elbow', 'axis': 'y'},

    15: {'name': 'right_shoulder_pitch', 'axis': 'y'},
    16: {'name': 'right_shoulder_roll', 'axis': 'z'},
    17: {'name': 'right_shoulder_yaw', 'axis': 'x'},
    18: {'name': 'right_elbow', 'axis': 'y'},
}

H1HAND_BODY_INFO: Final = {
    # * Relative to h1hand_pos.xml in Humanoid-Benchmark.
    # * Lower-Body: y <-> x
    0: {'name': 'left_hip_yaw', 'axis': 'z'},
    1: {'name': 'left_hip_roll', 'axis': 'y'},
    2: {'name': 'left_hip_pitch', 'axis': 'x'},
    3: {'name': 'left_knee', 'axis': 'x'},
    4: {'name': 'left_ankle', 'axis': 'x'},

    5: {'name': 'right_hip_yaw', 'axis': 'z'},
    6: {'name': 'right_hip_roll', 'axis': 'y'},
    7: {'name': 'right_hip_pitch', 'axis': 'x'},
    8: {'name': 'right_knee', 'axis': 'x'},
    9: {'name': 'right_ankle', 'axis': 'x'},

    10: {'name': 'torso', 'axis': 'y'},  # * z -> y: not consistent with h1_pos.xml

    # * Upper-Body: z <-> x
    11: {'name': 'left_shoulder_pitch', 'axis': 'y'},
    12: {'name': 'left_shoulder_roll', 'axis': 'z'},
    13: {'name': 'left_shoulder_yaw', 'axis': 'x'},
    14: {'name': 'left_elbow', 'axis': 'y'},
    15: {'name': 'left_wrist_yaw', 'axis': 'x'},  # * New.

    16: {'name': 'right_shoulder_pitch', 'axis': 'y'},
    17: {'name': 'right_shoulder_roll', 'axis': 'z'},
    18: {'name': 'right_shoulder_yaw', 'axis': 'x'},
    19: {'name': 'right_elbow', 'axis': 'y'},
    20: {'name': 'right_wrist_yaw', 'axis': 'x'},  # * New.
}

SMPLX_BODY_NAME2IDX: Final = {
    'left_hip_yaw': 1,
    'left_hip_roll': 1,
    'left_hip_pitch': 1,
    'left_knee': 4,
    'left_ankle': 7,

    'right_hip_yaw': 0,
    'right_hip_roll': 0,
    'right_hip_pitch': 0,
    'right_knee': 3,
    'right_ankle': 6,

    'torso': 5,

    'left_shoulder_pitch': 16,
    'left_shoulder_roll': 16,
    'left_shoulder_yaw': 16,
    'left_elbow': 18,
    'left_wrist_yaw': 20,

    'right_shoulder_pitch': 15,
    'right_shoulder_roll': 15,
    'right_shoulder_yaw': 15,
    'right_elbow': 17,
    'right_wrist_yaw': 19,
}

SMPLX_JOINTS_DIRECTION: Final = {
    # * Only contains h1/h1hand robot joint rotation axis, w/o hands/fingers.
    # * positive: +1 to add degree, negative: -1 to add degree.
    'x': {'positive': {0, 3, 1, 4, 15, 19, 20}, 'negative': {6, 7, 16}},
    'y': {'positive': {0, 15, 17, 16}, 'negative': {1, 5, 18}},
    'z': {'positive': {0, 1}, 'negative': {15, 16}}
}


# * ------------------------------------------------------ * #
# * -------------- Humanoid-Bench to SMPL-X -------------- * #
# * ------------------------------------------------------ * #

def init_smplx_body_pose(
        clip_len,
        device=None,
        smplx_clip_dim=178,
        align_leg=True,
        align_hand=False,
        only_joints_orient=False
) -> torch.Tensor:
    """
    Since the initial pose of SMPL-X body is different from the robot
    initial pose in Humanoid-Bench environment, so slightly change
    SMPL-X body pose for alignment.
    """
    D_90 = np.pi / 2
    # * Joints pose/orientations, 63, 21 joints * 3 dof
    joints_orient = torch.zeros(clip_len, 21, 3)
    joints_orient[:, 15] = torch.tensor([0, 0, -1.2])  # * Shoulder.
    joints_orient[:, 16] = torch.tensor([0, 0, 1.2])
    joints_orient[:, 17] = torch.tensor([0, -D_90, 0])  # * Elbow.
    joints_orient[:, 18] = torch.tensor([0, D_90, 0])
    if align_leg:
        # * Hip, Knee, Ankle are set to the opposite direction,
        # * so that robot joint pose + leg init pose is a normal human
        # * pose (SMPL-X body pose).
        # * If you want init_pose to be similar to robot_pose, the
        # * following should be set to [-, +, +] for x-axis.
        joints_orient[:, 0] = torch.tensor([0.4, 0, 0])  # * Hip.
        joints_orient[:, 1] = torch.tensor([0.4, 0, 0])
        joints_orient[:, 3] = torch.tensor([-0.5, 0, 0])  # * Knee.
        joints_orient[:, 4] = torch.tensor([-0.5, 0, 0])
        joints_orient[:, 6] = torch.tensor([-0.5, 0, 0])  # * Ankle.
        joints_orient[:, 7] = torch.tensor([-0.5, 0, 0])
    if align_hand:
        joints_orient[:, 19] = torch.tensor([-D_90, 0, 0])  # * Wrist.
        joints_orient[:, 20] = torch.tensor([-D_90, 0, 0])

    if only_joints_orient:
        if device is not None:
            joints_orient = joints_orient.to(device)
        return joints_orient

    smplx_clip = torch.zeros(clip_len, smplx_clip_dim)
    smplx_clip[:, 0:3] = torch.tensor([D_90, 0, 0])
    smplx_clip[:, 3:6] = torch.tensor([0, 0, 0])
    smplx_clip[:, 16:(16 + 63)] = joints_orient.reshape(-1, 63)

    if device is not None:
        smplx_clip = smplx_clip.to(device)
    return smplx_clip


def select_target_env_obs(args_env, env_obs, reshape=True, target_obs_name='proprio'):
    """
    Params:
        - env_obs: env.step() output obs
    """
    if (
            args_env.obs_wrapper == 'True'
            and (isinstance(env_obs, dict) or isinstance(env_obs, TensorDict))
    ):
        obs = env_obs[target_obs_name]
    else:
        obs = env_obs
    if reshape:
        # obs = obs.reshape(args_env.num_envs, -1)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
    return obs


def select_env_obs_transl_body_part(transl, device, ignore_wrist=True) -> torch.Tensor:
    """
    Params:
        - env_obs: related to self._env.named.data.xpos in the env.
    """
    assert transl.shape[-2:] == (71, 3)
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl)
    transl = transl.to(device)
    if len(transl.shape) == 2:
        transl = transl.unsqueeze(0)
    if ignore_wrist:
        body_transl = torch.cat([transl[..., 2:17, :], transl[..., 42:46, :]], dim=1)
    else:
        body_transl = torch.cat([transl[..., 2:18, :], transl[..., 42:47, :]], dim=1)
    return body_transl


def select_env_obs_global_orient_body_part(
        global_orient, device, ignore_wrist=True
) -> torch.Tensor:
    """
    Params:
        - global_orient: related to self._env.named.data.xquat in the env.
    """
    assert global_orient.shape[-2:] == (71, 4)
    if isinstance(global_orient, np.ndarray):
        global_orient = torch.from_numpy(global_orient)
    global_orient = global_orient.to(device)
    if len(global_orient.shape) == 2:
        global_orient = global_orient.unsqueeze(0)
    if ignore_wrist:
        body_orient = torch.cat([global_orient[..., 2:17, :], global_orient[..., 42:46, :]], dim=1)
    else:
        body_orient = torch.cat([global_orient[..., 2:18, :], global_orient[..., 42:47, :]], dim=1)
    return body_orient


def convert_env_obs_to_tensor(
        args_env,
        env_obs,
        device=None,
        only_body_part=True,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True,
        zero_wrist=False,
        cat_direction=1,
        target_obs_name='proprio'
):
    if (
            args_env.obs_wrapper == 'True'
            and (isinstance(env_obs, dict) or isinstance(env_obs, TensorDict))
    ):
        obs = select_target_env_obs(args_env, env_obs, target_obs_name=target_obs_name)
        transl = env_obs['transl']
        global_orient = env_obs['global_orient']
    else:
        raise NotImplementedError
    if only_body_part:
        obs_qpos = convert_robot_qpos_to_body_qpos(
            obs[..., :69], device, robot_type, ignore_wrist, zero_wrist, cat_direction
        )
        obs_qvel = convert_robot_qpos_to_body_qpos(
            obs[..., 69:138], device, robot_type, ignore_wrist, zero_wrist, cat_direction
        )
        transl = select_env_obs_transl_body_part(transl, device, ignore_wrist)
        global_orient = select_env_obs_global_orient_body_part(global_orient, device, ignore_wrist)
        obs_tensor = torch.cat(
            [obs_qpos.unsqueeze(-1), obs_qvel.unsqueeze(-1), transl, global_orient], dim=-1
        )  # * (num_envs, 19, 9), qpos(1) + qvel(1) + transl(3) + global_orient(4)
    else:
        obs = torch.from_numpy(obs).to(device)
        transl = torch.from_numpy(transl).to(device)
        global_orient = torch.from_numpy(global_orient).to(device)
        if len(transl.shape) == 2:
            transl = transl.unsqueeze(0)
        if len(global_orient.shape) == 2:
            global_orient = global_orient.unsqueeze(0)
        obs_tensor = torch.zeros((obs.shape[0], 71, 9)).to(device)  # * (num_envs, 71, 9)
        obs_tensor[:, 2:, 0] = obs[:, :69]
        obs_tensor[:, 2:, 1] = obs[:, 69:138]
        obs_tensor[:, :, 2:5] = transl
        obs_tensor[:, :, 5:] = global_orient
    return obs_tensor.reshape(obs_tensor.shape[0], -1).float()


def preprocess_robot_data(
        args_env,
        h1_data,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        device=None,
        orient_index=1,
        transl_index=1,
        smplx_joints_dim=21,
        ignore_wrist=True
) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate global_orient, transl, joints_pose of each frame in
    h1_body_data, which is collected from Humanoid-Bench environment.

    For robot h1hand joint dimensions:
    - Relative to self._env.named.data.qpos in Humanoid-Bench.robots.py
    - left leg & right leg: [0:10]
    - torso: [10:11]
    - left arm: [11:16]
    - left hand/fingers: [16:16+24]
    - right arm: [40: 45]
    - right hand/fingers: [45:45+24]

    Params:
        - h1_body_data: list of observation
            items.keys:
                - Orientation and translation: global_orient, transl
                - Joint qpos: proprio
                - Camera pose: camera_orient, camera_pos
                - [image_left_eye, image_right_eye],
                - # [tactile, tactile_torso]
        - orient_index:
            - Relative to Humanoid-Bench self._env.named.data.xquat.
            - 0: world, 1: pelvis, 12: torso_link
            - data['global_orient'].shape: (71, 4)
    """
    if robot_type == 'h1':
        ROBOT_BODY_INFO = H1_BODY_INFO
        h1_joint_dim = 19
    elif robot_type in {'h1hand', 'h1touch', 'h1strong'}:
        ROBOT_BODY_INFO = H1HAND_BODY_INFO
        h1_joint_dim = 21
    else:
        raise NotImplementedError

    h1_global_orient = []
    h1_transl = []
    h1_joints_orient = []
    for i, data in enumerate(h1_data):
        # * 01. Orientation.
        global_orient = data['global_orient']
        orient_size = global_orient.shape[1]  # * obs_wrapper is True.
        if orient_size == 9:  # * For xmat.
            global_orient = torch.from_numpy(global_orient[orient_index].reshape(1, 3, 3))
            global_orient_quat = matrix_to_quaternion(global_orient)
        elif orient_size == 4:  # * For xquat.
            global_orient_quat = torch.tensor(global_orient[orient_index])
        else:
            raise ValueError('Invalid orientation size: {}'.format(orient_size))

        # * 02. Translation.
        transl = data['transl'][transl_index]
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl)

        # * 03. Joint pose.
        joints_orient = torch.zeros((smplx_joints_dim, 3))
        body_qpos = select_target_env_obs(args_env, data, reshape=False)  # * reshape=False.
        if robot_type in {'h1hand', 'h1touch', 'h1strong'}:
            hand_joint_dim = 24
            r_arm_start = 16 + hand_joint_dim
            body_qpos = np.concatenate([body_qpos[:16], body_qpos[r_arm_start: r_arm_start + 5]])
            body_qpos = torch.from_numpy(body_qpos)
        for i in range(h1_joint_dim):
            joint_qpos = body_qpos[i]
            joint_name = ROBOT_BODY_INFO[i]['name']
            if ignore_wrist and ('wrist' in joint_name):
                continue
            joint_axis = ROBOT_BODY_INFO[i]['axis']
            smplx_joint_idx = SMPLX_BODY_NAME2IDX[joint_name]
            smplx_joint_axis_idx = XYZ_INDEX[joint_axis]
            coef = 1 if smplx_joint_idx in SMPLX_JOINTS_DIRECTION[joint_axis]['positive'] else -1
            joints_orient[smplx_joint_idx, smplx_joint_axis_idx] = joint_qpos * coef

        h1_global_orient.append(global_orient_quat)
        h1_transl.append(transl)
        h1_joints_orient.append(joints_orient.reshape(-1, 63))

    h1_global_orient = torch.stack(h1_global_orient).squeeze(1)  # * (bs, 1, 4) -> (bs, 4)
    h1_transl = torch.stack(h1_transl)
    h1_joints_orient = torch.stack(h1_joints_orient).squeeze()
    if device is not None:
        h1_global_orient = h1_global_orient.to(device)
        h1_transl = h1_transl.to(device)
        h1_joints_orient = h1_joints_orient.to(device)
    return h1_global_orient, h1_transl, h1_joints_orient


def _convert_robot_pose_to_smplx(
        args_env,
        h1_data: list,
        smplx_init_pose: torch.Tensor,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        device=None,
        ignore_wrist=False
) -> torch.Tensor:
    h1_global_orient, h1_transl, h1_joints_orient = preprocess_robot_data(
        args_env, h1_data, robot_type, device, ignore_wrist=ignore_wrist
    )
    # * 01. Orientation.
    # smplx_global_quat = h1_global_orient
    smplx_pose = smplx_init_pose.clone()
    init_quat = axis_angle_to_quaternion(smplx_pose[:, 0:3])
    smplx_global_quat = quaternion_multiply(init_quat, h1_global_orient)
    smplx_global_orient = quaternion_to_axis_angle(smplx_global_quat)
    smplx_pose[:, 0:3] = smplx_global_orient
    # * 02. Transl.
    smplx_pose[:, 3:6] = h1_transl
    # * 03. Joint pose.
    smplx_pose[:, 16:(16 + 63)] += h1_joints_orient  # * Add ! ! !

    if device is not None:
        smplx_pose = smplx_pose.to(device)
    return smplx_pose


def convert_robot_pose_to_smplx(
        args_env,
        h1_data: list,
        device=None,
        smplx_init_pose=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True
):
    """
    Params:
        - h1_data: robot h1 data collected from Humanoid-Bench.
    Returns:
        - smplx_clip: for SMPL-X render or RoHM inference.
        If render_or_inference is 'render',
            - return [joints_clip, smplx_verts].
                - joints_clip: SMPL-X skeleton for image render.
                - smplx_verts: SMPL-X body mesh for image render.
        If render_or_inference is 'inference',
            - return [data] for RoHM inference.
    """
    if smplx_init_pose is None:
        clip_len = len(h1_data)
        smplx_init_pose = init_smplx_body_pose(clip_len=clip_len, device=device)
    smplx_clip = _convert_robot_pose_to_smplx(
        args_env, h1_data, smplx_init_pose, robot_type, device, ignore_wrist=ignore_wrist
    )
    return smplx_clip


def process_robot_data_for_smplx_render(
        args_env,
        h1_data: list,
        smplx_model,
        device,
        cano=True,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand'
):
    """
    Returns:
        - joints_clip: SMPL-X skeleton for image render.
        - smplx_verts: SMPL-X body mesh for image render.
    """
    smplx_clip = convert_robot_pose_to_smplx(args_env, h1_data, device, robot_type=robot_type)
    # * smplx_clip --> joints_clip: qpos --> xpos.
    joints_clip, smplx_verts = convert_smplx_clip_to_joints_clip(smplx_clip, smplx_model, device,
                                                                 cano)
    return joints_clip, smplx_verts


def process_robot_data_for_rohm_inference(
        args_env,
        h1_data: list,
        smplx_model,
        device,
        smplx_init_pose=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        process_container=None,
        process_index=None
) -> dict:
    """
    Process robot data (collected from Humanoid-Bench) for RoHM inference.
    Returns:
        - rohm_input_data: list
    """
    smplx_clip = convert_robot_pose_to_smplx(args_env, h1_data, device,
                                             smplx_init_pose=smplx_init_pose,
                                             robot_type=robot_type)
    processed_data = rohm_process_smplx_clip_for_inference(smplx_clip, smplx_model, device)
    if (process_container is not None) and (process_index is not None):
        process_container.put({'index': process_index, 'output': processed_data})
    else:
        return processed_data


def process_robot_data_for_rohm_inference_batch(
        batch_size,
        args_env,
        h1_data: list,
        smplx_model,
        device,
        smplx_init_pose=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        clip_len=145
) -> dict:
    # * For start method, use 'forkserver' instead of 'spawn'.
    # * Reference: https://zhuanlan.zhihu.com/p/597327498
    mp.set_start_method('spawn', force=True)  # * Necessary.
    # mp.set_start_method('forkserver', force=True)  # * Necessary.
    p = mp.Pool(batch_size)
    input_data = []
    for i in range(batch_size):
        data = h1_data[i * clip_len: (i + 1) * clip_len]
        input_data.append(
            p.apply_async(
                process_robot_data_for_rohm_inference,
                args=(args_env, data, smplx_model, device, smplx_init_pose, robot_type)
            )
        )
    # * https://docs.python.org/3/library/multiprocessing.html
    p.close()  # * Prevents any more tasks from being submitted to the pool.
    p.join()  # * Wait for the worker processes to exit. Must call close() or terminate() before.

    # * Merge all processed data.
    processed_data = defaultdict(list)
    for i in range(batch_size):
        tmp_data = input_data[i].get()
        for key, value in tmp_data.items():
            processed_data[key].append(value.clone())
            del value  # * Avoid memory leak.
        del tmp_data
    input_data = {key: torch.cat(value, dim=0) for key, value in processed_data.items()}

    p.terminate()
    return input_data


def convert_robot_camera_transform_to_smplx(camera_orient, camera_pos):
    cam_trans = torch.zeros((4, 4))
    cam_trans[3, 3] = 1
    cam_trans[:3, :3] = -torch.tensor(camera_orient).reshape(3, 3)  # * - !!!
    cam_trans = torch.stack([
        cam_trans[1],
        cam_trans[0],
        cam_trans[2],
        cam_trans[3]
    ])
    cam_trans[:3, 3] = torch.tensor(camera_pos)  # * [2, 3, 1.79]
    return cam_trans


# * ------------------------------------------------------ * #
# * -------------- SMPL-X to Humanoid-Bench -------------- * #
# * ------------------------------------------------------ * #
SMPLX_BODY_2_H1: Final = {
    # * SMPL-X joints index: {axis: robot_joint_index, ...}
    1: {'x': 2, 'y': 1, 'z': 0},  # * left_hip
    4: {'x': 3},  # * left_knee
    7: {'x': 4},  # * left_ankle

    0: {'x': 7, 'y': 6, 'z': 5},  # * right_hip
    3: {'x': 8},  # * right_knee
    6: {'x': 9},  # * right_ankle

    5: {'y': 10},  # * torso

    16: {'x': 13, 'y': 11, 'z': 12},  # * left_shoulder
    18: {'y': 14},  # * left_elbow

    15: {'x': 17, 'y': 15, 'z': 16},  # * right_shoulder
    17: {'y': 18},  # * right_elbow
}

SMPLX_BODY_2_H1HAND: Final = {
    # * SMPL-X joints index: {axis: robot_joint_index, ...}
    1: {'x': 2, 'y': 1, 'z': 0},  # * left_hip
    4: {'x': 3},  # * left_knee
    7: {'x': 4},  # * left_ankle

    0: {'x': 7, 'y': 6, 'z': 5},  # * right_hip
    3: {'x': 8},  # * right_knee
    6: {'x': 9},  # * right_ankle

    5: {'y': 10},  # * torso

    16: {'x': 13, 'y': 11, 'z': 12},  # * left_shoulder
    18: {'y': 14},  # * left_elbow
    20: {'x': 15},  # * left_wrist_yaw

    15: {'x': 18, 'y': 16, 'z': 17},  # * right_shoulder
    17: {'y': 19},  # * right_elbow
    19: {'x': 20},  # * right_wrist_yaw
}


def convert_robot_qpos_to_body_qpos(
        robot_qpos,
        device=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True,
        zero_wrist=False,
        cat_direction=1
) -> torch.Tensor:
    """
    Convert robot qpos (body + wrists + hands) to body qpos tensor.
    """
    if isinstance(robot_qpos, np.ndarray):
        robot_qpos = torch.from_numpy(robot_qpos).float()
    if (device is not None) and (not robot_qpos.is_cuda):
        robot_qpos = robot_qpos.to(device)

    if robot_type == 'h1':
        qpos = robot_qpos[..., :19].float()
    elif robot_type in {'h1', 'h1hand', 'h1touch', 'h1strong'}:
        if ignore_wrist:  # * (, 19)
            qpos = torch.cat((robot_qpos[..., :15], robot_qpos[..., 40:44]), cat_direction).float()
        else:  # * (, 21)
            qpos = torch.cat((robot_qpos[..., :16], robot_qpos[..., 40:45]), cat_direction).float()
            if zero_wrist:
                qpos[..., 16] = 0
                qpos[..., 45] = 0
    else:
        raise ValueError(f'Invalid robot type: {robot_type}')
    return qpos


def convert_body_qpos_to_robot_qpos(
        body_qpos,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        ignore_wrist=True
):
    if robot_type == 'h1':
        qpos = body_qpos
    elif robot_type in {'h1', 'h1hand', 'h1touch', 'h1strong'}:
        qpos_shape = list(body_qpos.shape)
        qpos_shape[-1] = 61
        if isinstance(body_qpos, torch.Tensor):
            qpos = torch.zeros(qpos_shape).to(body_qpos.device)
        elif isinstance(body_qpos, np.ndarray):
            qpos = np.zeros(qpos_shape)
        else:
            raise NotImplementedError
        if ignore_wrist:  # * (, 19)
            qpos[..., :15] = body_qpos[..., :15]
            qpos[..., 16:20] = body_qpos[..., 15:19]
        else:  # * (, 21)
            qpos[..., :21] = body_qpos[..., :21]
    else:
        raise ValueError(f'Invalid robot type: {robot_type}')
    return qpos


def convert_smplx_joints_pose_to_robot(
        smplx_joints_pose: torch.Tensor,
        joints_init_orient=None,
        robot_type: Literal['h1', 'h1hand', 'h1touch', 'h1strong'] = 'h1hand',
        smplx_joints_dim=21,
        ignore_wrist=True,
) -> np.ndarray:
    """
    Convert SMPL-X body pose to robot pose.
    Since the robot in Humanoid-Bench env is position controlled, the
    robot pose is used as an action and will be executed in the env.
    SMPL-X body qpos --> robot qpos (used as an action).
    For actuators, total dimensions 61.
        - Body: [0: 21]
        - Left hand: [21: 40]
        - Right hand: [41: ]
    Params:
        - smplx_pose: smplx_clip body_pose/joints_orient part.
    """
    if smplx_joints_pose.is_cuda:
        smplx_joints_pose = smplx_joints_pose.detach().cpu()
    assert smplx_joints_pose.shape[1:] == torch.Size([smplx_joints_dim, 3])
    if robot_type == 'h1':
        BODY_JOINTS_INFO = SMPLX_BODY_2_H1
        robot_qpos = np.zeros((smplx_joints_pose.shape[0], 19))
        smplx_wrist_joints = ()
    elif robot_type in {'h1hand', 'h1touch', 'h1strong'}:
        BODY_JOINTS_INFO = SMPLX_BODY_2_H1HAND
        robot_qpos = np.zeros((smplx_joints_pose.shape[0], 61))  # * Only body, w/o hands.
        smplx_wrist_joints = (19, 20)
    else:
        raise ValueError(f'Unknown robot type: {robot_type}')

    curr_clip_len = smplx_joints_pose.shape[0]  # * bs * (args_rohm.clip_len - 2)
    if joints_init_orient is None:
        # * init_joints_orient = smplx_init_pose[:, 16:(16 + 63)].reshape(-1, 21, 3)
        joints_init_orient = init_smplx_body_pose(clip_len=curr_clip_len, only_joints_orient=True)
    else:
        batch_size = smplx_joints_pose.shape[0] // joints_init_orient.shape[0]
        if batch_size != 1:
            joints_init_orient = joints_init_orient.repeat(batch_size, 1, 1)

    smplx_joints_pose -= joints_init_orient
    for i in range(smplx_joints_dim):
        if (i not in BODY_JOINTS_INFO) or (ignore_wrist and (i in smplx_wrist_joints)):
            continue
        for joint_axis, joint_idx in BODY_JOINTS_INFO[i].items():  # * {'x': x, 'y': x, 'z': x}
            joint_axis_idx = XYZ_INDEX[joint_axis]
            coef = 1 if i in SMPLX_JOINTS_DIRECTION[joint_axis]['positive'] else -1
            robot_qpos[:, joint_idx] = smplx_joints_pose[:, i, joint_axis_idx] * coef

    return robot_qpos
