"""
@Project     ：FLAM
@File        ：utils_smplx.py
@Author      ：Xianqi-Zhang
@Date        ：2024/6/15
@Last        : 2024/7/23
@Description : SMPL-X utils functions. Some code is based on RoHM.
"""
import os
import torch
import trimesh
import pyrender
import numpy as np

from src.pkgs.RoHM.utils.other_utils import LIMBS_BODY_SMPL
from src.pkgs.RoHM.utils.render_util import material_joint_vis, material_skel_vis, material_debug, \
    material_skel_debug, material_body_rec_vis, create_render_cam, create_floor, \
    create_pyrender_scene, create_pyrender_mesh, create_pyrender_skel
from src.pkgs.RoHM.data_loaders.motion_representation import cano_seq_smplx, get_repr_smplx, \
    recover_from_repr_smpl
from src.utils.utils_rohm import rohm_get_cam_trans, rohm_render_img


# * ------------------------------------------------------ * #
# * ----------------------- Render ----------------------- * #
# * ------------------------------------------------------ * #

def render_smplx_skelecton(
        joints,
        H=1080,
        W=1920,
        cam_x=960,
        cam_y=540,
        save_path='./outputs/render_imgs/skelecton',
        img_name='skeleton.png',
        render_ground=True,
        debug=False
):
    cam_trans = rohm_get_cam_trans(debug=debug)
    camera, camera_pose, light = create_render_cam(cam_x=cam_x, cam_y=cam_y)
    body_scene = create_pyrender_scene(camera, camera_pose, light)
    if render_ground:
        # * Ground.
        ground_mesh = create_floor(cam_trans)
        body_scene.add(ground_mesh, 'mesh')

    skeleton_mesh_list = []
    joints_num = joints.shape[0]
    for j in range(joints_num):
        sphere = trimesh.creation.icosphere(radius=0.025)
        transformation = np.identity(4)
        transformation[:3, 3] = joints[j]
        sphere.apply_transform(transformation)
        sphere.apply_transform(np.linalg.inv(cam_trans))
        material = material_debug if debug else material_joint_vis  # * red: debug, blue: else.
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
        skeleton_mesh_list.append(sphere_mesh)

    for index_pair in LIMBS_BODY_SMPL:
        if (index_pair[0] >= joints_num) or (index_pair[1] >= joints_num):
            continue
        p1 = joints[index_pair[0]]
        p2 = joints[index_pair[1]]

        segment = np.array([p1, p2])
        cyl = trimesh.creation.cylinder(0.01, height=None, segment=segment)
        cyl.apply_transform(np.linalg.inv(cam_trans))

        material = material_skel_debug if debug else material_skel_vis
        cyl_mesh_rec = pyrender.Mesh.from_trimesh(cyl, material=material)
        skeleton_mesh_list.append(cyl_mesh_rec)

    # * Create scene and add skeleton.
    skel_scene = create_pyrender_scene(camera, camera_pose, light)
    for mesh in skeleton_mesh_list:
        skel_scene.add(mesh, 'pred_joint')

    # * Render.
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    os.makedirs(save_path, exist_ok=True)
    # color_skel = render_img(r, skel_scene, alpha=1.0)
    # color_skel = color_skel.transpose(Image.FLIP_LEFT_RIGHT)
    # color_skel.save(os.path.join(img_save_path, img_name))
    rohm_render_img(r, body_scene, skel_scene, save_path, img_name)


def render_smplx_body(
        smplx_verts,
        smplx_model,
        H=1080,
        W=1920,
        cam_x=960,
        cam_y=540,
        save_path='./outputs/render_imgs/body',
        img_name='body.png',
        render_ground=True,
        debug=False
):
    cam_trans = rohm_get_cam_trans(debug=debug)
    camera, camera_pose, light = create_render_cam(cam_x=cam_x, cam_y=cam_y)
    body_scene = create_pyrender_scene(camera, camera_pose, light)
    if render_ground:
        # * Ground.
        ground_mesh = create_floor(cam_trans)
        body_scene.add(ground_mesh, 'mesh')
    # * Body.
    body_mesh = create_pyrender_mesh(smplx_verts, smplx_model.faces, cam_trans,
                                     material_body_rec_vis)
    body_scene.add(body_mesh, 'mesh')
    # * Render.
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    os.makedirs(save_path, exist_ok=True)
    rohm_render_img(r, body_scene, None, save_path, img_name)


def render_smplx_body_skeleton(
        smpl_verts,
        joints,
        smplx_model,
        offscreen_render=None,
        flag_full_mask_area=False,
        mask_scheme=None,
        mask_joint_id=tuple(),
        contact_lbl=None,
        material_body=material_body_rec_vis,
        save_path='./outputs/render_imgs',
        img_name='smplx_body_skeleton.png',
        sekl_name='skeleton',
        alpha=1.0,
        add_occ_joints=False,
        colors_verts=None,
        cam_x=960,
        cam_y=540,
        H=1080,
        W=1920,
        cam_trans=None,
        render_skeleton=True,
        render_ground=True,
        debug=False
):
    cam_trans = cam_trans if cam_trans is not None else rohm_get_cam_trans(debug=debug)
    camera, camera_pose, light = create_render_cam(cam_x=cam_x, cam_y=cam_y)
    body = create_pyrender_scene(camera, camera_pose, light)
    if render_ground:
        ground_mesh = create_floor(cam_trans)
        body.add(ground_mesh, 'mesh')
    body_mesh = create_pyrender_mesh(smpl_verts, smplx_model.faces, cam_trans, material_body,
                                     colors_verts)
    body.add(body_mesh, 'mesh')
    # * Add body skeleton.
    skel = None
    if render_skeleton:
        skeleton_mesh_list = create_pyrender_skel(
            joints=joints,
            add_trans=np.linalg.inv(cam_trans),
            mask_scheme=mask_scheme,
            mask_joint_id=mask_joint_id,
            add_occ_joints=add_occ_joints,
            in_mask_area=flag_full_mask_area,
            contact_lbl=contact_lbl
        )
        skel = create_pyrender_scene(camera, camera_pose, light)
        for mesh in skeleton_mesh_list:
            skel.add(mesh, sekl_name)

    if offscreen_render is None:
        offscreen_render = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H,
                                                      point_size=1.0)
    rohm_render_img(offscreen_render, body, skel, save_path, img_name, alpha=alpha)


# * ------------------------------------------------------ * #
# * ---------- SMPL-X data (smplx_clip) process ---------- * #
# * ------------------------------------------------------ * #

def convert_smplx_clip_to_params_dict(smplx_clip, device):
    bs = smplx_clip.shape[0]
    smplx_params_dict = {  # * The body_pose is joint orientation.
        'global_orient': smplx_clip[:, 0:3],  # * 3
        'transl': smplx_clip[:, 3:6],  # * 3
        'betas': smplx_clip[:, 6:16],  # * 10
        'body_pose': smplx_clip[:, 16:(16 + 63)].reshape(-1, 21, 3),  # * 63, 21 * 3

        'jaw_pose': torch.zeros(bs, 3).to(device),
        'leye_pose': torch.zeros(bs, 3).to(device),
        'reye_pose': torch.zeros(bs, 3).to(device),
        # 'hand_pose': torch.zeros(bs, 90).to(device),
        'left_hand_pose': torch.zeros(bs, 45).to(device),
        'right_hand_pose': torch.zeros(bs, 45).to(device),
        'expression': torch.zeros(bs, 10).to(device),
    }
    return smplx_params_dict


def convert_smplx_params_dict_to_clip(smplx_params_dict, device) -> torch.Tensor:
    smplx_clip = [
        torch.from_numpy(value).to(device).float() for value in smplx_params_dict.values()
    ]
    return torch.cat(smplx_clip, dim=1)


def convert_smplx_clip_to_joints_clip(
        smplx_clip: torch.Tensor,
        smplx_model,
        device,
        cano=True,
) -> [np.ndarray, np.ndarray]:
    """
    Convert smplx_clip (joint orientation) to joint canonical position.
    joint qpos --> joint xpos.
    Similar function as motion_representation.recover_from_repr_smpl()

    Params:
        - smplx_clip: info with body_pose, torch.Tensor or np.ndarray
    Returns:
        - joint_positions: joint global positions for skeleton render.
        - smplx_output: type smplx.utils.SMPLXOutput
            - vertices
            - joints: shape (bs, 127, 3)
            - global_orient
            - transl
            - v_shaped
    """
    body_params = convert_smplx_clip_to_params_dict(smplx_clip, device)
    smplx_output = smplx_model(**body_params)
    smplx_joints = smplx_output.joints[:, 0:22]  # Joint positions [bs, 22, 3].
    smplx_verts = smplx_output.vertices

    smplx_clip = smplx_clip.cpu().numpy()
    joints_clip = smplx_joints.cpu().numpy()
    smplx_verts = smplx_verts.cpu().numpy()

    if cano:
        # * Canonicalization: change position and orientation.
        # * Perform canonicalization to the original motion sequence.
        # * IMPORTANT: if not canonicalize, the rendered body and
        # * skeleton will be 180 degrees different.
        joints_clip, smplx_verts = cano_joints_and_smplx_clip(joints_clip, smplx_clip,
                                                              smplx_model, device)
    return joints_clip, smplx_verts


def convert_smplx_clip_to_repr(joints_clip, smplx_clip, smplx_model, device, to_tensor=False):
    """
    Params:
        joint_clip: shape [clip_len, 22, 3]
        smplx_clip: shape [clip_len, 178]
    """
    smplx_params_dict = {
        'global_orient': smplx_clip[:, 0:3],  # * 3
        'transl': smplx_clip[:, 3:6],  # * 3
        'betas': smplx_clip[:, 6:16],  # * 10
        'body_pose': smplx_clip[:, 16:(16 + 63)],  # * 63, 21 * 3
    }
    # * cano_positions: shape [clip_len (145), 22, 3]
    # * cano_smplx_params_dict.keys(): ['global_orient', 'transl', 'betas', 'body_pose']
    cano_positions, cano_smplx_params = cano_seq_smplx(positions=joints_clip,
                                                       smplx_params_dict=smplx_params_dict,
                                                       smpl_model=smplx_model,
                                                       device=device)
    repr_dict = get_repr_smplx(positions=cano_positions,
                               smplx_params_dict=cano_smplx_params,
                               feet_vel_thre=5e-5)  # * motion representation, a dict of reprs

    if to_tensor:
        # * For repr recover --> recover_from_repr_smpl().
        # * Add batch_size dim, i.e., unsqueeze(0), and convert to torch.Tensor.
        repr_dict = {k: torch.from_numpy(v).to(device).float().unsqueeze(0) for k, v in
                     repr_dict.items()}

    return repr_dict, cano_positions, cano_smplx_params


def cano_joints_and_smplx_clip(joints_clip: np.ndarray, smplx_clip: np.ndarray, smplx_model,
                               device, to_numpy=True):
    # * Change position and orientation.
    # * Perform canonicalization to the original motion sequence.
    repr_dict, cano_positions, cano_smplx_params = convert_smplx_clip_to_repr(
        joints_clip, smplx_clip, smplx_model, device, to_tensor=True
    )
    joints_clip, smplx_verts = recover_from_repr_smpl(repr_dict, smplx_model, to_numpy=to_numpy)

    # * Remove batch_size dimension.
    cano_joints_clip = joints_clip.squeeze(0)  # * (clip_len-1, 22, 3)
    cano_smplx_verts = smplx_verts.squeeze(0)  # * (clip_len-1, 10475, 3)
    return cano_joints_clip, cano_smplx_verts
