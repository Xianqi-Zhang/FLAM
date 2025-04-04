"""
@Project ：FLAM
@File    ：config_rohm.py
@Author  ：Xianqi-Zhang
@Date    ：2024/5/20 
"""
import os
import configargparse


def config_rohm():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'RoHM configuration'
    group = configargparse.ArgParser(formatter_class=arg_formatter,
                                     config_file_parser_class=cfg_parser,
                                     description=description,
                                     prog='')
    group.add_argument('--config', is_config_file=True, default='', help='config file path')
    group.add_argument('--device', default=0, type=int, help='Device id to use.')
    group.add_argument('--seed', default=0, type=int, help='For fixing random seed.')

    # * Diffusion setups.
    group.add_argument('--diffusion_steps_posenet', default=100, type=int,
                       help='diffusion time steps, default=1000')
    group.add_argument('--diffusion_steps_trajnet', default=100, type=int,
                       help='diffusion time steps')
    group.add_argument('--noise_schedule', default='cosine', choices=['linear', 'cosine'],
                       type=str, help='Noise schedule type')
    group.add_argument('--timestep_respacing_eval', default='',
                       help='if use ddim, set to ddimN, where N denotes ddim sampling steps',
                       type=str)
    group.add_argument('--sigma_small', default='True', type=lambda x: x.lower() in ['true', '1'],
                       help='Use smaller sigma values.')

    # * Path to AMASS and body model.
    group.add_argument('--body_model_path', type=str, help='path to smplx model',
                       default='./src/pkgs/RoHM/data/body_models/smplx_model')
    group.add_argument('--dataset_root', type=str, help='path to datas',
                       default='./src/pkgs/RoHM/datasets/AMASS_smplx_preprocessed')

    # * Model setups.
    group.add_argument('--clip_len', default=145, type=int, help='sequence length for each clip')
    group.add_argument('--repr_abs_only', default='True',
                       type=lambda x: x.lower() in ['true', '1'],
                       help='if True, only include absolute trajectory repr for TrajNet')
    t_path = './src/pkgs/RoHM/data/checkpoints/trajnet_checkpoint/model000450000.pt'
    t_c_path = './src/pkgs/RoHM/data/checkpoints/trajnet_control_checkpoint/model000400000.pt'
    p_path = './src/pkgs/RoHM/data/checkpoints/posenet_checkpoint/model000200000.pt'
    n_path = './src/pkgs/RoHM/data/eval_noise_smplx/smplx_noise_level_{}.pkl'
    group.add_argument('--model_path_trajnet', type=str, default=t_path)
    group.add_argument('--model_path_trajnet_control', type=str, default=t_c_path)
    group.add_argument('--model_path_posenet', type=str, default=p_path)
    group.add_argument('--noise_pkl_path', type=str, default=n_path)

    # * Input noise scaling setups.
    group.add_argument('--input_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
    group.add_argument('--noise_std_smplx_global_rot', default=3, type=float)
    group.add_argument('--noise_std_smplx_body_rot', default=3, type=float)
    group.add_argument('--noise_std_smplx_trans', default=0.03, type=float)
    group.add_argument('--noise_std_smplx_betas', default=0.1, type=float)
    group.add_argument('--load_noise', default='True', type=lambda x: x.lower() in ['true', '1'])
    group.add_argument('--load_noise_level', default=3, type=int)

    # * Test setups.
    group.add_argument('--batch_size', default=8, type=int, help='Batch size during test.')
    group.add_argument('--cond_fn_with_grad', default='True',
                       type=lambda x: x.lower() in ['true', '1'],
                       help='use test-time guidance or not')
    group.add_argument('--infill_traj', default='True', type=lambda x: x.lower() in ['true', '1'])
    group.add_argument('--traj_mask_ratio', default=0.1, type=float,
                       help='occlusion ratio for traj infilling, when traj is occlude, we assume '
                            'full body pose is also occluded')
    group.add_argument('--mask_scheme', default='full', type=str,
                       choices=['lower', 'upper', 'full'], help='occlusion scheme for poseNet')
    group.add_argument('--sample_iter', default=2, type=int,
                       help='inference iterations during test, default is 2 for results in paper')
    group.add_argument('--iter2_cond_noisy_traj', default='False',
                       type=lambda x: x.lower() in ['true', '1'],
                       help='in inference iteration>1, if TrajNet conditions on noisy input '
                            'instead of predicted traj from inderence iteration 1')
    group.add_argument('--iter2_cond_noisy_pose', default='True',
                       type=lambda x: x.lower() in ['true', '1'],
                       help='in inference iteration>1, if PoseNet conditions on noisy input '
                            'instead of predicted pose from inderence iteration 1')
    group.add_argument('--early_stop', default='False', type=lambda x: x.lower() in ['true', '1'],
                       help='if stop denoising earlier for PoseNet (for only 980 steps)')
    group.add_argument('--save_root', type=str,
                       default='./outputs/test_results/results_amass')

    # * Related file saved in RoHMDataStorage.save() in utils_rohm.py.
    save_data_path = './outputs/test_results/results_amass/'
    save_data_file_name = 'amass_fgTrue_m{}_n3_infill_t0.1_i2_i2tnFalse_i2pnTrue_esFalse_s0.pkl'
    group.add_argument('--saved_data_path', type=str,
                       default=os.path.join(save_data_path, save_data_file_name))

    # * Evaluation and Render.
    group.add_argument('--visualize', default='False', type=lambda x: x.lower() in ['true', '1'])
    group.add_argument('--vis_interval', default=100, type=int)
    group.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'])
    group.add_argument('--render_interval', default=100, type=int)
    group.add_argument('--render_save_dir', type=str,
                       default='./outputs/render_imgs/render_amass/')
    group.add_argument('--storage_path', type=str, default=None)

    args, _ = group.parse_known_args()
    return args

