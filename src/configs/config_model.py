"""
@Project     ：FLAM
@File        ：config_model.py
@Author      ：Xianqi-Zhang
@Date        ：2024/12/24
@Last        : 2024/12/24
@Description : 
"""
import configargparse


def config_model():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'Model/Policy configuration'
    group = configargparse.ArgParser(formatter_class=arg_formatter,
                                     config_file_parser_class=cfg_parser,
                                     description=description,
                                     prog='')
    group.add_argument('--seed', default=0, type=int)
    group.add_argument('--obs', default='state', type=str)
    group.add_argument('--env_obs_type', default='privileged', type=str,
                       choices=['privileged', 'proprio'])
    group.add_argument('--checkpoint', default='./outputs', type=str)

    # * Hyperparameters: for Trainer.update_reward() in train.py
    group.add_argument('--gamma', default=1.0, type=float, help='refer to lambda in the paper.')
    group.add_argument('--expect_return', default=750, type=int,
                       help='750 for locomotion and 350 for manipulation.')
    group.add_argument('--reward_merge_type', default='add', type=str,
                       choices=['add', 'replace', 'merge'])

    # * Logging.
    group.add_argument('--wandb_name', default='walk_add_1_seed_0', type=str)  # * TODO
    group.add_argument('--wandb_project', default='FLAM', type=str)
    group.add_argument('--wandb_entity', default='YourEntity', type=str,
                       help='change to your entity name')
    group.add_argument('--wandb_silent', default=False, type=bool)
    group.add_argument('--disable_wandb', default=False, type=bool)
    group.add_argument('--save_csv', default=True, type=bool)
    group.add_argument('--save_video', default=True, type=bool)
    group.add_argument('--save_agent', default=True, type=bool)

    # * Evaluation.
    group.add_argument('--eval_episodes', default=10, type=int)
    group.add_argument('--eval_freq', default=10_000, type=int)

    # * Training.
    group.add_argument('--steps', default=2_000_000, type=int)
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--reward_coef', default=0.1, type=float)
    group.add_argument('--value_coef', default=0.1, type=float)
    group.add_argument('--consistency_coef', default=20, type=int)
    group.add_argument('--rho', default=0.5, type=float)
    group.add_argument('--lr', default=3e-4, type=float)
    group.add_argument('--enc_lr_scale', default=0.3, type=float)
    group.add_argument('--grad_clip_norm', default=20, type=int)
    group.add_argument('--tau', default=0.01, type=float)
    group.add_argument('--discount_denom', default=5, type=int)
    group.add_argument('--discount_min', default=0.95, type=float)
    group.add_argument('--discount_max', default=0.995, type=float)
    group.add_argument('--buffer_size', default=2_000_000, type=int)
    group.add_argument('--exp_name', default='default', type=str)
    group.add_argument('--data_dir', default=None, type=str)

    # * Reward Update.
    group.add_argument('--floor_env', default=False, type=bool, help='for balance, stair, slide')
    group.add_argument('--return_threshold', default=40, type=int)
    group.add_argument('--stable_pretrain_step', default=500_000, type=int)

    # * Planning.
    group.add_argument('--mpc', default=True, type=bool)
    group.add_argument('--iterations', default=6, type=int)
    group.add_argument('--num_samples', default=512, type=int)
    group.add_argument('--num_elites', default=64, type=int)
    group.add_argument('--num_pi_trajs', default=24, type=int)
    group.add_argument('--horizon', default=3, type=int)
    group.add_argument('--min_std', default=0.05, type=float)
    group.add_argument('--max_std', default=2, type=int)
    group.add_argument('--temperature', default=0.5, type=float)

    # * Actor.
    group.add_argument('--log_std_min', default=-10, type=int)
    group.add_argument('--log_std_max', default=2, type=int)
    group.add_argument('--entropy_coef', default=1e-4, type=float)

    # * Critic.
    group.add_argument('--num_bins', default=101, type=int)
    group.add_argument('--vmin', default=-10, type=int)
    group.add_argument('--vmax', default=10, type=float)

    # * Architecture.
    group.add_argument('--model_size', default=5, type=int)
    group.add_argument('--num_enc_layers', default=2, type=int)
    group.add_argument('--enc_dim', default=256, type=int)
    group.add_argument('--num_channels', default=32, type=int)
    group.add_argument('--mlp_dim', default=512, type=int)
    group.add_argument('--latent_dim', default=512, type=int)
    group.add_argument('--task_dim', default=96, type=float)
    group.add_argument('--num_q', default=5, type=int)
    group.add_argument('--dropout', default=0.01, type=float)
    group.add_argument('--simnorm_dim', default=8, type=int)

    args, _ = group.parse_known_args()
    return args
