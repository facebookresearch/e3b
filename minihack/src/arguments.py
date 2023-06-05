# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import argparse

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

# General Settings.
parser.add_argument('--env', type=str, default='MiniHack-KeyRoom-S5-v0',
                    help='Gym environment. Can include other MiniHack envs or Vizdoom ones')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')
parser.add_argument('--num_input_frames', default=1, type=int,
                    help='Number of input frames to the model and state embedding including the current frame \
                    When num_input_frames > 1, it will also take the previous num_input_frames - 1 frames as input.')
parser.add_argument('--run_id', default=0, type=int,
                    help='Run id used for running multiple instances of the same HP set \
                    (instead of a different random seed since torchbeast does not accept this).')
parser.add_argument('--seed', default=0, type=int,
                    help='Environment seed.')
parser.add_argument('--save_interval', default=30, type=int, metavar='N',
                    help='Time interval (in minutes) at which to save the model.')    
parser.add_argument('--checkpoint_num_frames', default=10000000, type=int,
                    help='Number of frames for checkpoint to load.')
parser.add_argument('--device', default=0, type=int, help='GPU device')
parser.add_argument('--checkpointpath', default='', type=str)
parser.add_argument('--clip_rewards', default=1, type=int, help='Clip rewards to [-1, 1]. Set 1 for true, 0 for false.')
parser.add_argument('--reward_norm', default='int', type=str,
                    help='Running normalization for reward. Can be int (normalize intrinsic reward), \
                    ext (normalize extrinsic reward) or all (normalize the sum of the two)')
parser.add_argument('--decay_lr', default=0, type=int, help='decay learning rate (1 for true, 0 for false)')
parser.add_argument('--num_contexts', default=-1, type=int,
                    help='Number of distinct contexts. Set to -1 for infinite.')



# Training settings.
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint.')
parser.add_argument('--savedir', default='./results/',
                    help='Root dir where experiment data will be saved.')
parser.add_argument('--num_actors', default=256, type=int, metavar='N',
                    help='Number of actors.')
parser.add_argument('--total_frames', default=5e7, type=int, metavar='T',
                    help='Total environment frames to train for.')
parser.add_argument('--batch_size', default=8, type=int, metavar='B',
                    help='Learner batch size.')
parser.add_argument('--unroll_length', default=80, type=int, metavar='T',
                    help='The unroll length (time dimension).')
parser.add_argument('--queue_timeout', default=1, type=int,
                    metavar='S', help='Error timeout for queue.')
parser.add_argument('--num_buffers', default=80, type=int,
                    metavar='N', help='Number of shared-memory buffers.')
parser.add_argument('--num_threads', default=4, type=int,
                    metavar='N', help='Number learner threads.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')
parser.add_argument('--max_grad_norm', default=40., type=float, metavar='MGN',
                    help='Max norm of gradients.')
parser.add_argument('--hidden_dim', default=1024, type=int)
parser.add_argument('--msg_model', default="lt_cnn", type=str,
                    help='Architecture for reading messages')
parser.add_argument('--use_lstm', default=1, type=int,
                    help='Use a lstm version of policy network.')
parser.add_argument('--use_lstm_intrinsic', action='store_true',
                    help='Use a lstm version of intrinsic embedding network.')
parser.add_argument('--state_embedding_dim', default=256, type=int,
                    help='Embedding dimension of last layer of network')
parser.add_argument('--scale_fac', default=0.1, type=float,
                    help='coefficient for scaling visitation count difference')


# Loss settings.
parser.add_argument('--entropy_cost', default=0.005, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')
parser.add_argument('--forward_loss_coef', default=10.0, type=float,
                    help='Coefficient for the forward dynamics loss.')
parser.add_argument('--inverse_loss_coef', default=1.0, type=float,
                    help='Coefficient for the inverse dynamics loss.')
parser.add_argument('--pg_loss_coef', default=1.0, type=float)


# Optimizer settings.
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--predictor_learning_rate', default=0.0001, type=float,
                    metavar='LR', help='Learning rate for RND predictor.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon.')

# Exploration Settings.
parser.add_argument('--intrinsic_reward_coef', default=1.0, type=float,
                    help='Coefficient for the intrinsic reward. \
                    This weighs the intrinsic reaward against the extrinsic one. \
                    Should be larger than 0.')
parser.add_argument('--rnd_loss_coef', default=1.0, type=float,
                    help='Coefficient for the RND loss coefficient relative to the IMPALA one.')
parser.add_argument('--count_reward_type', default='ind', type=str)
parser.add_argument('--encoder_momentum_update', default=1.0, type=float)
parser.add_argument('--episodic_bonus_type', default='counts-pos', type=str)
parser.add_argument('--global_bonus_type', default='none', type=str)
parser.add_argument('--bonus_combine', default='mult', type=str)
parser.add_argument('--global_bonus_coeff', default=1.0, type=float)
parser.add_argument('--ridge', default=0.1, type=float, help='covariance matrix regularizer for E3B')

# Singleton Environments.
parser.add_argument('--fix_seed', action='store_true',
                    help='Fix the environment seed so that it is \
                    no longer procedurally generated but rather the same layout every episode.')
parser.add_argument('--env_seed', default=1, type=int,
                    help='The seed used to generate the environment if we are using a \
                    singleton (i.e. not procedurally generated) environment.')
parser.add_argument('--no_reward', action='store_true',
                    help='No extrinsic reward. The agent uses only intrinsic reward to learn.')

# Training Models.
parser.add_argument('--model', default='vanilla', type=str,
                    help='Model used for training the agent.')

parser.add_argument('--dropout', default=0.0, type=float)

# Baselines for AMIGo paper.
parser.add_argument('--use_fullobs_policy', action='store_true',
                    help='Use a full view of the environment as input to the policy network.')
parser.add_argument('--use_fullobs_intrinsic', action='store_true',
                    help='Use a full view of the environment for computing the intrinsic reward.')
parser.add_argument('--target_update_freq', default=2, type=int,
                    help='Number of time steps for updating target')
parser.add_argument('--init_num_frames', default=1e6, type=int,
                    help='Number of frames for updating teacher network')
parser.add_argument('--planning_intrinsic_reward_coef', default=0.5, type=float,
                    help='Coefficient for the planning intrinsic reward. \
                    This weighs the intrinsic reaward against the extrinsic one. \
                    Should be larger than 0.')
parser.add_argument('--ema_momentum', default=1.0, type=float,
                    help='Coefficient for the EMA update of the RND network')
