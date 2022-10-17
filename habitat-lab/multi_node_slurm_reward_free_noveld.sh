# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=slurm/logs/%j.log
#SBATCH --error=slurm/logs/%j.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=5GB
#SBATCH --constraint=volta32gb
#SBATCH --partition=devlab
#SBATCH --time=72:00:00
#SBATCH --requeue

export MAGNUM_LOG=quiet

export MAGNUM_GPU_VALIDATION=ON
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}

module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.7.8-1-cuda.11.0

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR

echo $name

set -x


entropy=0.000005
bonus_coef=0.01
seed=1
tag=hm3d-noreward-noveld-sub-rgbd-bc_${bonus_coef}-entropy_${entropy}-seed_${seed}
echo $tag

srun python -u -m habitat_baselines.run \
     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_hm3d.yaml \
     --run-type train TENSORBOARD_DIR data/hm3d/tb/${tag} CHECKPOINT_FOLDER data/hm3d/ckpt/${tag} TASK_CONFIG.SEED ${seed} \
     TRAINER_NAME ddppo-noveld \
     RL.PPO.entropy_coef $entropy \
     RL.E2B.bonus_coef $bonus_coef \
     RL.E2B.inv_dynamics_epochs 3 \
     TOTAL_NUM_STEPS 5e8 \
     NUM_UPDATES -1 \
     NUM_CHECKPOINTS 100




