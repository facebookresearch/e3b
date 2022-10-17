# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


#!/bin/bash

rm -rf data/tb/test/
rm -rf data/ckpt/test/

entropy=0.000005
bonus_coef=1.0
seed=1
#tag=hm3d-noreward-e3b-rgbd-bc_${bonus_coef}-entropy_${entropy}-seed_${seed}
tag=test
echo $tag

/private/home/mikaelhenaff/.conda/envs/habitat/bin/python -u -m habitat_baselines.run \
     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_hm3d.yaml \
     --run-type train TENSORBOARD_DIR data/hm3d/tb/${tag} CHECKPOINT_FOLDER data/hm3d/ckpt/${tag} TASK_CONFIG.SEED ${seed} \
     TRAINER_NAME ddppo-e3b \
     RL.PPO.entropy_coef $entropy \
     RL.E2B.bonus_coef $bonus_coef \
     RL.E2B.inv_dynamics_epochs 3 \
     TOTAL_NUM_STEPS 5e8 \
     NUM_UPDATES -1 \
     NUM_CHECKPOINTS 100


