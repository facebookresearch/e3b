# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#!/bin/bash

python -u -m habitat_baselines.run \
     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_hm3d.yaml \
     --run-type eval \
     NUM_ENVIRONMENTS 20 \
     EVAL_CKPT_PATH_DIR data/hm3d/ckpt/${1}/ckpt.45.pth \
     VIDEO_DIR data/hm3d/videos/${1}/ \
     TEST_EPISODE_COUNT 100 \
     TORCH_GPU_ID 1 \
     TRAINER_NAME ddppo-noveld \
     TENSORBOARD_DIR data/hm3d/tb/${1}/
