# E3B for reward-free exploration on Habitat

Here is the code for running E3B, RND, ICM and NovelD for reward-free exploration on Habitat.

First, you will need to install the Habitat simulator. To do this, follow the instructions from the official Habitat repo [here](https://github.com/facebookresearch/habitat-lab), and make sure you can run the DD-PPO baseline. You will also need to download the HM3D dataset [here](https://github.com/facebookresearch/habitat-matterport3d-dataset). When training the agents, you will need to change [this line](https://github.com/facebookresearch/e3b/blob/5f78e88509737cc9fdeb01c55be9487347802a94/habitat-lab/configs/tasks/pointnav_hm3d.yaml#L35) to point to where you saved the HM3D data.

To run E3B locally on a single machine, do:

```
./run_local_reward_free.sh
```

This is useful for debugging, but is too slow otherwise. To run for enough steps, you will need to run distributed over multiple GPUs.

To run E3B, ICM, RND, NovelD with 32 GPUs on a Slurm cluster, do:

```
sbatch multi_node_reward_free_{e3b, icm, rnd, noveld}.sh
```

Depending on your cluster setup, you may need to edit the `--ntasks-per-node` and `--nodes` arguments in the script (the total number of GPUs used is these two numbers multiplied together). 

With 32 GPUs this will take about 3 days for 1 run. You can edit the seed parameter in the file as well as any other hyperparameters.
You may want to change the CHECKPOINT_FOLDER and TENSORBOARD_DIR arguments to where you want to save the results. 

To evaluate a trained model and generate videos, first open `run_checkpt.sh` and change EVAL_CKPT_PATH_DIR and TENSORBOARD_DIR to where your checkpoints and Tensorboard files are saved, and VIDEO_DIR to where you want the videos to be saved.
The coverage (the metric used to evaluate reward-free exploration) can be seen in the tensorboard logs after evaluation. 

All the code for E3B is in `habitat_baselines/rl/ppo/e3b_trainer.py`, and the code for the other agents is in the same folder. 

