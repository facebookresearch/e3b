# Exploration via Elliptical Episodic Bonuses

To install the dependencies, run:

```
conda create -n e3b python=3.8
conda activate e3b
pip install -r requirements.txt
```

Make sure to install MiniHack following the instructions [here](https://github.com/facebookresearch/minihack). 

To train an agent with E3B using the hyperparameters from the paper, run:

```
OMP_NUM_THREADS=1 python main.py  --learning_rate 0.0001 --model e3b --episodic_bonus_type elliptical-icm --savedir ./results/elliptical/ --env MiniHack-MultiRoom-N6-v0 --ridge 0.1 --reward_norm int --intrinsic_reward_coef 1.0 --seed 1
```

The file `sweep_slurm.py` will submit SLURM jobs for the experiments in the paper (may require some editing based on your computing infrastructure). It can also be run with the argument `--dry`, which will print out a list of commands instead. For example:

```
python sweep_slurm_neurips_2022.py --task train-elliptical 
```

trains E3B on all MiniHack tasks.

```
python sweep_slurm_neurips_2022.py --task train-noveld --dry
```

will print out all the commands to train NovelD (standard NovelD and the 3 variants described in the paper) on MiniHack.

The `sweep_slurm_icml_2023.py` file contains the commands to run the MiniHack experiments from the paper [A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs](https://arxiv.org/abs/2306.03236). This includes both the count-based global and episodic bonus experiments in Section 3 and the E3BxRND algorithm in Section 4. 


