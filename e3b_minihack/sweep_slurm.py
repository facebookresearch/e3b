#!/usr/bin/env python3
import numpy as np
import numpy.random as npr

import time
import os
import sys
import argparse
import pdb
import itertools
from subprocess import Popen, DEVNULL



        


# creates commands and job file names for a grid search over lists of hyperparameters
class Overrides(object):
    def __init__(self):
        self.args = []

    def add(self, arg_name, arg_values):
        self.args.append([{'name': arg_name, 'value': v} for v in arg_values])
                
    def parse(self, basecmd, cmd_format='argparse'):
        cmd, job = [], []
        for combos in itertools.product(*self.args):
            c = basecmd
            j = 'job'
            d = dict()
            for arg in combos:
                if cmd_format == 'argparse':
                    c += f" --{arg['name']} {str(arg['value'])}"
                elif cmd_format == 'hydra':
                    c += f" {arg['name']}={str(arg['value'])}"
                if arg['name'] != 'savedir':
                    j += f"_{arg['name']}={str(arg['value'])}"
                d[arg['name']] = arg['value']
            cmd.append(c)
            job.append(j)                
        return cmd, job


# copies the code before the sweep is run 
def make_code_snapshot(savedir):
    snap_dir = f'{savedir}/code/'
    dirs_to_copy = ['.', 'src', 'src/algos', 'src/core']
    
    def copy_dir(dir, pat):
        dst_dir = f'{snap_dir}/{dir}/'
        os.system(f'mkdir -p {dst_dir}')
        os.system(f'cp {dir}/{pat} {dst_dir}/')

    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        
        
    
    

def write_slurm_script(name, cmd, partition, device=0, ncpu=1):
    assert device in [-1, 0]
    scriptfile = f'slurm/scripts/run.{name}.sh'
    slurmfile = f'slurm/scripts/run.{name}.slrm'
    os.system('mkdir -p slurm/scripts/')
    with open(slurmfile, 'w') as s:
        s.write("#!/bin/sh\n")
        s.write(f"#SBATCH --job-name={name}\n")
        s.write(f"#SBATCH --output=slurm/stdout/{name}.%j\n")
        s.write(f"#SBATCH --error=slurm/stderr/{name}.%j\n")
        s.write(f"#SBATCH --partition={partition}\n")
        s.write("#SBATCH --mem=200000\n")
        s.write("#SBATCH --time=4320\n")
        s.write("#SBATCH --nodes=1\n")
        s.write(f"#SBATCH --cpus-per-task={ncpu}\n")
        s.write("#SBATCH --ntasks-per-node=1\n")
        s.write("#SBATCH --requeue\n")
        if device >= 0: 
            s.write("#SBATCH --gres=gpu:1\n")
        elif device == -1:
            s.write("#SBATCH --gres=gpu:0\n")
            s.write("#SBATCH --constraint=pascal")        
        s.write(f"srun sh {scriptfile}\n")

    with open(scriptfile, 'w') as s:
        s.write("#!/bin/sh\n")
        s.write("nvidia-smi\n")
        s.write(f"cd {os.getcwd()}\n")
        s.write("source activate my_pytorch_env\n")
        s.write(f"{cmd}\n")
        s.write("nvidia-smi\n")
        




ALL_TASKS = ['MiniHack-KeyRoom-S5-v0',
         'MiniHack-MultiRoom-N4-Locked-v0',
         'MiniHack-MultiRoom-N6-Locked-v0',
         'MiniHack-MultiRoom-N6-Lava-v0', 
         'MiniHack-MultiRoom-N6-Lava-OpenDoor-v0', 
         'MiniHack-MultiRoom-N6-LavaMonsters-v0',
         'MiniHack-MultiRoom-N10-v0',
         'MiniHack-MultiRoom-N10-OpenDoor-v0',
         'MiniHack-MultiRoom-N10-Lava-v0',
         'MiniHack-MultiRoom-N10-Lava-OpenDoor-v0',
         'MiniHack-LavaCrossingS19N13-v0',
         'MiniHack-LavaCrossingS19N17-v0',
         'MiniHack-Labyrinth-Small-v0',
         'MiniHack-Labyrinth-Big-v0',
         'MiniHack-LavaCross-Levitate-Potion-Pickup-v1',
         'MiniHack-Levitate-Potion-v1', 
         'MiniHack-Levitate-Ring-v1',
         'MiniHack-Levitate-Boots-v1',                 
         'MiniHack-WoD-Medium-v1',
         'MiniHack-Freeze-Horn-v1',
         'MiniHack-Freeze-Wand-v1',
         'MiniHack-Freeze-Random-v1'             
         ]


TUNE_TASKS = ['MiniHack-KeyRoom-S5-v0',
         'MiniHack-MultiRoom-N10-v0',
         'MiniHack-LavaCrossingS19N17-v0',
         'MiniHack-Labyrinth-Small-v0',
         'MiniHack-Levitate-Potion-v1', 
         'MiniHack-WoD-Medium-v1',
         'MiniHack-Freeze-Horn-v1',
         ]

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train-elliptical')
    parser.add_argument('--queue', default='')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--indx', type=int, default=-1)
    args = parser.parse_args()

    overrides = Overrides()
    

    # full E3B on MiniHack
    if args.task == 'train-elliptical':
        SAVEDIR = './results/elliptical/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b'])
        overrides.add('episodic_bonus_type', ['elliptical-icm'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('ridge', [0.1])
        overrides.add('reward_norm', ['int'])
        overrides.add('intrinsic_reward_coef', [1.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

        
    # ablation: E3B (non-episodic), i.e. with lifelong novelty bonus
    elif args.task == 'train-elliptical-lifelong':
        SAVEDIR = './results/elliptical_lifelong/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b'])
        overrides.add('episodic_bonus_type', ['elliptical-icm-lifelong'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('ridge', [0.1])
        overrides.add('reward_norm', ['int'])
        overrides.add('intrinsic_reward_coef', [10.0]) # 10.0 was better than 1.0 on tuning tasks
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

        

    # ablation: E3B with fixed random encoder 
    elif args.task == 'train-elliptical-rand-encoder':
        SAVEDIR = './results/elliptical_rand_encoder/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b'])
        overrides.add('predictor_learning_rate', [0.0])
        overrides.add('episodic_bonus_type', ['elliptical-icm'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('ridge', [0.1])
        overrides.add('reward_norm', ['int'])
        overrides.add('intrinsic_reward_coef', [1.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # ablation: E3B with policy network encoder
    elif args.task == 'train-elliptical-policy-encoder':
        SAVEDIR = './results/elliptical_policy_encoder/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b'])
        overrides.add('predictor_learning_rate', [0.0])
        overrides.add('episodic_bonus_type', ['elliptical-policy'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('ridge', [0.1])
        overrides.add('reward_norm', ['int'])
        overrides.add('intrinsic_reward_coef', [1.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        


    # E3B on Vizdoom
    elif args.task == 'train-elliptical-vizdoom':
        SAVEDIR = './results/vizdoom/elliptical_final/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b'])
        overrides.add('episodic_bonus_type', ['elliptical-icm'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ['VizdoomMyWayHomeDense-v0', 'VizdoomMyWayHomeSparse-v0', 'VizdoomMyWayHomeVerySparse-v0'])
        overrides.add('ridge', [0.1])
        overrides.add('reward_norm', ['none'])
        overrides.add('hidden_dim', [288])
        overrides.add('intrinsic_reward_coef', [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 0.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        

        

                

    # NovelD with 4 variants for the episodic bonus:
    # -standard (episodic_bonus_type = counts-obs)
    # -symbolic image (episodic_bonus_type = counts-glyphs)
    # -(x, y) coordinates (episodic_bonus_type = counts-pos)
    # -message (episodic_bonus_type = counts-msg)
    elif args.task == 'train-noveld':
        SAVEDIR = './results/noveld/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['bebold'])
        overrides.add('episodic_bonus_type', ['counts-glyphs', 'counts-pos', 'counts-msg', 'counts-obs'])
        overrides.add('count_reward_type', ['ind'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('reward_norm', ['int'])
        overrides.add('intrinsic_reward_coef', [10.0]) 
        overrides.add('clip_rewards', [1]) 
        overrides.add('scale_fac', [0.1]) # tried 0.5, 0.1 in early experiments, 0.1 worked best
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # NovelD on Vizdoom 
    elif args.task == 'train-noveld-vizdoom':
        SAVEDIR = './results/vizdoom/noveld/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0005])
        overrides.add('model', ['bebold'])
        overrides.add('episodic_bonus_type', ['counts-img'])
        overrides.add('count_reward_type', ['ind'])
        overrides.add('total_frames', [int(1e7)])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('hidden_dim', [288])
        overrides.add('entropy_cost', [0.005, 0.0005])
        overrides.add('env', ['VizdoomMyWayHomeDense-v0', 'VizdoomMyWayHomeSparse-v0', 'VizdoomMyWayHomeVerySparse-v0'])
        overrides.add('intrinsic_reward_coef', [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        

        
    # train RIDE on MiniHack
    elif args.task == 'train-ride':
        SAVEDIR = './results/ride/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['ride'])
        overrides.add('episodic_bonus_type', ['counts-obs'])
        overrides.add('count_reward_type', ['ind'])
        overrides.add('forward_loss_coef', [1.0]) # from MiniHack paper
        overrides.add('inverse_loss_coef', [0.1]) # MiniHack paper uses 0.1, also try 1.0
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('intrinsic_reward_coef', [0.001, 0.01, 0.1, 1.0]) # MiniHack paper uses 0.1, sweep around
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # train RIDE on Vizdoom
    elif args.task == 'train-ride-vizdoom':
        SAVEDIR = './results/vizdoom/ride/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0005])
        overrides.add('model', ['ride'])
        overrides.add('episodic_bonus_type', ['counts-img'])
        overrides.add('count_reward_type', ['ind'])
        overrides.add('forward_loss_coef', [0.5]) # from author
        overrides.add('inverse_loss_coef', [0.8]) # from author
        overrides.add('pg_loss_coef', [0.1]) # from author
        overrides.add('total_frames', [int(1e7)])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ['VizdoomMyWayHomeDense-v0', 'VizdoomMyWayHomeSparse-v0', 'VizdoomMyWayHomeVerySparse-v0'])
        overrides.add('intrinsic_reward_coef', [1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # train RND on MiniHack
    elif args.task == 'train-rnd':
        SAVEDIR = './results/rnd/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['rnd'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('intrinsic_reward_coef', [0.001]) 
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

    # train ICM on MiniHack
    elif args.task == 'train-icm':
        SAVEDIR = './results/icm/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['curiosity'])
        overrides.add('episodic_bonus_type', ['counts-obs'])
        overrides.add('forward_loss_coef', [1.0]) # from MiniHack paper
        overrides.add('inverse_loss_coef', [0.1]) # MiniHack paper uses 0.1
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ALL_TASKS)
        overrides.add('intrinsic_reward_coef', [0.1]) 
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # train ICM on Vizdoom
    elif args.task == 'train-icm-vizdoom':
        SAVEDIR = './results/vizdoom/icm/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0005])
        overrides.add('model', ['curiosity'])
        overrides.add('episodic_bonus_type', ['counts-img'])
        overrides.add('entropy_cost', [0.005]) # from author
        overrides.add('forward_loss_coef', [0.2]) # from author
        overrides.add('inverse_loss_coef', [0.8]) # from author
        overrides.add('total_frames', [int(1e7)])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('env', ['VizdoomMyWayHomeDense-v0', 'VizdoomMyWayHomeSparse-v0', 'VizdoomMyWayHomeVerySparse-v0'])
        overrides.add('intrinsic_reward_coef', [0.001, 0.003, 0.01, 0.03]) 
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

        
        
        
    # train IMPALA on MiniHack
    elif args.task == 'train-impala':
        SAVEDIR = './results/impala/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['vanilla'])
        overrides.add('env', ALL_TASKS)
        overrides.add('savedir', [SAVEDIR])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        
        
        
        
        
        

    os.system('mkdir -p slurm/stdout slurm/stderr')
#    cmds = [cmd + f' --device {args.device}' for cmd in cmds]
    n_jobs = len(cmds)
    print(f'submitting {n_jobs} jobs')
    for i in range(n_jobs):
        if args.dry:
            print(cmds[i])
            if i == len(cmds) + args.indx:
                os.system(cmds[i])
        else:
            print(cmds[i])
            write_slurm_script(jobs[i], cmds[i], partition=args.queue, ncpu=args.ncpu)
            os.system(f'sbatch slurm/scripts/run.{jobs[i]}.slrm')
        

if __name__ == '__main__':
    main()
