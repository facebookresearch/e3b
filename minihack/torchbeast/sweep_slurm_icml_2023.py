# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

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
                c += f" --{arg['name']} {str(arg['value'])}"
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
        
        
    
    
# Note: may need to adapt this based on compute infrastructure
def write_slurm_script(name, cmd, partition, device=0, ncpu=1):
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
        s.write("#SBATCH --gres=gpu:1\n")
        s.write(f"srun sh {scriptfile}\n")

    with open(scriptfile, 'w') as s:
        s.write("#!/bin/sh\n")
        s.write("nvidia-smi\n")
        s.write(f"cd {os.getcwd()}\n")
        s.write(f"{cmd}\n")
        s.write("nvidia-smi\n")
        




SKILL_TASKS = [
         'MiniHack-Levitate-Potion-Restricted-v0',
         'MiniHack-Levitate-Boots-Restricted-v0',
         'MiniHack-Freeze-Horn-Restricted-v0', 
         'MiniHack-Freeze-Wand-Restricted-v0', 
         'MiniHack-Freeze-Random-Restricted-v0',
         'MiniHack-LavaCross-Restricted-v0', 
         'MiniHack-WoD-Hard-Restricted-v0'
         ] 

NAV_TASKS = [
         'MiniHack-MultiRoom-N4-Locked-v0',
         'MiniHack-MultiRoom-N6-Lava-v0',
         'MiniHack-MultiRoom-N6-Lava-OpenDoor-v0',
         'MiniHack-MultiRoom-N6-LavaMonsters-v0', 
         'MiniHack-MultiRoom-N10-OpenDoor-v0',  
         'MiniHack-MultiRoom-N10-Lava-OpenDoor-v0', 
         'MiniHack-LavaCrossingS19N13-v0',
         'MiniHack-LavaCrossingS19N17-v0',             
         'MiniHack-Labyrinth-Big-v0',
         ]

ALL_TASKS = SKILL_TASKS + NAV_TASKS

        

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
        

    # MultiRoom with episodic bonus and positional encodings as described in Section 3.1
    elif args.task == 'train-counts-episodic-multiroom':
        SAVEDIR = './results/counts_episodic_multiroom/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['counts-pos'])
        overrides.add('global_bonus_type', ['none'])
        overrides.add('env', ['MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('num_contexts', [1, 3, 5, 10, -1])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

    # MultiRoom with global bonus and positional encodings as described in Section 3.1
    elif args.task == 'train-counts-global-multiroom':
        SAVEDIR = './results/counts_global_multiroom/'
#        SAVEDIR = './results/tmp/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['none'])
        overrides.add('global_bonus_type', ['counts-pos'])
        overrides.add('env', ['MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('num_contexts', [1, 3, 5, 10, -1])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # Corridors with episodic bonus and positional encodings as described in Section 3.2
    elif args.task == 'train-counts-episodic-corridors':
        SAVEDIR = './results/counts_episodic_corridors/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['counts-pos'])
        overrides.add('global_bonus_type', ['none'])
        overrides.add('env', ['MiniHack-Corridor-R5-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('num_contexts', [1])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        

    # Corridors with global bonus and positional encodings as described in Section 3.2
    elif args.task == 'train-counts-global-corridors':
        SAVEDIR = './results/counts_global_corridors/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['none'])
        overrides.add('global_bonus_type', ['counts-pos'])
        overrides.add('env', ['MiniHack-Corridor-R5-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('num_contexts', [1])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # KeyRoom and MultiRoom with episodic bonus and message encodings as described in Section 3.2
    elif args.task == 'train-counts-episodic-msg':
        SAVEDIR = './results/counts_episodic_msg/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['counts-msg'])
        overrides.add('global_bonus_type', ['none'])
        overrides.add('env', ['MiniHack-KeyRoom-S10-v0', 'MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('num_contexts', [-1])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        

    # KeyRoom and MultiRoom with global bonus and message encodings as described in Section 3.2
    elif args.task == 'train-counts-global-msg':
        SAVEDIR = './results/counts_global_msg/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['none'])
        overrides.add('global_bonus_type', ['counts-msg'])
        overrides.add('env', ['MiniHack-KeyRoom-S10-v0', 'MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('num_contexts', [-1])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # Combined bonus with positional encodings (Section 3.4)
    elif args.task == 'train-counts-combined-pos':
        SAVEDIR = './results/counts_combined_pos/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['counts-pos'])
        overrides.add('global_bonus_type', ['counts-pos'])
        overrides.add('env', ['MiniHack-Corridor-R5-v0', 'MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('num_contexts', [1, -1])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 10.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # Combined bonus with message encodings (Section 3.4)
    elif args.task == 'train-counts-combined-msg':
        SAVEDIR = './results/counts_combined_msg/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['count'])
        overrides.add('episodic_bonus_type', ['counts-msg'])
        overrides.add('global_bonus_type', ['counts-msg'])
        overrides.add('env', ['MiniHack-KeyRoom-S10-v0', 'MiniHack-MultiRoom-N6-Lava-v0'])
        overrides.add('savedir', [SAVEDIR])
        overrides.add('num_contexts', [-1])
        overrides.add('intrinsic_reward_coef', [0.1, 1.0, 2.0, 10.0]) # we include 2.0 for MultiRoom, this one is a bit more sensitive
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)


    # Combined bonus for E3Bx{NovelD, RND} (Section 4.2)
    # args.scale_fac=0 corresponds to E3BxRND
    elif args.task == 'train-e3b-noveld':
        SAVEDIR = './results/e3b_noveld/'
        make_code_snapshot(SAVEDIR)
        overrides.add('learning_rate', [0.0001])
        overrides.add('model', ['e3b-noveld'])
        overrides.add('episodic_bonus_type', ['elliptical-icm'])
        overrides.add('env', ALL_TASKS)
        overrides.add('savedir', [SAVEDIR])
        overrides.add('scale_fac', [0.0, 0.1])
        overrides.add('num_contexts', [-1])
        overrides.add('intrinsic_reward_coef', [1.0])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('OMP_NUM_THREADS=1 python main.py ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        
        
        
        
            
        
        

    os.system('mkdir -p slurm/stdout slurm/stderr')
#    cmds = [cmd + f' --device {args.device}' for cmd in cmds]
    n_jobs = len(cmds)
    print(f'submitting {n_jobs} jobs')
    import pdb; pdb.set_trace()
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
