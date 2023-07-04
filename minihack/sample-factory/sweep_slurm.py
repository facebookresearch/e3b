# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#!/usr/bin/env python3
import numpy as np
import numpy.random as npr

import math
import time
import os
import sys
import argparse
import pdb
import itertools
import subprocess
from subprocess import Popen, DEVNULL

MAX_JOBS = 120
        
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
            tag = ''
            train_dir = None
            for arg in combos:
                if arg['name'] != 'train_dir':
                    c += f" --{arg['name']}={str(arg['value'])}"
                    if arg['name'] != 'wandb_project':
                        j += f"_{arg['name']}={str(arg['value'])}"
                    k = ''.join([a[0] for a in arg['name'].split('_')])
                    tag += f"{k}={str(arg['value'])}-"
                else:
                    train_dir = arg['value']
                d[arg['name']] = arg['value']
            tag = tag[:-1]
            train_dir += f'/{tag}'
            c += f' --train_dir={train_dir}'
            cmd.append(c)
            job.append(j)
        return cmd, job


# copies the code before the sweep is run 
def make_code_snapshot(savedir):
    print('[copying code snapshot...]')
    snap_dir = f'{savedir}/code/'
    os.system('mkdir -p ' + snap_dir)
    cmd = "find . -name '*.py' -exec cp --parents \{\} " + snap_dir + "/ \;"
    os.system(cmd)
    print('[done]')
        
    
    
# Note: may need to adapt this based on compute infrastructure
def write_slurm_script(name, cmds, partition, device=0, ncpu=1):
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
        for cmd in cmds:
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
    parser.add_argument('--task', default='train-appo')
    parser.add_argument('--queue', default='learnlab')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--indx', type=int, default=-1)
    args = parser.parse_args()

    overrides = Overrides()
            
        
    # train APPO on MiniHack
    if args.task == 'train-appo':
        tag = 'sf_appo'
        savedir = f'./results/{tag}/'
        make_code_snapshot(savedir)
        overrides.add('env', ALL_TASKS)
        overrides.add('experiment', [tag])
        overrides.add('exploration_loss_coeff', [0.1, 0.01, 0.001])
        overrides.add('intrinsic_reward_episodic', ['none'])
        overrides.add('learning_rate', [2.5e-4])
        overrides.add('train_dir', [savedir])
        overrides.add('wandb_project', [tag])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('python -m sf_examples.minihack.train_minihack ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)

        
        
    if args.task == 'train-appo-e3b':
        tag = 'sf_appo_e3b'
        savedir = f'./results/{tag}/'
        make_code_snapshot(savedir)
        overrides.add('env', ALL_TASKS)
        overrides.add('experiment', [tag])
        overrides.add('exploration_loss_coeff', [0.3, 0.1, 0.03, 0.01])
        overrides.add('intrinsic_reward_episodic', ['e3b'])
        overrides.add('intrinsic_reward_coeff', [0.1, 0.03, 0.01, 0.003])
        overrides.add('learning_rate', [2.5e-4])
        overrides.add('e3b_ridge', [0.1])
        overrides.add('train_dir', [savedir])
        overrides.add('wandb_project', [tag])
        overrides.add('seed', [1, 2, 3, 4, 5])
        cmds, jobs = overrides.parse('python -m sf_examples.minihack.train_minihack ', cmd_format='argparse')
        args.ncpu = 40
        print(cmds)
        
        
        
                
        

    # schedule jobs
    os.system('mkdir -p slurm/stdout slurm/stderr')
    n_jobs_running = int(subprocess.check_output('squeue -u mikaelhenaff -h | wc -l', shell=True))
    n_cmds = len(cmds)
    n_jobs = min(n_cmds, MAX_JOBS - n_jobs_running)
    n_cmds_per_job = math.ceil(n_cmds / n_jobs)
    

    
    print(f'submitting {n_cmds} commands in {math.ceil(n_cmds / n_cmds_per_job)} jobs')
    pdb.set_trace()
    for i in range(0, n_cmds, n_cmds_per_job):
        cmd_chunk = cmds[i:i+n_cmds_per_job]
        if args.dry:
            print(cmd_chunk[0])
            if i == len(cmds) + args.indx:
                os.system(cmd_chunk[i])
        else:
            print(cmd_chunk)
            write_slurm_script(jobs[i], cmd_chunk, partition=args.queue, ncpu=args.ncpu)
            os.system(f'sbatch slurm/scripts/run.{jobs[i]}.slrm')
        

if __name__ == '__main__':
    main()
