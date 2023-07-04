# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from src.arguments import parser 

from src.algos.torchbeast import train as train_vanilla 
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity 
from src.algos.rnd import train as train_rnd
from src.algos.ride import train as train_ride
from src.algos.bebold import train as train_bebold
from src.algos.e3b import train as train_e3b
from src.algos.e3b_noveld import train as train_e3b_noveld

# Necessary for multithreading.
import os
import pdb
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"


def main(flags):
    print(flags)
    flags.use_lstm = flags.use_lstm==1


    if flags.num_contexts != -1:
        # if we are limiting number of contexts, sample random seeds used to generate envs
        flags.fix_seed = True
        flags.env_seed = list(np.random.choice(range(100000), flags.num_contexts, replace=False))
        flags.env_seed = [int(i) for i in flags.env_seed]
        print(flags.env_seed)
    else:
        flags.fix_seed = False
    
    
    if flags.model == 'vanilla':
        train_vanilla(flags)
    elif flags.model == 'count':
        train_count(flags)
    elif flags.model == 'curiosity':
        train_curiosity(flags)
    elif flags.model == 'rnd':
        train_rnd(flags)
    elif flags.model == 'ride':
        train_ride(flags)
    elif flags.model == 'bebold':
        train_bebold(flags)
    elif flags.model == 'e3b':
        train_e3b(flags)
    elif flags.model == 'e3b-noveld':
        train_e3b_noveld(flags)
    else:
        raise NotImplementedError("This model has not been implemented.")

if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)
