# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.arguments import parser 

from src.algos.torchbeast import train as train_vanilla 
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity 
from src.algos.rnd import train as train_rnd
from src.algos.ride import train as train_ride
from src.algos.bebold import train as train_bebold
from src.algos.e3b import train as train_e3b

# Necessary for multithreading.
import os
import pdb
os.environ["OMP_NUM_THREADS"] = "1"


def main(flags):
    print(flags)
    flags.use_lstm = flags.use_lstm==1
    
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
    elif flags.model == 'bebold-test':
        test(flags)
    else:
        raise NotImplementedError("This model has not been implemented. "\
        "The available options are: vanilla, count, curiosity, rnd, ride, \
        no-episodic-counts, and only-episodic-count.")

if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)
