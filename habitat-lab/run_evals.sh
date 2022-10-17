# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

#!/bin/bash

exp_dir=/private/home/mikaelhenaff/projects/genexp/habitat-challenge/habitat-lab/data/hm3d/ckpt/

for exp in "$exp_dir"/*icm*bc_0.01*; do
    f="$(basename -- $exp)"
    echo $f
    ./run_checkpt.sh $f
done
	   
