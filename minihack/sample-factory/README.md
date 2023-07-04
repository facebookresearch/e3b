# Overview

This repo includes E3B implemented using the APPO algorithm from the [Sample Factory](https://github.com/alex-petrenko/sample-factory) repo. In addition to changing the underlying RL algorithm, we also make a few changes to the network architecture for both the policy network and feature encoder:

- Remove the trunk mapping the full 21x79 image, which was expensive to process
- Expand the crop from 9x9 to 16x16
- Simplify the message encoder
- Add Layer Norm to the output of the feature encoder

The resulting implementation runs at about ~10k FPS on a P100 GPU. This enables experiments to be run in 1-2 hours, instead of about 20 with the previous TorchBeast implementation.

# Running the code

To run the code, follow the installation instructions from the original [Sample Factory](https://github.com/alex-petrenko/sample-factory) repo.

The `sweep_slurm.py` script can be used to submit jobs on Slurm. To run vanilla APPO, run:

```
python sweep_slurm.py --task train-appo
```

To run APPO with the E3B bonus, run:

```
python sweep_slurm.py --task train-appo-e3b
```

Use the `--dry` flag to get individual commands that can be run locally.

# Acknowledgements

Many thanks to [Aleksei Petrenko](https://alex-petrenko.github.io/) for answering questions about Sample Factory on Discord. Check out his amazing repo!