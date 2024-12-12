#!/bin/bash

# Install additional packages in the active conda environment
conda install -n $1 \
    numpy \
    tqdm \
    pyyaml \
    imageio \
    tensorboard \
    anaconda::numba \
    -y
