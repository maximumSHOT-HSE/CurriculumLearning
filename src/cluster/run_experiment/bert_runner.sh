#!/bin/bash

module add singularity hpcx/hpcx-ompi
singularity exec --nv /home/aomelchenko/containers/container.sif python bert.py --dataset base
