#!/bin/bash

module add singularity hpcx/hpcx-ompi
singularity exec --nv ../containers/container.sif python bert.py --dataset='base'