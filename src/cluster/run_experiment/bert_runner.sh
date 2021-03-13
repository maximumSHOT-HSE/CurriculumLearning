#!/bin/bash

#SBATCH --job-name=Fbase
#SBATCH --output=Fbase.log
#SBATCH --time=25-00:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module add singularity hpcx/hpcx-ompi
singularity exec --nv /home/aomelchenko/containers/container.sif python bert.py --dataset base --curriculum F
