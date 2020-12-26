#!/bin/bash

sbatch --gpus=2 --cpus-per-task=2 --time=25-00:00:00 --nodes=1 bert_runner.sh