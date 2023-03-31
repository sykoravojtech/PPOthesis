#!/bin/bash

#SBATCH --job-name table
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --partition amd
#SBATCH -o slurm/table-gustySides-%j.out # File to which STDOUT will be written

cd /mnt/personal/sykorvo1/PPOthesis/ppo
# singularity run --bind /mnt/personal/sykorvo1:/mnt/personal/sykorvo1 --nv tensorflow_2.10.0-gpu.sif

# conda activate newest
python run_model.py --load_model BEST/gustySides/ep780_4to5