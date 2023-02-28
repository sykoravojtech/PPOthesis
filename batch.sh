#!/bin/bash

#SBATCH --job-name car
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --partition amd
#SBATCH -o slurm/slurm-%j.out # File to which STDOUT will be written
#SBATCH -e slurm/slurm-%j.err # File to which STDERR will be written
#SBATCH --time 1-00:00:00

cd /mnt/personal/sykorvo1/PPOthesis/ppo
# singularity run --bind /mnt/personal/sykorvo1:/mnt/personal/sykorvo1 --nv tensorflow_2.10.0-gpu.sif

# conda activate newest
python train_model.py