#!/bin/bash

#SBATCH --job-name car
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --partition amdgpu
#SBATCH --gres gpu:1
#SBATCH -o slurm-%j.out # File to which STDOUT will be written
#SBATCH -e slurm-%j.err # File to which STDERR will be written

cd /mnt/personal/sykorvo1/PPOthesis/ppo
singularity run --bind /mnt/personal/sykorvo1:/mnt/personal/sykorvo1 --nv tensorflow_2.10.0-gpu.sif

# python run_model.py
python train_model.py