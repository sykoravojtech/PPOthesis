#!/bin/bash
#SBATCH --ntasks 8
#SBATCH --nodes 4

#SBATCH --partition amd
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --job-name gSide
#SBATCH --output slurm/PRETgustySides4to5-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out

cd /mnt/personal/sykorvo1/PPOthesis/ppo
srun --ntasks=8 -l --multi-prog ../commands.conf