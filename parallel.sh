#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --nodes 2

#SBATCH --partition amd
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --job-name car
#SBATCH --output slurm/PRETgustySides-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out

cd /mnt/personal/sykorvo1/PPOthesis/ppo
srun --ntasks=4 -l --multi-prog ../commands.conf