#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --nodes 1

#SBATCH --partition amd
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --job-name car
#SBATCH --output slurm/gustyRight-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out

cd /mnt/personal/sykorvo1/PPOthesis/ppo
srun --ntasks=2 -l --multi-prog ../parallel.conf