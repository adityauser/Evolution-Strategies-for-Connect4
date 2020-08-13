#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name="ucbSteps"
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=cpu



module load cuda/10.1
python run_parallel5.py 
