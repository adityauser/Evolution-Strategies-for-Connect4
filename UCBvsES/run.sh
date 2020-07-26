#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=100
#SBATCH --job-name=""
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=cpu



module load cuda/10.1
python run_parallel.py 
