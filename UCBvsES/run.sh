#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name="ucbes_smp"
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=cpu



module load cuda/10.1
python run_parallel1.py 
