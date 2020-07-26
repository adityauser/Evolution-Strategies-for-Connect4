#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name="ds"
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2


source activate env
module load cuda/10.1
python main.py 
