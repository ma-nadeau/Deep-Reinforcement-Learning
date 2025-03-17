#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=winter2025-comp579 

module load cuda/cuda-12.6 
module load python/3.10

pip cache purge
pip install -r requirements.txt

python ../Code/code.ipynb
