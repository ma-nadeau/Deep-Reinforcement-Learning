#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --account=winter2025-comp579
#SBATCH --gres=gpu:1  # Request a GPU if needed
#SBATCH --qos=normal
    # -> normal: max 4 hours jobs (no other restrictions 2 jobs at a time)
    # -> comp579-0gpu-4cpu-72h: 3 day max jobs with no GPUs allowed (4 x CPU only)
    # -> comp579-1gpu-12h: 12 hours max jobs with 1 gpu usage

module load miniconda/miniconda-winter2025

pip cache purge
pip install -r requirements.txt  # Install requirements locally

# Run the Python script (ensure it's a script, not a Jupyter notebook)
python ../../Deep-Reinforcement-Learning/Code/code.py