#!/bin/bash

#SBATCH --partition=all                     # partition (queue)
#SBATCH --gres=gpu:1                        # number of GPUs (per node)
#SBATCH --cpus-per-task=4                   # number of cores
#SBATCH --mem=50G
#SBATCH --propagate=NONE                    # IMPORTANT for long jobs
#SBATCH --time=0-12:00                      # time (D-HH:MM)
#SBATCH --output=experiment_output.log
#SBATCH --error=experiment_error.log
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --account=winter2025-comp579


# QoS options:
# comp579-0gpu-4cpu-72h     # QoS: 0 GPU, 4 CPU, max 72h
# comp579-1gpu-12h          # QoS: 1 GPU, max 12h
# normal                    # QoS: max 4h (no other restrictions 2 jobs at a time)

# module load miniconda/miniconda-winter2025
module load miniconda/miniconda-fall2024

pip cache purge

# Install required dependencies
pip install -r requirements.txt

# File to Run
python ../../Deep-Reinforcement-Learning/Code/a3.py