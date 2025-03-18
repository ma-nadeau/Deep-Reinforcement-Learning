#!/bin/bash
#
#SBATCH -p all # partition (queue)
#SBATCH -c 4 # number of cores
#SBATCH --mem=4G
#SBATCH --propagate=NONE # IMPORTANT for long jobs
#SBATCH -t 0-2:00 # time (D-HH:MM)
#SBATCH --output=experiment_output.log
#SBATCH --error=experiment_error.log
#SBATCH --qos=comp579-0gpu-4cpu-72h
#SBATCH --account=winter2025-comp579

module load miniconda/miniconda-winter2025

pip cache purge
pip install -r requirements.txt

python ../../Deep-Reinforcement-Learning/Code/code.py