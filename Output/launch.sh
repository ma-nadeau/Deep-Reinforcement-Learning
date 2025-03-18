#!/bin/bash
#SBATCH --job-name=rl_experiment
#SBATCH --propagate=NONE # IMPORTANT for long jobs
#SBATCH --account=winter2025-comp579
#SBATCH --qos=comp579-0gpu-4cpu-72h
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=experiment_output.log
#SBATCH --error=experiment_error.log

# Load necessary modules
module load slurm
PYTHON_EXEC=$(which python3)
# Upgrade pip and install dependencies
$PYTHON_EXEC -m pip install --user --upgrade pip --no-warn-script-location
$PYTHON_EXEC -m pip install --user numpy torch gym[accept-rom-license] gym[atari] matplotlib 
# Navigate to the COMP579 directory
cd ../../Deep-Reinforcement-Learning/Code/
# Run the Python script

$PYTHON_EXEC code.py
