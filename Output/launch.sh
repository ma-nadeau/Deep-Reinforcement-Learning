#!/bin/bash

#SBATCH -p all                 # Partition (queue)
#SBATCH -c 4                   # Number of cores
#SBATCH --mem=4G               # Memory allocation
#SBATCH --propagate=NONE       # Important for long jobs
#SBATCH -t 0-2:00              # Time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out     # STDOUT file
#SBATCH -e slurm.%N.%j.err     # STDERR file
#SBATCH --qos=comp579-0gpu-4cpu-72h  # QoS for COMP 579 (Modify as needed)
#SBATCH --account=winter2025-comp579 # Account to use for SOCS servers

module load miniconda/miniconda-fall2024  # Load necessary modules

# Add your Python commands or other necessary runs here
# Example:
# source activate my_env  # Activate your conda environment
# python my_script.py     # Run your Python script

# Notes for long-running jobs:
# - Debug on your local machine first.
# - Once everything is working, run on the server.
# - Use the correct QoS based on job needs (e.g., CPU vs. GPU requirements).

# Additional resources:
# - SOCS GPU usage instructions: https://docs.sci.mcgill.ca/COMP/slurm/
# - Contact slurm-admins.science@campus.mcgill.ca for assistance.