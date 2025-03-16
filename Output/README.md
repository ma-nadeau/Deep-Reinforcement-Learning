# SLURM x MIMI

## Running a Job on the SLURM Cluster
You must first go to the `Output` Folder
```bash
cd Deep-Reinforcement-Learning/Output
```

To load the Slurm module and submit a job, use the following commands:
```bash
module load slurm 
sbatch launch.sh
```

## Job Information
To see what's happening with all jobs:
```bash
squeue
```

To see the status of your jobs:
```bash
squeue -u #USER
```
## Output Files
The output of a job will be saved `*.out` file.

To display the output of the GPU:
```bash
cat gpu-grad-01-63.out
```
Replace `gpu-grad-01-63` with the name of the computer.

To cancel a job:
```bash
scancel <job_id>
```
## Clearing *.out Files on MIMI
On mimi, you may end up with many *.out files. To clear them, first go to the `Output` Folder:
```bash
cd Emotion-Classification-using-LLMs/Output
```
Then run the following command:
```bash
python3 clear-output-files.py
```
