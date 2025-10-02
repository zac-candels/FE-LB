#!/bin/bash
#SBATCH --job-name=BayesOpt   # Job name
#SBATCH --output=output.log # Standard output log file
#SBATCH --error=error.log   # Standard error log file
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --partition=standard      # Partition name
#SBATCH --account=e283
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task

# Run the executable
module load cray-python

srun python TG_lbfem.py