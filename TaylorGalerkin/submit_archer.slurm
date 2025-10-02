#!/bin/bash
#SBATCH --job-name=TG-fem   # Job name
#SBATCH --output=output.log # Standard output log file
#SBATCH --error=error.log   # Standard error log file
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --partition=standard      # Partition name
#SBATCH --account=e283
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task

SIF=../../fenics.sif
WORKDIR=$SLURM_SUBMIT_DIR

module load cray-python

# Bind the working directory into /root/shared inside the container
singularity exec --bind $SLURM_SUBMIT_DIR:/root/shared $SIF \
    bash -c "cd /root/shared && python3 TG_lbfem.py"