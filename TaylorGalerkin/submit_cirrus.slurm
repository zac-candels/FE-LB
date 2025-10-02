#!/bin/bash
#SBATCH --job-name=TG-fem
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=38:00:00
#SBATCH --partition=standard
#SBATCH --account=sc139-wetting
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

module load singularity

SIF=../../fenics.sif
WORKDIR=$SLURM_SUBMIT_DIR

# Bind the working directory into /root/shared inside the container
singularity exec --bind $SLURM_SUBMIT_DIR:/root/shared $SIF \
    bash -c "cd /root/shared && python3 TG_lbfem.py"