#!/bin/bash
#SBATCH --job-name=Cn0.1
#SBATCH --output=Cn0.1_out.log
#SBATCH --error=Cn0.1_err.log
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --account=ec288-wetting
#SBATCH --qos=lowpriority
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=240

module load singularity

SIF=../../../../../../fenics-gmsh.sif
WORKDIR=$SLURM_SUBMIT_DIR

# Bind the working directory into /root/shared inside the container
singularity exec --bind $SLURM_SUBMIT_DIR:/root/shared $SIF \
    bash -c "cd /root/shared && python3 mainMatin.py"