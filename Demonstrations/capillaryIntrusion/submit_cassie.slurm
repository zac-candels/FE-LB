#!/bin/bash
#SBATCH --job-name=capIntrusion
#SBATCH --output=out.log
#SBATCH --error=error.log
#SBATCH --time=96:00:00

#SBATCH --ntasks=1     # MPI ranks per node (adjust)
#SBATCH --cpus-per-task=1

#SBATCH --partition=login


# Load Cray Python (required for srun/MPI environment)
module load cray-python
module load FEniCS/2019.1
module load openmpi/4.1.1-gcc12.3

# Launch MPI ranks with srun
python capIntrusion.py

