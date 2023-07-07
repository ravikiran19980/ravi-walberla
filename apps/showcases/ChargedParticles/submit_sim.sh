#!/bin/bash -l
#SBATCH --nodes=58
#SBATCH --ntasks-per-node=20
#SBATCH --time=1:0:0
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load intel intelmpi

srun ./ChargedParticles >&log.migrationvelocity