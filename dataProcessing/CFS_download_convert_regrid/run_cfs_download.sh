#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --account=ees210017
#SBATCH --ntasks=12
#SBATCH --job-name=dwnCFS
#SBATCH --partition=defq

mpiexec -n 12 python scf_download_mpi4.py
