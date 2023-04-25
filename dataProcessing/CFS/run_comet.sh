#!/bin/bash
#SBATCH --job-name="dldCFS"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL
#SBATCH -t 10:00:00
python downloadCFS_JJA.py