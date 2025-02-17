#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=ees210017
#SBATCH --ntasks=2
#SBATCH --job-name=BC
#SBATCH --partition=defq

python run_post_bias_correction.py
