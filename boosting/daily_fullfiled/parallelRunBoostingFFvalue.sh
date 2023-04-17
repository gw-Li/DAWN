#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=ees210017
#SBATCH --ntasks=24
#SBATCH --job-name=both
#SBATCH --partition=defq

python  both_boosting_JJA.py    'T2MAX'  'quantile'          &
python  both_boosting_JJA.py    'T2MAX'  'absolute_error'    &
python  both_boosting_JJA.py    'T2MAX'  'squared_error'     &
python  both_boosting_JJA.py    'T2MAX'  'huber'             &

python  both_boosting_JJA.py    'T2MIN'  'quantile'          &
python  both_boosting_JJA.py    'T2MIN'  'absolute_error'    &
python  both_boosting_JJA.py    'T2MIN'  'squared_error'     &
python  both_boosting_JJA.py    'T2MIN'  'huber'             &


python  CFS_boosting_JJA.py    'T2MAX'  'quantile'           &
python  CFS_boosting_JJA.py    'T2MAX'  'absolute_error'     &
python  CFS_boosting_JJA.py    'T2MAX'  'squared_error'      &
python  CFS_boosting_JJA.py    'T2MAX'  'huber'              &

python  CFS_boosting_JJA.py    'T2MIN'  'quantile'           &
python  CFS_boosting_JJA.py    'T2MIN'  'absolute_error'     &
python  CFS_boosting_JJA.py    'T2MIN'  'squared_error'      &
python  CFS_boosting_JJA.py    'T2MIN'  'huber'              &



python  CFS_boosting_JJA.py    'T2MAX'  'quantile'           &
python  CFS_boosting_JJA.py    'T2MAX'  'absolute_error'     &
python  CFS_boosting_JJA.py    'T2MAX'  'squared_error'      &
python  CFS_boosting_JJA.py    'T2MAX'  'huber'              &

python  CFS_boosting_JJA.py    'T2MIN'  'quantile'           &
python  CFS_boosting_JJA.py    'T2MIN'  'absolute_error'     &
python  CFS_boosting_JJA.py    'T2MIN'  'squared_error'      &
python  CFS_boosting_JJA.py    'T2MIN'  'huber'              &

wait
echo "All Python scripts have finished."
