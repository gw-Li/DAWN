#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --account=ees210017
#SBATCH --ntasks=24
#SBATCH --job-name=both
#SBATCH --partition=defq

python  both_anomaly_boosting_JJA.py    'T2MAX'  'quantile'          &
python  both_anomaly_boosting_JJA.py    'T2MAX'  'absolute_error'    &
python  both_anomaly_boosting_JJA.py    'T2MAX'  'squared_error'     &
python  both_anomaly_boosting_JJA.py    'T2MAX'  'huber'             &

python  both_anomaly_boosting_JJA.py    'T2MIN'  'quantile'          &
python  both_anomaly_boosting_JJA.py    'T2MIN'  'absolute_error'    &
python  both_anomaly_boosting_JJA.py    'T2MIN'  'squared_error'     &
python  both_anomaly_boosting_JJA.py    'T2MIN'  'huber'             &



python  CFS_anomaly_boosting_JJA.py    'T2MAX'  'quantile'           &
python  CFS_anomaly_boosting_JJA.py    'T2MAX'  'absolute_error'     &
python  CFS_anomaly_boosting_JJA.py    'T2MAX'  'squared_error'      &
python  CFS_anomaly_boosting_JJA.py    'T2MAX'  'huber'              &

python  CFS_anomaly_boosting_JJA.py    'T2MIN'  'quantile'           &
python  CFS_anomaly_boosting_JJA.py    'T2MIN'  'absolute_error'     &
python  CFS_anomaly_boosting_JJA.py    'T2MIN'  'squared_error'      &
python  CFS_anomaly_boosting_JJA.py    'T2MIN'  'huber'              &



python  CWRF_anomaly_boosting_JJA.py    'T2MAX'  'quantile'          &
python  CWRF_anomaly_boosting_JJA.py    'T2MAX'  'absolute_error'    &
python  CWRF_anomaly_boosting_JJA.py    'T2MAX'  'squared_error'     &
python  CWRF_anomaly_boosting_JJA.py    'T2MAX'  'huber'             &

python  CWRF_anomaly_boosting_JJA.py    'T2MIN'  'quantile'          &
python  CWRF_anomaly_boosting_JJA.py    'T2MIN'  'absolute_error'    &
python  CWRF_anomaly_boosting_JJA.py    'T2MIN'  'squared_error'     &
python  CWRF_anomaly_boosting_JJA.py    'T2MIN'  'huber'             &


wait
echo "All Python scripts have finished."