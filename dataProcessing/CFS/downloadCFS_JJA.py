import subprocess
import datetime 
# for JJA
data_dir = '/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/CFS/'
CFSnclscript = '/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/src/preparation/DownloadCFS.ncl'
TIMEFMT  = '%Y-%m-%d-%H'
vnames   = ["PRAVG",'T2MAX',"T2MIN"]
years    = range(2013,2022)
months   = range(3,5)
# Loop through the variable and year arguments
for vname in vnames:
    for year in years:
        time_beg = datetime.datetime(year, 6,  1,  0).strftime(TIMEFMT)
        time_end = datetime.datetime(year, 8, 31, 18).strftime(TIMEFMT)
        for stmonth in months:
            time_IC  = datetime.datetime(year, stmonth,  1,  0).strftime(TIMEFMT)
            command  = f' ncl  \'varname=\"{vname}\"\'   \'time_IC=\"{time_IC}\"\'    \'time_beg=\"{time_beg}\"\'    \'time_end=\"{time_end}\"\'  \'cfs_path=\"{data_dir}\"\'  {CFSnclscript}  >& log.txt '
            subprocess.run(command, shell=True, check=True)




#CFS for MAM
import subprocess
import datetime 
data_dir = '/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/CFS/'
CFSnclscript = '/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/src/preparation/DownloadCFS.ncl'
TIMEFMT  = '%Y-%m-%d-%H'
vnames   = ["PRAVG",'T2MAX',"T2MIN"]
years    = range(2013,2022)
months   = range(1,2)
# Loop through the variable and year arguments
for vname in vnames:
    for year in years:
        time_beg = datetime.datetime(year, 3,  1,  0).strftime(TIMEFMT)
        time_end = datetime.datetime(year, 5, 31, 18).strftime(TIMEFMT)
        for stmonth in months:
            time_IC  = datetime.datetime(year, stmonth,  1,  0).strftime(TIMEFMT)
            command  = f'ncl  \'varname=\"{vname}\"\'   \'time_IC=\"{time_IC}\"\'    \'time_beg=\"{time_beg}\"\'    \'time_end=\"{time_end}\"\'  \'cfs_path=\"{data_dir}\"\'  {CFSnclscript}  >& log.txt'
            subprocess.run(command, shell=True, check=True)
