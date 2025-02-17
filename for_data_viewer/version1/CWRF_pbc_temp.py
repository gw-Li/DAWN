import os
import datetime
import requests
import subprocess
import numpy        as np
import pandas       as pd
import xarray       as xr
print('library has been loaded!')

# basic information
path_cwrf_hindcast= '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
path_operational  = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_climatology  = '/mnt/gfs01/PUB/S2S/V2023-07/gwli/transfer/climatology/'
static_path       = '/mnt/gfs01/PUB/OBS/script/static_data/'
MASK_US           = xr.open_dataset(f'{static_path}US_MASK_logic.nc')['MASK']
list_exp_hours    = [['00',6],['02',6],['00',0]]
var_names         = ['T2MAX','T2MIN']
years             = range(2012,2023+1)
months            = range(1,12+1)
days_per_month    = {     1: [ 1, 6,11,16,21,26,31],    2: [ 5,10,15,20,25],    3: [ 2, 7,12,17,22,27],    4: [ 1, 6,11,16,21,26],    5: [ 1, 6,11,16,21,26,31],    6: [ 5,10,15,20,25,30],    7: [ 5,10,15,20,25,30],    8: [ 4, 9,14,19,24,29],    9: [ 3, 8,13,18,23,28],    10:[ 3, 8,13,18,23,28],    11:[ 2, 7,12,17,22,27],    12:[ 2, 7,12,17,22,27],}
raw_init_dates    = []
for year in years:
    for month in months:
        days      = days_per_month[month]
        for day in days:
            raw_init_dates.append(f'{year}{month:02}{day:02}')
print(len(raw_init_dates))

def adjustment(raw_init_date,var_name):
    year,month,day      = int(raw_init_date[:4]), int(raw_init_date[4:6]),int(raw_init_date[6:8])
    for exp_hour in list_exp_hours:
        exp, hour       = exp_hour[0],exp_hour[1]
        try:
            # Read the climatology of observation
            obs_cli     = xr.open_dataset(f'{path_climatology}OBS/OBS_climatology_{var_name}_{years[0]}-{years[-1]}.nc')
            # Read the climatology of CWRF
            if hour     == 6: 
                cwrf_cli = xr.open_dataset(f'{path_climatology}CWRF/CWRF_exp{exp}_climatology_{var_name}_{years[0]}-{years[-1]}_{month:02}{day:02}.nc')
            elif hour   == 0:
                cwrf_cli= xr.open_dataset(f'{path_climatology}CWRF/CWRF_00Z_exp{exp}_climatology_{var_name}_{years[0]}-{years[-1]}_{month:02}{day:02}.nc')
            # Read the CWRF data
            cwrf_ds     = xr.open_dataset(f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily_no_adj.nc')

            # Create an empty dataset to store the adjusted data
            adjusted_ds = cwrf_ds.copy(deep=True)
            adjusted_ds[var_name].values[:] = np.nan  # Initialize with NaNs

            # Loop through each month in the CWRF dataset to apply adjustment
            for m in np.unique(cwrf_ds['time'].dt.month.values[:-1]):
                # Find the adjustment for the month
                Adjustment = obs_cli[var_name].sel(month=m) - cwrf_cli[var_name].sel(month=m)
                # Apply the adjustment for the specific month
                mask    = cwrf_ds['time'].dt.month == m
                adjusted_ds[var_name].loc[{'time': mask}] = cwrf_ds[var_name].loc[{'time': mask}] + Adjustment
            # Drop the 'month' coordinate if it's no longer needed
            adjusted_ds = adjusted_ds.drop_vars('month', errors='ignore')
            adjusted_ds = xr.where(MASK_US, adjusted_ds, cwrf_ds)
            adjusted_ds = adjusted_ds.transpose('time', 'bottom_top', 'south_north', 'west_east')
            adjusted_ds[var_name].loc[{'time':cwrf_ds['time'][-1] }] = cwrf_ds[var_name].loc[{'time': cwrf_ds['time'][-1] }]
            # Preserving global attributes
            adjusted_ds.attrs          = cwrf_ds.attrs
            # Preserving variable attributes for all variables including coordinates
            for var in adjusted_ds.variables:
                adjusted_ds[var].attrs = cwrf_ds[var].attrs
            adjusted_ds.to_netcdf(f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily.nc')
        except Exception as exception:
            print(exception)


for raw_init_date   in raw_init_dates:
    print(raw_init_date)
    for var_name in var_names:
        adjustment(raw_init_date,var_name)

