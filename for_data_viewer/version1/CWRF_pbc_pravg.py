import os
import shutil
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
path_recover      = '/mnt/gfs01/guangwei/data/BC/data_recover/'
static_path       = '/mnt/gfs01/PUB/OBS/script/static_data/'
MASK_US           = xr.open_dataset(f'{static_path}US_MASK_logic.nc')['MASK']
list_exp_hours    = [['00',6],['02',6],['00',0]]
var_names         = ['T2MAX','T2MIN']
years             = range(2013,2013+1)
months            = range(1,12+1)
none_bc_months    = [1,2,6,7,8,9,10,11,12]
bc_months         = [3,4,5]
var_name          = 'PRAVG'
days_per_month    = {     1: [ 1, 6,11,16,21,26,31],    2: [ 5,10,15,20,25],    3: [ 2, 7,12,17,22,27],    4: [ 1, 6,11,16,21,26],    5: [ 1, 6,11,16,26,31],    6: [ 5,10,15,20,25,30],    7: [ 5,10,15,20,25,30],    8: [ 4, 9,14,19,24,29],    9: [ 3, 8,13,18,23,28],    10:[ 3, 8,13,18,23,28],    11:[ 2, 7,12,17,22,27],    12:[ 2, 7,12,17,22,27],}


def copy_none_bc_files(raw_init_date,var_name):
    year,month,day      = int(raw_init_date[:4]), int(raw_init_date[4:6]),int(raw_init_date[6:8])
    for exp_hour in list_exp_hours:
        exp, hour       = exp_hour[0],exp_hour[1]
        try:
            src   = f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily_no_adj.nc'
            dst   = f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily.nc'
            shutil.copy(src, dst)
        except Exception as exception:
            print(exception)


# raw_init_dates = [f'{year}{month:02}{day:02}' for year in years for month in none_bc_months for day in days_per_month[month]]
# for raw_init_date   in raw_init_dates:
#     print(raw_init_date)
#     copy_none_bc_files(raw_init_date,var_name)



raw_init_dates    = [f'{year}{month:02}{day:02}' for year in years for month in bc_months for day in days_per_month[month]]
for raw_init_date in raw_init_dates:
    # raw_init_date     =  raw_init_dates[0]
    year,month,day    = int(raw_init_date[:4]), int(raw_init_date[4:6]),int(raw_init_date[6:8])
    for exp_hour  in list_exp_hours:
        try:
            exp, hour         = exp_hour[0],exp_hour[1]
            cwrf_ds           = xr.open_dataset(f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily_no_adj.nc')
            recover_ds        = xr.open_dataset(f'{path_recover}recover_1day_CWRF_PRAVG_EQM_2012-2023_mnth-{month:02}_day-{day:02}_JJA_{hour:02}_icbc01_exp{exp}.nc')
            recover_ds_1y     = recover_ds.sel(time=recover_ds['time'].dt.year == year)
            common_times      = np.intersect1d(cwrf_ds['time'].values, recover_ds_1y['time'].values)
            # cwrf_ds_common    = cwrf_ds.sel(time=common_times)
            recover_ds_common = recover_ds_1y.sel(time=common_times)
            cwrf_ds['PRAVG'].loc[dict(time=common_times)] = recover_ds_common['PRAVG']
            cwrf_ds.to_netcdf(f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily.nc')
            print(f'{path_cwrf_hindcast}{raw_init_date}/{year}{month:02}{day:02}{hour:02}_icbc01_exp{exp}_{var_name}_daily.nc')
        except Exception as e:
            print(e)
