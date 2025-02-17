#!/usr/bin/env python3

"""
Script Name: post_bias_correction
Description: A script to apply bias-correction and transfer to DAWN server.

Version Information:
    Author   : Guangwei Li
    Date     : 2024-02-29
    Version  : 2.0.1
    Changelog:
        2023-11-01: Initial release.
        2023-11-02: Download the corresponding CFS data.
        2023-11-03: Transfer the enesemble mean.
        2023-11-17: Add attributes to ensemble result.
        2023-11-28: In ensemble calculation, allows some variables not exist.
                    set path_cwrf_raw as an argument instead of hardcoding.
        2023-11-29: Fixed the bug of calculation error in regions outside the US.
        2024-02-29: Major version upgrade, for V0


Dependencies:
    - Required libraries as follows:
    - os,datetime,requests,subprocess,numpy,pandas,xarray,xesmf,xclim.sdba.adjustment, cfgrib,requests

Usage:
    - import post_bias_correction   as     pbc
    - pbc.post_bias_correction(raw_init_date,var_name,user_server)
"""

import os
import datetime
import requests
import subprocess
import numpy        as np
import pandas       as pd
import xarray       as xr
print('library has been loaded!')


# basic information
path_obs_monthly  = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/OBS/regrid_monthly/'
path_cwrf_hindcast= '/scratch16/umd-xliang/shinsa11/Exp2023Dec_NonBC/CWRF-post/'
path_climatology  = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/cwrf_operational/climatology/'
path_operational  = '/scratch16/umd-xliang/aditya/cwrf_operational/CWRF-post/V0/'
path_adjustment   = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/CWRF_v0_adjustment/'
static_path       = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/static/'
MASK_US           = xr.open_dataset(f'{static_path}US_MASK_logic.nc')['MASK']
path_v0_adj       = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/CWRF_v0_adjustment/'
path_climatology  = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/cwrf_operational/climatology/'
years             = range(2012,2023+1)
months            = range(1,12+1)
exps              = ['00','02']
var_names         = ['T2MAX','T2MIN']
vnames            = ['AQ2M','ASNOW','ASNOWH','AGHT_PL','AT2M','ATSK','AU_PL','AV_PL','AXTSS','AXWICE','AXWLIQ','PSFC','RH','uv_10','T2MAX','T2MIN','PRAVG']




def check_mkdir_dawn(user_server,  raw_init_date):
    path_dawn     = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
    dir_path = f'{path_dawn}{raw_init_date}/' 
    remote_check = f'ssh -p 2322 -o StrictHostKeyChecking=no {user_server} "[ -d \"{dir_path}\" ] && echo Exists || echo NotExists"'
    
    result = subprocess.run(remote_check, shell=True, capture_output=True, text=True)
    
    if "NotExists" in result.stdout:
        command = f'ssh -p 2322 -o StrictHostKeyChecking=no {user_server} "mkdir -p \"{dir_path}\" && chmod 777 \"{dir_path}\""'
        exit_status = os.system(command)
        if exit_status == 0:
            print("Command executed successfully!")
        else:
            print(f"Command failed with exit status {exit_status}.")
    else:
        print("Directory already exists!")
        return
def mkdir_adj_folder(path_adj_folder):
    folder_path = path_adj_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder was created at {folder_path}")
    else:
        print(f"The folder already exists at {folder_path}")


def check_mkdir_dawn(user_server,  raw_init_date):
    path_dawn     = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
    dir_path = f'{path_dawn}{raw_init_date}/' 
    remote_check = f'ssh -p 2322 -o StrictHostKeyChecking=no {user_server} "[ -d \"{dir_path}\" ] && echo Exists || echo NotExists"'
    
    result = subprocess.run(remote_check, shell=True, capture_output=True, text=True)
    
    if "NotExists" in result.stdout:
        command = f'ssh -p 2322 -o StrictHostKeyChecking=no {user_server} "mkdir -p \"{dir_path}\" && chmod 777 \"{dir_path}\""'
        exit_status = os.system(command)
        if exit_status == 0:
            print("Command executed successfully!")
        else:
            print(f"Command failed with exit status {exit_status}.")
    else:
        print("Directory already exists!")
        return

    
def adjustment(raw_init_date,var_name,path_cwrf_raw,path_adj_folder):
    year,month,day =int(raw_init_date[:4]), int(raw_init_date[4:6]),int(raw_init_date[6:8])
    for exp in exps:
        # Read the climatology of observation
        obs_cli = xr.open_dataset(f'{path_climatology}OBS/OBS_climatology_{var_name}_{years[0]}-{years[-1]}.nc')
        # Read the climatology of CWRF
        cwrf_cli = xr.open_dataset(f'{path_climatology}CWRF/CWRF_exp{exp}_climatology_{var_name}_{years[0]}-{years[-1]}_{month:02}{day:02}.nc')
        # Read the CWRF data
        cwrf_ds = xr.open_dataset(f'{path_cwrf_raw}{year}{month:02}{day:02}_icbc01_exp{exp}_{var_name}_daily.nc')

        # Create an empty dataset to store the adjusted data
        adjusted_ds = cwrf_ds.copy(deep=True)
        adjusted_ds[var_name].values[:] = np.nan  # Initialize with NaNs

        # Loop through each month in the CWRF dataset to apply adjustment
        for m in np.unique(cwrf_ds['time'].dt.month.values[:-1]):
            # Find the adjustment for the month
            Adjustment = obs_cli[var_name].sel(month=m) - cwrf_cli[var_name].sel(month=m)
            # Apply the adjustment for the specific month
            mask = cwrf_ds['time'].dt.month == m
            adjusted_ds[var_name].loc[{'time': mask}] = cwrf_ds[var_name].loc[{'time': mask}] + Adjustment
        # Drop the 'month' coordinate if it's no longer needed
        adjusted_ds = adjusted_ds.drop_vars('month', errors='ignore')
        adjusted_ds = xr.where(MASK_US, adjusted_ds, cwrf_ds)
        adjusted_ds = adjusted_ds.transpose('time', 'bottom_top', 'south_north', 'west_east')
        adjusted_ds[var_name].loc[{'time':cwrf_ds['time'][-1] }] = cwrf_ds[var_name].loc[{'time': cwrf_ds['time'][-1] }]
        # Preserving global attributes
        adjusted_ds.attrs = cwrf_ds.attrs
        # Preserving variable attributes for all variables including coordinates
        for var in adjusted_ds.variables:
            adjusted_ds[var].attrs = cwrf_ds[var].attrs
        adjusted_ds.to_netcdf(f'{path_adj_folder}{year}{month:02}{day:02}_icbc01_exp{exp}_{var_name}_daily.nc')

def calculate_the_ensemble_mean(vname,path_cwrf_raw, raw_init_date, path_adj):
    # Pattern for file matching
    pattern = f'{raw_init_date}_icbc01_exp*_{vname}_daily.nc'
    # Collect files matching the pattern
    files = []
    for dirpath, dirnames, filenames in os.walk(path_cwrf_raw):
        for filename in filenames:
            if filename.startswith(pattern.split('*')[0]) and filename.endswith(pattern.split('*')[1]):
                files.append(os.path.join(dirpath, filename))
    # Load datasets and store attributes
    datasets = []
    global_attributes = None
    variable_attributes = {}
    ensemble_specific_attributes = {}  # For attributes that vary across files
    for file in files:
        ds = xr.open_dataset(file, chunks={'south_north': 23})
        if global_attributes is None:
            global_attributes = ds.attrs
            for var in ds.variables:
                if var not in ds.coords:
                    variable_attributes[var] = ds[var].attrs
        # Store varying attributes as a list
        for attr in ['CU_PHYSICS', 'RA_LW_PHYSICS', 'RA_SW_PHYSICS', 'BL_PBL_PHYSICS']:  # Add other attributes as needed
            if attr in ds.attrs:
                if attr not in ensemble_specific_attributes:
                    ensemble_specific_attributes[attr] = []
                ensemble_specific_attributes[attr].append(ds.attrs[attr])
        datasets.append(ds)
    # Concatenate datasets into an ensemble
    ensemble_data = xr.concat(datasets, dim='ensemble')
    # Calculate the ensemble mean
    ensemble_mean = ensemble_data.mean(dim='ensemble')
    # Apply global and variable attributes
    ensemble_mean.attrs = global_attributes
    for var in ensemble_mean.variables:
        if var in variable_attributes:
            ensemble_mean[var].attrs = variable_attributes[var]
    # Modify global attributes for ensemble-specific values
    for attr, values in ensemble_specific_attributes.items():
        ensemble_mean.attrs[attr] = ', '.join(map(str, values))
    # Modify specific attributes to reflect the ensemble nature
    filenames_only = [os.path.basename(file) for file in files]
    ensemble_history_str = "The script for ensemble was written by Guangwei Li. Ensemble mean computed from multiple files: " + ', '.join(filenames_only)
    ensemble_mean.attrs['ensemble_history'] = ensemble_history_str
    # Save the ensemble mean dataset with the preserved attributes
    if vname not in ['T2MAX', 'T2MIN']:
        outfile = os.path.join(path_adj, f'{raw_init_date}_icbc01_ensemble_mean_{vname}_daily.nc')
    else:
        if path_cwrf_raw == path_adj:
            outfile = os.path.join(path_adj, f'{raw_init_date}_icbc01_ensemble_mean_{vname}_daily.nc')
        else:
            outfile = os.path.join(path_adj, f'{raw_init_date}_icbc01_ensemble_mean_{vname}_daily_no_adj.nc')            
    ensemble_mean.to_netcdf(outfile)
    print(f'{outfile} has been calculated!')


def post_bias_correction(raw_init_date,user_server,path_cwrf_raw):

    # create a folder to adjusted file and ensemble file. simulation file is also coped
    path_adj_folder = f'{path_v0_adj}{raw_init_date}/'
    mkdir_adj_folder(path_adj_folder)

    # Copy file, so that to transfer to DAWN by one scp command line.
    command = f'cp  {path_cwrf_raw}{raw_init_date}*  {path_adj_folder}'
    subprocess.run(command, shell=True)

    # Bias-correction （Executed after copy to avoid being overwritten by the source file）
    for var_name in var_names:
        adjustment(raw_init_date,var_name,path_cwrf_raw,path_adj_folder)

    # Ensemble mean
    for var_name in vnames:
        try:
            calculate_the_ensemble_mean(var_name,path_cwrf_raw, raw_init_date, path_adj_folder)
        except:
            print(f'{var_name} raw file not exists.')
    for var_name in var_names:
        calculate_the_ensemble_mean(var_name,path_adj_folder, raw_init_date, path_adj_folder)


    # Transfer to DAWN server
    check_mkdir_dawn(user_server,raw_init_date)
    command =f'scp -P 2322 {path_adj_folder}*   {user_server}:/mnt/gfs01/PUB/S2S/V2023-07/Operational/{raw_init_date}/'
    subprocess.run(command, shell=True)
    subprocess.run(f'rm -rf {path_adj_folder}', shell=True)