#!/usr/bin/env python3

"""
Script Name: post_bias_correction
Description: A script to apply bias-correction and transfer to DAWN server.

Version Information:
    Author   : Guangwei Li
    Date     : 2023-02-14
    Version  : 1.0.7
    Changelog:
        2023-11-01: Initial release.
        2023-11-02: Download the corresponding CFS data.
        2023-11-03: Transfer the enesemble mean.
        2023-11-17: Add attributes to ensemble result.
        2023-11-28: In ensemble calculation, allows some variables not exist.
                    set path_cwrf_raw as an argument instead of hardcoding.
        2023-11-29: Fixed the bug of calculation error in regions outside the US.
        2024-02-14: Not apply post bias_correction this time.
        2025-02-14: Adjusted the order of if structures （in line 286-293).


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
import xesmf        as xe
import xclim.sdba.adjustment   as     xclim_bc
print('library has been loaded!')


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


def find_nearest_month(raw_init_date):

    # Extract year, month, and day from the given date string
    init_year = int(raw_init_date[:4])
    init_mnth = int(raw_init_date[4:6])
    init_day  = int(raw_init_date[6:8])
    target_date = datetime.date(init_year, init_mnth, init_day)

    # List of months
    months = [11, 12, 1, 2, 3, 4, 5]
    # Find the nearest date

    nearest_month = min(months, key=lambda month: abs(target_date - datetime.date(init_year if month >= init_mnth else init_year + 1, month, 1)))
    return init_year, init_mnth, init_day, nearest_month


def season_trimdate(pred_season_str):
    if   pred_season_str == 'MAM': 
        beg_month, beg_day, end_month, end_day = 3,1,5,31
    elif pred_season_str == 'JJA':
        beg_month, beg_day, end_month, end_day = 6,1,8,31
    return beg_month, beg_day, end_month, end_day


def predyear_season(init_year,init_month):

    if  init_month== 1 or init_month==2:
        pred_season_str    = 'MAM'
        pred_year          = init_year
    elif init_month==10 or init_month==11 or init_month==12:
        pred_season_str    = 'MAM'
        pred_year          = init_year+1
    elif init_month==3 or init_month==4  or init_month==5:
        pred_season_str    = 'JJA'
        pred_year          = init_year

    return pred_year, pred_season_str


def time_beg_time_end(init_year,init_month):

    TIMEFMT     = '%Y-%m-%d-%H'
    pred_year, pred_season_str   =   predyear_season(init_year,init_month)
    beg_month,beg_day, end_month, end_day = season_trimdate(pred_season_str)
    time_beg   = datetime.datetime(pred_year, beg_month,  beg_day,  0).strftime(TIMEFMT)
    time_end   = datetime.datetime(pred_year, end_month,  end_day, 18).strftime(TIMEFMT)

    return time_beg,time_end

def rdmf_obs(path_obs,  var_name, pred_season_str, years):
# Usage : obs_dataset =  rdmf_obs(path_obs,  var_name, pred_season_str, years)
#     Args:
#     - path_obs (str): Path of observation data.
#     - var_name,pred_season_str (str): variable name and prediction seasion
#     - years (list): years to be combined.
#     Returns:
#     - xarray.DataArray: observation dataset with trimed prediction season

    # Generate time variables for prediction seasons.
    pred_months = season_to_pred_months(pred_season_str)
    TIMEFMT     = '%Y-%m-%d-%H'
    mod_name     = 'OBS_'

    # read the OBS data set, and saved as one file 
    obs_files    =[]
    for year in years:
        time_beg = datetime.datetime(year, 1,  1,  0).strftime(TIMEFMT)
        time_end = datetime.datetime(year,12, 31, 18).strftime(TIMEFMT)
        filename = path_obs + mod_name + var_name + '_' + time_beg + '_' + time_end + '.nc'
        obs_files= np.concatenate((obs_files,[filename]))
    
    # Open multi OBS data files
    obs_ds       = xr.open_mfdataset(obs_files,combine='by_coords')
    
    if 'crs' in obs_ds.dims:
        obs_ds   = obs_ds.drop_vars('crs')
    obs_subset = obs_ds.sel(time=obs_ds.time.dt.month.isin(pred_months))
    del obs_ds
    
    return obs_subset

def trim_cwrf(path_raw,  path_cwrf, var_name, init_year, init_month,init_day ,Eexp):

    # Generate the f_name_in and f_name_out following the standard format
    time_beg,time_end = time_beg_time_end(init_year,init_month)
    f_name_in  = f'{init_year}{init_month:02}{init_day:02}_cc00_icbc01_exp{Eexp}_{var_name}_daily.nc'
    # There are  or three naming conversions, later is the second one.
    if not os.path.exists( path_raw + f_name_in):
        f_name_in  = f'{init_year}{init_month:02}{init_day:02}_icbc01_exp{Eexp}_{var_name}_daily.nc'
        if var_name == 'PRAVG'  and (not os.path.exists( path_raw + f_name_in)):
            f_name_in  = f'{init_year}{init_month:02}{init_day:02}_icbc01_exp{Eexp}_PR_daily.nc'

    # If the file raw not exist, skip it
    if not os.path.exists( path_raw + f_name_in):
        print(f'{f_name_in} not exists')
    else:
        # Read the data and trim
        cwrf_ds   = xr.open_dataset(path_raw + f_name_in)
        # Drop useless dimension if necessary
        if 'bottom_top' in cwrf_ds.dims and cwrf_ds.dims['bottom_top'] == 1:
            cwrf_ds = cwrf_ds.squeeze('bottom_top',drop=True)
        # The time dimension of CWRF raw file is off by one day, correct this.
        cwrf_ds['time'] = cwrf_ds['time'] - np.timedelta64(1, 'D')
        # Trim the dataset, only data between time_beg and time_end are maintained.
        cwrf_ds_subset = cwrf_ds.sel(time=slice(time_beg, time_end))
        # Drop other variables, for saving disk space
        cwrf_ds_subset = cwrf_ds_subset[['time', var_name,'lat', 'lon']]
        if var_name  == 'PRAVG':
            if   f_name_in  != f'{init_year}{init_month:02}{init_day:02}_icbc01_exp{Eexp}_PR_daily.nc' : # new file style
                # Old version of c-post, convert units of precipitation.
                cwrf_ds_subset[var_name]  = cwrf_ds_subset[var_name]*86400.0
                cwrf_ds_subset[var_name].attrs['units']   = 'mm/d'
        return cwrf_ds_subset
        
def trim_cwrf_save(path_raw,  path_cwrf, var_name, init_year, init_month,init_day ,Eexp):
#     Trim CWRF post-processed daily data into standard format for bias-correction
#     Args:
#     - path_raw,path_cwrf,(string): Path of Raw and trimed data
#     - var_name (str)             : Variables to be trimed
#     - init_year, init_month (int): Year and month of raw file
#     - Eexp  (str)                : index of physics option selected.
#     Returns:
#     - An .nc file saved as standard format after trim.

    # Generate the f_name_in and f_name_out following the standard format
    time_beg,time_end = time_beg_time_end(init_year,init_month)
    f_name_out = f'CWRF_{var_name}_{time_beg}_{time_end}_E{Eexp}_{init_year}-{init_month:02}-{init_day:02}-00.nc'
    # Call function to read trim CWRF dataset.
    cwrf_ds_subset = trim_cwrf(path_raw,  path_cwrf, var_name, init_year, init_month,init_day ,Eexp)
    
    # Save the file in given path.
    if cwrf_ds_subset is None:
        print("No data returned from trim_cwrf function.")
    else:
        cwrf_ds_subset.to_netcdf( path_cwrf + f_name_out)
        print(f'{f_name_out} is converted.')


def rdmf_cwrf(path_cwrf, var_name, pred_season_str, years, init_month, Eexp):
# Usage : cwrf_dataset =  rdmf_cwrf(path_cwrf, var_name, pred_season_str, years, init_month, Eexp)
#     Args:
#     - path_cwrf ,var_name, pred_season_str(str): Path of observation data, variable name, prediction season string ('MAM' or 'JJA')
#     - years (list): years to be combined.
#     - init_month,Eexp (int, str): initial month and CWRF physics index
#     Returns:
#     - xarray.DataArray: CWRF dataset of given initial month and  physics
    TIMEFMT      = '%Y-%m-%d-%H'
    beg_month, beg_day, end_month, end_day = season_trimdate(pred_season_str)
    
    # Read CWRF data, and combin as one netCDF file.
    mod_name     = 'CWRF_'
    files        = np.array([])
    for year in years:            # different year, must in inner loop for  xr.open_mfdataset
        time_beg = datetime.datetime(year, beg_month,  beg_day,  0).strftime(TIMEFMT)
        time_end = datetime.datetime(year, end_month,  end_day, 18).strftime(TIMEFMT)
        if init_month == 11 or init_month == 12:
            time_IC  = datetime.datetime(year-1, init_month, 1,0).strftime(TIMEFMT)
        else:
            time_IC  = datetime.datetime(year, init_month, 1,0).strftime(TIMEFMT)
        filename = f'{path_cwrf}{mod_name}{var_name}_{time_beg}_{time_end}_E{Eexp}_{time_IC}.nc'
        if os.path.exists(filename):
            files= np.concatenate((files,  [filename] ))
    
    # Open multiple files and combine them into a single Dataset
    cwrf_ds      = xr.open_mfdataset(files, combine='by_coords')
    cwrf_ds      = cwrf_ds.drop(['lat', 'lon'])
    
    return cwrf_ds


def calculate_the_ensemble_mean(vname,path_cwrf_raw, raw_init_date, path_pbc):
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
    ensemble_history_str = "Ensemble mean computed from multiple files: " + ', '.join(filenames_only)
    ensemble_mean.attrs['ensemble_history'] = ensemble_history_str
    # Save the ensemble mean dataset with the preserved attributes
    outfile = os.path.join(path_pbc, f'{raw_init_date}_icbc01_ensemble_mean_{vname}_daily.nc')
    ensemble_mean.to_netcdf(outfile)
    print(f'{outfile} has been calculated!')
    # Clean up
    del ensemble_mean



def kinds(var_name):
    
    if var_name == 'PRAVG' or var_name == 'ASWDNS':
        kind    = '*'
    else:
        kind    = '+'
    
    return kind


def post_bias_correction(raw_init_date,var_name,user_server,path_cwrf_raw):
    
    # Set the path of data, and other informations.
    # path_cwrf_raw = '/scratch16/umd-xliang/aditya/cwrf_operational/CWRF-post/'
    path_pbc      = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/cwrf_operational/post_biascorrection/'
    path_season   = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/cwrf_operational/season/'
    path_combined = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/cwrf_operational/combined/'
    path_static   = '/scratch16/umd-xliang/CFS_seasonal_forecast/DATA/static/'
    path_dawn     = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
    path_dawn_soil= '/mnt/gfs01/PUB/SMAP/post-bias_correction/prediction_nopbc/'

    
    if var_name == 'T2MAX':
        # Transfer all related files to Dawn.
        os.system(f'scp -P 2322 {path_cwrf_raw}{raw_init_date}_icbc01_exp*_*_*.nc  {user_server}:{path_dawn}{raw_init_date}/')
        # Calculate the ensemble mean and save them (for variables that do not participate in bias-correction)
        vnames = ['AGHT_PL','AQ2M','ASNOW','ASNOWH','AT2M','ATSK','AU_PL','AV_PL','AXTSS','AXWICE','AXWLIQ','PSFC','RH','uv_10']
        for vname in vnames:
            try:
                calculate_the_ensemble_mean(vname,path_cwrf_raw, raw_init_date, path_pbc)
            except Exception as e:
                print(f"Error processing {vname}: {e}")


    # Generate time related information.
    init_year, init_month, init_day, nearest_month = find_nearest_month(raw_init_date)
    pred_year, pred_season_str = predyear_season(init_year,init_month)
    time_beg,time_end          = time_beg_time_end(init_year,init_month)

    # Variables for loops (soil moisture and temperature are processed on DAWN server.)
    Eexps         = ['00','02','03','05','06']
    years         = range(2012,pred_year)
    TIMEFMT       = '%Y-%m-%d-%H'

    # apply bias-correction
    # Read the predictand for training.
    obs_comb_name = f'{path_combined}OBS_{var_name}_{years[0]}-{years[-1]}_{pred_season_str}.nc'
    if not os.path.exists(obs_comb_name):
        obs_ds    = rdmf_obs(path_season, var_name, pred_season_str, years)
        obs_ds.to_netcdf(obs_comb_name)
    obs_ds        = xr.open_dataset(obs_comb_name)

    
    if var_name == 'PRAVG':
        # The observed value and missing value of precipitation are both 0. 
        # Therefore, the mask file is used to determine whether it is an ocean.
        us_mask = xr.open_dataset(path_combined + 'US_MASK.nc')['MASK']

    for Eexp in Eexps:

        # Read the predictor for training.
        cwrf_comb_name = f'{path_combined}CWRF_{var_name}_{years[0]}-{years[-1]}_{pred_season_str}_E{Eexp}_init_{nearest_month}.nc'
        if not os.path.exists(cwrf_comb_name):
            cwrf_ds    = rdmf_cwrf(path_season, var_name, pred_season_str, years, nearest_month, Eexp)
            cwrf_ds.to_netcdf(cwrf_comb_name)
        cwrf_ds   = xr.open_dataset(cwrf_comb_name)

        # Read the simulations (trim)
        simu_ds   = trim_cwrf(path_cwrf_raw,  path_season, var_name, init_year, init_month, init_day,Eexp)

        # Read the CWRFraw file.
        simu_file = f'{raw_init_date}_icbc01_exp{Eexp}_{var_name}_daily.nc'
        CWRF_raw  = xr.open_dataset(f'{path_cwrf_raw}{simu_file}')

        # apply post-process Bias Adjust
        # Transfer from dataSet to dataArray 
        ref       = obs_ds[var_name]
        if var_name == 'PRAVG':
            ref   = xr.where(np.isnan(us_mask), np.nan, ref )
        else:
            ref   = xr.where(ref == 0, np.nan, ref)        
        ref.attrs['units']  = obs_ds[var_name].attrs['units']
        hist, sim = cwrf_ds[var_name], simu_ds[var_name]
        
        # Apply bias-adjust
        ADJ       = xclim_bc.Scaling.train(ref, hist, group='time', kind=kinds(var_name))
        adjusted  = ADJ.adjust(sim)

        # Switch the time dimension offset back
        adjusted['time'] = adjusted['time'] + np.timedelta64(1, 'D')
        
        # Convert the unit of precipitation back.
        if var_name == 'PRAVG':
            adjusted  = adjusted/86400.0
            adjusted.attrs['units']   = 'mm s-1'

        # Subset CWRF_raw for times that overlap with adjusted
        overlap_times = adjusted['time'].values
        subset        = CWRF_raw.sel(time=overlap_times)

        # Replace T2MAX values in the subset
        # If CWRF_raw's T2MAX has a 'bottom_top' dimension but adjusted doesn't, we'll add a singleton dimension for matching shapes.
        if 'bottom_top' in CWRF_raw.dims and 'bottom_top' not in adjusted.dims:
            adjusted_with_bottom_top = adjusted.expand_dims({'bottom_top': [0]})
        else:
            adjusted_with_bottom_top = adjusted

        # Transpose to match the order of dimensions in CWRF_raw['T2MAX']
        adjusted_with_bottom_top = adjusted_with_bottom_top.transpose('time', 'bottom_top', 'south_north', 'west_east')

        # replace the adjusted data back to raw file.
        subset[var_name] = xr.where(np.isnan(adjusted_with_bottom_top), subset[var_name], adjusted_with_bottom_top)

        # Reassign this subset back to the original CWRF_raw dataset
        CWRF_raw[var_name].loc[dict(time=overlap_times)] = subset[var_name]
        
        # Save the result
        CWRF_raw.to_netcdf(f'{path_pbc}{simu_file}')
        os.system(f'scp -P 2322 {path_pbc}{simu_file}  {user_server}:{path_dawn}{raw_init_date}/')
        print(f'{var_name}_{Eexp} has been adjusted and transferred to DAWN.')

    # Calculate the ensemble mean and save them (for variables after bias-correction)
    try :
        calculate_the_ensemble_mean(var_name,path_pbc, raw_init_date, path_pbc)
    except Exception as e:
        print(f"Error processing {var_name}: {e}")

    if var_name == 'ASWDNS':
        # Transfer the ensemble mean to Dawn
        os.system(f'scp -P 2322 {path_pbc}{raw_init_date}_icbc01_ensemble_mean_*_daily.nc  {user_server}:{path_dawn}{raw_init_date}/')

        # # Download the corresponding CFS data    
        cmd = f' python  /scratch16/umd-xliang/CFS_seasonal_forecast/DATA/CFS/cfs_download_for_dawn.py {raw_init_date}'
        print(cmd)
        os.system(cmd)
        cfs_paths = ['grib_file','raw_daily','regrid_daily']
        for cfs_path in cfs_paths:
            os.system(f'scp -P 2322 /scratch16/umd-xliang/CFS_seasonal_forecast/DATA/CFS/{cfs_path}/*  {user_server}:/mnt/gfs01/PUB/CFS/{cfs_path}/')
            os.system(f'rm /scratch16/umd-xliang/CFS_seasonal_forecast/DATA/CFS/{cfs_path}/*   ')



