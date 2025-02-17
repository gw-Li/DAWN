import sys,os
import numpy    as np
import pandas   as pd
import xarray   as xr

# # initial date
init_year  = 2024
init_month = 7
init_day   = 5


# **** Base information ***
path_obs_daily = '/mnt/gfs01/PUB/OBS/regrid_daily/'
path_viewer    = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/Data_Viewer/'
path_operational = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_hindcast  = '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
path_static    = '/mnt/gfs01/PUB/OBS/script/static_data/'
da_US_MASK     = xr.open_dataset(f'{path_static}/US_MASK_logic.nc')['MASK']
years_hist     = range(1981,2023+1)
years          = range(2012,2023+1)
months         = range(1,12+1)
vnames         = ['T2MAX','T2MIN','PRAVG']
hour_exps      = ['00_icbc01_exp00','06_icbc01_exp00','06_icbc01_exp02']
days_per_month = { 1: [ 1, 6,11,16,21,26,31], 2: [ 5,10,15,20,25],3: [ 2, 7,12,17,22,27],4: [ 1, 6,11,16,21,26],5: [ 1, 6,11,16,21,26,31],6: [ 5,10,15,20,25,30],7: [ 5,10,15,20,25,30],8: [ 4, 9,14,19,24,29],9: [ 3, 8,13,18,23,28],10:[ 3, 8,13,18,23,28],11:[ 2, 7,12,17,22,27],12:[ 2, 7,12,17,22,27]}


# **** functions ***
def mask_us(da_quantile):
    return xr.where(da_US_MASK,da_quantile,np.nan)

def dawn_calculate_quantile(da_obs_mnth,quantile_value):
    list_quantile = []
    for month in range(1,12+1):
        month_data  = da_obs_mnth.sel(time = da_obs_mnth['time'].dt.month == month).chunk({'time': -1})
        list_quantile.append( mask_us(month_data.quantile(quantile_value, dim="time").compute()).expand_dims(month = [month])   )
    da_quantile = xr.concat(list_quantile,dim='month')
    return da_quantile

def dawn_calculate_max_min(da_obs_mnth):
    list_max = [mask_us(da_obs_mnth.sel(time=da_obs_mnth['time'].dt.month == month).chunk({'time': -1}).max(dim="time").compute()).expand_dims(month=[month])  for month in range(1, 13)    ]
    list_min = [mask_us(da_obs_mnth.sel(time=da_obs_mnth['time'].dt.month == month).chunk({'time': -1}).min(dim="time").compute()).expand_dims(month=[month])  for month in range(1, 13)    ]
    da_max   = xr.concat(list_max, dim='month')
    da_min   = xr.concat(list_min, dim='month')
    return da_max, da_min

def month_of_last_one_day(month):
    return (month + 7 - 1) % 12 + 1

def remove_height_above_ground(ds):
    if 'heightAboveGround' in ds.coords:
        ds = ds.drop_vars('heightAboveGround')
    if 'surface' in ds.coords:
        ds = ds.drop_vars('surface')
    if 'quantile' in ds.coords:
        ds = ds.drop_vars('quantile')
    if 'time' in ds.coords:
        ds = ds.drop_vars('time')
    return ds



def sum_of_60km_searching(count_between):
    shifts = [        (0, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1),        (0, 1), (1, 1), (1, 0), (1, -1)    ]
    list_count_between = [ xr.DataArray(np.roll(np.roll(count_between, shift[0], axis=0), shift[1], axis=1), dims=["south_north", "west_east"])   for shift in shifts   ]
    da_60km_searching_sum = xr.concat(list_count_between, dim='searching').sum(dim='searching')#.astype(np.int16)
    return da_60km_searching_sum



def count_one_realization_fall_tertiles_after_searching(da_cwrf_mnth_sel,ds_quantile):
    ds_count_tertiles = xr.Dataset()
    list_count_less_33_searching,list_count_abov_66_searching,list_count_between_searching = [],[],[]
    for tgt_mnth in  da_cwrf_mnth_sel['time'].dt.month.values:
        da_cwrf_tgt_mnth = da_cwrf_mnth_sel.sel(time = da_cwrf_mnth_sel['time'].dt.month == tgt_mnth).squeeze(['time','bottom_top'], drop = True)
        da_quantile_33_tgt_month = ds_quantile['da_quantile_33'].sel(month =ds_quantile['da_quantile_33']['month'] == tgt_mnth ).squeeze('month', drop = True)
        da_quantile_66_tgt_month = ds_quantile['da_quantile_66'].sel(month =ds_quantile['da_quantile_66']['month'] == tgt_mnth ).squeeze('month', drop = True)
        count_less_33 = xr.where(da_cwrf_tgt_mnth <= da_quantile_33_tgt_month, 1, 0)
        count_abov_66 = xr.where(da_cwrf_tgt_mnth >= da_quantile_66_tgt_month, 1, 0)
        count_between = xr.where((da_cwrf_tgt_mnth < da_quantile_66_tgt_month) & (da_cwrf_tgt_mnth > da_quantile_33_tgt_month), 1, 0)
        list_count_less_33_searching.append(sum_of_60km_searching(count_less_33).expand_dims(month = [tgt_mnth])  )
        list_count_abov_66_searching.append(sum_of_60km_searching(count_abov_66).expand_dims(month = [tgt_mnth])  )
        list_count_between_searching.append(sum_of_60km_searching(count_between).expand_dims(month = [tgt_mnth])  )
    ds_count_tertiles['count_less_33'] = xr.concat(list_count_less_33_searching,dim = 'month' )
    ds_count_tertiles['count_abov_66'] = xr.concat(list_count_abov_66_searching,dim = 'month' )
    ds_count_tertiles['count_between'] = xr.concat(list_count_between_searching,dim = 'month' )
    return ds_count_tertiles


def count_forecast_init_1day_fall_into_the_historical_tertile_and_save(year,month,day,path_operational):
    list_ds_init_1day_count_tertiles_3variables,list_da_ensemble_mean=[],[]
    for vname in vnames:
        ds_quantile = xr.open_dataset(f'{path_viewer}/{vname}_OBS_quantile.nc')
        list_ds_1realization_count_tertiles = []
        list_da = []
        for hour_exp in hour_exps:
            ds_cwrf_daily    = xr.open_dataset(f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}{hour_exp}_{vname}_daily.nc')
            if vname != 'PRAVG':
                ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').mean()
                da_cwrf_mnth_f  = (ds_cwrf_mnth[vname]- 273.15 )  * 9/5   + 32 
                da_cwrf_mnth_f.attrs['units'] = 'Fahrenheit'
            else:
                ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').sum()
                da_cwrf_mnth_f  = ds_cwrf_mnth[vname] * 0.0393701 * 86400 # original unit of PRAVG from CWRF is mm/s
                da_cwrf_mnth_f.attrs['units'] = 'inches per month'
            da_cwrf_mnth_sel = da_cwrf_mnth_f.sel(time=~da_cwrf_mnth_f['time'].dt.month.isin([month,month_of_last_one_day(month)]))
            list_da.append(da_cwrf_mnth_sel.expand_dims(hour_exp = [hour_exp] ))
            list_ds_1realization_count_tertiles.append(count_one_realization_fall_tertiles_after_searching(da_cwrf_mnth_sel,ds_quantile).astype(np.int16).expand_dims(hour_exp = [hour_exp]))
        da_ensemble_mean       = xr.concat(list_da,dim = 'hour_exp').mean(dim = 'hour_exp')
        da_ensemble_mean['time'] = da_ensemble_mean['time'].dt.month
        da_ensemble_mean       = da_ensemble_mean.rename({'time': 'month'})
        ds_init_1day_count_tertiles = xr.concat(list_ds_1realization_count_tertiles,dim = 'hour_exp')
        list_da_ensemble_mean.append(da_ensemble_mean.expand_dims(vname = [vname]))
        list_ds_init_1day_count_tertiles_3variables.append(ds_init_1day_count_tertiles.expand_dims(vname = [vname]))
    list_da_ensemble_mean      = [remove_height_above_ground(ds) for ds in list_da_ensemble_mean]
    da_init_1mnth_ensemble_mean= xr.concat(list_da_ensemble_mean, dim = 'vname')
    ds_init_1day_count_tertiles_3variables = xr.concat(list_ds_init_1day_count_tertiles_3variables, dim = 'vname')
    ds_init_1day_count_tertiles_3variables['ensemble_mean'] = da_init_1mnth_ensemble_mean
    ds_init_1day_count_tertiles_3variables.to_netcdf(f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}_count_tertiles_3variables.nc')


def list_month_day_6_cycle(init_month,init_day):
    list_month_full, list_day_full = [],[]
    for month    in range(1,12+1):
        for day  in days_per_month[month]:
            list_month_full.append(month)
            list_day_full.append(day)
    indices_day  = [i for i in range(len(list_day_full  )) if list_day_full[i]   == init_day]
    indices_mnth = [i for i in range(len(list_month_full)) if list_month_full[i] == init_month]
    common_indices = list(set(indices_day) & set(indices_mnth))[0]
    list_indces  = [common_indices - x for x in range(6)]
    list_indces  = [x + 73 if x < 0 else x  for x in list_indces]
    list_month   = [list_month_full[x] for x in list_indces ]
    list_day     = [list_day_full[x]   for x in list_indces ]
    return list_month, list_day
 
# **** Historical Values ***  # only need to be run once
for vname  in vnames:
    list_file_name = []
    for year in years_hist:
        file_name  = f'{path_obs_daily}OBS_{vname}_{year}-01-01-00_{year}-12-31-18.nc'
        list_file_name.append(file_name)
        ds_obs_daily   = xr.open_mfdataset(list_file_name,combine = 'nested', concat_dim='time')
    if vname != 'PRAVG':
        ds_obs_mnth    = ds_obs_daily.resample(time='ME').mean()
        da_obs_mnth_f  = (ds_obs_mnth[vname]- 273.15 )  * 9/5   + 32 
        da_obs_mnth_f.attrs['units'] = 'Fahrenheit'
    else:
        ds_obs_mnth    = ds_obs_daily.resample(time='ME').sum()
        da_obs_mnth_f  = ds_obs_mnth[vname] * 0.0393701
        da_obs_mnth_f.attrs['units'] = 'inches per month'
    ds_quantile        = xr.Dataset()
    ds_quantile['da_quantile_33'] = xr.apply_ufunc(sum_of_60km_searching,    dawn_calculate_quantile(da_obs_mnth_f,0.333) ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile['da_quantile_66'] = xr.apply_ufunc(sum_of_60km_searching,    dawn_calculate_quantile(da_obs_mnth_f,0.666) ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile['da_quantile_25'] = xr.apply_ufunc(sum_of_60km_searching,    dawn_calculate_quantile(da_obs_mnth_f,0.25 ) ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile['da_quantile_50'] = xr.apply_ufunc(sum_of_60km_searching,    dawn_calculate_quantile(da_obs_mnth_f,0.50 ) ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile['da_quantile_75'] = xr.apply_ufunc(sum_of_60km_searching,    dawn_calculate_quantile(da_obs_mnth_f,0.75 ) ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    da_maximum,da_minimum         = dawn_calculate_max_min(da_obs_mnth_f)
    ds_quantile['da_maximum']     = xr.apply_ufunc(sum_of_60km_searching,    da_maximum ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile['da_minimum']     = xr.apply_ufunc(sum_of_60km_searching,    da_minimum ,      input_core_dims=[['south_north', 'west_east']],    output_core_dims=[['south_north', 'west_east']],    vectorize=True,    dask='parallelized',    output_dtypes=[np.float32])/9.0
    ds_quantile.to_netcdf(f'{path_viewer}/{vname}_OBS_quantile.nc')

