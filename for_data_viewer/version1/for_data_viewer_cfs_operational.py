import sys , os
import glob, calendar
import numpy    as np
import pandas   as pd
import xarray   as xr
from datetime import date
from datetime import date, timedelta,datetime


import argparse
parser     = argparse.ArgumentParser(description="Process a single date parameter.")
parser.add_argument('date', type=str, help='Date in the format YYYYMMDD')
date_str   = parser.parse_args().date
dw_year    = int(date_str[:4])
dw_month   = int(date_str[4:6])
dw_day     = int(date_str[6:])



# **** Base information ***
path_obs_daily   = '/mnt/gfs01/PUB/OBS/regrid_daily/'
path_viewer      = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/Data_Viewer/'
path_operational = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_cfs_mnth    = '/mnt/gfs01/PUB/CFS/regrid_full_monthly/'
path_cfs_D_V     = '/mnt/gfs01/PUB/CFS/regrid_full_monthly/for_data_viewer/'
path_static      = '/mnt/gfs01/PUB/OBS/script/static_data/'
path_ens_stgy    = '/mnt/gfs01/PUB/CFS/regrid_full_monthly/for_data_viewer/ensemble_strategy/'
da_US_MASK       = xr.open_dataset(f'{path_static}/US_MASK_logic.nc')['MASK']
years_hist       = range(1981,2023+1)
vnames           = ['T2MAX','T2MIN','PRAVG']
hour_exps        = ['00_icbc01_exp00','06_icbc01_exp00','06_icbc01_exp02']
hours            = [0,6,12,18]
days_per_month   = { 1: [ 1, 6,11,16,21,26,31], 2: [ 5,10,15,20,25],3: [ 2, 7,12,17,22,27],4: [ 1, 6,11,16,21,26],5: [ 1, 6,11,16,21,26,31],6: [ 5,10,15,20,25,30],7: [ 5,10,15,20,25,30],8: [ 4, 9,14,19,24,29],9: [ 3, 8,13,18,23,28],10:[ 3, 8,13,18,23,28],11:[ 2, 7,12,17,22,27],12:[ 2, 7,12,17,22,27]}
month_to_delete  = { 1:[11,12, 1],    2:[12, 1, 2],    3:[ 1, 2, 3],    4:[ 2, 3, 4],    5:[ 3, 4, 5],    6:[ 4, 5, 6],    7:[ 5, 6, 7],    8:[ 6, 7, 8],    9:[ 7, 8, 9],    10:[8, 9,10],    11:[9,10,11],    12:[10,11,12]}
possible_combinations   = [[year,month,day] for year in range(2010, 2100) for month in range(1,12+1) for day in days_per_month[month]   ]

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

def month_of_last_one_day(month):
    return (month + 7 - 1) % 12 + 1

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

def remove_height_above_ground(ds):
    for coord in ['heightAboveGround', 'surface', 'quantile', 'time']:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds

def remove_height_above_ground2(ds):
    for coord in ['heightAboveGround', 'surface', 'quantile']:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds

def count_one_cfs_fall_tertiles_after_searching(da_cwrf_mnth_sel,ds_quantile):
    ds_count_tertiles = xr.Dataset()
    list_count_less_33_searching,list_count_abov_66_searching,list_count_between_searching = [],[],[]
    for tgt_mnth in  da_cwrf_mnth_sel['time'].dt.month.values:
        da_cwrf_tgt_mnth = da_cwrf_mnth_sel.sel(time = da_cwrf_mnth_sel['time'].dt.month == tgt_mnth).squeeze('time', drop = True)
        da_quantile_33_tgt_month = ds_quantile['da_quantile_33'].sel(month =ds_quantile['da_quantile_33']['month'] == tgt_mnth ).squeeze('month', drop = True)
        da_quantile_66_tgt_month = ds_quantile['da_quantile_66'].sel(month =ds_quantile['da_quantile_66']['month'] == tgt_mnth ).squeeze('month', drop = True)
        count_less_33 = xr.where(da_cwrf_tgt_mnth <= da_quantile_33_tgt_month, 1, 0)
        count_abov_66 = xr.where(da_cwrf_tgt_mnth >= da_quantile_66_tgt_month, 1, 0)
        count_between = xr.where((da_cwrf_tgt_mnth < da_quantile_66_tgt_month) & (da_cwrf_tgt_mnth > da_quantile_33_tgt_month), 1, 0)
        list_count_less_33_searching.append(sum_of_60km_searching(count_less_33).expand_dims(month = [tgt_mnth])  )
        list_count_abov_66_searching.append(sum_of_60km_searching(count_abov_66).expand_dims(month = [tgt_mnth])  )
        list_count_between_searching.append(sum_of_60km_searching(count_between).expand_dims(month = [tgt_mnth])  )
    ds_count_tertiles['count_less_33'] = xr.concat(list_count_less_33_searching,dim = 'month' ).astype('int16')
    ds_count_tertiles['count_abov_66'] = xr.concat(list_count_abov_66_searching,dim = 'month' ).astype('int16')
    ds_count_tertiles['count_between'] = xr.concat(list_count_between_searching,dim = 'month' ).astype('int16')
    return ds_count_tertiles

def count_forecast_init_1day_fall_into_the_historical_tertile_and_save(year,month,day,path_operational):
    list_ds_init_1day_count_tertiles_3variables=[]
    for vname in vnames:
        ds_quantile = xr.open_dataset(f'{path_viewer}/{vname}_OBS_quantile.nc')
        list_ds_1realization_count_tertiles = []
        for hour_exp in hour_exps:
            ds_cwrf_daily    = xr.open_dataset(f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}{hour_exp}_{vname}_daily.nc') 
            if vname != 'PRAVG':
                ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').mean()
                da_cwrf_mnth_f  = (ds_cwrf_mnth[vname]- 273.15 )  * 9/5   + 32 
                da_cwrf_mnth_f.attrs['units'] = 'Fahrenheit'
            else:
                ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').sum()
                da_cwrf_mnth_f  = ds_cwrf_mnth[vname] * 0.0393701
                da_cwrf_mnth_f.attrs['units'] = 'inches per month'
            da_cwrf_mnth_sel = da_cwrf_mnth_f.sel(time=~da_cwrf_mnth_f['time'].dt.month.isin([month,month_of_last_one_day(month)]))
            list_ds_1realization_count_tertiles.append(count_one_realization_fall_tertiles_after_searching(da_cwrf_mnth_sel,ds_quantile).expand_dims(hour_exp = [hour_exp]))
        ds_init_1day_count_tertiles = xr.concat(list_ds_1realization_count_tertiles,dim = 'hour_exp')
        list_ds_init_1day_count_tertiles_3variables.append(ds_init_1day_count_tertiles.expand_dims(vname = [vname]))
    ds_init_1day_count_tertiles_3variables = xr.concat(list_ds_init_1day_count_tertiles_3variables, dim = 'vname')
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

# Intermediate data for data-viewer ensemble from realizations initial at the same month.
def CFSv2_calculate_intermediate_data_ensemble_from_one_month(year, month):
    list_ds_init_1mnth_count_tertiles_3variables,list_da_ensemble_mean =[], []
    for vname in vnames:
        print(vname)
        ds_quantile      = xr.open_dataset(f'{path_viewer}/{vname}_OBS_quantile.nc')
        list_ds_1realization_count_tertiles = []
        file_list        = glob.glob(f'{path_cfs_mnth}{vname}/CFS_full_time_{vname}_{year}-{month:02}-*.nc')
        print('number of ensemble',len(file_list))
        list_da          = []
        for file_name in file_list:
            ds_cfs_daily = xr.open_dataset(file_name)
            if vname    != 'PRAVG':
                da_cfs_mnth_f  = (ds_cfs_daily[vname]- 273.15 )  * 9/5   + 32 
                da_cfs_mnth_f.attrs['units'] = 'Fahrenheit'
            else:
                da_cfs_mnth_f  = ds_cfs_daily[vname] * calendar.monthrange(year, month)[1] * 0.0393701
                da_cfs_mnth_f.attrs['units'] = 'inches per month'
            da_cfs_mnth_sel    = da_cfs_mnth_f.sel(time=~da_cfs_mnth_f['time'].dt.month.isin(month_to_delete[month])).compute()
            list_da.append(da_cfs_mnth_sel.expand_dims(day_hour = [ file_name[-8:-3]]))    # for ensemble mean
            list_ds_1realization_count_tertiles.append(count_one_cfs_fall_tertiles_after_searching(da_cfs_mnth_sel,ds_quantile).expand_dims(day_hour = [ file_name[-8:-3]]))
        da_ensemble_mean       = xr.concat(list_da,dim = 'day_hour').mean(dim = 'day_hour')
        da_ensemble_mean['time'] = da_ensemble_mean['time'].dt.month
        da_ensemble_mean       = da_ensemble_mean.rename({'time': 'month'})
        ds_init_1month_count_tertiles = xr.concat(list_ds_1realization_count_tertiles,dim = 'day_hour').sum(dim = 'day_hour')
        list_da_ensemble_mean.append(da_ensemble_mean.expand_dims(vname = [vname]))
        list_ds_init_1mnth_count_tertiles_3variables.append(ds_init_1month_count_tertiles.expand_dims(vname = [vname]))
    list_da_ensemble_mean      = [remove_height_above_ground(ds) for ds in list_da_ensemble_mean]
    list_ds_init_1mnth_count_tertiles_3variables = [remove_height_above_ground(ds) for ds in list_ds_init_1mnth_count_tertiles_3variables]
    da_init_1mnth_ensemble_mean= xr.concat(list_da_ensemble_mean, dim = 'vname')
    ds_init_1mnth_count_tertiles_3variables = xr.concat(list_ds_init_1mnth_count_tertiles_3variables, dim = 'vname')
    # sum across 'day_hour' dimension, and save as .astype(np.int16)
    ds_init_1mnth_count_tertiles_3variables['ensemble_mean'] = da_init_1mnth_ensemble_mean
    ds_init_1mnth_count_tertiles_3variables['count_less_33'] = ds_init_1mnth_count_tertiles_3variables['count_less_33'].astype(np.int16)
    ds_init_1mnth_count_tertiles_3variables['count_abov_66'] = ds_init_1mnth_count_tertiles_3variables['count_abov_66'].astype(np.int16)
    ds_init_1mnth_count_tertiles_3variables['count_between'] = ds_init_1mnth_count_tertiles_3variables['count_between'].astype(np.int16)
    ds_init_1mnth_count_tertiles_3variables.to_netcdf(f'{path_cfs_D_V}1month_interval_CFS_intermediate/CFS_{year}-{month:02}_intermediate_for_data_viewer.nc')
    print(f'{path_cfs_D_V}1month_interval_CFS_intermediate/CFS_{year}-{month:02}_intermediate_for_data_viewer.nc  is DONE!')

def CFSv2_calculate_intermediate_data_ensemble_from_5days(init_year, init_month,init_day):
    index_of_init = possible_combinations.index([init_year, init_month,init_day ])
    si_year,si_month,si_day = possible_combinations[index_of_init-1]
    init_date     = date(init_year, init_month, init_day)
    si_date       = date(si_year, si_month, si_day)
    # Generate a list of dates between si_date and init_date
    date_list     = []
    current_date  = si_date
    while current_date <= init_date:
        date_list.append([current_date.year, current_date.month, current_date.day])
        current_date += timedelta(days=1)
    # Calculate and Save
    list_ds       =[]
    for vname in vnames:
        ds_quantile      = xr.open_dataset(f'{path_viewer}/{vname}_OBS_quantile.nc')
        hours     = [0,6,12,18]
        list_files= []
        for year, month, day in date_list[1::]:
            for hour in hours:
                fname = f'{path_cfs_mnth}{vname}/CFS_full_time_{vname}_{year}-{month:02}-{day:02}-{hour:02}.nc'
                if os.path.exists(fname):
                    list_files.append(fname)
        ds_cfs            = xr.open_mfdataset(list_files,combine='nested',concat_dim = 'ensemble')
        subset_ds_cfs     = ds_cfs.sel(time=~ds_cfs['time.month'].isin(month_to_delete[init_month]))
        if vname != 'PRAVG':
            da_cfs_mnth_f = (subset_ds_cfs[vname]- 273.15 )  * 9/5   + 32 
            da_cfs_mnth_f.attrs['units'] = 'Fahrenheit'
        else:
            days_in_month = subset_ds_cfs['time'].dt.days_in_month
            da_cfs_sum    = subset_ds_cfs[vname] * days_in_month
            da_cfs_mnth_f = da_cfs_sum * 0.0393701     # original unit of PRAVG from CWRF is mm/s
            da_cfs_mnth_f.attrs['units'] = 'inches per month'
        da_cfs_mnth_f_em  = da_cfs_mnth_f.mean('ensemble')
        list_ds_quantile  = [count_one_cfs_fall_tertiles_after_searching(da_cfs_mnth_f.sel(ensemble =x ),ds_quantile) for x in da_cfs_mnth_f['ensemble'].values]
        ds_quantile_5DI   = xr.concat(list_ds_quantile,dim='ensemble').sum(dim = 'ensemble')
        da_ensemble_mean  = da_cfs_mnth_f.mean('ensemble').astype('float32')
        da_ensemble_mean['time'] = da_ensemble_mean['time'].dt.month
        ds_quantile_5DI['ensemble_mean']       = da_ensemble_mean.rename({'time': 'month'})
        ds_quantile_5DI['num_of_files']  = len(list_files)
        list_ds.append(ds_quantile_5DI.expand_dims(vname = [vname]))
    list_ds      = [remove_height_above_ground2(ds) for ds in list_ds]
    ds_5day_interval_CFS_intermediate_3variables = xr.concat(list_ds, dim = 'vname')
    ds_5day_interval_CFS_intermediate_3variables['num_of_files']   = ds_5day_interval_CFS_intermediate_3variables['num_of_files'].astype('int16')
    ds_5day_interval_CFS_intermediate_3variables['count_less_33']  = ds_5day_interval_CFS_intermediate_3variables['count_less_33'].astype('int16')
    ds_5day_interval_CFS_intermediate_3variables['count_abov_66']  = ds_5day_interval_CFS_intermediate_3variables['count_abov_66'].astype('int16')
    ds_5day_interval_CFS_intermediate_3variables['count_between']  = ds_5day_interval_CFS_intermediate_3variables['count_between'].astype('int16')
    ds_5day_interval_CFS_intermediate_3variables.to_netcdf(f'{path_cfs_D_V}5day_interval_CFS_intermediate/CFS_5day_{init_year}-{init_month:02}-{init_day:02}_intermediate_for_data_viewer.nc')




# Calculate the input data for Data-viewer
# lead month between  using ensemble strategy result, but for  lead month between, applying  all the data we have.
import json
path_cfs_D_V     = '/mnt/gfs01/PUB/CFS/regrid_full_monthly/for_data_viewer/'
path_ens_stgy    = '/mnt/gfs01/PUB/CFS/regrid_full_monthly/for_data_viewer/ensemble_strategy/'
dck_ens_stgy_0mq = json.load( open(f'{path_ens_stgy}initial_month_0mq.json' ) )  # Load
dck_ens_stgy_1mq = json.load( open(f'{path_ens_stgy}initial_month_1mq.json' ) )  # Load
dck_ens_stgy_2mq = json.load( open(f'{path_ens_stgy}initial_month_2mq.json' ) )  # Load
dck_ens_stgy_3mq = json.load( open(f'{path_ens_stgy}initial_month_3mq.json' ) )  # Load
list_dck_ens_stgy= [dck_ens_stgy_1mq,dck_ens_stgy_2mq,dck_ens_stgy_3mq]
dck_full_list    = [dck_ens_stgy_0mq, dck_ens_stgy_1mq, dck_ens_stgy_2mq, dck_ens_stgy_3mq]


def month_next_year(month):
    return (month - 1) % 12 + 1


def intermediate_across_ensemble(ds_tgt):
    ds_out = xr.Dataset()
    ds_out['count_less_33'] = ds_tgt['count_less_33'].sum(dim='ensemble')
    ds_out['count_abov_66'] = ds_tgt['count_abov_66'].sum(dim='ensemble')
    ds_out['count_between'] = ds_tgt['count_between'].sum(dim='ensemble')
    ds_out['ensemble_mean'] = ds_tgt['ensemble_mean'].mean(dim='ensemble')
    return ds_out


def intermediate_across_ensemble_weight(ds_5day,weights):
    ds_out = xr.Dataset()
    ds_out['count_less_33'] = ds_5day['count_less_33'].sum(dim='ensemble')
    ds_out['count_abov_66'] = ds_5day['count_abov_66'].sum(dim='ensemble')
    ds_out['count_between'] = ds_5day['count_between'].sum(dim='ensemble')
    ds_out['ensemble_mean'] = ds_5day['ensemble_mean'].weighted(weights).mean(dim='ensemble')
    ds_out = ds_out.transpose('vname', 'month', 'south_north', 'west_east')
    return ds_out

def year_month_list_for_checking(dw_year, dw_month):
    year_month_list = []
    start_date = datetime(dw_year, dw_month, 1) - timedelta(days=9*30)
    for i in range(8, 0, -1):
        date = start_date + timedelta(days=i*30)
        year_month_list.append((date.year, date.month))
    return year_month_list


def calculate_CFS_predictions_for_data_viewer_and_save(dw_year,dw_month,dw_day):
    dates_before_dw_day = [[dw_year, dw_month, d] for d in days_per_month[dw_month] if d <= dw_day]
    list_tgt_mnth       = [month_next_year(dw_month + x)   for x in    range(1,10)]   # List of target month
    # Check the intermediate dataset. if not exists, calculate it.
    for dates in dates_before_dw_day:
        _, _, right_border_day = dates
        if not os.path.exists(f'{path_cfs_D_V}5day_interval_CFS_intermediate/CFS_5day_{dw_year}-{dw_month:02}-{right_border_day:02}_intermediate_for_data_viewer.nc'):
            try:
                CFSv2_calculate_intermediate_data_ensemble_from_5days(dw_year, dw_month, right_border_day)
            except Exception as e:
                print(e)
    for init_year, init_month in year_month_list_for_checking(dw_year, dw_month):
        if not os.path.exists( f'{path_cfs_D_V}1month_interval_CFS_intermediate/CFS_{init_year}-{init_month:02}_intermediate_for_data_viewer.nc'):
            try:
                CFSv2_calculate_intermediate_data_ensemble_from_one_month(dw_year,dw_month,dw_day)
            except Exception as e:
                print(e)                
    # combine the 1 month intermediate dataset
    list_ds_tgt         = []
    for kk in range(3):          # Apply the conclusion of ensemble strategy study
        tgt_month       = list_tgt_mnth[kk]
        list_init_month , fname_list = list_dck_ens_stgy[kk][f'{tgt_month}'], []
        for init_month in list_init_month:
            init_year   = dw_year - 1 if init_month > tgt_month else dw_year
            fname_list.append(f'{path_cfs_D_V}1month_interval_CFS_intermediate/CFS_{init_year}-{init_month:02}_intermediate_for_data_viewer.nc')
        ds_tgt_ensemble = xr.open_mfdataset(fname_list,combine='nested',concat_dim='ensemble').sel(month = tgt_month)
        list_ds_tgt.append(  intermediate_across_ensemble(ds_tgt_ensemble)  )
    for kk in range(5):
        tgt_month       = list_tgt_mnth[3+kk]
        list_init_month =  [month_next_year(dw_month - x)   for x in    range(1,6-kk)]
        for init_month in list_init_month:
            init_year   = dw_year - 1 if init_month > tgt_month else dw_year
            fname_list.append(f'{path_cfs_D_V}1month_interval_CFS_intermediate/CFS_{init_year}-{init_month:02}_intermediate_for_data_viewer.nc')
        ds_tgt_ensemble = xr.open_mfdataset(fname_list,combine='nested',concat_dim='ensemble').sel(month = tgt_month)
        list_ds_tgt.append(  intermediate_across_ensemble(ds_tgt_ensemble)  )
    ds_from_1month      = xr.concat( list_ds_tgt, dim='month').transpose('vname', 'month', 'south_north', 'west_east')
    # # combine the 5 days interval intermediate dataset
    list_weight,list_ds_5day   =[],[]
    for i, dates in enumerate(dates_before_dw_day):
        _, _, right_border_day = dates
        fname = f'{path_cfs_D_V}5day_interval_CFS_intermediate/CFS_5day_{dw_year}-{dw_month:02}-{right_border_day:02}_intermediate_for_data_viewer.nc'
        if os.path.exists(fname):
            dataset    = xr.open_dataset(fname)
            list_ds_5day.append(dataset )
            list_weight.append(dataset['num_of_files'])
    ds_5day  = xr.concat(  list_ds_5day, dim='ensemble')
    weights  = xr.DataArray(list_weight, dims=['ensemble', 'vname'])
    weights.loc[0] = 4 * dates_before_dw_day[0][2]
    ds_from_5day = intermediate_across_ensemble_weight(ds_5day,weights)
    # Combine the 1month intermediate and 5-day intermediate dataset togater, and run
    # Find which target month need 5day intermediate data
    keys_with_value = []
    for dck in dck_full_list:
        keys_with_value.extend([key for key, value in dck.items() if dw_month in value])
    unique_keys = list(set(map(int, keys_with_value)))    
    # The weight of 1month ensemble and 5days ensemble based on ensemble member.
    n5day = ds_from_5day['count_less_33'][:,:,55,55] + ds_from_5day['count_abov_66'][:,:,55,55] + ds_from_5day['count_between'][:,:,55,55]
    n1m   = ds_from_1month['count_less_33'][:,:,55,55]+ds_from_1month['count_abov_66'][:,:,55,55]+ds_from_1month['count_between'][:,:,55,55]
    # Adjust the weight file
    for tgt_month  in list_tgt_mnth[:8]:
        if not tgt_month in unique_keys:
            n5day.loc[:, tgt_month] = 0
    # Calculate the data need for data-viewer month by month
    list_ds_data_view_tgt      = []
    for tgt_month  in list_tgt_mnth[:8]:
        ds_data_view = xr.Dataset()
        ds_5d_tgt_m = ds_from_5day.sel(month =tgt_month )
        ds_1m_tgt_m = ds_from_1month.sel(month =tgt_month )
        if not n5day.sel(month = tgt_month)[0].values.item() == 0: #(the intermediate data from the 1month ensemble and the 5day ensemble are counted)
            da_dv_tgt_33 = ds_5d_tgt_m['count_less_33'] + ds_1m_tgt_m['count_less_33']
            da_dv_tgt_66 = ds_5d_tgt_m['count_abov_66'] + ds_1m_tgt_m['count_abov_66']
            da_dv_tgt_bt = ds_5d_tgt_m['count_between'] + ds_1m_tgt_m['count_between']
            weighted_sum = ds_1m_tgt_m['ensemble_mean'] * n1m.sel(month = tgt_month) + ds_5d_tgt_m['ensemble_mean'] * n5day.sel(month = tgt_month)
            total_weight = n1m.sel(month   = tgt_month) + n5day.sel(month = tgt_month)
            ds_data_view['ensemble_mean']  = (weighted_sum / total_weight).astype(np.float32)
        else:
            da_dv_tgt_33 = ds_1m_tgt_m['count_less_33']
            da_dv_tgt_66 = ds_1m_tgt_m['count_abov_66']
            da_dv_tgt_bt = ds_1m_tgt_m['count_between']
            ds_data_view['ensemble_mean']  = ds_1m_tgt_m['ensemble_mean']
        da_sum           = da_dv_tgt_33    + da_dv_tgt_66   + da_dv_tgt_bt
        ds_data_view['percentage_less_33'] = ((da_dv_tgt_33 / da_sum) * 100.0 ).astype(np.float32)
        ds_data_view['percentage_abov_66'] = ((da_dv_tgt_66 / da_sum) * 100.0 ).astype(np.float32)
        ds_data_view['percentage_between'] = ((da_dv_tgt_bt / da_sum) * 100.0 ).astype(np.float32)
        ds_data_view.compute()
        list_ds_data_view_tgt.append(ds_data_view)
    # for last target month, only ds_from_5day have predictions.
    tgt_month    = list_tgt_mnth[8]
    ds_data_view = xr.Dataset()
    ds_5d_tgt_m  = ds_from_5day.sel(month  = tgt_month)
    da_dv_tgt_33 = ds_5d_tgt_m['count_less_33']
    da_dv_tgt_66 = ds_5d_tgt_m['count_abov_66']
    da_dv_tgt_bt = ds_5d_tgt_m['count_between']
    ds_data_view['ensemble_mean']      = ds_5d_tgt_m['ensemble_mean'].astype(np.float32)
    da_sum       = da_dv_tgt_33        + da_dv_tgt_66   + da_dv_tgt_bt
    ds_data_view['percentage_less_33'] = ((da_dv_tgt_33 / da_sum) * 100.0 ).astype(np.float32)
    ds_data_view['percentage_abov_66'] = ((da_dv_tgt_66 / da_sum) * 100.0 ).astype(np.float32)
    ds_data_view['percentage_between'] = ((da_dv_tgt_bt / da_sum) * 100.0 ).astype(np.float32)
    ds_data_view.compute()
    list_ds_data_view_tgt.append(ds_data_view)
    # Save the data
    ds_data_view_noaa = mask_us(xr.concat(list_ds_data_view_tgt,dim='month')).transpose('vname', 'month', 'south_north', 'west_east')
    ds_data_view_noaa.to_netcdf( f'{path_cfs_D_V}CFS_for_data_viewer/CFS_{dw_year}{dw_month:02}{dw_day:02}_for_data_viewer.nc')

calculate_CFS_predictions_for_data_viewer_and_save(dw_year,dw_month,dw_day)