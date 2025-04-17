import os
import xarray                  as     xr
import numpy                   as     np
import pandas                  as     pd
from   properscoring           import crps_ensemble
from   sklearn.model_selection import KFold
from   scipy.stats             import genextreme
from   dask.distributed        import Client
from   icecream                import ic
from   mpi4py                  import MPI
comm              = MPI.COMM_WORLD
rank,size         = comm.Get_rank(), comm.Get_size()

## *************** Path and base information ***************
# --- File Paht ---
path_obs          = '/mnt/gfs01/PUB/OBS/regrid_daily/'
path_cwrf         = '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
path_noaa         = '/mnt/gfs01/PUB/CFS/regrid_full_daily/PRAVG/'
path_static       = '/data/gwli/data/static_data/'
path_project      = '/data/gwli/data/BC/'
path_simu         = f'{path_project}data_simu/'
path_bc           = f'{path_project}data_bc/'
path_ind_simu     = f'{path_project}indices_simu/'
path_ind_bc       = f'{path_project}indices_bc/'
path_simu_metric  = f'{path_project}metric_simu/'
path_bc_metric    = f'{path_project}metric_bc/'
path_stttc_simu   = f'{path_project}stttc_simu/'
path_stttc_bc     = f'{path_project}stttc_bc/'
path_rx1day       = f'{path_project}CNN_Rx1day/'
path_rx5day       = f'{path_project}CNN_Rx5day/'
path_quantile     = f'{path_project}quantile/'
path_recover      = f'{path_project}data_recover/'
path_figures      = f'{path_project}Figures_EQM/'
path_moment       = f'{path_project}moment/'
path_combine      = path_simu

# --- Loop Indices ---
var_name          = 'PRAVG'
pred_season_str   = 'JJA'
years             = range(2012,2023+1)
init_mnths        = [3,4,5]
# init_mnths        = [5]
ih_exps           = ['00_icbc01_exp00','06_icbc01_exp00']
bc_methods        = ['EQM','DQM','QDM']
cli_indices       = ['Rx1day','Rx5day']
init_hours        = [0,6]
days_per_month    = {3: [ 2, 7,12,17,22,27],4: [ 1, 6,11,16,21,26],5: [ 1, 6,11,16,21,26,31]}
recover_methods   = ['recover_1day_','recover_5day_','recover_5to1day_']
# --- Combinations for parallel run ---
noaa_combination  = [[init_mnth,day,init_hour] for init_mnth in init_mnths  for day in days_per_month[init_mnth] for init_hour in init_hours  ]
cwrf_combination  = [[init_mnth,day,ih_exp]    for init_mnth in init_mnths  for day in days_per_month[init_mnth] for ih_exp    in ih_exps ]
bc_combination    = [[bc_method,init_mnth,day,ih_exp] for bc_method in bc_methods for init_mnth in init_mnths for day in days_per_month[init_mnth] for ih_exp in ih_exps]
print(len(cwrf_combination))
## *************** Functions ***************
# --- Mask ---
da_US    = xr.open_dataset(f'{path_static}US_MASK_logic.nc')['MASK']
da_CB    = xr.open_dataset(f'{path_static}CB_MASK_logic.nc')['MASK']
def US_MASK(data_array):
    return xr.where(da_US,data_array,np.nan)
def CB_MASK(data_array):
    return xr.where(da_CB,data_array,np.nan)

# # Define the GEV fitting function
def fit_gev_and_get_return_level(data, return_period):
    data = data.data if isinstance(data, xr.core.variable.Variable) else data
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return np.nan
    shape, loc, scale = genextreme.fit(data)
    return genextreme.ppf(1 - 1.0 / return_period, shape, loc=loc, scale=scale)

def compute_indices(total_size, rank, size):
    n, remainder = divmod(total_size, size)
    start= rank  * n + min(rank, remainder)
    end  = start + n + (1 if rank < remainder else 0)
    return start, end

# --- For apply Bias correction ---
da_max   = xr.open_dataset(f'{path_simu}OBS_100year_return_level_12y_2.5max_estimation.nc')['PRAVG'] # based on precipitation observation from 2012 to 2023
def MAX_MASK(data_array):
    return xr.where(data_array<da_max,data_array,da_max).transpose('time', 'south_north', 'west_east')

def bias_correction_export(var_name,list_adjusted,new_dataset,valid_times,export_file_name):
    # The 'adjusted_all' will have the same shape and dims as X
    adjusted_all  = xr.concat(list_adjusted, dim='time').transpose('time', 'south_north', 'west_east')
    # Sort the adjusted data by time
    adjusted_all  = adjusted_all.sortby('time').transpose()
    # transpose the dataset, change the order of the dimensions.
    adjusted_all  = adjusted_all.transpose('time', 'south_north', 'west_east')
    # new dataset will have the same dimension as cwrf_da
    new_dataset.loc[dict(time=valid_times)] = adjusted_all#.values
    # Save the result
    # Convert DataArray to Dataset with new variable name
    ds            = new_dataset.to_dataset(name=var_name)
    ds_mask       = US_MASK(ds).transpose('time', 'south_north', 'west_east')
    # Convert scalar coordinates to attributes
    MAX_MASK(ds_mask).to_netcdf(export_file_name)
    print(f'{export_file_name} has been saved!')

def bias_correction_read_data_cwrf(path_combine,var_name,years,init_mnth,day,pred_season_str,ih_exp):
    obs_da        = xr.open_dataset(f'{path_combine}OBS_PRAVG_2012-2023_JJA.nc')[var_name]
    cwrf_da       = xr.open_dataset(f'{path_combine}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')[var_name]
    valid_times   = cwrf_da.dropna('time').time
    # obs_da_subset,cwrf_da_subset =  xr.align(obs_da,cwrf_da, join = 'inner')
    obs_da_subset = obs_da.sel( time = valid_times)
    cwrf_da_subset= cwrf_da.sel(time = valid_times)
    obs_da_subset.attrs['units']     = 'mm/d'
    cwrf_da_subset.attrs['units']    = 'mm/d'
    new_dataset   = xr.full_like(cwrf_da, fill_value=np.nan)
    uniq_years    = np.unique(cwrf_da_subset.time.dt.year.values)
    num_years     = len(uniq_years)
    return obs_da_subset,cwrf_da_subset,new_dataset,valid_times,num_years

def bias_correction_read_data_noaa(path_combine,var_name,years,init_mnth,day,pred_season_str,init_hour):
    obs_da        = xr.open_dataset(f'{path_combine}OBS_PRAVG_2012-2023_JJA.nc')[var_name]
    noaa_da       = xr.open_dataset(f'{path_combine}NOAA_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_hour-{init_hour:02}_{pred_season_str}.nc')[var_name]
    valid_times   = noaa_da.dropna('time').time
    # obs_da_subset,noaa_da_subset =  xr.align(obs_da,noaa_da, join = 'inner')
    obs_da_subset = obs_da.sel( time = valid_times)
    noaa_da_subset= noaa_da.sel(time = valid_times)
    obs_da_subset.attrs['units']     = 'mm/d'
    noaa_da_subset.attrs['units']    = 'mm/d'
    new_dataset   = xr.full_like(noaa_da, fill_value=np.nan)
    uniq_years    = np.unique(noaa_da_subset.time.dt.year.values)
    num_years     = len(uniq_years)
    return obs_da_subset,noaa_da_subset,new_dataset,valid_times,num_years

def xclim_parameters(bc_method, var_name, pred_season_str):
    group_rank = 2 if bc_method == 'DQM' and var_name == 'PRAVG' and pred_season_str == 'MAM' else 1 if bc_method == 'LOCI' else 0
    kind = '*' if var_name == 'PRAVG' else '+'
    return group_rank, kind, 20, '1mm/day'


# # --- For extreme indices calculation ---
def calculate_R95p(pr_data, threshold=1.0):
    wet_days      = pr_data.where(pr_data >= threshold)                                    # Identify wet days
    pr_95th_percentile = wet_days.quantile(0.95, dim='time', skipna=True)                  # Calculate the 95th percentile for wet days
    R95p          = (pr_data > pr_95th_percentile).groupby('time.year').sum(dim='time')    # Count the number of days exceeding the 95th percentile for each year
    pr_99th_percentile = wet_days.quantile(0.99, dim='time', skipna=True)                  # Calculate the 99th percentile for wet days
    R99p          = (pr_data > pr_99th_percentile).groupby('time.year').sum(dim='time')    # Count the number of days exceeding the 99th percentile for each year
    PRCPTOT       = wet_days.groupby('time.year').sum(dim='time')                          # Calculate PRCPTOT: Total annual precipitation on wet days
    R95p,R99p     = R95p.drop_vars('quantile') ,R99p.drop_vars('quantile')                 # drop coordinates
    return R95p,R99p,PRCPTOT

def make_continuous_daily(obs_da): 
    # Convert an xarray DataArray to a continuous daily dataset, filling missing dates with NaN.
    # Create a full time range from the start to the end of the dataset
    start_date    = obs_da['time'].min().values
    end_date      = obs_da['time'].max().values
    full_time_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # Reindex the dataset to this full time range, filling gaps with NaN
    obs_da_full   = obs_da.reindex(time=full_time_range, fill_value=np.nan)
    return obs_da_full

# Define a helper function to compute the maximum consecutive True values in a 1D array.
def max_consecutive(bool_arr):
    max_run = 0
    run = 0
    for val in bool_arr:
        if val:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run

def precipitation_indices_jja(data_in):
    dry_thresh = 1.0
    da_summer  = data_in.where(data_in.time.dt.month.isin([6, 7, 8]), drop=True)

    rx1day_JJA = da_summer.groupby('time.year').max(dim='time')
    rx5day_JJA = da_summer.rolling(time=5, min_periods=5).sum().groupby('time.year').max(dim='time')

    years      = []
    cdd_list   = []
    cwd_list   = []
    sdii_list  = []
    wetdays_list  = []
    r10mm_list    = []
    r20mm_list    = []
    for year, group in da_summer.groupby('time.year'):
        years.append(year)
        dry    = (group <  dry_thresh)
        wet    = (group >= dry_thresh)
        cdd_vals  = np.apply_along_axis(max_consecutive, 0, dry.values)
        cwd_vals  = np.apply_along_axis(max_consecutive, 0, wet.values)
        cdd_list.append(cdd_vals)
        cwd_list.append(cwd_vals)
        sum_wet   = group.where(wet).sum(dim='time')
        count_wet = wet.sum(dim='time')
        sdii      = sum_wet.where(count_wet > 0) / count_wet.where(count_wet > 0)
        sdii_list.append(sdii.values)
        wetdays_count = wet.sum(dim='time')
        wetdays_list.append(wetdays_count.values)
        r10mm_count   = (group >= 10.0).sum(dim='time')
        r10mm_list.append(r10mm_count.values)
        r20mm_count   = (group >= 20.0).sum(dim='time')
        r20mm_list.append(r20mm_count.values)

    cdd_arr   = np.stack(cdd_list, axis=0)
    cwd_arr   = np.stack(cwd_list, axis=0)
    sdii_arr  = np.stack(sdii_list, axis=0)
    wetdays_arr   = np.stack(wetdays_list, axis=0)
    r10mm_arr = np.stack(r10mm_list, axis=0)
    r20mm_arr = np.stack(r20mm_list, axis=0)

    cdd_JJA   = xr.DataArray(cdd_arr,       dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})
    cwd_JJA   = xr.DataArray(cwd_arr,       dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})
    SDII_JJA  = xr.DataArray(sdii_arr,      dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})
    wetdays_JJA = xr.DataArray(wetdays_arr, dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})
    R10mm_JJA   = xr.DataArray(r10mm_arr,   dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})
    R20mm_JJA   = xr.DataArray(r20mm_arr,   dims=['year', 'south_north', 'west_east'],coords={'year': years, 'south_north': da_summer.south_north,'west_east': da_summer.west_east})

    R95p,R99p,PRCPTOT = calculate_R95p(da_summer)
    JJA_total_prec    = da_summer.groupby('time.year').sum(dim='time')

    ds_combined       = xr.Dataset({
        'Rx1day'      : rx1day_JJA.astype(np.float32),
        'Rx5day'      : rx5day_JJA.astype(np.float32),
        'SDII'        : SDII_JJA.astype(np.float32),
        'R10mm'       : R10mm_JJA.astype(np.float32),
        'R20mm'       : R20mm_JJA.astype(np.float32),
        'CDD'         : cdd_JJA.astype(np.float32),
        'CWD'         : cwd_JJA.astype(np.float32),
        'R95p'        : R95p.astype(np.float32),
        'R99p'        : R99p.astype(np.float32),
        'PRCPTOT'     : PRCPTOT,
        'JJATOTAL'    : JJA_total_prec,
        'wetdays'     : wetdays_JJA.astype(np.float32),
    })
    return ds_combined

# --- Static of Extreme indices ---
def ym(da):                              #annual average for ACC calculation.
    return(da.resample(time='YE').mean())

def dawn_rmse(da1, da2, dim): # dim='time'
    diff_square               = ((da1 - da2) ** 2)
    mean_diff_square          = diff_square.mean(dim, skipna=True)
    return np.sqrt(mean_diff_square)

def gb_mnth(daily_data):
    daily_data['year_month']  = daily_data['time'].dt.strftime('%Y-%m')
    monthly_data              = daily_data.groupby('year_month').mean()
    return monthly_data

def to_mnth(daily_data,var_name):
    monthly_data  = gb_mnth(daily_data)  # group by month
    dims          = list(monthly_data[var_name].dims)
    dims[dims.index('year_month')] = 'time'
    new_time      = pd.to_datetime(monthly_data['year_month'].values)
    new_ds        = xr.Dataset()
    new_ds[var_name]  = (dims, monthly_data[var_name].values)
    new_ds['time']= new_time
    new_ds[var_name].attrs['units'] = monthly_data[var_name].attrs['units']
    return new_ds

def dawn_acc_ds(obs_align,cwrf_align,var_name = 'PRAVG'):
    obs_align_us,cwrf_align_us    = US_MASK(obs_align),US_MASK(cwrf_align)
    obs_align_cb,cwrf_align_cb    = CB_MASK(obs_align),CB_MASK(cwrf_align)
    # Calculate the anomaly
    obs_us_anomy  = obs_align_us  - obs_align_us.mean( dim = ['south_north', 'west_east'])
    cwrf_us_anomy = cwrf_align_us - cwrf_align_us.mean(dim = ['south_north', 'west_east'])
    # Calculate ACC (apply mask to avoid affected by outer boundary values)
    acc_us        = xr.corr(US_MASK(obs_us_anomy[var_name]), US_MASK(cwrf_us_anomy[var_name]) ,dim = ['south_north', 'west_east'])
    # Calculate the anomaly based on Corn Belt.
    obs_cb_anomy  = obs_align_cb  - obs_align_cb.mean( dim = ['south_north', 'west_east'])
    cwrf_cb_anomy = cwrf_align_cb - cwrf_align_cb.mean(dim = ['south_north', 'west_east'])
    # Calculate ACC (apply mask to avoid affected by outer boundary values) 
    acc_cb        = xr.corr(CB_MASK(obs_cb_anomy[var_name]), CB_MASK(cwrf_cb_anomy[var_name]) ,dim = ['south_north', 'west_east'])
    return acc_us,acc_cb

def dawn_acc_da(obs_align,cwrf_align):
    obs_align_us,cwrf_align_us    = US_MASK(obs_align),US_MASK(cwrf_align)
    obs_align_cb,cwrf_align_cb    = CB_MASK(obs_align),CB_MASK(cwrf_align)
    # Calculate the anomaly
    obs_us_anomy  = obs_align_us  - obs_align_us.mean( dim = ['south_north', 'west_east'])
    cwrf_us_anomy = cwrf_align_us - cwrf_align_us.mean(dim = ['south_north', 'west_east'])
    # Calculate ACC (apply mask to avoid affected by outer boundary values)
    acc_us        = xr.corr(US_MASK(obs_us_anomy), US_MASK(cwrf_us_anomy) ,dim = ['south_north', 'west_east'])
    # Calculate the anomaly based on Corn Belt.
    obs_cb_anomy  = obs_align_cb  - obs_align_cb.mean( dim = ['south_north', 'west_east'])
    cwrf_cb_anomy = cwrf_align_cb - cwrf_align_cb.mean(dim = ['south_north', 'west_east'])
    # Calculate ACC (apply mask to avoid affected by outer boundary values) 
    acc_cb        = xr.corr(CB_MASK(obs_cb_anomy), CB_MASK(cwrf_cb_anomy) ,dim = ['south_north', 'west_east'])
    return acc_us,acc_cb

def calculate_simu_metric(ds_obs_us, ds_cwrf):
    ds_cwrf['PRAVG'].attrs['units']  = 'mm/d'
    ds_cwrf_us    = to_mnth(ds_cwrf,'PRAVG')
    metric_simu   = xr.Dataset()
    # Align the observation and simulation
    obs_align_us, cwrf_align_us = xr.align(ym(ds_obs_us),ym(ds_cwrf_us), join = 'inner')
    # Calculate RMSE, IAC, ACC
    metric_simu['IAC']          = xr.corr(  obs_align_us['PRAVG'], cwrf_align_us['PRAVG'] ,dim = 'time')
    metric_simu['RMSE']         = dawn_rmse(obs_align_us['PRAVG'], cwrf_align_us['PRAVG'] ,dim = 'time')
    metric_simu['bias']         = cwrf_align_us['PRAVG'] - obs_align_us['PRAVG']
    metric_simu['ACC_US'],metric_simu['ACC_CB'] = dawn_acc_ds(obs_align_us,cwrf_align_us,var_name = 'PRAVG')
    return metric_simu

def calculate_statistics_of_precipitation_indices(ds_cwrf,ds_obs):
    inds_year     = ['R95p','R99p','PRCPTOT','JJATOTAL','wetdays','CDD','CWD','SDII','Rx1day','Rx5day','R10mm','R20mm']
    list_metric   = []
    for ind in inds_year:
        ds_metric = xr.Dataset()
        obs_align_us,cwrf_align_us = xr.align(ds_obs[ind]   ,ds_cwrf[ind]   , join = 'inner')
        ds_metric['IAC']        = xr.corr(  obs_align_us, cwrf_align_us ,dim = 'year')
        ds_metric['RMSE']       = dawn_rmse(obs_align_us, cwrf_align_us ,dim = 'year')
        ds_metric['bias']       =           cwrf_align_us - obs_align_us 
        ds_metric['ACC_US'],ds_metric['ACC_CB'] = dawn_acc_da(obs_align_us,cwrf_align_us)
        list_metric.append(ds_metric.expand_dims(ind = [ind]))
    result        = xr.concat(list_metric,dim='ind')
    return result


def quantile_of_pravg(da_obs):
    q_list = [x*0.01 for x in range(1,100)]
    obs_da = US_MASK(da_obs ).transpose('time', 'south_north', 'west_east') # Mask
    obs_da = xr.where(obs_da>=1 ,obs_da,np.nan)                             # Wet day only
    obs_quantile_list = [obs_da.quantile(x,skipna=True) for x in q_list]
    da_quantile       = xr.concat(obs_quantile_list,dim= 'quantile')
    return da_quantile

def quantile_of_pravg_all_days(da_obs):
    q_list = [x*0.01 for x in range(1,100)]
    obs_da = US_MASK(da_obs ).transpose('time', 'south_north', 'west_east') # Mask
    obs_quantile_list = [obs_da.quantile(x,skipna=True) for x in q_list]
    da_quantile       = xr.concat(obs_quantile_list,dim= 'quantile')
    return da_quantile


# --- Ploting related ---
def get_list_simulation_metric(metric):
    ih_exp = ih_exps[0]
    list_cwrf,list_bc0,list_bc1,list_bc2,list_re = [],[],[],[],[]
    for comb in cwrf_combination:
        init_mnth,day,ih_exp     = comb
        list_cwrf.append(            xr.open_dataset(f'{path_simu_metric}CWRF_{var_name}_{                years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc0.append(             xr.open_dataset(f'{path_bc_metric  }CWRF_{var_name}_{bc_methods[0]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc1.append(             xr.open_dataset(f'{path_bc_metric  }CWRF_{var_name}_{bc_methods[1]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc2.append(             xr.open_dataset(f'{path_bc_metric  }CWRF_{var_name}_{bc_methods[2]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_re.append(              xr.open_dataset(f'{path_bc_metric  }recover_5to1day_CWRF_{var_name}_{bc_methods[0]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
    da_cwrf   = xr.concat(list_cwrf,dim='ensemble').mean(dim='ensemble')
    da_bc0    = xr.concat(list_bc0 ,dim='ensemble').mean(dim='ensemble')
    da_bc1    = xr.concat(list_bc1 ,dim='ensemble').mean(dim='ensemble')
    da_bc2    = xr.concat(list_bc2 ,dim='ensemble').mean(dim='ensemble')
    da_re     = xr.concat(list_re  ,dim='ensemble').mean(dim='ensemble')
    # list_da = [da_cwrf, da_bc0, da_bc1 , da_bc2,da_re]
    list_da = [da_cwrf,da_bc0,da_re]
    return list_da

def get_ind_simulation_metric(metric):
    list_cwrf,list_bc0,list_bc1,list_bc2,list_re = [],[],[],[],[]
    for comb in cwrf_combination:
        init_mnth,day,ih_exp   = comb
        list_cwrf.append(  xr.open_dataset(f'{path_stttc_simu}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc0.append(   xr.open_dataset(f'{path_stttc_bc}CWRF_{var_name}_{bc_methods[0]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc1.append(   xr.open_dataset(f'{path_stttc_bc}CWRF_{var_name}_{bc_methods[1]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_bc2.append(   xr.open_dataset(f'{path_stttc_bc}CWRF_{var_name}_{bc_methods[2]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
        list_re.append(    xr.open_dataset(f'{path_stttc_bc}recover_5to1day_CWRF_{var_name}_{bc_methods[0]}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc')[metric] )
    da_cwrf   = xr.concat(list_cwrf,dim='ensemble').mean(dim='ensemble')
    da_bc0    = xr.concat(list_bc0 ,dim='ensemble').mean(dim='ensemble')
    da_bc1    = xr.concat(list_bc1 ,dim='ensemble').mean(dim='ensemble')
    da_bc2    = xr.concat(list_bc2 ,dim='ensemble').mean(dim='ensemble')
    da_re     = xr.concat(list_re  ,dim='ensemble').mean(dim='ensemble')
    list_da = [da_cwrf, da_bc0, da_bc1 , da_bc2, da_re]
    return list_da

def calculate_crps_brier_score(da_obs,cwrf_ensemble,dim='member',threshold = 1.0):
    # resample to yearly mean
    da_obs_yearly        = da_obs.resample(time='YE').mean()
    cwrf_ensemble_yearly = cwrf_ensemble.resample(time='YE').mean()
    # Calculate the CRPS
    crps = xr.apply_ufunc(
        lambda obs, ens: crps_ensemble(obs, ens, axis=0),
        da_obs_yearly, cwrf_ensemble_yearly,
        input_core_dims=[[], ["member"]],  # 'obs' has no core dim, 'ens' has core dim "member"
        vectorize = True,                    # automatically vectorize over non-core dims
        dask="parallelized",               # if using dask, parallelize the operation
        output_dtypes=[float]
    )
    # Calculate the brier_score
    ensemble_prob = (cwrf_ensemble_yearly >= threshold).mean(dim="member") # Convert the ensemble forecasts to a probability of exceeding the threshold
    obs_binary    = (da_obs_yearly >= threshold).astype(float)# Convert the observations to binary (1 if event occurred, 0 otherwise)
    brier_score   = (ensemble_prob - obs_binary)**2          # The Brier Score is the squared difference between the forecast probability and the observed outcome:
    return crps, brier_score

def avg_list(list_cnn):
        return xr.concat(list_cnn ,dim='ensemble').mean(dim='ensemble')

# for equitable_threat_score calculation
def ets_func(obs_series, fcst_series):
    H = np.sum((obs_series == 1) & (fcst_series == 1))
    F = np.sum((obs_series == 0) & (fcst_series == 1))
    M = np.sum((obs_series == 1) & (fcst_series == 0))
    N = len(obs_series)
    H_r = ((H + F) * (H + M)) / N
    denominator = (H + F + M - H_r)
    if denominator == 0:
        return np.nan
    else:
        return (H - H_r) / denominator

def equitable_threat_score(da_obs, da_cwrf):
    obs_align_us,cwrf_align_us = xr.align(da_obs, da_cwrf, join = 'inner') # Align
    # Convert observations and forecasts to binary events (1 if event occurs, else 0)
    threshold= 1.0
    obs_bin  = (obs_align_us  >= threshold).astype(int)
    fcst_bin = (cwrf_align_us >= threshold).astype(int)
    ets_grid = xr.apply_ufunc( ets_func,   obs_bin, fcst_bin,
        input_core_dims=[['time'], ['time']],
        vectorize=True,       # Automatically vectorize over non-core dims (south_north, west_east)
        dask='parallelized',  # Use parallelization if your arrays are dask-backed
        output_dtypes=[float]  )
    return ets_grid

def kdeplot_seasonal(list_da,metric,figfmt):
    colors     = ['blue','darkorange','cyan','skyblue', 'r']
    labels     = ['CWRF','EQM','DQM','QDM','CNN-BC']
    list_np    = [US_MASK(x).values.flatten() for x in list_da]
    plt.figure(figsize=(2.99, 1.9182))
    for i, color in enumerate(colors):
        sns.kdeplot(list_np[i], label=labels[i], color = color, fill=False)
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{path_figures}summer_avg/KDE_{metric}_plot.{figfmt}')
print('end of self_defined function')

# # *************** Combine Dataset ***************
# # --- CWRF ---
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp = comb
#     file_name_output     = f'{path_combine}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#     if not os.path.exists(file_name_output):
#         file_list        = []
#         for year in years:
#             file_name    = f'{path_cwrf}{year}{init_mnth:02}{day:02}/{year}{init_mnth:02}{day:02}{ih_exp}_{var_name}_daily.nc'
#             if os.path.exists(file_name):
#                 file_list.append(file_name)
#         ds_CWRF          = xr.open_mfdataset(file_list)
#         ds_sel           = ds_CWRF.sel(time = ds_CWRF['time'].dt.month.isin([6,7,8]))
#         if 'bottom_top' in ds_sel.dims and ds_sel.dims['bottom_top'] == 1:
#             ds_sel       = ds_sel.squeeze('bottom_top',drop=True)
#         ds_sel[var_name] = ds_sel[var_name]*86400.0
#         ds_sel.to_netcdf(path = file_name_output )
#         print(f'{file_name_output} has been saved!')

# # --- Observation ---
# if rank == 0:
#     file_list    = []
#     for year in years:
#         file_list.append(f'{path_obs}OBS_{var_name}_{year}-01-01-00_{year}-12-31-18.nc')
#     ds_obs       = xr.open_mfdataset(file_list)
#     ds_JJA       = ds_obs.sel(time = ds_obs['time'].dt.month.isin([6,7,8]))
#     if 'crs' in ds_JJA.dims:
#         ds_JJA   = ds_JJA.drop_vars('crs')
#     file_name_output = f'{path_combine}OBS_{var_name}_{years[0]}-{years[-1]}_{pred_season_str}.nc'
#     ds_JJA.to_netcdf(path = file_name_output )

# Synchronize all processes before finishing
# comm.Barrier()

# # # --- 100 year return maximum precipitation ---
# if not os.path.exists(f'{path_combine}/newOBS_100year_return_level_estimation.nc'):
#     return_period   = 100
#     if rank    == 0:
#         da     = xr.open_dataset(f'{path_combine}OBS_PRAVG_2012-2023_JJA.nc')['PRAVG']
#         south_north_dim, west_east_dim = da.south_north.size, da.west_east.size
#     else:
#         da     = south_north_dim = west_east_dim = None
#     south_north_dim = comm.bcast(south_north_dim, root=0)
#     west_east_dim   = comm.bcast(west_east_dim, root=0)


#     start, end = compute_indices(south_north_dim, rank, size)
#     da_subset  = xr.open_dataset(f'{path_combine}OBS_PRAVG_2012-2023_JJA.nc', chunks={'south_north': end - start})['PRAVG'].isel(south_north=slice(start, end))
#     da_subset  = da_subset.assign_coords(year=da_subset['time.year'])
#     annual_max = da_subset.groupby('year').max(dim='time').chunk({'year': -1})
#     return_level_subset = xr.apply_ufunc(
#         fit_gev_and_get_return_level,     annual_max,    input_core_dims=[['year']],        output_core_dims=[[]],    kwargs={'return_period': return_period},
#               vectorize=True,    dask='parallelized',    output_dtypes=[annual_max.dtype],  dask_gufunc_kwargs={'allow_rechunk': True})
#     return_level_np = return_level_subset.compute().values
#     data_list       = comm.gather(return_level_np, root=0)
#     if rank == 0:
#         return_level_full = np.concatenate(data_list, axis=0)
#         return_level      = xr.DataArray(data=return_level_full, name='PRAVG', dims=('south_north', 'west_east'), coords={'south_north':da.south_north.values,'west_east': da.west_east.values})
#         return_level.to_dataset(name='PRAVG').to_netcdf(f'{path_combine}/newOBS_100year_return_level_estimation.nc')

# # Synchronize all processes before finishing
# comm.Barrier()

# # ## *************** Bias-Correction ***************
# import xclim
# from   xclim                   import sdba
# import xclim.sdba.adjustment   as     xclim_bc 

# groups             = [ 'time', sdba.Grouper("time.dayofyear", window=3), 'time.month']                                        # Generate the groups list
# # --- Apply BC to CWRF ---
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp = comb
#     obs_da_subset,cwrf_da_subset,new_dataset,valid_times,num_years = bias_correction_read_data_cwrf(path_combine,var_name,years,init_mnth,day,pred_season_str,ih_exp)

#     bc_method       = 'EQM'                                                                                           # Bias-correction method
#     group_rank,kind,nquantile,thresh = xclim_parameters(bc_method,var_name,pred_season_str)                           # Parameters
#     kind            = '*'
#     print(bc_method,group_rank,kind,nquantile,thresh)
#     export_file_name= f'{path_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'  # Export file name
#     kf,list_adjusted= KFold(n_splits = num_years, random_state=None, shuffle=False),[]                                # Apply KFold (need year information)
#     for train_index,test_index in kf.split(obs_da_subset):                                                            # Leave one year out
#         ref,hist,sim= obs_da_subset[train_index], cwrf_da_subset[train_index], cwrf_da_subset[test_index]             # Split the data into 'training' and 'testing'
#         ADJ         = xclim_bc.EmpiricalQuantileMapping.train(ref, hist, group=groups[group_rank],kind=kind,nquantiles=nquantile)      # Training
#         list_adjusted.append(ADJ.adjust(sim))       
#     bias_correction_export(var_name,list_adjusted,new_dataset,valid_times,export_file_name)                           # Exporting
#     print('BC is applied and saved as: ',export_file_name)

#     bc_method       = 'DQM'                                                                                           # Bias-correction method
#     group_rank,kind,nquantile,thresh = xclim_parameters(bc_method,var_name,pred_season_str)                           # Parameters
#     kind            = '*'
#     print(bc_method,group_rank,kind,nquantile,thresh)
#     export_file_name= f'{path_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'  # Export file name
#     kf,list_adjusted= KFold(n_splits = num_years, random_state=None, shuffle=False),[]                                # Apply KFold (need year information)
#     for train_index,test_index in kf.split(obs_da_subset):                                                            # Leave one year out
#         ref,hist,sim= obs_da_subset[train_index], cwrf_da_subset[train_index], cwrf_da_subset[test_index]             # Split the data into 'training' and 'testing'
#         ADJ         = xclim_bc.DetrendedQuantileMapping.train(ref, hist, group=groups[group_rank],kind=kind,nquantiles=nquantile)      # Training
#         list_adjusted.append(ADJ.adjust(sim))                                                                         # Fitting
#     bias_correction_export(var_name,list_adjusted,new_dataset,valid_times,export_file_name)                           # Exporting

#     bc_method       = 'QDM'                                                                                           # Bias-correction method
#     group_rank,kind,nquantile,thresh = xclim_parameters(bc_method,var_name,pred_season_str)                           # Parameters
#     kind            = '*'
#     print(bc_method,group_rank,kind,nquantile,thresh)
#     export_file_name= f'{path_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'  # Export file name
#     kf,list_adjusted= KFold(n_splits = num_years, random_state=None, shuffle=False),[]                                # Apply KFold (need year information)
#     for train_index,test_index in kf.split(obs_da_subset):                                                            # Leave one year out
#         ref,hist,sim= obs_da_subset[train_index], cwrf_da_subset[train_index], cwrf_da_subset[test_index]             # Split the data into 'training' and 'testing'
#         ADJ         = xclim_bc.QuantileDeltaMapping.train(ref, hist, group=groups[group_rank],kind=kind,nquantiles=nquantile)          # Training
#         list_adjusted.append(ADJ.adjust(sim))                                                                         # Fitting
#     bias_correction_export(var_name,list_adjusted,new_dataset,valid_times,export_file_name)                           # Exporting

# comm.Barrier()

# # ************ Extreme precipitation indices1 ****************
# # --- Calculate Extreme indices of precipitation ---
# # 1 Observation
# if rank == 0:
#     obs_ds            = xr.open_dataset( f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')
#     obs_da            = obs_ds['PRAVG']
#     obs_da.attrs['units']  = 'mm/d'
#     obs_da_indices    = precipitation_indices_jja(obs_da)
#     obs_da_indices.to_netcdf(f'{path_ind_simu}OBS.nc')
#     print('Extreme precipitation indices is calculated and saved as:',f'{path_ind_simu}OBS.nc')

# # 2 CWRF simulation
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     filename      = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#     in_fname      = f'{path_simu    }{filename}'
#     out_fname     = f'{path_ind_simu}{filename}'
#     if not os.path.exists(out_fname):
#         da_cwrf   = xr.open_dataset(in_fname)['PRAVG']
#         da_cwrf.attrs['units']  = 'mm/d'
#         da_cwrf_jja_ind         = precipitation_indices_jja(da_cwrf)
#         da_cwrf_jja_ind.to_netcdf(out_fname)
#     else:
#         print(f"The file {out_fname} exists.")

# # 3 CWRF Bias-correction result
# for comb in bc_combination[rank::size]:
#     bc_method, init_mnth,day,ih_exp    = comb
#     filename      = f'CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#     in_fname      = f'{path_bc    }{filename}'
#     out_fname     = f'{path_ind_bc}{filename}'
#     if not os.path.exists(out_fname):
#         da_cwrf   = xr.open_dataset(in_fname)['PRAVG']
#         da_cwrf.attrs['units']  = 'mm/d'
#         da_cwrf_jja_ind         = precipitation_indices_jja(da_cwrf)
#         da_cwrf_jja_ind.to_netcdf(out_fname)
#     else:
#         print(f"The file {out_fname} exists.")
# comm.Barrier()
# ### ************ CNN for Rx1day and Rx5day ****************
# import tensorflow              as     tf
# from   tensorflow.keras.models import Sequential
# from   tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
# from   sklearn.model_selection import train_test_split  # For splitting validation from training
# indices                      = ['Rx1day','Rx5day']
# # # --- For CWRF ---
# if rank      == 0:
#     for comb in cwrf_combination:
#         init_mnth,day,ih_exp     = comb
#         for index in indices:
#             path_cnn             = f'{path_project}CNN_{index}/'
#             da_obs               = xr.open_dataset(f'{path_ind_simu}OBS.nc')[index]
#             file_name            = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#             ic(f'{path_cnn}{file_name}')
#             if not os.path.exists(f'{path_cnn}{file_name}'):
#                 # try:
#                 da_simu          = xr.open_dataset(f'{path_ind_simu}{file_name}')[index]
#                 da_simu_align,da_obs_align = xr.align(da_simu,da_obs,join='inner')
#                 # Convert xarray DataArrays to numpy arrays
#                 obs_data         = da_obs_align.values
#                 simu_data        = da_simu_align.values
#                 # Define the shape (time, height, width)
#                 time_steps       = obs_data.shape[0]       # Number of time steps
#                 height           = obs_data.shape[1]       # Spatial dimension 1 (height)
#                 width            = obs_data.shape[2]       # Spatial dimension 2 (width)
#                 # Reshape the predictor (da_simu) to match the format (time, height, width, channels)
#                 simu_data_reshaped = simu_data.reshape((time_steps, height, width, 1))
#                 list_opt_result  = []
#                 # Loop through each year, making it the test year
#                 for test_year in range(time_steps):
#                     print(f"\n--- Testing Year: {test_year + 1} ---")
#                     # Create a list of all years except the current test year
#                     train_val_years = np.delete(np.arange(time_steps), test_year)
#                     # Randomly select 3 years for validation from the remaining 11 years
#                     np.random.shuffle(train_val_years)
#                     val_years    = train_val_years[:3]
#                     train_years  = train_val_years[3:]
#                     # Extract training data
#                     train_simu   = simu_data_reshaped[train_years]
#                     train_obs    = obs_data[train_years].reshape((len(train_years), height, width, 1))
#                     # Extract validation data
#                     val_simu     = simu_data_reshaped[val_years]
#                     val_obs      = obs_data[val_years].reshape((len(val_years), height, width, 1))
#                     # Extract test data
#                     test_simu    = simu_data_reshaped[test_year:test_year + 1]  # Only the current test year
#                     test_obs     = obs_data[test_year:test_year + 1].reshape((1, height, width, 1))
#                     # Build the CNN model (rebuild for each iteration to avoid interference from previous runs)
#                     model        = Sequential()
#                     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)))
#                     model.add(MaxPooling2D(pool_size=(2, 2)))
#                     model.add(Conv2D(64, (3, 3), activation='relu'))
#                     model.add(MaxPooling2D(pool_size=(2, 2)))
#                     model.add(Flatten())
#                     model.add(Dense(128, activation='relu'))
#                     model.add(Dense(height * width, activation='linear'))
#                     model.add(Reshape((height, width, 1)))
#                     # Compile the model
#                     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#                     # Train the model
#                     history      = model.fit(train_simu, train_obs, validation_data=(val_simu, val_obs), epochs=20, batch_size=2)
#                     # Evaluate the model on the test set
#                     test_loss, test_mae       = model.evaluate(test_simu, test_obs)
#                     print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
#                     # Make predictions on the test set
#                     test_predictions          = model.predict(test_simu)
#                     test_predictions_reshaped = test_predictions.squeeze(axis=-1)
#                     # Convert predictions to xarray for further analysis
#                     test_pred_da = xr.DataArray(test_predictions_reshaped, dims=["year", "south_north", "west_east"], coords={"year": da_obs_align.year[test_year:test_year + 1]})
#                     test_pred_ds = test_pred_da.to_dataset(name=index)
#                     list_opt_result.append(test_pred_ds)
#                 opt_ds           = xr.concat(list_opt_result,dim='year')
#                 opt_ds.to_netcdf(f'{path_cnn}{file_name}')


#  ### ************ Recover ****************
# obs_wetdays            = xr.open_dataset(f'{path_ind_simu}OBS.nc')['wetdays'].mean(dim='year')    # Area with wetdasy less than 5 days, not apply recover
# # --- For Rx1day only ---
# for comb in  cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp     = comb
#     findex_name        = f'CWRF_PRAVG_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#     da_rx1d            = xr.open_dataset(f'{path_rx1day}{findex_name}')['Rx1day']
#     for bc_method in  bc_methods:
#         fbc_name       = f'CWRF_PRAVG_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#         da_bc          = xr.open_dataset(f'{path_bc}{fbc_name}')['PRAVG']
#         da_bc_modified = da_bc.copy()
#         list_recover   = []
#         for year in sorted(set(da_bc['time.year'].values)):  # Iterate over each year in da_rx1d
#             rx1d_value = da_rx1d.sel(year=year).values
#             max_values = da_bc.sel(time=str(year)).max(dim='time')
#             da_replace =  xr.where(da_bc_modified.sel(time=str(year)) == max_values,rx1d_value, da_bc_modified.sel(time=str(year)))
#             da_replace =  xr.where(obs_wetdays >=5, da_replace, da_bc_modified.sel(time=str(year)) )  # for threshold 
#             list_recover.append( da_replace )
#         da_recover     = xr.concat(list_recover,dim='time').transpose('time','south_north', 'west_east')
#         ds             = da_recover.to_dataset(name = 'PRAVG')
#         ds.to_netcdf(f'{path_recover}recover_1day_{fbc_name}')
# comm.Barrier()

# # --- For Rx5day only ---
# for comb in  cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp     = comb
#     findex_name   = f'CWRF_PRAVG_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#     da_rx5d       = xr.open_dataset(f'{path_rx5day}{findex_name}')['Rx5day']
#     for bc_method in  bc_methods:
#         fbc_name  = f'CWRF_PRAVG_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#         if not os.path.exists(f'{path_recover}recover_5day_{fbc_name}'):
#             da_bc            = xr.open_dataset(f'{path_bc}{fbc_name}')['PRAVG']
#             list_recover     = []
#             unique_years     = sorted(set(da_bc['time.year'].values))
#             for year in unique_years:
#                 da_bc_year   = da_bc.sel(time=str(year))
#                 da_rollsum   = da_bc_year.rolling(time=5, min_periods=5).sum()
#                 max_5day_sum = da_rollsum.max(dim='time')
#                 idx_of_max   = da_rollsum.fillna(-9999).argmax(dim='time')
#                 rx5d_value   = da_rx5d.sel(year=year)#.squeeze('year', drop=True)
#                 scaling_factor = rx5d_value / max_5day_sum
#                 scaling_factor = xr.where(max_5day_sum == 0, 1 ,scaling_factor)
#                 da_bc_year_modified = da_bc_year.copy()
#                 time_dim_size     = da_bc_year.sizes['time']
#                 south_north_size  = da_bc_year.sizes['south_north']
#                 west_east_size    = da_bc_year.sizes['west_east']
#                 window_indices    = idx_of_max.values[..., None] + np.arange(5)
#                 window_indices    = np.clip(window_indices, 0, time_dim_size - 1)
#                 da_bc_values      = da_bc_year.values  # Shape: (time, south_north, west_east)
#                 da_bc_values      = da_bc_values.transpose(1, 2, 0)
#                 adjusted_values   = np.zeros((south_north_size, west_east_size, 5))
#                 for i in range(south_north_size):
#                     for j in range(west_east_size):
#                         indices   = window_indices[i, j, :]
#                         original_values = da_bc_values[i, j, indices]
#                         factor    = scaling_factor.values[i, j]
#                         adjusted_values[i, j, :] = original_values * factor
#                 for k in range(5):
#                     indices_k     = window_indices[:, :, k]
#                     for i in range(south_north_size):
#                         for j in range(west_east_size):
#                             time_idx = indices_k[i, j]
#                             da_bc_year_modified.values[time_idx, i, j] = adjusted_values[i, j, k]
#                 da_bc_year_modified  = xr.where(obs_wetdays >=5, da_bc_year_modified, da_bc_year).transpose('south_north', 'west_east', 'time')
#                 list_recover.append(da_bc_year_modified)
#             da_recover = xr.concat(list_recover, dim='time').transpose('time','south_north', 'west_east')
#             ds    = da_recover.to_dataset(name='PRAVG')
#             ds.to_netcdf(f'{path_recover}recover_5day_{fbc_name}')
# comm.Barrier()

# #  For Rx5day first than Rx1day
# for comb in  cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp     = comb
#     findex_name   = f'CWRF_PRAVG_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#     da_rx1d            = xr.open_dataset(f'{path_rx1day}{findex_name}')['Rx1day']
#     for bc_method in  bc_methods:
#         fbc_name  = f'CWRF_PRAVG_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#         da_bc          = xr.open_dataset(f'{path_recover}recover_5day_{fbc_name}')['PRAVG']  # based on which dataset is the recover performed?
#         da_bc_modified = da_bc.copy()
#         list_recover   = []
#         for year in sorted(set(da_bc['time.year'].values)):
#             max_values = da_bc.sel(time=str(year)).max(dim='time')
#             rx1d_value = da_rx1d.sel(year=year).values
#             rx1d_value = xr.where(obs_wetdays >=5, rx1d_value, max_values ).values
#             list_recover.append( xr.where(da_bc_modified.sel(time=str(year)) == max_values,rx1d_value, da_bc_modified.sel(time=str(year))))
#         da_recover     = xr.concat(list_recover,dim='time').transpose('time','south_north', 'west_east')
#         ds             = da_recover.to_dataset(name = 'PRAVG')
#         ds.to_netcdf(f'{path_recover}recover_5to1day_{fbc_name}')
# comm.Barrier()

# ### ************ Calculate the quantiles ****************
# da_obs       = xr.open_dataset(f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')['PRAVG']  
# if rank      == 1:                                         # --- For OBS ---
#     file_name     = f'OBS_PRAVG_2012-2023_JJA.nc'
#     if not os.path.exists(f'{path_quantile}{file_name}'):
#         obs_q     = quantile_of_pravg(da_obs)
#         ds_obs_q  = obs_q.to_dataset(name = 'PRAVG')
#         ds_obs_q.to_netcdf(f'{path_quantile}{file_name}')

# comm.Barrier()

# for comb in  cwrf_combination[rank::size]:                 # --- For CWRF ---
#     init_mnth,day,ih_exp     = comb
#     file_name = f'CWRF_PRAVG_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#     print(file_name)
#     da_cwrf   = xr.open_dataset(f'{path_simu}{file_name}')['PRAVG']
#     obs_da_subset,cwrf_da_subset =  xr.align(da_obs,da_cwrf, join = 'inner')
#     cwrf_q    = quantile_of_pravg(cwrf_da_subset)
#     ds_cwrf_q = cwrf_q.to_dataset(name = 'PRAVG')
#     ds_cwrf_q.to_netcdf(f'{path_quantile}{file_name}')

# # #NOAA 
# for comb in noaa_combination[rank::size]:
#     init_mnth,day,init_hour = comb
#     file_name      = f'NOAA_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_hour-{init_hour:02}_{pred_season_str}.nc'
#     da_cfs         = xr.open_dataset(f'{path_simu}{file_name}')['PRAVG']
#     obs_da_subset,cfs_da_subset =  xr.align(da_obs,da_cfs, join = 'inner')
#     cfs_q          = quantile_of_pravg(cfs_da_subset)
#     ds_cfs_q       = cfs_q.to_dataset(name = 'PRAVG')
#     ds_cfs_q.to_netcdf(f'{path_quantile}{file_name}')


# for comb in  cwrf_combination[rank::size]:                 # --- For BC result  ---
#     init_mnth,day,ih_exp   = comb
#     for bc_method in bc_methods:
#         file_name          = f'CWRF_PRAVG_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#         if not os.path.exists(f'{path_quantile}all_days_{file_name}'):
#             da_cwrf        = xr.open_dataset(f'{path_bc}{file_name}')['PRAVG']
#             obs_da_subset,cwrf_da_subset =  xr.align(da_obs,da_cwrf, join = 'inner')
#             cwrf_q       = quantile_of_pravg(cwrf_da_subset)
#             ds_cwrf_q    = cwrf_q.to_dataset(name = 'PRAVG')
#             ds_cwrf_q.to_netcdf(f'{path_quantile}{file_name}')

# recover_methods = ['recover_1day_','recover_5day_','recover_5to1day_']
# for comb in  cwrf_combination[rank::size]:                 # # --- For CWRF recover---
#     init_mnth,day,ih_exp   = comb
#     for recover_method in recover_methods:
#         for bc_method  in bc_methods:
#             file_name  = f'CWRF_PRAVG_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_JJA_{ih_exp}.nc'
#             if not os.path.exists(f'{path_quantile}all_days_{recover_method}{file_name}'):
#                 da_cwrf    = xr.open_dataset(f'{path_recover}{recover_method}{file_name}')['PRAVG']
#                 obs_da_subset,cwrf_da_subset =  xr.align(da_obs,da_cwrf, join = 'inner')
#                 cwrf_q     = quantile_of_pravg(cwrf_da_subset)
#                 ds_cwrf_q  = cwrf_q.to_dataset(name = 'PRAVG')
#                 ds_cwrf_q.to_netcdf(f'{path_quantile}{recover_method}{file_name}')
# comm.Barrier()


# # # ************ Extreme precipitation indices2  ****************
# # 4 CWRF Bias-correction recover result
# recover_methods = ['recover_1day_','recover_5day_','recover_5to1day_']
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     for recover_method in recover_methods:
#         for bc_method in bc_methods:
#             filename      = f'{recover_method}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#             in_fname      = f'{path_recover}{filename}'
#             out_fname     = f'{path_ind_bc }{filename}'
#             if not os.path.exists(out_fname):
#                 da_cwrf   = xr.open_dataset(in_fname)['PRAVG']
#                 da_cwrf.attrs['units']  = 'mm/d'
#                 da_cwrf_jja_ind         = precipitation_indices_jja(da_cwrf)
#                 da_cwrf_jja_ind.to_netcdf(out_fname)
#             else:
#                 print(f"The file {out_fname} exists.")
# comm.Barrier()


# # # ************ Metric of monthly and extreme precipitation ************
# # # # --- Calculat the metric of monthly precipitation ---
# # Read the observation data
# obs_ds        = xr.open_dataset( f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')
# obs_ds['PRAVG'].attrs['units']  = 'mm/d'
# ds_obs_us     = to_mnth(obs_ds,'PRAVG')
# # 1 statistics of CWRF
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     ds_cwrf   = xr.open_dataset(f'{path_simu}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
#     calculate_simu_metric(ds_obs_us, ds_cwrf).to_netcdf(f'{path_simu_metric}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
# # 2 statistics of BCs-CWRF
# for comb in bc_combination[rank::size]:
#     bc_method, init_mnth,day,ih_exp    = comb
#     ds_cwrf   = xr.open_dataset(f'{path_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
#     calculate_simu_metric(ds_obs_us, ds_cwrf).to_netcdf(f'{path_bc_metric}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
# # 3 statistics of BCs-recover
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     for recover_method in recover_methods:
#         for bc_method in bc_methods:
#             filename      = f'{recover_method}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#             ds_recover    = xr.open_dataset(f'{path_recover}{filename}')
#             calculate_simu_metric(ds_obs_us, ds_recover).to_netcdf(f'{path_bc_metric}{filename}')
# comm.Barrier()

# # # --- Calculat the metric of Extreme precipitation indices ---
# ds_obs         = xr.open_dataset((f'{path_ind_simu}OBS.nc'))
# # 1 statistics of CWRF
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     ds_cwrf    = xr.open_dataset(f'{path_ind_simu}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
#     metric     = calculate_statistics_of_precipitation_indices(ds_cwrf,ds_obs)
#     metric.to_netcdf(f'{path_stttc_simu}CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
# # 2 statistics of BCs-CWRF
# for comb in bc_combination[rank::size]:
#     bc_method, init_mnth,day,ih_exp    = comb
#     ds_cwrf    = xr.open_dataset(f'{path_ind_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
#     metric     = calculate_statistics_of_precipitation_indices(ds_cwrf,ds_obs)
#     metric.to_netcdf(f'{path_stttc_bc}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')
# # 3 statistics of BCs-recover
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp  = comb
#     for recover_method in recover_methods:
#         for bc_method in bc_methods:
#             filename      = f'{recover_method}CWRF_{var_name}_{bc_method}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'
#             ds_recover    = xr.open_dataset(f'{path_ind_bc}{filename}')
#             metric        = calculate_statistics_of_precipitation_indices(ds_recover,ds_obs)
#             metric.to_netcdf(f'{path_stttc_bc}{filename}')
# comm.Barrier()


# # generate mask for each region
# path_static      = '/Volumes/ssd3/DATA/static/'
# da_US            = xr.open_dataset(f'{path_static}US_MASK_logic.nc')['MASK']
# name_regions     = ['Cascades','Northern_Rockies','Central_Great_Plains','Midwest','Northeast','Southeast','Gulf_States','North_American_monsoon']

# ds               = xr.Dataset()
# for n_region in name_regions:
#     ds[n_region] = xr.full_like(da_US, fill_value=False)

# ds['Cascades'              ].loc[dict(south_north=slice(20,  36), west_east=slice(85, 121))] = True
# ds['Northern_Rockies'      ].loc[dict(south_north=slice(42,  62), west_east=slice(89, 117))] = True
# ds['Central_Great_Plains'  ].loc[dict(south_north=slice(74,  91), west_east=slice(55,  90))] = True
# ds['Midwest'               ].loc[dict(south_north=slice(104,133), west_east=slice(66,  93))] = True
# ds['Northeast'             ].loc[dict(south_north=slice(139,166), west_east=slice(80, 116))] = True
# ds['Southeast'             ].loc[dict(south_north=slice(134,155), west_east=slice(28,  69))] = True
# ds['Gulf_States'           ].loc[dict(south_north=slice(94, 128), west_east=slice(39,  63))] = True
# ds['North_American_monsoon'].loc[dict(south_north=slice(51,  70), west_east=slice(26,  68))] = True
# ds['name_regions']         = name_regions
# ds.to_netcdf(f'{path_static}regions_MASK_logic.nc')

# # ---For moment calculation ---
import xarray as xr
from scipy.stats import skew, kurtosis

# def pravg_moment_calculation_yearly(da_obs_aligned):
#     ds_mean = da_obs_aligned.groupby("time.year").mean(dim="time", skipna=True)
#     ds_std  = da_obs_aligned.groupby("time.year").std(dim="time", skipna=True)
#     ds_skew = da_obs_aligned.groupby("time.year").apply( lambda x: xr.apply_ufunc(skew,    x, input_core_dims=[["time"]], kwargs={"nan_policy": "omit"}, vectorize=True )   )
#     ds_kurt = da_obs_aligned.groupby("time.year").apply( lambda x: xr.apply_ufunc(kurtosis,x, input_core_dims=[["time"]], kwargs={"nan_policy": "omit"}, vectorize=True )   )
#     ds_moment = xr.Dataset({ "mean": ds_mean, "std": ds_std,   "skewness": ds_skew,     "kurtosis": ds_kurt  })    
#     return ds_moment

list_moment_names = ['mean','std','skewness','kurtosis']
def metric_moment_yearly(ds_obs,ds_cwrf):
    # align
    obs_ds_align,cwrf_ds_align =  xr.align(ds_obs,ds_cwrf, join = 'inner')
    list_da_rmse,list_da_acc = [],[]
    for moment_name in list_moment_names:
        # data_array
        da_obs, da_cwrf = obs_ds_align[moment_name],cwrf_ds_align[moment_name]
        # calculate  RMSE and ACC
        da_rmse = dawn_rmse(da_obs, da_cwrf, dim = 'year')
        da_acc, da_acc_cb = dawn_acc_da(da_obs, da_cwrf)
        list_da_rmse.append(da_rmse.astype(np.float32).expand_dims(moment = [moment_name]))
        list_da_acc.append(  da_acc.astype(np.float32).expand_dims(moment = [moment_name]))
    ds = xr.Dataset()
    ds['RMSE'] = xr.concat(list_da_rmse,dim='moment')
    ds['ACC']  = xr.concat(list_da_acc ,dim='moment')
    return ds


# # Calculate the moment of Observation
# da_obs      = xr.open_dataset(  f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')['PRAVG']
# pravg_moment_calculation_yearly(da_obs).to_netcdf(f'{path_moment}yearly_OBS_PRAVG_2012-2023_JJA.nc')

# # Calculate the moment of Simulation and bias-correction result.
# for comb in cwrf_combination[rank::size]:
#     init_mnth,day,ih_exp    = comb
#     # CWRF
#     filename    = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'     # Name of CWRF and CNN
#     da_cwrf     = xr.open_dataset(  f'{path_simu}{filename}')['PRAVG']
#     pravg_moment_calculation_yearly(da_cwrf).to_netcdf(f'{path_moment}yearly_{filename}')
#     # EQM
#     filename    = f'CWRF_{var_name}_EQM_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'     # Name of CWRF and CNN
#     da_eqm      = xr.open_dataset(  f'{path_bc}{filename}')['PRAVG']
#     pravg_moment_calculation_yearly(da_eqm).to_netcdf(f'{path_moment}yearly_{filename}')
#     # CQM
#     da_cqm      = xr.open_dataset(  f'{path_recover}recover_5to1day_{filename}')['PRAVG']
#     pravg_moment_calculation_yearly(da_cqm).to_netcdf(f'{path_moment}yearly_recover_5to1day_{filename}')



# # Calculate the metric of moment
# ds_obs      = US_MASK(xr.open_dataset(f'{path_moment}yearly_OBS_PRAVG_2012-2023_JJA.nc'))

# list_metric_cwrf,list_metric_eqm,list_metric_cqm = [],[],[]
# for comb   in cwrf_combination:
#     init_mnth,day,ih_exp    = comb
#     filename    = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'        
#     ds_cwrf     =  US_MASK(xr.open_dataset( f'{path_moment}yearly_{filename}'))           # CWRF
#     filename    = f'CWRF_{var_name}_EQM_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'    
#     ds_eqm      = xr.open_dataset(f'{path_moment}yearly_{filename}')                      # EQM
#     ds_cqm      = xr.open_dataset( f'{path_moment}yearly_recover_5to1day_{filename}')     # CQM
#     list_metric_cwrf.append( metric_moment_yearly(ds_obs,ds_cwrf).expand_dims(comb = [comb]) )
#     list_metric_eqm.append(  metric_moment_yearly(ds_obs,ds_eqm ).expand_dims(comb = [comb]) )
#     list_metric_cqm.append(  metric_moment_yearly(ds_obs,ds_cqm ).expand_dims(comb = [comb]) )

# ds_metric_cwrf = xr.concat(list_metric_cwrf, dim='comb' )
# ds_metric_cwrf.mean(dim='comb').to_netcdf(f'{path_moment}metric_moment_cwrf.nc')
# ds_metric_eqm  = xr.concat(list_metric_eqm , dim='comb' )
# ds_metric_eqm.mean(dim='comb').to_netcdf(f'{path_moment}metric_moment_eqm.nc')
# ds_metric_cqm  = xr.concat(list_metric_cqm , dim='comb' )
# ds_metric_cqm.mean(dim='comb').to_netcdf(f'{path_moment}metric_moment_cqm.nc')





# # ### ************ Plot figures ****************
import matplotlib.pyplot        as      plt
import seaborn                  as      sns
from   matplotlib.colors        import ListedColormap
from   base_information         import (contour_single_US_plot,raster_single_US_plot,contour_single_US_ploti_Logarithmic,contour_single_US_ploti_Logarithmic_mask,contour_panel_mRnC_US_plot3,contour_panel_mRnC_US_plot4,
cmap_normalize,cmap_grads_rainbow,contour_panel_mRnC_US_plot,contour_panel_mRnC_US_plot2,acc_after_mask)
import matplotlib
matplotlib.use('Agg')

print('Load packages for plotting')

def normalize_rgb(rgb):
    return tuple(np.array(rgb) / 255.0)

da_obs        = xr.open_dataset(  f'{path_quantile}OBS_PRAVG_2012-2023_JJA.nc')['PRAVG']
def MAPE_of_ensemble(list_cwrf):
    da_cwrf   = xr.concat(list_cwrf, dim='ensemble').mean(dim='ensemble')
    MAPE_CWRF = 100 * abs(da_cwrf - da_obs)  / da_obs
    # MAPE_CWRF  =  da_cwrf
    return(MAPE_CWRF)

def wasserstein_1d(quantiles_forecast, quantiles_obs, p_grid):
    # p_grid is the array of probabilities (e.g. np.linspace(0,1,N))
    # The integrand is abs(Q_f(p) - Q_o(p)), approximate integral via trapezoid rule
    return np.trapz(np.abs(quantiles_forecast - quantiles_obs), x=p_grid)

def show_mean_std(cwrf_xr):
    print('The mean is', cwrf_xr.mean().values,';      The Std is', cwrf_xr.std(dim='ensemble').mean().values)
print('end of function for plotting')
# # --- funcitons and settings for plotting ---
# Colors and levels
cmap_RGB_ARMSE    = [(255, 255, 255),    (171, 193, 252), (128, 168, 251),  (114, 137, 250),  ( 88,  98, 242),  (  0, 148,  39), ( 31, 181,  63), (167, 204, 106),  (255, 249,  58),  (255, 153,  42), (232,   0,  24), (121,   0,  11) ] 
cmap_armse        = ListedColormap( cmap_normalize(cmap_RGB_ARMSE) )
levels_armse      = [0.2,0.4,0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

levels_proportion = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
cmap_RGB_propor   = [(171, 193, 252),   (128, 168, 251),   ( 88,  98, 242),  (  0, 148,  39), ( 31, 181,  63), (167, 204, 106),  (255, 249,  58),(255, 153,  42),] 
cmap_propor       = ListedColormap( cmap_normalize(cmap_RGB_propor) )

cmap_RGB_freq     = [(128, 168, 251),   ( 88,  98, 242),  (  0, 148,  39), ( 31, 181,  63), (167, 204, 106),  (255, 249,  58),(255, 153,  42), (232,   0,  24),] 
cmap_freq         = ListedColormap( cmap_normalize(cmap_RGB_freq) )

cmap_RGB_freq_r   = cmap_RGB_freq[::-1]
cmap_freq_r       = ListedColormap( cmap_normalize(cmap_RGB_freq_r) )

cmap_acc          = ListedColormap( cmap_normalize( [ ( 70, 111, 235),  (112, 214, 251),  (128, 245, 253), (158, 252, 175),(199, 253, 173),  (205, 224, 191),  (209, 209, 194), (241, 241, 175),(250, 224, 107),  (242, 160,  72),  (238, 112,  60), (237,  81,  55),(154,  45,  61) ] ) )
levels_acc        = [-0.6,-0.5,-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7]
cmap_noaa_prec    = ListedColormap( cmap_normalize([ (110,  79,  70),  (162,  47,  26),  (185,  54,  37), (237,  89,  56),  (243, 175,  76),  (255, 255, 255), (255, 255, 255), (182, 244, 162), (143, 239, 119),  (102, 186,  77),  ( 90, 168,  66), ( 65, 114, 209),]) )
levels_noaa_prec  = [-1,-0.8,-0.6, -0.4,-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1]
cmap_temp         = ListedColormap( cmap_normalize( [( 17,  47,  85),  ( 28,  64, 118),  ( 53,  97, 162),( 75, 126, 185),  (122, 169, 207),  (189, 223, 232), (251, 229, 228),(236, 129, 101),  (225,  93,  74),  (195,  65,  60), (163,  54,  48),(103,  16,   8) ]) )
levels_temp       = [0.0,1.0,2.0, 3.0,4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
cmap_parvg2       = ListedColormap( cmap_normalize([(255, 255, 255),  (232, 254, 250),  (177, 220, 209), (139, 199, 175),  (112, 180, 135),  (103, 174, 162), (122, 169, 207), (103, 156, 201),  (108, 152, 199),  ( 28,  64, 118), ( 17,  47,  85), ( 86,  10, 125) ] ) )
cmap_dry          = ListedColormap( cmap_normalize( [( 95,  36, 134), ( 33,  77,  66), ( 61, 116, 111), ( 89, 161, 155), (158, 209, 200), (212, 236, 232), (246, 246, 246), (245, 235, 206), (223, 202, 147), (190, 145,  80), (145, 100,  48), ( 96,  67,  21)]) )
print('end of color')

# # ------ Fig.1 QQplot, ratio between Rx1d,Rx5d/total ------
# figfmt            = 'png'
# # 1 Plot precipitation quantiles (Q-Q)
# da_obs            = xr.open_dataset(  f'{path_quantile}OBS_PRAVG_2012-2023_JJA.nc')['PRAVG']
# da_noaa           = xr.open_mfdataset(f'{path_quantile}NOAA_PRAVG_2012-2023*.nc'                ,combine = 'nested',concat_dim='ensemble').mean(dim='ensemble')['PRAVG']
# da_cwrf           = xr.open_mfdataset(f'{path_quantile}CWRF_PRAVG_2012*.nc'                     ,combine = 'nested',concat_dim='ensemble').mean(dim='ensemble')['PRAVG']
# da_cwrf_eqm       = xr.open_mfdataset(f'{path_quantile}CWRF_PRAVG_EQM_2012*.nc'                 ,combine = 'nested',concat_dim='ensemble').mean(dim='ensemble')['PRAVG']
# da_cwrf_cnn       = xr.open_mfdataset(f'{path_quantile}recover_5to1day_CWRF_PRAVG_EQM_2012*.nc' ,combine = 'nested',concat_dim='ensemble').mean(dim='ensemble')['PRAVG']
# plt.figure(figsize=(2.268,3.675))
# plt.plot([0, 70], [0, 70], 'k-',)# label='Y = X')
# plt.xlim(0, 70),plt.ylim(0, 85)
# def nprg(da_obs):
#     return da_obs.values[0:100]
# plt.plot( nprg(da_obs), nprg(da_noaa)     ,marker='x' ,color ='green'     ,linewidth=0.2, label = 'CFSv2' ,ms=2 )
# plt.plot( nprg(da_obs), nprg(da_cwrf)     ,marker='x' ,color ='blue'      ,linewidth=0.3, label = 'CWRF'  ,ms=2 )
# plt.plot( nprg(da_obs), nprg(da_cwrf_eqm) ,marker='x' ,color ='darkorange',linewidth=0.3, label = 'EQM'   ,ms=2 )
# # plt.plot( nprg(da_obs), nprg(da_cwrf_cnn) ,marker='x' ,color ='r'         , label = 'CNN-BC')
# plt.xlabel('Observation')
# plt.ylabel('Prediction')
# plt.grid(True),
# legend = plt.legend()
# legend.get_frame().set_linewidth(0)
# plt.savefig(f'{path_figures}qqplot_proportion/precipitation_quantiles.{figfmt}', dpi=300)
# # 2 Plot the area of wetdays less than 5 day
# da_obs       = xr.open_dataset(f'{path_ind_simu}OBS.nc')['wetdays']                       # wet days greater than 5 days
# da_plot      = xr.where( da_obs.mean(dim='year') >5, 0.8, 2.2)
# # raster_single_US_plot(da_plot,cmap_armse,levels_armse).savefig(f'{path_figures}qqplot_proportion/distribution_of_wetdays_obs.{figfmt}')
# # 3 Contour plot, the ratio between Rx1d, Rx5d and Total precipitation
# levels_exp   = [0.1,0.2,0.4,0.8]
# cmap_RGB_exp = [ (171, 193, 252),   ( 131,  189, 178),  (  63, 121, 155), ( 31, 181,  63),  (219, 219, 192),(251, 232, 231),]
# cmap_exp     = ListedColormap( cmap_normalize(cmap_RGB_exp) )
# da_rx1d      = xr.open_dataset(f'{path_ind_simu}OBS.nc')['Rx1day'].mean(dim='year')       # ratop between Rx1day to total precipitation.
# da_rx5d      = xr.open_dataset(f'{path_ind_simu}OBS.nc')['Rx5day'].mean(dim='year')       # ratop between Rx5day to total precipitation.
# da_total_all = xr.open_dataset(  f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')[var_name].resample(time='YE').sum()
# da_total     = da_total_all.mean(dim='time')
# proportion1d = da_rx1d / da_total#,    ic(da_rx1d.sum()/da_total.sum())
# proportion5d = da_rx5d / da_total#,    ic(da_rx5d.sum()/da_total.sum())
# contour_single_US_ploti_Logarithmic(proportion1d,cmap_exp,levels_exp).savefig(f'{path_figures}qqplot_proportion/rx1d_total_proportion.{figfmt}')
# contour_single_US_ploti_Logarithmic_mask(    proportion5d, cmap_exp, levels_exp, mask_array=da_plot, mask_threshold=1).savefig(
#     f'{path_figures}qqplot_proportion/rx5d_total_proportion.{figfmt}')

# print("***Figure 1 is Done!")

# # ------ Figure 2 For Rxnd Ploting (contour and KDE) ------

# figfmt     = 'png'
# ratio_avg  = {'Rx1day':9,'Rx5day':16}
# ratio_rmse = {'Rx1day':4,'Rx5day':6}
# for cli_index in cli_indices:
#     path_cnn    = f'{path_project}CNN_{cli_index}/'
#     da_obs      = xr.open_dataset(f'{path_ind_simu}OBS.nc')[cli_index]
#     list_cwrf,list_bc, list_cnn, list_rmse_cwrf,list_rmse_bc,list_rmse_cnn = [],[],[],[],[],[]
#     for comb in cwrf_combination:
#         init_mnth,day,ih_exp    = comb
#         filename    = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'     # Name of CWRF and CNN
#         da_cwrf     = xr.open_mfdataset(f'{path_ind_simu}{filename}')[cli_index]
#         da_bc       = xr.open_mfdataset(f'{path_ind_bc}CWRF_{var_name}_EQM_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc')[cli_index]
#         da_cnn      = xr.open_mfdataset(f'{path_cnn}{filename}')[cli_index]
#         obs_align, cwrf_align   = xr.align(da_obs,da_cwrf, join = 'inner') 
#         rmse_cwrf   = dawn_rmse(  US_MASK(cwrf_align), US_MASK(obs_align),dim='year')
#         obs_align, bc_align     = xr.align(da_obs,da_bc, join = 'inner')
#         rmse_bc     = dawn_rmse(  US_MASK(bc_align), US_MASK(obs_align),dim='year')
#         obs_align, cnn_align    = xr.align(da_obs,da_cnn , join = 'inner')
#         rmse_cnn    = dawn_rmse(  US_MASK(cnn_align) , US_MASK(obs_align),dim='year')
#         list_cwrf.append(da_cwrf),list_bc.append(da_bc),list_cnn.append(da_cnn),list_rmse_cwrf.append(rmse_cwrf),list_rmse_bc.append(rmse_bc),list_rmse_cnn.append(rmse_cnn)
#     da_cwrf_avg     = xr.concat(list_cwrf     ,dim='member').mean(dim='member')
#     da_bc_avg       = xr.concat(list_bc       ,dim='member').mean(dim='member')
#     da_cnn_avg      = xr.concat(list_cnn      ,dim='member').mean(dim='member')
#     da_cwrf_rmse    = xr.concat(list_rmse_cwrf,dim='member').mean(dim='member')
#     da_bc_rmse      = xr.concat(list_rmse_bc  ,dim='member').mean(dim='member')
#     da_cnn_rmse     = xr.concat(list_rmse_cnn ,dim='member').mean(dim='member')
#     # 1 Contour plot of Rxd avg
#     levels = [x*ratio_avg[cli_index]  for x in range(1,8)]
#     cmap_rxnd_list  = [(233,137,66),(253,254,207),(209, 234, 247),  (176, 219, 245), (132, 184, 227), ( 90, 140, 198),   (31,  44,  94) ]
#     cmap_interior_colors = [normalize_rgb(c) for c in cmap_rxnd_list[1:-1]]
#     cmap_rxnd = ListedColormap(cmap_interior_colors)
#     cmap_rxnd.set_under(normalize_rgb(cmap_rxnd_list[0]))  # for values <10
#     cmap_rxnd.set_over(normalize_rgb(cmap_rxnd_list[-1]))    # for values >70
#     list_rxnd_avg = [ US_MASK(da_obs.mean(dim='year')), US_MASK(da_cwrf_avg.mean(dim='year')),US_MASK(da_bc_avg.mean(dim='year')), US_MASK(da_cnn_avg.mean( dim='year'))]
#     contour_panel_mRnC_US_plot( 1,4,12,2,list_rxnd_avg, cmap_rxnd, levels).savefig(f'{path_figures}Rxnd/{cli_index}_avg_panel_contour_plot.{figfmt}')
#     contour_panel_mRnC_US_plot4(4,1, 3,8,list_rxnd_avg, cmap_rxnd, levels).savefig(f'{path_figures}Rxnd/vertical_{cli_index}_avg_panel_contour_plot.{figfmt}')
#     # # 2 Contour plot of Rxd RMSE
#     levels = [x*ratio_rmse[cli_index] for x in levels_temp]      # 4,6
#     list_rxnd_rmse = [  US_MASK(da_cwrf_rmse), US_MASK(da_bc_rmse), US_MASK(da_cnn_rmse)]
#     contour_panel_mRnC_US_plot( 1,3,9,2,list_rxnd_rmse, cmap_armse, levels).savefig(f'{path_figures}Rxnd/{cli_index}_rmse_panel_contour_plot.{figfmt}')
#     contour_panel_mRnC_US_plot2(3,1,3,6,list_rxnd_rmse, cmap_armse, levels).savefig(f'{path_figures}Rxnd/vertical_{cli_index}_rmse_panel_contour_plot.{figfmt}')
#     # 3 KDE plot
#     list_rxnd_rmse = [  US_MASK(da_cwrf_rmse), US_MASK(da_bc_rmse), US_MASK(da_cnn_rmse)]
#     list_rmse = [x.values.flatten() for x in list_rxnd_rmse]
#     plt.figure(figsize=(2.99, 1.9182))
#     colors     = ['blue', 'darkorange', 'r']
#     sns.kdeplot(list_rmse[0], label="CWRF", color = colors[0], fill=False)
#     sns.kdeplot(list_rmse[1], label="EQM" , color = colors[1], fill=False)
#     sns.kdeplot(list_rmse[2], label="CNN" , color = colors[2], fill=False)
#     plt.xlabel('RMSE')
#     plt.ylabel('Density')
#     # plt.title('KDE Distribution of RMSE')
#     plt.legend()
#     plt.savefig(f'{path_figures}Rxnd/{cli_index}_KDE_rmse_plot.{figfmt}')

# print("***Figure 2 is Done!")



# # ------Figure 3. ACC of Rx1d, Rx5d ------
# figfmt           = 'eps'
# # ACC of Rx1d and Rx5d
# for cli_index in cli_indices:
#     da_obs       = xr.open_dataset(  f'{path_ind_simu}OBS.nc')[cli_index]    # Read the observation
#     # Calculate the ACC, convert to xarray, and store in a list.
#     list_cwrf,list_cnn,list_bc  = [],[],[]
#     for comb in cwrf_combination:
#         init_mnth,day,ih_exp    = comb
#         filename    = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'     # Name of CWRF and CNN
#         bcfilename  = f'CWRF_{var_name}_EQM_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc' # Name of BC
#         path_cnn    = f'{path_project}CNN_{cli_index}/'
#         da_cwrf     = xr.open_dataset(f'{path_ind_simu}{filename}')[cli_index]
#         da_cnn      = xr.open_dataset(f'{path_cnn     }{filename}')[cli_index]
#         da_bc       = xr.open_dataset(f'{path_ind_bc}{bcfilename}')[cli_index]
#         obs_align, cwrf_align   = xr.align(da_obs,da_cwrf, join = 'inner')                                   # Align
#         obs_align, cnn_align    = xr.align(da_obs,da_cnn , join = 'inner')
#         obs_align, bc_align     = xr.align(da_obs,da_bc  , join = 'inner')
#         acc_cwrf,acc_cnn,acc_bc = [],[],[]                                                                   # Start ACC calculation
#         for year in cwrf_align['year'].values:
#             acc_cwrf.append(acc_after_mask(obs_align.sel( year = year),cwrf_align.sel(year = year),da_US ))  # Calculate ACC
#             acc_cnn.append( acc_after_mask(obs_align.sel( year = year),cnn_align.sel( year = year),da_US ))
#             acc_bc.append(  acc_after_mask(obs_align.sel( year = year),bc_align.sel(  year = year),da_US ))
#         da_acc_cnn  = xr.DataArray(acc_cnn , dims=["year"], coords={"year":obs_align['year'].values })       # Convert to xr.dataarray 
#         da_acc_cwrf = xr.DataArray(acc_cwrf, dims=["year"], coords={"year":obs_align['year'].values })
#         da_acc_bc   = xr.DataArray(acc_bc  , dims=["year"], coords={"year":obs_align['year'].values })
#         list_cnn.append(  da_acc_cnn.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))                # Append to the corresponding list.
#         list_cwrf.append(da_acc_cwrf.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))
#         list_bc.append(    da_acc_bc.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))
    
#     plt.figure(figsize=(4.5,2.3))                                                                            # xy plot
#     x_shared = xr.concat(list_cnn, dim='ensemble').mean(dim='ensemble')['year'].values                       # 0) Create the shared x-axis from the concatenated+mean approach:
#     for i, da in enumerate(list_cnn):               # AI                                                     # 1) Plot the Lines of each ensemble member.
#         y_list  = [da.sel(year=da['year'] == x).values[0][0] if da.sel(year=da['year'] == x).values.size > 0 else np.nan for x in x_shared]
#         plt.plot(x_shared, y_list, marker='.', color = 'pink',linewidth=0.6,markersize=5)
#     for i, da in enumerate(list_cwrf):              # CWRF
#         y_list  = [da.sel(year=da['year'] == x).values[0][0] if da.sel(year=da['year'] == x).values.size > 0 else np.nan for x in x_shared]
#         plt.plot(x_shared, y_list, marker='.', color = 'cornflowerblue',linewidth=0.6,markersize=5)
#     for i, da in enumerate(list_bc):                # Bias-correction
#         y_list  = [da.sel(year=da['year'] == x).values[0][0] if da.sel(year=da['year'] == x).values.size > 0 else np.nan for x in x_shared]
#         plt.plot(x_shared, y_list, marker='.', color = 'bisque',linewidth=0.6,markersize=5)
#     plt.plot(years, avg_list(list_cwrf).values, label='CWRF',   marker='h', linestyle='-',  linewidth=1.0, color='blue') # 2)Plot the Lines of each ensemble mean.
#     plt.plot(years, avg_list(list_cnn ).values, label='CNN' ,   marker='h', linestyle='-',  linewidth=1.0, color='r'   )
#     plt.plot(years, avg_list(list_bc  ).values, label='EQM' ,   marker='h', linestyle='-',  linewidth=1.0, color='darkorange'   )
#     plt.xlabel("Year"),    plt.ylabel(f"ACC ( {cli_index} )"),    plt.savefig(f'{path_figures}ACC/xy_acc_{cli_index}.{figfmt}') 

#     # box plot
#     cwrf_array = xr.concat(list_cwrf ,dim='ensemble').values.flatten()
#     bc_array   = xr.concat(list_bc   ,dim='ensemble').values.flatten()
#     cnn_array  = xr.concat(list_cnn  ,dim='ensemble').values.flatten()
#     cwrf_array = cwrf_array[~np.isnan(cwrf_array)]
#     bc_array   = bc_array[  ~np.isnan(bc_array)]
#     cnn_array  = cnn_array[ ~np.isnan(cnn_array)]
#     data       = [ cwrf_array, bc_array, cnn_array]
#     colors     = ['blue', 'darkorange', 'r']
#     plt.figure(figsize=(1.8, 2.3))
#     box        = plt.boxplot(data,patch_artist=True, showfliers=False)
#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in box['medians']:
#         median.set_color('black')
#     plt.xticks([1, 2, 3], [ 'CWRF', 'EQM', 'CNN'])
#     # plt.xlabel('Dataset'), plt.ylabel('ACC')
#     plt.savefig(f'{path_figures}ACC/box_ACC_{cli_index}.{figfmt}', dpi=300, bbox_inches='tight')

# print("***Figure 3 is Done!")

# # ------Fig4. Total precipitation ------
# figfmt     = 'eps'
# metric     = 'RMSE'
# print('start_calculation')
# list_da    = get_list_simulation_metric(metric)
# print(list_da)
# contour_panel_mRnC_US_plot(1,3,6.3,1.4,list_da, cmap_armse,levels_armse).savefig(
#     f'{path_figures}total_prec/contour_plot_{metric}_PRAVG.{figfmt}', dpi=200,bbox_inches='tight'),plt.close(),print(f'{path_figures}Average_precipitation/contour_plot_{metric}_PRAVG.png have been ploted')

# metric     = 'bias'
# levels     = [x*0.5 for x in levels_acc]
# list_da    = get_list_simulation_metric(metric)
# list_da_mean = [x.mean(dim='time') for x in list_da]
# contour_panel_mRnC_US_plot(1,3,6.3,1.4,list_da_mean, cmap_noaa_prec,levels_noaa_prec).savefig(
#     f'{path_figures}total_prec/contour_plot_{metric}_PRAVG.{figfmt}', dpi=200,bbox_inches='tight'),plt.close(),print(f'{path_figures}Average_precipitation/contour_plot_{metric}_PRAVG.png have been ploted')

# metric     = 'IAC'
# levels     = [x*0.5 for x in levels_acc]   #If have 5 columns, then contour_panel_mRnC_US_plot2(5,1,2,6,list_da, cmap_armse,levels_armse)
# list_da    = get_list_simulation_metric(metric)

# cmap_acc_p_list =  [ ( 70, 111, 235),  (128, 245, 253), (199, 253, 173),  (243, 243,243),(243, 243,243 ),(250, 224, 107),   (238, 112,  60),(154,  45,  61) ]
# cmap_interior_colors = [normalize_rgb(c) for c in cmap_acc_p_list[1:-1]]
# cmap_acc_p      = ListedColormap(cmap_interior_colors)
# cmap_acc_p.set_under(normalize_rgb(cmap_acc_p_list[0])) 
# cmap_acc_p.set_over(normalize_rgb(cmap_acc_p_list[-1])) 
# levels_acc      = [ -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
# contour_panel_mRnC_US_plot2(3,1,2.53,3.795,list_da, cmap_acc_p,levels_acc).savefig(
#     f'{path_figures}total_prec/contour_plot_{metric}_PRAVG.{figfmt}', dpi=200,bbox_inches='tight'),plt.close(),print(f'{path_figures}Average_precipitation/contour_plot_{metric}_PRAVG.png have been ploted')

# # ACC of total preciptation
# da_obs          = xr.open_dataset(  f'{path_simu}OBS_PRAVG_2012-2023_JJA.nc')[var_name].resample(time='YE').sum()
# list_cwrf,list_cnn,list_bc  = [],[],[]
# for comb in cwrf_combination:
#     init_mnth,day,ih_exp    = comb
#     filename    = f'CWRF_{var_name}_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc'     # Name of CWRF and CNN
#     bcfilename  = f'CWRF_{var_name}_EQM_{years[0]}-{years[-1]}_mnth-{init_mnth:02}_day-{day:02}_{pred_season_str}_{ih_exp}.nc' # Name of BC
#     da_cwrf     = xr.open_dataset(f'{path_simu   }{  filename}')[var_name].resample(time='YE').sum()
#     da_cnn      = xr.open_dataset(f'{path_recover}recover_5to1day_{bcfilename}')[var_name].resample(time='YE').sum()
#     da_bc       = xr.open_dataset(f'{path_bc     }{bcfilename}')[var_name].resample(time='YE').sum()
#     obs_align, cwrf_align   = xr.align(da_obs,da_cwrf, join = 'inner')                                   # Align
#     obs_align, cnn_align    = xr.align(da_obs,da_cnn , join = 'inner')
#     obs_align, bc_align     = xr.align(da_obs,da_bc  , join = 'inner')
#     acc_cwrf,acc_cnn,acc_bc = [],[],[]                                                                   # Start ACC calculation
#     for time in cwrf_align['time'].values:
#         acc_cwrf.append(acc_after_mask(obs_align.sel( time = time),cwrf_align.sel(time = time),da_US ))  # Calculate ACC
#         acc_cnn.append( acc_after_mask(obs_align.sel( time = time),cnn_align.sel( time = time),da_US ))
#         acc_bc.append(  acc_after_mask(obs_align.sel( time = time),bc_align.sel(  time = time),da_US ))
#     da_acc_cnn  = xr.DataArray(acc_cnn , dims=["time"], coords={"time":obs_align['time'].values })       # Convert to xr.dataarray 
#     da_acc_cwrf = xr.DataArray(acc_cwrf, dims=["time"], coords={"time":obs_align['time'].values })
#     da_acc_bc   = xr.DataArray(acc_bc  , dims=["time"], coords={"time":obs_align['time'].values })
#     list_cnn.append(  da_acc_cnn.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))                # Append to the corresponding list.
#     list_cwrf.append(da_acc_cwrf.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))
#     list_bc.append(    da_acc_bc.expand_dims(ensemble = [f'{init_mnth}_{day}_{ih_exp}']))
# plt.figure(figsize=(4.5,2.3))
# x_shared = xr.concat(list_cnn, dim='ensemble').mean(dim='ensemble')['time'].dt.year.values               # 0) Create the shared x-axis from the concatenated+mean approach:
# for i, da in enumerate(list_cnn):               # AI                                                     # 1) Plot the Lines of each ensemble member.
#     y_list  = [da.sel(time=da.time.dt.year == x).values[0][0] if da.sel(time=da.time.dt.year == x).values.size > 0 else np.nan for x in x_shared]
#     plt.plot(x_shared, y_list, marker='.', color = 'pink',linewidth=0.6,markersize=5)
# for i, da in enumerate(list_cwrf):              # CWRF
#     y_list  = [da.sel(time=da.time.dt.year == x).values[0][0] if da.sel(time=da.time.dt.year == x).values.size > 0 else np.nan for x in x_shared]
#     plt.plot(x_shared, y_list, marker='.', color = 'cornflowerblue',linewidth=0.6,markersize=5)
# for i, da in enumerate(list_bc):                # Bias-correction
#     y_list  = [da.sel(time=da.time.dt.year == x).values[0][0] if da.sel(time=da.time.dt.year == x).values.size > 0 else np.nan for x in x_shared]
#     plt.plot(x_shared, y_list, marker='.', color = 'bisque',linewidth=0.6,markersize=5)
# plt.plot(years, avg_list(list_cwrf).values, label='CWRF',      marker='h', linestyle='-',  linewidth=1.0, color='blue') # 2)Plot the Lines of each ensemble mean.
# plt.plot(years, avg_list(list_cnn ).values, label='CNN_BC'  ,  marker='h', linestyle='-',  linewidth=1.0, color='r'   )
# plt.plot(years, avg_list(list_bc  ).values, label='EQM'    ,   marker='h', linestyle='-',  linewidth=1.0, color='darkorange'   )
# plt.xlabel("Year"),    plt.ylabel(f"ACC (total)"),    plt.savefig(f'{path_figures}total_prec/xy_acc_total_precipitation.{figfmt}') 

# cwrf_xr = xr.concat(list_cwrf,dim='ensemble')
# bc_xr   = xr.concat(list_bc  ,dim='ensemble')
# cnn_xr  = xr.concat(list_cnn  ,dim='ensemble')
# show_mean_std(cwrf_xr)
# show_mean_std(bc_xr)
# show_mean_std(cnn_xr)
# print('ACC of total precipitation predicted by CWRF, BC, CNN')
# cwrf_array = cwrf_xr.values.flatten()
# bc_array   = bc_xr.values.flatten()
# cnn_array  = cnn_xr.values.flatten()

# cwrf_array = cwrf_array[~np.isnan(cwrf_array)]
# bc_array   = bc_array[  ~np.isnan(bc_array)]
# cnn_array  = cnn_array[ ~np.isnan(cnn_array)]
# data       = [ cwrf_array, bc_array, cnn_array]
# colors     = ['blue', 'darkorange', 'r']
# plt.figure(figsize=(1.8, 2.3))
# box        = plt.boxplot(data,patch_artist=True, showfliers=False)
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# for median in box['medians']:
#     median.set_color('black')
# plt.xticks([1, 2, 3], [ 'CWRF', 'EQM', 'CQM'])
# # plt.xlabel('Dataset'), plt.ylabel('ACC')
# plt.savefig(f'{path_figures}total_prec/box_ACC_total.{figfmt}', dpi=300, bbox_inches='tight')

# print("***Figure 4 is Done!")


# # ------ Fig.5 For precipitation quantiles  and extreme indices ------
# figfmt        = 'eps'
# #1 MAPE plot
# list_cwrf,list_eqm,list_both,list_rx1d,list_rx5d=[],[],[],[],[] # Read the data
# for comb in noaa_combination:
#     init_mnth,day,init_hour = comb
#     # list_noaa.append(xr.open_dataset(f'{path_quantile}NOAA_PRAVG_2012-2023_mnth-{init_mnth:02}_day-{day:02}_hour-{init_hour:02}_JJA.nc')['PRAVG'])
#     list_cwrf.append(xr.open_dataset(f'{path_quantile}CWRF_PRAVG_2012-2023_mnth-{init_mnth:02}_day-{day:02}_JJA_{init_hour:02}_icbc01_exp00.nc')['PRAVG'])
#     list_eqm.append( xr.open_dataset(f'{path_quantile}CWRF_PRAVG_EQM_2012-2023_mnth-{init_mnth:02}_day-{day:02}_JJA_{init_hour:02}_icbc01_exp00.nc')['PRAVG'])
#     list_both.append( xr.open_dataset(f'{path_quantile}recover_5to1day_CWRF_PRAVG_EQM_2012-2023_mnth-{init_mnth:02}_day-{day:02}_JJA_{init_hour:02}_icbc01_exp00.nc')['PRAVG'])
#     list_rx1d.append( xr.open_dataset(f'{path_quantile}recover_1day_CWRF_PRAVG_EQM_2012-2023_mnth-{init_mnth:02}_day-{day:02}_JJA_{init_hour:02}_icbc01_exp00.nc')['PRAVG'])
#     list_rx5d.append( xr.open_dataset(f'{path_quantile}recover_5day_CWRF_PRAVG_EQM_2012-2023_mnth-{init_mnth:02}_day-{day:02}_JJA_{init_hour:02}_icbc01_exp00.nc')['PRAVG'])
# plt.figure(figsize=(3.5, 2.5))  # Plot the MAPE xy line
# x_varialbe = range(1,100)
# plt.plot(x_varialbe, MAPE_of_ensemble(list_cwrf), label='CWRF'    ,color = 'blue')
# plt.plot(x_varialbe, MAPE_of_ensemble(list_eqm),  label='EQM'     ,color = 'darkorange')
# plt.plot(x_varialbe, MAPE_of_ensemble(list_both), label='Recover' ,color = 'r')
# plt.xlabel('Percentile'), plt.ylabel('MAPE Values') ,plt.legend() # plt.grid(True)
# plt.savefig(f'{path_figures}extreme/xy_MAPE.{figfmt}', dpi=300, bbox_inches='tight')

# # 2 Bar-Plot
# list_cwrf_wd,list_eqm_wd,list_both_wd,list_rx1d_wd,list_rx5d_wd=[],[],[],[],[]
# for i in range(len(list_cwrf)):
#     p_grid  = np.linspace(0, 1, len(da_obs.values))
#     # WD_NOAA = wasserstein_1d(list_noaa[i].values, da_obs.values, p_grid)
#     WD_CWRF = wasserstein_1d(list_cwrf[i].values, da_obs.values, p_grid)
#     WD_EQM  = wasserstein_1d(list_eqm[ i].values, da_obs.values, p_grid)
#     WD_Both = wasserstein_1d(list_both[i].values, da_obs.values, p_grid)
#     WD_rx1d = wasserstein_1d(list_rx1d[i].values, da_obs.values, p_grid)
#     WD_rx5d = wasserstein_1d(list_rx5d[i].values, da_obs.values, p_grid)
#     list_cwrf_wd.append(WD_CWRF),list_eqm_wd.append(WD_EQM),list_both_wd.append(WD_Both),list_rx1d_wd.append(WD_rx1d),list_rx5d_wd.append(WD_rx5d)
# data = [ list_cwrf_wd, list_eqm_wd, list_both_wd, list_rx1d_wd, list_rx5d_wd]
# plt.figure(figsize=(2.5, 2.5))
# plt.boxplot(data, showfliers=False)
# plt.xticks([1, 2, 3, 4, 5], [ 'CWRF', 'EQM', 'Eb', 'E1d', 'E5d'])
# plt.xlabel('Predictions'), plt.ylabel('Wasserstein distance')  #plt.title('Box Plot of Wasserstein distance')
# plt.savefig(f'{path_figures}extreme/box_Wasserstein_distance.{figfmt}', dpi=300, bbox_inches='tight')

# # # 3 heatmap, rmse of indices ---
# figfmt           = 'eps'
# metric           = 'RMSE'
# list_ind_metric  = get_ind_simulation_metric(metric)
# list_us_avg_all  = [ US_MASK(x).mean(dim = ['south_north','west_east'])  for x in list_ind_metric]
# new_order = ['Rx1day','Rx5day','SDII','R10mm','R20mm','CDD','CWD','R95p','R99p','PRCPTOT','wetdays','JJATOTAL']
# list_us_avg = [x.reset_coords('time',drop=True).reindex(ind=new_order).values[0:10] for x in list_us_avg_all ]
# relative_rmse_us =(list_us_avg[1::]-list_us_avg[0])*100.0 / list_us_avg[0]

# y_tick_labels    = ['EQM','DQM','QDM','CQM']
# x_tick_labels    = [ "Rx1d", "Rx5d","SDII", "R10","R20", "CDD", "CWD", "R95p", "R99p", "PRCPTOT"]
# plt.figure(figsize=(8, 3)),sns.set(font_scale=0.85)
# ax               = sns.heatmap(relative_rmse_us, annot=True, cmap='coolwarm', center=0,linewidths=2, linecolor='white')
# ax.set_xticklabels(x_tick_labels,rotation=90),ax.set_yticklabels(y_tick_labels)
# plt.savefig( f'{path_figures}extreme/Heatmap_{metric}_indices_US.{figfmt}', dpi=200,bbox_inches='tight'),plt.close(),print(f'{path_figures}indices/Heatmap_{metric}_indices_US.png have been ploted')

# print("***Figure 5 is Done!")


# # ------ Fig.6 For moment ------
ds_cwrf = xr.open_dataset(f'{path_moment}metric_moment_cwrf.nc')
ds_eqm  = xr.open_dataset(f'{path_moment}metric_moment_eqm.nc')
ds_cqm  = xr.open_dataset(f'{path_moment}metric_moment_cqm.nc')

cmap_armse        = ListedColormap( cmap_normalize(cmap_RGB_ARMSE) )
levels_armse      = [0.2,0.4,0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
lelve_weights     = [1,3,1,15]
figfmt = 'png'
for i, moment_name in enumerate( list_moment_names):
    data_list   = [ds_cwrf['RMSE'].sel(moment = moment_name), ds_eqm['RMSE'].sel(moment = moment_name),ds_cqm['RMSE'].sel(moment = moment_name)]
    # levels      = [x * lelve_weights[i] for x in levels_armse]
    # contour_panel_mRnC_US_plot(1,3,6.3,1.4,data_list, cmap_armse,levels).savefig(
    #     f'{path_figures}moment/contour_plot_RMSE_{moment_name}_PRAVG.{figfmt}', dpi=200,bbox_inches='tight'),plt.close()

ic(data_list)