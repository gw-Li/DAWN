import sys,os
import numpy     as np
import pandas    as pd
import xarray    as xr
from scipy.stats import gaussian_kde
from icecream    import ic
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process climate data.')
parser.add_argument('--path_operational', type=str, required=True, help='Path to operational data')
parser.add_argument('--vname', type=str, required=True, help='Variable name')
parser.add_argument('--init_year', type=int, required=True, help='Initialization year')
parser.add_argument('--init_month', type=int, required=True, help='Initialization month')
parser.add_argument('--init_day', type=int, required=True, help='Initialization day')
args = parser.parse_args()

# Assign variables from arguments
path_operational = args.path_operational
vname            = args.vname
init_year        = args.init_year
init_month       = args.init_month
init_day         = args.init_day
ic(init_year,init_month,init_day)
# **** Base information ***
path_obs_daily = '/mnt/gfs01/PUB/OBS/regrid_daily/'
path_viewer    = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/Data_Viewer/'
# path_operational = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_hindcast  = '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
path_static    = '/mnt/gfs01/PUB/OBS/script/static_data/'
da_US_MASK     = xr.open_dataset(f'{path_static}/US_MASK_logic.nc')['MASK']
years_hist     = range(1981,2023+1)
years          = range(2012,2023+1)
months         = range(1,12+1)
vnames         = ['T2MAX','T2MIN','PRAVG']
hour_exps      = ['00_icbc01_exp00','06_icbc01_exp00','06_icbc01_exp02']
days_per_month = { 1: [ 1, 6,11,16,21,26,31], 2: [ 5,10,15,20,25],3: [ 2, 7,12,17,22,27],4: [ 1, 6,11,16,21,26],5: [ 1, 6,11,16,21,26,31],6: [ 5,10,15,20,25,30],7: [ 5,10,15,20,25,30],8: [ 4, 9,14,19,24,29],9: [ 3, 8,13,18,23,28],10:[ 3, 8,13,18,23,28],11:[ 2, 7,12,17,22,27],12:[ 2, 7,12,17,22,27]}
shifts         = [        (0, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1),        (0, 1), (1, 1), (1, 0), (1, -1)    ]
ih_exps        = ['00_icbc01_exp00','06_icbc01_exp00','06_icbc01_exp02']

# **** Functions ***
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

def get_grid_pdf_values(my_data):
    if np.isnan(my_data).any():
        x_grid, pdf_values = np.full(100, np.nan), np.full(100, np.nan)
    else:
        kde = gaussian_kde(my_data)
        x_min, x_max = np.min(my_data), np.max(my_data)
        x_grid = np.linspace(x_min, x_max, 100)
        pdf_values = kde(x_grid)
    return np.array([x_grid, pdf_values])


# Get the list of file name in ensemble.
combination    = [[year,month,day] for year in range(2012, 2050) for month in months for day in days_per_month[month]   ]
index_of_init  = combination.index([init_year, init_month,init_day ])
list_file_name = []
k              = index_of_init
count          = 0
while count    < 6:
    year, month, day = combination[k]
    file_name1  = f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}{ih_exps[0]}_{vname}_daily.nc'
    file_name2  = f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}{ih_exps[1]}_{vname}_daily.nc'
    file_name3  = f'{path_operational}{year}{month:02}{day:02}/{year}{month:02}{day:02}{ih_exps[2]}_{vname}_daily.nc'
    if os.path.exists(file_name1):
        list_file_name.append(file_name1)
        list_file_name.append(file_name2)
        list_file_name.append(file_name3)
        count += 1
    k         -= 1
# Iterate over the list in reverse order to avoid index issues while removing elements
for file in list_file_name[::-1]:
    if not os.path.exists(file):
        list_file_name.remove(file)
# Read the CWRF data,
ds_cwrf_daily    = xr.open_mfdataset(list_file_name,combine='nested',concat_dim = 'ensemble')
if vname != 'PRAVG':
    ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').mean()
    ds_cwrf_mnth_f  = (ds_cwrf_mnth[vname]- 273.15 )  * 9/5   + 32 
    ds_cwrf_mnth_f.attrs['units'] = 'Fahrenheit'
else:
    ds_cwrf_mnth    = ds_cwrf_daily.resample(time='ME').sum()
    ds_cwrf_mnth_f  = ds_cwrf_mnth[vname] * 0.0393701
    ds_cwrf_mnth_f.attrs['units'] = 'inches per month'


# Find the prediction month
ens_months = ds_cwrf_mnth_f['time'].dt.month.values
init_month_index = np.where(ens_months == init_month)[0][0]
month_subset = ens_months[init_month_index + 1:init_month_index + 7]
# Calculate PDF month by month.
list_data_vw = []
for month  in month_subset:
    # Select the target month
    month_data     = ds_cwrf_mnth_f.sel(time = ds_cwrf_mnth_f['time'].dt.month == month).chunk({'time': -1})
    # Grid search
    shifted_dataarrays = []
    for dx, dy in shifts:
        shifted    = month_data.shift(south_north=dy, west_east=dx, fill_value=np.nan)
        shifted_dataarrays.append(shifted)
    searched_data  = xr.concat(shifted_dataarrays,dim='searching')
    # Flatten 'searching' and 'ens] dimension into 'se', calculate the maximum and minimum
    flattened_da   = searched_data.stack(st=("searching", "ensemble"))#.transpose("south_north", "west_east", "st")
    da_max  = flattened_da.max(dim='st')
    da_min  = flattened_da.min(dim='st')
    # Calculate PDF
    flattened_da_dask = flattened_da.chunk({'south_north': 10, 'west_east': 10, 'st': -1})
    result            = xr.apply_ufunc(
        get_grid_pdf_values,
        flattened_da_dask,
        input_core_dims   =[['st']],
        output_core_dims  =[['output', 'linspace']],
        vectorize         = True,
        dask              = 'parallelized',
        output_dtypes     = [np.float64],
        dask_gufunc_kwargs={'output_sizes': {'output': 2, 'linspace': 100}}
    )
    x_grid_da     = result.isel(output=0)
    pdf_values_da = result.isel(output=1)
    # # dataArray to Dataset
    ds_data_vw               = xr.Dataset()
    ds_data_vw['da_maximum'] = da_max
    ds_data_vw['da_minimum'] = da_min
    ds_data_vw['x_grid']     = x_grid_da
    ds_data_vw['pdf']        = pdf_values_da
    ds_data_vw = ds_data_vw.where((ds_data_vw >= 0) & (ds_data_vw <= 10000), np.nan)
    list_data_vw.append(ds_data_vw)

ds_data_vw_all = xr.concat(list_data_vw,dim='time')
ds_data_vw_all.to_netcdf( f'{path_operational}{init_year}{init_month:02}{init_day:02}/{init_year}{init_month:02}{init_day:02}_{vname}_PDF.nc')
