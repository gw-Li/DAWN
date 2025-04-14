import cdsapi
import sys,os
import datetime 
import calendar
import cfgrib
import pandas   as     pd
import xarray   as     xr
import xesmf    as     xe

# Modify the path if neecessary 
path_raw            = './'
path_raw_nc         = './'
path_regrid_daily   = './'
path_regrid_monthly = './'

# Information
path_static         = './'
origin              = 'ecmwf'
wtg_file            = f'{path_static}ecmwf_to_CWRF_weights_file.nc'
ds_cwrf             = xr.open_dataset(f'{path_static}geo_em.d01_30.nc')
list_lead_hours     = [str(i) for i in range(24, 5160+1,24)]
years               = range(1991,2025)
months              = range(1,12+1)
dataset             = "seasonal-original-single-levels"

# Define functions.
def download_from_cds(year,month):
	request = {
	    "originating_centre": "ecmwf",
	    "system": "51",
	    "variable": [
	        "maximum_2m_temperature_in_the_last_24_hours",
	        "minimum_2m_temperature_in_the_last_24_hours",
	        "total_precipitation"
	    ],
	    "year": [f"{year}"],
	    "month": [f"{month}"],
	    "day": ["01"],
	    "leadtime_hour":list_lead_hours,
	    "data_format": "grib"
	}
	target = f'ecmwf_T2MAX_T2MIN_PRAVG_year-{year}_month-{month:02}.grib'
	client = cdsapi.Client()
	client.retrieve(dataset, request,target)

def ungrib_interpolate_resamp(year,month):
	
	# 1st, ungrid the files to netcdf format.
	file_name = f'ecmwf_T2MAX_T2MIN_PRAVG_year-{year}_month-{month:02}'
	ds        = xr.open_dataset(f'{path_raw}{file_name}.grib', engine='cfgrib')
	ds = ds.drop_vars('time') 
	ds = ds.swap_dims({'step': 'valid_time'})
	ds = ds.rename({'valid_time': 'time'})
	ds = ds.drop_vars('step')
	ds = ds.drop_vars('surface')
	# ds.to_netcdf(f'{path_raw_nc}{file_name}.nc') 
	
	# 2nd, interpolate to CWRF grid.
	ds_in           = ds
	del(ds)
	ds_out          = xr.Dataset({'lat': ds_cwrf['XLAT_M'].isel(Time=0), 'lon': ds_cwrf['XLONG_M'].isel(Time=0)})
	regridder       = xe.Regridder(ds_in, ds_out, method='bilinear', filename=wtg_file, reuse_weights=True)
	ds_in_regridded = regridder(ds_in)
	# Copy the attribute
	ds_in_regridded.attrs          =ds_in.attrs
	# ds_in_regridded['tp'].attrs    =ds_in['tp'].attrs
	# ds_in_regridded['mx2t24'].attrs=ds_in['mx2t24'].attrs
	# ds_in_regridded['mn2t24'].attrs=ds_in['mn2t24'].attrs
	
	# 3, convert the unit of the variables.
	# convert to daily precipitation
	ds = ds_in_regridded
	ds = ds.rename({'tp': 'PRAVG'})
	ds["PRAVG"] = ds["PRAVG"].diff(dim="time")        # convert from total precipitation to daily precipitation.
	ds['PRAVG'] = ds['PRAVG'].where(ds['PRAVG'] >= 0, 0)
	ds['PRAVG'] = ds['PRAVG']/86.4
	ds['PRAVG'].attrs          = ds_in['tp'].attrs
	ds['PRAVG'].attrs['units'] = 'kg m-2 s-1'
	ds = ds.rename({'mx2t24': 'T2MAX'})
	ds['T2MAX'] = (ds['T2MAX']- 273.15) * 9 / 5 + 32
	ds['T2MAX'].attrs          = ds_in['mx2t24'].attrs
	ds['T2MAX'].attrs['units'] = 'F'
	ds = ds.rename({'mn2t24': 'T2MIN'})
	ds['T2MIN'] = (ds['T2MIN']- 273.15) * 9 / 5 + 32
	ds['T2MIN'].attrs          = ds_in['mn2t24'].attrs
	ds['T2MIN'].attrs['units'] = 'F'
	# Save daily data
	ds['PRAVG'].to_netcdf(f'{path_regrid_daily}ECMWF_PRAVG_year-{year}_month-{month:02}.nc')
	ds['T2MAX'].to_netcdf(f'{path_regrid_daily}ECMWF_T2MAX_year-{year}_month-{month:02}.nc')
	ds['T2MIN'].to_netcdf(f'{path_regrid_daily}ECMWF_T2MIN_year-{year}_month-{month:02}.nc')

	# 4, resample to monthly data
	ds['PRAVG'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_PRAVG_year-{year}_month-{month:02}.nc')
	ds['T2MAX'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_T2MAX_year-{year}_month-{month:02}.nc')
	ds['T2MIN'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_T2MIN_year-{year}_month-{month:02}.nc')




# Run the script
for year  in years:
	for month in months:
		download_from_cds(year,month)
		ungrib_interpolate_resamp(year,month)