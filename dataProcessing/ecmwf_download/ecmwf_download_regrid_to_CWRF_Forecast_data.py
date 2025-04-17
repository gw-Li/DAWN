import cdsapi
import sys,os
import datetime 
import calendar
import cfgrib
import numpy    as     np
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



# Define functions.
def download_temp_dew_temp_from_cds(year,month):
	request = {
	    "originating_centre": "ecmwf",
	    "system": "51",
	    "variable": [
	        "2m_dewpoint_temperature",
	        "2m_temperature"
	    ],
	    "year": [f"{year}"],
	    "month": [f"{month}"],
	    "day": ["01"],
	    "leadtime_hour":list_lead_6hourly,
	    "data_format": "grib"
	}
	target = f'ecmwf_dewpoint_year-{year}_month-{month:02}.grib'
	client = cdsapi.Client()
	client.retrieve(dataset, request,target)

def ungrib_interpolate_resamp_relative_humidity(year,month):
	# 1st, ungrid the files to netcdf format.
	file_name = f'ecmwf_dewpoint_year-{year}_month-{month:02}'
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
	
	# 3, convert the unit of the variables.
	# 3.1 Convert Kelvin to Celsius.
	T_c   = ds_in_regridded['t2m'] - 273.15
	Td_c  = ds_in_regridded['d2m'] - 273.15

	# 3.2 Compute saturation vapor pressure (es) and actual vapor pressure (e)
	# Using the Tetens formula (with T in Celsius), where the unit of pressure is hPa.
	es    = 6.112 * np.exp((17.67 * T_c)  / (T_c + 243.5))
	e     = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))

	# 3.3 Calculate Relative Humidity (RH) in percent:
	RH    = 100.0 * (e / es)

	ds_in_regridded['RH'] = RH
	ds_in_regridded['RH'].attrs['units'] = '%'
	ds_in_regridded['RH'].attrs['long_name'] = 'Relative Humidity'
	ds_in_regridded['RH'].resample(time='D').mean().to_netcdf(f'{path_regrid_daily}ECMWF_RH_year-{year}_month-{month:02}.nc')

	# 4, resample to monthly data
	ds_in_regridded['RH'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_RH_year-{year}_month-{month:02}.nc')


# Define functions.
def download_srad_from_cds(year,month):
	request = {
	    "originating_centre": "ecmwf",
	    "system": "51",
	    "variable": ["surface_solar_radiation_downwards"],
	    "year": [f"{year}"],
	    "month": [f"{month}"],
	    "day": ["01"],
	    "leadtime_hour":list_lead_hours,
	    "data_format": "grib"
	}
	target = f'ecmwf_srad_year-{year}_month-{month:02}.grib'
	client = cdsapi.Client()
	client.retrieve(dataset, request,target)

def ungrib_interpolate_resamp_srad(year,month):
	# 1st, ungrid the files to netcdf format.
	file_name = f'ecmwf_srad_year-{year}_month-{month:02}'
	ds        = xr.open_dataset(f'{path_raw}{file_name}.grib', engine='cfgrib')
	ds = ds.drop_vars('time') 
	ds = ds.swap_dims({'step': 'valid_time'})
	ds = ds.rename({'valid_time': 'time'})
	ds = ds.drop_vars('step')
	ds = ds.drop_vars('surface')
	# 2nd, interpolate to CWRF grid.
	ds_in           = ds
	del(ds)
	ds_out          = xr.Dataset({'lat': ds_cwrf['XLAT_M'].isel(Time=0), 'lon': ds_cwrf['XLONG_M'].isel(Time=0)})
	regridder       = xe.Regridder(ds_in, ds_out, method='bilinear', filename=wtg_file, reuse_weights=True)
	ds_in_regridded = regridder(ds_in)
	# Copy the attribute
	ds_in_regridded.attrs          =ds_in.attrs
	ds_in_regridded
	# 3, convert the unit of the variables.
	ds = ds_in_regridded
	ds = ds.rename({'ssrd': 'ASWDNS'})
	ds["ASWDNS"] = ds["ASWDNS"].diff(dim="time")       # Accumulate value to daily value
	ds['ASWDNS'] = ds['ASWDNS'].where(ds['ASWDNS'] >= 0, 0)
	ds['ASWDNS'].attrs = ds_in['ssrd'].attrs
	# Compute the average flux in W m**-2 by dividing by the number of seconds in 24 hours.
	ds['ASWDNS']=ds['ASWDNS']/ 86400.0
	ds['ASWDNS'].attrs['units'] = 'W m**-2'
	# Save daily data
	ds['ASWDNS'].to_netcdf(f'{path_regrid_daily}ECMWF_ASWDNS_year-{year}_month-{month:02}.nc')
	# 4, resample to monthly data
	ds['ASWDNS'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_ASWDNS_year-{year}_month-{month:02}.nc')

def download_volumetric_soil_moisture_from_cds(year,month):
	request = {
	    "originating_centre": "ecmwf",
	    "system": "51",
	    "variable": ["volumetric_soil_moisture"],
	    "year": [f"{year}"],
	    "month": [f"{month}"],
	    "day": ["01"],
	    "leadtime_hour":list_lead_hours,
	    "data_format": "grib"
	}
	target = f'ecmwf_volumetric_soil_moisture_year-{year}_month-{month:02}.grib'
	client = cdsapi.Client()
	client.retrieve(dataset, request,target)

def ungrib_interpolate_resamp_volumetric_soil_moisture(year,month):
	
	# 1st, ungrid the files to netcdf format.
	file_name = f'ecmwf_volumetric_soil_moisture_year-{year}_month-{month:02}'
	ds        = xr.open_dataset(f'{path_raw}{file_name}.grib', engine='cfgrib')
	ds = ds.drop_vars('time') 
	ds = ds.swap_dims({'step': 'valid_time'})
	ds = ds.rename({'valid_time': 'time'})
	ds = ds.drop_vars('step')
	# ds.to_netcdf(f'{path_raw_nc}{file_name}.nc') 
	
	# 2nd, interpolate to CWRF grid.
	ds_in           = ds
	del(ds)
	ds_out          = xr.Dataset({'lat': ds_cwrf['XLAT_M'].isel(Time=0), 'lon': ds_cwrf['XLONG_M'].isel(Time=0)})
	regridder       = xe.Regridder(ds_in, ds_out, method='bilinear', filename=wtg_file, reuse_weights=True)
	ds_in_regridded = regridder(ds_in)
	# Copy the attribute
	ds_in_regridded.attrs          =ds_in.attrs
	
	# 3, split the data and save monthly_daily data
	ds = ds_in_regridded.sel(soilLayer=1)
	ds = ds.rename({'vsw': 'SOILM1'})
	ds['SOILM1'].attrs = ds_in['vsw'].attrs
	ds['SOILM1'].to_netcdf(f'{path_regrid_daily}ECMWF_SOILM1_year-{year}_month-{month:02}.nc')
	ds['SOILM1'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_SOILM1_year-{year}_month-{month:02}.nc')
	del(ds)

	ds = ds_in_regridded.sel(soilLayer=2)
	ds = ds.rename({'vsw': 'SOILM2'})
	ds['SOILM2'].attrs = ds_in['vsw'].attrs
	ds['SOILM2'].to_netcdf(f'{path_regrid_daily}ECMWF_SOILM2_year-{year}_month-{month:02}.nc')
	ds['SOILM2'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_SOILM2_year-{year}_month-{month:02}.nc')
	del(ds)

	ds = ds_in_regridded.sel(soilLayer=3)
	ds = ds.rename({'vsw': 'SOILM3'})
	ds['SOILM3'].attrs = ds_in['vsw'].attrs
	ds['SOILM3'].to_netcdf(f'{path_regrid_daily}ECMWF_SOILM3_year-{year}_month-{month:02}.nc')
	ds['SOILM3'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_SOILM3_year-{year}_month-{month:02}.nc')
	del(ds)

	ds = ds_in_regridded.sel(soilLayer=4)
	ds = ds.rename({'vsw': 'SOILM4'})
	ds['SOILM4'].attrs = ds_in['vsw'].attrs
	ds['SOILM4'].to_netcdf(f'{path_regrid_daily}ECMWF_SOILM4_year-{year}_month-{month:02}.nc')
	ds['SOILM4'].resample(time='ME').mean().to_netcdf(f'{path_regrid_monthly}ECMWF_monthly_SOILM4_year-{year}_month-{month:02}.nc')
	del(ds)




# Run the script
year,month= 2025,4
# for tmin (F),tmax (F),prate (kg m**-2 s**-1)
download_from_cds(year,month)
ungrib_interpolate_resamp(year,month)
# for relative humidity (we previously generated this from pressfc & q2m)
download_temp_dew_temp_from_cds(year,month)
ungrib_interpolate_resamp_relative_humidity(year,month)
# for srad (W m**-2)
download_srad_from_cds(year,month)
ungrib_interpolate_resamp_srad(year,month)
# for soil moisture (percentage)
download_volumetric_soil_moisture_from_cds(year,month)
ungrib_interpolate_resamp_volumetric_soil_moisture(year,month)