import sys,os
import subprocess
import datetime 
import cfgrib
import requests
import xarray   as xr
import xesmf    as xe
from multiprocessing import Pool
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from mpi4py import MPI


# Modify the begin and end month if necessary, as well as years and the file directory.
beg_month_int,beg_day_int,end_month_int, end_day_int = 6,1,8,31
init_month_ints = [3,4,5]
init_day_int    = 1
init_hour_int   = 0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


years    = range(2011,2023)
data_dir = '/home/umd-gwli/scratch16-umd-xliang/gwli/DATA/CFS/'
stic_dir = '/home/umd-gwli/scratch16-umd-xliang/gwli/DATA/static/'
TIMEFMT  = '%Y-%m-%d-%H'

# Add more variables if necessary.
var_names= ['PRAVG','T2MAX','T2MIN','ASWDNS','AQ2M','SOILT1','SOILM1','SOILM2','SOILM3','SOILM4','AT2M',]


varname_mapping = {
    "PRAVG": {
        "filename_fragment" :  "prate",
        "original_var_name" :  "prate",
    },
    "T2MAX": {
        "filename_fragment" :  "tmax",
        "original_var_name" :  "tmax",
    },
    "T2MIN": {
        "filename_fragment" :  "tmin",
        "original_var_name" :  "tmin",
    },
    "ASWDNS": {
        "filename_fragment" :  "dswsfc",
        "original_var_name" :  "dswrf",
    },
    "AT2M": {
        "filename_fragment" :  "tmp2m",
        "original_var_name" :  "t2m",
    },
    "AQ2M": {
        "filename_fragment" :  "q2m",
        "original_var_name" :  "sh2",
    },
    "SOILT1": {
        "filename_fragment" :  "soilt1",
        "original_var_name" :  "t",
    },
    "SOILM1": {
        "filename_fragment" :  "soilm1",
        "original_var_name" :  "soilw",
    },
    "SOILM2": {
        "filename_fragment" :  "soilm2",
        "original_var_name" :  "soilw",
    },
    "SOILM3": {
        "filename_fragment" :  "soilm3",
        "original_var_name" :  "soilw",
    },
    "SOILM4": {
        "filename_fragment" :  "soilm4",
        "original_var_name" :  "soilw",
    },
}

# Specify the algorithm for calculating daily
def calculate_daily_values(data, var_name):
    if var_name == 'T2MIN':
        dailydata =  data.resample(time='1D').min()
        dailydata.attrs['units'] = data.attrs['units']
        return dailydata
    elif var_name == 'T2MAX':
        dailydata =  data.resample(time='1D').max()
        dailydata.attrs['units'] = data.attrs['units']
        return dailydata
    elif var_name == 'PRAVG':
        dailydata = data.resample(time='1D').mean() * 86400
        dailydata.attrs['units'] = 'mm/d'
        return  dailydata
    else:
        dailydata =  data.resample(time='1D').mean()
        dailydata.attrs['units'] = data.attrs['units']
        return  dailydata


# Define a class for Data preparation and downloading
class DataDownloader:
    def __init__(self, var_name, time_init, data_dir):
        self.var_name   = var_name
        self.time_init  = time_init
        self.data_dir   = data_dir
        self.link_remot = 'https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/time-series/'

    def assemble_filenames(self):
        init_year, init_month, init_day, init_hour = self.time_init.split("-")
        file_name_original = varname_mapping[self.var_name]['filename_fragment'] + ".01." +  str(init_year) + \
                             str(init_month)+ str(init_day)+ str(init_hour) + '.daily.grb2'
        self.file_name_original = file_name_original
        self.file_name_raw_6h   = 'CFSraw_6h_'    + var_name +  "_" + time_init +'.nc'
        self.url = self.link_remot + init_year + "/" + init_year + init_month + "/" + init_year + init_month + \
              init_day + "/" + init_year + init_month + init_day + init_hour + "/" + self.file_name_original

    def download_and_rename(self):
        response = requests.get(self.url, stream=True)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            print(f"Download failed: {response.status_code}, {response.reason}")
            return False
        with open(self.data_dir + 'grib_file/' + self.file_name_original, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=8192):
                fd.write(chunk)
        print(f"Download succeeded: {self.file_name_original}")
        return True



# Define a class for handling file conversion
class FileConverter:
    def __init__(self, downloader):
        self.downloader = downloader

    def convert_file(self):
        # Convert data from grib2 file format to netcdf file format.
        self.ds = xr.open_dataset(self.downloader.data_dir + 'grib_file/' + self.downloader.file_name_original , engine='cfgrib')

        # Delete temporary files when reading grib2 files
        os.system(f'rm {self.downloader.data_dir}grib_file/{self.downloader.file_name_original}*.idx')

        # Convet the 'step' to a 'time' coordinate
        self.ds['time'] = self.ds['time'] + self.ds['step']
        self.ds = self.ds.set_coords('time')
        self.ds = self.ds.swap_dims({'step': 'time'})

        # Modify the variable name and coordinate name
        self.ds = self.ds.rename({varname_mapping[self.downloader.var_name]['original_var_name']:self.downloader.var_name ,})

        # Save CFS raw data
        self.ds.to_netcdf(self.downloader.data_dir + 'raw_data/' + self.downloader.file_name_raw_6h)




class DataRegridder:
    def __init__(self, converter, time_beg, time_end, stic_dir, var_name):
        self.converter = converter
        self.time_beg  = time_beg
        self.time_end  = time_end
        self.stic_dir  = stic_dir
        self.var_name  = var_name

    def regrid_data(self):
        # Read in the meteorology data on a regular lat/lon grid
        wtg_file = self.stic_dir + 'CFS2CWRF_weights_file.nc'
        ds_in    = self.converter.ds

        # Read in the CWRF grid file
        ds_cwrf  = xr.open_dataset(self.stic_dir + 'geo_em.d01_30.nc')

        # Create a new dataset with the latitude and longitude from the WRF grid
        ds_out   = xr.Dataset({'lat': ds_cwrf['XLAT_M'].isel(Time=0), 'lon': ds_cwrf['XLONG_M'].isel(Time=0)})

        # Create the regridder using the pre-generated weights
        regridder = xe.Regridder(ds_in, ds_out, method='bilinear', filename=wtg_file, reuse_weights=True)

        # Regrid the meteorology data
        ds_in_regridded = regridder(ds_in)

        # copy the attribute of units to regrided dataset.
        ds_in_regridded[self.var_name].attrs['units'] = ds_in[self.var_name].attrs.get('units')

        # Convert to daily data
        daily_data = calculate_daily_values(ds_in_regridded[self.var_name], self.var_name)

        # Trim the dataset along the 'time' dimension
        daily_data_trim = daily_data.sel(time=slice(self.time_beg, self.time_end))


        # Export the result to netcdf format.
        file_name_CFS   = 'CFS_' + var_name +  "_" + time_beg  + "_" + time_end + "_EX_"+ time_init +'.nc'
        daily_data_trim.to_netcdf(self.converter.downloader.data_dir + 'regrid_daily/' + file_name_CFS)



# Continue defining classes for each major step of the function...



def process_year(var_name, year, time_beg, time_end, time_init, data_dir, stic_dir):
    # Initialize a DataDownloader instance
    downloader = DataDownloader(var_name, time_init, data_dir)
    downloader.assemble_filenames()

    # Download the data
    download_success = downloader.download_and_rename()

    # If the download was successful, convert and regrid the data
    if download_success:
        # Initialize a FileConverter instance
        converter = FileConverter(downloader)
        converter.convert_file()

        # Initialize a DataRegridder instance
        regridder = DataRegridder(converter, time_beg, time_end, stic_dir, var_name)
        regridder.regrid_data()


for var_name in var_names[0:3]:
    for year in years[rank::size]:
        for init_month_int in init_month_ints:
            # set time_beg and time_end for trim the data through time dimension
            time_beg = datetime.datetime(year, beg_month_int,  beg_day_int,  0).strftime(TIMEFMT)
            time_end = datetime.datetime(year, end_month_int,  end_day_int, 18).strftime(TIMEFMT)
            #time_init= datetime.datetime(year, init_month_int, init_day_int, init_hour_int).strftime(TIMEFMT)
            time_IC_5d=[(datetime.datetime(year, init_month_int, init_day_int, 0) - datetime.timedelta(days=i)).strftime(TIMEFMT) for i in range(2, -3, -1) ]
            indices = [0,1,3,4]
            for time_init in [time_IC_5d[i] for i in indices]:
                process_year(var_name, year, time_beg, time_end, time_init, data_dir, stic_dir)





