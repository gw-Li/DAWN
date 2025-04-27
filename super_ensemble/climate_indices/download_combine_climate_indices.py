import os,re
import cftime
import requests
import numpy      as np
import pandas     as pd
import xarray     as xr
from urllib.parse import urlparse
from io           import StringIO


output_dir    = 'indices'
list_ascii    = ['aao','ao','nao','pna']
list_data     = ['meiv2','noi','np','qbo','tpi','whwp']
list_ncfiles  = ['amm','dmi','nat','tasi','tna','tsa']
list_nc_T     = ['ea_wr','ea','epnp','epv10','peu','pt','sand','tnh','wp']

dict_var_names={'amm' :'AMM' ,     'dmi'  :'DMI',     'nat'  :'NAT' ,    'tasi':'TASI',
                'tna' :'TNA' ,     'tsa'  :'TSA',     'ea_wr':'EAWR',    'ea'  :'EA'  ,
                'epnp':'EPNP',     'epv10':'ExplVar', 'peu'  :'POL' ,    'pt'  :'PNA' ,
                'sand':'SCA' ,     'tnh'  :'TNH',     'wp'   :'WP'  ,    'npgo':'NPGO',
                'amo' :'AMO' ,     'aao'  :'AAO',     'ao'   :'AO'  ,    'nao' :'NAO' ,
                'pna' :'PNA' ,     'meiv2':'MEIV2',   'noi'  :'NOI' ,    'np'  :'NP'  ,
                'qbo' :'QBO' ,     'tpi'  :'TPI',     'whwp' :'WHWP',    'pdo' :'PDO'  ,
                'soi' :'SOI' ,}

dict_nc_names ={'amm' :'value',    'dmi'  :'value',   'nat'  :'NAT' ,    'tasi':'TASI',
                'tna' :'TNA',      'tsa'  :'TSA',     'ea_wr':'EAWR',    'ea'  :'EA'  ,
                'epnp':'EPNP',     'epv10':'ExplVar', 'peu'  :'POL' ,    'pt'  :'PNA' ,
                'sand':'SCA',      'tnh'  :'TNH',     'wp'   :'WP'  ,    'npgo':'value'}

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# URLs for downloading (note the commas between entries)
url_indices = {
'pdo'  : 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat',
'nino' : 'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices',
'aao'  : 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii',
'ao'   : 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii',
'nao'  : 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii',
'pna'  : 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii',
'meiv2': 'https://psl.noaa.gov/enso/mei/data/meiv2.data',
'soi'  : 'https://www.cpc.ncep.noaa.gov/data/indices/soi',
'tna'  : 'https://stateoftheocean.osmc.noaa.gov/sur/data/tna.nc',
'tsa'  : 'https://stateoftheocean.osmc.noaa.gov/sur/data/tsa.nc',
'nat'  : 'https://stateoftheocean.osmc.noaa.gov/sur/data/nat.nc',
'tasi' : 'https://stateoftheocean.osmc.noaa.gov/sur/data/tasi.nc',
'whwp' : 'https://psl.noaa.gov/data/correlation/whwp.data',
'tpi'  : 'https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.data',
'noi'  : 'https://psl.noaa.gov/data/correlation/noi.data',
'np'   : 'https://psl.noaa.gov/data/timeseries/month/data/np.long.data',
'amo'  : 'https://tropical.colostate.edu/Forecast/downloadable/csu_amo.csv',
'amm'  : 'https://www.psl.noaa.gov/data/timeseries/month/data/amm.nc',
'dmi'  : 'https://www.psl.noaa.gov/data/timeseries/month/data/dmi.had.long.nc',
'npgo' : 'https://www.psl.noaa.gov/data/correlation/npgo.nc',
'qbo'  : 'https://psl.noaa.gov/data/correlation/qbo.data',
'ea'   : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/EA/data.nc',
'ea_wr': 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/EAWR/data.nc',
'epnp' : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/EPNP/data.nc',
'peu'  : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/POL/data.nc',
'pt'   : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/PNA/data.nc',
'sand' : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/SCA/data.nc',
'tnh'  : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/TNH/data.nc',
'wp'   : 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/WP/data.nc',
'epv10': 'https://iridl.ldeo.columbia.edu/SOURCES/.Indices/.CPC_Indices/.NHTI/ExplVar/data.nc',
}

# Self-defined functions for read data and convert to xarray dataset.
def func_read_ascii_convert2xr( fname: str, var_name: str, missing_val: float = -99.0) -> xr.Dataset:
    df           = pd.read_csv( fname, delim_whitespace=True, header=None, names=['year', 'month', var_name])
    df           = df.dropna(subset=[var_name])  # 2. Drop any missing or invalid lines (if needed)
    # 3. Create a datetime column at the first of each month
    df['time']   = pd.to_datetime({ 'year' : df['year'].astype(int), 'month': df['month'].astype(int), 'day'  : 1 })
    aaots        = df.set_index('time')[var_name].sort_index() # 4. Set the time index and sort
    ds           = aaots.to_xarray().to_dataset(name=var_name)  # 5. Convert to xarray Dataset
    return ds

def func_read_csv_convert2xr( fname: str, var_name: str ) -> xr.Dataset:
    df          = pd.read_csv(f'{output_dir}/amo.csv', sep=',')                              # 1. Read the CSV. Adjust sep if your file is tab‑ or space‑delimited.
    df_long     = df.melt(    id_vars='Year',    var_name='Month',    value_name=var_name)   # 2. Melt from wide to long format
    df_long['month'] = df_long['Month'].map({m: i+1 for i, m in enumerate(MONTHS)})
    df_long          = df_long.dropna(subset=[var_name])                                     # 3. Drop any rows where AMO is missing
    # 4. Create a datetime column at the first of each month
    df_long['time']  = pd.to_datetime({ 'year' : df_long['Year'].astype(int),'month': df_long['month'].astype(int),'day'  : 1 })
    amo_ts      = df_long.set_index('time')[var_name].sort_index()                           # 5. Pivot to a time‑indexed Series and convert to xarray
    ds          = amo_ts.to_xarray().to_dataset(name=var_name)
    return ds

def func_read_data_convert2xr( fname: str, var_name: str, start_header: int = 1, missing_val: float = -99.0) -> xr.Dataset:
    # 1) Slurp valid rows
    records = []
    with open(fname) as f:
        for _ in range(start_header):
            next(f)
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if len(parts) < 13:
                continue
            vals = [float(tok) for tok in parts[1:13]]
            records.append([year] + vals)
    # 2) Build a DataFrame
    df      = pd.DataFrame(records, columns=['year'] + MONTHS)
    df[MONTHS]   = df[MONTHS].replace(missing_val, np.nan)
    # 3) Melt to long form
    df_long = df.melt( id_vars='year', value_vars=MONTHS, var_name='month', value_name=var_name )
    # 4) Map month names → numbers
    month_map = {m: i+1 for i, m in enumerate(MONTHS)}
    df_long['month_num'] = df_long['month'].map(month_map)
    # 5) Drop missing
    df_long = df_long.dropna(subset=[var_name])
    # 6) Build a true datetime index (first of each month)
    df_long['time'] = pd.to_datetime({ 'year':  df_long['year'].astype(int), 'month': df_long['month_num'].astype(int),  'day':   1 })
    # 7) Sort and set index
    df_long = df_long.sort_values('time').set_index('time')
    # 8) Convert to xarray
    ds = df_long[var_name].to_xarray().to_dataset()
    return ds

def func_read_pdodat_convert2xr(fname, var_name, na_values):
    # 1) find the header
    with open(fname) as f:
        for i, line in enumerate(f):
            if line.strip().startswith('Year') and 'Jan' in line:
                header_row = i
                break
    # 2) read the table, treating only your flag as missing
    df = pd.read_csv(  fname,
        delim_whitespace=True,         skiprows=header_row,
        header    = 0,                 names=['year'] + MONTHS,
        na_values =na_values,          keep_default_na=False    )
    # 3) drop any year that's entirely empty
    df = df.dropna(subset=MONTHS, how='all')
    # 4) melt to long form — **do not drop NaNs here**!
    df_long = df.melt( id_vars=['year'],  value_vars=MONTHS,  var_name='month',   value_name=var_name  )
    # 5) build a datetime index
    month_map    = {m: i+1 for i, m in enumerate(MONTHS)}
    df_long['time'] = pd.to_datetime({ 'year' :  df_long['year'], 'month': df_long['month'].map(month_map), 'day'  :   1, })
    df_long      = df_long.sort_values('time')
    return df_long.set_index('time')[var_name].to_xarray().to_dataset()

def func_read_soidat_convert2xr(fname, var_name, na_values):
    # 1) read all lines
    with open(fname) as f:
        lines    = f.readlines()
    # 2) find the “YEAR  Jan  Feb …” header
    header_row   = next( i for i, line in enumerate(lines)
        if line.strip().upper().startswith('YEAR') and 'JAN' in line.upper()     )
    # 3) find the start of the STANDARDIZED block (we’ll foot there)
    footer_row   = next(        (i for i, line in enumerate(lines[header_row+1:], start=header_row+1)
         if line.strip().upper().startswith('(STAND TAHITI')),           len(lines)    )
    # 4) pull out only the anomaly‐block lines
    block_lines  = lines[header_row:footer_row]
    # 4a) drop any truly malformed lines (like ones with no space after the 4‐digit year)
    clean_lines  = [  ln for ln in block_lines   if len(ln) > 4 and ln[4].isspace()  ]
    # 4b) **inject a space** before every “-” that follows a digit
    #     so that “2.8-999.9-999.9” → “2.8 -999.9 -999.9”
    fixed_lines  = [     re.sub(r'(?<=\d)(-)', r' \1', ln)   for ln in clean_lines  ]
    block        = ''.join(fixed_lines)
    # 5) read into pandas
    df           = pd.read_csv(         StringIO(block),
        delim_whitespace=True,          header=0,
        names    =['year'] + MONTHS,    na_values=na_values,
        keep_default_na=False         ).dropna(subset=MONTHS, how='all')
    # 6) melt into long form
    df_long      = df.melt(             id_vars='year',
        value_vars=MONTHS,              var_name='month',
        value_name=var_name   )
    # 7) construct proper datetime index
    month_map    = {m: i+1 for i, m in enumerate(MONTHS)}
    df_long['time'] = pd.to_datetime({  'year':  df_long['year'],  'month': df_long['month'].map(month_map),    'day':   1  })
    df_long      = df_long.sort_values('time')
    # 8) push into xarray
    return df_long.set_index('time')[var_name].to_xarray().to_dataset()

def func_read_indices_convert2xr():
    df = pd.read_csv(
        f'{output_dir}/nino.indices',
        delim_whitespace=True,
        header=None,        skiprows=1,      # <-- skip the "YR MON NINO1+2 ..." line
        names =[ 'year','month', 'NINO12','ANOM12',  'NINO3', 'ANOM3',
            'NINO4', 'ANOM4',  'NINO34','ANOM34'    ])
    # 2) Build a proper pandas datetime index (first of each month)
    df['time'] = pd.to_datetime({  'year':  df.year,  'month': df.month, 'day':   1 })
    # 3) Convert to xarray, keeping only the anomaly columns and renaming them
    ds = xr.Dataset(    {
            'NINO12': ('time', df.ANOM12.values),
            'NINO3':  ('time', df.ANOM3.values),
            'NINO4':  ('time', df.ANOM4.values),
            'NINO34': ('time', df.ANOM34.values),
        },    coords={'time': df.time.values}    )
    return ds


# *************** Download the datafiles ****************
output_dir   = 'indices'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for name, url in url_indices.items():
    print(f'Downloading {name} from {url}...')
    resp     = requests.get(url, stream=True)
    resp.raise_for_status()
    
    # Derive a sensible filename: <key> + original extension (if any)
    parsed   = urlparse(url)
    ext      = os.path.splitext(parsed.path)[1] or '.dat'
    out_path = os.path.join(output_dir, f"{name}{ext}")
    
    with open(out_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f'  → saved to {out_path}')

print("All downloads complete.")





# *************** Read and Convert to netcdf format ****************
ds_ind = xr.Dataset()
# --- for *.nc files ---
# with time dimension
list_ncfiles= ['amm','dmi','nat','tasi','tna','tsa']
for i in range(len(list_ncfiles)):
    ncfile  = list_ncfiles[i]
    ds      = xr.open_dataset(f'{output_dir}/{ncfile}.nc')
    if 'TIME' in ds.coords:
        ds = ds.rename({'TIME': 'time'})
    ds_ind[dict_var_names[list_ncfiles[i]]]=ds[dict_nc_names[list_ncfiles[i]]]
# with T as time dimension
list_nc_T   = ['ea_wr','ea','epnp','epv10','peu','pt','sand','tnh','wp']
for i in range(len(list_nc_T)):
    ncfile  = list_nc_T[i]
    ds = (    xr.open_dataset(f'{output_dir}/{ncfile}.nc', decode_times=False).rename({'T':'time'}))
    ds = ds.assign_coords(time = pd.date_range('1950-01-01',periods=ds.sizes['time'],freq='MS'))
    ds_ind[dict_var_names[list_nc_T[i]]]=ds[dict_nc_names[list_nc_T[i]]]
# NPGO
ds     =    xr.open_dataset(f'{output_dir}/npgo.nc', decode_times=False)
ds     = ds.assign_coords(time = pd.date_range('1950-01-01',periods=ds.sizes['time'],freq='MS'))
ds_ind[dict_var_names['npgo']]=ds[dict_nc_names['npgo']]

# --- for *.csv files ---
ds_ind[dict_var_names['amo']]    = func_read_csv_convert2xr(fname = f'{output_dir}/amo.csv',var_name = 'AMO' )['AMO']

# --- for *.ascii files ---
list_ascii  = ['aao','ao','nao','pna']
for i in range(len(list_ascii)):
    ds      = func_read_ascii_convert2xr(fname=f'{output_dir}/{list_ascii[i]}.ascii', var_name=dict_var_names[list_ascii[i]])
    ds_ind[dict_var_names[list_ascii[i]]] = ds[dict_var_names[list_ascii[i]]]

# --- for *.data files ---
list_data   = ['meiv2','noi','np','qbo','tpi','whwp']
list_mis    = [-999.00,-999.0,-999,-999.0,-99, -99.99]
for i in range(len(list_data)):
    ds      = func_read_data_convert2xr( fname=f'{output_dir}/{list_data[i]}.data', var_name=dict_var_names[list_data[i]],start_header=1, missing_val=list_mis[i] )
    ds_ind[dict_var_names[list_data[i]]] = ds[dict_var_names[list_data[i]]]

# --- for *.dat files ---
ds_pdo = func_read_pdodat_convert2xr(f'{output_dir}/pdo.dat', 'PDO', na_values=[99.99])
ds_soi = func_read_soidat_convert2xr(f'{output_dir}/soi.dat', 'SOI', na_values=[-999.9])
ds_ind[dict_var_names['pdo']] = ds_pdo[dict_var_names['pdo']]
ds_ind[dict_var_names['soi']] = ds_soi[dict_var_names['soi']]

# # # --- for *.indices file ---
ds_sstoi         = func_read_indices_convert2xr()
ds_ind['NINO12'] =ds_sstoi['NINO12']
ds_ind['NINO3']  =ds_sstoi['NINO3']
ds_ind['NINO4']  =ds_sstoi['NINO4']
ds_ind['NINO34'] =ds_sstoi['NINO34']



# *************** organize as input of AI model ****************
init_year, init_mnth = 2025,1

ds_sub = ds_ind.sel(time=slice('1990-01', f'{init_year}-{init_mnth:02}'))

# Check the missing values.
nan_vars = [
    var for var in ds_sub.data_vars
    if ds_sub[var].isnull().any(dim="time").item()]
print("Variables with at least one NaN:", nan_vars)
nan_counts = {
    var: int(ds_sub[var].isnull().sum(dim="time").item())
    for var in ds_sub.data_vars
    if ds_sub[var].isnull().any(dim="time").item() }

print("NaN counts per variable:", nan_counts)

#Drop the missing values.
ds_no_nans = ds_sub.drop_vars(nan_vars)
num_inds   = len(ds_no_nans.data_vars)
print('There are ',num_inds,' climate indices in total.')

def select_year_month(year, init_month, MBI):
    if init_month-MBI>0:
        year  = year
        month = init_month-MBI
    else:
        year  = year -1
        month = init_month-MBI + 12
        if month <=0:
            year  = year -1
            month = month + 12
    return year, month


sel_years = range(1993,init_year+1)

list_MBI_year = []
for year in sel_years:
    list_MBI  = []
    for MBI in MBIs:
        sel_year, sel_month = select_year_month(init_year, init_mnth, MBI)
        sel_data = ds_no_nans.sel(time=f'{sel_year}-{sel_month:02}').squeeze('time',drop=True)
        list_MBI.append(sel_data.expand_dims(MBI = [MBI]))
    ds_mbi  = xr.concat(list_MBI,dim='MBI')
    list_MBI_year.append(ds_mbi.expand_dims(year = [year]))
ds_mbi_year = xr.concat(list_MBI_year,dim='year')
fname = f'CI_4_AI_{num_inds}indices_init_{init_year}-{init_mnth:02}_24MBI.nc'
ds_mbi_year.to_netcdf(fname)
print(fname 'is saved')


