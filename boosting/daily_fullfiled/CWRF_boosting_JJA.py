import os
import numpy   as np
import netCDF4 as nc
import argparse
import itertools
from   sklearn.ensemble        import GradientBoostingRegressor
from   sklearn.model_selection import KFold
from   concurrent.futures      import ProcessPoolExecutor, as_completed

# python   boosting_CWRF_T2MAXanomaly_JJA.py  "T2MAX" 'squared_error' 
def main(arg1, arg2):
    print("Argument 1:", arg1)
    print("Argument 2:", arg2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that takes two arguments.")
    parser.add_argument("arg1", help="The first argument.")
    parser.add_argument("arg2", help="The second argument.")
    args = parser.parse_args()
    main(args.arg1, args.arg2)

# training variables and cost functions.
Vname       = args.arg1#"T2MAX"
lossfunc    = args.arg2#'squared_error'   # loss='quantile','absolute_error', 'squared_error', 'huber'

# read data in
pathin      = "/home/umd-gwli/boosting/data/"
pathout     = "/home/umd-gwli/boosting/boostingresult/fullfiled/"
fileOBS     = 'OBS_2013-2022_JJA_daily.nc'
fileCWRF    = 'CWRF_2013-2022_JJA_daily.nc'
fileCFS     = 'CFS_2013-2022_JJA_daily.nc'

# Load the WRF output NetCDF file
datasetOBS  = nc.Dataset(pathin+fileOBS)
DATA_OBS    = datasetOBS.variables[Vname]
timeOBS     = datasetOBS.variables["time"]

datasetCWRF = nc.Dataset(pathin+fileCWRF)
DATA_CWRF   = datasetCWRF.variables[Vname]

datasetCFS  = nc.Dataset(pathin+fileCFS)
DATA_CFS    = datasetCFS.variables[Vname]

datasetMASK = nc.Dataset('/home/umd-gwli/data/staticData/USMASK_OBS.nc')
USMASK      = datasetMASK.variables['USMASK']
print("data read in")


#define the boosting method (one point)
def process_point(i, j):
    gb_reg  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss=lossfunc)  # loss='quantile','absolute_error', 'squared_error', 'huber'
    # Split CWRF data into train and test, and transpose.
    C3D     = DATA_CWRF[:, :, :, i, j]
    X       = C3D.reshape(-1, C3D.shape[2]).transpose()
    # Split OBS data into train and test
    OBS     = DATA_OBS[:, i, j]
    kf      = KFold(n_splits=10)          # for cross-validation
    y_out   = np.array([])
    for train_index, test_index in kf.split(OBS):
        y_train, y_test = OBS[train_index], OBS[test_index]
        X_train, X_test = X[train_index], X[test_index]
        gb_reg.fit(X_train, y_train)
        y_out  = np.concatenate((y_out, gb_reg.predict(X_test)))
    #y_out = np.where((y_out >= 220) & (y_out <= 330), y_out, np.nan)  # Set values outside the range to missing (np.nan)
    return (i, j, y_out)


#  use nested loops to iterate over the range of points 
num_points_i = DATA_OBS.shape[1]
num_points_j = DATA_OBS.shape[2]
output = np.full((num_points_i, num_points_j, DATA_OBS.shape[0]), np.nan, dtype=float)

for i in range(num_points_i):
    print(f"Processing row {i}")
    for j in range(num_points_j):
        if USMASK[i, j] == 1 :
            _, _, y_out  = process_point(i, j)
            output[i, j] = y_out

# bugs in parallel run

# Create a new NetCDF file
#os.remove(pathout+"CWRF_"+Vname+"_anomaly_JJA_"+lossfunc+".nc") 
output_file = nc.Dataset(pathout+"CWRF_"+Vname+"_anomaly_JJA_"+lossfunc+".nc", "w", format="NETCDF4")

# Create dimensions
output_file.createDimension("x", num_points_i)
output_file.createDimension("y", num_points_j)
output_file.createDimension("time", DATA_OBS.shape[0])

# Create variables
x    = output_file.createVariable("x", "f4", ("x",))
y    = output_file.createVariable("y", "f4", ("y",))
time = output_file.createVariable("time", "f4", ("time",))
predicted_values = output_file.createVariable("predicted_values", "f4", ("time","x", "y"))

# Add data to the variables
x[:] = np.arange(num_points_i)
y[:] = np.arange(num_points_j)
time = np.arange(DATA_OBS.shape[0])

output_transposed = np.transpose(output, (2, 0, 1)) # change the dimensions.
predicted_values[:, :, :] = output_transposed

# Add some attributes to the variables (optional)
x.units    = "grid index"
y.units    = "grid index"
#time.units = "index of cross-validation fold"
predicted_values.units = "temperature (K)"

# Close the output file
output_file.close()