import os
import post_bias_correction   as     pbc

#user_server  = 'shinsa11@129.2.80.228'
path_cwrf_raw = '/scratch16/umd-xliang/aditya/cwrf_operational/CWRF-post/'

user_server   = 'guangwei@129.2.80.228'
var_names     = ['T2MAX','T2MIN','PRAVG','ASWDNS']
raw_init_dates= ['20231003','20231008','20231013','20231018','20231023','20231028','20231102','20231107','20231112','20231117']

for raw_init_date in raw_init_dates:
    # Make directory to save post bias correction resut.
    pbc.check_mkdir_dawn(user_server,raw_init_date)

    for var_name in var_names:
        pbc.post_bias_correction(raw_init_date,var_name,user_server,path_cwrf_raw)