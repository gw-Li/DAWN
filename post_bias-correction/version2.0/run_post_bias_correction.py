raw_init_date = '20240225'
# user_server   = 'shinsa11@129.2.80.228'
path_cwrf_raw = '/scratch16/umd-xliang/aditya/cwrf_operational/CWRF-post/V0/'
user_server   = 'guangwei@129.2.80.228'

import os
import post_bias_correction   as     pbc

pbc.post_bias_correction(raw_init_date,user_server,path_cwrf_raw)
