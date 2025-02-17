raw_init_date = '20231018'
user_server   = 'guangwei@129.2.80.228'




import os
import post_bias_correction   as     pbc

# Make directory to save post bias correction resut.
pbc.check_mkdir_dawn(user_server,raw_init_date)

var_names     = ['T2MAX','T2MIN','PRAVG','ASWDNS']
for var_name in var_names:
	pbc.post_bias_correction(raw_init_date,var_name,user_server)
