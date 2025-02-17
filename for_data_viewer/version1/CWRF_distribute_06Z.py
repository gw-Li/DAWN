import sys,os,glob
import shutil
import subprocess

path_v0_hist  = '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
path_trans_06Z= '/mnt/gfs01/shinsa11/CWRF_hindcast/Exp2023Dec_NonBC/'

years             = range(2012,2023+1)
months            = range(1,12+1)
days_per_month    = {     1: [ 1, 6,11,16,21,26,31],    2: [ 5,10,15,20,25],    3: [ 2, 7,12,17,22,27],    4: [ 1, 6,11,16,21,26],    5: [ 1, 6,11,16,21,26,31],    6: [ 5,10,15,20,25,30],    7: [ 5,10,15,20,25,30],    8: [ 4, 9,14,19,24,29],    9: [ 3, 8,13,18,23,28],    10:[ 3, 8,13,18,23,28],    11:[ 2, 7,12,17,22,27],    12:[ 2, 7,12,17,22,27],}
raw_init_dates    = []
phys              = ['00','02']
for year in years:
    for month in months:
        days = days_per_month[month]
        for day in days:
            raw_init_dates.append(f'{year}{month:02}{day:02}')
print(len(raw_init_dates))

vnames       = ['T2MAX','T2MIN','PRAVG']
vnames_npbc1 = ['AGHT_PL','AQ2M','ARNOF','ASNOW','ASNOWH','ASWDNS','AT2M','ATSK','AU_PL','AV_PL','axsmtg05','axsmtg','axsttg05','axsttg','AXTSS','AXWICE','PSFC','RH','uv_10']
vnames_npbc2 = ['axsmlg05','axsmlg']

for raw_init_date in raw_init_dates:
    print(raw_init_date)

    # for vname in vnames_npbc1:
    #     for phy in phys:
    #         src = f'{path_trans_06Z}CWRF-post/{raw_init_date}_icbc01_exp{phy}_{vname}_daily.nc'
    #         dst = f'{path_v0_hist}{raw_init_date}/{raw_init_date}06_icbc01_exp{phy}_{vname}_daily.nc'
    #         try:
    #             shutil.copy(src, dst)
    #         except Exception as e:
    #             print(e)
                
    # for vname in vnames_npbc2:
    #     for phy in phys:
    #         src = f'{path_trans_06Z}CWRF-post-casper/{raw_init_date}_icbc01_exp{phy}_{vname}_daily.nc'
    #         dst = f'{path_v0_hist}{raw_init_date}/{raw_init_date}06_icbc01_exp{phy}_{vname}_daily.nc'
    #         try:
    #             shutil.copy(src, dst)
    #         except Exception as e:
                # print(e)

    for vname in vnames:
        for phy in phys:
            src = f'{path_trans_06Z}CWRF-post/{raw_init_date}_icbc01_exp{phy}_{vname}_daily.nc'
            dst = f'{path_v0_hist}{raw_init_date}/{raw_init_date}06_icbc01_exp{phy}_{vname}_daily_no_adj.nc'
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(e)