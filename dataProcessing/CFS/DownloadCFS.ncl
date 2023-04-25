begin
;0 pre-define 
  ;pass from commend line to ncl script.
  ;varname   = "PRAVG"
  ;time_IC   = "2015-05-01-00"
  ;time_beg  = "2015-06-01-00"
  ;time_end  = "2015-08-31-18"
  ;cfs_path      = "/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/CFS/"
  ; path of GFS and wtgfile.
  remote_path   = "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/time-series/"
  wtg_file      = "/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/static/CFS2CWRF_weights_file.nc"

  ; time variables (string format)
  ICyear    = str_get_field(time_IC  , 1, "-")
  ICmonth   = str_get_field(time_IC  , 2, "-")
  ICday     = str_get_field(time_IC  , 3, "-")
  IChour    = str_get_field(time_IC  , 4, "-")
  BegYear   = str_get_field(time_beg , 1, "-")
  BegMonth  = str_get_field(time_beg , 2, "-")
  BegDay    = str_get_field(time_beg , 3, "-")
  BegHour   = str_get_field(time_beg , 4, "-")
  EndYear   = str_get_field(time_end , 1, "-")
  EndMonth  = str_get_field(time_end , 2, "-")
  EndDay    = str_get_field(time_end , 3, "-")
  EndHour   = str_get_field(time_end , 4, "-")
  
  ; time variables (integer format)
  iBegYear  = stringtoint(BegYear)
  iBegMonth = stringtoint(BegMonth)
  iBegDay   = stringtoint(BegDay)
  iBegHour  = stringtoint(BegHour)
  iEndYear  = stringtoint(EndYear)
  iEndMonth = stringtoint(EndMonth)
  iEndDay   = stringtoint(EndDay)
  iEndHour  = stringtoint(EndHour)
  itimebegin=iBegYear*1000000 + iBegMonth *10000 + iBegDay *100 + iBegHour
  itimeend  =iEndYear*1000000 + iEndMonth *10000 + iEndDay *100 + iEndHour
  ;simple debug
  if ( ismissing( ind(  (/"00","06","12","18"/) .eq. IChour     )   ) ) then
   print("ERROR :  IC hour can only be 00, 06, 12, 12")
   exit
  end if


  ; get other names
print(varname)
  if      ( varname .eq. "T2MAX") then
            Vname = "tmax"
            Vcfsn = "TMAX_2maboveground"
            arith = "max"
  else if ( varname .eq. "T2MIN") then
            Vname = "tmax"      
            Vcfsn = "TMIN_2maboveground"
            arith = "min"
  else if ( varname .eq. "PRAVG") then
            Vname = "prate" 
            Vcfsn = "PRATE_surface"
            arith = "avg"
  end if
  end if
  end if
    variablename = Vname+".01."
    
;1st download CFS grib2 file online
  cfsgrb2file =  variablename +ICyear+ ICmonth + ICday +  IChour +".daily.grb2"
  cfsrawFile  =  "CFSraw_"+varname +"_"+ time_IC + ".nc"
  
  system("wget  --directory-prefix=" +cfs_path +"   " + remote_path +  ICyear + "/" +  ICyear + ICmonth +"/" + ICyear + ICmonth +ICday + "/" + ICyear + ICmonth +ICday  + IChour + "/" +  cfsgrb2file )
  system("ls  " + cfs_path +  cfsgrb2file)

;2nd convert frome grib2 to nc file format

  system("  wgrib2    " +  cfs_path + cfsgrb2file + "  -netcdf   " +  cfs_path + cfsrawFile)
  system("  rm        " +  cfs_path + cfsgrb2file )

;read the data for regrid data to CWRF grid and 4 trim data, select correct start and end date.
  fin_from = addfile(cfs_path + cfsrawFile,"r")
  data     = fin_from->$Vcfsn$
  time     = fin_from->time
  ymdh     = cd_calendar(time,-3)

;3 regrid data
  data_regrid  = ESMF_regrid_with_weights(data, wtg_file,False)

;4 trim data
  data_trim    = data_regrid(ind(ymdh .eq. itimebegin ): ind(ymdh .eq. itimeend),:,:)
  time_trim    =        time(ind(ymdh .eq. itimebegin ): ind(ymdh .eq. itimeend ))

;5 export data
  fout            = addfile(cfs_path + "CFS_" + varname + "_"+ time_beg+"_"+time_end+"_EX_"+time_IC+".nc" , "c")
  fout->$varname$ = data_trim
  fout->time      = time_trim


;6 calculate and save daily and monthly data.

  ; 6.1 daily
  Dailydata = calculate_daily_values(data_trim, arith , 0, False)
  if ( varname .eq. "PRAVG") then                  ;convert precipitation unit
    Dailydata           = Dailydata * 86400.0
    Dailydata@units     =  "mm/d"
    Dailydata@long_name =  "Daily Precipitation"
  end if
  fout            = addfile(cfs_path + "CFS_daily_" + varname +   "_" + time_beg+"_"+time_end+"_EX_"+time_IC+".nc" , "c")
  fout->$varname$ = Dailydata

  ; 6.2 monthly
  Monthlydata = calculate_monthly_values(Dailydata, "avg" , 0, False)
  fout            = addfile(cfs_path + "CFS_monthly_" + varname +   "_" + time_beg+"_"+time_end+"_EX_"+time_IC+".nc" , "c")
  fout->$varname$ = Monthlydata
end
