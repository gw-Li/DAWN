begin
; regrid tmmx to us_cwrf

;input data and its grid




  pathin   = "/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/CFS/"

  fin_from = addfile(pathin+"CFSraw_tmax20150500.nc","r")
  lon      = doubletofloat(fin_from->longitude)
  lat      = doubletofloat(fin_from->latitude)
  time     = fin_from->time
  data     = fin_from->TMAX_2maboveground


;output grid
  fgeo     = addfile("/cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/static/geo_em.d01_30.nc","r")
  lat2d    = fgeo->XLAT_M(0,:,:)               ; (south_north,west_east)
  lon2d    = fgeo->XLONG_M(0,:,:)

;set some optins 
  Opt                = True    
  Opt@SrcGridType    = "rectilinear"
  Opt@SrcGridLat     = lat
  Opt@SrcGridLon     = lon
  Opt@SrcGridMask    = where(.not.ismissing(data(0,:,:)),1,0) ;for missing value


  Opt@DstRegional    = True
  Opt@DstGridLat     = lat2d
  Opt@DstGridLon     = lon2d

 
;regrid
  data_regrid = ESMF_regrid(data, Opt)


   
;store the data
  ;fout = addfile("CFS202302Tmin.nc", "c")
  ;fout->Tmax    = air_temperature_regrid
  ;fout->time     = time
  ;fout->lat2d   = lat2d
  ;fout->lon2d   = lon2d

  ;system("rm    PET0.RegridWeightGen.Log")
  ;system("rm    destination_grid_file.nc")
  ;system("rm    source_grid_file.nc     ")
  system("mv    weights_file.nc       /cw3e/mead/projects/cdd103/shared/CWRF_seasonal_forecast/DATA/static/CFS2CWRF_weights_file.nc  ")

end
