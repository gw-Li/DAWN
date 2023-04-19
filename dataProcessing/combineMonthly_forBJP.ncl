;逐个读入数据，下载的数据为1972-2022年，一共51年的逐月资料
;每年选择三个月的资料，为JJA同时，提前一个月，提前两个月，提前三个月，一直到提前12个月。
;为了方便bjp方法运行， 只整理50年的资料。及只分析提前5个月时，扔掉1972年的资料。
begin
;OBS
yearSrt= 1979
yearEnd= 2022

pathin = "/home/umd-gwli/scratch16-umd-xliang/gwli/OBS/"
pathout= "/scratch16/umd-xliang/gwli/OBS/seasonal_monthly/monthly/"
Vname  = (/"T2MAX","T2MIN","PRAVG"/)
iyear  = ispan(yearSrt, yearEnd, 1)
year   = tostring(iyear)
Nyears = dimsizes(iyear)
print("There are " + tostring(Nyears) + " years of data that to requared be combined")

do   i = 0,3-1        ; for different variables
  data = new((/3*Nyears,138,195/), float)
do   j = 0,Nyears-1   ; readdata from each year
   fin = addfile(pathin+"OBS_"+Vname(i)+"_"+year(j)+"_JJA_monthly.nc", "r")
   data(j*3:j*3+2,:,:) = fin->$Vname(i)$ 
end do
   fout = addfile(pathout+"OBS_"+Vname(i)+"_"+tostring(yearSrt)+"-"+tostring(yearEnd)+".JJA.nc","c")
   fout->$Vname(i)$ =data
   delete(data)
end do



;CWRF
pathin = "/home/umd-gwli/scratch16-umd-xliang/gwli/CWRF/eachdata/"
pathout= "/scratch16/umd-xliang/gwli/CWRF/seasonal_monthly/monthly/"
Vname  = (/"T2MAX","T2MIN","PRAVG"/)
iyear  = ispan(2013, 2022, 1)
year   = tostring(iyear)
exps   = (/"exp00","exp02","exp03","exp05","exp06"/)
start  = (/"03","04","05"/)

do   i = 0,3-1  ;for each variables
  data = new((/5,3,30,138,195/), float)
  do ee = 0,5-1
    do ss=0,3-1
       do   j = 0,10-1 ;for each years
          fin = addfile(pathin+Vname(i)+"/"+"monthly_"+year(j)+start(ss)+"01_cc00_icbc01_"+exps(ee)+"_"+Vname(i)+"_daily.nc", "r")
          data(ee,ss,j*3:j*3+2,:,:) = fin->$Vname(i)$ 
       end do
     end do 
   end do
   fout = addfile(pathout+"CWRF_"+Vname(i)+"_2013-2022.JJA.nc","c")
   fout->$Vname(i)$ =data
   delete(data)
end do



;CFS
pathin = "/home/umd-gwli/scratch16-umd-xliang/gwli/CFS/eachdata/"
pathout= "/home/umd-gwli/scratch16-umd-xliang/gwli/CFS/seasonal_monthly/monthly/"
Vname  = (/"T2MAX","T2MIN","PRAVG"/)
namesV = (/"tmax","tmin","prate"/)
iyear  = ispan(2013, 2022, 1)
year   = tostring(iyear)
start  = (/"03","04","05"/)

do   i = 0,3-1  ;for each variables
  data = new((/3,30,138,195/), float)
    do ss=0,3-1
       do   j = 0,10-1 ;for each years
          fin = addfile(pathin+Vname(i)+"/"+"monthly_nctcdf_"+namesV(i)+".01."+year(j)+start(ss)+"0100.daily.grb2.nc", "r")
          data(ss,j*3:j*3+2,:,:) = fin->$Vname(i)$ 
       end do
    end do
   fout = addfile(pathout+"CFS_"+Vname(i)+"_2013-2022.JJA.nc","c")
   fout->$Vname(i)$ =data
   delete(data)
end do

