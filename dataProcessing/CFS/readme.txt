3.0generate_weight_file.ncl 
	a script to generate the wig file.

DownloadCFS.ncl
	the ncl script to download, convert format, trim, and calculate daily and monthly CFS data.
	the script have many argument which pass from commend line to ncl script.

downloadCFS_JJA.py
	a python script, same as in the notebook.
	call ncl script to download CFS data, and pass the different argument to ncl script by nested loops.

run_comet.sh
	a .sh file to submit the job on comet. 
