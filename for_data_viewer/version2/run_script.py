import subprocess
from   mpi4py                  import MPI
comm              = MPI.COMM_WORLD
rank,size         = comm.Get_rank(), comm.Get_size()

path_operational = '/mnt/gfs01/PUB/S2S/V2023-07/V0_hindcast/'
days_per_month = { 1: [ 1, 6,11,16,21,26,31], 2: [ 5,10,15,20,25],3: [ 2, 7,12,17,22,27],4: [ 1, 6,11,16,21,26],5: [ 1, 6,11,16,21,26,31],6: [ 5,10,15,20,25,30],7: [ 5,10,15,20,25,30],8: [ 4, 9,14,19,24,29],9: [ 3, 8,13,18,23,28],10:[ 3, 8,13,18,23,28],11:[ 2, 7,12,17,22,27],12:[ 2, 7,12,17,22,27]}
# months         = range(1,12+1)
months         = range(1,12+1)
init_years     = range(2012,2023+1)
vnames         = ['T2MAX','T2MIN']
combination    = [ [ vname,year,month, day] for month in  months for day in days_per_month[month]  for year in init_years   for vname in vnames  ]

# # Define the parameters
for comb in combination[rank::size]:
    vname, init_year,init_month,init_day = comb
    print(vname, init_month,init_day)
    # Create a command to run the original script with the parameters
    command = [
        'python', 'cwrf_pdf_dw2.py',
        '--path_operational', path_operational,
        '--vname', vname,
        '--init_year', str(init_year),
        '--init_month', str(init_month),
        '--init_day', str(init_day)
    ]
    # Run the command
    subprocess.run(command)

