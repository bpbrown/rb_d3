mpiexec_mpt -n 1024 python3 $1 --Nz=64 --run_time_iter=$2
mpiexec_mpt -n 512 python3 $1 --Nz=64 --run_time_iter=$2
mpiexec_mpt -n 256 python3 $1 --Nz=64 --run_time_iter=$2
mpiexec_mpt -n 128 python3 $1 --Nz=64 --run_time_iter=$2
mpiexec_mpt -n 64 python3 $1 --Nz=64 --run_time_iter=$2


mpiexec_mpt -n 4096 python3 $1 --Nz=128 --run_time_iter=$2
mpiexec_mpt -n 2048 python3 $1 --Nz=128 --run_time_iter=$2
mpiexec_mpt -n 1024 python3 $1 --Nz=128 --run_time_iter=$2
mpiexec_mpt -n 512 python3 $1 --Nz=128 --run_time_iter=$2
mpiexec_mpt -n 256 python3 $1 --Nz=128 --run_time_iter=$2
mpiexec_mpt -n 128 python3 $1 --Nz=128 --run_time_iter=$2

mpiexec_mpt -n 4096 python3 $1 --Nz=256 --run_time_iter=$2
mpiexec_mpt -n 2048 python3 $1 --Nz=256 --run_time_iter=$2
mpiexec_mpt -n 1024 python3 $1 --Nz=256 --run_time_iter=$2
mpiexec_mpt -n 512 python3 $1 --Nz=256 --run_time_iter=$2
#mpiexec_mpt -n 256 python3 $1 --Nz=256 --run_time_iter=$2

mpiexec_mpt -n 4096 python3 $1 --Nz=512 --run_time_iter=$2
