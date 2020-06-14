#!/bin/bash
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=q48node" # Queue name
#PJM –L "elapse=60" # Elapse time in seconds (there are other time formats available)
#PJM -L "node=1" # Number of nodes (up to 48)

export BASEDIR=/opt/FJSVxtclanga/tcsds-1.2.25
export TCS=${BASEDIR}/bin
export PATH=$TCS:$PATH
export PATH=/home/users/scs/scs-4/install/cmake-3.16.1/install/cmake/bin:$PATH
export LD_LIBRARY_PATH=${BASEDIR}/lib64:$LD_LIBRARY_PATH

cd ${PJM_O_WORKDIR} # Change directory to the working directory

FCC -Nclang -O3 -funroll-loops -fno-vectorize -march=armv8.2-a+sve compare_sve_no_vec.cpp -o a.out
numactl --physcpubind=12 --membind=4 ./a.out 8000000 1
numactl --physcpubind=12 --membind=4 ./a.out 8000000 1
numactl --physcpubind=12 --membind=4 ./a.out 8000000 1
numactl --physcpubind=12 --membind=4 ./a.out 8000000 1
rm a.out
echo "DONE"

