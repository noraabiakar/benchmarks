#g++ -O3 -funroll-loops -fno-tree-vectorize -march=skylake-avx512 compare_no_vec.cpp -o a.out
#g++ -O3 -funroll-loops                     -march=skylake-avx512 compare_no_vec.cpp -o a.out
g++ -O3 -funroll-loops                     -march=skylake-avx512 compare_avx512.cpp -o a.out

srun numactl --physcpubind=0 --membind=0 ./a.out 8000000 1
srun numactl --physcpubind=0 --membind=0 ./a.out 8000000 1
srun numactl --physcpubind=0 --membind=0 ./a.out 8000000 1
srun numactl --physcpubind=0 --membind=0 ./a.out 8000000 1
rm a.out
echo "DONE"

