#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/prctl.h>
#include <unistd.h>
#include <arm_sve.h>
 
using cclock = std::chrono::high_resolution_clock;

inline void flop_0(int64_t size, double* data_a) // 2 flops - 8 bytes = 1/4 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];
       auto t0 = 3.0*a + 2.0;
       data_a[i] = t0;
    }
}

inline void flop_1(int64_t size, double* data_a) // 5 flops - 8 bytes = 5/8 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       t0 = t0 + t1; 

       data_a[i] = t0;
    }
}

inline void flop_2(int64_t size, double* data_a) // 11 flops - 8 bytes = 11/8 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;

       t0 = t0 + t1 + t2 + t3; 

       data_a[i] = t0;
    }
}

inline void flop_3(int64_t size, double* data_a) // 17 flops - 8 bytes = 2 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;

       t0 = t0 + t1 + t2 + t3 + t4 + t5; 

       data_a[i] = t0;
    }
}

inline void flop_4(int64_t size, double* data_a) // 35 flops - 8 bytes = 4.4 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       t0 = t3*t4 + t0;
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 
       t3 = t5*t4 + t3; 
       t4 = t7*t6 + t4; 
       t5 = t8*t6 + t5; 

       t0 = t0 + t1 + t2 + t3 + t4 + t5; 

       data_a[i] = t0;
    }
}

inline void flop_5(int64_t size, double* data_a) // 62 flops - 8 bytes = 7.75 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       t0 = t3*t4 + t0;
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 
       t3 = t5*t4 + t3; 
       t4 = t7*t6 + t4; 
       t5 = t8*t6 + t5; 
       t6 = 3.0*t7 + t6; 
       t7 = 1.3*t8 + t7; 
       t8 = 2.0*t8 + t8; 

       t0 = t3*t4 + t0;
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 
       t3 = t5*t4 + t3; 
       t4 = t7*t6 + t4; 
       t5 = t8*t6 + t5; 
       t6 = 3.0*t7 + t6; 
       t7 = 1.3*t8 + t7; 
       t8 = 2.0*t8 + t8; 

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

inline void flop_6(int64_t size, double* data_a) // 116 flops - 8 bytes = 14.5 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       for (unsigned j = 0; j<5; ++j) {
           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 
       }

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

inline void flop_7(int64_t size, double* data_a) // 242 flops - 8 bytes = 30.25 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       for (unsigned j = 0; j<12; ++j) {
           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 
       }

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

inline void flop_8(int64_t size, double* data_a) // 512 flops - 8 bytes = 64 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       for (unsigned j = 0; j<9; ++j) {
           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 
       }

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

inline void flop_9(int64_t size, double* data_a) // 1034 flops - 8 bytes = 129.25 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;
 
       for (unsigned j = 0; j<14; ++j) {
           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 
       }

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

inline void flop_10(int64_t size, double* data_a) //2168 flops - 8 bytes = 271 flops/byte 
{
    for (unsigned i = 0; i < size; i++) {
       auto a = data_a[i];

       auto t0 = 3.0*a + 2.0;
       auto t1 = 7.0*a + 1.3;
       auto t2 = 2.0*a + 3.0;
       auto t3 = 1.3*a + 7.0;
       auto t4 = 1.3*a + 3.0;
       auto t5 = 2.0*a + 7.0;
       auto t6 = 1.3*a + 2.0;
       auto t7 = 3.0*a + 7.0;
       auto t8 = 3.0*a + 1.3;

       for (unsigned j = 0; j<17; ++j) {
           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 

           t0 = t3*t4 + t0;
           t1 = t5*t6 + t1; 
           t2 = t7*t8 + t2; 
           t3 = t5*t4 + t3; 
           t4 = t7*t6 + t4; 
           t5 = t8*t6 + t5; 
           t6 = 3.0*t7 + t6; 
           t7 = 1.3*t8 + t7; 
           t8 = 2.0*t8 + t8; 
       }

       // REDUCE
       t0 = t3*t4 + t0; 
       t1 = t5*t6 + t1; 
       t2 = t7*t8 + t2; 

       t0 = t0 + t1 + t2; 

       data_a[i] = t0;
    }
}

template <typename L>
void run(L&& fn, double* a, unsigned size) {

    for(uint64_t i = 0; i < size; i++) {
        a[i] = (double)(i + 1)/(1e20*size);
    }

    usleep(1000000);  

    auto t0 = cclock::now();
    fn(size, a);
    auto t1 = cclock::now();
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::cout<< count << std::endl;
}

int main(int argc, char **argv) {
    unsigned SIZE = std::atoi(argv[1]);
    unsigned N    = std::atoi(argv[2]);
    std::cout << SIZE << " x " << N << std::endl;
    double *a;

    a   = (double *)malloc(sizeof(double) * SIZE);

    run(flop_0, a, SIZE);
    run(flop_1, a, SIZE);
    run(flop_2, a, SIZE);
    run(flop_3, a, SIZE);
    run(flop_4, a, SIZE);
    run(flop_5, a, SIZE);
    run(flop_6, a, SIZE);
    run(flop_7, a, SIZE);
    run(flop_8, a, SIZE);
    run(flop_9, a, SIZE);
    run(flop_10, a, SIZE);
    
    free(a);

    return 0;
}
