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
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);
       auto t0 = svmad_f64_z(pg, three, a, two);
       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_1(int64_t size, double* data_a) // 5 flops - 8 bytes = 5/8 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       t0 = svadd_f64_z(pg, t0, t1);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_2(int64_t size, double* data_a) // 11 flops - 8 bytes = 11/8 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);

       t0 = svadd_f64_z(pg, t0, t1);
       t2 = svadd_f64_z(pg, t2, t3);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_3(int64_t size, double* data_a) // 17 flops - 8 bytes = 2 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);

       t0 = svadd_f64_z(pg, t0, t1);
       t2 = svadd_f64_z(pg, t2, t3);
       t4 = svadd_f64_z(pg, t4, t5);
       t0 = svadd_f64_z(pg, t0, t2);
       t0 = svadd_f64_z(pg, t0, t4);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_4(int64_t size, double* data_a) // 35 flops - 8 bytes = 4.4 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 
       t3 = svmad_f64_z(pg, t5, t4, t3); 
       t4 = svmad_f64_z(pg, t7, t6, t4); 
       t5 = svmad_f64_z(pg, t8, t6, t5); 

       t0 = svadd_f64_z(pg, t0, t1);
       t2 = svadd_f64_z(pg, t2, t3);
       t4 = svadd_f64_z(pg, t4, t5);
       t0 = svadd_f64_z(pg, t0, t2);
       t0 = svadd_f64_z(pg, t0, t4);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_5(int64_t size, double* data_a) // 62 flops - 8 bytes = 7.75 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       t0 = svmad_f64_z(pg, t3,    t4, t0); 
       t1 = svmad_f64_z(pg, t5,    t6, t1); 
       t2 = svmad_f64_z(pg, t7,    t8, t2); 
       t3 = svmad_f64_z(pg, t5,    t4, t3); 
       t4 = svmad_f64_z(pg, t7,    t6, t4); 
       t5 = svmad_f64_z(pg, t8,    t6, t5); 
       t6 = svmad_f64_z(pg, three, t7, t6); 
       t7 = svmad_f64_z(pg, one,   t8, t7); 
       t8 = svmad_f64_z(pg, two,   t8, t8); 

       t0 = svmad_f64_z(pg, t3,    t4, t0); 
       t1 = svmad_f64_z(pg, t5,    t6, t1); 
       t2 = svmad_f64_z(pg, t7,    t8, t2); 
       t3 = svmad_f64_z(pg, t5,    t4, t3); 
       t4 = svmad_f64_z(pg, t7,    t6, t4); 
       t5 = svmad_f64_z(pg, t8,    t6, t5); 
       t6 = svmad_f64_z(pg, three, t7, t6); 
       t7 = svmad_f64_z(pg, one,   t8, t7); 
       t8 = svmad_f64_z(pg, two,   t8, t8); 

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_6(int64_t size, double* data_a) // 116 flops - 8 bytes = 14.5 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       for (unsigned j = 0; j < 5; j++) {
           t0 = svmad_f64_z(pg, t3,    t4, t0); 
           t1 = svmad_f64_z(pg, t5,    t6, t1); 
           t2 = svmad_f64_z(pg, t7,    t8, t2); 
           t3 = svmad_f64_z(pg, t5,    t4, t3); 
           t4 = svmad_f64_z(pg, t7,    t6, t4); 
           t5 = svmad_f64_z(pg, t8,    t6, t5); 
           t6 = svmad_f64_z(pg, three, t7, t6); 
           t7 = svmad_f64_z(pg, one,   t8, t7); 
           t8 = svmad_f64_z(pg, two,   t8, t8); 
       }

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_7(int64_t size, double* data_a) // 242 flops - 8 bytes = 30.25 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       asm volatile ("starting_now:");
       for (unsigned j = 0; j < 12; j++) {
           t0 = svmad_f64_z(pg, t3,    t4, t0); 
           t1 = svmad_f64_z(pg, t5,    t6, t1); 
           t2 = svmad_f64_z(pg, t7,    t8, t2); 
           t3 = svmad_f64_z(pg, t5,    t4, t3); 
           t4 = svmad_f64_z(pg, t7,    t6, t4); 
           t5 = svmad_f64_z(pg, t8,    t6, t5); 
           t6 = svmad_f64_z(pg, three, t7, t6); 
           t7 = svmad_f64_z(pg, one,   t8, t7); 
           t8 = svmad_f64_z(pg, two,   t8, t8); 
       }
       asm volatile ("ending_now:");

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_8(int64_t size, double* data_a) // 512 flops - 8 bytes = 64 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       for (unsigned j = 0; j < 27; j++) {
           t0 = svmad_f64_z(pg, t3,    t4, t0); 
           t1 = svmad_f64_z(pg, t5,    t6, t1); 
           t2 = svmad_f64_z(pg, t7,    t8, t2); 
           t3 = svmad_f64_z(pg, t5,    t4, t3); 
           t4 = svmad_f64_z(pg, t7,    t6, t4); 
           t5 = svmad_f64_z(pg, t8,    t6, t5); 
           t6 = svmad_f64_z(pg, three, t7, t6); 
           t7 = svmad_f64_z(pg, one,   t8, t7); 
           t8 = svmad_f64_z(pg, two,   t8, t8); 
       }

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_9(int64_t size, double* data_a) // 1034 flops - 8 bytes = 129.25 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);
 
       for (unsigned j = 0; j < 56; j++) {
           t0 = svmad_f64_z(pg, t3,    t4, t0); 
           t1 = svmad_f64_z(pg, t5,    t6, t1); 
           t2 = svmad_f64_z(pg, t7,    t8, t2); 
           t3 = svmad_f64_z(pg, t5,    t4, t3); 
           t4 = svmad_f64_z(pg, t7,    t6, t4); 
           t5 = svmad_f64_z(pg, t8,    t6, t5); 
           t6 = svmad_f64_z(pg, three, t7, t6); 
           t7 = svmad_f64_z(pg, one,   t8, t7); 
           t8 = svmad_f64_z(pg, two,   t8, t8); 
       }

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
    }
}

inline void flop_10(int64_t size, double* data_a) //2168 flops - 8 bytes = 271 flops/byte 
{
    auto pg    = svptrue_b64();
    auto three = svdup_n_f64(3.0);
    auto two   = svdup_n_f64(2.0);
    auto seven = svdup_n_f64(7.0);
    auto one   = svdup_n_f64(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(pg, data_a+i);

       auto t0 = svmad_f64_z(pg, three, a, two);
       auto t1 = svmad_f64_z(pg, seven, a, one);
       auto t2 = svmad_f64_z(pg, two, a, three);
       auto t3 = svmad_f64_z(pg, one, a, seven);
       auto t4 = svmad_f64_z(pg, one, a, three);
       auto t5 = svmad_f64_z(pg, two, a, seven);
       auto t6 = svmad_f64_z(pg, one, a, two);
       auto t7 = svmad_f64_z(pg, three, a, seven);
       auto t8 = svmad_f64_z(pg, three, a, one);

       for (unsigned j = 0; j < 119; j++) {
           t0 = svmad_f64_z(pg, t3,    t4, t0); 
           t1 = svmad_f64_z(pg, t5,    t6, t1); 
           t2 = svmad_f64_z(pg, t7,    t8, t2); 
           t3 = svmad_f64_z(pg, t5,    t4, t3); 
           t4 = svmad_f64_z(pg, t7,    t6, t4); 
           t5 = svmad_f64_z(pg, t8,    t6, t5); 
           t6 = svmad_f64_z(pg, three, t7, t6); 
           t7 = svmad_f64_z(pg, one,   t8, t7); 
           t8 = svmad_f64_z(pg, two,   t8, t8); 
       }

       // REDUCE
       t0 = svmad_f64_z(pg, t3, t4, t0); 
       t1 = svmad_f64_z(pg, t5, t6, t1); 
       t2 = svmad_f64_z(pg, t7, t8, t2); 

       t0 = svadd_f64_z(pg, t0, t1);
       t0 = svadd_f64_z(pg, t0, t2);

       svst1_f64(pg,  data_a+i, t0);
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
