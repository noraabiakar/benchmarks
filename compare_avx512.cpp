#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/prctl.h>
#include <immintrin.h>
 
using cclock = std::chrono::high_resolution_clock;

inline void flop_0(int64_t size, double* data_a) // 2 flops - 8 bytes = 1/4 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);
       auto t0 = _mm512_fmadd_pd(three, a, two);
       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_1(int64_t size, double* data_a) // 5 flops - 8 bytes = 5/8 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       t0 = _mm512_add_pd(t0, t1);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_2(int64_t size, double* data_a) // 11 flops - 8 bytes = 11/8 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);

       t0 = _mm512_add_pd(t0, t1);
       t2 = _mm512_add_pd(t2, t3);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_3(int64_t size, double* data_a) // 17 flops - 8 bytes = 2 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);

       t0 = _mm512_add_pd(t0, t1);
       t2 = _mm512_add_pd(t2, t3);
       t4 = _mm512_add_pd(t4, t5);
       t0 = _mm512_add_pd(t0, t2);
       t0 = _mm512_add_pd(t0, t4);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_4(int64_t size, double* data_a) // 35 flops - 8 bytes = 4.4 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 
       t3 = _mm512_fmadd_pd(t5, t4, t3); 
       t4 = _mm512_fmadd_pd(t7, t6, t4); 
       t5 = _mm512_fmadd_pd(t8, t6, t5); 

       t0 = _mm512_add_pd(t0, t1);
       t2 = _mm512_add_pd(t2, t3);
       t4 = _mm512_add_pd(t4, t5);
       t0 = _mm512_add_pd(t0, t2);
       t0 = _mm512_add_pd(t0, t4);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_5(int64_t size, double* data_a) // 62 flops - 8 bytes = 7.75 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_6(int64_t size, double* data_a) // 116 flops - 8 bytes = 14.5 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 


       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_7(int64_t size, double* data_a) // 242 flops - 8 bytes = 30.25 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_8(int64_t size, double* data_a) // 512 flops - 8 bytes = 64 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 


       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_9(int64_t size, double* data_a) // 1034 flops - 8 bytes = 129.25 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);
 
       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}

inline void flop_10(int64_t size, double* data_a) //2168 flops - 8 bytes = 271 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);
    auto seven = _mm512_set1_pd(7.0);
    auto one   = _mm512_set1_pd(1.3);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);

       auto t0 = _mm512_fmadd_pd(three, a, two);
       auto t1 = _mm512_fmadd_pd(seven, a, one);
       auto t2 = _mm512_fmadd_pd(two, a, three);
       auto t3 = _mm512_fmadd_pd(one, a, seven);
       auto t4 = _mm512_fmadd_pd(one, a, three);
       auto t5 = _mm512_fmadd_pd(two, a, seven);
       auto t6 = _mm512_fmadd_pd(one, a, two);
       auto t7 = _mm512_fmadd_pd(three, a, seven);
       auto t8 = _mm512_fmadd_pd(three, a, one);

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);   // 10 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 20

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 30

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 40

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 50

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 
 
       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 60 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 70

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 80

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 90

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); // 100

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 110 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8); 

       t0 = _mm512_fmadd_pd(t3,    t4, t0); 
       t1 = _mm512_fmadd_pd(t5,    t6, t1); 
       t2 = _mm512_fmadd_pd(t7,    t8, t2); 
       t3 = _mm512_fmadd_pd(t5,    t4, t3); 
       t4 = _mm512_fmadd_pd(t7,    t6, t4); 
       t5 = _mm512_fmadd_pd(t8,    t6, t5); 
       t6 = _mm512_fmadd_pd(three, t7, t6); 
       t7 = _mm512_fmadd_pd(one,   t8, t7); 
       t8 = _mm512_fmadd_pd(two,   t8, t8);  // 120 

       // REDUCE
       t0 = _mm512_fmadd_pd(t3, t4, t0); 
       t1 = _mm512_fmadd_pd(t5, t6, t1); 
       t2 = _mm512_fmadd_pd(t7, t8, t2); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
    }
}


int main(int argc, char **argv) {
    unsigned SIZE = std::atoi(argv[1]);
    unsigned N    = std::atoi(argv[2]);
    std::cout << SIZE << " x " << N << std::endl;
    double *a;

    a   = (double *)malloc(sizeof(double) * SIZE);

    for(uint64_t i = 0; i < SIZE; i++) {
        a[i] = (double)(i + 1)/(1e20*SIZE);
    }
    
    {
        auto t0 = cclock::now();
        flop_0(SIZE, a);
        auto t1 = cclock::now();
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout<< count << std::endl;
    }
    
    free(a);

    return 0;
}
