#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/prctl.h>
#include <unistd.h>
#include <immintrin.h>
 
using cclock = std::chrono::high_resolution_clock;

inline void flop_0(int64_t size, double* data_a) // 2 flops - 8 bytes = 1/4 flops/byte 
{
    auto three = _mm512_set1_pd(3.0);
    auto two   = _mm512_set1_pd(2.0);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);
       auto t0 = _mm512_div_pd(three, _mm512_add_pd(a, two));
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

       auto t0 = _mm512_div_pd(three, _mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven, _mm512_add_pd(a, one));
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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,_mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,_mm512_add_pd(a, seven));
                                      
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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 
       t3 = _mm512_div_pd(t5,_mm512_add_pd(t4, t3)); 
       t4 = _mm512_div_pd(t7,_mm512_add_pd(t6, t4)); 
       t5 = _mm512_div_pd(t8,_mm512_add_pd(t6, t5)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
       t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
       t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
       t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
       t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
       t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
       t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 

       t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
       t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
       t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
       t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
       t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
       t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
       t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       for (unsigned j = 0; j < 5; j++) {
           t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
           t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
           t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
           t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
           t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
           t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
           t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
           t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
           t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 
       }

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       for (unsigned j = 0; j < 12; j++) {
           t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
           t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
           t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
           t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
           t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
           t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
           t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
           t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
           t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 
       }

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       for (unsigned j = 0; j < 27; j++) {
           t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
           t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
           t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
           t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
           t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
           t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
           t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
           t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
           t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 
       }

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));
 
       for (unsigned j = 0; j < 56; j++) {
           t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
           t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
           t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
           t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
           t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
           t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
           t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
           t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
           t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 
       }

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

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

       auto t0 = _mm512_div_pd(three,_mm512_add_pd(a, two));
       auto t1 = _mm512_div_pd(seven,_mm512_add_pd(a, one));
       auto t2 = _mm512_div_pd(two,  _mm512_add_pd(a, three));
       auto t3 = _mm512_div_pd(one,  _mm512_add_pd(a, seven));
       auto t4 = _mm512_div_pd(one,  _mm512_add_pd(a, three));
       auto t5 = _mm512_div_pd(two,  _mm512_add_pd(a, seven));
       auto t6 = _mm512_div_pd(one,  _mm512_add_pd(a, two));
       auto t7 = _mm512_div_pd(three,_mm512_add_pd(a, seven));
       auto t8 = _mm512_div_pd(three,_mm512_add_pd(a, one));

       for (unsigned j = 0; j < 119; j++) {
           t0 = _mm512_div_pd(t3,   _mm512_add_pd(t4, t0)); 
           t1 = _mm512_div_pd(t5,   _mm512_add_pd(t6, t1)); 
           t2 = _mm512_div_pd(t7,   _mm512_add_pd(t8, t2)); 
           t3 = _mm512_div_pd(t5,   _mm512_add_pd(t4, t3)); 
           t4 = _mm512_div_pd(t7,   _mm512_add_pd(t6, t4)); 
           t5 = _mm512_div_pd(t8,   _mm512_add_pd(t6, t5)); 
           t6 = _mm512_div_pd(three,_mm512_add_pd(t7, t6)); 
           t7 = _mm512_div_pd(one,  _mm512_add_pd(t8, t7)); 
           t8 = _mm512_div_pd(two,  _mm512_add_pd(t8, t8)); 
       }

       // REDUCE
       t0 = _mm512_div_pd(t3,_mm512_add_pd(t4, t0)); 
       t1 = _mm512_div_pd(t5,_mm512_add_pd(t6, t1)); 
       t2 = _mm512_div_pd(t7,_mm512_add_pd(t8, t2)); 

       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);

       _mm512_storeu_pd( data_a+i, t0);
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
