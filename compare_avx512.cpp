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

inline void compute_v(int64_t size, double* data_a)
{
    auto three = _mm512_set1_pd(1/30.);
    auto two   = _mm512_set1_pd(1/20.);
    auto seven = _mm512_set1_pd(1/70.);

    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);
       auto b = _mm512_fmadd_pd(three, a, two); 
       auto c = _mm512_fmadd_pd(two, a, three); 
       auto d = _mm512_fmadd_pd(seven, a, three); //6 flops 

       // fmadd: res = (a * b) + c
       // t = (a)a + (a)b                     -- 3  t0
       //   + (a)c + (a)d                     -- 4  t1
       //   + (b)b + (b)c                     -- 4  t2   
       //   + (b)d + (c)c                     -- 4  t3
       //   + (c)d + (d)d                     -- 4  t4
       //   + (aa)a + (aa)b + (aa)c + (aa)d   -- 8  t0
       //   + (bb)a + (bb)b + (bb)c + (bb)d   -- 8  t1
       //   + (cc)a + (cc)b + (cc)c + (cc)d   -- 8  t2
       //   + (dd)a + (dd)b + (dd)c + (dd)d   -- 8  t3
       //   + (ab)a + (ab)b + (ab)c + (ab)d   -- 8  t4
       //   + (bc)a + (bc)b + (bc)c + (bc)d   -- 8  t0
       //   + (cd)a + (cd)b + (cd)c + (cd)d   -- 8  t1
       //   + (ac)a + (ac)b + (ac)c + (ac)d   -- 8  t2                                  
       //   + (bd)a + (db)b + (db)c + (db)d   -- 8  t3                                   
       //   + (ad)a + (ad)b + (ad)c + (ad)d   -- 8  t4                                   

       auto aa = _mm512_mul_pd( a, a);
       auto ad = _mm512_mul_pd( a, d);
       auto bc = _mm512_mul_pd( b, c);
       auto cc = _mm512_mul_pd( c, c);
       auto dd = _mm512_mul_pd( d, d);  // 10 flops
       
       auto t0 = _mm512_fmadd_pd( a, b, aa); 
       auto t1 = _mm512_fmadd_pd( a, c, ad);
       auto t2 = _mm512_fmadd_pd( b, b, bc);
       auto t3 = _mm512_fmadd_pd( b, d, cc);
       auto t4 = _mm512_fmadd_pd( c, d, dd); // 5 flops

       t0 = _mm512_fmadd_pd( aa, a, t0);
       t0 = _mm512_fmadd_pd( aa, b, t0);
       t0 = _mm512_fmadd_pd( aa, c, t0);
       t0 = _mm512_fmadd_pd( aa, d, t0);   // 8 flops

       auto bb = _mm512_mul_pd( b, b);
       t1 = _mm512_fmadd_pd( bb, a, t1);
       t1 = _mm512_fmadd_pd( bb, b, t1);
       t1 = _mm512_fmadd_pd( bb, c, t1);
       t1 = _mm512_fmadd_pd( bb, d, t1);  // 9 flops

       t2 = _mm512_fmadd_pd( cc, a, t2);
       t2 = _mm512_fmadd_pd( cc, b, t2);
       t2 = _mm512_fmadd_pd( cc, c, t2);
       t2 = _mm512_fmadd_pd( cc, d, t2);  // 8 flops

       t3 = _mm512_fmadd_pd( dd, a, t3);
       t3 = _mm512_fmadd_pd( dd, b, t3);
       t3 = _mm512_fmadd_pd( dd, c, t3);
       t3 = _mm512_fmadd_pd( dd, d, t3);  // 8 flops

       auto ab = _mm512_mul_pd( a, b);
       t4 = _mm512_fmadd_pd( ab, a, t4);
       t4 = _mm512_fmadd_pd( ab, b, t4);
       t4 = _mm512_fmadd_pd( ab, c, t4);
       t4 = _mm512_fmadd_pd( ab, d, t4);  // 9 flops

       t0 = _mm512_fmadd_pd( bc, a, t0);
       t0 = _mm512_fmadd_pd( bc, b, t0);
       t0 = _mm512_fmadd_pd( bc, c, t0);
       t0 = _mm512_fmadd_pd( bc, d, t0);  // 8 flops

       auto cd = _mm512_mul_pd( c, d);
       t1 = _mm512_fmadd_pd( cd, a, t1);
       t1 = _mm512_fmadd_pd( cd, b, t1);
       t1 = _mm512_fmadd_pd( cd, c, t1);
       t1 = _mm512_fmadd_pd( cd, d, t1);  // 9 flops

       auto ac = _mm512_mul_pd( a, c);
       t2 = _mm512_fmadd_pd( ac, a, t2);
       t2 = _mm512_fmadd_pd( ac, b, t2);
       t2 = _mm512_fmadd_pd( ac, c, t2);
       t2 = _mm512_fmadd_pd( ac, d, t2);  // 9 flops

       auto bd = _mm512_mul_pd( b, d);
       t3 = _mm512_fmadd_pd( bd, a, t3);
       t3 = _mm512_fmadd_pd( bd, b, t3);
       t3 = _mm512_fmadd_pd( bd, c, t3);
       t3 = _mm512_fmadd_pd( bd, d, t3);  // 9 flops

       t4 = _mm512_fmadd_pd( ad, a, t4);
       t4 = _mm512_fmadd_pd( ad, b, t4);
       t4 = _mm512_fmadd_pd( ad, c, t4);
       t4 = _mm512_fmadd_pd( ad, d, t4);  // 8 flops

       t1 = _mm512_add_pd( t1, t2);
       t3 = _mm512_add_pd( t3, t4);
       t0 = _mm512_add_pd( t0, t1);
       t0 = _mm512_add_pd( t0, t3);     // 4 flops

       //-----------------------------    110 flops
        
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
    
    double sum = 0;
    {
        auto t0 = cclock::now();
        for (int i = 0; i < N; i++) {
            compute_v(SIZE, a);
            sum += a[i%SIZE];
        }
        auto t1 = cclock::now();
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout<<"compute_v takes: \t"<< count << "\t" << sum << std::endl;
    }
    
    free(a);

    return 0;
}
