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

inline void compute_v(int64_t size, double* data_a, double* data_b, double* data_c, double* data_d)
{
    for (unsigned i = 0; i < size; i+=8) {
       auto a = _mm512_loadu_pd(data_a+i);
       auto b = _mm512_loadu_pd(data_b+i);
       auto c = _mm512_loadu_pd(data_c+i);
       auto d = _mm512_loadu_pd(data_d+i);

       // fmadd: res = (a * b) + c
       // t = a + b + c + d                  -- 3
       //  + (a)a + (a)b + (a)c + (a)d       -- 8
       //  + (b)b + (b)c + (b)d              -- 6
       //  + (c)c + (c)d                     -- 4
       //  + (d)d                            -- 2
       //  + (aa)a + (aa)b + (aa)c + (aa)d   -- 8
       //  + (bb)a + (bb)b + (bb)c + (bb)d   -- 8
       //  + (cc)a + (cc)b + (cc)c + (cc)d   -- 8
       //  + (dd)a + (dd)b + (dd)c + (dd)d   -- 8
       //  + (ab)a + (ab)b + (ab)c + (ab)d   -- 8
       //  + (bc)a + (bc)b + (bc)c + (bc)d   -- 8
       //  + (cd)a + (cd)b + (cd)c + (cd)d   -- 8
       //  + (ac)a + (ac)b + (ac)c + (ac)d   -- 8                                     
       //  + (bd)a + (db)b + (db)c + (db)d   -- 8                                     
       //  + (ad)a + (ad)b + (ad)c + (ad)d   -- 8                                     

       auto t0 = _mm512_add_pd(a, b); 
       auto t1 = _mm512_add_pd(c, d); 
       t0 = _mm512_add_pd(t0, t1);      // 3 flops

       auto aa = _mm512_mul_pd(a, a);
       auto ab = _mm512_mul_pd(a, b);
       auto ac = _mm512_mul_pd(a, c);
       auto ad = _mm512_mul_pd(a, d);
       auto bb = _mm512_mul_pd(b, b);
       auto bc = _mm512_mul_pd(b, c);
       auto bd = _mm512_mul_pd(b, d);
       auto cc = _mm512_mul_pd(c, c);
       auto cd = _mm512_mul_pd(c, d);
       auto dd = _mm512_mul_pd(d, d);  // 10 flops

       t1 = _mm512_add_pd(aa, ab);
       auto t2 = _mm512_add_pd(ac, ad);
       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);      // 4 flops

       t2 = _mm512_add_pd(bb, bc);
       t1 = _mm512_add_pd(bd, cc);
       t0 = _mm512_add_pd(t0, t1);
       t0 = _mm512_add_pd(t0, t2);      // 4 flops

       t1 = _mm512_add_pd(cd, dd);
       t0 = _mm512_add_pd(t0, t1);      // 2 flops
                                         
       t0 = _mm512_fmadd_pd(aa, a, t0);
       t0 = _mm512_fmadd_pd(aa, b, t0);
       t0 = _mm512_fmadd_pd(aa, c, t0);
       t0 = _mm512_fmadd_pd(aa, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(bb, a, t0);
       t0 = _mm512_fmadd_pd(bb, b, t0);
       t0 = _mm512_fmadd_pd(bb, c, t0);
       t0 = _mm512_fmadd_pd(bb, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(cc, a, t0);
       t0 = _mm512_fmadd_pd(cc, b, t0);
       t0 = _mm512_fmadd_pd(cc, c, t0);
       t0 = _mm512_fmadd_pd(cc, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(dd, a, t0);
       t0 = _mm512_fmadd_pd(dd, b, t0);
       t0 = _mm512_fmadd_pd(dd, c, t0);
       t0 = _mm512_fmadd_pd(dd, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(ab, a, t0);
       t0 = _mm512_fmadd_pd(ab, b, t0);
       t0 = _mm512_fmadd_pd(ab, c, t0);
       t0 = _mm512_fmadd_pd(ab, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(bc, a, t0);
       t0 = _mm512_fmadd_pd(bc, b, t0);
       t0 = _mm512_fmadd_pd(bc, c, t0);
       t0 = _mm512_fmadd_pd(bc, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(cd, a, t0);
       t0 = _mm512_fmadd_pd(cd, b, t0);
       t0 = _mm512_fmadd_pd(cd, c, t0);
       t0 = _mm512_fmadd_pd(cd, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(ac, a, t0);
       t0 = _mm512_fmadd_pd(ac, b, t0);
       t0 = _mm512_fmadd_pd(ac, c, t0);
       t0 = _mm512_fmadd_pd(ac, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(bd, a, t0);
       t0 = _mm512_fmadd_pd(bd, b, t0);
       t0 = _mm512_fmadd_pd(bd, c, t0);
       t0 = _mm512_fmadd_pd(bd, d, t0);  // 8 flops

       t0 = _mm512_fmadd_pd(ad, a, t0);
       t0 = _mm512_fmadd_pd(ad, b, t0);
       t0 = _mm512_fmadd_pd(ad, c, t0);
       t0 = _mm512_fmadd_pd(ad, d, t0);  // 8 flops
       //-----------------------------    103 flops
       
       _mm512_storeu_pd(data_d+i, t0);
    }
}

int main(int argc, char **argv) {
    unsigned SIZE = std::atoi(argv[1]);
    unsigned N    = std::atoi(argv[2]);
    std::cout << SIZE << " x " << N << std::endl;
    double *a, *b, *c, *out;

    a   = (double *)malloc(sizeof(double) * SIZE);
    b   = (double *)malloc(sizeof(double) * SIZE);
    c   = (double *)malloc(sizeof(double) * SIZE);
    out = (double *)malloc(sizeof(double) * SIZE);

    for(uint64_t i = 0; i < SIZE; i++) {
        a[i] = (double)(i + 1)/(1e20*SIZE);
        b[i] = (double)(i + 2)/(1e20*SIZE);
        c[i] = (double)(i + 3)/(1e20*SIZE);
        out[i] = (double)(i + 4)/(1e20*SIZE);
    }
    
    double sum = 0;
    {
        auto t0 = cclock::now();
        for (int i = 0; i < N; i++) {
            compute_v(SIZE, a, b, c, out);
            sum += out[i%SIZE];
        }
        auto t1 = cclock::now();
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::cout<<"compute_v takes: \t"<< count << "\t" << sum << std::endl;
    }
    
    free(a);
    free(b);
    free(c);
    free(out);

    return 0;
}
