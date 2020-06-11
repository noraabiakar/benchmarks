#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/prctl.h>
#include <arm_sve.h>
 
using cclock = std::chrono::high_resolution_clock;

inline void compute_v(int64_t size, double* data_a, double* data_b, double* data_c, double* data_d)
{
    for (unsigned i = 0; i < size; i+=8) {
       auto a = svld1_f64(svptrue_b64(),data_a+i);
       auto b = svld1_f64(svptrue_b64(),data_b+i);
       auto c = svld1_f64(svptrue_b64(),data_c+i);
       auto d = svld1_f64(svptrue_b64(),data_d+i);

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

       auto aa = svmul_f64_z(svptrue_b64(), a, a);
       auto ad = svmul_f64_z(svptrue_b64(), a, d);
       auto bc = svmul_f64_z(svptrue_b64(), b, c);
       auto cc = svmul_f64_z(svptrue_b64(), c, c);
       auto dd = svmul_f64_z(svptrue_b64(), d, d);  // 5 flops
       
       auto t0 = svmad_f64_z(svptrue_b64(), a, b, aa); 
       auto t1 = svmad_f64_z(svptrue_b64(), a, c, ad);
       auto t2 = svmad_f64_z(svptrue_b64(), b, b, bc);
       auto t3 = svmad_f64_z(svptrue_b64(), b, d, cc);
       auto t4 = svmad_f64_z(svptrue_b64(), c, d, dd); // 10 flops

       t0 = svmad_f64_z(svptrue_b64(), aa, a, t0);
       t0 = svmad_f64_z(svptrue_b64(), aa, b, t0);
       t0 = svmad_f64_z(svptrue_b64(), aa, c, t0);
       t0 = svmad_f64_z(svptrue_b64(), aa, d, t0);   // 8 flops

       auto bb = svmul_f64_z(svptrue_b64(), b, b);
       t1 = svmad_f64_z(svptrue_b64(), bb, a, t1);
       t1 = svmad_f64_z(svptrue_b64(), bb, b, t1);
       t1 = svmad_f64_z(svptrue_b64(), bb, c, t1);
       t1 = svmad_f64_z(svptrue_b64(), bb, d, t1);  // 9 flops

       t2 = svmad_f64_z(svptrue_b64(), cc, a, t2);
       t2 = svmad_f64_z(svptrue_b64(), cc, b, t2);
       t2 = svmad_f64_z(svptrue_b64(), cc, c, t2);
       t2 = svmad_f64_z(svptrue_b64(), cc, d, t2);  // 8 flops

       t3 = svmad_f64_z(svptrue_b64(), dd, a, t3);
       t3 = svmad_f64_z(svptrue_b64(), dd, b, t3);
       t3 = svmad_f64_z(svptrue_b64(), dd, c, t3);
       t3 = svmad_f64_z(svptrue_b64(), dd, d, t3);  // 8 flops

       auto ab = svmul_f64_z(svptrue_b64(), a, b);
       t4 = svmad_f64_z(svptrue_b64(), ab, a, t4);
       t4 = svmad_f64_z(svptrue_b64(), ab, b, t4);
       t4 = svmad_f64_z(svptrue_b64(), ab, c, t4);
       t4 = svmad_f64_z(svptrue_b64(), ab, d, t4);  // 9 flops

       t0 = svmad_f64_z(svptrue_b64(), bc, a, t0);
       t0 = svmad_f64_z(svptrue_b64(), bc, b, t0);
       t0 = svmad_f64_z(svptrue_b64(), bc, c, t0);
       t0 = svmad_f64_z(svptrue_b64(), bc, d, t0);  // 8 flops

       auto cd = svmul_f64_z(svptrue_b64(), c, d);
       t1 = svmad_f64_z(svptrue_b64(), cd, a, t1);
       t1 = svmad_f64_z(svptrue_b64(), cd, b, t1);
       t1 = svmad_f64_z(svptrue_b64(), cd, c, t1);
       t1 = svmad_f64_z(svptrue_b64(), cd, d, t1);  // 9 flops

       auto ac = svmul_f64_z(svptrue_b64(), a, c);
       t2 = svmad_f64_z(svptrue_b64(), ac, a, t2);
       t2 = svmad_f64_z(svptrue_b64(), ac, b, t2);
       t2 = svmad_f64_z(svptrue_b64(), ac, c, t2);
       t2 = svmad_f64_z(svptrue_b64(), ac, d, t2);  // 9 flops

       auto bd = svmul_f64_z(svptrue_b64(), b, d);
       t3 = svmad_f64_z(svptrue_b64(), bd, a, t3);
       t3 = svmad_f64_z(svptrue_b64(), bd, b, t3);
       t3 = svmad_f64_z(svptrue_b64(), bd, c, t3);
       t3 = svmad_f64_z(svptrue_b64(), bd, d, t3);  // 8 flops

       t4 = svmad_f64_z(svptrue_b64(), ad, a, t4);
       t4 = svmad_f64_z(svptrue_b64(), ad, b, t4);
       t4 = svmad_f64_z(svptrue_b64(), ad, c, t4);
       t4 = svmad_f64_z(svptrue_b64(), ad, d, t4);  // 8 flops

       t1 = svadd_f64_z(svptrue_b64(), t1, t2);
       t3 = svadd_f64_z(svptrue_b64(), t3, t4);
       t0 = svadd_f64_z(svptrue_b64(), t0, t1);
       t0 = svadd_f64_z(svptrue_b64(), t0, t3);     // 4 flops

       //-----------------------------    104 flops
       svst1_f64(svptrue_b64(), data_d+i, t0);
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
