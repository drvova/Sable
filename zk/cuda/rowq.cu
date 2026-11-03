#include "field.cuh"

// Requant8 / ShiftQ8 row quantization (model._rowq), one block per row.
// e = least c in [emin, emax] with 2a in [-255 2^c, 255 2^c) for the whole row (emax if none);
// q = floor((2a + s) / 2s), s = 2^e; jstar = first j with 4a_j outside [-255 s, 255 s) when e > emin.
#define RQ_NT 256
#ifndef LLONG_MAX
#define LLONG_MAX 9223372036854775807ll
#define LLONG_MIN (-LLONG_MAX - 1)
#endif

template <typename TA>
__device__ __forceinline__ void rowq_impl(const TA* __restrict__ a, long long N, int emin, int emax,
                                TA* __restrict__ q, long long* __restrict__ e_out,
                                long long* __restrict__ js_out, long long* __restrict__ sig_out,
                                long long* __restrict__ f_out, long long* __restrict__ asel_out,
                                int* __restrict__ err) {
    __shared__ long long smax[RQ_NT / 32], smin[RQ_NT / 32];
    __shared__ long long sidx[RQ_NT / 32];
    __shared__ int se;
    const TA* row = a + (long long)blockIdx.x * N;
    TA* qrow = q + (long long)blockIdx.x * N;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    long long hi = LLONG_MIN, lo = LLONG_MAX;
    for (long long j = threadIdx.x; j < N; j += RQ_NT) {
        long long v = row[j];
        hi = v > hi ? v : hi;
        lo = v < lo ? v : lo;
    }
    for (int o = 16; o; o >>= 1) {
        long long h2 = __shfl_xor_sync(0xffffffff, hi, o), l2 = __shfl_xor_sync(0xffffffff, lo, o);
        hi = h2 > hi ? h2 : hi;
        lo = l2 < lo ? l2 : lo;
    }
    if (lane == 0) { smax[wid] = hi; smin[wid] = lo; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < RQ_NT / 32; w++) {
            hi = smax[w] > hi ? smax[w] : hi;
            lo = smin[w] < lo ? smin[w] : lo;
        }
        hi = smax[0] > hi ? smax[0] : hi;
        lo = smin[0] < lo ? smin[0] : lo;
        int e = emax;
        for (int c = emax; c >= emin; c--) {
            long long s = 1ll << c;
            if (2 * hi < 255 * s && 2 * lo >= -255 * s) e = c;
        }
        se = e;
    }
    __syncthreads();
    int e = se;
    long long s = 1ll << e;
    bool f = e > emin;
    long long first = LLONG_MAX;
    bool big = false;
    for (long long j = threadIdx.x; j < N; j += RQ_NT) {
        long long v = row[j];
        long long qq = (2 * v + s) >> (e + 1);
        big |= (qq > 127 || qq < -127);
        qrow[j] = (TA)qq;
        if (j < first && (4 * v >= 255 * s || 4 * v < -255 * s)) first = j;
    }
    if (big) atomicAdd(err, 1);
    for (int o = 16; o; o >>= 1) {
        long long f2 = __shfl_xor_sync(0xffffffff, first, o);
        first = f2 < first ? f2 : first;
    }
    if (lane == 0) sidx[wid] = first;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 0; w < RQ_NT / 32; w++) first = sidx[w] < first ? sidx[w] : first;
        long long js = 0;
        if (f) {
            if (first == LLONG_MAX) atomicAdd(err + 1, 1);
            else js = first;
        }
        long long as = row[js];
        e_out[blockIdx.x] = e;
        js_out[blockIdx.x] = js;
        f_out[blockIdx.x] = f ? 1 : 0;
        asel_out[blockIdx.x] = as;
        sig_out[blockIdx.x] = f ? (4 * as >= 255 * s ? 1 : 0) : 0;
    }
}

extern "C" __global__ void rowq(const long long* __restrict__ a, long long N, int emin, int emax,
                                long long* __restrict__ q, long long* __restrict__ e_out,
                                long long* __restrict__ js_out, long long* __restrict__ sig_out,
                                long long* __restrict__ f_out, long long* __restrict__ asel_out,
                                int* __restrict__ err) {
    rowq_impl<long long>(a, N, emin, emax, q, e_out, js_out, sig_out, f_out, asel_out, err);
}

// int32 rows and q (|a| < 2^31)
extern "C" __global__ void rowq32(const int* __restrict__ a, long long N, int emin, int emax,
                                  int* __restrict__ q, long long* __restrict__ e_out,
                                  long long* __restrict__ js_out, long long* __restrict__ sig_out,
                                  long long* __restrict__ f_out, long long* __restrict__ asel_out,
                                  int* __restrict__ err) {
    rowq_impl<int>(a, N, emin, emax, q, e_out, js_out, sig_out, f_out, asel_out, err);
}
