#pragma once
#include "field.cuh"
#include "b3w.cuh"

// One transcript round on the device (as tr_round in trk.cu): send nw words, draw a D-challenge.
// Called by all 32 lanes of warp 0.
__device__ __forceinline__ void sc_tr_round(B3S* st, const uint32_t* sums, int nw, uint32_t* msg, uint32_t* r, int D) {
  b3w_round(st, sums, nw, msg, r, D);
}

HD uint32_t mulx(uint32_t a, uint32_t b) { return fmul(a, b); }
HD EF mulx(EF a, uint32_t b) { return mulb(a, b); }
HD EF mulx(uint32_t a, EF b) { return mulb(b, a); }
HD EF mulx(EF a, EF b) { return mul(a, b); }

HD uint32_t addx(uint32_t a, uint32_t b) { return fadd(a, b); }
HD EF addx(EF a, EF b) { return add(a, b); }
HD EF addx(EF a, uint32_t b) { uint32_t* q = (uint32_t*)&a; q[0] = fadd(q[0], b); return a; }
HD EF addx(uint32_t b, EF a) { return addx(a, b); }

HD uint32_t subx(uint32_t a, uint32_t b) { return fsub(a, b); }
HD EF subx(EF a, EF b) { return sub(a, b); }
HD EF subx(EF a, uint32_t b) { uint32_t* q = (uint32_t*)&a; q[0] = fsub(q[0], b); return a; }
HD EF subx(uint32_t b, EF a) { return sub(ef_base(b), a); }

HD EF tox(EF a) { return a; }
HD EF tox(uint32_t a) { return ef_base(a); }

HD EF lerp_b(uint32_t a, uint32_t b, EF r) { return addx(mulb(r, fsub(b, a)), a); }

// d times the evaluation point (0, 2, 3, 4, ...)[pi] of the looped round bodies, additions up to 4
HD uint32_t xpv(uint32_t d, int pi) {
  const uint32_t d2 = fadd(d, d);
  return pi == 0 ? 0u : pi == 1 ? d2 : pi == 2 ? fadd(d2, d) : pi == 3 ? fadd(d2, d2) : fmul(d, (uint32_t)(pi + 1));
}
HD EF xpv(EF d, int pi) {
  const EF d2 = add(d, d);
  return pi == 0 ? ef_zero() : pi == 1 ? d2 : pi == 2 ? add(d2, d) : pi == 3 ? add(d2, d2) : mulb(d, (uint32_t)(pi + 1));
}

__device__ unsigned int sc_ticket = 0;

template <int NPT>
HD void block_reduce_store(EF* acc, uint32_t* out, B3S* trst = nullptr, uint32_t* trmsg = nullptr,
                          uint32_t* trr = nullptr) {
  __shared__ uint32_t sh[8][NPT * EXT_D];
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
  for (int p = 0; p < NPT; p++) {
    EF v = ef_warp_sum(acc[p]);
    if (l == 0) {
      const uint32_t* q = (const uint32_t*)&v;
#pragma unroll
      for (int k = 0; k < EXT_D; k++) sh[w][p * EXT_D + k] = q[k];
    }
  }
  __syncthreads();
  if (threadIdx.x < NPT * EXT_D) {
    int p = threadIdx.x / EXT_D, k = threadIdx.x % EXT_D;
    uint32_t s = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s = fadd(s, sh[i][p * EXT_D + k]);
    out[blockIdx.x * NPT * EXT_D + threadIdx.x] = s;
  }
  // the last block to finish sums the per-block rows into row gridDim.x; with trst, that
  // block also runs the round's transcript step (send the sums, draw the challenge)
  if (gridDim.x == 1) {
    if (trst) {
      __syncthreads();
      if (threadIdx.x < 32) sc_tr_round(trst, out, NPT * EXT_D, trmsg, trr, EXT_D);
    }
    return;
  }
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&sc_ticket, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int W = NPT * EXT_D;
  for (int c = threadIdx.x >> 5; c < W; c += blockDim.x >> 5) {
    uint64_t s = 0;
    for (int i = threadIdx.x & 31; i < (int)gridDim.x; i += 32) s += ((volatile uint32_t*)out)[i * W + c];
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) out[gridDim.x * W + c] = (uint32_t)(s % MP);
  }
  if (trst) {
    __syncthreads();
    if (threadIdx.x < 32) sc_tr_round(trst, out + gridDim.x * W, W, trmsg, trr, EXT_D);
  }
  if (threadIdx.x == 0) sc_ticket = 0;
}
