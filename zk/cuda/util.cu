// Small extension-field vector kernels (NVRTC).
#include "field.cuh"

HD uint32_t mulx(uint32_t a, uint32_t b) { return fmul(a, b); }
HD EF mulx(EF a, uint32_t b) { return mulb(a, b); }
HD EF mulx(uint32_t a, EF b) { return mulb(b, a); }
HD EF mulx(EF a, EF b) { return mul(a, b); }

#define R8ARGS uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7
#define GID (blockIdx.x * (int64_t)blockDim.x + threadIdx.x)
#define GSTRIDE ((int64_t)gridDim.x * blockDim.x)

// out[i] = in[2i] + r (in[2i+1] - in[2i]),  i < m.  Ext SoA.
extern "C" __global__ void ext_fold(const uint32_t* __restrict__ in, int64_t in_n,
                                    uint32_t* __restrict__ out, int64_t out_n,
                                    R8ARGS, int64_t m) {
  EF rr = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    EF a = ef_load(in, 2 * i, in_n), b = ef_load(in, 2 * i + 1, in_n);
    ef_store(out, i, out_n, ef_lerp(a, b, rr));
  }
}

// Batched fold of nt tables of 2*kp entries spaced sp apart: out[t*kp + i].
extern "C" __global__ void ext_fold_many(const uint32_t* __restrict__ in, int64_t in_n, int64_t sp, int64_t kp,
                                         uint32_t* __restrict__ out, int64_t out_n, R8ARGS, int64_t m) {
  EF rr = R8VAL;
  for (int64_t g = GID; g < m; g += GSTRIDE) {
    int64_t t = g / kp, i = g - t * kp;
    int64_t e = t * sp + 2 * i;
    EF a = ef_load(in, e, in_n), b = ef_load(in, e + 1, in_n);
    ef_store(out, g, out_n, ef_lerp(a, b, rr));
  }
}

// Base (int32 field values) -> ext fold.
extern "C" __global__ void base_fold(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                     int64_t out_n, R8ARGS, int64_t m) {
  EF rr = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    uint32_t a = in[2 * i], b = in[2 * i + 1];
    ef_store(out, i, out_n, add(ef_base(a), mulb(rr, fsub(b, a))));
  }
}

// The same folds with the challenge read from device memory (GPU transcript).
extern "C" __global__ void ext_fold_p(const uint32_t* __restrict__ in, int64_t in_n,
                                      uint32_t* __restrict__ out, int64_t out_n,
                                      const uint32_t* __restrict__ rp, int64_t m) {
  EF rr = ef_load(rp, 0, 1);
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    EF a = ef_load(in, 2 * i, in_n), b = ef_load(in, 2 * i + 1, in_n);
    ef_store(out, i, out_n, ef_lerp(a, b, rr));
  }
}

extern "C" __global__ void ext_fold_many_p(const uint32_t* __restrict__ in, int64_t in_n, int64_t sp, int64_t kp,
                                           uint32_t* __restrict__ out, int64_t out_n,
                                           const uint32_t* __restrict__ rp, int64_t m) {
  EF rr = ef_load(rp, 0, 1);
  for (int64_t g = GID; g < m; g += GSTRIDE) {
    int64_t t = g / kp, i = g - t * kp;
    int64_t e = t * sp + 2 * i;
    EF a = ef_load(in, e, in_n), b = ef_load(in, e + 1, in_n);
    ef_store(out, g, out_n, ef_lerp(a, b, rr));
  }
}

// Folds of up to 32 separate tables (base or ext, any lengths) with the challenge at rp; job = blockIdx.y.
struct FoldJobs { long long src[32], sn[32], dst[32], dn[32], m[32], base[32]; };
extern "C" __global__ void fold_list_p(const FoldJobs J, int nj, const uint32_t* __restrict__ rp) {
  const int j = blockIdx.y;
  if (j >= nj) return;
  EF rr = ef_load(rp, 0, 1);
  const uint32_t* in = (const uint32_t*)J.src[j];
  uint32_t* out = (uint32_t*)J.dst[j];
  const int64_t m = J.m[j], in_n = J.sn[j], out_n = J.dn[j];
  if (J.base[j]) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < m; i += (int64_t)gridDim.x * blockDim.x) {
      uint32_t a = in[2 * i], b = in[2 * i + 1];
      ef_store(out, i, out_n, add(ef_base(a), mulb(rr, fsub(b, a))));
    }
  } else {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < m; i += (int64_t)gridDim.x * blockDim.x)
      ef_store(out, i, out_n, ef_lerp(ef_load(in, 2 * i, in_n), ef_load(in, 2 * i + 1, in_n), rr));
  }
}

extern "C" __global__ void base_fold_p(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                       int64_t out_n, const uint32_t* __restrict__ rp, int64_t m) {
  EF rr = ef_load(rp, 0, 1);
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    uint32_t a = in[2 * i], b = in[2 * i + 1];
    ef_store(out, i, out_n, add(ef_base(a), mulb(rr, fsub(b, a))));
  }
}

// Sumcheck weight tables: part p is out[oo + (b << lb) + y] = bw[b] * buf[so + y].
extern "C" __global__ void eh_build(const uint32_t* __restrict__ bw, int64_t bw_n, int64_t K,
                                    const uint32_t* __restrict__ buf, int64_t buf_n,
                                    const int64_t* __restrict__ meta, uint32_t* __restrict__ out, int64_t out_n) {
  const int64_t so = meta[3 * blockIdx.y], lb = meta[3 * blockIdx.y + 1], oo = meta[3 * blockIdx.y + 2];
  const int64_t m = K << lb, mk = (1ll << lb) - 1;
  for (int64_t i = GID; i < m; i += GSTRIDE)
    ef_store(out, oo + i, out_n, mul(ef_load(bw, i >> lb, bw_n), ef_load(buf, so + (i & mk), buf_n)));
}

// Fold over the HIGH half: out[i] = in[i] + r (in[i + m] - in[i]).
extern "C" __global__ void ext_fold_hi(const uint32_t* __restrict__ in, int64_t in_n,
                                       uint32_t* __restrict__ out, int64_t out_n,
                                       R8ARGS, int64_t m) {
  EF rr = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    EF a = ef_load(in, i, in_n), b = ef_load(in, i + m, in_n);
    ef_store(out, i, out_n, ef_lerp(a, b, rr));
  }
}

extern "C" __global__ void ext_fold_hi_p(const uint32_t* __restrict__ in, int64_t in_n,
                                         uint32_t* __restrict__ out, int64_t out_n,
                                         const uint32_t* __restrict__ rp, int64_t m) {
  EF rr = ef_load(rp, 0, 1);
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    EF a = ef_load(in, i, in_n), b = ef_load(in, i + m, in_n);
    ef_store(out, i, out_n, ef_lerp(a, b, rr));
  }
}

// out (D, nk) [SoA] = sum_i beta_i C_ik: beta rows (n, D), C (n, nk, D) AoS.
extern "C" __global__ void comb_coefs(const uint32_t* __restrict__ beta, const uint32_t* __restrict__ C, int n,
                                      int nk, uint32_t* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nk) return;
  EF s = ef_zero();
  for (int i = 0; i < n; i++)
    s = add(s, mul(ef_load(beta + (int64_t)i * EXT_D, 0, 1), ef_load(C + ((int64_t)i * nk + k) * EXT_D, 0, 1)));
  ef_store(out, k, nk, s);
}

// Sumcheck tail (sumcheck.py _tail): table k (desc row of 8 int64: mode, is_base, ptr, a1..a5) at
// the tail's first round as a dense ext array of S = 2 npairs elements, out + k D S (row stride S).
// mode 0: FULL raw (ptr, Tn, vb, vs); 1: FULL fold with *rp (ptr, Tn, vb, vs);
// 2: non-FULL varying (ptr, Tn, sa, sb, sc, mk); 3: non-FULL constant (same).
extern "C" __global__ void tail_expand(const int64_t* __restrict__ desc, int64_t S, uint32_t* __restrict__ out,
                                       const uint32_t* __restrict__ rp) {
  const int k = blockIdx.y;
  const int64_t* d = desc + 8 * k;
  const int mode = (int)d[0], isb = (int)d[1];
  const uint32_t* T = (const uint32_t*)d[2];
  const int64_t Tn = d[3];
  uint32_t* o = out + (int64_t)k * EXT_D * S;
  EF r = mode == 1 ? ef_load(rp, 0, 1) : ef_zero();
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < S; e += (int64_t)gridDim.x * blockDim.x) {
    const int64_t x = e >> 1, bit = e & 1;
    EF v;
    if (mode <= 1) {
      const int vb = (int)d[4];
      const int64_t vs = d[5], msk = (1ll << vb) - 1;
      const int64_t sc = mode == 0 ? 2 : 4;
      const int64_t q = (((sc * x) >> vb) * vs + ((sc * x) & msk)) + (mode == 0 ? bit : 2 * bit);
      if (mode == 0) {
        v = isb ? ef_base(T[q]) : ef_load(T, q, Tn);
      } else {
        EF a = isb ? ef_base(T[q]) : ef_load(T, q, Tn), b = isb ? ef_base(T[q + 1]) : ef_load(T, q + 1, Tn);
        v = ef_lerp(a, b, r);
      }
    } else {
      const int64_t sa = d[4], sb = d[5], sc = d[6], mk = d[7];
      const int64_t i = ((x >> sa) << sb) | ((x >> sc) & mk);
      const int64_t q = mode == 2 ? 2 * i + bit : i;
      v = isb ? ef_base(T[q]) : ef_load(T, q, Tn);
    }
    ef_store(o, e, S, v);
  }
}

// eq table expansion step: t has 2^k valid entries; produce 2^(k+1):
// t[i + 2^k] = t[i] * r ; t[i] = t[i] * (1 - r).
extern "C" __global__ void eq_expand(uint32_t* __restrict__ t, int64_t n, int64_t half,
                                     R8ARGS) {
  EF rr = R8VAL;
  for (int64_t i = GID; i < half; i += GSTRIDE) {
    EF v = ef_load(t, i, n);
    EF hi = mul(v, rr);
    ef_store(t, i + half, n, hi);
    ef_store(t, i, n, sub(v, hi));
  }
}

// out[x] = lo[x & (2^kl - 1)] * hi[x >> kl] for x < n (eq table of a split point).
extern "C" __global__ void eq_outer(const uint32_t* __restrict__ buf, int64_t bn, int kl, int64_t hoff,
                                    uint32_t* __restrict__ out, int64_t n) {
  const int64_t m = (1ll << kl) - 1;
  for (int64_t i = GID; i < n; i += GSTRIDE)
    ef_store(out, i, n, mul(ef_load(buf, i & m, bn), ef_load(buf, hoff + (i >> kl), bn)));
}

// Many eq tables at once.  rbuf: n points AoS (n * D words).  Segment s covers
// variables [st[s], st[s] + ln[s]) and is written at out[off[s] ...].
extern "C" __global__ void eq_segments(const uint32_t* __restrict__ rbuf,
                                       const int* __restrict__ st, const int* __restrict__ ln,
                                       const int64_t* __restrict__ off, uint32_t* __restrict__ out,
                                       int64_t outn) {
  int s = blockIdx.y;
  int k = ln[s];
  int64_t cnt = 1ll << k;
  for (int64_t i = GID; i < cnt; i += GSTRIDE) {
    EF v = ef_one();
    for (int b = 0; b < k; b++) {
      EF r = ef_c(rbuf + (st[s] + b) * EXT_D);
      v = ((i >> b) & 1) ? mul(v, r) : sub(v, mul(v, r));
    }
    ef_store(out, off[s] + i, outn, v);
  }
}

// One GKR fractional-sum layer: children (b, h, x), h the top bit of each block.
extern "C" __global__ void gkr_pair(const uint32_t* __restrict__ p, int64_t pn,
                                    const uint32_t* __restrict__ q, int64_t qn,
                                    uint32_t* __restrict__ po, int64_t pon,
                                    uint32_t* __restrict__ qo, int64_t qon, int k, int64_t total) {
  for (int64_t e = GID; e < total; e += GSTRIDE) {
    int64_t b = e >> k, x = e & ((1ll << k) - 1);
    int64_t i0 = (b << (k + 1)) + x, i1 = i0 + (1ll << k);
    EF q0 = ef_load(q, i0, qn), q1 = ef_load(q, i1, qn);
    EF p0 = ef_load(p, i0, pn), p1 = ef_load(p, i1, pn);
    ef_store(po, e, pon, add(mul(p0, q1), mul(p1, q0)));
    ef_store(qo, e, qon, mul(q0, q1));
  }
}

// gkr_pair for up to 32 lookups of one layer at once (lookup = blockIdx.y).
struct PairJobs { long long p[32], pn[32], q[32], qn[32], po[32], pon[32], qo[32], qon[32]; };
extern "C" __global__ void gkr_pair_list(const PairJobs J, int nj, int k, int64_t total) {
  const int j = blockIdx.y;
  if (j >= nj) return;
  const uint32_t* p = (const uint32_t*)J.p[j]; const uint32_t* q = (const uint32_t*)J.q[j];
  uint32_t* po = (uint32_t*)J.po[j]; uint32_t* qo = (uint32_t*)J.qo[j];
  const int64_t pn = J.pn[j], qn = J.qn[j], pon = J.pon[j], qon = J.qon[j];
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < total; e += (int64_t)gridDim.x * blockDim.x) {
    int64_t b = e >> k, x = e & ((1ll << k) - 1);
    int64_t i0 = (b << (k + 1)) + x, i1 = i0 + (1ll << k);
    EF q0 = ef_load(q, i0, qn), q1 = ef_load(q, i1, qn);
    EF p0 = ef_load(p, i0, pn), p1 = ef_load(p, i1, pn);
    ef_store(po, e, pon, add(mul(p0, q1), mul(p1, q0)));
    ef_store(qo, e, qon, mul(q0, q1));
  }
}

extern "C" __global__ void base_inv(const uint32_t* __restrict__ a, uint32_t* __restrict__ o, int64_t m) {
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    uint32_t x = a[i], r = 1;
    uint32_t e = MP - 2;
    while (e) { if (e & 1) r = fmul(r, x); x = fmul(x, x); e >>= 1; }
    o[i] = r;
  }
}

// Codeword fold: out[s] = f0 + r (f1 - f0), f0 = (a+b)/2, f1 = (a-b) * itw2[s]
// with (a, b) = in[2s], in[2s+1] and itw2[s] = 1 / (2 tw[s]).
extern "C" __global__ void cw_fold(const uint32_t* __restrict__ in, int64_t in_n,
                                   const uint32_t* __restrict__ itw2, R8ARGS,
                                   uint32_t* __restrict__ out, int64_t out_n, int64_t m) {
  EF rr = R8VAL;
  const uint32_t inv2 = (MP + 1) / 2;
  for (int64_t s = GID; s < m; s += GSTRIDE) {
    EF a = ef_load(in, 2 * s, in_n), b = ef_load(in, 2 * s + 1, in_n);
    EF f0 = mulb(add(a, b), inv2), f1 = mulb(sub(a, b), itw2[s]);
    ef_store(out, s, out_n, add(f0, mul(rr, sub(f1, f0))));
  }
}

extern "C" __global__ void ext_mul_vec(const uint32_t* __restrict__ a, int64_t an,
                                       const uint32_t* __restrict__ b, int64_t bn,
                                       uint32_t* __restrict__ c, int64_t cn, int64_t m) {
  for (int64_t i = GID; i < m; i += GSTRIDE)
    ef_store(c, i, cn, mul(ef_load(a, i, an), ef_load(b, i, bn)));
}

// c[i] = a[i] * s  (s constant ext)
extern "C" __global__ void ext_scale(const uint32_t* __restrict__ a, int64_t an,
                                     R8ARGS, uint32_t* __restrict__ c,
                                     int64_t cn, int64_t m) {
  EF ss = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE) ef_store(c, i, cn, mul(ef_load(a, i, an), ss));
}

// c[i] += a[i] * s
extern "C" __global__ void ext_axpy(const uint32_t* __restrict__ a, int64_t an,
                                    R8ARGS, uint32_t* __restrict__ c,
                                    int64_t cn, int64_t m) {
  EF ss = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE)
    ef_store(c, i, cn, add(ef_load(c, i, cn), mul(ef_load(a, i, an), ss)));
}

// c[i] += base a[i] * s
extern "C" __global__ void base_axpy(const uint32_t* __restrict__ a, R8ARGS,
                                     uint32_t* __restrict__ c, int64_t cn, int64_t m) {
  EF ss = R8VAL;
  for (int64_t i = GID; i < m; i += GSTRIDE)
    ef_store(c, i, cn, add(ef_load(c, i, cn), mulb(ss, a[i])));
}

// Block partial sums of a[i] * b[i] (ext x ext) -> out[block] (D words AoS).
extern "C" __global__ void ext_dot_partial(const uint32_t* __restrict__ a, int64_t an,
                                           const uint32_t* __restrict__ b, int64_t bn, int64_t m,
                                           uint32_t* __restrict__ out) {
  EF acc = ef_zero();
  for (int64_t i = GID; i < m; i += GSTRIDE) acc = add(acc, mul(ef_load(a, i, an), ef_load(b, i, bn)));
  __shared__ uint32_t sh[32][EXT_D];
  acc = ef_warp_sum(acc);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { const uint32_t* q = (const uint32_t*)&acc; for (int k = 0; k < EXT_D; k++) sh[w][k] = q[k]; }
  __syncthreads();
  if (threadIdx.x == 0) {
    EF s = ef_zero();
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s = add(s, ef_c(sh[i]));
    const uint32_t* q = (const uint32_t*)&s;
    for (int k = 0; k < EXT_D; k++) out[blockIdx.x * EXT_D + k] = q[k];
  }
}

// Block partial sums of base a[i] * ext b[i].
extern "C" __global__ void base_ext_dot_partial(const uint32_t* __restrict__ a,
                                                const uint32_t* __restrict__ b, int64_t bn,
                                                int64_t m, uint32_t* __restrict__ out) {
  EF acc = ef_zero();
  for (int64_t i = GID; i < m; i += GSTRIDE) acc = add(acc, mulb(ef_load(b, i, bn), a[i]));
  __shared__ uint32_t sh[32][EXT_D];
  acc = ef_warp_sum(acc);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { const uint32_t* q = (const uint32_t*)&acc; for (int k = 0; k < EXT_D; k++) sh[w][k] = q[k]; }
  __syncthreads();
  if (threadIdx.x == 0) {
    EF s = ef_zero();
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s = add(s, ef_c(sh[i]));
    const uint32_t* q = (const uint32_t*)&s;
    for (int k = 0; k < EXT_D; k++) out[blockIdx.x * EXT_D + k] = q[k];
  }
}

// Embed signed int32 values into the field (in place OK).
extern "C" __global__ void to_field(const int32_t* __restrict__ in, uint32_t* __restrict__ out,
                                    int64_t m) {
  for (int64_t i = GID; i < m; i += GSTRIDE) out[i] = fs32(in[i]);
}

// ---- dots of base tables with an ext vector (int32 field or signed int8 data)
template <typename T> HD uint32_t tofield(T v);
template <> HD uint32_t tofield<uint32_t>(uint32_t v) { return v; }
template <> HD uint32_t tofield<int8_t>(int8_t v) { return fs32((int32_t)v); }
template <> HD uint32_t tofield<int16_t>(int16_t v) { return fs32((int32_t)v); }

// part[row * nch + ch] = sum_{j in chunk ch} e[j] X[row * L + j]   (AoS D words)
template <typename T>
__device__ void rowdot_impl(const T* __restrict__ X, int64_t L, const uint32_t* __restrict__ e, int64_t en,
                            int64_t nch, int64_t CH, uint32_t* __restrict__ part) {
  int64_t row = blockIdx.x / nch, ch = blockIdx.x % nch;
  int64_t j0 = ch * CH, j1 = min(L, j0 + CH);
  EF acc = ef_zero();
  for (int64_t j = j0 + threadIdx.x; j < j1; j += blockDim.x)
    acc = add(acc, mulb(ef_load(e, j, en), tofield<T>(X[row * L + j])));
  __shared__ uint32_t sh[32][EXT_D];
  acc = ef_warp_sum(acc);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { const uint32_t* q = (const uint32_t*)&acc; for (int k = 0; k < EXT_D; k++) sh[w][k] = q[k]; }
  __syncthreads();
  if (threadIdx.x == 0) {
    EF s = ef_zero();
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s = add(s, ef_c(sh[i]));
    const uint32_t* q = (const uint32_t*)&s;
    for (int k = 0; k < EXT_D; k++) part[blockIdx.x * EXT_D + k] = q[k];
  }
}
extern "C" __global__ void rowdot_b(const uint32_t* X, int64_t L, const uint32_t* e, int64_t en,
                                    int64_t nch, int64_t CH, uint32_t* part) {
  rowdot_impl<uint32_t>(X, L, e, en, nch, CH, part);
}
extern "C" __global__ void rowdot_i8(const int8_t* X, int64_t L, const uint32_t* e, int64_t en,
                                     int64_t nch, int64_t CH, uint32_t* part) {
  rowdot_impl<int8_t>(X, L, e, en, nch, CH, part);
}

// Several tables at once over int16 rows (one read of X): part[(row * nch + ch) * ns + s] (AoS D
// words) = sum_{j in chunk ch} e_s[j] X[row * L + j], with e_s = e + s * EXT_D * L (SoA, stride L).
#define RM_MAX 8
extern "C" __global__ void rowdot_m16(const int16_t* __restrict__ X, int64_t L, const uint32_t* __restrict__ e,
                                      int ns, int64_t nch, int64_t CH, uint32_t* __restrict__ part) {
  int64_t row = blockIdx.x / nch, ch = blockIdx.x % nch;
  int64_t j0 = ch * CH, j1 = min(L, j0 + CH);
  EF acc[RM_MAX];
#pragma unroll
  for (int s = 0; s < RM_MAX; s++) acc[s] = ef_zero();
  for (int64_t j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
    const uint32_t v = tofield<int16_t>(X[row * L + j]);
#pragma unroll
    for (int s = 0; s < RM_MAX; s++)
      if (s < ns) acc[s] = add(acc[s], mulb(ef_load(e + (int64_t)s * EXT_D * L, j, L), v));
  }
  __shared__ uint32_t sh[32][EXT_D];
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  for (int s = 0; s < ns; s++) {
    EF a = ef_warp_sum(acc[s]);
    if (l == 0) { const uint32_t* q = (const uint32_t*)&a; for (int k = 0; k < EXT_D; k++) sh[w][k] = q[k]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      EF t = ef_zero();
      for (int i = 0; i < (int)(blockDim.x >> 5); i++) t = add(t, ef_c(sh[i]));
      const uint32_t* q = (const uint32_t*)&t;
      for (int k = 0; k < EXT_D; k++) part[(blockIdx.x * (int64_t)ns + s) * EXT_D + k] = q[k];
    }
    __syncthreads();
  }
}

// part[mch][k * C + c] (SoA, stride pn) = sum_{m in chunk mch} e[m] X[(k M + m) C + c]
template <typename T>
__device__ void coldot_impl(const T* __restrict__ X, int64_t K, int64_t M, int64_t C,
                            const uint32_t* __restrict__ e, int64_t en, int64_t MCH,
                            uint32_t* __restrict__ part, int64_t pn) {
  int64_t nct = (C + blockDim.x - 1) / blockDim.x;
  int64_t nm = (M + MCH - 1) / MCH;
  int64_t b = blockIdx.x;
  int64_t ct = b % nct; b /= nct;
  int64_t mch = b % nm; int64_t k = b / nm;
  int64_t c = ct * blockDim.x + threadIdx.x;
  if (c >= C) return;
  int64_t m0 = mch * MCH, m1 = min(M, m0 + MCH);
  EF acc = ef_zero();
  for (int64_t m = m0; m < m1; m++) acc = add(acc, mulb(ef_load(e, m, en), tofield<T>(X[(k * M + m) * C + c])));
  ef_store(part + mch * (int64_t)EXT_D * pn, k * C + c, pn, acc);
}
extern "C" __global__ void coldot_b(const uint32_t* X, int64_t K, int64_t M, int64_t C, const uint32_t* e,
                                    int64_t en, int64_t MCH, uint32_t* part, int64_t pn) {
  coldot_impl<uint32_t>(X, K, M, C, e, en, MCH, part, pn);
}
extern "C" __global__ void coldot_i8(const int8_t* X, int64_t K, int64_t M, int64_t C, const uint32_t* e,
                                     int64_t en, int64_t MCH, uint32_t* part, int64_t pn) {
  coldot_impl<int8_t>(X, K, M, C, e, en, MCH, part, pn);
}
