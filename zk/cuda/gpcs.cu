// Kernels of the global PCS: batched linear combinations of base columns and
// coset-of-4 Merkle leaves.
#include "field.cuh"
#include "blake3.cuh"

#define GID (blockIdx.x * (int64_t)blockDim.x + threadIdx.x)
#define GSTRIDE ((int64_t)gridDim.x * blockDim.x)
#define LC_MAX 128

// acc[i] += sum_k coef_k * col_k[i] for i < len_k (columns: base field, int32).
extern "C" __global__ void lincomb(const int64_t* __restrict__ ptrs, const int64_t* __restrict__ lens,
                                   const uint32_t* __restrict__ coefs, int n,
                                   uint32_t* __restrict__ acc, int64_t accn, int64_t m) {
  __shared__ uint32_t cf[LC_MAX * EXT_D];
  __shared__ int64_t ln[LC_MAX];
  __shared__ const uint32_t* cp[LC_MAX];
  for (int t = threadIdx.x; t < n * EXT_D; t += blockDim.x) cf[t] = coefs[t];
  for (int t = threadIdx.x; t < n; t += blockDim.x) {
    ln[t] = lens[t];
    cp[t] = reinterpret_cast<const uint32_t*>(ptrs[t]);
  }
  __syncthreads();
  // delayed reduction: per coordinate, products < p^2 summed 4 at a time (< 2^64) before one reduction
  for (int64_t i = GID; i < m; i += GSTRIDE) {
    uint32_t s[EXT_D];
    uint64_t t[EXT_D];
#pragma unroll
    for (int d = 0; d < EXT_D; d++) { s[d] = acc[d * accn + i]; t[d] = 0; }
    int c = 0;
    for (int k = 0; k < n; k++) {
      if (i < ln[k]) {
        const uint64_t v = cp[k][i];
#pragma unroll
        for (int d = 0; d < EXT_D; d++) t[d] += cf[k * EXT_D + d] * v;
        if (++c == 4) {
#pragma unroll
          for (int d = 0; d < EXT_D; d++) { s[d] = fadd(s[d], fred64(t[d])); t[d] = 0; }
          c = 0;
        }
      }
    }
    if (c) {
#pragma unroll
      for (int d = 0; d < EXT_D; d++) s[d] = fadd(s[d], fred64(t[d]));
    }
#pragma unroll
    for (int d = 0; d < EXT_D; d++) acc[d * accn + i] = s[d];
  }
}

// Partial folds of base columns at their high variables: column k (length len_k, zero beyond)
// read as H rows of Lo, part[hc][k * Lo + c] (SoA, stride pn) = sum_{h in chunk hc} e[h] col_k[h Lo + c].
// Products of e (< p) and column values (< p) are summed 4 at a time in 64 bits (delayed reduction).
extern "C" __global__ void cols_fold(const int64_t* __restrict__ ptrs, const int64_t* __restrict__ lens,
                                     int64_t Lo, int64_t H, const uint32_t* __restrict__ e, int64_t en,
                                     int64_t HCH, uint32_t* __restrict__ part, int64_t pn) {
  int64_t nct = (Lo + blockDim.x - 1) / blockDim.x;
  int64_t nh = (H + HCH - 1) / HCH;
  int64_t b = blockIdx.x;
  int64_t ct = b % nct; b /= nct;
  int64_t hc = b % nh; int64_t k = b / nh;
  int64_t c = ct * blockDim.x + threadIdx.x;
  if (c >= Lo) return;
  const uint32_t* col = reinterpret_cast<const uint32_t*>(ptrs[k]);
  const int64_t len = lens[k];
  int64_t h0 = hc * HCH, h1 = min(H, h0 + HCH);
  uint32_t s[EXT_D];
  uint64_t t[EXT_D];
#pragma unroll
  for (int d = 0; d < EXT_D; d++) { s[d] = 0; t[d] = 0; }
  int n = 0;
  for (int64_t h = h0; h < h1; h++) {
    int64_t i = h * Lo + c;
    if (i >= len) break;
    const uint64_t v = col[i];
#pragma unroll
    for (int d = 0; d < EXT_D; d++) t[d] += (uint64_t)e[d * en + h] * v;
    if (++n == 4) {
#pragma unroll
      for (int d = 0; d < EXT_D; d++) { s[d] = fadd(s[d], fred64(t[d])); t[d] = 0; }
      n = 0;
    }
  }
  if (n) {
#pragma unroll
    for (int d = 0; d < EXT_D; d++) s[d] = fadd(s[d], fred64(t[d]));
  }
#pragma unroll
  for (int d = 0; d < EXT_D; d++) part[(hc * EXT_D + d) * pn + k * Lo + c] = s[d];
}

// Leaf l of a chunk of ncols <= 64 codeword rows: the bytes of symbols
// 4l..4l+3 of each row, row-major (16 bytes per row), hashed as one BLAKE3
// input of 16 ncols bytes.  chain = 0: state[l] = digest;
// chain = 1: state[l] = BLAKE3(state[l] || digest).
extern "C" __global__ void leaf_chunk(const uint32_t* __restrict__ cw, int64_t stride, int ncols,
                                      int64_t nleaves, uint32_t* __restrict__ state, int chain) {
  for (int64_t leaf = GID; leaf < nleaves; leaf += GSTRIDE) {
    uint32_t cv[8];
    b3_init(cv);
    int nb = (ncols + 3) / 4;
    for (int blk = 0; blk < nb; blk++) {
      uint32_t m[16];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        int col = 4 * blk + q;
        if (col < ncols) {
          uint4 v = *reinterpret_cast<const uint4*>(cw + col * stride + 4 * leaf);
          m[4 * q] = v.x; m[4 * q + 1] = v.y; m[4 * q + 2] = v.z; m[4 * q + 3] = v.w;
        } else {
          m[4 * q] = m[4 * q + 1] = m[4 * q + 2] = m[4 * q + 3] = 0;
        }
      }
      uint32_t len = blk == nb - 1 ? 16u * (ncols - 4 * blk) : 64u;
      uint32_t fl = (blk == 0 ? B3_CHUNK_START : 0u) | (blk == nb - 1 ? (B3_CHUNK_END | B3_ROOT) : 0u);
      b3_compress(cv, m, 0, len, fl);
    }
    uint32_t* st = state + leaf * 8;
    if (chain) {
      uint32_t m[16];
#pragma unroll
      for (int k = 0; k < 8; k++) { m[k] = st[k]; m[8 + k] = cv[k]; }
      b3_hash64(m, cv);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) st[k] = cv[k];
  }
}
