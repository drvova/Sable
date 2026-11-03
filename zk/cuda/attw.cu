// Attention witness of one head and one sequence (S x S causal scores, exact
// integers in fp32): row maxima, probability tiles, PV limbs and row sums.
#include "field.cuh"

#define XMIN ((1 << 23) - 1)

// One warp per row i: max and first argmax of acc[i, 0..i].
extern "C" __global__ void att_rowmax(const float* __restrict__ acc, int S, float* __restrict__ mx,
                                      int64_t* __restrict__ js) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (w >= S) return;
  const float* row = acc + (int64_t)w * S;
  float best = -1e30f;
  int bj = 0x7fffffff;
  for (int j = l; j <= w; j += 32) {
    float v = row[j];
    if (v > best) { best = v; bj = j; }
  }
  for (int o = 16; o; o >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, best, o);
    int oj = __shfl_down_sync(0xffffffffu, bj, o);
    if (ov > best || (ov == best && oj < bj)) { best = ov; bj = oj; }
  }
  if (l == 0) { mx[w] = best; js[w] = bj; }
}

// One block (128 threads, thread = column) per causal tile t = (ta[t], tb[t]).
// P = texp[x], x = M_i - acc_ij (XMIN above the diagonal); writes the tile,
// P >> 5 and P & 31 into key-block-major limb matrices [kb][i][j mod KB], and
// adds row sums into D.
extern "C" __global__ void att_tiles(const float* __restrict__ acc, int S, const float* __restrict__ mx,
                                     const int* __restrict__ ta, const int* __restrict__ tb,
                                     const int16_t* __restrict__ texp, int16_t* __restrict__ Pt,
                                     int8_t* __restrict__ phi, int8_t* __restrict__ plo, int lkb,
                                     unsigned long long* __restrict__ D) {
  __shared__ unsigned int rs[128][4];
  int t = blockIdx.x, c = threadIdx.x;
  int a = ta[t], b = tb[t];
  int j = b * 128 + c;
  int KB = 1 << lkb;
  int kb = j >> lkb, jj = j & (KB - 1);
  int64_t lb = (int64_t)kb * S * KB + jj;
  int16_t* out = Pt + (int64_t)t * 16384;
  for (int r = 0; r < 128; r++) {
    int i = a * 128 + r;
    int x = XMIN;
    if (j <= i) {
      int d = (int)(mx[i] - acc[(int64_t)i * S + j]);
      x = d < 0 ? 0 : (d > XMIN ? XMIN : d);
    }
    int p = texp[x];
    out[r * 128 + c] = (int16_t)p;
    int64_t li = lb + (int64_t)i * KB;
    phi[li] = (int8_t)(p >> 5);
    plo[li] = (int8_t)(p & 31);
    unsigned int s = (unsigned int)p;
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((c & 31) == 0) rs[r][c >> 5] = s;
  }
  __syncthreads();
  unsigned long long s = (unsigned long long)rs[c][0] + rs[c][1] + rs[c][2] + rs[c][3];
  if (s) atomicAdd(D + a * 128 + c, s);
}

// Stored tiles -> the limb matrices of att_tiles (positions outside the causal tiles are left as they are).
extern "C" __global__ void att_split(const int16_t* __restrict__ Pt, const int* __restrict__ ta,
                                     const int* __restrict__ tb, int S, int8_t* __restrict__ phi,
                                     int8_t* __restrict__ plo, int lkb) {
  int t = blockIdx.x, c = threadIdx.x;
  int a = ta[t], b = tb[t];
  int j = b * 128 + c;
  int KB = 1 << lkb;
  int kb = j >> lkb, jj = j & (KB - 1);
  int64_t lb = (int64_t)kb * S * KB + jj;
  const int16_t* in = Pt + (int64_t)t * 16384;
  for (int r = 0; r < 128; r++) {
    int p = in[r * 128 + c];
    int64_t li = lb + (int64_t)(a * 128 + r) * KB;
    phi[li] = (int8_t)(p >> 5);
    plo[li] = (int8_t)(p & 31);
  }
}

// Per causal tile t and column c: part{1,2}[t][k][c] = sum_r e{1,2}[k][a*128 + r] P_t[r][c] mod p
// (e: (D, *) field rows with row strides es1, es2, P in [0, 2^15)).
extern "C" __global__ void att_coldot(const int16_t* __restrict__ Pt, const int* __restrict__ ta, int D,
                                      const uint32_t* __restrict__ e1, const uint32_t* __restrict__ e2,
                                      int64_t es1, int64_t es2, uint32_t* __restrict__ part1, uint32_t* __restrict__ part2) {
  int t = blockIdx.x, c = threadIdx.x;
  int a = ta[t];
  const int16_t* in = Pt + (int64_t)t * 16384;
  unsigned long long s1[8], s2[8];
  for (int k = 0; k < 8; k++) s1[k] = s2[k] = 0;
  for (int r = 0; r < 128; r++) {
    unsigned long long p = (unsigned long long)(unsigned int)(int)in[r * 128 + c];
    int64_t row = (int64_t)a * 128 + r;
    for (int k = 0; k < D; k++) {
      s1[k] += p * e1[k * es1 + row];
      s2[k] += p * e2[k * es2 + row];
    }
  }
  for (int k = 0; k < D; k++) {
    part1[((int64_t)t * D + k) * 128 + c] = (uint32_t)(s1[k] % MP);
    part2[((int64_t)t * D + k) * 128 + c] = (uint32_t)(s2[k] % MP);
  }
}

// ---- head-batched variants: acc holds HB heads' (S x S) score matrices back to back.
// Row w of the batch is row i = w % S of head h = w / S; mx (HB * S) and js[h * jsT + i].
extern "C" __global__ void att_rowmax_h(const float* __restrict__ acc, int S, int nrows, float* __restrict__ mx,
                                        int64_t* __restrict__ js, int64_t jsT) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (w >= nrows) return;
  int h = w / S, i = w - h * S;
  const float* row = acc + (int64_t)w * S;
  float best = -1e30f;
  int bj = 0x7fffffff;
  for (int j = l; j <= i; j += 32) {
    float v = row[j];
    if (v > best) { best = v; bj = j; }
  }
  for (int o = 16; o; o >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, best, o);
    int oj = __shfl_down_sync(0xffffffffu, bj, o);
    if (ov > best || (ov == best && oj < bj)) { best = ov; bj = oj; }
  }
  if (l == 0) { mx[w] = best; js[h * jsT + i] = bj; }
}

// att_tiles for head h = blockIdx.y of the batch: tiles at Pt + h * pstride, limbs at
// [kb][h][i][jj] of an (nkb, HB, S, KB) array, row sums at D + h * dstride.  Block x does tiles
// tpb * blockIdx.x .. + tpb - 1 (of nt).
// hist (optional): per head (stride hstride) counts of the lookup input x over the tiles (the
// multiplicities of the softmax lookups (EXP, x, P)); inputs below HSH are counted in shared
// memory and flushed once per block, masked entries (x = XMIN) per block, others globally.
#define HSH 4096
extern "C" __global__ void att_tiles_h(const float* __restrict__ acc, int S, const float* __restrict__ mx,
                                       const int* __restrict__ ta, const int* __restrict__ tb,
                                       const int16_t* __restrict__ texp, int16_t* __restrict__ Pt, int64_t pstride,
                                       int8_t* __restrict__ phi, int8_t* __restrict__ plo, int lkb, int HB,
                                       unsigned long long* __restrict__ D, int64_t dstride,
                                       int* __restrict__ hist, int64_t hstride, int nt, int tpb) {
  __shared__ unsigned int rs[128][4];
  __shared__ int nmask;
  __shared__ int sh[HSH];
  int* hh = hist ? hist + (int64_t)blockIdx.y * hstride : nullptr;
  int c = threadIdx.x, h = blockIdx.y;
  if (hh) {
    for (int k = c; k < HSH; k += blockDim.x) sh[k] = 0;
    if (c == 0) nmask = 0;
  }
  __syncthreads();
  int lmask = 0;
  acc += (int64_t)h * S * S;
  mx += (int64_t)h * S;
  D += (int64_t)h * dstride;
  int KB = 1 << lkb;
  int t1 = min(nt, (blockIdx.x + 1) * tpb);
  for (int t = blockIdx.x * tpb; t < t1; t++) {
    int a = ta[t], b = tb[t];
    int j = b * 128 + c;
    int kb = j >> lkb, jj = j & (KB - 1);
    int64_t lb = ((int64_t)kb * HB + h) * S * KB + jj;
    int16_t* out = Pt + (int64_t)h * pstride + (int64_t)t * 16384;
    for (int r = 0; r < 128; r++) {
      int i = a * 128 + r;
      int x = XMIN;
      if (j <= i) {
        int d = (int)(mx[i] - acc[(int64_t)i * S + j]);
        x = d < 0 ? 0 : (d > XMIN ? XMIN : d);
      }
      if (hh) {
        if (x < HSH) atomicAdd(sh + x, 1);
        else if (x == XMIN) lmask++;
        else atomicAdd(hh + x, 1);
      }
      int p = texp[x];
      out[r * 128 + c] = (int16_t)p;
      int64_t li = lb + (int64_t)i * KB;
      phi[li] = (int8_t)(p >> 5);
      plo[li] = (int8_t)(p & 31);
      unsigned int s = (unsigned int)p;
      for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
      if ((c & 31) == 0) rs[r][c >> 5] = s;
    }
    __syncthreads();
    unsigned long long s = (unsigned long long)rs[c][0] + rs[c][1] + rs[c][2] + rs[c][3];
    if (s) atomicAdd(D + a * 128 + c, s);
    __syncthreads();
  }
  if (hh) {
    if (lmask) atomicAdd(&nmask, lmask);
    __syncthreads();
    for (int k = c; k < HSH; k += blockDim.x)
      if (sh[k]) atomicAdd(hh + k, sh[k]);
    if (c == 0 && nmask) atomicAdd(hh + XMIN, nmask);
  }
}

// Nr[h][b*S + i][kb][c] = ShiftRound(32 hi + lo, km[h][b*S + i]) from (nkb, HB*S, dh) int32 limb products.
extern "C" __global__ void att_nr_h(const int* __restrict__ hi, const int* __restrict__ lo, int nkb, int HBS,
                                    int S, int dh, const int64_t* __restrict__ km, int64_t kmT, int64_t boff,
                                    int64_t* __restrict__ Nr, int64_t nrT) {
  int64_t n = (int64_t)nkb * HBS * dh;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n; e += (int64_t)gridDim.x * blockDim.x) {
    int c = (int)(e % dh);
    int64_t r = e / dh;
    int w = (int)(r % HBS), kb = (int)(r / HBS);
    int h = w / S, i = w - h * S;
    long long k = km[h * kmT + boff + i];
    long long v = ((long long)hi[e] * 32 + (long long)lo[e] + (1ll << (k - 1))) >> k;
    Nr[h * nrT + ((boff + i) * nkb + kb) * dh + c] = v;
  }
}
