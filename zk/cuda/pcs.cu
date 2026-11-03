// Circle-code encoding (rate 1/2), codeword folding, BLAKE3 Merkle trees.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "field.cuh"
#include "blake3.cuh"

#define U32(t) reinterpret_cast<uint32_t*>((t).data_ptr<int32_t>())
#define CU32(t) reinterpret_cast<const uint32_t*>((t).data_ptr<int32_t>())
#define STREAM at::cuda::getCurrentCUDAStream()

__constant__ int64_t c_tw_off[40];

// One pass of butterflies on bit positions [b_lo, b_lo + C).
// Element index i = lo + (mid << b_lo) + (hi << (b_lo + C)).
// If from_msg, element i reads src[i & (half - 1)] (rate-1/2 duplication).
#define FFT_PAD(e) ((e) + ((e) >> 5))
#ifndef FFT_NT
#define FFT_NT 512
#endif

// R butterfly levels k0 .. k0-R+1 in registers: each task owns the 2^R elements that differ in those mid bits.
template <int C, int WL, int R, bool HL>
__device__ __forceinline__ void fft_stage(uint32_t* s, const uint32_t* __restrict__ tw, int b_lo, int hi, int k0) {
  constexpr int W = 1 << WL;
  constexpr int G = 1 << R;
  const int klo = k0 - R + 1;
  for (int p = threadIdx.x; p < (W << C) / G; p += blockDim.x) {
    int lo = p & (W - 1), q = p >> WL;
    int rest = ((q >> klo) << (k0 + 1)) | (q & ((1 << klo) - 1));
    uint32_t v[G];
#pragma unroll
    for (int j = 0; j < G; j++) v[j] = s[FFT_PAD(((rest | (j << klo)) << WL) + lo)];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int k = k0 - r, bit = R - 1 - r;
      // twiddle index (hi << (C - k - 1)) + (mid >> (k + 1)) for mid in this butterfly
      const uint32_t* twb = tw + c_tw_off[b_lo + k] + ((int64_t)(HL ? hi + lo : hi) << (C - k - 1));
#pragma unroll
      for (int j = 0; j < G; j++) {
        if (j & (1 << bit)) continue;
        int mid = rest | (j << klo);
        uint32_t t = twb[mid >> (k + 1)];
        uint32_t tc = fmul(t, v[j | (1 << bit)]);
        uint32_t x = v[j];
        v[j] = fadd(x, tc);
        v[j | (1 << bit)] = fsub(x, tc);
      }
    }
#pragma unroll
    for (int j = 0; j < G; j++) s[FFT_PAD(((rest | (j << klo)) << WL) + lo)] = v[j];
  }
}

// HL (b_lo = 0 only): the W lanes are W consecutive tiles hi = blockIdx.x W + lo instead of low bits.
template <int C, int WL, bool HL>
__global__ void __launch_bounds__(FFT_NT) fft_pass_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                const uint32_t* __restrict__ tw, int n, int b_lo,
                                int from_msg, int64_t src_stride, int64_t dst_stride,
                                const int* __restrict__ sel) {
  constexpr int W = 1 << WL;
  constexpr int M = 1 << C;
  __shared__ uint32_t s[FFT_PAD(W * M) + 1];
  const int64_t col = blockIdx.y;
  const uint32_t* sp = src + col * src_stride;
  int hi;
  if (HL) {
    hi = (sel ? sel[blockIdx.x] : (int)blockIdx.x) * W;      // sel: only these groups of W tiles
    const int64_t base = (int64_t)hi << C;
    const int64_t hmask = from_msg ? (1ll << (n - 1)) - 1 : -1ll;
    for (int e = threadIdx.x; e < W * M; e += blockDim.x) {
      int lo = e >> C, mid = e & (M - 1);
      s[FFT_PAD(mid * W + lo)] = sp[(base + e) & hmask];
    }
    __syncthreads();
  } else {
    const int lo_tiles = (1 << b_lo) >> WL;
    const int tile = blockIdx.x;
    const int lo_base = (tile % lo_tiles) << WL;
    hi = tile / lo_tiles;
    // element (lo, mid) of this block sits at base + lo + (mid << b_lo); lo_base + lo < 2^b_lo
    const int64_t base = lo_base + ((int64_t)hi << (b_lo + C));
    if (from_msg) {
      // rate-1/2 duplication: element i reads src[i mod 2^(n-1)]
      const int64_t hmask = (1ll << (n - 1)) - 1;
      for (int e = threadIdx.x; e < W * M; e += blockDim.x) {
        int lo = e & (W - 1), mid = e >> WL;
        s[FFT_PAD(e)] = sp[(base + lo + ((int64_t)mid << b_lo)) & hmask];
      }
    } else {
      const uint32_t* spb = sp + base;
      for (int e = threadIdx.x; e < W * M; e += blockDim.x) {
        int lo = e & (W - 1), mid = e >> WL;
        s[FFT_PAD(e)] = spb[lo + (mid << b_lo)];
      }
    }
    __syncthreads();
  }
  int k0 = C - 1;
  for (; k0 >= 2; k0 -= 3) {
    fft_stage<C, WL, 3, HL>(s, tw, b_lo, hi, k0);
    __syncthreads();
  }
  if (k0 == 1) {
    fft_stage<C, WL, 2, HL>(s, tw, b_lo, hi, 1);
    __syncthreads();
  } else if (k0 == 0) {
    fft_stage<C, WL, 1, HL>(s, tw, b_lo, hi, 0);
    __syncthreads();
  }
  uint32_t* dpc = dst + col * dst_stride;
  if (HL) {
    uint32_t* dp = dpc + ((int64_t)hi << C);
    for (int e = threadIdx.x; e < W * M; e += blockDim.x) {
      int lo = e >> C, mid = e & (M - 1);
      dp[e] = s[FFT_PAD(mid * W + lo)];
    }
  } else {
    const int lo_tiles = (1 << b_lo) >> WL;
    const int lo_base = (blockIdx.x % lo_tiles) << WL;
    uint32_t* dp = dpc + lo_base + ((int64_t)hi << (b_lo + C));
    for (int e = threadIdx.x; e < W * M; e += blockDim.x) {
      int lo = e & (W - 1), mid = e >> WL;
      dp[lo + (mid << b_lo)] = s[FFT_PAD(e)];
    }
  }
}

// A pass holds at most FFT_TILE symbols (W lanes x 2^C) in shared memory.
#define FFT_TILE_L 13

template <int C, int WL, bool HL>
static void go_pass(dim3 grid, const uint32_t* src, uint32_t* dst, const uint32_t* tw, int n, int b_lo,
                    int from_msg, int64_t ss, int64_t ds, const int* sel = nullptr) {
  if constexpr (C + WL <= FFT_TILE_L) {
    fft_pass_kernel<C, WL, HL><<<grid, FFT_NT, 0, STREAM>>>(src, dst, tw, n, b_lo, from_msg, ss, ds, sel);
  } else {
    TORCH_CHECK(false, "fft pass tile too large");
  }
}

static int last_wl(int n, int C) { return std::min(std::min(4, n - C), FFT_TILE_L - C); }

template <int C>
static void launch_pass(const uint32_t* src, uint32_t* dst, const uint32_t* tw, int n, int b_lo,
                        int from_msg, int64_t ss, int64_t ds, int ncols, const int* sel = nullptr, int nsel = 0) {
#define GO(wl, hl) go_pass<C, wl, hl>(grid, src, dst, tw, n, b_lo, from_msg, ss, ds, sel); break;
  if (b_lo == 0 && n - C >= 1) {
    // W lanes = W consecutive tiles of 2^C contiguous symbols
    int WL = last_wl(n, C);
    dim3 grid(sel ? (unsigned)nsel : (unsigned)((1ll << (n - C)) >> WL), ncols);
    switch (WL) {
      case 4: GO(4, true) case 3: GO(3, true) case 2: GO(2, true) case 1: GO(1, true) default: GO(0, true)
    }
    return;
  }
  // W lanes over the low bits: at least 16 when C <= 9 (64-byte runs), 8 for C = 10 (32-byte runs)
  int WL = std::min(std::min(b_lo, 8), std::max(4, FFT_TILE_L - C));
  if (C > 9) WL = std::min(b_lo, FFT_TILE_L - C);
  int64_t tiles = ((1ll << b_lo) >> WL) * (1ll << (n - b_lo - C));
  dim3 grid((unsigned)tiles, ncols);
  switch (WL) {
    case 8: GO(8, false) case 7: GO(7, false) case 6: GO(6, false) case 5: GO(5, false)
    case 4: GO(4, false) case 3: GO(3, false) case 2: GO(2, false) case 1: GO(1, false)
    default: GO(0, false)
  }
#undef GO
}

static void dispatch_pass(int C, const uint32_t* src, uint32_t* dst, const uint32_t* tw, int n,
                          int b_lo, int from_msg, int64_t ss, int64_t ds, int ncols,
                          const int* sel = nullptr, int nsel = 0) {
  switch (C) {
#define CASE(c) case c: launch_pass<c>(src, dst, tw, n, b_lo, from_msg, ss, ds, ncols, sel, nsel); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13)
#undef CASE
    default: TORCH_CHECK(false, "bad pass size");
  }
}

void set_tw_offsets(torch::Tensor off) {
  TORCH_CHECK(off.numel() <= 40);
  cudaMemcpyToSymbolAsync(c_tw_off, off.data_ptr<int64_t>(), off.numel() * 8, 0,
                          cudaMemcpyHostToDevice, STREAM);
}

// Encode msg (ncols, 2^(n-1)) -> out (ncols, 2^n) on the level-`level`
// domain of the twiddle tables (level 0 = circle, >= 1 = line).
// tw holds all levels concatenated; offsets were set by set_tw_offsets for
// the table's full n (levels shift with `level`).
void encode_impl(torch::Tensor msg, torch::Tensor out, torch::Tensor tw, int64_t n, int64_t level,
                 const torch::Tensor* pos) {
  int ncols = msg.size(0);
  const int64_t half = 1ll << (n - 1);
  TORCH_CHECK(msg.size(1) == half && out.size(1) == 2 * half);
  // butterfly bit positions n-2 .. 0 relative to this level; table level = level + b
  // We implement level offset by shifting the offset table on the host.
  // the short pass goes first, so that the last pass (b_lo = 0, one lane per
  // block) still covers 512 contiguous symbols
  // passes from the top bits down; the last pass (b_lo = 0) covers up to 13 contiguous bits and
  // the rest at most 10 bits each (2 passes up to 2^24), the first being the short one
  int top = n - 2;
  bool first = true;
  const int last = (int)std::min<int64_t>(FFT_TILE_L, n - 1);
  const int rest = (int)(n - 1) - last;
  const int nrest = (rest + 9) / 10;
  while (top >= 0) {
    int C;
    if (top < last) C = top + 1;
    else if (first && nrest > 0) C = rest - 10 * (nrest - 1);
    else C = 10;
    int b_lo = top - C + 1;
    if (pos && b_lo == 0 && n - C >= 1) {
      // only the groups of W tiles of the last pass that hold a requested position
      int WL = last_wl((int)n, C);
      torch::Tensor g = std::get<0>(at::_unique(torch::floor_divide(*pos, (int64_t)1 << (C + WL)), false))
                            .to(torch::kInt32).contiguous();
      dispatch_pass(C, first ? CU32(msg) : CU32(out), U32(out), CU32(tw), n, b_lo, first ? 1 : 0,
                    first ? msg.stride(0) : out.stride(0), out.stride(0), ncols,
                    reinterpret_cast<const int*>(g.data_ptr<int>()), (int)g.numel());
    } else {
      dispatch_pass(C, first ? CU32(msg) : CU32(out), U32(out), CU32(tw), n, b_lo, first ? 1 : 0,
                    first ? msg.stride(0) : out.stride(0), out.stride(0), ncols);
    }
    first = false;
    top = b_lo - 1;
  }
  if (n == 1) {  // no butterflies: just duplicate
    out.narrow(1, 0, half).copy_(msg);
    out.narrow(1, half, half).copy_(msg);
  }
}

void encode(torch::Tensor msg, torch::Tensor out, torch::Tensor tw, int64_t n, int64_t level) {
  encode_impl(msg, out, tw, n, level, nullptr);
}

// encode, but only the symbols at positions pos (int64, device) are guaranteed in out.
void encode_sel(torch::Tensor msg, torch::Tensor out, torch::Tensor tw, int64_t n, int64_t level, torch::Tensor pos) {
  encode_impl(msg, out, tw, n, level, &pos);
}

// ----------------------------------------------------------- Merkle / BLAKE3
// Leaves: for coset c, 16 consecutive symbols of each of 16 columns
// (column k occupies message block k).  cw: (16, N) int32.
__global__ void leaf_hash_kernel(const uint32_t* __restrict__ cw, int64_t stride, int ncols,
                                 int64_t nleaves, uint32_t* __restrict__ out) {
  int64_t leaf = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (leaf >= nleaves) return;
  uint32_t cv[8];
  b3_init(cv);
  uint32_t m[16];
  for (int k = 0; k < 16; k++) {
    if (k < ncols) {
      const uint4* p = reinterpret_cast<const uint4*>(cw + k * stride + leaf * 16);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint4 v = p[q];
        m[4 * q] = v.x; m[4 * q + 1] = v.y; m[4 * q + 2] = v.z; m[4 * q + 3] = v.w;
      }
    } else {
#pragma unroll
      for (int q = 0; q < 16; q++) m[q] = 0;
    }
    uint32_t fl = (k == 0 ? B3_CHUNK_START : 0u) | (k == 15 ? (B3_CHUNK_END | B3_ROOT) : 0u);
    b3_compress(cv, m, 0, 64, fl);
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[leaf * 8 + i] = cv[i];
}

void leaf_hash(torch::Tensor cw, torch::Tensor out) {
  int ncols = cw.size(0);
  TORCH_CHECK(ncols <= 16);
  int64_t nleaves = cw.size(1) / 16;
  leaf_hash_kernel<<<(nleaves + 127) / 128, 128, 0, STREAM>>>(CU32(cw), cw.stride(0), ncols,
                                                              nleaves, U32(out));
}

__global__ void node_hash_kernel(const uint32_t* __restrict__ in, int64_t nout,
                                 uint32_t* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= nout) return;
  uint32_t m[16];
  const uint4* p = reinterpret_cast<const uint4*>(in + i * 16);
#pragma unroll
  for (int q = 0; q < 4; q++) {
    uint4 v = p[q];
    m[4 * q] = v.x; m[4 * q + 1] = v.y; m[4 * q + 2] = v.z; m[4 * q + 3] = v.w;
  }
  uint32_t h[8];
  b3_hash64(m, h);
#pragma unroll
  for (int k = 0; k < 8; k++) out[i * 8 + k] = h[k];
}

void node_hash(torch::Tensor in, torch::Tensor out) {
  int64_t nout = out.size(0);
  node_hash_kernel<<<(nout + 255) / 256, 256, 0, STREAM>>>(CU32(in), nout, U32(out));
}

// ------------------------------------------------------------ field tests
__global__ void ext_mul_kernel(const uint32_t* a, const uint32_t* b, uint32_t* c, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  ef_store(c, i, n, mul(ef_load(a, i, n), ef_load(b, i, n)));
}

void ext_mul(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  int64_t n = a.size(1);
  ext_mul_kernel<<<(n + 255) / 256, 256, 0, STREAM>>>(CU32(a), CU32(b), U32(c), n);
}

// ------------------------------------------------------------ grinding
// Smallest nonce in [start, start + count) with BLAKE3(base32 || nonce_le64)[0..8]
// (as LE u64) < 2^(64 - bits); result atomically min-ed into *res.
__global__ void grind_kernel(const uint32_t* __restrict__ base, uint64_t start, uint64_t count,
                             int bits, unsigned long long* res) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint64_t nonce = start + i;
  uint32_t m[16];
#pragma unroll
  for (int k = 0; k < 8; k++) m[k] = base[k];
  m[8] = (uint32_t)nonce;
  m[9] = (uint32_t)(nonce >> 32);
#pragma unroll
  for (int k = 10; k < 16; k++) m[k] = 0;
  uint32_t cv[8];
  b3_init(cv);
  b3_compress(cv, m, 0, 40, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
  uint64_t v = (uint64_t)cv[0] | ((uint64_t)cv[1] << 32);
  if ((v >> (64 - bits)) == 0) atomicMin(res, (unsigned long long)nonce);
}

int64_t grind(torch::Tensor base, int64_t bits) {
  auto res = torch::full({1}, (int64_t)-1, base.options().dtype(torch::kInt64));
  const uint64_t chunk = 1ull << 24;
  for (uint64_t start = 0;; start += chunk) {
    grind_kernel<<<(chunk + 255) / 256, 256, 0, STREAM>>>(CU32(base), start, chunk, (int)bits,
                                                         reinterpret_cast<unsigned long long*>(res.data_ptr<int64_t>()));
    int64_t r = res.item<int64_t>();
    if (r != -1) return r;
  }
}
