// Transcript steps on the GPU (b3s.h is prepended by the loader).
// Launched with one warp: the transcript is sequential; the warp steps it together (b3w.cuh).
// One sumcheck round: send the reduced round message (nw field words), then draw
// one D-element challenge.
extern "C" __global__ void tr_round(B3S* st, const uint32_t* __restrict__ sums, int nw,
                                    uint32_t* __restrict__ msg, uint32_t* __restrict__ r, int D) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  b3w_round(st, sums, nw, msg, r, D);
}

// nsend messages of nw words each: message s is src[s * ss + i * sw], i < nw;
// each absorbed as one send, and copied to log (s * nw + i).
extern "C" __global__ void tr_sends(B3S* st, const uint32_t* __restrict__ src, int nsend, int nw,
                                    int64_t ss, int64_t sw, uint32_t* __restrict__ log) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  B3W f;
  b3w_load(f, st);
  for (int s = 0; s < nsend; s++) {
    const uint32_t* row = src + s * ss;
    for (int i = threadIdx.x; i < nw; i += 32) log[s * nw + i] = row[i * sw];
    b3w_send_words(f, [&](uint64_t i) -> uint32_t { return i < (uint64_t)nw ? row[i * sw] : 0u; },
                   4 * (uint64_t)nw);
  }
  b3w_store(f);
}

// absorb(d) = update(len u64 LE) + update(d)
extern "C" __global__ void tr_bytes(B3S* st, const uint8_t* __restrict__ d, int64_t n) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  B3W f;
  b3w_load(f, st);
  b3w_absorb(f, [&](uint64_t s) -> uint32_t {
    if (s < 8) return (uint32_t)((uint64_t)n >> (8 * s));
    return s - 8 < (uint64_t)n ? (uint32_t)d[s - 8] : 0u;
  }, 8 + (uint64_t)n);
  b3w_store(f);
}

extern "C" __global__ void tr_chal(B3S* st, int k, uint32_t* __restrict__ out) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  B3W f;
  b3w_load(f, st);
  b3w_challenge(f, k, out);
  b3w_store(f);
}

// absorb n <= 64 bytes passed by value
struct TrImm { unsigned long long w[8]; };
extern "C" __global__ void tr_bytes_imm(B3S* st, int n, const TrImm b) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  B3W f;
  b3w_load(f, st);
  b3w_absorb(f, [&](uint64_t s) -> uint32_t {
    if (s < 8) return (uint32_t)((uint64_t)n >> (8 * s));
    const uint64_t t = s - 8 < (uint64_t)n ? s - 8 : 0;
    return (uint32_t)(b.w[t >> 3] >> (8 * (t & 7)));
  }, 8 + (uint64_t)n);
  b3w_store(f);
}

// nsend messages of D words: message s = column cols[s] of an (D, *) table with row stride rs
struct TrCols { int c[256]; };
extern "C" __global__ void tr_send_cols(B3S* st, const uint32_t* __restrict__ src, int64_t rs, int nsend, int D,
                                        const TrCols cols, uint32_t* __restrict__ log) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  B3W f;
  b3w_load(f, st);
  for (int s = 0; s < nsend; s++) {
    const int c = cols.c[s];
    for (int i = threadIdx.x; i < D; i += 32) log[s * D + i] = src[i * rs + c];
    b3w_send_words(f, [&](uint64_t i) -> uint32_t { return i < (uint64_t)D ? src[i * rs + c] : 0u; },
                   4 * (uint64_t)D);
  }
  b3w_store(f);
}
