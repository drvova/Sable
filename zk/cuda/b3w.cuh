// Warp-cooperative transcript steps on a device B3S (b3s.h).  All 32 lanes of one warp call
// these with the same arguments and take the same branches; lane j < 16 holds word j of the
// current block, so a block is filled in one parallel step and compressions read their message
// through shuffles (registers only).  Bit-identical to tr_absorb / tr_challenge of b3s.h.
#pragma once
#include "b3s.h"

#define B3W_ALL 0xffffffffu

struct B3W {
  uint32_t cv[8];
  uint32_t bw;             // lane j < 16: pending block word j (bytes beyond len are 0)
  uint64_t cc;             // blocks compressed
  uint32_t len;            // pending block length
  B3S* g;
};

__device__ __forceinline__ int b3w_lane() { return threadIdx.x & 31; }

__device__ __forceinline__ void b3w_load(B3W& f, B3S* s) {
  const int l = b3w_lane();
  f.g = s;
#pragma unroll
  for (int i = 0; i < 8; i++) f.cv[i] = s->cv[i];
  f.bw = l < 16 ? ((const uint32_t*)s->block)[l] : 0u;
  f.cc = s->counter;
  f.len = s->block_len;
}

__device__ __forceinline__ void b3w_store(const B3W& f) {
  const int l = b3w_lane();
  B3S* s = f.g;
  __syncwarp();
  if (l < 16) ((uint32_t*)s->block)[l] = f.bw;
  if (l == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) s->cv[i] = f.cv[i];
    s->counter = f.cc;
    s->block_len = f.len;
  }
  __syncwarp();
}

__device__ __forceinline__ void b3w_msg(const B3W& f, uint32_t m[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = __shfl_sync(B3W_ALL, f.bw, i);
}

// compress the full pending block into the chaining value
__device__ __forceinline__ void b3w_block_done(B3W& f) {
  uint32_t m[16], o[16];
  b3w_msg(f, m);
  b3s_compress(f.cv, m, f.cc, 64, 0u, o);
#pragma unroll
  for (int i = 0; i < 8; i++) f.cv[i] = o[i];
  f.cc++;
  f.len = 0;
  f.bw = 0;
}

// absorb L bytes; get(s) returns stream byte s (called by every lane, s may be out of range)
template <class Get>
__device__ __forceinline__ void b3w_absorb(B3W& f, Get get, uint64_t L) {
  const int l = b3w_lane();
  uint64_t pos = 0;
  while (pos < L) {
    if (f.len == 64) b3w_block_done(f);
    uint32_t n = 64 - f.len;
    if (L - pos < n) n = (uint32_t)(L - pos);
    uint32_t w = f.bw;
#pragma unroll
    for (int b = 0; b < 4; b++) {
      const int p = 4 * (l & 15) + b;
      const bool in = l < 16 && p >= (int)f.len && p < (int)(f.len + n);
      const uint32_t v = get(in ? pos + (uint64_t)(p - (int)f.len) : pos);
      if (in) w |= (v & 0xffu) << (8 * b);
    }
    f.bw = w;
    f.len += n;
    pos += n;
  }
}

// Transcript.challenge_base(k) into vals (written by lane 0): the digest of 4 (k - got + 4) bytes
// is read 8 words at a time; its first min(32, 4 (k - got + 4)) bytes are re-absorbed after
// "squeeze".
__device__ __forceinline__ void b3w_challenge(B3W& f, int k, uint32_t* vals) {
  const int l = b3w_lane();
  int got = 0;
  while (got < k) {
    const int nw = k - got + 4;
    uint32_t w[16];
    b3w_msg(f, w);
    uint32_t fw = 0;               // lane i < 8: word i of the first output block
    int taken = 0;
    for (int blk = 0; 8 * blk < nw; blk++) {
      uint32_t o[16];
      b3s_compress(f.cv, w, (uint64_t)(blk >> 1), f.len, B3S_CHUNK_END | B3S_ROOT, o);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t d = o[(blk & 1) * 8 + i];
        if (blk == 0 && l == i) fw = d;
        const uint32_t x = d & 0x7FFFFFFFu;
        if (8 * blk + i < nw && x < MP && got + taken < k) {
          if (l == 0) vals[got + taken] = x;
          taken++;
        }
      }
    }
    // update(state, "squeeze"), update(state, first min(32, 4 nw) bytes)
    const uint64_t nf = nw < 8 ? 4 * (uint64_t)nw : 32;
    b3w_absorb(f, [&](uint64_t s) -> uint32_t {
      const uint32_t sq = s < 4 ? 0x65757173u : 0x00657a65u;   // "sque" "eze"
      const int q = s < 7 ? 0 : (int)((s - 7) >> 2);
      const uint32_t wd = __shfl_sync(B3W_ALL, fw, q & 7);
      if (s < 7) return sq >> (8 * (s & 3));
      return wd >> (8 * ((s - 7) & 3));
    }, 7 + nf);
    got += taken;
  }
  __syncwarp();
}

// stream = the 8-byte LE length nb, then the bytes of words src[0..]: one transcript send
template <class Word>
__device__ __forceinline__ void b3w_send_words(B3W& f, Word word, uint64_t nb) {
  b3w_absorb(f, [&](uint64_t s) -> uint32_t {
    if (s < 8) return (uint32_t)(nb >> (8 * s));
    const uint64_t t = s - 8;
    return word(t >> 2) >> (8 * (t & 3));
  }, 8 + nb);
}

// transcript round: absorb nw field words (as bytes; copied to msg), then a D-challenge into r
__device__ __forceinline__ void b3w_round(B3S* st, const uint32_t* sums, int nw, uint32_t* msg, uint32_t* r, int D) {
  B3W f;
  b3w_load(f, st);
  const volatile uint32_t* vs = (const volatile uint32_t*)sums;
  if (msg)
    for (int i = b3w_lane(); i < nw; i += 32) msg[i] = vs[i];
  b3w_send_words(f, [&](uint64_t i) -> uint32_t { return i < (uint64_t)nw ? vs[i] : 0u; }, 4 * (uint64_t)nw);
  if (r) b3w_challenge(f, D, r);
  b3w_store(f);
}
