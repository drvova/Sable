// BLAKE3 compression (single-chunk inputs only: leaves of <= 1024 bytes and
// 64-byte internal Merkle nodes).
#pragma once
#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

#define B3_CHUNK_START 1u
#define B3_CHUNK_END 2u
#define B3_ROOT 8u

__device__ __constant__ uint32_t B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return __funnelshift_r(x, x, n); }

#define B3G(a, b, c, d, mx, my)          \
  a = a + b + mx; d = rotr32(d ^ a, 16); \
  c = c + d;      b = rotr32(b ^ c, 12); \
  a = a + b + my; d = rotr32(d ^ a, 8);  \
  c = c + d;      b = rotr32(b ^ c, 7);

// Compress one 64-byte block (m: 16 words) into chaining value cv (8 words).
__device__ __forceinline__ void b3_compress(uint32_t cv[8], const uint32_t m_in[16],
                                            uint32_t counter_lo, uint32_t block_len,
                                            uint32_t flags) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = m_in[i];
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3];
  uint32_t v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = B3_IV[0], v9 = B3_IV[1], v10 = B3_IV[2], v11 = B3_IV[3];
  uint32_t v12 = counter_lo, v13 = 0, v14 = block_len, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; r++) {
    B3G(v0, v4, v8, v12, m[0], m[1]);
    B3G(v1, v5, v9, v13, m[2], m[3]);
    B3G(v2, v6, v10, v14, m[4], m[5]);
    B3G(v3, v7, v11, v15, m[6], m[7]);
    B3G(v0, v5, v10, v15, m[8], m[9]);
    B3G(v1, v6, v11, v12, m[10], m[11]);
    B3G(v2, v7, v8, v13, m[12], m[13]);
    B3G(v3, v4, v9, v14, m[14], m[15]);
    if (r < 6) {
      // permutation: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
      uint32_t t[16];
      t[0] = m[2]; t[1] = m[6]; t[2] = m[3]; t[3] = m[10];
      t[4] = m[7]; t[5] = m[0]; t[6] = m[4]; t[7] = m[13];
      t[8] = m[1]; t[9] = m[11]; t[10] = m[12]; t[11] = m[5];
      t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8];
#pragma unroll
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
  cv[0] = v0 ^ v8; cv[1] = v1 ^ v9; cv[2] = v2 ^ v10; cv[3] = v3 ^ v11;
  cv[4] = v4 ^ v12; cv[5] = v5 ^ v13; cv[6] = v6 ^ v14; cv[7] = v7 ^ v15;
}

__device__ __forceinline__ void b3_init(uint32_t cv[8]) {
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
}

// Hash of a 64-byte message (two 32-byte children).
__device__ __forceinline__ void b3_hash64(const uint32_t m[16], uint32_t out[8]) {
  b3_init(out);
  b3_compress(out, m, 0, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
}
