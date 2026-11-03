// The Fiat-Shamir hash: the BLAKE3 compression function in sequential mode (a chaining value
// updated once per 64-byte block, flags 0, counter = block index; outputs from the pending last
// block with its length and flags CHUNK_END | ROOT), with a flat, copyable state, compiled both for
// the host (ctypes library) and the device (NVRTC), plus the transcript operations of
// zk/transcript.py.  Every message is length-prefixed and outputs carry their own flags, so the
// encoding is prefix-free and absorb and output compressions are domain-separated.  A challenge
// costs one compression (BLAKE3's tree mode needs one per level of its chunk tree).
#pragma once
#ifdef __CUDACC__
#define B3HD __host__ __device__
#else
#define B3HD
#include <cstdint>
#include <cstring>
#endif

#define B3S_CHUNK_END 2u
#define B3S_ROOT 8u

struct B3S {
  uint32_t cv[8];          // chaining value
  uint8_t block[64];       // pending block (compressed once more input arrives)
  uint64_t counter;        // blocks compressed so far
  uint32_t block_len, pad_;
};

B3HD inline uint32_t b3s_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

B3HD inline void b3s_g(uint32_t* v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx; v[d] = b3s_rotr(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];      v[b] = b3s_rotr(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my; v[d] = b3s_rotr(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];      v[b] = b3s_rotr(v[b] ^ v[c], 7);
}

B3HD inline void b3s_iv(uint32_t* o) {
  o[0] = 0x6A09E667u; o[1] = 0xBB67AE85u; o[2] = 0x3C6EF372u; o[3] = 0xA54FF53Au;
  o[4] = 0x510E527Fu; o[5] = 0x9B05688Cu; o[6] = 0x1F83D9ABu; o[7] = 0x5BE0CD19u;
}

// full 16-word compression output
B3HD inline void b3s_compress(const uint32_t cv[8], const uint32_t m_in[16], uint64_t counter,
                              uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  uint32_t m[16], v[16], iv[8];
  b3s_iv(iv);
  for (int i = 0; i < 16; i++) m[i] = m_in[i];
  for (int i = 0; i < 8; i++) { v[i] = cv[i]; v[i + 8] = iv[i]; }
  v[12] = (uint32_t)counter; v[13] = (uint32_t)(counter >> 32); v[14] = block_len; v[15] = flags;
  for (int r = 0; r < 7; r++) {
    b3s_g(v, 0, 4, 8, 12, m[0], m[1]);
    b3s_g(v, 1, 5, 9, 13, m[2], m[3]);
    b3s_g(v, 2, 6, 10, 14, m[4], m[5]);
    b3s_g(v, 3, 7, 11, 15, m[6], m[7]);
    b3s_g(v, 0, 5, 10, 15, m[8], m[9]);
    b3s_g(v, 1, 6, 11, 12, m[10], m[11]);
    b3s_g(v, 2, 7, 8, 13, m[12], m[13]);
    b3s_g(v, 3, 4, 9, 14, m[14], m[15]);
    uint32_t t[16];
    t[0] = m[2]; t[1] = m[6]; t[2] = m[3]; t[3] = m[10]; t[4] = m[7]; t[5] = m[0]; t[6] = m[4]; t[7] = m[13];
    t[8] = m[1]; t[9] = m[11]; t[10] = m[12]; t[11] = m[5]; t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8];
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
  for (int i = 0; i < 8; i++) { out[i] = v[i] ^ v[i + 8]; out[i + 8] = v[i + 8] ^ cv[i]; }
}

B3HD inline void b3s_words(const uint8_t* b, uint32_t* w) {
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)b[4 * i] | ((uint32_t)b[4 * i + 1] << 8) | ((uint32_t)b[4 * i + 2] << 16) |
           ((uint32_t)b[4 * i + 3] << 24);
}

B3HD inline void b3s_init(B3S* s) {
  b3s_iv(s->cv);
  for (int i = 0; i < 64; i++) s->block[i] = 0;
  s->counter = 0; s->block_len = 0; s->pad_ = 0;
}

// sequential mode: a full pending block is compressed (flags 0, counter = its index) when more
// input arrives, so the last block, possibly empty, is always pending
B3HD inline void b3s_update(B3S* s, const uint8_t* in, uint64_t n) {
  while (n) {
    if (s->block_len == 64) {
      uint32_t w[16], o[16];
      b3s_words(s->block, w);
      b3s_compress(s->cv, w, s->counter, 64, 0u, o);
      for (int i = 0; i < 8; i++) s->cv[i] = o[i];
      s->counter++;
      s->block_len = 0;
      for (int i = 0; i < 64; i++) s->block[i] = 0;
    }
    uint32_t bw = 64 - s->block_len;
    uint64_t take = n < bw ? n : bw;
    for (uint64_t i = 0; i < take; i++) s->block[s->block_len + i] = in[i];
    s->block_len += (uint32_t)take;
    in += take; n -= take;
  }
}

// output of everything absorbed so far (state unchanged): 64-byte block i is the compression of
// the pending block with counter i, its length and flags CHUNK_END | ROOT
B3HD inline void b3s_digest(const B3S* s, uint8_t* out, uint64_t n) {
  uint32_t w[16];
  b3s_words(s->block, w);
  uint64_t ctr = 0;
  while (n) {
    uint32_t o[16];
    b3s_compress(s->cv, w, ctr, s->block_len, B3S_CHUNK_END | B3S_ROOT, o);
    for (int i = 0; i < 64 && n; i++, n--) *out++ = (uint8_t)(o[i >> 2] >> (8 * (i & 3)));
    ctr++;
  }
}

// transcript: absorb(data) = update(len u64 LE) + update(data)
B3HD inline void tr_absorb(B3S* s, const uint8_t* d, uint64_t n) {
  uint8_t l[8];
  for (int i = 0; i < 8; i++) l[i] = (uint8_t)(n >> (8 * i));
  b3s_update(s, l, 8);
  b3s_update(s, d, n);
}

// transcript: k base-field challenges by rejection sampling (k <= 250)
B3HD inline void tr_challenge(B3S* s, int k, uint32_t* vals) {
  const uint32_t MPR = 0x7f000001u;
  int got = 0;
  const uint8_t sq[7] = {'s', 'q', 'u', 'e', 'e', 'z', 'e'};
  while (got < k) {
    uint8_t buf[4 * 256];
    int nb = 4 * (k - got + 4);
    b3s_digest(s, buf, nb);
    b3s_update(s, sq, 7);
    b3s_update(s, buf, nb < 32 ? nb : 32);   // the first min(32, nb) output bytes
    for (int i = 0; i < nb / 4; i++) {
      uint32_t w = ((uint32_t)buf[4 * i] | ((uint32_t)buf[4 * i + 1] << 8) | ((uint32_t)buf[4 * i + 2] << 16) |
                    ((uint32_t)buf[4 * i + 3] << 24)) & 0x7FFFFFFFu;
      if (w < MPR && got < k) vals[got++] = w;
    }
  }
}
