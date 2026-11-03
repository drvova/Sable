// KoalaBear field p = 2^31 - 2^24 + 1 and the tower F_p2 = F_p[x]/(x^2-3),
// F_p4 = F_p2[y]/(y^2-x), F_p8 = F_p4[z]/(z^2-y)  (= F_p[X]/(X^8-3), z = X).
// Elements are canonical in [0, p).
// Extension vectors are stored structure-of-arrays: coordinate k of
// element j lives at ptr[k * n + j].
#pragma once
#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;
#else
#include <cstdint>
#endif

#ifndef EXT_D
#define EXT_D 8
#endif

#define MP 0x7f000001u
#define MP_MU 0x20408101cull     // floor(2^64 / p)
#define HD __device__ __forceinline__

HD uint32_t fadd(uint32_t a, uint32_t b) { uint32_t r = a + b; return min(r, r - MP); }
HD uint32_t fsub(uint32_t a, uint32_t b) { uint32_t r = a - b; return min(r, r + MP); }
HD uint32_t fneg(uint32_t a) { return fsub(0u, a); }
// Barrett: any t < 2^64.  q is floor(t/p) or one less, so t - q p < 2p < 2^32.
HD uint32_t fred64(uint64_t t) {
  uint64_t q = __umul64hi(t, MP_MU);
  uint32_t r = (uint32_t)t - (uint32_t)q * MP;
  return min(r, r - MP);
}
HD uint32_t fmul(uint32_t a, uint32_t b) { return fred64((uint64_t)a * b); }
HD uint32_t fmul3(uint32_t a) { return fadd(fadd(a, a), a); }
// signed small integer -> field
HD uint32_t fs(int64_t v) {
  int64_t m = v % (int64_t)MP; if (m < 0) m += MP; return (uint32_t)m;
}
HD uint32_t fs32(int32_t v) { return v >= 0 ? (uint32_t)v : (uint32_t)((int64_t)v + MP); }
HD int32_t fsigned(uint32_t v) { return v > (MP >> 1) ? (int32_t)((int64_t)v - MP) : (int32_t)v; }

struct F2 { uint32_t a, b; };
struct F4 { F2 a, b; };
struct F8 { F4 a, b; };

// ---- F2 = F_p[x]/(x^2 - 3)
HD F2 add(F2 x, F2 y) { return {fadd(x.a, y.a), fadd(x.b, y.b)}; }
HD F2 sub(F2 x, F2 y) { return {fsub(x.a, y.a), fsub(x.b, y.b)}; }
HD F2 neg(F2 x) { return {fneg(x.a), fneg(x.b)}; }
HD F2 mul(F2 x, F2 y) {
  // products < p^2 < 2^62, so p0 + 3 p1 and the Karatsuba middle term fit in 64 bits
  uint64_t p0 = (uint64_t)x.a * y.a, p1 = (uint64_t)x.b * y.b;
  uint64_t pm = (uint64_t)(x.a + x.b) * (uint64_t)(y.a + y.b);
  return {fred64(p0 + 3 * p1), fred64(pm - p0 - p1)};
}
HD F2 mulb(F2 x, uint32_t s) { return {fmul(x.a, s), fmul(x.b, s)}; }
HD F2 mul_x(F2 v) {  // x (a + b x) = 3b + a x
  return {fmul3(v.b), v.a};
}
HD F2 dbl(F2 x) { return add(x, x); }

// ---- F4 = F2[y]/(y^2 - x)
HD F4 add(F4 x, F4 y) { return {add(x.a, y.a), add(x.b, y.b)}; }
HD F4 sub(F4 x, F4 y) { return {sub(x.a, y.a), sub(x.b, y.b)}; }
HD F4 neg(F4 x) { return {neg(x.a), neg(x.b)}; }
#if defined(F8_NOINLINE) && EXT_D == 4 && !defined(F4_INLINE)
__device__ __noinline__
#else
HD
#endif
F4 mul(F4 x, F4 y) {
  // schoolbook with one reduction per coordinate: with the factors 3 (x^2 = 3, y^2 = x) folded into
  // the operands, each coordinate is a sum of 4 products < 4 p^2 < 2^64.  About 1.45x the throughput
  // of Karatsuba over F2 (9 products, 6 reductions).
  const uint32_t a0 = x.a.a, a1 = x.a.b, b0 = x.b.a, b1 = x.b.b;
  const uint32_t c0 = y.a.a, c1 = y.a.b, d0 = y.b.a, d1 = y.b.b;
  const uint32_t a13 = fmul3(a1), b03 = fmul3(b0), b13 = fmul3(b1);
  const uint64_t r0 = (uint64_t)a0 * c0 + (uint64_t)a13 * c1 + (uint64_t)b03 * d1 + (uint64_t)b13 * d0;
  const uint64_t r1 = (uint64_t)a0 * c1 + (uint64_t)a1 * c0 + (uint64_t)b0 * d0 + (uint64_t)b13 * d1;
  const uint64_t r2 = (uint64_t)a0 * d0 + (uint64_t)a13 * d1 + (uint64_t)b0 * c0 + (uint64_t)b13 * c1;
  const uint64_t r3 = (uint64_t)a0 * d1 + (uint64_t)a1 * d0 + (uint64_t)b0 * c1 + (uint64_t)b1 * c0;
  return {{fred64(r0), fred64(r1)}, {fred64(r2), fred64(r3)}};
}
HD F4 mulb(F4 x, uint32_t s) { return {mulb(x.a, s), mulb(x.b, s)}; }
HD F4 mul_u(F4 x) { return {mul_x(x.b), x.a}; }   // times y
HD F4 mul2(F4 x, F2 s) { return {mul(x.a, s), mul(x.b, s)}; }

// ---- F8
HD F8 add(F8 x, F8 y) { return {add(x.a, y.a), add(x.b, y.b)}; }
HD F8 sub(F8 x, F8 y) { return {sub(x.a, y.a), sub(x.b, y.b)}; }
HD F8 neg(F8 x) { return {neg(x.a), neg(x.b)}; }
#ifdef F8_NOINLINE
__device__ __noinline__
#else
HD
#endif
F8 mul(F8 x, F8 y) {
  F4 aa = mul(x.a, y.a), bb = mul(x.b, y.b);
  F4 m = mul(add(x.a, x.b), add(y.a, y.b));
  return {add(aa, mul_u(bb)), sub(m, add(aa, bb))};
}
HD F8 mulb(F8 x, uint32_t s) { return {mulb(x.a, s), mulb(x.b, s)}; }

#if EXT_D == 8
typedef F8 EF;
#else
typedef F4 EF;
#endif

HD void ef_zero(F4& x) { x = {{0, 0}, {0, 0}}; }
HD void ef_zero(F8& x) { ef_zero(x.a); ef_zero(x.b); }
HD EF ef_zero() { EF z; ef_zero(z); return z; }
HD EF ef_base(uint32_t v) { EF z; ef_zero(z); ((uint32_t*)&z)[0] = v; return z; }
HD EF ef_one() { return ef_base(1u); }

HD EF ef_load(const uint32_t* __restrict__ p, int64_t i, int64_t n) {
  EF r; uint32_t* q = (uint32_t*)&r;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) q[k] = p[k * n + i];
  return r;
}
HD void ef_load2(const uint32_t* __restrict__ p, int64_t i, int64_t n, EF& a, EF& b) {
  uint32_t* qa = (uint32_t*)&a, * qb = (uint32_t*)&b;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) {
    const uint32_t* row = p + k * n + i;
    qa[k] = row[0];
    qb[k] = row[1];
  }
}
HD void ef_store(uint32_t* __restrict__ p, int64_t i, int64_t n, EF v) {
  const uint32_t* q = (const uint32_t*)&v;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) p[k * n + i] = q[k];
}
HD void ef_store2(uint32_t* __restrict__ p, int64_t i, int64_t n, EF a, EF b) {
  const uint32_t* qa = (const uint32_t*)&a, * qb = (const uint32_t*)&b;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) {
    uint32_t* row = p + k * n + i;
    row[0] = qa[k];
    row[1] = qb[k];
  }
}
// Constant ext element passed as a small array (AoS, D words)
HD EF ef_c(const uint32_t* p) {
  EF r; uint32_t* q = (uint32_t*)&r;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) q[k] = p[k];
  return r;
}
#define R8VAL ef_r8(r0, r1, r2, r3, r4, r5, r6, r7)
HD EF ef_r8(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5,
            uint32_t r6, uint32_t r7) {
  uint32_t t[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
  return ef_c(t);
}
// a + (b - a) * r
HD EF ef_lerp(EF a, EF b, EF r) { return add(a, mul(sub(b, a), r)); }
// warp reduction of an ext value
HD EF ef_shfl_down(EF v, int off) {
  EF r; uint32_t* q = (uint32_t*)&r; const uint32_t* s = (const uint32_t*)&v;
#pragma unroll
  for (int k = 0; k < EXT_D; k++) q[k] = __shfl_down_sync(0xffffffffu, s[k], off);
  return r;
}
HD EF ef_warp_sum(EF v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = add(v, ef_shfl_down(v, o));
  return v;
}
