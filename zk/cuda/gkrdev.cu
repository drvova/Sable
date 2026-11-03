// GKR layer glue on the device (device transcript mode).
#include "field.cuh"

// _layer_terms coefficients, rows of EXT_D words: per lookup l: a_l, a_l, a_l lam.
extern "C" __global__ void gkr_coefs(const uint32_t* __restrict__ a, const uint32_t* __restrict__ lam, int L,
                                     uint32_t* __restrict__ out) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= L) return;
  EF al = ef_load(a + l * EXT_D, 0, 1), lm = ef_load(lam, 0, 1);
  ef_store(out + (3 * l) * EXT_D, 0, 1, al);
  ef_store(out + (3 * l + 1) * EXT_D, 0, 1, al);
  ef_store(out + (3 * l + 2) * EXT_D, 0, 1, mul(al, lm));
}

// finals f: rows (4L, EXT_D) [p0, p1, q0, q1] per lookup -> cp_l = p0 + t (p1 - p0), cq_l likewise.
extern "C" __global__ void gkr_next(const uint32_t* __restrict__ f, const uint32_t* __restrict__ t, int L,
                                    uint32_t* __restrict__ cp, uint32_t* __restrict__ cq) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= L) return;
  EF tt = ef_load(t, 0, 1);
  EF p0 = ef_load(f + (4 * l) * EXT_D, 0, 1), p1 = ef_load(f + (4 * l + 1) * EXT_D, 0, 1);
  EF q0 = ef_load(f + (4 * l + 2) * EXT_D, 0, 1), q1 = ef_load(f + (4 * l + 3) * EXT_D, 0, 1);
  ef_store(cp + l * EXT_D, 0, 1, ef_lerp(p0, p1, tt));
  ef_store(cq + l * EXT_D, 0, 1, ef_lerp(q0, q1, tt));
}

// x^e for e given as little-endian 32-bit words (one thread): mu = lam^(p^D - 2) = 1 / lam.
extern "C" __global__ void ef_pow1(const uint32_t* __restrict__ x, const uint32_t* __restrict__ ew, int nbits,
                                   uint32_t* __restrict__ out) {
  EF b = ef_load(x, 0, 1), r = ef_one();
  for (int i = nbits - 1; i >= 0; i--) {
    r = mul(r, r);
    if ((ew[i >> 5] >> (i & 31)) & 1u) r = mul(r, b);
  }
  ef_store(out, 0, 1, r);
}

// Domain program coefficients (domain.py COEF_KINDS) from device challenges; AoS rows of EXT_D.
// kind: 0 Z, 1 beta_idx (tuple coefficient), 2 a_idx, 3 a_idx lam, 4 gam_idx, 5 gam_idx eta, 6 base constant bval,
// 7 a_idx lam (Z + mu - bval), 8 a_i lam beta_j (idx = 256 i + j), 9 Z + mu - bval, 10 -a_idx mu
// 11 bval +- lp[idx >> 1] (a layer parameter).
// (mu = 1 / lam: the factored lookup leaf a (q0 + q1) + a lam q0 q1 = a lam (q0 + mu)(q1 + mu) - a mu).
extern "C" __global__ void dom_coefs(const int* __restrict__ kind, const int* __restrict__ idx,
                                     const uint32_t* __restrict__ bval, int n, const uint32_t* __restrict__ Z,
                                     const uint32_t* __restrict__ beta, const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ lam, const uint32_t* __restrict__ gam,
                                     const uint32_t* __restrict__ eta, const uint32_t* __restrict__ mu,
                                     const uint32_t* __restrict__ lp, uint32_t* __restrict__ out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  EF v;
  switch (kind[k]) {
    case 0: v = ef_load(Z, 0, 1); break;
    case 1: v = ef_load(beta + idx[k] * EXT_D, 0, 1); break;
    case 2: v = ef_load(a + idx[k] * EXT_D, 0, 1); break;
    case 3: v = mul(ef_load(a + idx[k] * EXT_D, 0, 1), ef_load(lam, 0, 1)); break;
    case 4: v = ef_load(gam + idx[k] * EXT_D, 0, 1); break;
    case 5: v = mul(ef_load(gam + idx[k] * EXT_D, 0, 1), ef_load(eta, 0, 1)); break;
    case 7: v = mul(mul(ef_load(a + idx[k] * EXT_D, 0, 1), ef_load(lam, 0, 1)),
                    sub(add(ef_load(Z, 0, 1), ef_load(mu, 0, 1)), ef_base(bval[k]))); break;
    case 8: v = mul(mul(ef_load(a + (idx[k] >> 8) * EXT_D, 0, 1), ef_load(lam, 0, 1)),
                    ef_load(beta + (idx[k] & 255) * EXT_D, 0, 1)); break;
    case 9: v = sub(add(ef_load(Z, 0, 1), ef_load(mu, 0, 1)), ef_base(bval[k])); break;
    case 10: v = sub(ef_zero(), mul(ef_load(a + idx[k] * EXT_D, 0, 1), ef_load(mu, 0, 1))); break;
    case 11: {      // layer parameter: bval +- lp[idx >> 1] (layer constants read on the device)
      const uint32_t pv = lp[idx[k] >> 1];
      v = ef_base((idx[k] & 1) ? fsub(bval[k], pv) : fadd(bval[k], pv));
      break; }
    default: v = ef_base(bval[k]); break;
  }
  ef_store(out + k * EXT_D, 0, 1, v);
}
