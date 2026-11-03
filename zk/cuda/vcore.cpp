// Compiled verifier core (host, single thread): extension-field arithmetic and the sumcheck and
// LogUp-GKR verifiers of zk/sumcheck.py and zk/gkr.py, run on the transcript state of b3s.h and on
// the proof's field messages packed as one int64 array with offsets.  Same transcript bytes and
// the same checks as the Python functions they replace (zk/vcore.py).
#include <cstdint>
#include <cstring>
#include <vector>
#include "b3s.h"

typedef uint64_t u64;
static const u64 MP = 0x7f000001ull;

static inline u64 md(u64 x) { return x % MP; }
static inline u64 ad(u64 a, u64 b) { u64 r = a + b; return r >= MP ? r - MP : r; }
static inline u64 sb(u64 a, u64 b) { return a >= b ? a - b : a + MP - b; }
static inline u64 ml(u64 a, u64 b) { return (a * b) % MP; }
static u64 pw(u64 a, u64 e) { u64 r = 1; a %= MP; while (e) { if (e & 1) r = ml(r, a); a = ml(a, a); e >>= 1; } return r; }
static inline u64 inv_b(u64 a) { return pw(a, MP - 2); }

// Tower F2 = Fp[x]/(x^2 - 3), F4 = F2[y]/(y^2 - x), F8 = F4[z]/(z^2 - y); coordinates in tower order.
struct X { u64 v[8]; };
static int GD = 4;

static inline X zero() { X r; memset(r.v, 0, sizeof(r.v)); return r; }
static inline X one() { X r = zero(); r.v[0] = 1; return r; }
static inline X add(const X& a, const X& b) { X r = zero(); for (int i = 0; i < GD; i++) r.v[i] = ad(a.v[i], b.v[i]); return r; }
static inline X sub(const X& a, const X& b) { X r = zero(); for (int i = 0; i < GD; i++) r.v[i] = sb(a.v[i], b.v[i]); return r; }
static inline X smul(const X& a, u64 s) { X r = zero(); for (int i = 0; i < GD; i++) r.v[i] = ml(a.v[i], s); return r; }
static inline bool eqx(const X& a, const X& b) { for (int i = 0; i < GD; i++) if (a.v[i] != b.v[i]) return false; return true; }

static inline void f2m(u64 a0, u64 a1, u64 b0, u64 b1, u64& r0, u64& r1) {
  r0 = (a0 * b0 % MP + 3 * (a1 * b1 % MP)) % MP;
  r1 = (a0 * b1 % MP + a1 * b0 % MP) % MP;
}
static void f4m(const u64* a, const u64* b, u64* r) {
  u64 x0, x1, y0, y1, m0, m1;
  f2m(a[0], a[1], b[0], b[1], x0, x1);
  f2m(a[2], a[3], b[2], b[3], y0, y1);
  f2m(ad(a[0], a[2]), ad(a[1], a[3]), ad(b[0], b[2]), ad(b[1], b[3]), m0, m1);
  r[0] = ad(x0, ml(3, y1)); r[1] = ad(x1, y0);
  r[2] = sb(sb(m0, x0), y0); r[3] = sb(sb(m1, x1), y1);
}
static inline void f4mu(const u64* a, u64* r) { u64 t0 = ml(3, a[3]), t1 = a[2], t2 = a[0], t3 = a[1]; r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; }
static X mul(const X& a, const X& b) {
  X r = zero();
  if (GD == 4) { f4m(a.v, b.v, r.v); return r; }
  u64 x[4], y[4], m[4], s0[4], s1[4], yu[4];
  f4m(a.v, b.v, x); f4m(a.v + 4, b.v + 4, y);
  for (int i = 0; i < 4; i++) { s0[i] = ad(a.v[i], a.v[i + 4]); s1[i] = ad(b.v[i], b.v[i + 4]); }
  f4m(s0, s1, m); f4mu(y, yu);
  for (int i = 0; i < 4; i++) { r.v[i] = ad(x[i], yu[i]); r.v[i + 4] = sb(sb(m[i], x[i]), y[i]); }
  return r;
}
static void f2inv(u64 a0, u64 a1, u64& r0, u64& r1) {
  u64 n = inv_b(sb(ml(a0, a0), ml(3, ml(a1, a1))));
  r0 = ml(a0, n); r1 = ml(sb(0, a1), n);
}
static void f4inv(const u64* a, u64* r) {
  u64 s0, s1, t0, t1, i0, i1, c0, c1, d0, d1;
  f2m(a[0], a[1], a[0], a[1], s0, s1);
  f2m(a[2], a[3], a[2], a[3], t0, t1);
  f2inv(sb(s0, ml(3, t1)), sb(s1, t0), i0, i1);
  f2m(a[0], a[1], i0, i1, c0, c1);
  f2m(a[2], a[3], i0, i1, d0, d1);
  r[0] = c0; r[1] = c1; r[2] = sb(0, d0); r[3] = sb(0, d1);
}
static X inv(const X& a) {
  X r = zero();
  if (GD == 4) { f4inv(a.v, r.v); return r; }
  u64 aa[4], bb[4], bu[4], n[4], ni[4];
  f4m(a.v, a.v, aa); f4m(a.v + 4, a.v + 4, bb); f4mu(bb, bu);
  for (int i = 0; i < 4; i++) n[i] = sb(aa[i], bu[i]);
  f4inv(n, ni);
  f4m(a.v, ni, r.v); f4m(a.v + 4, ni, r.v + 4);
  for (int i = 4; i < 8; i++) r.v[i] = sb(0, r.v[i]);
  return r;
}

// ------------------------------------------------------------------ messages and transcript
struct Msgs { const int64_t* flat; const int64_t* off; int64_t n; int64_t pos; };
static int ERR = 0;   // 1: message shape, 2: check failed

static const int64_t* take(Msgs& M, int64_t len) {
  if (M.pos >= M.n || M.off[M.pos + 1] - M.off[M.pos] != len || M.off[M.pos] < 0) { ERR = 1; return nullptr; }
  return M.flat + M.off[M.pos++];
}
static inline u64 canon(int64_t v) { int64_t m = v % (int64_t)MP; return (u64)(m < 0 ? m + (int64_t)MP : m); }

// Transcript.send of a field message of len words (canonical values absorbed as u32 LE)
static void absorb_words(B3S* st, const int64_t* w, int64_t len) {
  std::vector<uint8_t> b(4 * len);
  for (int64_t i = 0; i < len; i++) { uint32_t v = (uint32_t)canon(w[i]); memcpy(&b[4 * i], &v, 4); }
  tr_absorb(st, b.data(), (uint64_t)b.size());
}
// Transcript.challenge_base(k): rejection sampling of 31-bit words, any k
static void chal_k(B3S* st, int k, uint32_t* vals) {
  static const uint8_t sq[7] = {'s', 'q', 'u', 'e', 'e', 'z', 'e'};
  int got = 0;
  std::vector<uint8_t> buf;
  while (got < k) {
    int nb = 4 * (k - got + 4);
    buf.resize(nb);
    b3s_digest(st, buf.data(), nb);
    b3s_update(st, sq, 7);
    b3s_update(st, buf.data(), nb < 32 ? nb : 32);         // the first min(32, nb) output bytes
    for (int i = 0; i < nb / 4 && got < k; i++) {
      uint32_t w;
      memcpy(&w, &buf[4 * i], 4);
      w &= 0x7FFFFFFFu;
      if (w < MP) vals[got++] = w;
    }
  }
}
static X chal(B3S* st) {
  uint32_t v[8]; chal_k(st, GD, v);
  X r = zero(); for (int i = 0; i < GD; i++) r.v[i] = v[i]; return r;
}
static X row(const int64_t* w) { X r = zero(); for (int i = 0; i < GD; i++) r.v[i] = canon(w[i]); return r; }

static X lagrange(const std::vector<X>& vals, const X& x) {
  int d = (int)vals.size() - 1;
  std::vector<u64> w(d + 1);
  for (int i = 0; i <= d; i++) {
    u64 den = 1;
    for (int j = 0; j <= d; j++) if (j != i) den = ml(den, canon(i - j));
    w[i] = inv_b(den);
  }
  std::vector<X> xm(d + 1), pre(d + 1);
  for (int j = 0; j <= d; j++) { xm[j] = x; xm[j].v[0] = sb(x.v[0], (u64)j); }
  pre[0] = one();
  for (int j = 0; j < d; j++) pre[j + 1] = mul(pre[j], xm[j]);
  X res = zero(), suf = one();
  for (int i = d; i >= 0; i--) {
    res = add(res, smul(mul(mul(pre[i], suf), vals[i]), w[i]));
    suf = mul(suf, xm[i]);
  }
  return res;
}

// sumcheck.verify: hdr is the label header (absorbed); eq_r (n, D) rows or null.  rho out (n, D).
static X sc_verify(B3S* st, Msgs& M, const uint8_t* hdr, int hdrlen, int n, int d, X claim, const X* eq_r,
                   std::vector<X>& rho) {
  tr_absorb(st, hdr, hdrlen);
  int npts = d > 1 ? d : 1;
  rho.resize(n);
  for (int j = 0; j < n; j++) {
    const int64_t* m = take(M, (int64_t)npts * GD);
    if (!m) return claim;
    absorb_words(st, m, (int64_t)npts * GD);
    X rj = chal(st);
    rho[j] = rj;
    X g0 = row(m), g1;
    if (eq_r) {
      const X& r = eq_r[j];
      g1 = mul(sub(claim, mul(sub(one(), r), g0)), inv(r));
    } else {
      g1 = sub(claim, g0);
    }
    std::vector<X> vals = {g0, g1};
    for (int i = 1; i < npts; i++) vals.push_back(row(m + (int64_t)i * GD));
    vals.resize((d > 1 ? d : 1) + 1);
    claim = lagrange(vals, rj);
  }
  return claim;
}

static void hdr_bytes(std::vector<uint8_t>& h, const uint8_t* label, int lablen, int n, int d, int K) {
  h.assign(label, label + lablen);
  h.push_back('|');
  h.push_back(n & 255); h.push_back((n >> 8) & 255);
  h.push_back(d & 255); h.push_back((d >> 8) & 255);
  for (int i = 0; i < 4; i++) h.push_back((K >> (8 * i)) & 255);
}

static std::vector<X> eq_tab(const std::vector<X>& r) {   // eq(r, x) for x < 2^|r|, x LSB = r[0]
  std::vector<X> v(1ull << r.size());
  v[0] = one();
  for (size_t b = 0; b < r.size(); b++) {
    u64 h = 1ull << b;
    for (u64 x = 0; x < h; x++) { X t = mul(v[x], r[b]); v[x + h] = t; v[x] = sub(v[x], t); }
  }
  return v;
}

extern "C" {

// eq(r, x) for x < 2^k (x LSB = r_0): r (D, k) row-major numpy layout, out (D, 2^k)
void vc_eq_table(int D, int k, const int64_t* r, int64_t* out) {
  GD = D;
  int64_t n = 1ll << k;
  std::vector<X> v(n);
  v[0] = one();
  for (int b = 0; b < k; b++) {
    X rb = zero();
    for (int i = 0; i < D; i++) rb.v[i] = canon(r[(int64_t)i * k + b]);
    int64_t h = 1ll << b;
    for (int64_t x = 0; x < h; x++) {
      X t = mul(v[x], rb);
      v[x + h] = t;
      v[x] = sub(v[x], t);
    }
  }
  for (int64_t x = 0; x < n; x++) for (int i = 0; i < D; i++) out[(int64_t)i * n + x] = (int64_t)v[x].v[i];
}

// MLE of a (D, n) ext vector (base: D = 1 rows given, ext = 0) at r (D, k), n = 2^k; folds LSB first
void vc_mle(int D, int k, int base, const int64_t* t, const int64_t* r, int64_t* out) {
  GD = D;
  int64_t n = 1ll << k;
  std::vector<X> cur(n);
  for (int64_t x = 0; x < n; x++) {
    X e = zero();
    if (base) e.v[0] = canon(t[x]);
    else for (int i = 0; i < D; i++) e.v[i] = canon(t[(int64_t)i * n + x]);
    cur[x] = e;
  }
  for (int j = 0; j < k; j++) {
    X rj = zero();
    for (int i = 0; i < D; i++) rj.v[i] = canon(r[(int64_t)i * k + j]);
    int64_t m = (n >> j) >> 1;
    for (int64_t x = 0; x < m; x++) cur[x] = add(cur[2 * x], mul(rj, sub(cur[2 * x + 1], cur[2 * x])));
  }
  for (int i = 0; i < D; i++) out[i] = (int64_t)cur[0].v[i];
}

// out = a * b elementwise, (D, n) each (a may be base: abase = 1, (n,))
void vc_vmul(int D, int64_t n, int abase, const int64_t* a, const int64_t* b, int64_t* out) {
  GD = D;
  for (int64_t x = 0; x < n; x++) {
    X u = zero(), w = zero();
    if (abase) u.v[0] = canon(a[x]); else for (int i = 0; i < D; i++) u.v[i] = canon(a[(int64_t)i * n + x]);
    for (int i = 0; i < D; i++) w.v[i] = canon(b[(int64_t)i * n + x]);
    X p = mul(u, w);
    for (int i = 0; i < D; i++) out[(int64_t)i * n + x] = (int64_t)p.v[i];
  }
}

// sum_x a_x b_x
void vc_dot(int D, int64_t n, int abase, const int64_t* a, const int64_t* b, int64_t* out) {
  GD = D;
  X s = zero();
  for (int64_t x = 0; x < n; x++) {
    X u = zero(), w = zero();
    if (abase) u.v[0] = canon(a[x]); else for (int i = 0; i < D; i++) u.v[i] = canon(a[(int64_t)i * n + x]);
    for (int i = 0; i < D; i++) w.v[i] = canon(b[(int64_t)i * n + x]);
    s = add(s, mul(u, w));
  }
  for (int i = 0; i < D; i++) out[i] = (int64_t)s.v[i];
}

// prod_j eq(a_j, b_j) over n coordinates; a, b (D, n) row-major as numpy (D, n).
void vc_eqv(int D, int n, const int64_t* a, const int64_t* b, int64_t* out) {
  GD = D;
  X e = one();
  for (int j = 0; j < n; j++) {
    X x = zero(), y = zero();
    for (int i = 0; i < D; i++) { x.v[i] = canon(a[(int64_t)i * n + j]); y.v[i] = canon(b[(int64_t)i * n + j]); }
    X xy = mul(x, y);
    X t = zero();                            // xy + (1 - x)(1 - y) = 2xy - x - y + 1
    for (int i = 0; i < D; i++) t.v[i] = sb(sb(ad(xy.v[i], xy.v[i]), x.v[i]), y.v[i]);
    t.v[0] = ad(t.v[0], 1);
    e = mul(e, t);
  }
  for (int i = 0; i < D; i++) out[i] = (int64_t)e.v[i];
}

// sumcheck.eval_prog: sum_b bw_b E(finals[:, b]).  Program: polys as (nmono, then per monomial: coef,
// degree, table ids...), expression in postfix: op codes 0 poly i, 1 coef i (ext row), 2 base const,
// 3 kc i (base value coef[i][0]), 4 +, 5 -, 6 *.  finals: table t has width w[t] (1: shared) at
// column offset fo[t] of an (D, ftot) array.  bw (D, K) or null.
void vc_eval_prog(int D, int K, const int64_t* polys, int npoly, const int64_t* code, int ncode,
                  const int64_t* coefs, const int64_t* fin, int64_t ftot, const int64_t* fo, const int64_t* fw,
                  int ntab, const int64_t* bw, int64_t* out) {
  GD = D;
  std::vector<int64_t> pstart(npoly);
  {
    int64_t q = 0;
    for (int i = 0; i < npoly; i++) {
      pstart[i] = q;
      int64_t nm = polys[q++];
      for (int64_t m = 0; m < nm; m++) { int64_t deg = polys[q + 1]; q += 2 + deg; }
    }
  }
  std::vector<X> vals(ntab), pv(npoly), stk;
  X acc = zero();
  for (int b = 0; b < K; b++) {
    for (int t = 0; t < ntab; t++) {
      int64_t c = fo[t] + (fw[t] > 1 ? b : 0);
      X v = zero();
      for (int i = 0; i < D; i++) v.v[i] = canon(fin[(int64_t)i * ftot + c]);
      vals[t] = v;
    }
    for (int p = 0; p < npoly; p++) {
      int64_t q = pstart[p];
      int64_t nm = polys[q++];
      X s = zero();
      for (int64_t m = 0; m < nm; m++) {
        u64 cf = canon(polys[q]); int64_t deg = polys[q + 1]; q += 2;
        X mv = zero(); mv.v[0] = cf;
        for (int64_t k = 0; k < deg; k++) mv = mul(mv, vals[polys[q + k]]);
        q += deg;
        s = add(s, mv);
      }
      pv[p] = s;
    }
    stk.clear();
    for (int i = 0; i < ncode; i += 2) {
      int64_t op = code[i], arg = code[i + 1];
      if (op == 0) stk.push_back(pv[arg]);
      else if (op == 1) { X c = zero(); for (int j = 0; j < D; j++) c.v[j] = canon(coefs[arg * D + j]); stk.push_back(c); }
      else if (op == 2) { X c = zero(); c.v[0] = canon(arg); stk.push_back(c); }
      else if (op == 3) { X c = zero(); c.v[0] = canon(coefs[arg * D]); stk.push_back(c); }
      else {
        X y = stk.back(); stk.pop_back();
        X x = stk.back(); stk.pop_back();
        stk.push_back(op == 4 ? add(x, y) : op == 5 ? sub(x, y) : mul(x, y));
      }
    }
    X s = stk.back();
    if (bw) { X w = zero(); for (int i = 0; i < D; i++) w.v[i] = canon(bw[(int64_t)i * K + b]); s = mul(s, w); }
    acc = add(acc, s);
  }
  for (int i = 0; i < D; i++) out[i] = (int64_t)acc.v[i];
}

// sumcheck.verify.  claim (D), eq_r (n, D) or null, rho out (n, D), red out (D).  Returns new message
// position, or -1 (message shape).
int64_t vc_sc_verify(B3S* st, int D, const int64_t* flat, const int64_t* off, int64_t nmsg, int64_t pos,
                     const uint8_t* label, int lablen, int n, int d, int K, const int64_t* claim,
                     const int64_t* eq_r, int64_t* rho_out, int64_t* red_out) {
  GD = D; ERR = 0;
  Msgs M{flat, off, nmsg, pos};
  std::vector<uint8_t> h;
  hdr_bytes(h, label, lablen, n, d, K);
  std::vector<X> er;
  if (eq_r) for (int j = 0; j < n; j++) er.push_back(row(eq_r + (int64_t)j * D));
  std::vector<X> rho;
  X red = sc_verify(st, M, h.data(), (int)h.size(), n, d, row(claim), eq_r ? er.data() : nullptr, rho);
  if (ERR) return -1;
  for (int j = 0; j < n; j++) for (int i = 0; i < D; i++) rho_out[(int64_t)j * D + i] = (int64_t)rho[j].v[i];
  for (int i = 0; i < D; i++) red_out[i] = (int64_t)red.v[i];
  return M.pos;
}

// gkr.verify_upper.  d2: degree of the block-phase rounds.  Outputs: roots (L, 2, D), ri (n-1, D),
// rb (lk, D), cp (L, D), cq (L, D).  Returns the new position, -1 (message shape) or -2 (top layer
// check) or -3 (layer check).
int64_t vc_gkr_upper(B3S* st, int D, const int64_t* flat, const int64_t* off, int64_t nmsg, int64_t pos,
                     const uint8_t* label, int lablen, int n, int K, int L, int d2,
                     int64_t* roots, int64_t* ri_out, int64_t* rb_out, int64_t* cp_out, int64_t* cq_out) {
  GD = D; ERR = 0;
  Msgs M{flat, off, nmsg, pos};
  int lk = 0; while ((1 << lk) < K) lk++;
  std::vector<X> cp(L), cq(L);
  for (int l = 0; l < L; l++) {
    const int64_t* m = take(M, 2 * (int64_t)D);
    if (!m) return -1;
    absorb_words(st, m, 2 * (int64_t)D);
    cp[l] = row(m); cq[l] = row(m + D);
    for (int i = 0; i < D; i++) { roots[((int64_t)l * 2) * D + i] = (int64_t)cp[l].v[i]; roots[((int64_t)l * 2 + 1) * D + i] = (int64_t)cq[l].v[i]; }
  }
  std::vector<uint8_t> lab(label, label + lablen), labtop = lab, labblk = lab, h;
  const char* t1 = "-top"; labtop.insert(labtop.end(), t1, t1 + 4);
  const char* t2 = "-blk"; labblk.insert(labblk.end(), t2, t2 + 4);
  auto layer_chal = [&](std::vector<X>& a, X& lam) {
    std::vector<uint32_t> buf(GD * L);
    chal_k(st, GD * L, buf.data());
    a.resize(L);
    for (int l = 0; l < L; l++) { a[l] = zero(); for (int i = 0; i < GD; i++) a[l].v[i] = buf[l * GD + i]; }
    lam = chal(st);
  };
  auto claimf = [&](const std::vector<X>& a, const X& lam) {
    X s = zero();
    for (int l = 0; l < L; l++) s = add(s, mul(a[l], add(cp[l], mul(lam, cq[l]))));
    return s;
  };
  // finals of a layer: 4 L messages (1, D), then t; new claims; returns sum_t c_t prod finals
  auto recv = [&](const std::vector<X>& a, const X& lam, X& t) -> X {
    std::vector<X> f(4 * L);
    for (int k = 0; k < 4 * L; k++) {
      const int64_t* m = take(M, D);
      if (!m) return zero();
      absorb_words(st, m, D);
      f[k] = row(m);
    }
    t = chal(st);
    X s = zero();
    for (int l = 0; l < L; l++) {
      const X &p0 = f[4 * l], &p1 = f[4 * l + 1], &q0 = f[4 * l + 2], &q1 = f[4 * l + 3];
      s = add(s, mul(a[l], add(mul(p0, q1), mul(p1, q0))));
      s = add(s, mul(mul(a[l], lam), mul(q0, q1)));
      cp[l] = add(p0, mul(t, sub(p1, p0)));
      cq[l] = add(q0, mul(t, sub(q1, q0)));
    }
    return s;
  };
  std::vector<X> r;                          // top tree point
  for (int j = 0; j < lk; j++) {
    std::vector<X> a; X lam; layer_chal(a, lam);
    X cl = claimf(a, lam);
    std::vector<X> rho;
    hdr_bytes(h, labtop.data(), (int)labtop.size(), j, j ? 2 : 0, 1);
    X red = sc_verify(st, M, h.data(), (int)h.size(), j, j ? 2 : 0, cl, j ? r.data() : nullptr, rho);
    if (ERR) return -1;
    X t; X ev = recv(a, lam, t);
    if (ERR) return -1;
    if (!eqx(red, ev)) return -2;
    r = rho; r.push_back(t);
  }
  std::vector<X> rb = r;
  if ((1 << lk) != K) {                      // padded per-block roots: denominators 1 at b >= K
    X pq = zero();
    std::vector<X> e = eq_tab(rb);
    for (u64 b = K; b < (1ull << lk); b++) pq = add(pq, e[b]);
    for (int l = 0; l < L; l++) cq[l] = sub(cq[l], pq);
  }
  std::vector<X> ri;
  for (int k = 0; k < n - 1; k++) {
    std::vector<X> a; X lam; layer_chal(a, lam);
    X cl = claimf(a, lam);
    std::vector<X> bw = lk ? eq_tab(rb) : std::vector<X>(1, one());
    std::vector<X> rho;
    hdr_bytes(h, lab.data(), (int)lab.size(), k, k ? 2 : 0, K);
    X red = sc_verify(st, M, h.data(), (int)h.size(), k, k ? 2 : 0, cl, k ? ri.data() : nullptr, rho);
    if (ERR) return -1;
    std::vector<X> rbn;
    if (lk) {
      hdr_bytes(h, labblk.data(), (int)labblk.size(), lk, d2, 1);
      red = sc_verify(st, M, h.data(), (int)h.size(), lk, d2, red, nullptr, rbn);
      if (ERR) return -1;
    }
    X t; X ev = recv(a, lam, t);
    if (ERR) return -1;
    X bwv = zero();
    if (K > 1) { std::vector<X> e = eq_tab(rbn); for (int b = 0; b < K; b++) bwv = add(bwv, mul(bw[b], e[b])); }
    else bwv = bw[0];
    if (!eqx(red, mul(bwv, ev))) return -3;
    ri = rho; ri.push_back(t);
    rb = rbn;
  }
  for (int k = 0; k < (int)ri.size(); k++) for (int i = 0; i < D; i++) ri_out[(int64_t)k * D + i] = (int64_t)ri[k].v[i];
  for (int k = 0; k < (int)rb.size(); k++) for (int i = 0; i < D; i++) rb_out[(int64_t)k * D + i] = (int64_t)rb[k].v[i];
  for (int l = 0; l < L; l++) for (int i = 0; i < D; i++) { cp_out[(int64_t)l * D + i] = (int64_t)cp[l].v[i]; cq_out[(int64_t)l * D + i] = (int64_t)cq[l].v[i]; }
  return M.pos;
}

}
