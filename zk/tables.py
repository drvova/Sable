"""The global tagged lookup table T (profile O) and multiplicity counting.

A row is a tuple (tag, v0, v1, ..., v5) compressed as tag + beta_0 v0 + beta_1 v1 + ... with independent
challenges beta_i (domain.tuple_coefs).
Static tags:
  k in [0, 24]   (k, v)                      0 <= v < 2^k          range
  STAR8          (v)                         0 <= v < 255          q + 127
  STAR9          (v)                         0 <= v < 361          RoPE output + 180
  POW            (e + 16, 2^e)               -16 <= e <= 30        2^e in F_p
  SR             (k + 16, s, R, tR, tB, oB)  -16 <= k <= 24        shift regimes:
                 s = 2^k in F_p (inverse for k < 0), R = floor(2^(k-1)), tR = max(k, 0),
                 (tB, oB) = (31 - k, 2^(30-k)) for k >= 7, else (24, 2^23)
  SRA            (k + 16, tA, oA, f)          -16 <= k <= 24        input bound of a left shift:
                 f = [k <= 0], (tA, oA) = (24 + k, 2^(23+k)) if f else (0, 0); the gadget looks up
                 (tA, f acc + oA), i.e. |acc| < 2^(23+k), so c = 2^-k acc cannot wrap
  NPOW           (E + 15, E+, 2^E+, 2^E-, [E even], [E < 0])  -15 <= E <= 15  signed Norm
  RSQ            (delta - 2^14, floor(127 2^15 / sqrt(delta)))  2^14 <= delta < 2^16
  RECIP          (delta - 2^15, floor(2^32 / delta))         2^15 <= delta < 2^16
  SILU           (c + 2^14, tau_SiLU(c))                     |c| <= 2^14
  EXP            (x, tau_exp(x))                             0 <= x < 2^23
  EXPT           (d, tau_exp,theta(d))                       0 <= d < 2^24   sampling (theta > 0)
  RTEXP          (x, tau_r(x))                               0 <= x < 2^16   MoE router weights
Dynamic tables (witness-defined, e.g. gathered key rows) use tags >= DYN0 and
their own multiplicity columns."""
import math
import numpy as np
import torch
from .field import P
from . import kern
from .kern import ptr, i64
from .ext import _grid, NT
from .gkr import _col_expr, _poly_c
from .ext import dev as _dev

STAR8, STAR9, POW, SR, RSQ, RECIP, SILU, EXP, NPOW, EXPT, SRA, RTEXP = range(25, 37)
POW_OFF = 16
RSQ_LO = 1 << 14
RECIP_NUM = 1 << 32
DYN0 = 64
NV = 6
LOG_ROWS = 26
SR_KMIN, SR_KMAX = -16, 24


def p2(k):
    """2^k as a field element (k may be negative)."""
    return pow(2, k, P)


_P2T = {}


def p2t(k):
    """Elementwise 2^k as int32 field values for an integer tensor k in [-64, 30]."""
    dev = k.device
    if dev not in _P2T:
        _P2T[dev] = torch.tensor([p2(j) for j in range(-64, 31)], dtype=torch.int64, device=dev).to(torch.int32)
    return _P2T[dev][(k.to(torch.int64) + 64)]


def sr_row(k):
    s = p2(k)
    R = (1 << (k - 1)) if k >= 1 else 0
    tR = max(k, 0)
    tB, oB = ((31 - k), 1 << (30 - k)) if k >= 7 else (24, 1 << 23)
    return [k - SR_KMIN, s, R, tR, tB, oB]


def sra_row(k):
    if k <= 0:
        return [k - SR_KMIN, 24 + k, 1 << (23 + k), 1]
    return [k - SR_KMIN, 0, 0, 0]


def npow_row(E):
    return [E + 15, max(E, 0), 1 << max(E, 0), 1 << max(-E, 0), 1 - (E & 1), int(E < 0)]


class Layout:
    """(tag -> (base row, lo, size))."""

    def __init__(self):
        self.ent = {}
        off = 0
        for k in range(25):
            self.ent[k] = (off, 0, 1 << k)
            off += 1 << k
        for tag, lo, size in ((STAR8, 0, 255), (STAR9, 0, 361), (POW, 0, 31 + POW_OFF),
                              (SR, 0, SR_KMAX - SR_KMIN + 1), (RSQ, 0, (1 << 16) - RSQ_LO), (RECIP, 0, 1 << 15),
                              (SILU, 0, (1 << 15) + 1), (EXP, 0, 1 << 23), (NPOW, 0, 31), (EXPT, 0, 1 << 24),
                              (SRA, 0, SR_KMAX - SR_KMIN + 1), (RTEXP, 0, 1 << 16)):
            self.ent[tag] = (off, lo, size)
            off += size
        self.rows = off
        assert off <= 1 << LOG_ROWS

    def dev(self):
        base = torch.zeros(DYN0, dtype=torch.int64)
        lo = torch.zeros(DYN0, dtype=torch.int64)
        size = torch.zeros(DYN0, dtype=torch.int64)
        for t, (b, l, s) in self.ent.items():
            base[t], lo[t], size[t] = b, l, s
        return base.to(_dev()), lo.to(_dev()), size.to(_dev())


LAYOUT = Layout()


def build_table(exp_vals, silu_vals, samp_vals=None, router_vals=None):
    """Columns (tag, v0..v5) as int32 field tensors of 2^LOG_ROWS rows (padding rows
    repeat (0, 0)).  exp_vals: int64 (2^23,), silu_vals: int64 (2^15 + 1,) for c = -2^14..,
    samp_vals: int64 (2^24,) tau_exp,theta of the policy's temperature (rows left at (0, 0)
    for greedy decoding)."""
    N = 1 << LOG_ROWS
    cols = [torch.zeros(N, dtype=torch.int64, device=_dev()) for _ in range(NV + 1)]
    L = LAYOUT.ent

    def put(tag, vals):
        b, lo, size = L[tag]
        cols[0][b:b + size] = tag
        for i, v in enumerate(vals):
            cols[1 + i][b:b + size] = v
    for k in range(25):
        put(k, [torch.arange(1 << k, device=_dev())])
    put(STAR8, [torch.arange(255, device=_dev())])
    put(STAR9, [torch.arange(361, device=_dev())])
    e = torch.arange(31 + POW_OFF, device=_dev())
    put(POW, [e, torch.tensor([p2(int(x) - POW_OFF) for x in e], dtype=torch.int64, device=_dev())])
    rows = torch.tensor([sr_row(k) for k in range(SR_KMIN, SR_KMAX + 1)], dtype=torch.int64, device=_dev())
    put(SR, [rows[:, i] for i in range(6)])
    rows = torch.tensor([sra_row(k) for k in range(SR_KMIN, SR_KMAX + 1)], dtype=torch.int64, device=_dev())
    put(SRA, [rows[:, i] for i in range(4)])
    rows = torch.tensor([npow_row(E) for E in range(-15, 16)], dtype=torch.int64, device=_dev())
    put(NPOW, [rows[:, i] for i in range(6)])
    d = np.arange(RSQ_LO, 1 << 16, dtype=np.int64)
    C2 = (127 << 15) ** 2
    mu = np.array([math.isqrt(C2 // int(x)) for x in d], dtype=np.int64)
    put(RSQ, [torch.from_numpy(d - RSQ_LO).to(_dev()), torch.from_numpy(mu).to(_dev())])
    dd = torch.arange(1 << 15, 1 << 16, device=_dev())
    put(RECIP, [dd - (1 << 15), RECIP_NUM // dd])
    c = torch.arange(-(1 << 14), (1 << 14) + 1, device=_dev())
    put(SILU, [c + (1 << 14), silu_vals.to(torch.int64).to(_dev())])
    x = torch.arange(1 << 23, device=_dev())
    put(EXP, [x, exp_vals.to(torch.int64).to(_dev())])
    if samp_vals is not None:
        put(EXPT, [torch.arange(1 << 24, device=_dev()), samp_vals.to(torch.int64).to(_dev())])
    if router_vals is not None:          # MoE models only (the other tables are unchanged without it)
        put(RTEXP, [torch.arange(1 << 16, device=_dev()), router_vals.to(torch.int64).to(_dev())])
    # SR's first entry is k + 16 >= 0 and its other entries are already field elements
    return [(t % P).to(torch.int32) for t in cols]


# ----------------------------------------------------------- debugging
_DBGT = {}


def debug_tuples(dom, tcols):
    """Every static lookup tuple of dom equals the table row its first entry selects."""
    from .sumcheck import FULL as _F, ROW as _R
    base, lo, _ = LAYOUT.dev()
    n = dom.n
    per = max(1, (1 << 24) >> n)

    def ev(poly, b0, b1):
        e = torch.arange(b0 << n, b1 << n, device=_dev(), dtype=torch.int64)
        b, idx = e >> n, e & ((1 << n) - 1)
        out = torch.zeros_like(e)
        for mono, c in poly.t.items():
            term = torch.full_like(e, int(c) % P)
            for v in mono:
                col = dom.cols[v]
                if col.kind == _F:
                    ix = e
                elif col.kind == _R:
                    ix = (idx >> col.c) + (0 if col.shared else (b << (n - col.c)))
                else:
                    ix = (idx & ((1 << col.c) - 1)) + (0 if col.shared else (b << col.c))
                term = term * col.data[ix].to(torch.int64) % P
            out = (out + term) % P
        return out
    if tcols is None:
        fps = []
        for li, ((tag, vals, num), dy) in enumerate(zip(dom.lookups, dom.dyn)):
            s = 0
            for b0 in range(0, dom.K, per):
                b1 = min(dom.K, b0 + per)
                key = ev(tag, b0, b1) * 1000003 + ev(vals[0], b0, b1)
                s += int(((key * 2654435761) % 2147483647).sum())
            fps.append(s)
        return fps
    bad = []
    for li, ((tag, vals, num), dy) in enumerate(zip(dom.lookups, dom.dyn)):
        if dy or num is not None:
            continue
        for b0 in range(0, dom.K, per):
            b1 = min(dom.K, b0 + per)
            tg = ev(tag, b0, b1)
            v0 = ev(vals[0], b0, b1)
            row = base[tg] + v0 - lo[tg]
            ok = tcols[0][row].to(torch.int64) == tg
            for j, vp in enumerate(vals[1:], 1):
                ok &= tcols[1 + j][row].to(torch.int64) == ev(vp, b0, b1)
            nb = int((~ok).sum())
            if nb:
                w = int((~ok).nonzero()[0])
                bad.append((li, b0 + (w >> n), nb, int(tg[w]), [int(ev(vp, b0, b1)[w]) for vp in vals]))
                break
    if bad:
        print(f"[dbg] domain {dom.name}: tuple mismatches {bad}", flush=True)


# ----------------------------------------------------------- multiplicities
_MK = {}


def _mult_kernel(n, cols, lookups):
    def pk(q):
        return None if q is None else tuple(sorted(q.t.items()))
    key = (n, tuple((t.kind, t.c, t.shared) for t in cols), tuple((pk(t), pk(vs[0])) for t, vs, _ in lookups))
    if key in _MK:
        return _MK[key]
    used = sorted(set(i for t, vs, _ in lookups for q in (t, vs[0]) for i in q.vars()))
    L = ['#include "field.cuh"']
    prm = [f"const uint32_t* __restrict__ C{k}" for k in range(len(cols))]
    prm += ["const int64_t* __restrict__ tbase", "const int64_t* __restrict__ tlo", "const int64_t* __restrict__ tsize",
            "int32_t* __restrict__ mult", "int32_t* __restrict__ err", "int64_t total"]
    L.append(f'extern "C" __global__ void mk({", ".join(prm)}) {{')
    # Warp-strided loop (every lane runs every iteration).  Equal addresses within a warp step are
    # merged by __match_any_sync; on top, each warp keeps one hot row per lookup whose hits are
    # summed in a warp-uniform register (dominant rows, e.g. the clamped exp row, otherwise
    # serialize every warp's atomics on one L2 address).
    L.append("  const unsigned FULLM = 0xffffffffu; const int lane = threadIdx.x & 31;")
    L.append("  const int64_t nw = ((int64_t)gridDim.x * blockDim.x) >> 5;")
    L.append("  const int64_t w0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;")
    for l in range(len(lookups)):
        L.append(f"  long long wa{l} = -1; unsigned wc{l} = 0;")
    L.append("  for (int64_t base = w0 * 32; base < total; base += nw * 32) {")
    L.append("    const bool ok = base + lane < total;")
    L.append("    const int64_t e = ok ? base + lane : 0;")
    L.append(f"    int64_t b = e >> {n}, idx = e & {(1 << n) - 1}ll;")
    for k in used:
        L.append(f"    uint32_t x{k}_0 = {_col_expr(k, cols[k], n)};")
    for l, (tag, vs, _) in enumerate(lookups):
        L.append("    { long long a = -1;")
        L.append(f"      if (ok) {{ uint32_t tg = {_poly_c(tag, 0)}; uint32_t v = {_poly_c(vs[0], 0)};")
        L.append(f"      if (tg >= {DYN0}u) {{ atomicAdd(err, 1); }} else {{")
        L.append("        int64_t r = (int64_t)v - tlo[tg];")
        L.append("        if (r < 0 || r >= tsize[tg]) { atomicAdd(err + 1, 1); err[2] = " + str(l) + "; err[3] = (int32_t)tg; err[4] = (int32_t)v; }")
        L.append("        else a = tbase[tg] + r; } }")
        L.append(f"      unsigned hits = __ballot_sync(FULLM, a >= 0 && a == wa{l});")
        L.append(f"      wc{l} += __popc(hits);")
        L.append(f"      bool miss = a >= 0 && a != wa{l};")
        L.append("      unsigned peers = __match_any_sync(FULLM, (unsigned long long)(miss ? a : -1ll));")
        L.append("      bool lead = miss && lane == __ffs(peers) - 1;")
        L.append("      unsigned cnt = __popc(peers);")
        L.append("      unsigned best = __reduce_max_sync(FULLM, lead ? ((cnt << 5) | lane) : 0u);")
        L.append("      if (hits == 0 && (best >> 5) >= 8) {")
        L.append(f"        if (lane == 0 && wa{l} >= 0 && wc{l}) atomicAdd(mult + wa{l}, (int)wc{l});")
        L.append(f"        wa{l} = __shfl_sync(FULLM, a, best & 31); wc{l} = best >> 5;")
        L.append("        if (lead && lane != (int)(best & 31)) atomicAdd(mult + a, (int)cnt);")
        L.append("      } else if (lead) atomicAdd(mult + a, (int)cnt); }")
    L.append("  }")
    for l in range(len(lookups)):
        L.append(f"  if (lane == 0 && wa{l} >= 0 && wc{l}) atomicAdd(mult + wa{l}, (int)wc{l});")
    L.append("}")
    fn = kern.compile_module("\n".join(L), 8).get_function("mk")
    _MK[key] = fn
    return fn


_DEV = None


def count(dom, chunks, Tab):
    """Add the static-table lookups of a domain (with data) into their chunks' multiplicities."""
    global _DEV
    if _DEV is None:
        _DEV = LAYOUT.dev()
    idx = [i for i, (l, dy) in enumerate(zip(dom.lookups, dom.dyn)) if not dy and l[2] is None]
    if not idx:
        return
    total = dom.K << dom.n
    cols = [Tab(c.data, c.kind, c.c, shared=c.shared) for c in dom.cols]
    group = []
    for i in idx:
        if group and not chunks.fits(total):
            _count_group(dom, cols, *group)
            group = []
        m = chunks.take((dom.name, i), total)
        if group:
            group[1].append(i)
        else:
            group = [m, [i]]
    if group:
        _count_group(dom, cols, *group)


def _count_group(dom, cols, mult, ls):
    total = dom.K << dom.n
    base, lo, size = _DEV
    lk = [dom.lookups[i] for i in ls]
    fn = _mult_kernel(dom.n, cols, lk)
    err = torch.zeros(8, dtype=torch.int32, device=_dev())
    kern.launch(fn, _grid(total), NT, [ptr(t.data) for t in cols] + [ptr(base), ptr(lo), ptr(size), ptr(mult),
                                                                       ptr(err), i64(total)])
    _ERRS.append((err, dom.name, list(ls)))


_ERRS = []


def check_counts():
    """Raise if any counted lookup fell outside its table (one synchronization for the pass)."""
    if not _ERRS:
        return
    errs = list(_ERRS)
    _ERRS.clear()
    allv = torch.stack([e for e, _, _ in errs]).cpu().tolist()
    for e, (_, name, ls) in zip(allv, errs):
        if e[0] or e[1]:
            raise ValueError(f"domain {name}: lookup {ls[e[2]]} out of table (tag {e[3]}, v {e[4]}); "
                             f"dynamic-tag hits {e[0]}, misses {e[1]}")
