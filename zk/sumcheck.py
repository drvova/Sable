"""Generic sumcheck over sums of products of multilinear tables, batched over
K blocks that share the round challenges.

Proves  sum_b bw_b sum_{x in {0,1}^n} w(x) sum_t c_t prod_{k in t} T_k[b](x) = claim
with w(x) = eq(r, x) (eq mode) or 1, and bw_b block weights (default 1).

Tables (per block b; data laid out block-major)
  FULL   all n variables      base int32 (K*2^n,) or ext (D, K*2^n); an optional
         view (vbits, vstride, voff) maps element e of the logical table to
         voff + (e >> vbits) * vstride + (e & (2^vbits - 1)) in the round-0 read
  COL c  variables [0, c)     K*2^c entries, or 2^c if shared
  ROW c  variables [c, n)     K*2^(n-c) entries
Variables are bound LSB first; FULL tables are folded inside the round kernel.
Small-value (profile O): the first round loads base-field tables (`raw_b`) and
the first fold uses `lerp_b`; later rounds are in the extension. GKR leaf
values are computed from base columns (gkr.leaf_layer).
Round message: g(X) at X in {0, 2, ..., d}; g(1) follows from the claim.  In eq
mode g excludes the eq factor of the round variable, and the reduced claim
excludes the whole eq factor.
"""
import os
import time
import numpy as np
import torch
from .field import P
from . import kern
from .kern import ptr, i64, h2d
from .fe import fe
from .ext import gops
from .poly import Poly
from .devtr import DevTr
from .ext import dev as _dev

FULL, ROW, COL = 0, 1, 2
NTHREADS = 256
# minimum resident blocks per SM requested from the compiler for the round kernels (register cap)
SC_MINB = int(os.environ.get("ZK_SC_MINB", "1"))
NOVIEW = 62
LOOP_MONOS = int(os.environ.get("ZK_LOOP_MONOS", "48"))


class Tab:
    def __init__(self, data, kind=FULL, c=0, view=None, shared=False):
        self.data = data
        self.kind = kind
        self.c = c
        self.view = view
        self.shared = shared

    @property
    def is_base(self):
        return self.data.dim() == 1


class Prog:
    """Structured round expression: polys (Poly over table ids, base coefs)
    combined by expr, a tree of ('p', i) | ('c', j) | ('k', int) |
    ('+'|'-'|'*', a, b).  Coefficients c_j are ext values."""

    def __init__(self, polys, expr, coefs):
        self.polys = polys
        self.expr = expr
        self.coefs = coefs          # list of numpy (D,)

    def key(self):
        return (tuple(tuple(sorted(p.t.items())) for p in self.polys), repr(self.expr))

    def degree(self, kinds, cs, n):
        def pdeg(p):
            return max(round_degree([(None, list(m)) for m in p.t], kinds, cs, n, j) for j in range(n)) if n else 0
        pd = [pdeg(p) for p in self.polys]

        def rec(e):
            if e[0] == "p":
                return pd[e[1]]
            if e[0] in ("c", "k", "kc"):
                return 0
            a, b = rec(e[1]), rec(e[2])
            return a + b if e[0] == "*" else max(a, b)
        return rec(self.expr)

    def cexpr(self, p):
        def rec(e):
            if e[0] == "p":
                return f"P{e[1]}_{p}"
            if e[0] == "c":
                return f"cf[{e[1]}]"
            if e[0] == "k":
                return f"((uint32_t){int(e[1]) % P}u)"
            if e[0] == "kc":
                return f"((uint32_t)coef[{e[1]} * EXT_D])"
            f = {"+": "addx", "-": "subx", "*": "mulx"}[e[0]]
            return f"{f}({rec(e[1])}, {rec(e[2])})"
        return f"tox({rec(self.expr)})"

    def evaluate(self, F, vals):
        """vals: list of FE tuples per table id -> FE tuple."""
        pv = []
        for p in self.polys:
            acc = F.zero
            for mono, c in p.t.items():
                m = F.base(c)
                for i in mono:
                    m = F.mul(m, vals[i])
                acc = F.add(acc, m)
            pv.append(acc)

        def rec(e):
            if e[0] == "p":
                return pv[e[1]]
            if e[0] == "c":
                return F.c(self.coefs[e[1]])
            if e[0] == "k":
                return F.base(e[1])
            if e[0] == "kc":
                return F.base(int(self.coefs[e[1]][0]))
            a, b = rec(e[1]), rec(e[2])
            return {"+": F.add, "-": F.sub, "*": F.mul}[e[0]](a, b)
        return rec(self.expr)


def round_degree(terms, kinds, cs, n, j):
    d = 0
    for _, refs in terms:
        k = 0
        for t in refs:
            if kinds[t] == FULL or (kinds[t] == COL and j < cs[t]) or (kinds[t] == ROW and j >= cs[t]):
                k += 1
        d = max(d, k)
    return d


_MDEG = {}


def max_degree(terms, kinds, cs, n):
    key = (tuple(tuple(r) for _, r in terms), tuple(kinds), tuple(cs), n)
    if key not in _MDEG:
        if len(_MDEG) > 1 << 14:
            _MDEG.clear()
        _MDEG[key] = max(round_degree(terms, kinds, cs, n, j) for j in range(n)) if n else 0
    return _MDEG[key]


# --------------------------------------------------------------- codegen
def nslots(nt):
    return 6 * nt + 21


def _slot_buf(n):
    dt = np.dtype([("v", np.int64, n)])
    b = np.zeros((), dtype=dt)
    return b, b["v"]


def _horner(t, name):
    """Code for sum_mono c * prod(vars), factoring out the most frequent variable
    recursively (x * Q + R) to cut the number of field multiplications."""
    cnt = {}
    for mono in t:
        for k in set(mono):
            cnt[k] = cnt.get(k, 0) + 1
    if cnt and max(cnt.values()) >= 2:
        x = max(sorted(cnt), key=lambda k: cnt[k])
        Q, R = {}, {}
        for mono, c in t.items():
            if x in mono:
                m = list(mono)
                m.remove(x)
                m = tuple(m)
                Q[m] = (Q.get(m, 0) + c) % P
            else:
                R[mono] = c
        q = f"mulx({name(x)}, {_horner(Q, name)})"
        return f"addx({q}, {_horner(R, name)})" if R else q
    pos, neg = [], []
    for mono, c in t.items():
        c %= P
        if c == 0:
            continue
        f = [name(k) for k in mono]
        e = f[0] if f else None
        for g in f[1:]:
            e = f"mulx({e}, {g})"
        sgn = c > P // 2
        if sgn:
            c = P - c
        if e is None:
            e = f"((uint32_t){c}u)"
        elif c != 1:
            e = f"mulx({e}, (uint32_t){c}u)"
        (neg if sgn else pos).append(e)
    e = pos[0] if pos else "((uint32_t)0u)"
    for q in pos[1:]:
        e = f"addx({e}, {q})"
    for q in neg:
        e = f"subx({e}, {q})"
    return e


def _poly_cexpr(poly, vary, suf="_L"):
    if not os.environ.get("ZK_NOHORNER"):
        return _horner(dict(poly.t), lambda k: f"v{k}{suf}" if vary[k] else f"a{k}")
    parts = []
    for mono, c in poly.t.items():
        fac = [(f"v{k}{suf}" if vary[k] else f"a{k}") for k in mono]
        if not fac:
            parts.append(f"((uint32_t){c}u)")
            continue
        e = fac[0]
        for f in fac[1:]:
            e = f"mulx({e}, {f})"
        parts.append(e if c == 1 else f"mulx({e}, (uint32_t){c}u)")
    e = parts[0] if parts else "((uint32_t)0u)"
    for q in parts[1:]:
        e = f"addx({e}, {q})"
    return e


def _gen_grouped(L, ld, prog, kinds, vary, npts, use_w):
    """Long programs.  expr = outer factors * (sum of summands); summands sharing
    polys form a group, and each group loads only its own tables and loops over
    the evaluation points, so only one group's tables are live at a time."""
    def flat(e, op):
        return flat(e[1], op) + flat(e[2], op) if e[0] == op else [e]

    def polys_of(e):
        if e[0] == "p":
            return {e[1]}
        if e[0] in ("c", "k", "kc"):
            return set()
        return polys_of(e[1]) | polys_of(e[2])

    def cx(e):
        if e[0] == "p":
            return f"P{e[1]}_L"
        if e[0] == "c":
            return f"ef_c(coef + {e[1]} * EXT_D)"
        if e[0] == "k":
            return f"((uint32_t){int(e[1]) % P}u)"
        if e[0] == "kc":
            return f"((uint32_t)coef[{e[1]} * EXT_D])"
        f = {"+": "addx", "-": "subx", "*": "mulx"}[e[0]]
        return f"{f}({cx(e[1])}, {cx(e[2])})"

    facs = flat(prog.expr, "*")
    si = max(range(len(facs)), key=lambda i: len(flat(facs[i], "+")))
    outer = facs[:si] + facs[si + 1:]
    summ = flat(facs[si], "+")
    par = list(range(len(summ)))

    def find(i):
        while par[i] != i:
            i = par[i]
        return i
    owner = {}
    for i, s in enumerate(summ):
        for q in polys_of(s):
            if q in owner:
                par[find(i)] = find(owner[q])
            else:
                owner[q] = i
    groups = {}
    for i in range(len(summ)):
        groups.setdefault(find(i), []).append(summ[i])
    jobs = ([("O", outer)] if outer else []) + [("S", g) for g in groups.values()]

    def tabs_of(es):
        ps = set().union(*[polys_of(e) for e in es])
        return sorted(set(k for q in ps for m in prog.polys[q].t for k in m)), sorted(ps)

    if use_w:
        L.append("    EF w = mul(ef_load(EL, x & lomask, ELn), ef_load(EH, x >> lo_bits, EHn));")
    done = set()
    allt = set(k for _, es in jobs for k in tabs_of(es)[0])
    for k in range(len(kinds)):
        if k not in allt and any("ef_store" in s for s in ld[k]):
            L.extend(ld[k])
            done.add(k)
    L.append(f"    EF S[{npts}];")
    L.append(f"    for (int p = 0; p < {npts}; p++) S[p] = ef_zero();")
    if outer:
        L.append(f"    EF Of[{npts}];")
    for kind, es in jobs:
        tabs, ps = tabs_of(es)
        L.append("    {")
        for k in tabs:
            L.extend(s for s in ld[k] if k not in done or "ef_store" not in s)
            done.add(k)
            if vary[k]:
                L.append(f"    auto d{k} = subx(b{k}, a{k});")
        L.append("    #pragma unroll 1")
        L.append(f"    for (int pi = 0; pi < {npts}; pi++) {{")
        for k in tabs:
            if vary[k]:
                L.append(f"      auto v{k}_L = addx(a{k}, xpv(d{k}, pi));")
        for q in ps:
            L.append(f"      auto P{q}_L = {_poly_cexpr(prog.polys[q], vary)};")
        if kind == "O":
            e = cx(es[0])
            for f in es[1:]:
                e = f"mulx({e}, {cx(f)})"
            L.append(f"      Of[pi] = tox({e});")
        else:
            e = cx(es[0])
            for f in es[1:]:
                e = f"addx({e}, {cx(f)})"
            L.append(f"      S[pi] = add(S[pi], tox({e}));")
        L.append("    }")
        L.append("    }")
    for k in range(len(kinds)):
        if k not in done and any("ef_store" in s for s in ld[k]):
            L.extend(ld[k])
    v = "mul(Of[p], S[p])" if outer else "S[p]"
    L.append(f"    for (int p = 0; p < {npts}; p++) acc[p] = add(acc[p], {'mul(w, ' + v + ')' if use_w else v});")
    L.append("  }")
    L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
    L.append("}")
    return "\n".join(L)


def _gen_grouped_terms(L, ld, terms, kinds, vary, npts, use_w):
    """Term lists: tables in every term form an outer factor; the remaining terms
    are grouped by shared tables, each group loading only its own tables."""
    refs = [list(r) for _, r in terms]
    U = set(refs[0]).intersection(*map(set, refs[1:])) if refs else set()
    U = sorted(k for k in U if all(r.count(k) == 1 for r in refs))
    rest = [[k for k in r if k not in U] for r in refs]
    par = list(range(len(terms)))

    def find(i):
        while par[i] != i:
            i = par[i]
        return i
    owner = {}
    for i, r in enumerate(rest):
        for k in r:
            if k in owner:
                par[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups = {}
    for i in range(len(terms)):
        groups.setdefault(find(i), []).append(i)

    def nm(k):
        return f"cf[{-1 - k}]" if k < 0 else (f"v{k}_L" if vary[k] else f"a{k}")
    jobs = []
    if U:
        jobs.append(("O", U, "Of[pi] = tox(%s);" % _horner({tuple(U): 1}, nm)))
    for g in groups.values():
        hp = {}
        for t in g:
            m = tuple(sorted(rest[t] + [-1 - t]))
            hp[m] = hp.get(m, 0) + 1
        tabs = sorted(set(k for t in g for k in rest[t]))
        jobs.append(("S", tabs, "S[pi] = add(S[pi], tox(%s));" % _horner(hp, nm)))
    if use_w:
        L.append("    EF w = mul(ef_load(EL, x & lomask, ELn), ef_load(EH, x >> lo_bits, EHn));")
    done = set()
    allt = set(k for _, tabs, _ in jobs for k in tabs)
    for k in range(len(kinds)):
        if k not in allt and any("ef_store" in s for s in ld[k]):
            L.extend(ld[k])
            done.add(k)
    L.append(f"    EF S[{npts}];")
    L.append(f"    for (int p = 0; p < {npts}; p++) S[p] = ef_zero();")
    if U:
        L.append(f"    EF Of[{npts}];")
    for kind, tabs, code in jobs:
        L.append("    {")
        for k in tabs:
            L.extend(s for s in ld[k] if k not in done or "ef_store" not in s)
            done.add(k)
            if vary[k]:
                L.append(f"    auto d{k} = subx(b{k}, a{k});")
        for k in tabs:
            if vary[k]:
                L.append(f"    auto v{k}_L = a{k};")
        L.append("    #pragma unroll")
        L.append(f"    for (int pi = 0; pi < {npts}; pi++) {{")
        for k in tabs:
            if vary[k]:
                L.append(f"      if (pi == 1) v{k}_L = addx(b{k}, d{k}); else if (pi > 1) v{k}_L = addx(v{k}_L, d{k});")
        L.append("      " + code)
        L.append("    }")
        L.append("    }")
    v = "mul(Of[p], S[p])" if U else "S[p]"
    L.append(f"    for (int p = 0; p < {npts}; p++) acc[p] = add(acc[p], {'mul(w, ' + v + ')' if use_w else v});")
    L.append("  }")
    L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
    L.append("}")
    return "\n".join(L)


def _gkr_lookups(terms, kinds):
    """L if terms are GKR layer products (optionally times a trailing block-weight table)."""
    if not terms:
        return 0
    refs = [tuple(r) for _, r in terms]
    n2 = sum(1 for r in refs if len(r) == 2)
    n3 = sum(1 for r in refs if len(r) == 3)
    if n2 == len(refs) and len(kinds) % 4 == 0:
        L = len(kinds) // 4
        if len(refs) == 3 * L and all(kinds[k] == FULL for k in range(len(kinds))):
            if all(refs[3 * l] == (4 * l, 4 * l + 3) and refs[3 * l + 1] == (4 * l + 1, 4 * l + 2)
                   and refs[3 * l + 2] == (4 * l + 2, 4 * l + 3) for l in range(L)):
                return L
    if n3 == len(refs) and len(kinds) >= 5 and (len(kinds) - 1) % 4 == 0:
        L = (len(kinds) - 1) // 4
        bw = len(kinds) - 1
        if len(refs) == 3 * L and all(kinds[k] == FULL for k in range(len(kinds))):
            if all(refs[3 * l] == (4 * l, 4 * l + 3, bw) and refs[3 * l + 1] == (4 * l + 1, 4 * l + 2, bw)
                   and refs[3 * l + 2] == (4 * l + 2, 4 * l + 3, bw) for l in range(L)):
                return L
    return 0


def _gen_gkr(L0, ld, kinds, vary, npts, use_w, nlook, flat=False):
    """One lookup at a time: 4 child tables -> a (p0 q1 + p1 q0) + (a lam) q0 q1.

    GKR views share (vb, vs, row stride). Pointers live in shared memory; the
    per-thread loop keeps four tables live.  flat: one thread per (pair, lookup) instead of
    per pair (rounds with few pairs: the lookups spread over blocks instead of a serial loop).
    """
    nt = len(kinds)
    bw = nt - 1 if nt == 4 * nlook + 1 else None
    sample = "\n".join(ld[0])
    do_fold = "ef_store" in sample
    raw_b = "uint32_t a0" in sample
    scale = 4 if do_fold else 2
    pre = [
        f"  __shared__ const uint32_t* sT[{nt}];",
        "  if (threadIdx.x == 0) {",
    ]
    for k in range(nt):
        pre.append(f"    sT[{k}] = T{k};")
    pre.append("  }")
    if do_fold:
        pre += [f"  __shared__ uint32_t* sO[{nt}];", "  if (threadIdx.x == 0) {"]
        for k in range(nt):
            pre.append(f"    sO[{k}] = O{k};")
        pre.append("  }")
    pre.append("  __syncthreads();")
    # L0 is header + "for (x) {"; inject shared setup before the grid-stride loop
    for i in range(len(L0) - 1, -1, -1):
        if L0[i].lstrip().startswith("for (int64_t x"):
            hdr, rest = L0[:i], L0[i:]
            break
    else:
        hdr, rest = L0, []
    if flat:
        assert len(rest) == 1
        rest = ["  for (int64_t wi = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; wi < npairs * %d;"
                " wi += (int64_t)gridDim.x * blockDim.x) {" % nlook,
                "    const int64_t x = wi % npairs; const int lf = (int)(wi / npairs);"]
    L = hdr + pre + rest
    if use_w:
        L.append("    EF w = mul(ef_load(EL, x & lomask, ELn), ef_load(EH, x >> lo_bits, EHn));")
    L.append(f"    int64_t e = (({scale} * x) >> vb0) * vs0 + (({scale} * x) & ((1ll << vb0) - 1));")

    def load_ab(idx, a, b, indent="    "):
        if do_fold:
            return [
                f"{indent}EF {a}t0, {a}t1, {b}t0, {b}t1;",
                f"{indent}ef_load2(sT[{idx}], e, Tn0, {a}t0, {a}t1);",
                f"{indent}ef_load2(sT[{idx}], e + 2, Tn0, {b}t0, {b}t1);",
                f"{indent}EF {a} = ef_lerp({a}t0, {a}t1, rp), {b} = ef_lerp({b}t0, {b}t1, rp);",
                f"{indent}ef_store2(sO[{idx}], 2 * x, On0, {a}, {b});",
            ]
        if raw_b:
            return [f"{indent}uint32_t {a} = sT[{idx}][e], {b} = sT[{idx}][e + 1];"]
        return [f"{indent}EF {a}, {b}; ef_load2(sT[{idx}], e, Tn0, {a}, {b});"]

    if bw is not None:
        L.extend(load_ab(str(bw), "abw", "bbw"))
        if npts >= 2:
            L.append("    auto dbw = subx(bbw, abw);")
            L.append("    auto vbw_2 = addx(bbw, dbw);")
            for p in range(3, npts + 1):
                L.append(f"    auto vbw_{p} = addx(vbw_{p - 1}, dbw);")
    if flat:
        L.append("    { const int l = lf;")
    else:
        L.append("    #pragma unroll 1")
        L.append(f"    for (int l = 0; l < {nlook}; l++) {{")
    for i, name in enumerate(("p0", "p1", "q0", "q1")):
        L.extend(load_ab(f"4 * l + {i}", f"a{name}", f"b{name}", "      "))
    pts = [0] + list(range(2, npts + 1))

    def val(name, p):
        if p == 0:
            return f"a{name}"
        if p == 2:
            return f"addx(b{name}, subx(b{name}, a{name}))"
        return f"addx(a{name}, mulx(subx(b{name}, a{name}), {p}u))"

    for pi, p in enumerate(pts):
        p0, p1, q0, q1 = val("p0", p), val("p1", p), val("q0", p), val("q1", p)
        L.append(f"      {{ auto u = addx(mulx({p0}, {q1}), mulx({p1}, {q0})); "
                 f"auto v = mulx({q0}, {q1}); "
                 f"auto s = add(mulx(cf[3 * l], u), mulx(cf[3 * l + 2], v));")
        if bw is not None:
            bwv = "abw" if p == 0 else f"vbw_{p}"
            L.append(f"        s = mulx(s, {bwv});")
        e = "mul(w, s)" if use_w else "s"
        L.append(f"        acc[{pi}] = add(acc[{pi}], {e}); }}")
    L.append("    }")
    L.append("  }")
    L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
    L.append("}")
    return "\n".join(L)


def _gen_kernel(name, terms, kinds, state, npts, use_w, prog=None, flat=False):
    nt = len(kinds)
    if prog is not None:
        terms = [(None, list(m)) for p in prog.polys for m in p.t]
    params = []
    for k in range(nt):
        if kinds[k] == FULL:
            params += [f"const uint32_t* __restrict__ T{k}", f"int64_t Tn{k}",
                       f"uint32_t* __restrict__ O{k}", f"int64_t On{k}", f"int vb{k}", f"int64_t vs{k}"]
        else:
            params += [f"const uint32_t* __restrict__ T{k}", f"int64_t Tn{k}",
                       f"int sa{k}", f"int sb{k}", f"int sc{k}", f"int64_t mk{k}"]
    params += ["const uint32_t* __restrict__ EL", "int64_t ELn", "const uint32_t* __restrict__ EH",
               "int64_t EHn", "int lo_bits", "int64_t lomask", "const uint32_t* __restrict__ coef",
               "const uint32_t* __restrict__ rptr"]
    params += [f"uint32_t r{i}" for i in range(8)] + ["void* trst", "uint32_t* trmsg", "uint32_t* trr"]
    params += ["int64_t npairs", "uint32_t* __restrict__ out"]
    assert len(params) == nslots(nt)
    pts = [0] + list(range(2, npts + 1))
    # all arguments arrive as int64 slots of one struct (one kernel parameter)
    L = [f"struct ScArgs {{ long long v[{len(params)}]; }};",
         f'extern "C" __global__ void __launch_bounds__({NTHREADS}, {SC_MINB}) {name}(const ScArgs A) {{']
    for i, p in enumerate(params):
        ty, nm = p.rsplit(" ", 1)
        L.append(f"  {p} = ({ty.replace(' __restrict__', '')})A.v[{i}];")
    L.append("  EF acc[%d];" % npts)
    L.append("  for (int p = 0; p < %d; p++) acc[p] = ef_zero();" % npts)
    L.append("  EF rp = rptr ? ef_load(rptr, 0, 1) : R8VAL;")
    if "fold_b2" in state:               # the previous round's challenge: the row before rptr (device rho)
        L.append("  EF rq = ef_load(rptr - EXT_D, 0, 1);")
    nterm = len(terms) if prog is None else len(prog.coefs)
    L.append(f"  EF cf[{max(1, nterm)}];")
    for t in range(nterm):
        L.append(f"  cf[{t}] = ef_c(coef + {t} * EXT_D);")
    L.append("  for (int64_t x = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; x < npairs;"
             " x += (int64_t)gridDim.x * blockDim.x) {")
    vary = {}
    hdr = len(L)
    ld = {k: [] for k in range(nt)}
    for k in range(nt):
        st = state[k]
        if kinds[k] == FULL:
            vary[k] = True
            if st in ("raw_b", "raw_e"):
                ld[k].append(f"    int64_t e{k} = ((2 * x) >> vb{k}) * vs{k} + ((2 * x) & ((1ll << vb{k}) - 1));")
            if st == "raw_b":
                ld[k].append(f"    uint32_t a{k} = T{k}[e{k}], b{k} = T{k}[e{k} + 1];")
            elif st == "raw_e":
                ld[k].append(f"    EF a{k} = ef_load(T{k}, e{k}, Tn{k}), b{k} = ef_load(T{k}, e{k} + 1, Tn{k});")
            elif st in ("fold_b", "lerp_b"):
                ld[k].append(f"    int64_t e{k} = ((4 * x) >> vb{k}) * vs{k} + ((4 * x) & ((1ll << vb{k}) - 1));")
                ld[k].append(f"    EF a{k} = lerp_b(T{k}[e{k}], T{k}[e{k} + 1], rp);")
                ld[k].append(f"    EF b{k} = lerp_b(T{k}[e{k} + 2], T{k}[e{k} + 3], rp);")
                if st == "fold_b":
                    ld[k].append(f"    ef_store(O{k}, 2 * x, On{k}, a{k}); ef_store(O{k}, 2 * x + 1, On{k}, b{k});")
            elif st == "fold_b2":
                # round 2 straight from the base table (round 1 stored nothing): two folds, r_{j-2} then r_{j-1}
                ld[k].append(f"    int64_t e{k} = ((8 * x) >> vb{k}) * vs{k} + ((8 * x) & ((1ll << vb{k}) - 1));")
                ld[k].append(f"    EF a{k} = ef_lerp(lerp_b(T{k}[e{k}], T{k}[e{k} + 1], rq), "
                             f"lerp_b(T{k}[e{k} + 2], T{k}[e{k} + 3], rq), rp);")
                ld[k].append(f"    EF b{k} = ef_lerp(lerp_b(T{k}[e{k} + 4], T{k}[e{k} + 5], rq), "
                             f"lerp_b(T{k}[e{k} + 6], T{k}[e{k} + 7], rq), rp);")
                ld[k].append(f"    ef_store(O{k}, 2 * x, On{k}, a{k}); ef_store(O{k}, 2 * x + 1, On{k}, b{k});")
            else:
                ld[k].append(f"    int64_t e{k} = ((4 * x) >> vb{k}) * vs{k} + ((4 * x) & ((1ll << vb{k}) - 1));")
                ld[k].append(f"    EF a{k} = ef_lerp(ef_load(T{k}, e{k}, Tn{k}), ef_load(T{k}, e{k} + 1, Tn{k}), rp);")
                ld[k].append(f"    EF b{k} = ef_lerp(ef_load(T{k}, e{k} + 2, Tn{k}), ef_load(T{k}, e{k} + 3, Tn{k}), rp);")
                ld[k].append(f"    ef_store(O{k}, 2 * x, On{k}, a{k}); ef_store(O{k}, 2 * x + 1, On{k}, b{k});")
        else:
            mode, ty = st
            ld[k].append(f"    int64_t i{k} = ((x >> sa{k}) << sb{k}) | ((x >> sc{k}) & mk{k});")
            if mode == "vary":
                vary[k] = True
                if ty == "b":
                    ld[k].append(f"    uint32_t a{k} = T{k}[2 * i{k}], b{k} = T{k}[2 * i{k} + 1];")
                else:
                    ld[k].append(f"    EF a{k} = ef_load(T{k}, 2 * i{k}, Tn{k}), b{k} = ef_load(T{k}, 2 * i{k} + 1, Tn{k});")
            else:
                vary[k] = False
                if ty == "b":
                    ld[k].append(f"    uint32_t a{k} = T{k}[i{k}];")
                else:
                    ld[k].append(f"    EF a{k} = ef_load(T{k}, i{k}, Tn{k});")
    for k in range(nt):
        L.extend(ld[k])
    used = set(k for _, refs in terms for k in refs)
    for k in range(nt):
        if not vary[k] or k not in used:
            continue
        L.append(f"    auto d{k} = subx(b{k}, a{k});")
        L.append(f"    auto v{k}_0 = a{k};")
        if npts >= 2:
            L.append(f"    auto v{k}_2 = addx(b{k}, d{k});")
        for p in range(3, npts + 1):
            L.append(f"    auto v{k}_{p} = addx(v{k}_{p - 1}, d{k});")
    if use_w:
        L.append("    EF w = mul(ef_load(EL, x & lomask, ELn), ef_load(EH, x >> lo_bits, EHn));")
    if prog is not None and sum(len(p.t) for p in prog.polys) > LOOP_MONOS and not os.environ.get("ZK_SCOLD"):
        return _gen_grouped(L[:hdr], ld, prog, kinds, vary, npts, use_w)
    if prog is not None and sum(len(p.t) for p in prog.polys) > LOOP_MONOS:
        # long programs: one copy of the body, looping over evaluation points (NVRTC
        # compile time grows superlinearly with straight-line code)
        L.append("    #pragma unroll 1")
        L.append(f"    for (int pi = 0; pi < {npts}; pi++) {{")
        for k in range(nt):
            if vary[k] and k in used:
                L.append(f"      auto v{k}_L = addx(a{k}, xpv(d{k}, pi));")
        for i, poly in enumerate(prog.polys):
            L.append(f"      auto P{i}_L = {_poly_cexpr(poly, vary)};")
        ce = prog.cexpr("L")
        L.append(f"      acc[pi] = add(acc[pi], {'mul(w, ' + ce + ')' if use_w else ce});")
        L.append("    }")
        L.append("  }")
        L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
        L.append("}")
        return "\n".join(L)
    if prog is not None:
        for pi, p in enumerate(pts):
            for i, poly in enumerate(prog.polys):
                e = _poly_cexpr(poly, vary, f"_{p}")
                L.append(f"    auto P{i}_{p} = {e};")
            ce = prog.cexpr(p)
            L.append(f"    acc[{pi}] = add(acc[{pi}], {'mul(w, ' + ce + ')' if use_w else ce});")
        L.append("  }")
        L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
        L.append("}")
        return "\n".join(L)
    ng = _gkr_lookups(terms, kinds)
    if ng and not os.environ.get("ZK_NOGKR"):
        return _gen_gkr(L[:hdr], ld, kinds, vary, npts, use_w, ng, flat)
    if len(kinds) > 8 and not os.environ.get("ZK_SCOLD"):
        return _gen_grouped_terms(L[:hdr], ld, terms, kinds, vary, npts, use_w)
    if not os.environ.get("ZK_NOHORNER"):
        # coefficient t is the symbol -1-t
        hp = {}
        for t, (_, refs) in enumerate(terms):
            m = tuple(sorted(list(refs) + [-1 - t]))
            hp[m] = hp.get(m, 0) + 1
        for pi, p in enumerate(pts):
            e = _horner(dict(hp), lambda k: f"cf[{-1 - k}]" if k < 0 else (f"v{k}_{p}" if vary[k] else f"a{k}"))
            e = f"tox({e})"
            L.append(f"    acc[{pi}] = add(acc[{pi}], {'mul(w, ' + e + ')' if use_w else e});")
        L.append("  }")
        L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
        L.append("}")
        return "\n".join(L)
    for pi, p in enumerate(pts):
        parts = []
        for t, (_, refs) in enumerate(terms):
            if not refs:
                parts.append(f"cf[{t}]")
                continue
            fac = [(f"v{k}_{p}" if vary[k] else f"a{k}") for k in refs]
            expr = fac[0]
            for f in fac[1:]:
                expr = f"mulx({expr}, {f})"
            parts.append(f"mulx(cf[{t}], {expr})")
        e = parts[0]
        for q in parts[1:]:
            e = f"add({e}, {q})"
        if use_w:
            L.append(f"    acc[{pi}] = add(acc[{pi}], mul(w, {e}));")
        else:
            L.append(f"    acc[{pi}] = add(acc[{pi}], {e});")
    L.append("  }")
    L.append(f"  block_reduce_store<{npts}>(acc, out, (B3S*)trst, trmsg, trr);")
    L.append("}")
    return "\n".join(L)


_KCACHE = {}
_HDR = ('#include "sc_common.cuh"\n#define R8ARGS uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, '
        'uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7\n')


# Round kernels with long sources (per-row domains: many columns, few rows) take minutes to
# optimize; when the sumcheck has at most SMALL_PAIRS pairs they are built in fast-compile
# mode, whose slower code costs microseconds there.
SMALL_PAIRS = 1 << 13
# GKR rounds with at most this many pairs run one thread per (pair, lookup) (_gen_gkr flat)
FLAT_PAIRS = int(os.environ.get("ZK_FLAT", "32768"))
# base FULL tables: no stored round-1 fold, round 2 folds twice from the base table (see _prove)
FOLD2 = bool(int(os.environ.get("ZK_FOLD2", "1")))
LONG_SRC = 64 << 10


# where the round kernels keep the ext coefficients: "smem" (shared memory, read at each use), "reg"
# (an EF array in registers for the whole loop: up to ~160 registers in large domain programs, which
# then run at one block per SM and spill), "gmem" (global memory, read at each use).
# 1x4K, GPU time of one layer's prove pass: reg 0.889 s, smem 0.857 s.
CF_MODE = os.environ.get("ZK_CFMODE", "smem")


def _cf_mode(src):
    import re
    m = re.search(r"\n  EF cf\[(\d+)\];\n", src)
    if CF_MODE == "reg" or m is None:
        return src
    n = int(m.group(1))
    src = re.sub(r"\n  cf\[\d+\] = ef_c\(coef \+ \d+ \* EXT_D\);", "", src)
    if CF_MODE == "smem":
        decl = (f"\n  __shared__ uint32_t scf[{n} * EXT_D];\n"
                f"  for (int i_ = threadIdx.x; i_ < {n} * EXT_D; i_ += blockDim.x) scf[i_] = coef[i_];\n"
                f"  __syncthreads();\n")
        base = "scf"
    else:
        decl, base = "\n", "coef"
    src = src[:m.start()] + decl + src[m.end():]
    return re.sub(r"\bcf\[([^\]]+)\]", lambda g: f"ef_c({base} + ({g.group(1)}) * EXT_D)", src)


def _get_kernel(D, terms_struct, kinds, state, npts, use_w, prog=None, small=False, flat=False):
    key = (D, terms_struct if prog is None else prog.key(), tuple(kinds), tuple(state), npts, use_w, small, flat)
    if key not in _KCACHE:
        terms = [(None, list(r)) for r in terms_struct] if prog is None else None
        src = _cf_mode(_gen_kernel("sc_round", terms, kinds, state, npts, use_w, prog, flat))
        mod = kern.compile_module(_HDR + src, D, fast=small and len(src) > LONG_SRC)
        _KCACHE[key] = mod.get_function("sc_round")
    return _KCACHE[key]


# --------------------------------------------------------------- tail
# Once a sumcheck has at most TAIL_PAIRS pairs left, the remaining rounds run in one single-block
# kernel (sc_tail): each round evaluates the round body over dense ext tables, reduces, runs the
# transcript step on warp 0 and folds every table, without host round trips or launches per round.
TAIL_PAIRS = int(os.environ.get("ZK_TAIL", "0"))   # opt-in: no gain measured (e.g. ZK_TAIL=256)
_TCACHE = {}

_TAIL_DRIVER = r"""
struct TailP { long long v[10]; };
extern "C" __global__ void __launch_bounds__(%(NT)d, 1) sc_tail(const ScArgs A0, const TailP Q) {
  __shared__ ScArgs A;
  __shared__ uint32_t sums[%(NPTS)d * EXT_D];
  __shared__ uint32_t shr[%(NT)d / 32][%(NPTS)d * EXT_D];
  const int nt = (int)Q.v[0], nr = (int)Q.v[1];
  const int64_t S = Q.v[2];
  uint32_t* src = (uint32_t*)Q.v[3];
  uint32_t* dst = (uint32_t*)Q.v[4];
  const int64_t* meta = (const int64_t*)Q.v[5];
  B3S* trst = (B3S*)Q.v[6];
  uint32_t* FB = (uint32_t*)Q.v[7];
  const int64_t FBn = Q.v[8];
  const int64_t* fbo = (const int64_t*)Q.v[9];
  const int base = 6 * nt;
  const int NS = (int)(sizeof(ScArgs) / 8);
  for (int i = threadIdx.x; i < NS; i += blockDim.x) A.v[i] = A0.v[i];
  __syncthreads();
  for (int r = 0; r < nr; r++) {
    const int64_t* m = meta + 7 * r;
    for (int k = threadIdx.x; k < nt; k += blockDim.x) {
      A.v[6 * k] = (long long)(src + (int64_t)k * EXT_D * S);
      A.v[6 * k + 1] = S; A.v[6 * k + 2] = 0; A.v[6 * k + 3] = 0; A.v[6 * k + 4] = 62; A.v[6 * k + 5] = 0;
    }
    if (threadIdx.x == 0) {
      A.v[base] = m[0]; A.v[base + 2] = m[1]; A.v[base + 4] = m[2]; A.v[base + 5] = m[3];
      A.v[NS - 2] = m[4];
    }
    __syncthreads();
    EF acc[%(NPTS)d];
    sc_body(A, acc);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int p = 0; p < %(NPTS)d; p++) {
      EF v = ef_warp_sum(acc[p]);
      if (l == 0) {
        const uint32_t* q = (const uint32_t*)&v;
        for (int k = 0; k < EXT_D; k++) shr[w][p * EXT_D + k] = q[k];
      }
    }
    __syncthreads();
    if (threadIdx.x < %(NPTS)d * EXT_D) {
      uint32_t t = 0;
      for (int i = 0; i < (int)(blockDim.x >> 5); i++) t = fadd(t, shr[i][threadIdx.x]);
      sums[threadIdx.x] = t;
    }
    __syncthreads();
    if (threadIdx.x < 32) b3w_round(trst, sums, %(NPTS)d * EXT_D, (uint32_t*)m[5], (uint32_t*)m[6], EXT_D);
    __syncthreads();
    const EF rr = ef_load((const uint32_t*)m[6], 0, 1);
    const int64_t half = m[4];
    for (int64_t e = threadIdx.x; e < (int64_t)nt * half; e += blockDim.x) {
      const int k = (int)(e / half);
      const int64_t i = e - k * half;
      const uint32_t* s_ = src + (int64_t)k * EXT_D * S;
      uint32_t* d_ = dst + (int64_t)k * EXT_D * S;
      ef_store(d_, i, S, ef_lerp(ef_load(s_, 2 * i, S), ef_load(s_, 2 * i + 1, S), rr));
    }
    __syncthreads();
    uint32_t* t = src; src = dst; dst = t;
  }
  for (int k = 0; k < nt; k++) {
    const int64_t off = fbo[2 * k], sz = fbo[2 * k + 1];
    const uint32_t* s_ = src + (int64_t)k * EXT_D * S;
    for (int64_t e = threadIdx.x; e < sz * EXT_D; e += blockDim.x) {
      const int64_t d = e / sz, i = e - d * sz;
      FB[d * FBn + off + i] = s_[d * S + i];
    }
  }
}
"""


def _get_tail_kernel(D, terms_struct, nt, npts, use_w, prog=None):
    key = (D, terms_struct if prog is None else prog.key(), nt, npts, use_w)
    if key not in _TCACHE:
        terms = [(None, list(r)) for r in terms_struct] if prog is None else None
        body = _gen_kernel("sc_body", terms, [FULL] * nt, ["raw_e"] * nt, npts, use_w, prog)
        out = []
        for ln in body.split("\n"):
            if ln.startswith('extern "C" __global__ void') and "sc_body(const ScArgs A)" in ln:
                ln = "__device__ __forceinline__ void sc_body(const ScArgs& A, EF* accout) {"
            elif "block_reduce_store<" in ln:
                ln = f"  for (int p_ = 0; p_ < {npts}; p_++) accout[p_] = acc[p_];"
            out.append(ln)
        src = "\n".join(out) + _TAIL_DRIVER % {"NT": NTHREADS, "NPTS": npts}
        mod = kern.compile_module(_HDR + src, D, fast=len(src) > LONG_SRC)
        _TCACHE[key] = mod.get_function("sc_tail")
    return _TCACHE[key]


_TP = np.dtype([("v", np.int64, 10)])


def _tail(G, D, n, j0, K, tables, cur, targs, state, fidx, npts, use_w, eq_r, EL, EH, wmeta, cptr,
          sbuf, sv, base, dtr, rbase, terms_struct, prog):
    """Rounds j0 .. n-1 of _prove in one sc_tail launch (see TAIL_PAIRS).  Returns (FB, slices) as
    _prove's finals, or None when a table's finals do not map onto the dense per-block layout."""
    nt = len(tables)
    S = K << (n - j0)
    nr = n - j0
    szs = [K] * nt
    for k, t in enumerate(tables):
        if t.kind == FULL:
            continue
        L0 = cur[k].shape[-1] if cur[k].dim() == 2 else cur[k].numel()
        nv = (n - max(j0, t.c)) if t.kind == ROW else max(0, t.c - j0)
        szs[k] = L0 >> nv
        if szs[k] not in (1, K) or (szs[k] == 1 and K > 1 and not t.shared):
            return None
    # one upload: desc (nt, 8) | meta (nr, 7) | fbo (nt, 2)
    hb = np.zeros(8 * nt + 7 * nr + 2 * nt, dtype=np.int64)
    desc = hb[:8 * nt].reshape(nt, 8)
    desc[:, 2:] = targs
    for k, t in enumerate(tables):
        if t.kind == FULL:
            desc[k, 0] = 0 if j0 == 0 else 1
            desc[k, 1] = state[k] in ("raw_b", "fold_b")
            desc[k, 4:6] = targs[k, 4:6]
            desc[k, 6:] = 0
        else:
            desc[k, 0] = 2 if state[k][0] == "vary" else 3
            desc[k, 1] = state[k][1] == "b"
    meta = hb[8 * nt:8 * nt + 7 * nr].reshape(nr, 7)
    jj = np.arange(j0, n)
    if use_w:
        wm = np.asarray(wmeta[j0:], dtype=np.int64)
        meta[:, 0] = EL.data_ptr() + 4 * wm[:, 0]
        meta[:, 1] = EH.data_ptr() + 4 * wm[:, 2]
        meta[:, 2] = wm[:, 1]
        meta[:, 3] = ((1 << wm[:, 1]) - 1) if eq_r is not None else 0
    meta[:, 4] = K << (n - jj - 1)
    meta[:, 5] = dtr.alloc_many(nr, npts * D, (npts, D)) + 4 * npts * D * np.arange(nr)
    meta[:, 6] = rbase + 4 * D * jj
    order = fidx + [k for k in range(nt) if k not in fidx]
    fbo = hb[8 * nt + 7 * nr:].reshape(nt, 2)
    pos = 0
    for k in order:
        fbo[k] = (pos, szs[k])
        pos += szs[k]
    hd = h2d(hb)
    dbuf = torch.empty((2, nt, D, S), dtype=torch.int32, device=_dev())
    kern.launch(G.f["tail_expand"], (int(max(1, min((S + 255) // 256, 64))), nt), 256,
                [ptr(hd), np.int64(S), ptr(dbuf[0]), np.uint64(rbase + 4 * D * (j0 - 1) if j0 else 0)])
    FB = torch.empty((D, pos), dtype=torch.int32, device=_dev())
    sv[:] = 0
    if use_w:
        sv[base + 1] = EL.stride(0)
        sv[base + 3] = EH.stride(0)
    sv[base + 6] = cptr
    tp = np.zeros((), dtype=_TP)
    p0 = hd.data_ptr()
    tp["v"] = (nt, nr, S, dbuf[0].data_ptr(), dbuf[1].data_ptr(), p0 + 64 * nt, dtr.st.data_ptr(),
               FB.data_ptr(), FB.stride(0), p0 + 8 * (8 * nt + 7 * nr))
    fn = _get_tail_kernel(D, terms_struct, nt, npts, use_w, prog)
    fn((1,), (NTHREADS,), (sbuf[()], tp[()]))
    return FB, [slice(int(fbo[k, 0]), int(fbo[k, 0] + fbo[k, 1])) for k in range(nt)], (dbuf, hd)


# --------------------------------------------------------------- helpers
_HBUF = {}


def _host_buf(n):
    if n not in _HBUF:
        _HBUF[n] = torch.empty(n, dtype=torch.int32, pin_memory=True)
    return _HBUF[n]


def _next_claim(F, claim, msg, rj, x, d):
    """msg rows: g(0), g(2..npts).  Returns g(x) as numpy (D,)."""
    c = F.c(claim)
    g0 = F.c(msg[0])
    if rj is not None:
        r = F.c(rj)
        g1 = F.mul(F.sub(c, F.mul(F.sub(F.one, r), g0)), F.inv(r))
    else:
        g1 = F.sub(c, g0)
    vals = [g0, g1] + [F.c(m) for m in msg[1:]]
    return np.array(F.lagrange(vals[:max(d, 1) + 1], F.c(x)), dtype=np.int64)


_SEGC = {}
_STREAM = []


def _stream():
    if not _STREAM:
        _STREAM.append(torch.cuda.current_stream())
    return _STREAM[0]


def eq_segments(G, r, segs, offs, total):
    if getattr(G, "cpu", False):
        return G.eq_segments(r, segs, offs, total)
    D = G.D
    from .lazy import Lz
    if isinstance(r, Lz):            # host layout (D, k) on the device -> rows (k, D)
        r = r.dev.T.contiguous()
    if torch.is_tensor(r):
        rbuf = r.reshape(-1) if r.shape[0] else torch.zeros(D, dtype=torch.int32, device=_dev())
    else:
        rbuf = G.to_dev(r.T.reshape(-1)) if r.shape[1] else G.to_dev(np.zeros(D))
    key = (tuple(segs), tuple(offs))
    if key not in _SEGC:
        if len(_SEGC) > 4096:
            _SEGC.clear()
        _SEGC[key] = (h2d(np.array([s for s, _ in segs], dtype=np.int32)),
                      h2d(np.array([l for _, l in segs], dtype=np.int32)), h2d(np.array(offs, dtype=np.int64)))
    st, ln, of = _SEGC[key]
    buf = torch.empty((D, total), dtype=torch.int32, device=_dev())
    mx = max(l for _, l in segs)
    gx = int(max(1, min(((1 << mx) + 255) // 256, 512)))
    kern.launch(G.f["eq_segments"], (gx, len(segs)), 256,
                [ptr(rbuf), ptr(st), ptr(ln), ptr(of), ptr(buf), i64(total)])
    return buf


def _weight_tables(G, r, n, K, bw):
    """Per round j: w(x') = EL_j[x' & m] * EH_j[x' >> lo] over pairs x' = (b, y),
    y in {0,1}^(n-j-1), w = bw_b eq(r_{j+1..}, y).  Returns (EL, EH, meta)."""
    D = G.D
    h = n // 2
    if r is None:
        r = np.zeros((D, 0), dtype=np.int64)
        h = 0
    segs, offs = [(0, 0)], [0]
    off = 1
    hi_meta = []                     # per round: (el_off, lo_bits, eh_seg)

    def put(st, ln):
        nonlocal off
        segs.append((st, ln))
        offs.append(off)
        o = off
        off += 1 << ln
        return o, len(segs) - 1

    fixed = None
    for j in range(n):
        if j + 1 < h:
            if fixed is None:
                fixed = put(h, n - h)[1]
            el_off, _ = put(j + 1, h - 1 - j)
            hi_meta.append((el_off, h - 1 - j, fixed))
        elif (r.shape[0] if torch.is_tensor(r) else r.shape[1]):
            hi_meta.append((0, 0, put(j + 1, n - j - 1)[1] if j + 1 < n else 0))
        else:
            hi_meta.append((0, n - j - 1, 0))       # plain mode: EH = bw only
    buf = eq_segments(G, r, segs, offs, off)
    if K == 1 and bw is None:
        return buf, buf, [(el, lo, offs[s]) for el, lo, s in hi_meta]
    # EH tables with block weights: bw_b * seg[y]
    if bw is None:
        bwd = G.from_base(torch.ones(K, dtype=torch.int32, device=_dev()))
    else:
        bwd = bw if torch.is_tensor(bw) else G.to_dev(bw)
    parts, eh_off, cache = [], 0, {}
    meta = []
    for el, lo, s in hi_meta:
        if s not in cache:
            cache[s] = eh_off
            parts += [offs[s], segs[s][1], eh_off]
            eh_off += K << segs[s][1]
        meta.append((el, lo, cache[s]))
    EH = torch.empty((D, eh_off), dtype=torch.int32, device=_dev())
    mx = max(K << segs[s][1] for s in cache)
    kern.launch(G.f["eh_build"], (int(max(1, min((mx + 255) // 256, 1024))), len(cache)), 256,
                [ptr(bwd), i64(bwd.stride(0)), i64(K), ptr(buf), i64(buf.stride(0)),
                 ptr(h2d(np.array(parts, dtype=np.int64))), ptr(EH), i64(EH.stride(0))])
    return buf, EH, meta


def _materialize(G, t, nel):
    """Contiguous copy of a viewed FULL table with nel logical elements."""
    vb, vs, vo = t.view
    if vb < NOVIEW and nel % (1 << vb) == 0:
        x = t.data
        if x.dim() == 1:
            v = x.as_strided((nel >> vb, 1 << vb), (vs, 1), x.storage_offset() + vo)
            return v.reshape(nel)
        v = x.as_strided((x.shape[0], nel >> vb, 1 << vb), (x.stride(0), vs, 1), x.storage_offset() + vo)
        return v.reshape(x.shape[0], nel)
    e = torch.arange(nel, device=_dev(), dtype=torch.int64)
    idx = vo + (e >> vb) * vs + (e & ((1 << vb) - 1))
    return t.data[idx] if t.data.dim() == 1 else t.data[:, idx]


# --------------------------------------------------------------- prover
SCLOG = {}
# The prover never needs the reduced claim; tracking it is a self-check.
TRACK = bool(os.environ.get("ZK_TRACK"))
GPU_TR = not os.environ.get("ZK_HOSTTR")
NVTX = bool(os.environ.get("ZK_NVTX"))
RTIME = int(os.environ.get("ZK_RTIME", "0"))


def _session(tr):
    """The transcript's active device session (host Transcript.begin_device), or None."""
    return None if isinstance(tr, DevTr) else getattr(tr, "dev", None)


def _lazy_out(rho, FB, sl):
    from .lazy import Lz, rows_lz
    if FB is None:
        return rows_lz(rho), []
    fb = Lz(FB)                     # one transfer serves every table's finals
    return rows_lz(rho), [fb[:, s] for s in sl]


def prove(tr, D, n, tables, terms, claim, eq_r=None, K=1, bw=None, label=b"sc", prog=None, on_round=None,
          dev=False, coefs_dev=None):
    ss = _session(tr)
    if ss is not None and not TRACK and on_round is None:
        # run on the session; host-format results stay lazy (no synchronization)
        rho, (FB, sl), red = prove(ss, D, n, tables, terms, claim, eq_r, K, bw, label, prog, None, True, coefs_dev)
        if dev:
            return rho, (FB, sl), red
        r_, f_ = _lazy_out(rho, FB if tables else None, sl)
        return r_, f_, None
    """tables: list of Tab; terms: list of (coef numpy (D,), [table idx]).
    Returns (rho (D, n), finals: list of numpy (D, K_t), reduced claim); with dev=True the
    finals stay on the device as one (D, total) tensor and per-table column slices."""
    if NVTX:
        import re
        torch.cuda.nvtx.range_push("sc:" + re.sub(rb"[0-9]+", b"#", label).decode(errors="replace"))
        try:
            return _prove(tr, D, n, tables, terms, claim, eq_r, K, bw, label, prog, on_round, dev, coefs_dev)
        finally:
            torch.cuda.nvtx.range_pop()
    if not os.environ.get("ZK_SCLOG"):
        return _prove(tr, D, n, tables, terms, claim, eq_r, K, bw, label, prog, on_round, dev, coefs_dev)
    import re
    t0 = time.time()
    out = _prove(tr, D, n, tables, terms, claim, eq_r, K, bw, label, prog, on_round, dev, coefs_dev)
    key = re.sub(rb"[0-9]+", b"#", label).decode(errors="replace")
    e = SCLOG.setdefault(key, [0, 0, 0.0, 0])
    e[0] += 1
    e[1] += n
    e[2] += time.time() - t0
    e[3] += (K << n) * len(tables)
    return out


def _prove(tr, D, n, tables, terms, claim, eq_r=None, K=1, bw=None, label=b"sc", prog=None, on_round=None,
           dev=False, coefs_dev=None):
    G = gops(D)
    F = fe(D)
    kinds = [t.kind for t in tables]
    cs = [t.c for t in tables]
    if prog is not None:
        terms_struct = None
        d = prog.degree(kinds, cs, n)
        cl = prog.coefs
    else:
        terms_struct = tuple(tuple(r) for _, r in terms)
        d = max_degree(terms, kinds, cs, n)
        cl = [c for c, _ in terms]
    from .lazy import Lz
    if coefs_dev is not None:
        coefs = coefs_dev
    elif any(isinstance(c, Lz) or torch.is_tensor(c) for c in cl):     # device coefficients: no read
        coefs = torch.stack([c.dev.reshape(-1) if isinstance(c, Lz) else c.reshape(-1) if torch.is_tensor(c)
                             else G.to_dev(np.asarray(c, dtype=np.int64)) for c in cl]).reshape(-1)
    else:
        coefs = G.to_dev(np.stack([np.asarray(c, dtype=np.int64) for c in cl]).reshape(-1)) \
            if cl else G.to_dev(np.zeros(D))
    ext_tr = isinstance(tr, DevTr)
    use_w = eq_r is not None or bw is not None
    tr.absorb(label + b"|" + n.to_bytes(2, "little") + d.to_bytes(2, "little") + K.to_bytes(4, "little"))
    cur = [t.data for t in tables]
    claim_p = np.asarray(claim, dtype=np.int64) % P if TRACK else None
    rho = np.zeros((D, n), dtype=np.int64)
    rprev = np.zeros(D, dtype=np.int64)
    if use_w and n:
        EL, EH, wmeta = _weight_tables(G, eq_r, n, K, bw)
    npts = max(d, 1)
    # FULL fold outputs ping-pong inside one workspace: round j writes region j & 1
    fidx = [k for k, t in enumerate(tables) if t.kind == FULL]
    ws = None
    if n >= 2 and fidx:
        szA, szB = K << (n - 1), (K << (n - 2)) if n >= 3 else 0
        ws = torch.empty((D, len(fidx) * (szA + szB)), dtype=torch.int32, device=_dev())
        wbase, wstr = ws.data_ptr(), ws.stride(0)
        wreg = {k: (wbase + 4 * i * (szA + szB), wbase + 4 * (i * (szA + szB) + szA)) for i, k in enumerate(fidx)}
    gmax = int(max(1, min(((K << max(n - 1, 0)) + NTHREADS - 1) // NTHREADS, 188 * 8)))
    nlk = _gkr_lookups(terms, kinds) if prog is None and terms and not os.environ.get("ZK_NOGKR") else 0
    if nlk > 1:                       # flat GKR rounds: up to FLAT_PAIRS * nlk work items
        fw = min(FLAT_PAIRS, K << max(n - 1, 0)) * nlk
        gmax = max(gmax, int(min((fw + NTHREADS - 1) // NTHREADS, 188 * 8)))
    part = torch.empty((gmax + (gmax > 1), npts * D), dtype=torch.int32, device=_dev())
    pbase = part.data_ptr()
    hbuf = _host_buf(npts * D)
    cptr = coefs.data_ptr()
    sbuf, sv = _slot_buf(nslots(len(tables)))
    stream = _stream()
    # GPU transcript: all rounds are queued without host round trips
    gtr = ext_tr or (GPU_TR and not TRACK and on_round is None and n > 0)
    if gtr:
        dtr = tr if ext_tr else DevTr(tr, cap=max(1, n * npts * D))
        rho_d = torch.empty((n, D), dtype=torch.int32, device=_dev())
        rbase = rho_d.data_ptr()
    nt = len(tables)
    targs = np.zeros((nt, 6), dtype=np.int64)
    Fi = np.array(fidx, dtype=np.int64)
    nonfull = [k for k in range(nt) if tables[k].kind != FULL]
    if fidx:
        vw = [tables[k].view if tables[k].view else (NOVIEW, 0, 0) for k in fidx]
        p0 = np.array([cur[k].data_ptr() + 4 * v[2] for k, v in zip(fidx, vw)], dtype=np.int64)
        s0 = np.array([cur[k].stride(0) if cur[k].dim() == 2 else 0 for k in fidx], dtype=np.int64)
        vb0 = np.array([v[0] for v in vw], dtype=np.int64)
        vs0 = np.array([v[1] for v in vw], dtype=np.int64)
        z = np.zeros_like(p0)
        # base tables with FOLD2: round 1 folds on the fly without storing, round 2 folds twice from the
        # base table (less traffic than writing and re-reading the 4x larger ext table of round 1)
        f2 = [FOLD2 and tables[k].is_base and cur[k].dim() == 1 and vb0[i] >= 3 and n >= 3 and gtr
              and TAIL_PAIRS == 0 and ws is not None for i, k in enumerate(fidx)]
        fst = [["raw_" + ("b" if cur[k].dim() == 1 else "e") for k in fidx],
               [("lerp_b" if f2[i] else "fold_b") if tables[k].is_base else "fold_e" for i, k in enumerate(fidx)],
               ["fold_b2" if f2[i] else "fold_e" for i in range(len(fidx))], ["fold_e"] * len(fidx)]
        fargs = [np.stack([p0, s0, z, z, vb0, vs0], axis=1)]
        if ws is not None:
            W0 = np.array([wreg[k][0] for k in fidx], dtype=np.int64)
            W1 = np.array([wreg[k][1] for k in fidx], dtype=np.int64)
            ww = np.full_like(p0, wstr)
            fargs.append(np.stack([p0, s0, W0, ww, vb0, vs0], axis=1))
            fargs.append(np.stack([W0, ww, W1, ww, np.full_like(p0, NOVIEW), z], axis=1))
            fargs.append(np.stack([W1, ww, W0, ww, np.full_like(p0, NOVIEW), z], axis=1))
            fargs.append(fargs[2].copy())        # round 2 only (later even rounds use fargs[2])
            for i in range(len(fidx)):
                if f2[i]:                # round 2 reads the original base table
                    fargs[4][i] = (p0[i], s0[i], W1[i], wstr, vb0[i], vs0[i])
    state = [None] * nt
    fcache = {}
    base = 6 * nt
    tail_ok = gtr and not TRACK and on_round is None and TAIL_PAIRS > 0 and nt > 0
    tail_fb = None
    for j in range(n):
        m = n - j - 1
        ph = min(j, 3)
        if fidx:
            targs[Fi] = fargs[0 if j == 0 else (1 if j == 1 else (4 if j == 2 else 2 + (j & 1)))]
            if j <= 3:
                for i, k in enumerate(fidx):
                    state[k] = fst[ph][i]
        nfs = []
        for k in nonfull:
            t = tables[k]
            src = cur[k]
            ty = "b" if src.dim() == 1 else "e"
            stride = src.stride(0) if src.dim() == 2 else 0
            if t.kind == ROW:
                if j >= t.c:
                    st_ = ("vary", ty); sa, sb, sc_, mk = 63, 0, 0, ((1 << m) - 1) if t.shared else -1
                else:
                    st_ = ("const", ty)
                    sa, sb, sc_, mk = 63, 0, t.c - j - 1, ((1 << (n - t.c)) - 1) if t.shared else -1
            else:
                if j < t.c:
                    st_ = ("vary", ty)
                    sa, sb = (63, 0) if t.shared else (m, t.c - j - 1)
                    sc_, mk = 0, (1 << (t.c - j - 1)) - 1
                else:
                    st_ = ("const", ty)
                    sa, sb, sc_, mk = (63, 0, 0, 0) if t.shared else (m, 0, 0, 0)
            state[k] = st_
            nfs.append(st_)
            targs[k] = (src.data_ptr(), stride, sa, sb, sc_, mk)
        if tail_ok and (K << m) <= TAIL_PAIRS:
            tail_fb = _tail(G, D, n, j, K, tables, cur, targs, state, fidx, npts, use_w, eq_r,
                            EL if use_w else None, EH if use_w else None, wmeta if use_w else None,
                            cptr, sbuf, sv, base, dtr, rbase, terms_struct, prog)
            if tail_fb is not None:
                break
            tail_ok = False
        npairs = K << m
        flat = nlk > 1 and npairs <= FLAT_PAIRS
        fk = (ph, tuple(nfs), flat)
        fn = fcache.get(fk)
        if fn is None:
            fn = fcache[fk] = _get_kernel(D, terms_struct, kinds, state, npts, use_w, prog,
                                          small=(K << max(n - 1, 0)) <= SMALL_PAIRS, flat=flat)
        work = npairs * nlk if flat else npairs
        grid = int(max(1, min((work + NTHREADS - 1) // NTHREADS, 188 * 8)))
        sv[:base] = targs.reshape(-1)
        if use_w:
            el_off, lo_bits, eh_off = wmeta[j]
            sv[base:base + 6] = (EL.data_ptr() + 4 * el_off, EL.stride(0), EH.data_ptr() + 4 * eh_off, EH.stride(0),
                                 lo_bits, (1 << lo_bits) - 1 if eq_r is not None else 0)
        else:
            sv[base:base + 6] = 0
        sv[base + 6] = cptr
        sv[base + 7] = rbase + 4 * D * (j - 1) if gtr and j else 0
        sv[base + 8:base + 19] = 0
        if not gtr:
            sv[base + 8:base + 8 + D] = rprev
        else:
            # the round kernel's last block sends the sums and draws r_j (fused transcript step)
            rj = rbase + 4 * D * j
            sv[base + 16:base + 19] = (dtr.st.data_ptr(), dtr._alloc(npts * D, (npts, D)), rj)
        sv[-2] = npairs
        sv[-1] = pbase
        if RTIME and K == RTIME:
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        fn((grid,), (NTHREADS,), (sbuf[()],))
        if RTIME and K == RTIME:
            ev1.record(); ev1.synchronize()
            print(f"[rt] {label!r} n={n} j={j} npts={npts} nt={len(tables)} {ev0.elapsed_time(ev1):.3f} ms", flush=True)
        if gtr:
            fk_ = [k for k in nonfull if (tables[k].kind == COL and j < tables[k].c)
                   or (tables[k].kind == ROW and j >= tables[k].c)]
            if fk_:
                for k, o in zip(fk_, G.fold_list_p([cur[k] for k in fk_], rj)):
                    cur[k] = o
            continue
        hbuf.copy_(part[grid if grid > 1 else 0], non_blocking=True)
        stream.synchronize()
        sums = hbuf.numpy().view(np.uint32).astype(np.int64).reshape(npts, D)
        tr.send(sums)
        rj_ch = tr.challenge(D=D)
        rho[:, j] = rj_ch
        if TRACK:
            claim_p = _next_claim(F, claim_p, sums, None if eq_r is None else eq_r[:, j], rj_ch, d)
        rprev = rj_ch
        if on_round is not None:
            on_round(j, rj_ch)
        for k, t in enumerate(tables):
            if (t.kind == COL and j < t.c) or (t.kind == ROW and j >= t.c):
                cur[k] = G.base_fold(cur[k], rj_ch) if cur[k].dim() == 1 else G.fold(cur[k], rj_ch)
    if gtr and not ext_tr:
        rho_h = torch.empty((n, D), dtype=torch.int32, pin_memory=True)
        rho_h.copy_(rho_d, non_blocking=True)
        dtr.finish()
        rho = rho_h.numpy().view(np.uint32).astype(np.int64).T.copy()
    if ext_tr:
        assert dev and not TRACK
        rho = rho_d
        from .lazy import KEYS
        KEYS[rho_d] = dtr.draw_keys(n)          # one draw per round (in the round kernels)
    if tail_fb is not None:
        FB, sl = tail_fb[0], tail_fb[1]
        if dev:
            return rho, (FB, sl), claim_p
        allf = G.to_host(FB)
        return rho, [allf[:, s_] for s_ in sl], claim_p
    if not tables:
        return rho, ((None, []) if dev else []), claim_p
    sizes = [K if t.kind == FULL else (cur[k].shape[-1] if cur[k].dim() == 2 else cur[k].numel())
             for k, t in enumerate(tables)]
    order = fidx + [k for k in range(len(tables)) if k not in fidx]
    st, pos = {}, 0
    for k in order:
        st[k] = pos
        pos += sizes[k]
    FB = torch.empty((D, pos), dtype=torch.int32, device=_dev())
    if n >= 2 and fidx:
        c0 = 0 if (n & 1) == 0 else szA
        mm = len(fidx) * K
        if ext_tr:
            kern.launch(G.f["ext_fold_many_p"], int(max(1, min((mm + 255) // 256, 188 * 16))), 256,
                        [np.uint64(wbase + 4 * c0), i64(wstr), i64(szA + szB), i64(K), ptr(FB), i64(FB.stride(0)),
                         np.uint64(rbase + 4 * D * (n - 1)), i64(mm)])
        else:
            kern.launch(G.f["ext_fold_many"], int(max(1, min((mm + 255) // 256, 188 * 16))), 256,
                        [np.uint64(wbase + 4 * c0), i64(wstr), i64(szA + szB), i64(K), ptr(FB), i64(FB.stride(0)),
                         *G.r8(rho[:, n - 1]), i64(mm)])
    for k, t in enumerate(tables):
        if t.kind == FULL and n >= 2:
            continue
        v = cur[k]
        o = FB[:, st[k]:st[k] + sizes[k]]
        if t.kind == FULL and n == 1:
            if t.view:
                v = _materialize(G, t, K * 2)
            if ext_tr:
                v = G.base_fold_p(v, rbase) if v.dim() == 1 else G.fold_p(v, rbase)
            else:
                v = G.base_fold(v, rho[:, 0]) if v.dim() == 1 else G.fold(v, rho[:, 0])
        elif t.kind == FULL and t.view:
            v = _materialize(G, t, K)
        if v.dim() == 1:
            o[0].copy_(v)
            o[1:].zero_()
        else:
            o.copy_(v)
    if dev:
        return rho, (FB, [slice(st[k], st[k] + sizes[k]) for k in range(len(tables))]), claim_p
    allf = G.to_host(FB)
    return rho, [allf[:, st[k]:st[k] + sizes[k]] for k in range(len(tables))], claim_p


# --------------------------------------------------------------- block phase
# A K-block sumcheck followed by rounds over the block index: the in-block
# rounds leave sum_b bw_b E(finals_b), which is proved by a sumcheck over
# nbits(K) variables whose tables are the per-block finals (zero-padded) and
# bw.  Only one final per table is sent.
def nbits(K):
    return max(0, (K - 1).bit_length())


def _block_struct(prog, terms, nt):
    if prog is not None:
        return Prog(prog.polys + [Poly.var(nt)], ("*", ("p", len(prog.polys)), prog.expr), prog.coefs), None
    return None, [(c, list(r) + [nt]) for c, r in terms]


def _block_shapes(shared, lk):
    kinds = [ROW if s else FULL for s in shared] + [FULL]
    cs = [lk if s else 0 for s in shared] + [0]
    return kinds, cs


def block_degree(prog, terms, shared, lk):
    prog2, terms2 = _block_struct(prog, terms, len(shared))
    kinds, cs = _block_shapes(shared, lk)
    return prog2.degree(kinds, cs, lk) if prog2 is not None else max_degree(terms2, kinds, cs, lk)


def prove_blocks(tr, D, n, tables, terms, claim, K, bw, eq_r=None, label=b"sc", prog=None, coefs_dev=None):
    """Returns (rho (D, n + nbits(K)), finals [(D, 1)], reduced claim excluding bw~(rho_b)).
    On a device transcript: (rho (n + nbits(K), D) device, (FB, column slices), None)."""
    G = gops(D)
    lk = nbits(K)
    ss = _session(tr)
    if ss is not None and not TRACK:
        rho, (FB, sl), _ = prove_blocks(ss, D, n, tables, terms, claim, K, bw, eq_r, label, prog, coefs_dev)
        r_, f_ = _lazy_out(rho, FB, sl)
        return r_, f_, None
    ext_tr = isinstance(tr, DevTr)
    if lk == 0:
        return prove(tr, D, n, tables, terms, claim, eq_r=eq_r, K=K, bw=bw, label=label, prog=prog,
                     dev=ext_tr, coefs_dev=coefs_dev)
    rho, (FB, sl), red = prove(tr, D, n, tables, terms, claim, eq_r=eq_r, K=K, bw=bw, label=label, prog=prog,
                               dev=True, coefs_dev=coefs_dev)
    Kp = 1 << lk
    shared = [s.stop - s.start == 1 for s in sl]
    tabs = []
    ns = [s for s, sh in zip(sl, shared) if not sh] + [None]
    big = torch.zeros((D, len(ns), Kp), dtype=torch.int32, device=_dev())
    if len(ns) > 1:
        big[:, :-1, :K] = torch.stack([FB[:, s] for s in ns[:-1]], dim=1)
    big[:, -1, :K] = bw if torch.is_tensor(bw) else G.to_dev(bw)
    i = 0
    for s, sh in zip(sl, shared):
        if sh:
            tabs.append(Tab(FB[:, s].contiguous(), ROW, lk, shared=True))
        else:
            tabs.append(Tab(big[:, i], FULL))
            i += 1
    tabs.append(Tab(big[:, -1], FULL))
    prog2, terms2 = _block_struct(prog, terms, len(sl))
    if ext_tr:
        rb, (FB2, sl2), _ = prove(tr, D, lk, tabs, terms2, red, label=label + b"-blk", prog=prog2, dev=True,
                                  coefs_dev=coefs_dev)
        from .lazy import KEYS
        rr = torch.cat([rho, rb], dim=0)
        if rho in KEYS and rb in KEYS:
            KEYS[rr] = KEYS[rho] + KEYS[rb]
        return rr, (FB2, sl2[:-1]), None
    rb, fin2, red2 = prove(tr, D, lk, tabs, terms2, red, label=label + b"-blk", prog=prog2)
    return np.concatenate([rho, rb], axis=1), fin2[:-1], red2


def verify_blocks(tr, E, n, d, claim, K, shared, eq_r=None, label=b"sc", prog=None, terms=None):
    """shared: per table, True if it does not depend on the block.  Returns (rho, reduced claim);
    the caller checks reduced == bw~(rho_b) * E(finals)."""
    rho, red = verify(tr, E, n, d, claim, eq_r=eq_r, K=K, label=label)
    lk = nbits(K)
    if lk == 0:
        return rho, red
    d2 = block_degree(prog, terms, shared, lk)
    rb, red2 = verify(tr, E, lk, d2, red, label=label + b"-blk")
    return np.concatenate([rho, rb], axis=1), red2


def mle_vec(G, vec, pt):
    """MLE of the zero-padded vector vec (D, K) at pt (D, k) -> numpy (D,)."""
    K = vec.shape[1]
    if pt.shape[1] == 0:
        return np.asarray(vec[:, 0], dtype=np.int64) % P
    e = G.eq_table(pt)[:, :K].contiguous()
    return np.asarray(G.dot(e, G.to_dev(vec)), dtype=np.int64)


# --------------------------------------------------------------- verifier
def verify(tr, E, n, d, claim, eq_r=None, K=1, label=b"sc", on_round=None):
    """Replays the transcript.  Returns (rho, reduced_claim); the caller checks
    reduced_claim == sum_b bw_b sum_t c_t prod finals (eq factor excluded)."""
    from . import vcore
    if on_round is None and vcore.usable(tr):
        return vcore.sc_verify(tr, E.D, n, d, claim, eq_r, K, label)
    F = fe(E.D)
    tr.absorb(label + b"|" + n.to_bytes(2, "little") + d.to_bytes(2, "little") + K.to_bytes(4, "little"))
    claim_p = np.asarray(claim, dtype=np.int64) % P
    npts = max(d, 1)
    rho = np.zeros((E.D, n), dtype=np.int64)
    for j in range(n):
        msg = tr.send(None)
        assert msg.shape == (npts, E.D)
        rj_ch = tr.challenge(D=E.D)
        rho[:, j] = rj_ch
        claim_p = _next_claim(F, claim_p, msg, None if eq_r is None else eq_r[:, j], rj_ch, d)
        if on_round is not None:
            on_round(j, rj_ch)
    return rho, claim_p


def eval_prog(E, prog, finals, bw=None):
    from . import vcore
    if vcore.ENABLED and all(f.shape[1] in (1, max(g.shape[1] for g in finals)) for f in finals):
        return vcore.eval_prog(E.D, prog, finals, bw)
    F = fe(E.D)
    K = max(f.shape[1] for f in finals) if finals else 1
    acc = F.zero
    for b in range(K):
        vals = [F.c(f[:, b if f.shape[1] > 1 else 0]) for f in finals]
        s = prog.evaluate(F, vals)
        if bw is not None:
            s = F.mul(s, F.c(bw[:, b]))
        acc = F.add(acc, s)
    return np.array(acc, dtype=np.int64)


def eval_terms(E, terms, finals, bw=None):
    """sum_b bw_b sum_t c_t prod_k finals[k][:, b]; finals (D, K) or (D, 1) (shared)."""
    F = fe(E.D)
    K = max(f.shape[1] for f in finals) if finals else 1
    acc = F.zero
    for b in range(K):
        s = F.zero
        for c, refs in terms:
            v = F.c(c)
            for k in refs:
                f = finals[k]
                v = F.mul(v, F.c(f[:, b if f.shape[1] > 1 else 0]))
            s = F.add(s, v)
        if bw is not None:
            s = F.mul(s, F.c(bw[:, b]))
        acc = F.add(acc, s)
    return np.array(acc, dtype=np.int64)
