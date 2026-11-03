"""Batched LogUp-GKR fractional sums.

A domain has K blocks of 2^n leaves and L lookups; lookup l has one tree over
all K' 2^n leaves (K' = K padded to a power of two with leaves 0/1).  Leaf
(l, b, i): numerator N_l(i) (1 if absent) and denominator Z - Q_l(i) with
Q_l = tag_l + beta_0 v_l + beta_1 w_l + ..., polynomials in the domain's base columns (tuple_coefs).
In-block layers: children of (b, x) in a layer of 2^k entries per block are
(b, x) and (b, x + 2^k).  Above the per-block roots, a top tree over the
block index pairs b and b + 2^j.  The root fraction of each lookup is sent;
the leaf layer is proved by the caller (domain.py) over the columns.
Every layer is one sumcheck batched over lookups; in-block layers bind the
block index after the in-block variables (sumcheck block phase), so each
layer ends in one claim per table, at a point (x, b)."""
import os
import numpy as np
import torch
from .field import P
from . import kern
from .kern import ptr, i64, i32
from .ext import gops, _grid, NT
from .fe import fe
from . import sumcheck as sc
from .sumcheck import Tab, FULL, ROW
from .devtr import DevTr
from .ext import dev as _dev

_LK = {}


def _col_expr(k, t, n):
    if t.kind == FULL:
        return f"C{k}[(b << {n}) + idx]"
    if t.kind == ROW:
        return f"C{k}[{'' if t.shared else f'(b << {n - t.c}) + '}(idx >> {t.c})]"
    return f"C{k}[{'' if t.shared else f'(b << {t.c}) + '}(idx & {(1 << t.c) - 1}ll)]"


def _poly_c(p, h):
    parts = []
    for mono, c in p.t.items():
        if not mono:
            parts.append(f"{c}u")
            continue
        e = f"x{mono[0]}_{h}"
        for i in mono[1:]:
            e = f"fmul({e}, x{i}_{h})"
        parts.append(e if c == 1 else f"fmul({e}, {c}u)")
    if not parts:
        return "0u"
    e = parts[0]
    for q in parts[1:]:
        e = f"fadd({e}, {q})"
    return e


def _split_lp(q):
    """q without its layer-parameter terms (ids < 0, see poly.lparam) and those terms [(index, +-1)]."""
    from .poly import Poly, lparam_index
    lps = [(lparam_index(m[0]), c) for m, c in q.t.items() if m and m[0] < 0]
    if not lps:
        return q, []
    assert all(c in (1, P - 1) for _, c in lps), "layer parameters enter linearly with +-1"
    return Poly({m: c for m, c in q.t.items() if not (m and m[0] < 0)}), lps


def _has_k(q):
    """q's constant term is a kernel argument (kcst) rather than part of the source."""
    return q is not None and (not q.vars() or q.const_term() % P != 0)


def _kcst_vals(lookups):
    """Constant terms, in the same order leaf_kernel emits kcst[i]."""
    out = []
    for t, vs, nm in lookups:
        t, vs, nm = _split_lp(t)[0], [_split_lp(v)[0] for v in vs], None if nm is None else _split_lp(nm)[0]
        for _h in (0, 1):
            if _has_k(t):
                out.append(int(t.const_term()) % P)
            for v in vs:
                if v.t and _has_k(v):
                    out.append(int(v.const_term()) % P)
        if nm is not None:
            for _h in (0, 1):
                if _has_k(nm):
                    out.append(int(nm.const_term()) % P)
    return out


def leaf_kernel(D, n, cols, lookups, pair=False):
    """lookups: list of (tag, [v0, v1, ...], num|None) Polys over column ids.  pair: also write the
    next layer (n-2), pairing entries x and x + 2^(n-2) of layer n-1 as gkr_pair does, so the tree
    build starts one layer higher and never reads layer n-1 back."""
    def pk(q):
        if q is None:
            return None
        if not q.vars():
            return ("K",)
        return (_has_k(q),) + tuple(sorted((m, c) for m, c in q.t.items() if m != ()))
    key = (D, n, pair, tuple((t.kind, t.c, t.shared) for t in cols),
           tuple((pk(t), tuple(pk(v) for v in vs), pk(nm)) for t, vs, nm in lookups))
    if key in _LK:
        return _LK[key]
    A = max([len(vs) for _, vs, _ in lookups] + [1])
    allp = [q for t, vs, nm in lookups for q in [t, nm] + list(vs) if q is not None]
    used = sorted(set(i for q in allp for i in q.vars() if i >= 0))
    J = 2 if pair else 1
    L = ['#include "field.cuh"']
    prm = [f"const uint32_t* __restrict__ C{k}" for k in range(len(cols))]
    prm += ["const uint32_t* __restrict__ chal", "const uint32_t* __restrict__ kcst",
            "uint32_t* __restrict__ out", "int64_t on", "int64_t total", "const uint32_t* __restrict__ lp",
            "uint32_t* __restrict__ out2", "int64_t on2"]
    L.append(f'extern "C" __global__ void leaf({", ".join(prm)}) {{')
    L.append("  EF Z = ef_c(chal);")
    for i in range(A):
        L.append(f"  EF B{i} = ef_c(chal + {i + 1} * EXT_D);")
    sh = n - 1 - (1 if pair else 0)                     # bits of the per-block index this loop runs over
    L.append(f"  for (int64_t e2 = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e2 < (total >> {J - 1}); "
             f"e2 += (int64_t)gridDim.x * blockDim.x) {{")
    L.append(f"    int64_t b = e2 >> {sh}, z = e2 & {(1 << sh) - 1}ll;")
    if used:
        L.append(f"    uint32_t {', '.join(f'x{k}_{h}_j{j}' for k in used for h in (0, 1) for j in range(J))};")
    for j in range(J):
        L.append(f"    {{ int64_t x = z + {j << sh}ll;")
        for h in (0, 1):
            L.append(f"      {{ int64_t idx = x + {h << (n - 1)}ll;")
            for k in used:
                L.append(f"        x{k}_{h}_j{j} = {_col_expr(k, cols[k], n)};")
            L.append("      }")
        L.append("    }")
    ki = [0]

    def pc(q, h):
        q, lps = _split_lp(q)
        e = pc0(q, h)
        for jj, c in lps:          # layer parameters: read from the device (domain.LPARAMS_DEV)
            e = f"{'fadd' if c == 1 else 'fsub'}({e}, lp[{jj}])"
        return e

    def pc0(q, h):
        if _has_k(q):
            s_ = f"kcst[{ki[0]}]"
            ki[0] += 1
            return s_ if not q.vars() else f"fadd({_poly_c(q - q.const_term(), h)}, {s_})"
        return _poly_c(q, h)

    for l, (tag, vs, num) in enumerate(lookups):
        body = []                  # one entry of layer n-1 for lookup l, from the aliases x{k}_{h}
        for h in (0, 1):
            body.append(f"EF q_{h} = sub(Z, ef_base({pc(tag, h)}));")
            for i, v in enumerate(vs):
                if v.t:
                    body.append(f"q_{h} = sub(q_{h}, mulb(B{i}, {pc(v, h)}));")
        if num is None:
            body.append("EF p_ = add(q_0, q_1);")
        else:
            body.append(f"EF p_ = add(mulb(q_1, {pc(num, 0)}), mulb(q_0, {pc(num, 1)}));")
        body.append("EF q_ = mul(q_0, q_1);")
        L.append("    {")
        if pair:
            L.append("      EF pp0, qq0, pp1, qq1;")
        for j in range(J):
            L.append("      {")
            for k in used:
                for h in (0, 1):
                    L.append(f"        uint32_t x{k}_{h} = x{k}_{h}_j{j};")
            L.extend("        " + x for x in body)
            L.append(f"        int64_t e = (b << {n - 1}) + z + {j << sh}ll;")
            L.append(f"        ef_store(out, (int64_t){2 * l} * total + e, on, p_);")
            L.append(f"        ef_store(out, (int64_t){2 * l + 1} * total + e, on, q_);")
            if pair:
                L.append(f"        pp{j} = p_; qq{j} = q_;")
            L.append("      }")
        if pair:
            L.append(f"      ef_store(out2, (int64_t){2 * l} * (total >> 1) + e2, on2, add(mul(pp0, qq1), mul(pp1, qq0)));")
            L.append(f"      ef_store(out2, (int64_t){2 * l + 1} * (total >> 1) + e2, on2, mul(qq0, qq1));")
        L.append("    }")
    L.append("  }")
    L.append("}")
    fn = kern.compile_module("\n".join(L), D).get_function("leaf")
    _LK[key] = fn
    return fn


# Lookup tuples are compressed as tag + beta_0 v0 + beta_1 v1 + ... with NBETA independent challenges.
# With the powers of one challenge, a tuple of arity k makes the cleared LogUp identity a polynomial of
# degree k D in (Z, beta) instead of D, which costs log2(k) bits of soundness per chunk.
NBETA = 8


def tuple_coefs(beta, k):
    """Coefficients (D,) of tuple entries v0 .. v(k-1); beta: (D, NBETA) independent challenges."""
    b = np.asarray(beta, dtype=np.int64)
    assert b.ndim == 2 and b.shape[1] == NBETA and k <= NBETA, "beta must be NBETA independent challenges"
    return [b[:, i] % P for i in range(k)]


def leaf_layer(G, n, K, cols, lookups, Z, beta, pair=False):
    """Layer n-1 of every tree: list of (p, q) ext (D, K*2^(n-1)); with pair (n >= 3), also layer
    n-2: (layer n-1, layer n-2)."""
    D = G.D
    A = max([len(vs) for _, vs, _ in lookups] + [1])
    pw = [np.asarray(Z, dtype=np.int64)] + tuple_coefs(beta, A)
    chal = G.to_dev(np.concatenate(pw))
    kv = _kcst_vals(lookups)
    kcst = kern.h2d(np.array(kv or [0], dtype=np.uint32).view(np.int32))
    total = K << (n - 1)
    pair = pair and n >= 3
    out = torch.empty((D, 2 * len(lookups) * total), dtype=torch.int32, device=_dev())
    out2 = torch.empty((D, len(lookups) * total), dtype=torch.int32, device=_dev()) if pair else out
    fn = leaf_kernel(D, n, cols, lookups, pair)
    from . import domain as dm
    if dm.LPARAMS_DEV is None:
        dm.set_lparams(dm.LPARAMS)
    kern.launch(fn, _grid(total >> (1 if pair else 0)), NT,
                [ptr(t.data) for t in cols] + [ptr(chal), ptr(kcst), ptr(out), i64(out.stride(0)), i64(total),
                                               ptr(dm.LPARAMS_DEV), ptr(out2), i64(out2.stride(0))])
    top = [(out[:, (2 * l) * total:(2 * l + 1) * total], out[:, (2 * l + 1) * total:(2 * l + 2) * total])
           for l in range(len(lookups))]
    if not pair:
        return top
    h = total >> 1
    return top, [(out2[:, (2 * l) * h:(2 * l + 1) * h], out2[:, (2 * l + 1) * h:(2 * l + 2) * h])
                 for l in range(len(lookups))]


def _pair(G, p, q, k, total):
    D = G.D
    po = torch.empty((D, total), dtype=torch.int32, device=_dev())
    qo = torch.empty((D, total), dtype=torch.int32, device=_dev())
    kern.launch(G.f["gkr_pair"], _grid(total), NT,
                [ptr(p), i64(p.stride(0)), ptr(q), i64(q.stride(0)), ptr(po), i64(po.stride(0)),
                 ptr(qo), i64(qo.stride(0)), i32(k), i64(total)])
    return po, qo


_PJ = np.dtype([("v", np.int64, 8 * 32)])


def _pairs(G, pqs, k, total):
    """_pair of every (p, q) in pqs, 32 lookups per launch."""
    D = G.D
    out = []
    for i in range(0, len(pqs), 32):
        chunk = pqs[i:i + 32]
        J = np.zeros((), dtype=_PJ)
        v = J["v"].reshape(8, 32)
        for j, (p, q) in enumerate(chunk):
            po = torch.empty((D, total), dtype=torch.int32, device=_dev())
            qo = torch.empty((D, total), dtype=torch.int32, device=_dev())
            v[:, j] = (p.data_ptr(), p.stride(0), q.data_ptr(), q.stride(0), po.data_ptr(), po.stride(0),
                       qo.data_ptr(), qo.stride(0))
            out.append((po, qo))
        kern.launch(G.f["gkr_pair_list"], (_grid(total), len(chunk)), NT, [J[()], i32(len(chunk)), i32(k), i64(total)])
    return out


def leaf_bytes(D, n, K, L):
    return 2 * L * (K << max(n - 1, 0)) * D * 4


def build(G, top, K, n, drop_leaves=False, nxt=None):
    """top: layer n-1 per lookup; nxt: layer n-2, if the leaf kernel paired it already.  Returns
    (layers k -> [(p, q)] for k = 0..n-1, top tree j -> [(p, q)]).
    drop_leaves frees the 2^n leaf tables after the first fold; prove_upper rematerializes them last."""
    D = G.D
    layers = {n - 1: top}
    for k in range(n - 2, -1, -1):
        layers[k] = nxt if (nxt is not None and k == n - 2) else _pairs(G, layers[k + 1], k, K << k)
        if drop_leaves and k + 1 == n - 1:
            layers[n - 1] = None
            del top
    lk = sc.nbits(K)
    Kp = 1 << lk
    tt = {}
    cur = []
    for p, q in layers[0]:
        pp = torch.zeros((D, Kp), dtype=torch.int32, device=_dev())
        qq = torch.zeros((D, Kp), dtype=torch.int32, device=_dev())
        qq[0] = 1
        pp[:, :K] = p
        qq[:, :K] = q
        cur.append((pp, qq))
    tt[lk] = cur
    for j in range(lk - 1, -1, -1):
        tt[j] = _pairs(G, tt[j + 1], j, 1 << j)
    return layers, tt


def _layer_terms(F, L, lam, a):
    """Terms over tables [p0_l, p1_l, q0_l, q1_l] (index 4l + .)."""
    terms = []
    for l in range(L):
        al = np.array(a[l], dtype=np.int64)
        alam = np.array(F.mul(a[l], lam), dtype=np.int64)
        terms += [(al, [4 * l, 4 * l + 3]), (al, [4 * l + 1, 4 * l + 2]), (alam, [4 * l + 2, 4 * l + 3])]
    return terms


def _claim(F, cp, cq, a, lam):
    s = F.zero
    for l in range(len(cp)):
        s = F.add(s, F.mul(a[l], F.add(F.c(cp[l]), F.mul(lam, F.c(cq[l])))))
    return np.array(s, dtype=np.int64)


def lerp_v(a, b, t):
    """a + t (b - a) per column for numpy ext arrays (D, K) and an ext scalar t (on the host:
    K is small, and a device round trip per column costs a synchronization)."""
    from .field import Ext
    E = Ext(a.shape[0])
    a = np.asarray(a, dtype=np.int64) % P
    tt = np.asarray(t, dtype=np.int64).reshape(-1, 1) % P
    return E.add(a, E.mul(tt, E.sub(np.asarray(b, dtype=np.int64) % P, a)))


def _lerp(F, a, b, t):
    return np.array(F.add(F.c(a), F.mul(F.c(t), F.sub(F.c(b), F.c(a)))), dtype=np.int64)


def pad_q(G, K, rb):
    """sum_{b >= K} eq(rb, b): the denominators of the padded per-block roots."""
    lk = rb.shape[1]
    if (1 << lk) == K:
        return np.zeros(G.D, dtype=np.int64)
    e = G.eq_table(rb)
    return G.sum(e[:, K:].contiguous())


def block_weights(G, rb, K, dev=False):
    """eq(rb, b) for b < K -> numpy (D, K), or a device tensor if dev."""
    if rb.shape[1] == 0:
        w = np.zeros((G.D, 1), dtype=np.int64)
        w[0, 0] = 1
        return G.to_dev(w) if dev else w
    e = G.eq_table(rb)[:, :K].contiguous()
    return e if dev else G.to_host(e)


def _sends(tr, F, D, L, fin):
    """fin: 4L finals (D, 1) -> next claims (cp, cq) after a fresh t."""
    for f in fin:
        tr.send(f.T)
    t = tr.challenge()
    cp = [_lerp(F, fin[4 * l][:, 0], fin[4 * l + 1][:, 0], t) for l in range(L)]
    cq = [_lerp(F, fin[4 * l + 2][:, 0], fin[4 * l + 3][:, 0], t) for l in range(L)]
    return t, cp, cq


def _recv(tr, F, D, L):
    fin = [tr.send(None).T for _ in range(4 * L)]
    assert all(f.shape == (D, 1) for f in fin)
    t = tr.challenge()
    cp = [_lerp(F, fin[4 * l][:, 0], fin[4 * l + 1][:, 0], t) for l in range(L)]
    cq = [_lerp(F, fin[4 * l + 2][:, 0], fin[4 * l + 3][:, 0], t) for l in range(L)]
    return fin, t, cp, cq


_GD = {}


def _gdev(D):
    if D not in _GD:
        m = kern.file_module("gkrdev.cu", D)
        _GD[D] = {k: m.get_function(k) for k in ("gkr_coefs", "gkr_next", "dom_coefs", "ef_pow1")}
    return _GD[D]


def _sends_dev(dt, G, FB, sl, L):
    D = G.D
    fins = dt.send_cols(FB, [s.start for s in sl])
    t = dt.challenge()
    cp = torch.empty((L, D), dtype=torch.int32, device=_dev())
    cq = torch.empty((L, D), dtype=torch.int32, device=_dev())
    kern.launch(_gdev(D)["gkr_next"], 1, max(32, L), [ptr(fins), ptr(t), i32(L), ptr(cp), ptr(cq)])
    return t, cp, cq


def _layer_dev(dt, G, L):
    a = dt.challenge(L)
    lam = dt.challenge()
    co = torch.empty((3 * L, G.D), dtype=torch.int32, device=_dev())
    kern.launch(_gdev(G.D)["gkr_coefs"], 1, max(32, L), [ptr(a), ptr(lam), i32(L), ptr(co)])
    return co


def _eq_dev(G, r, K):
    """eq(r, b) for b < K, r (k, D) device -> (D, K)."""
    if r.shape[0] == 0:
        w = torch.zeros((G.D, 1), dtype=torch.int32, device=_dev())
        w[0].fill_(1)                 # no host scalar copy
        return w
    return sc.eq_segments(G, r, [(0, r.shape[0])], [0], 1 << r.shape[0])[:, :K].contiguous()


def prove_upper(tr, G, n, K, layers, tt, L, label=b"gkr", leaf_fn=None, dt=None):
    """dt: a device transcript to continue (the caller finishes it); the results are then device
    tensors (ri (n-1, D), rb (lk, D), cp (L, D), cq (L, D))."""
    if os.environ.get("ZK_HOSTGKR") and dt is None:
        return prove_upper_host(tr, G, n, K, layers, tt, L, label)
    D = G.D
    F = fe(D)
    lk = sc.nbits(K)
    keep = dt is not None
    sess = not keep and getattr(tr, "dev", None) is not None
    dt = dt if keep else (tr.dev if sess else DevTr(tr))
    for p, q in tt[0]:
        dt.send(torch.cat([p, q], dim=1).T)
    cp = torch.stack([p[:, 0] for p, _ in tt[0]])
    cq = torch.stack([q[:, 0] for _, q in tt[0]])
    zero_terms = _layer_terms(F, L, F.zero, [F.zero] * L)
    r = torch.zeros((0, D), dtype=torch.int32, device=_dev())
    for j in range(lk):
        co = _layer_dev(dt, G, L)
        half = 1 << j
        tabs = []
        for p, q in tt[j + 1]:
            tabs += [Tab(p, FULL, view=(j, 2 * half, 0)), Tab(p, FULL, view=(j, 2 * half, half)),
                     Tab(q, FULL, view=(j, 2 * half, 0)), Tab(q, FULL, view=(j, 2 * half, half))]
        rho, (FB, sl), _ = sc.prove(dt, D, j, tabs, zero_terms, None, eq_r=r if j else None,
                                    label=label + b"-top", dev=True, coefs_dev=co)
        t, cp, cq = _sends_dev(dt, G, FB, sl, L)
        r = torch.cat([rho, t.view(1, D)], dim=0)
    rb = r
    if lk and (1 << lk) != K:
        e = sc.eq_segments(G, rb, [(0, lk)], [0], 1 << lk)
        pq = e[:, K:].to(torch.int64).sum(dim=1) % P
        cq = ((cq.to(torch.int64) - pq.view(1, D)) % P).to(torch.int32)
    ri = torch.zeros((0, D), dtype=torch.int32, device=_dev())
    for k in range(n - 1):
        co = _layer_dev(dt, G, L)
        bw = _eq_dev(G, rb, K)
        half = 1 << k
        src = layers[k + 1]
        if src is None:
            torch.cuda.empty_cache()
            src = leaf_fn()
        tabs = []
        for p, q in src:
            tabs += [Tab(p, FULL, view=(k, 2 * half, 0)), Tab(p, FULL, view=(k, 2 * half, half)),
                     Tab(q, FULL, view=(k, 2 * half, 0)), Tab(q, FULL, view=(k, 2 * half, half))]
        rho, (FB, sl), _ = sc.prove_blocks(dt, D, k, tabs, zero_terms, None, K, bw, eq_r=ri if k else None,
                                           label=label, coefs_dev=co)
        t, cp, cq = _sends_dev(dt, G, FB, sl, L)
        ri = torch.cat([rho[:k], t.view(1, D)], dim=0)
        rb = rho[k:]
        layers[k + 1] = None
        del src, tabs
    if keep:
        return ri, rb, cp, cq
    if sess:
        from .lazy import Lz
        lp, lq = Lz(cp), Lz(cq)
        return Lz(ri.T), Lz(rb.T), [lp[l] for l in range(L)], [lq[l] for l in range(L)]
    dt.finish()

    def host(x):
        return x.cpu().numpy().view(np.uint32).astype(np.int64)
    cph, cqh = host(cp), host(cq)
    return host(ri).T.copy(), host(rb).T.copy(), [cph[l] for l in range(L)], [cqh[l] for l in range(L)]


def prove_upper_host(tr, G, n, K, layers, tt, L, label=b"gkr"):
    """Sends the root of every lookup and proves every layer down to layer n-1.
    Returns (in-block point (D, n-1), block point (D, nbits K), claims p, q per lookup)."""
    D = G.D
    F = fe(D)
    lk = sc.nbits(K)
    for p, q in tt[0]:
        tr.send(np.concatenate([G.to_host(p), G.to_host(q)], axis=1).T)
    r = np.zeros((D, 0), dtype=np.int64)
    cp = [G.to_host(p)[:, 0] for p, _ in tt[0]]
    cq = [G.to_host(q)[:, 0] for _, q in tt[0]]
    for j in range(lk):
        a = [F.c(x) for x in tr.challenge(L).T]
        lam = F.c(tr.challenge())
        claim = _claim(F, cp, cq, a, lam)
        half = 1 << j
        tabs = []
        for p, q in tt[j + 1]:
            tabs += [Tab(p, FULL, view=(j, 2 * half, 0)), Tab(p, FULL, view=(j, 2 * half, half)),
                     Tab(q, FULL, view=(j, 2 * half, 0)), Tab(q, FULL, view=(j, 2 * half, half))]
        rho, fin, _ = sc.prove(tr, D, j, tabs, _layer_terms(F, L, lam, a), claim,
                               eq_r=r if j else None, label=label + b"-top")
        t, cp, cq = _sends(tr, F, D, L, fin)
        r = np.concatenate([rho, t.reshape(D, 1)], axis=1)
    rb = r
    pq = pad_q(G, K, rb)
    cq = [(np.asarray(c) - pq) % P for c in cq]
    ri = np.zeros((D, 0), dtype=np.int64)
    for k in range(n - 1):
        a = [F.c(x) for x in tr.challenge(L).T]
        lam = F.c(tr.challenge())
        claim = _claim(F, cp, cq, a, lam)
        bw = block_weights(G, rb, K, dev=True)
        half = 1 << k
        tabs = []
        for p, q in layers[k + 1]:
            tabs += [Tab(p, FULL, view=(k, 2 * half, 0)), Tab(p, FULL, view=(k, 2 * half, half)),
                     Tab(q, FULL, view=(k, 2 * half, 0)), Tab(q, FULL, view=(k, 2 * half, half))]
        rho, fin, _ = sc.prove_blocks(tr, D, k, tabs, _layer_terms(F, L, lam, a), claim, K, bw,
                                      eq_r=ri if k else None, label=label)
        t, cp, cq = _sends(tr, F, D, L, fin)
        ri = np.concatenate([rho[:, :k], t.reshape(D, 1)], axis=1)
        rb = rho[:, k:]
    return ri, rb, cp, cq


def verify_upper(tr, E, n, K, L, label=b"gkr"):
    """Returns (roots [(P, Q)] per lookup, in-block point, block point, claims p, q)."""
    from . import vcore
    if vcore.usable(tr):
        lk = sc.nbits(K)
        d2 = sc.block_degree(None, _layer_terms(fe(E.D), L, fe(E.D).zero, [fe(E.D).zero] * L), [False] * (4 * L), lk) \
            if lk else 0
        return vcore.gkr_upper(tr, E.D, n, K, L, label, d2)
    D = E.D
    F = fe(D)
    G = gops(D)
    lk = sc.nbits(K)
    roots, cp, cq = [], [], []
    for _ in range(L):
        v = tr.send(None).T
        assert v.shape == (D, 2)
        roots.append((F.c(v[:, 0]), F.c(v[:, 1])))
        cp.append(v[:, 0])
        cq.append(v[:, 1])
    r = np.zeros((D, 0), dtype=np.int64)
    for j in range(lk):
        a = [F.c(x) for x in tr.challenge(L).T]
        lam = F.c(tr.challenge())
        claim = _claim(F, cp, cq, a, lam)
        terms = _layer_terms(F, L, lam, a)
        rho, red = sc.verify(tr, E, j, 2 if j else 0, claim, eq_r=r if j else None, label=label + b"-top")
        fin, t, cp, cq = _recv(tr, F, D, L)
        if not np.array_equal(red % P, sc.eval_terms(E, terms, fin)):
            raise ValueError("gkr top layer check failed")
        r = np.concatenate([rho, t.reshape(D, 1)], axis=1)
    rb = r
    pq = pad_q(G, K, rb)
    cq = [(np.asarray(c) - pq) % P for c in cq]
    ri = np.zeros((D, 0), dtype=np.int64)
    for k in range(n - 1):
        a = [F.c(x) for x in tr.challenge(L).T]
        lam = F.c(tr.challenge())
        claim = _claim(F, cp, cq, a, lam)
        terms = _layer_terms(F, L, lam, a)
        bw = block_weights(G, rb, K)
        rho, red = sc.verify_blocks(tr, E, k, 2 if k else 0, claim, K, [False] * (4 * L), eq_r=ri if k else None,
                                    label=label, terms=terms)
        fin, t, cp, cq = _recv(tr, F, D, L)
        bwv = sc.mle_vec(G, bw, rho[:, k:]) if K > 1 else bw[:, 0]
        if F.c(red) != F.mul(F.c(bwv), F.c(sc.eval_terms(E, terms, fin))):
            raise ValueError("gkr layer check failed")
        ri = np.concatenate([rho[:, :k], t.reshape(D, 1)], axis=1)
        rb = rho[:, k:]
    return roots, ri, rb, cp, cq
