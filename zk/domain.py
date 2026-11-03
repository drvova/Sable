"""A domain: K blocks of 2^n elements, base-field columns, zerocheck
constraints and lookups.  One proof per domain:

  1. batched LogUp-GKR over all lookups (gkr.py) down to layer n-1, at point r;
  2. one sumcheck over the column halves that proves the leaf layer of every
     tree and the zerocheck  sum_x eq(r, x) [C(x, 0) + eta C(x, 1)] = 0;
  3. a challenge t turns half evaluations into column claims at (rho, t).

Columns: FULL (K*2^n), ROW c (vars [c, n)), COL c (vars [0, c)); shared
ROW/COL tables are the same for every block.  Roles: 'commit' (claims go to
the PCS), 'virtual' (claims go to the protocol that defines the column),
'public' (the verifier evaluates them itself)."""
import os
import re
import numpy as np
import torch
from . import kern
from .field import P
from .poly import Poly, LPARAM_NAMES, lparam_index
from .fe import fe
from .ext import gops
from . import gkr
from . import sumcheck as sc
from .sumcheck import Tab, FULL, ROW, COL
from .transcript import label_name
from .ext import dev as _dev


class Column:
    def __init__(self, name, kind, c, shared, role, data):
        self.name, self.kind, self.c, self.shared, self.role, self.data = name, kind, c, shared, role, data


class Domain:
    """merge=True binds the block index inside the protocol (sumcheck block phase): claims
    are then single values at points (in-block point, block point)."""

    def __init__(self, name, n, K=1, merge=False):
        assert n >= 2
        self.name, self.n, self.K, self.merge = name, n, K, merge
        self.cols = []
        self.constraints = []
        self.lookups = []
        self.dyn = []

    def col(self, name, data=None, kind=FULL, c=0, shared=False, role="commit"):
        if kind == FULL:
            c = 0
        self.cols.append(Column(name, kind, c, shared, role, data))
        return Poly.var(len(self.cols) - 1)

    def zero(self, poly):
        self.constraints.append(Poly.lift(poly))

    def lookup(self, tag, *vals, num=None, dynamic=False):
        """tuple tag + beta_0 v0 + beta_1 v1 + ... (gkr.tuple_coefs); numerator num (default 1).  Lookups with
        num set (table sides) or dynamic=True are not counted in the global table."""
        self.lookups.append((Poly.lift(tag), [Poly.lift(v) for v in vals],
                             None if num is None else Poly.lift(num)))
        self.dyn.append(dynamic)

    def arity(self):
        return max([len(v) for _, v, _ in self.lookups] + [1])

    def col_index(self, name):
        return [c.name for c in self.cols].index(name)

    # ------------------------------------------------------------ structure
    def _tabmap(self):
        """col k -> (table id for half 0, half 1), plus per-table (col, half, kind, c, shared)."""
        n = self.n
        tmap, tabs = [], []
        for k, c in enumerate(self.cols):
            if c.kind == FULL or (c.kind == ROW and c.c < n):
                ids = (len(tabs), len(tabs) + 1)
                kind = FULL if c.kind == FULL else ROW
                tabs += [(k, 0, kind, c.c, c.shared), (k, 1, kind, c.c, c.shared)]
            elif c.kind == COL and c.c == n:     # all in-block variables (e.g. shared masks)
                ids = (len(tabs), len(tabs) + 1)
                tabs += [(k, 0, COL, n - 1, c.shared), (k, 1, COL, n - 1, c.shared)]
            elif c.kind == ROW:          # constant per block
                ids = (len(tabs),) * 2
                tabs.append((k, None, ROW, n - 1, c.shared))
            else:
                assert c.c < n
                ids = (len(tabs),) * 2
                tabs.append((k, None, COL, c.c, c.shared))
            tmap.append(ids)
        return tmap, tabs

    def prog(self, F, Z, beta, a, lam, gam, eta, recipe=None):
        """The leaf-layer + zerocheck program.  With Z None the coefficients are left symbolic:
        recipe receives one (kind, index, base value) per coefficient (see COEF_KINDS), which
        dom_coefs evaluates on the device from device challenges."""
        tmap, _ = self._tabmap()
        polys, coefs = [], []
        sym = Z is None
        bco = None if sym else [F.c(b) for b in gkr.tuple_coefs(beta, self.arity())]

        def addp(p, h):
            q = p.rename(lambda i: tmap[i][h])
            lp = {k: v for k, v in q.t.items() if k and k[0] < 0}       # layer-parameter terms
            if lp:
                (k_, c_), = lp.items()
                assert len(k_) == 1 and c_ in (1, P - 1), "layer parameters enter linearly with +-1"
                q = Poly({k: v for k, v in q.t.items() if k not in lp})
                j = lparam_index(k_[0])
                cp = addc(("lparam", 2 * j + (c_ != 1), int(q.const_term()) % P))
                q = q - q.const_term()
                if not q.vars():
                    return cp
                polys.append(q)
                return ("+", ("p", len(polys) - 1), cp)
            if not q.vars():
                return addc(("base", 0, int(q.const_term()) % P))
            # constant terms are kernel arguments, so per-layer format constants do not
            # change the generated source (one compile per constraint structure)
            k0 = q.const_term()
            polys.append(q - k0)
            if not k0:
                return ("p", len(polys) - 1)
            return ("+", ("p", len(polys) - 1), ("kc", addc(("base", 0, int(k0) % P))[1]))

        def value(rec):
            kind, i, v = rec
            if kind == "base":
                return F.base(v)
            if kind == "Z":
                return Z
            if kind == "lparam":
                pv = int(LPARAMS[i >> 1]) % P
                return F.base((v - pv) % P if i & 1 else (v + pv) % P)
            if kind == "bpow":
                return bco[i]
            if kind in ("alamZmu", "alamb", "Zmu", "namu"):
                mu = F.inv(lam)
                if kind == "alamZmu":
                    return F.mul(F.mul(a[i], lam), F.sub(F.add(Z, mu), F.base(v)))
                if kind == "alamb":
                    return F.mul(F.mul(a[i >> 8], lam), bco[i & 255])
                if kind == "Zmu":
                    return F.sub(F.add(Z, mu), F.base(v))
                return F.sub(F.zero, F.mul(a[i], mu))
            return {"a": lambda: a[i], "alam": lambda: F.mul(a[i], lam), "gam": lambda: gam[i],
                    "geta": lambda: F.mul(gam[i], eta)}[kind]()

        def addc(rec):
            if recipe is not None:
                recipe.append(rec)
            coefs.append(np.zeros(len(F.zero), dtype=np.int64) if sym else np.array(value(rec), dtype=np.int64))
            return ("c", len(coefs) - 1)

        cZ = addc(("Z", 0, 0))
        cb = [addc(("bpow", i, 0)) for i in range(self.arity())]
        terms = []
        for l, (tag, vals, num) in enumerate(self.lookups):
            if num is None and FACTOR_LEAF:
                kt = None if tag.vars() else int(tag.const_term()) % P
                q = []
                for h in (0, 1):
                    if h == 0:
                        if kt is not None:
                            e = addc(("alamZmu", l, kt))
                        else:
                            e = ("-", addc(("alamZmu", l, 0)), ("*", addc(("alam", l, 0)), addp(tag, 0)))
                    else:
                        e = addc(("Zmu", 0, kt)) if kt is not None else ("-", addc(("Zmu", 0, 0)), addp(tag, 1))
                    for i, v in enumerate(vals):
                        if v.t:
                            c = addc(("alamb", 256 * l + i, 0)) if h == 0 else cb[i]
                            e = ("-", e, ("*", c, addp(v, h)))
                    q.append(e)
                terms.append(("+", ("*", q[0], q[1]), addc(("namu", l, 0))))
                continue
            q, N = [], []
            for h in (0, 1):
                e = ("-", cZ, addp(tag, h))
                for i, v in enumerate(vals):
                    if v.t:
                        e = ("-", e, ("*", cb[i], addp(v, h)))
                q.append(e)
                N.append(addp(num, h) if num is not None else None)
            pn = ("+", q[1], q[0]) if num is None else ("+", ("*", N[0], q[1]), ("*", N[1], q[0]))
            terms.append(("*", addc(("a", l, 0)), pn))
            terms.append(("*", addc(("alam", l, 0)), ("*", q[0], q[1])))
        for i, cpoly in enumerate(self.constraints):
            terms.append(("*", addc(("gam", i, 0)), addp(cpoly, 0)))
            terms.append(("*", addc(("geta", i, 0)), addp(cpoly, 1)))
        expr = terms[0]
        for t in terms[1:]:
            expr = ("+", expr, t)
        return sc.Prog(polys, expr, coefs)

    def kinds(self):
        _, tabs = self._tabmap()
        return [t[2] for t in tabs], [t[3] for t in tabs]

    def shared(self):
        _, tabs = self._tabmap()
        return [bool(t[4]) for t in tabs]

    # ------------------------------------------------------------ prover data
    def _tables(self, G):
        n = self.n
        _, tabs = self._tabmap()
        out = []
        for k, h, kind, c, shared in tabs:
            col = self.cols[k]
            d = col.data
            if kind == FULL:
                half = 1 << (n - 1)
                out.append(Tab(d, FULL, view=(n - 1, 2 * half, h * half)))
            elif h is None:
                out.append(Tab(d, kind, c, shared=shared))
            elif kind == COL:
                half = 1 << (n - 1)
                v = d.reshape(-1, 2, half)[:, h, :].reshape(-1) if d.dim() == 1 else \
                    d.reshape(d.shape[0], -1, 2, half)[:, :, h, :].reshape(d.shape[0], -1)
                out.append(Tab(v.contiguous(), COL, c, shared=shared))
            else:
                m = 1 << (n - c)
                v = d.reshape(-1, 2, m // 2)[:, h, :].reshape(-1) if d.dim() == 1 else \
                    d.reshape(d.shape[0], -1, 2, m // 2)[:, :, h, :].reshape(d.shape[0], -1)
                out.append(Tab(v.contiguous(), ROW, c, shared=shared))
        return out


def _challenges(tr, dom):
    F = fe(tr.D)
    a = [F.c(x) for x in tr.challenge(max(1, len(dom.lookups))).T]
    lam = F.c(tr.challenge())
    gam = [F.c(x) for x in tr.challenge(max(1, len(dom.constraints))).T]
    eta = F.c(tr.challenge())
    return a, lam, gam, eta


def _claims(F, dom, rho, t, fin):
    """Unmerged: name -> (in-block point, values (D, K), or (D, 1) if shared).
    Merged: name -> (in-block point || block point, value (D, 1)); shared columns have no block part."""
    n = dom.n
    tmap, _ = dom._tabmap()
    ri, rb = rho[:, :n - 1], rho[:, n - 1:]
    out = {}
    tt = F.c(t)
    for k, col in enumerate(dom.cols):
        i0, i1 = tmap[k]
        f0, f1 = fin[i0], fin[i1]
        if i0 != i1:
            val = gkr.lerp_v(f0, f1, tt)
            full = np.concatenate([ri, t.reshape(-1, 1)], axis=1)
            pt = full if col.kind in (FULL, COL) else full[:, col.c:]
        else:
            val = f0
            pt = ri[:, :col.c] if col.kind == COL else ri[:, :0]
        if dom.merge and not col.shared:
            pt = np.concatenate([pt, rb], axis=1)
        out[col.name] = (pt, val)
    return out


def prove(tr, dom, Z, beta, label=b"dom"):
    if not os.environ.get("ZK_NVTX"):
        return _prove(tr, dom, Z, beta, label)
    torch.cuda.nvtx.range_push("dom:" + re.sub(r"[0-9]+", "#", dom.name))
    try:
        return _prove(tr, dom, Z, beta, label)
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()


_COEF_KIND = {"Z": 0, "bpow": 1, "a": 2, "alam": 3, "gam": 4, "geta": 5, "base": 6,
              "alamZmu": 7, "alamb": 8, "Zmu": 9, "namu": 10, "lparam": 11}
# current layer's parameters (poly.LPARAM_NAMES order): host ints (verifier, recipes) and the prover's
# device copy (uint32 mod p) read by dom_coefs, which a graph replay updates in place
LPARAMS = [0] * len(LPARAM_NAMES)
LPARAMS_DEV = None


def tag_value(tag):
    """A lookup tag's value: constant, or constant + layer parameter (current layer); else None."""
    v = int(tag.const_term())
    for k, c in tag.t.items():
        if not k:
            continue
        if len(k) != 1 or k[0] >= 0:
            return None
        v += int(c) * LPARAMS[lparam_index(k[0])]
    return v % P


def set_lparams(vals):
    global LPARAMS_DEV
    LPARAMS[:] = [int(v) for v in vals]
    from . import ext
    if ext.CPU[0]:                      # the device copy is for the prover's kernels
        return
    a = np.array([v % P for v in LPARAMS], dtype=np.uint32).view(np.int32)
    if LPARAMS_DEV is None:
        LPARAMS_DEV = torch.zeros(len(LPARAMS), dtype=torch.int32, device=_dev())
    # fixed device buffer (captured layer graphs read it), filled without a host sync
    LPARAMS_DEV.copy_(torch.from_numpy(a).pin_memory(), non_blocking=True)
# lookup leaves with numerator 1 as a lam (q0 + mu)(q1 + mu) - a mu (mu = 1 / lam), with a lam folded
# into q0's coefficients: half the extension multiplications of a (q0 + q1) + a lam q0 q1
FACTOR_LEAF = not os.environ.get("ZK_NOFACTOR")


def _claims_lazy(G, dom, rho, t, FB, sl):
    """_claims on the device: rho Lz (D, m), t device (D,), finals FB (D, *) with per-table column
    slices sl.  Every claim value is a view of one device tensor (one transfer when read)."""
    from .lazy import Lz, lcat, KEYS
    D = G.D
    n = dom.n
    tmap, _ = dom._tabmap()
    ri, rb = rho[:, :n - 1], rho[:, n - 1:]
    full = lcat([ri, Lz(t.view(D, 1), keys=KEYS.get(t))])
    # interpolated values of every two-half column at once: f0 + t (f1 - f0)
    i0s = [tmap[k][0] for k in range(len(dom.cols))]
    i1s = [tmap[k][1] for k in range(len(dom.cols))]
    idx0 = torch.cat([torch.arange(sl[i].start, sl[i].stop, device=_dev()) for i in i0s])
    idx1 = torch.cat([torch.arange(sl[i].start, sl[i].stop, device=_dev()) for i in i1s])
    f0, f1 = FB[:, idx0], FB[:, idx1]
    d = ((f1.to(torch.int64) - f0.to(torch.int64)) % P).to(torch.int32)
    tb = t.view(D, 1).expand(D, f0.shape[1]).contiguous()
    V = Lz(G.add(f0.contiguous(), G.mul(d.contiguous(), tb)))
    out, off = {}, 0
    for k, col in enumerate(dom.cols):
        i0, i1 = i0s[k], i1s[k]
        w = sl[i0].stop - sl[i0].start
        val = V[:, off:off + w]                   # (for one-half columns f1 = f0: the value is f0)
        off += w
        if i0 != i1:
            pt = full if col.kind in (FULL, COL) else full[:, col.c:]
        else:
            pt = ri[:, :col.c] if col.kind == COL else ri[:, :0]
        if dom.merge and not col.shared:
            pt = lcat([pt, rb])
        out[col.name] = (pt, val)
    return out


def _prove(tr, dom, Z, beta, label=b"dom"):
    if os.environ.get("ZK_DOMHOST"):
        return _prove_host(tr, dom, Z, beta, label)
    return _prove_dev(tr, dom, Z, beta, label)


_INV_EXP, _INV_BITS = {}, {}


def _inv_exp(D):
    """Device words of p^D - 2 (x^(p^D - 2) = 1 / x in F_p^D)."""
    if D not in _INV_EXP:
        e = P ** D - 2
        nw = (e.bit_length() + 31) // 32
        _INV_EXP[D] = h2d_words(np.array([(e >> (32 * i)) & 0xffffffff for i in range(nw)], dtype=np.uint32))
        _INV_BITS[D] = e.bit_length()
    return _INV_EXP[D]


def h2d_words(a):
    from .kern import h2d
    return h2d(a.view(np.int32))


def _prove_dev(tr, dom, Z, beta, label=b"dom"):
    """_prove_host with the whole domain proof (GKR, challenges, leaf/zerocheck sumcheck, finals)
    queued on one device transcript: one synchronization per domain.  Same transcript bytes."""
    from .kern import h2d, ptr
    D = tr.D
    G = gops(D)
    F = fe(D)
    n, K = dom.n, dom.K
    tr.absorb(label + label_name(dom.name))
    L = len(dom.lookups)
    tick = _ticker(dom.name)
    from . import devtr as _dv
    dt = _dv.get(tr)
    sess = dt.session
    if L:
        cols = [Tab(c.data, c.kind, c.c, shared=c.shared) for c in dom.cols]
        top, nxt = gkr.leaf_layer(G, n, K, cols, dom.lookups, Z, beta, pair=True) if n >= 3 else \
            (gkr.leaf_layer(G, n, K, cols, dom.lookups, Z, beta), None)
        tick("leaf")
        drop = gkr.leaf_bytes(D, n, K, L) > (24 << 30)
        layers, tt = gkr.build(G, top, K, n, drop_leaves=drop, nxt=nxt)
        del top, nxt
        tick("build")

        def _leaf():
            return gkr.leaf_layer(G, n, K, cols, dom.lookups, Z, beta)
        r, rb, _, _ = gkr.prove_upper(tr, G, n, K, layers, tt, L, leaf_fn=_leaf if drop else None, dt=dt)
        del layers, tt
        tick("upper")
    else:
        r = dt.challenge(n - 1)
        rb = dt.challenge(sc.nbits(K))
        tick("leaf")
        tick("build")
        tick("upper")
    bw = gkr._eq_dev(G, rb, K)
    # _challenges, on the device
    a = dt.challenge(max(1, L))
    lam = dt.challenge()
    gam = dt.challenge(max(1, len(dom.constraints)))
    eta = dt.challenge()
    recipe = []
    prog = dom.prog(F, None, None, None, None, None, None, recipe=recipe)
    kind = h2d(np.array([_COEF_KIND[k] for k, _, _ in recipe], dtype=np.int32))
    idx = h2d(np.array([i for _, i, _ in recipe], dtype=np.int32))
    bval = h2d(np.array([v for _, _, v in recipe], dtype=np.uint32).view(np.int32))
    coefs = torch.empty((len(recipe), D), dtype=torch.int32, device=_dev())
    Zd, bd = G.to_dev(np.asarray(Z) % P), G.to_dev(np.stack(gkr.tuple_coefs(beta, gkr.NBETA)))
    mu = torch.empty(D, dtype=torch.int32, device=_dev())
    kern.launch(gkr._gdev(D)["ef_pow1"], 1, 1, [ptr(lam), ptr(_inv_exp(D)), np.int32(_INV_BITS[D]), ptr(mu)])
    if LPARAMS_DEV is None:
        set_lparams(LPARAMS)
    kern.launch(gkr._gdev(D)["dom_coefs"], (len(recipe) + 63) // 64, 64,
                [ptr(kind), ptr(idx), ptr(bval), np.int32(len(recipe)), ptr(Zd), ptr(bd), ptr(a), ptr(lam),
                 ptr(gam), ptr(eta), ptr(mu), ptr(LPARAMS_DEV), ptr(coefs)])
    tabs = dom._tables(G)
    if os.environ.get("ZK_DOMDEG"):
        kinds_, cs_ = dom.kinds()
        print(f"[deg] {dom.name:28s} n={n} K={K} L={L} cols={len(dom.cols)} tabs={len(kinds_)} "
              f"deg={prog.degree(kinds_, cs_, n - 1)} monos={sum(len(p_.t) for p_ in prog.polys)} "
              f"polys={len(prog.polys)}", flush=True)
    claim = np.zeros(D, dtype=np.int64)            # the prover does not track the claim
    if dom.merge:
        rho, (FB, sl), _ = sc.prove_blocks(dt, D, n - 1, tabs, None, claim, K, bw, eq_r=r, label=label, prog=prog,
                                           coefs_dev=coefs)
    else:
        rho, (FB, sl), _ = sc.prove(dt, D, n - 1, tabs, None, claim, eq_r=r, K=K, bw=bw, label=label, prog=prog,
                                    dev=True, coefs_dev=coefs)
    tick("final")
    # finals: one (K_t, D) message per table, as tr.send(f.T)
    if all(s_.stop - s_.start == 1 for s_ in sl):
        dt.send_cols(FB, [s_.start for s_ in sl])
    else:
        for s_ in sl:
            dt.send(FB[:, s_].T.contiguous())
    t = dt.challenge()
    if sess:
        # no synchronization: the claims stay on the device until the host reads them
        from .lazy import rows_lz
        return _claims_lazy(G, dom, rows_lz(rho), t, FB, sl)
    rho_h = torch.empty(tuple(rho.shape), dtype=torch.int32, pin_memory=True)
    fb_h = torch.empty(tuple(FB.shape), dtype=torch.int32, pin_memory=True)
    t_h = torch.empty((D,), dtype=torch.int32, pin_memory=True)
    rho_h.copy_(rho, non_blocking=True)
    fb_h.copy_(FB, non_blocking=True)
    t_h.copy_(t, non_blocking=True)
    dt.finish()
    rho = rho_h.numpy().view(np.uint32).astype(np.int64).T.copy() if rho.shape[0] else np.zeros((D, 0), dtype=np.int64)
    fbh = fb_h.numpy().view(np.uint32).astype(np.int64)
    fin = [fbh[:, s_] for s_ in sl]
    th = t_h.numpy().view(np.uint32).astype(np.int64)
    return _claims(F, dom, rho, th, fin)


def _prove_host(tr, dom, Z, beta, label=b"dom"):
    """Returns claims: name -> (point, values) (see _claims)."""
    D = tr.D
    G = gops(D)
    F = fe(D)
    n, K = dom.n, dom.K
    tr.absorb(label + label_name(dom.name))
    L = len(dom.lookups)
    tick = _ticker(dom.name)
    if os.environ.get("ZK_DOMT") == "2":
        print(f"dom {dom.name} n={n} K={K} L={L} merge={dom.merge}", flush=True)
    if L:
        cols = [Tab(c.data, c.kind, c.c, shared=c.shared) for c in dom.cols]
        top = gkr.leaf_layer(G, n, K, cols, dom.lookups, Z, beta)
        tick("leaf")
        drop = gkr.leaf_bytes(D, n, K, L) > (24 << 30)
        layers, tt = gkr.build(G, top, K, n, drop_leaves=drop)
        del top
        tick("build")

        def _leaf():
            return gkr.leaf_layer(G, n, K, cols, dom.lookups, Z, beta)
        r, rb, cp, cq = gkr.prove_upper(tr, G, n, K, layers, tt, L, leaf_fn=_leaf if drop else None)
        del layers, tt
        tick("upper")
    else:
        r = tr.challenge(n - 1)
        rb = tr.challenge(sc.nbits(K))
        cp = cq = []
        tick("leaf")
        tick("build")
        tick("upper")
    bw = gkr.block_weights(G, rb, K, dev=True)
    a, lam, gam, eta = _challenges(tr, dom)
    claim = gkr._claim(F, cp, cq, a, lam) if L else np.zeros(D, dtype=np.int64)
    prog = dom.prog(F, F.c(Z), beta, a, lam, gam, eta)
    tabs = dom._tables(G)
    if dom.merge:
        rho, fin, red = sc.prove_blocks(tr, D, n - 1, tabs, None, claim, K, bw, eq_r=r, label=label, prog=prog)
    else:
        rho, fin, red = sc.prove(tr, D, n - 1, tabs, None, claim, eq_r=r, K=K, bw=bw, label=label, prog=prog)
    tick("final")
    for f in fin:
        tr.send(f.T)
    t = tr.challenge()
    return _claims(F, dom, rho, t, fin)


DOMT = {}


_NXT = {"leaf": "d.build", "build": "d.upper", "upper": "d.final", "final": "d.claims"}


def _ticker(name):
    import os
    import re
    import time
    if os.environ.get("ZK_NVTX"):
        torch.cuda.nvtx.range_push("d.leaf")

        def tick(what):
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(_NXT[what])
        return tick
    if not os.environ.get("ZK_DOMT"):
        return lambda what: None
    key = re.sub(r"[0-9]+", "#", name)
    st = [time.time()]

    def tick(what):
        torch.cuda.synchronize()
        t = time.time()
        DOMT[(key, what)] = DOMT.get((key, what), 0.0) + t - st[0]
        st[0] = t
    return tick


def verify(tr, dom, Z, beta, label=b"dom"):
    """Returns (claims, roots [(P, Q)] per lookup)."""
    D = tr.D
    G = gops(D)
    E = G.E
    F = fe(D)
    n, K = dom.n, dom.K
    tr.absorb(label + label_name(dom.name))
    L = len(dom.lookups)
    roots = []
    if L:
        roots, r, rb, cp, cq = gkr.verify_upper(tr, E, n, K, L)
    else:
        r = tr.challenge(n - 1)
        rb = tr.challenge(sc.nbits(K))
        cp = cq = []
    bw = gkr.block_weights(G, rb, K)
    a, lam, gam, eta = _challenges(tr, dom)
    claim = gkr._claim(F, cp, cq, a, lam) if L else np.zeros(D, dtype=np.int64)
    prog = dom.prog(F, F.c(Z), beta, a, lam, gam, eta)
    kinds, cs = dom.kinds()
    d = prog.degree(kinds, cs, n - 1)
    if dom.merge:
        rho, red = sc.verify_blocks(tr, E, n - 1, d, claim, K, dom.shared(), eq_r=r, label=label, prog=prog)
        fin = [tr.send(None).T for _ in kinds]
        assert all(f.shape == (D, 1) for f in fin)
        bwv = sc.mle_vec(G, bw, rho[:, n - 1:]) if K > 1 else bw[:, 0]
        if F.c(red) != F.mul(F.c(bwv), F.c(sc.eval_prog(E, prog, fin, None))):
            raise ValueError(f"domain {dom.name}: leaf/zerocheck sumcheck failed")
    else:
        rho, red = sc.verify(tr, E, n - 1, d, claim, eq_r=r, K=K, label=label)
        fin = [tr.send(None).T for _ in kinds]
        if not np.array_equal(red % P, sc.eval_prog(E, prog, fin, bw)):
            raise ValueError(f"domain {dom.name}: leaf/zerocheck sumcheck failed")
    t = tr.challenge()
    return _claims(F, dom, rho, t, fin), roots


def frac_sum(F, roots):
    """sum of P/Q over a list of (P, Q) FE tuples."""
    s = F.zero
    for p, q in roots:
        s = F.add(s, F.mul(p, F.inv(q)))
    return s
