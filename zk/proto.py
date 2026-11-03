"""Protocol context shared by prover and verifier.

The same step code runs in three modes:
  'commit'  prover, pass 1: witness -> commitments and global multiplicities
  'prove'   prover, pass 2: domains and sumchecks; claims routed to tensors
  'verify'  verifier: replays the transcript, data is None

Committed tensors are K blocks of 2^c base-field values owned by a commitment
group (a PCS unit, gpcs.py).  A claim is (point (D, c), {block: value}) or a
weighted claim (point, {block: coefficient}, value).  A group is opened
(per-unit point reduction) once every claim on it is known; the global
opening runs at the end (Ctx.finish).  The IOP runs in F_q (D = 4)."""
import os
import numpy as np
import torch
from .field import P
from .fe import fe
from .ext import gops
from . import sumcheck as sc
from . import domain as dm
from . import gpcs
from . import logup
from . import tables
from .sumcheck import Tab


class TInfo:
    def __init__(self, name, K, c, group):
        self.name, self.K, self.c, self.group = name, K, c, group


PACK_PAD = 0.10
PACK_ABS = int(os.environ.get("ZK_PACK_ABS", str(1 << 20)))


class Pack:
    """K = Kin * Kout blocks of 2^c stored as Kin * G PCS columns of 2^c2: column
    (tb, ob // bpc) holds blocks ob of row tb, low bits ob % bpc (zero padded)."""

    def __init__(self, K, c, shape, cmax, pad=None):
        self.Kin, self.Kout = shape if shape else (1, K)
        assert self.Kin * self.Kout == K
        self.c = c
        # widest columns whose zero padding stays within PACK_PAD of the data or within PACK_ABS
        # elements (small tensors: a few padded blocks cost little, while every column is opened
        # by every query)
        lb = min(cmax, c + nbits(self.Kout)) - c
        while lb > 0:
            npad = (((-(-self.Kout >> lb)) << lb) - self.Kout) * self.Kin << c
            if npad <= self.Kout * self.Kin * (PACK_PAD if pad is None else pad) * (1 << c) or npad <= PACK_ABS:
                break
            lb -= 1
        self.c2 = c + lb
        self.lb = lb
        self.bpc = 1 << self.lb
        self.G = -(-self.Kout // self.bpc)
        self.K2 = self.Kin * self.G

    def loc(self, b):
        tb, ob = divmod(b, self.Kout)
        return tb * self.G + ob // self.bpc, ob % self.bpc

    def nblk(self, j):
        """Real blocks of column j."""
        return min(self.bpc, self.Kout - (j % self.G) * self.bpc)

    def layout(self, t):
        t = t.reshape(self.Kin, self.Kout, 1 << self.c)
        if self.G * self.bpc != self.Kout:
            out = torch.zeros((self.Kin, self.G * self.bpc, 1 << self.c), dtype=t.dtype, device=t.device)
            out[:, :self.Kout] = t
            t = out
        return t.reshape(-1)


def nbits(K):
    return max(0, (K - 1).bit_length())


class Ctx:
    def __init__(self, mode, tr, D=4, **pcs_args):
        assert mode in ("commit", "prove", "verify")
        self.mode = mode
        self.tr, self.D = tr, D
        self.F, self.G = fe(D), gops(D)
        self.E = self.G.E
        self.tinfo = {}
        self.groups = {}             # gname -> [(name, K, c)]
        self.claims = {}             # name -> [(pt, {b: val})]
        self.wclaims = {}            # name -> [(pt, {b: coef}, value)]
        self.virt = {}
        self.pcs_args = dict(pcs_args)
        self.pcs = gpcs.GPCS(tr, mode != "verify", **pcs_args)
        self.chunks = logup.Chunks() if mode != "verify" else None
        self.assign = None           # verifier: (domain, lookup) -> chunk
        self.csum, self.ccount, self.dsum, self.seen = {}, {}, {}, set()
        self.Z = self.beta = None
        self.stats = {}

    @property
    def P(self):
        return self.mode != "verify"

    @property
    def proving(self):
        return self.mode == "prove"

    @property
    def active(self):
        return self.mode != "commit"

    # ------------------------------------------------------------ tensors
    def group(self, gname, items, pad=None):
        """Declare a commitment group: items [(name, K, c)] or [(name, K, c, (Kin, Kout))].
        Tensors of many small blocks are packed into larger PCS columns (Pack); pad: the padding
        allowance of this group's packing (default PACK_PAD)."""
        if not hasattr(self, "pack"):
            self.pack = {}
        pitems = []
        for it in items:
            name, K, c = it[:3]
            self.tinfo[name] = TInfo(name, K, c, gname)
            shape = it[3] if len(it) > 3 else None
            pk = (Pack(K, c, shape, self.pcs.cmax, pad)
                  if K > 1 and c < self.pcs.cmax and shape is not False else None)
            if pk is not None and pk.bpc > 1:
                self.pack[name] = pk
                pitems.append((name, pk.K2, pk.c2))
            else:
                pitems.append((name, K, c))
        self.groups[gname] = pitems
        self.pcs.declare(gname, pitems)

    def _ptensors(self, tensors):
        if tensors is None:
            return None
        return {k: (self.pack[k].layout(v) if k in self.pack else v) for k, v in tensors.items()}

    def commit(self, gname, tensors=None):
        """Commit pass (prover) / receive caps (verifier)."""
        self.pcs.commit(gname, self._ptensors(tensors))

    def open(self, gname, tensors=None):
        cl = []
        one = np.zeros(self.D, dtype=np.int64)
        one[0] = 1
        if self.P:
            # the prover's opening needs points and coefficients only (lazy, never read here)
            for name, K, c in self.groups[gname]:
                for pt, vals in self.claims.pop(name, []):
                    for b in vals:
                        cl.append((name, pt, {b: one}, None))
                for pt, wb, v in self.wclaims.pop(name, []):
                    cl.append((name, pt, dict(wb), None))
            self.pcs.open(gname, cl, self._ptensors(tensors))
            return
        from .lazy import materialize
        materialize([(self.claims.get(name), self.wclaims.get(name)) for name, _, _ in self.groups[gname]])
        for name, K, c in self.groups[gname]:
            for pt, vals in self.claims.pop(name, []):
                pt = _host(pt)
                for b, v in vals.items():
                    cl.append((name, pt, {b: one}, _host(v)))
            for pt, wb, v in self.wclaims.pop(name, []):
                cl.append((name, _host(pt), {b: _host(w) for b, w in wb.items()}, _host(v)))
        self.pcs.open(gname, cl, self._ptensors(tensors))

    def wclaim(self, name, pt, wb, value):
        """Weighted claim: sum_b wb[b] f~_{name, b}(pt) = value."""
        info = self.tinfo[name]
        assert pt.shape[1] == info.c, (name, pt.shape, info.c)
        assert name not in getattr(self, "pack", {}), name
        self._wclaim(name, pt, wb, value)

    def _wclaim(self, name, pt, wb, value):
        # stored as given (lazy device arrays stay lazy): converted when the group is opened
        self.wclaims.setdefault(name, []).append((pt, dict(wb), value))

    def finish(self, provider=None):
        """Global opening (provider: iterator factory of (gname, tensors) over every group)."""
        if provider is None:
            self.pcs.finish(None)
            return

        def prov(skip=None):
            try:
                it = provider(skip) if skip else provider()
            except TypeError:
                it = provider()
            for g, t in it:
                yield g, self._ptensors(t)
        self.pcs.finish(prov)

    # ------------------------------------------------------------ LogUp sets
    def _root(self, dom, l, pq):
        tag, _, num = dom.lookups[l]
        if not dom.dyn[l] and num is None:
            key = (dom.name, l)
            if key not in self.assign:
                raise ValueError(f"LogUp: lookup {key} has no chunk")
            if key in self.seen:
                raise ValueError(f"LogUp: lookup {key} proved twice")
            self.seen.add(key)
            c = self.assign[key]
            self.csum[c] = self.F.add(self.csum.get(c, self.F.zero), self.F.mul(pq[0], self.F.inv(pq[1])))
            self.ccount[c] = self.ccount.get(c, 0) + (dom.K << dom.n)
        else:
            sid = dm.tag_value(tag)
            if sid is None:
                raise ValueError(f"domain {dom.name}: dynamic lookup {l} without a constant tag")
            self.dsum[sid] = self.F.add(self.dsum.get(sid, self.F.zero), self.F.mul(pq[0], self.F.inv(pq[1])))

    def check_sets(self):
        """Verifier: dynamic sets sum to zero; chunk sizes; every assigned lookup proved."""
        for sid, s in self.dsum.items():
            if s != self.F.zero:
                raise ValueError(f"LogUp: dynamic set {sid} does not sum to zero")
        for c, n in self.ccount.items():
            if n > logup.LIMIT:
                raise ValueError(f"LogUp: chunk {c} has {n} lookups")
        if set(self.assign) != self.seen:
            raise ValueError("LogUp: assigned lookups not proved")

    def claim(self, name, pt, vals):
        """Block claim: pt (D, c) in-block point, vals dict b -> (D,) or array (D, K)."""
        info = self.tinfo[name]
        assert pt.shape[1] == info.c, (name, pt.shape, info.c)
        if not isinstance(vals, dict):
            vals = {b: vals[:, b] for b in range(vals.shape[1])}
        pk = getattr(self, "pack", {}).get(name)
        if pk is None:
            self._claim(name, pt, vals)
            return
        F = self.F
        bycol = {}
        for b, v in vals.items():
            j, lo = pk.loc(b)
            bycol.setdefault(j, {})[lo] = v
        full = {j: d for j, d in bycol.items() if len(d) == pk.nblk(j)}
        if full:
            tau = self.chal(pk.lb)
            if self.P:                   # the prover never reads claimed values (prover-side open)
                comb = {j: None for j in full}
            else:
                e = self.G.to_host(self.G.eq_table(tau))
                comb = {}
                for j, d in full.items():
                    s = F.zero
                    for lo, v in d.items():
                        s = F.add(s, F.mul(F.c(e[:, lo]), F.c(v)))
                    comb[j] = np.array(s, dtype=np.int64)
            self._claim(name, cat(pt, tau), comb)
        bylo = {}
        for j, d in bycol.items():
            if j not in full:
                for lo, v in d.items():
                    bylo.setdefault(lo, {})[j] = v
        for lo, d in bylo.items():
            self._claim(name, cat(pt, const_pt(self.D, [(lo >> i) & 1 for i in range(pk.lb)])), d)

    def _claim(self, name, pt, vals):
        self.claims.setdefault(name, []).append((pt, dict(vals)))

    def point_claim(self, name, pt, value=None, data=None):
        """Claim on the full tensor at pt (D, c + nbits(K)); the prover sends all block
        values (computed from data (K*2^c,)) and both sides check their eq-combination."""
        info = self.tinfo[name]
        c, K = info.c, info.K
        nb = nbits(K)
        assert pt.shape[1] == c + nb, (name, pt.shape, c, nb)
        pk = getattr(self, "pack", {}).get(name)
        if pk is not None and pk.Kin == 1 and value is not None:
            # block bits = (column-internal low bits, column index): a weighted claim, nothing sent
            hi = pt[:, c + pk.lb:]
            if hi.shape[1] and self.P:
                from .lazy import Lz
                e = Lz(self.G.eq_table(hi))
            elif hi.shape[1]:
                e = self.G.to_host(self.G.eq_table(hi))
            else:
                e = np.zeros((self.D, 1), dtype=np.int64)
                e[0, 0] = 1
            self._wclaim(name, pt[:, :c + pk.lb], {j: e[:, j] for j in range(pk.K2)}, value)
            return value if self.P else np.asarray(value, dtype=np.int64)
        if self.P:                       # device values sent, nothing read; no self-check
            e = self.G.eq_table(pt[:, :c])
            vals = self.send_vals(self.G.rowdot(data.reshape(-1), e, K, 1 << c))
            tot = self.blockcomb(pt[:, c:], vals)
            self.claim(name, pt[:, :c], vals)
            return tot
        else:
            vals = self.tr.send(None).T
            assert vals.shape == (self.D, K)
        tot = self.blockcomb(pt[:, c:], vals)
        if value is not None and tuple(int(x) for x in tot) != self.F.c(value):
            raise ValueError(f"point claim on {name}: block values do not match")
        self.claim(name, pt[:, :c], vals)
        return tot

    def claimsum(self, coefs, vals):
        """sum_i coefs[:, i] vals[i] for a sumcheck claim: the verifier's value; the prover never
        uses claimed values (zeros)."""
        F = self.F
        if self.P:
            return np.zeros(self.D, dtype=np.int64)
        s = F.zero
        for i, v in enumerate(vals):
            s = F.add(s, F.mul(F.c(coefs[:, i]), F.c(v)))
        return np.array(s, dtype=np.int64)

    def blockcomb(self, ptb, vals):
        """sum_b eq(ptb, b) vals[:, b] (the prover: a lazy device value, nothing read)."""
        F = self.F
        if self.P:
            from .lazy import Lz, to_dev
            if ptb.shape[1] == 0:
                return Lz(to_dev(vals)[:, 0].contiguous())
            e = self.G.eq_table(ptb)[:, :vals.shape[1]].contiguous()
            return Lz(self.G.dot_dev(e, to_dev(vals).contiguous()))
        if vals.shape[1] > 16:
            e = self.G.eq_table(ptb)[:, :vals.shape[1]].contiguous()
            return np.asarray(self.G.dot(e, self.G.to_dev(vals)), dtype=np.int64)
        s = F.zero
        for b in range(vals.shape[1]):
            e = F.one
            for j in range(ptb.shape[1]):
                bit = (b >> j) & 1
                x = F.c(ptb[:, j])
                e = F.mul(e, x if bit else F.sub(F.one, x))
            s = F.add(s, F.mul(e, F.c(vals[:, b])))
        return np.array(s, dtype=np.int64)

    # ------------------------------------------------------------ challenges
    def chal(self, k=None):
        return self.tr.challenge(k)

    # ------------------------------------------------------------ domains
    def run_domain(self, dom, routes):
        """routes: column name -> fn(pt, vals) (skipped in commit mode)."""
        if os.environ.get("ZK_DEBUG") and self.P:
            fp = tables.debug_tuples(dom, None)
            if self.mode == "commit":
                self.dbg_fp = getattr(self, "dbg_fp", {})
                self.dbg_fp[dom.name] = fp
            elif self.dbg_fp.get(dom.name) != fp:
                print(f"[dbg] domain {dom.name}: lookup data differs between passes "
                      f"{[i for i, (a, b) in enumerate(zip(fp, self.dbg_fp[dom.name])) if a != b]}", flush=True)
        if self.mode == "commit":
            tables.count(dom, self.chunks, Tab)
            if os.environ.get("ZK_DOMLOG"):
                N = dom.K << dom.n
                print(f"[dom] {dom.name:24s} n={dom.n:2d} K={dom.K:6d} N={N:.2e} L={len(dom.lookups)} "
                      f"cols={len(dom.cols)} cons={len(dom.constraints)} gkrGB={64 * len(dom.lookups) * N / 2**30:.1f}",
                      flush=True)
            return
        if self.proving:
            if getattr(self, "dbg_tcols", None) is not None:
                tables.debug_tuples(dom, self.dbg_tcols)
            if os.environ.get("ZK_MEMLOG"):
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
            cl = dm.prove(self.tr, dom, self.Z, self.beta)
            if os.environ.get("ZK_MEMLOG"):
                print(f"[mem] {dom.name:24s} before {m0 / 2**30:5.1f} peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB",
                      flush=True)
        else:
            cl, roots = dm.verify(self.tr, dom, self.Z, self.beta)
            for l, pq in enumerate(roots):
                self._root(dom, l, pq)
        for col in dom.cols:
            pt, vals = cl[col.name]
            r = routes.get(col.name)
            if r is None:
                raise KeyError(f"domain {dom.name}: no route for column {col.name}")
            r(pt, vals)

    # ------------------------------------------------------------ sumchecks
    def sumcheck(self, label, n, K, shapes, claim, tabs=None, terms=None, prog=None, eq_r=None, bw=None):
        """shapes: [(kind, c)] per table; tabs: prover tables (callable or list).
        Returns (rho (D, n), finals [(D, K_t)])."""
        tr = self.tr
        if self.proving:
            tl = tabs() if callable(tabs) else tabs
            if n == 0 or os.environ.get("ZK_HOSTTR"):
                rho, fin, red = sc.prove(tr, self.D, n, tl, terms, claim, eq_r=eq_r, K=K, bw=bw,
                                         label=label, prog=prog)
                for f in fin:
                    tr.send(f.T)
                return rho, fin
            # rounds, finals and the result download on one device transcript: one sync
            from .devtr import DevTr
            D = self.D
            sess = getattr(tr, "dev", None)
            dt = sess if sess is not None else DevTr(tr, cap=max(1, n * 16 * D + 64 * len(tl) * D))
            rho_d, (FB, sl), _ = sc.prove(dt, D, n, tl, terms, claim, eq_r=eq_r, K=K, bw=bw, label=label,
                                          prog=prog, dev=True)
            if all(s_.stop - s_.start == 1 for s_ in sl):
                dt.send_cols(FB, [s_.start for s_ in sl])
            else:
                for s_ in sl:
                    dt.send(FB[:, s_].T.contiguous())
            if sess is not None:
                from .lazy import Lz, rows_lz
                fb = Lz(FB)
                return rows_lz(rho_d), [fb[:, s_] for s_ in sl]
            rho_h = torch.empty(tuple(rho_d.shape), dtype=torch.int32, pin_memory=True)
            fb_h = torch.empty(tuple(FB.shape), dtype=torch.int32, pin_memory=True)
            rho_h.copy_(rho_d, non_blocking=True)
            fb_h.copy_(FB, non_blocking=True)
            dt.finish()
            rho = rho_h.numpy().view(np.uint32).astype(np.int64).T.copy()
            fbh = fb_h.numpy().view(np.uint32).astype(np.int64)
            return rho, [fbh[:, s_] for s_ in sl]
        kinds = [k for k, _ in shapes]
        cs = [c for _, c in shapes]
        d = prog.degree(kinds, cs, n) if prog is not None else sc.max_degree(terms, kinds, cs, n)
        rho, red = sc.verify(tr, self.E, n, d, claim, eq_r=eq_r, K=K, label=label)
        fin = [tr.send(None).T for _ in shapes]
        ev = sc.eval_prog(self.E, prog, fin, bw) if prog is not None else sc.eval_terms(self.E, terms, fin, bw)
        if not np.array_equal(red % P, ev % P):
            raise ValueError(f"sumcheck {label!r} failed")
        return rho, fin

    def send_vals(self, arr=None, shape=None):
        """Prover sends numpy or device (D, k) (device: returned lazy); verifier receives."""
        if self.P:
            if torch.is_tensor(arr):
                from .lazy import Lz
                if getattr(self.tr, "dev", None) is not None:
                    self.tr.send(arr.T.contiguous())
                    return Lz(arr)
                arr = arr.cpu().numpy().view(np.uint32).astype(np.int64)
            self.tr.send(np.asarray(arr).T)
            return np.asarray(arr, dtype=np.int64) % P
        v = self.tr.send(None).T
        if shape is not None:
            assert v.shape == shape, (v.shape, shape)
        return v

    # ------------------------------------------------------------ virtual
    def put(self, key, item):
        self.virt.setdefault(key, []).append(item)

    def take(self, key):
        return self.virt.pop(key, [])


# ---------------------------------------------------------------- helpers
def fe_arr(F, x):
    return np.array(x, dtype=np.int64)


def eq_point(F, a, b):
    """eq(a, b) for numpy (D, k) points -> FE."""
    e = F.one
    for j in range(a.shape[1]):
        e = F.mul(e, F.eq1(F.c(a[:, j]), F.c(b[:, j])))
    return e


def const_pt(D, bits):
    """numpy (D, len(bits)) with base-field constants (0/1 or any int)."""
    out = np.zeros((D, len(bits)), dtype=np.int64)
    out[0, :] = [int(b) % P for b in bits]
    return out


def _host(x):
    return np.asarray(x, dtype=np.int64) % P


def cat(*pts):
    from .lazy import Lz, lcat
    if not any(p.shape[1] for p in pts):
        return pts[0]
    if any(isinstance(p, Lz) for p in pts):
        return lcat(list(pts))
    return np.concatenate([p for p in pts if p.shape[1]], axis=1)
