"""LogUp chunks and the batched table side.

Static lookups (domain, lookup) are assigned in commit order to chunks of at
most LIMIT lookups; chunk i has its own multiplicity column m_i (a PCS unit
'mult{i}' of 2^LOG_ROWS entries).  The prover sends the assignment before Z.
The verifier sums the root fractions of each chunk into S_i, draws gamma, and
one GKR over the table proves
    sum_v M(v) / (Z - T(v)) = sum_i gamma_i S_i,   M = sum_i gamma_i m_i,
ending in claims m~_i(pt) for every chunk.  Multiplicities stay on the host
as uint8 with a sparse list of entries >= 255."""
import numpy as np
import torch
from .field import P
from .fe import fe
from .ext import gops
from . import gkr
from . import gpcs
from . import tables as tb
from . import sumcheck as sc
from .sumcheck import Tab, FULL
from .ext import dev as _dev

LIMIT = 1 << 30


class Chunks:
    def __init__(self, limit=LIMIT):
        self.limit = limit
        self.cur, self.cur_n = None, 0
        self.store = []              # (uint8 host, exc idx, exc vals)
        self.assign = {}             # (domain, lookup) -> chunk
        self.sizes = []
        self._pending, self._side = [], None
        self.total = 0                 # static lookups taken (statistics)

    def fits(self, cnt):
        """take(cnt) stays in the current chunk."""
        return self.cur is not None and self.cur_n + cnt <= self.limit

    def take(self, key, cnt):
        assert cnt <= self.limit and key not in self.assign
        if self.cur is None or self.cur_n + cnt > self.limit:
            self._close()
            self.cur = torch.zeros(1 << tb.LOG_ROWS, dtype=torch.int32, device=_dev())
            self.cur_n = 0
        self.cur_n += cnt
        self.total += cnt
        self.assign[key] = len(self.store)
        return self.cur

    def _close(self):
        """Close the current chunk.  Its compression to the host runs on a side stream after an
        event recorded here, one chunk later (or at finish), so the main stream is never drained."""
        if self.cur is None:
            return
        ev = torch.cuda.Event()
        ev.record()
        self._pending.append((self.cur, ev, len(self.store)))
        self.store.append(None)
        self.sizes.append(self.cur_n)
        self.cur = None
        self._drain(1)

    def _drain(self, keep):
        while len(self._pending) > keep:
            m, ev, i = self._pending.pop(0)
            if self._side is None:
                self._side = torch.cuda.Stream()
            with torch.cuda.stream(self._side):
                self._side.wait_event(ev)
                m.record_stream(self._side)
                big = (m >= 255).nonzero().view(-1)          # syncs the side stream only
                u8 = torch.empty(m.shape, dtype=torch.uint8, pin_memory=True)
                u8.copy_(m.clamp(max=255).to(torch.uint8), non_blocking=True)
                self.store[i] = (u8, big.cpu().pin_memory(), m[big].cpu().pin_memory())
                self._side.synchronize()

    def finish(self):
        self._close()
        self._drain(0)
        return len(self.store)

    def get(self, i):
        u8, idx, vals = self.store[i]
        t = u8.cuda(non_blocking=True).to(torch.int32)
        if idx.numel():
            t[idx.cuda(non_blocking=True)] = vals.cuda(non_blocking=True)
        return t

    def n(self):
        return len(self.store)

    def assign_bytes(self):
        lines = [f"{d}\t{l}\t{c}" for (d, l), c in self.assign.items()]
        return ("\n".join(lines)).encode()


def parse_assign(b):
    out = {}
    if not b:
        return out
    for ln in b.decode().split("\n"):
        d, l, c = ln.split("\t")
        out[(d, int(l))] = int(c)
    return out


def _one(D):
    o = np.zeros(D, dtype=np.int64)
    o[0] = 1
    return o


def _qtable(G, tcols, Z, beta):
    """Z - (T0 + beta_0 T1 + beta_1 T2 + ...) as ext (D, N)."""
    F = fe(G.D)
    N = tcols[0].numel()
    co = [np.array(F.neg(F.c(b)), dtype=np.int64) for b in [_one(G.D)] + gkr.tuple_coefs(beta, len(tcols) - 1)]
    Q = G.zeros(N)
    gpcs.lincomb(Q, list(tcols), np.stack(co, axis=1))
    z = torch.from_numpy((np.asarray(Z, dtype=np.int64) % P)).to(_dev()).view(-1, 1)
    return ((Q.to(torch.int64) + z) % P).to(torch.int32)


def table_side(ctx, tcols, chunks, S, tmle):
    """Prover (tcols: public table columns on the GPU, chunks: Chunks) or verifier
    (S: chunk sums, tmle(pt) -> numpy (D, NV+1) table MLEs).  Adds claims on mult{i}."""
    tr, D = ctx.tr, ctx.D
    F, G = fe(D), gops(D)
    n = tb.LOG_ROWS
    ncnk = chunks.n() if ctx.P else len(S)
    tr.absorb(b"table-side")
    gam = tr.challenge(ncnk)
    if ctx.P:
        M = G.zeros(1 << n)
        for i in range(ncnk):
            gpcs.lincomb(M, [chunks.get(i)], gam[:, i:i + 1])
        Q = _qtable(G, tcols, ctx.Z, ctx.beta)
        Pn = ((P - M.to(torch.int64)) % P).to(torch.int32)
        half = 1 << (n - 1)
        p0, p1, q0, q1 = Pn[:, :half], Pn[:, half:], Q[:, :half], Q[:, half:]
        top = [(G.add(G.mul(p0.contiguous(), q1.contiguous()), G.mul(p1.contiguous(), q0.contiguous())),
                G.mul(q0.contiguous(), q1.contiguous()))]
        del Pn
        layers, tt = gkr.build(G, top, 1, n)
        ri, rb, cp, cq = gkr.prove_upper(tr, G, n, 1, layers, tt, 1, label=b"tgkr")
        del layers, tt, top
    else:
        roots, ri, rb, cp, cq = gkr.verify_upper(tr, G.E, n, 1, 1, label=b"tgkr")
        R = F.mul(roots[0][0], F.inv(roots[0][1]))
        tot = R
        for i in range(ncnk):
            tot = F.add(tot, F.mul(F.c(gam[:, i]), S[i]))
        if tot != F.zero:
            raise ValueError("LogUp: chunk sums do not match the table side")
    a = F.c(tr.challenge())
    lam = F.c(tr.challenge())
    claim = gkr._claim(F, cp, cq, [a], lam)
    na = np.array(F.neg(a), dtype=np.int64)
    terms = [(na, [0, 3]), (na, [1, 2]), (np.array(F.mul(a, lam), dtype=np.int64), [2, 3])]
    if ctx.P:
        half = 1 << (n - 1)
        tabs = [Tab(M, FULL, view=(n - 1, 2 * half, 0)), Tab(M, FULL, view=(n - 1, 2 * half, half)),
                Tab(Q, FULL, view=(n - 1, 2 * half, 0)), Tab(Q, FULL, view=(n - 1, 2 * half, half))]
        rho, fin, _ = sc.prove(tr, D, n - 1, tabs, terms, claim, eq_r=ri, label=b"tleaf")
        del tabs, Q
        fm = [tr.send(fin[0].T).T, tr.send(fin[1].T).T]
    else:
        rho, red = sc.verify(tr, G.E, n - 1, 2, claim, eq_r=ri, label=b"tleaf")
        fm = [tr.send(None).T, tr.send(None).T]
    # the table's MLEs at (rho, 0) and (rho, 1): sent by the prover, claims on the committed "tab"
    # unit (checked by its opening) instead of evaluated by the verifier
    tvs = []
    for h in (0, 1):
        pt = np.concatenate([np.asarray(rho, dtype=np.int64) % P, _const(D, h)], axis=1)
        if ctx.P:
            tv = np.stack([np.asarray(G.mle(c, pt), dtype=np.int64) for c in tcols], axis=1)
            tv = tr.send(tv.T).T
        else:
            tv = tr.send(None).T
            assert tv.shape == (D, tb.NV + 1)
        for j in range(tv.shape[1]):
            ctx.claim(f"tab{j}", pt, {0: tv[:, j]})
        tvs.append(tv)
    if not ctx.P:
        qf = []
        for h in (0, 1):
            tv = tvs[h]
            qv = F.c(ctx.Z)
            for j, bp in enumerate([_one(D)] + gkr.tuple_coefs(ctx.beta, tv.shape[1] - 1)):
                qv = F.sub(qv, F.mul(F.c(bp), F.c(tv[:, j])))
            qf.append(np.array(qv, dtype=np.int64).reshape(D, 1))
        # finals [M0, M1, Q0, Q1] with the numerator sign folded into the terms
        if not np.array_equal(red % P, sc.eval_terms(G.E, terms, [fm[0], fm[1], qf[0], qf[1]])):
            raise ValueError("LogUp: table leaf sumcheck failed")
    t = tr.challenge()
    pt = np.concatenate([rho, t.reshape(D, 1)], axis=1)
    mt = gkr._lerp(F, fm[0][:, 0], fm[1][:, 0], t)
    if ctx.P:
        e = G.eq_table(pt)
        mv = np.stack([G.dot(chunks.get(i), e) for i in range(ncnk)], axis=1)
        del e, M
        tr.send(mv.T)
    else:
        mv = tr.send(None).T
        assert mv.shape == (D, ncnk)
        s = F.zero
        for i in range(ncnk):
            s = F.add(s, F.mul(F.c(gam[:, i]), F.c(mv[:, i])))
        if s != F.c(mt):
            raise ValueError("LogUp: multiplicity values do not match")
    for i in range(ncnk):
        ctx.claim(f"mult{i}", pt, {0: mv[:, i]})


def _const(D, v):
    out = np.zeros((D, 1), dtype=np.int64)
    out[0, 0] = v
    return out
