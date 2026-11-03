import time
import numpy as np
import torch
from zk.gkr import NBETA
from zk.field import P, Ext
from zk.fe import fe
from zk.transcript import Transcript
from zk import domain as dm
from zk.sumcheck import ROW, COL


def dev(x):
    return torch.from_numpy((np.asarray(x, dtype=np.int64) % P).astype(np.int32)).cuda()


def build(n, K, data, bad=False, merge=False):
    """data: dict of numpy arrays or None (verifier)."""
    g = (lambda k: dev(data[k])) if data else (lambda k: None)
    d = dm.Domain("toy", n, K, merge=merge)
    q = d.col("q", g("q"))
    u = d.col("u", g("u"))
    a = d.col("a", g("a"))
    s = d.col("s", g("s"), ROW, 4)
    pc = d.col("pc", g("pc"), COL, 3, shared=True, role="public")
    z = d.col("z", g("z"))
    d.zero(a - s * q - u)
    d.zero(z - pc * q)
    d.lookup(3, u)
    d.lookup(8, q + 127)
    return d


def table(nT, mult):
    g = (lambda k: dev(mult[k])) if mult else (lambda k: None)
    t = dm.Domain("table", nT, 1)
    tag = t.col("tag", g("tag"), role="public")
    v = t.col("v", g("v"), role="public")
    m = t.col("m", g("m"))
    t.lookup(tag, v, num=-m)
    return t


def run(D, n=8, K=3, seed=0, bad=False, merge=False):
    rng = np.random.default_rng(seed)
    E = Ext(D)
    F = fe(D)
    N = K << n
    q = rng.integers(-127, 128, N)
    u = rng.integers(0, 8, N)
    s = rng.choice([1, 2, 4], K << (n - 4))
    idx = np.arange(N)
    srep = s[(idx >> n << (n - 4)) + ((idx & ((1 << n) - 1)) >> 4)]
    a = srep * q + u
    pc = rng.integers(0, 50, 1 << 3)
    z = pc[idx & 7] * q
    if bad:
        u[5] = 9     # out of range: u in [0,8) violated, constraint kept consistent
        a[5] = srep[5] * q[5] + 9
    data = dict(q=q, u=u, a=a, s=s, pc=pc, z=z)
    rows = [(8, v) for v in range(255)] + [(3, v) for v in range(8)]
    nT = 9
    tag = np.zeros(1 << nT, dtype=np.int64)
    val = np.zeros(1 << nT, dtype=np.int64)
    m = np.zeros(1 << nT, dtype=np.int64)
    for i, (tg, v) in enumerate(rows):
        tag[i], val[i] = tg, v
    for v in u:
        if v < 8:
            m[255 + v] += 1
    for v in q:
        m[v + 127] += 1
    mult = dict(tag=tag, v=val, m=m)
    tr = Transcript(b"toy", D)
    Z, beta = tr.challenge(), tr.challenge(NBETA)
    t0 = time.time()
    cl = dm.prove(tr, build(n, K, data, merge=merge), Z, beta)
    dm.prove(tr, table(nT, mult), Z, beta)
    dt = time.time() - t0
    for name, (pt, val_) in cl.items():
        arr = data[name]
        dom = build(n, K, None, merge=merge)
        col = dom.cols[dom.col_index(name)]
        Kb = 1 if col.shared else K
        blk = arr.reshape(Kb, -1)
        if merge and not col.shared:
            lk = max(0, (K - 1).bit_length())
            full = np.zeros(blk.shape[1] << lk, dtype=np.int64)
            full[:arr.size] = arr % P
            assert E.eq(val_[:, 0], E.mle_eval(full, pt)), name
            continue
        for b in range(Kb):
            assert E.eq(val_[:, b], E.mle_eval(blk[b] % P, pt)), name
    tv = Transcript(b"toy", D, msgs=tr.msgs)
    Zv, bv = tv.challenge(), tv.challenge(NBETA)
    clv, roots = dm.verify(tv, build(n, K, None, merge=merge), Zv, bv)
    clvt, roots_t = dm.verify(tv, table(nT, None), Zv, bv)
    tot = F.zero
    tot = dm.frac_sum(F, roots + roots_t)
    for name in cl:
        assert np.array_equal(cl[name][1], clv[name][1])
    return tot == F.zero, dt


def test_domain():
    for D in (4, 8):
        for K in (1, 3, 4):
            for merge in (False, True):
                ok, dt = run(D, K=K, merge=merge)
                print("honest", D, K, merge, ok, f"{dt:.2f}s")
                assert ok
                ok, dt = run(D, K=K, merge=merge, bad=True)
                print("bad", D, K, merge, ok)
                assert not ok


if __name__ == "__main__":
    test_domain()
