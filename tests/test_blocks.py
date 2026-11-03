"""Block-phase sumcheck: K blocks, then rounds over the block index."""
import numpy as np
import torch
from zk.field import P
from zk.ext import gops
from zk.fe import fe
from zk.transcript import Transcript
from zk import sumcheck as sc
from zk.sumcheck import Tab, FULL, COL


def tracked(fn):
    """Run fn with sc.TRACK on (host-side claim tracking in the prover), restoring it afterwards."""
    def call(*a, **k):
        old, sc.TRACK = sc.TRACK, True
        try:
            return fn(*a, **k)
        finally:
            sc.TRACK = old
    call.__name__ = fn.__name__
    return call


def run(D, n, K, seed=0, bad=False):
    rng = np.random.default_rng(seed)
    G = gops(D)
    F = fe(D)
    N = 1 << n
    A = rng.integers(0, P, size=K * N)
    B = rng.integers(0, P, size=(D, K * N))
    C = rng.integers(0, P, size=(D, 1 << 2))           # shared COL 2
    terms = [(rng.integers(0, P, D), [0, 1]), (rng.integers(0, P, D), [0, 2]), (rng.integers(0, P, D), [1])]
    eq_r = rng.integers(0, P, size=(D, n))
    bw = rng.integers(0, P, size=(D, K))
    E = G.E
    ev = G.to_host(G.eq_table(eq_r))
    claim = np.zeros(D, dtype=np.int64)
    for b in range(K):
        a = A[b * N:(b + 1) * N]
        bb = B[:, b * N:(b + 1) * N]
        cc = C[:, np.arange(N) & 3]
        s = np.zeros((D, N), dtype=np.int64)
        for c, refs in terms:
            v = np.broadcast_to(np.asarray(c).reshape(-1, 1), (D, N)).copy()
            for k in refs:
                x = [E.from_base(a), bb, cc][k]
                v = E.mul(v, x)
            s = (s + v) % P
        tot = np.sum(E.mul(s, ev), axis=1) % P
        claim = (claim + np.array(F.mul(F.c(bw[:, b]), F.c(tot)))) % P
    if bad:
        claim = (claim + 1) % P
    tabs = [Tab(G.to_dev(A).reshape(-1)[0] if False else torch.from_numpy(A.astype(np.uint32).view(np.int32)).cuda(), FULL),
            Tab(G.to_dev(B), FULL), Tab(G.to_dev(C), COL, 2, shared=True)]
    tr = Transcript(b"t", D)
    rho, fin, red = sc.prove_blocks(tr, D, n, tabs, terms, claim, K, bw, eq_r=eq_r)
    for f in fin:
        tr.send(f.T)
    trv = Transcript(b"t", D, msgs=tr.msgs)
    d = sc.max_degree(terms, [FULL, FULL, COL], [0, 0, 2], n)
    rv, redv = sc.verify_blocks(trv, E, n, d, claim, K, [False, False, True], eq_r=eq_r, terms=terms)
    finv = [trv.send(None).T for _ in tabs]
    lk = sc.nbits(K)
    bwv = sc.mle_vec(G, bw, rv[:, n:])
    expv = F.mul(F.c(bwv), F.c(sc.eval_terms(E, terms, finv)))
    ok = F.c(redv) == expv
    # finals are MLEs of the zero-padded tables at the full point
    Ap = np.zeros(1 << (n + lk), dtype=np.int64); Ap[:K * N] = A
    fa = G.mle(torch.from_numpy(Ap.astype(np.uint32).view(np.int32)).cuda(), rv)
    ok2 = np.array_equal(np.asarray(fa) % P, finv[0][:, 0] % P)
    return ok, ok2


@tracked
def test_blocks():
    for D in (4, 8):
        for K in (1, 5, 8):
            ok, ok2 = run(D, 5, K)
            bad = run(D, 5, K, bad=True)[0]
            print(D, K, (ok, ok2), bad)
            assert ok and ok2 and not bad


if __name__ == "__main__":
    test_blocks()
