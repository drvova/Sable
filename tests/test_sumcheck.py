import time
import numpy as np
import torch
from zk.field import P
from zk.ext import gops
from zk.transcript import Transcript
from zk import sumcheck as sc
from zk.sumcheck import Tab, FULL, ROW, COL


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


def run(D, n, c, use_eq, K=1, use_bw=False, seed=0):
    rng = np.random.default_rng(seed)
    G = gops(D)
    E = G.E
    N = 1 << n
    # A is the upper half (msb=1) of each block of a 2N-per-block buffer: a view
    Abig = rng.integers(0, P, size=K * 2 * N)
    A = Abig.reshape(K, 2, N)[:, 1, :].reshape(-1)
    B = rng.integers(0, P, size=(D, K * N))
    R = rng.integers(0, P, size=(D, K << (n - c)))
    Cb = rng.integers(0, 100, size=K << c)
    Cs = rng.integers(0, P, size=(D, 1 << c))
    terms = [(rng.integers(0, P, D), [0, 1]),
             (rng.integers(0, P, D), [0, 0, 2]),
             (rng.integers(0, P, D), [1, 3, 2, 4]),
             (rng.integers(0, P, D), [0]),
             (rng.integers(0, P, D), [])]
    idx = np.arange(N)
    bw = rng.integers(0, P, size=(D, K)) if use_bw else None
    eq_r = rng.integers(0, P, size=(D, n)) if use_eq else None
    claim = E.zero()
    for b in range(K):
        vals = [E.from_base(A[b * N:(b + 1) * N]), B[:, b * N:(b + 1) * N],
                R[:, (b << (n - c)) + (idx >> c)], E.from_base(Cb[(b << c) + (idx & ((1 << c) - 1))]),
                Cs[:, idx & ((1 << c) - 1)]]
        tot = E.zero(N)
        for cf, refs in terms:
            v = np.broadcast_to(cf.reshape(-1, 1), (D, N)).copy()
            for k in refs:
                v = E.mul(v, vals[k])
            tot = E.add(tot, v)
        if use_eq:
            tot = E.mul(tot, E.eq_table(eq_r))
        s = np.sum(tot, axis=1) % P
        if use_bw:
            s = E.mul(s.reshape(-1, 1), bw[:, b:b + 1])[:, 0]
        claim = E.add(claim, s)
    tabs = [Tab(G.to_dev(Abig).reshape(-1), FULL, view=(n, 2 * N, N)), Tab(G.to_dev(B), FULL),
            Tab(G.to_dev(R), ROW, c), Tab(G.to_dev(Cb).reshape(-1), COL, c),
            Tab(G.to_dev(Cs), COL, c, shared=True)]
    tr = Transcript(b"t", D)
    t0 = time.time()
    rho, finals, cl = sc.prove(tr, D, n, tabs, terms, claim, eq_r=eq_r, K=K, bw=bw)
    dt = time.time() - t0
    for b in range(K):
        assert E.eq(finals[0][:, b], E.mle_eval(A[b * N:(b + 1) * N], rho))
        assert E.eq(finals[1][:, b], E.mle_eval(B[:, b * N:(b + 1) * N], rho))
        assert E.eq(finals[2][:, b], E.mle_eval(R[:, b << (n - c):(b + 1) << (n - c)], rho[:, c:]))
        assert E.eq(finals[3][:, b], E.mle_eval(Cb[b << c:(b + 1) << c], rho[:, :c]))
    assert finals[4].shape[1] == 1 and E.eq(finals[4][:, 0], E.mle_eval(Cs, rho[:, :c]))
    assert E.eq(cl, sc.eval_terms(E, terms, finals, bw))
    kinds = [FULL, FULL, ROW, COL, COL]
    d = sc.max_degree(terms, kinds, [0, 0, c, c, c], n)
    tv = Transcript(b"t", D, msgs=tr.msgs)
    rho_v, cl_v = sc.verify(tv, E, n, d, claim, eq_r=eq_r, K=K)
    assert np.array_equal(rho_v, rho) and E.eq(cl_v, cl)
    return dt


@tracked
def test_sumcheck():
    for D in (4, 8):
        for use_eq in (False, True):
            for K, bwf in ((1, False), (3, False), (3, True)):
                dt = run(D, 10, 4, use_eq, K, bwf)
                print("ok", D, use_eq, K, bwf, f"{dt:.2f}s")


@tracked
def bench():
    """Time of an eq * A * B sumcheck over 2^24 entries (D = 8)."""
    D = 8
    n = 24
    A = torch.randint(0, P, (1 << n,), dtype=torch.int32, device="cuda")
    B = torch.randint(0, P, (1 << n,), dtype=torch.int32, device="cuda")
    terms = [(np.ones(D, dtype=np.int64), [0, 1])]
    eq_r = np.random.default_rng(0).integers(0, P, size=(D, n))
    for it in range(2):
        tr = Transcript(b"t", D)
        torch.cuda.synchronize()
        t0 = time.time()
        sc.prove(tr, D, n, [Tab(A, FULL), Tab(B, FULL)], terms, np.zeros(D, dtype=np.int64), eq_r=eq_r)
        torch.cuda.synchronize()
        print(f"n={n} eq*A*B sumcheck: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    test_sumcheck()
    bench()
