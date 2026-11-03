import sys
import time
import numpy as np
import torch
from zk.field import P, Ext
from zk.transcript import Transcript
from zk import gpcs
from zk.fe import fe

UNITS = {
    "u0": [("a", 3, 9), ("b", 1, 12), ("c", 2, 3)],
    "u1": [("d", 20, 7), ("e", 2, 10)],
    "u2": [("f", 1, 5)],
}


def mk_claims(rng, E, data, items, n=2):
    F = fe(4)
    out = []
    for name, K, c in items:
        for _ in range(n):
            pt = rng.integers(0, P, size=(4, c))
            vals = [E.mle_eval(data[name][b], pt) for b in range(K)]
            if K > 1 and rng.random() < 0.5:
                w = {b: rng.integers(0, P, size=4) for b in range(K)}
                v = F.zero
                for b in range(K):
                    v = F.add(v, F.mul(F.c(w[b]), F.c(vals[b])))
                out.append((name, pt, w, np.array(v)))
            else:
                for b in range(K):
                    out.append((name, pt, {b: np.array([1, 0, 0, 0])}, vals[b]))
    return out


def run(tamper=None, seed=0, cmax=10, Q=20, grind=4):
    rng = np.random.default_rng(seed)
    E = Ext(4)
    data = {name: rng.integers(0, P, size=(K, 1 << c)) for it in UNITS.values() for name, K, c in it}
    tens = {k: torch.from_numpy(v.astype(np.int32)).cuda() for k, v in data.items()}

    commit_tens = tens
    if tamper == "data":
        data = {k: v.copy() for k, v in data.items()}
        data["d"][3, 5] = (data["d"][3, 5] + 1) % P
        tens = {k: torch.from_numpy(v.astype(np.int32)).cuda() for k, v in data.items()}
    calls = [0]

    def provider():
        calls[0] += 1
        src = commit_tens if calls[0] == 3 else tens
        for u, it in UNITS.items():
            yield u, {n: src[n] for n, _, _ in it}
    claims = {u: mk_claims(rng, E, data, it) for u, it in UNITS.items()}
    if tamper == "claim":
        cl = claims["u1"][0]
        v = cl[3].copy()
        v[0] = (v[0] + 1) % P
        claims["u1"][0] = (cl[0], cl[1], cl[2], v)
    t0 = time.time()
    tr = Transcript(b"gpcs", 4)
    pc = gpcs.GPCS(tr, True, cmax=cmax, queries=Q, grind=grind)
    for u, it in UNITS.items():
        pc.declare(u, it)
    for u, it in UNITS.items():
        pc.commit(u, {n: commit_tens[n] for n, _, _ in it})
    for u in UNITS:
        pc.open(u, claims[u], {n: tens[n] for n, _, _ in UNITS[u]})
    pc.finish(provider)
    dt = time.time() - t0
    msgs = list(tr.msgs)
    if tamper == "leaf":
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], np.ndarray) and msgs[i].ndim == 3 and msgs[i].shape[2] == 4:
                m = msgs[i].copy()
                m[0, 0, 0] = (m[0, 0, 0] + 1) % P
                msgs[i] = m
                break
    if tamper == "cap":
        b = bytearray(msgs[0])
        b[0] ^= 1
        msgs[0] = bytes(b)
    tv = Transcript(b"gpcs", 4, msgs=msgs)
    pv = gpcs.GPCS(tv, False, cmax=cmax, queries=Q, grind=grind)
    for u, it in UNITS.items():
        pv.declare(u, it)
    try:
        for u in UNITS:
            pv.commit(u)
        for u in UNITS:
            pv.open(u, claims[u])
        pv.finish()
        return True, dt, tr.sent_bytes
    except ValueError as e:
        print("  rejected:", e)
        return False, dt, tr.sent_bytes


def bench(c=23, ncols=64, Q=153, grind=30):
    """One unit of ncols columns of 2^c: time per pass."""
    rng = np.random.default_rng(1)
    tens = {"x": torch.randint(0, P, (ncols << c,), dtype=torch.int32, device="cuda")}
    items = [("x", ncols, c)]
    tr = Transcript(b"gpcs", 4)
    pc = gpcs.GPCS(tr, True, queries=Q, grind=grind)
    pc.declare("u", items)
    torch.cuda.synchronize()
    t0 = time.time()
    pc.commit("u", tens)
    torch.cuda.synchronize()
    t1 = time.time()
    pt = rng.integers(0, P, size=(4, c))
    xs = tens["x"].reshape(ncols, -1)
    G = gpcs.gops(4)
    e = G.eq_table(pt)
    cl = [("x", pt, {b: np.array([1, 0, 0, 0])}, G.dot(xs[b].contiguous(), e)) for b in range(ncols)]
    pc.open("u", cl, tens)
    torch.cuda.synchronize()
    t2 = time.time()
    pc.finish(lambda: iter([("u", tens)]))
    torch.cuda.synchronize()
    t3 = time.time()
    print(f"bench {ncols}x2^{c}: commit {t1 - t0:.2f}s open {t2 - t1:.2f}s finish {t3 - t2:.2f}s "
          f"proof {tr.sent_bytes / 2**20:.1f} MiB")


def test_gpcs():
    ok, dt, nb = run()
    print("honest", ok, f"{dt:.2f}s", f"{nb / 1e6:.2f} MB")
    assert ok
    for t in ("claim", "leaf", "cap", "data"):
        ok, _, _ = run(tamper=t)
        print("tamper", t, ok)
        assert not ok
    ok, dt, nb = run(cmax=23, seed=3)
    print("honest cmax=23", ok)
    assert ok
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench()
    else:
        test_gpcs()
