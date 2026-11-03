"""Cost of the BLAKE3 circuit of the recursive verifier (zk/recur/b3dom.py) per block of N
compressions: commitment of the trace (encoding and Merkle hashing), domain proofs (wired G rows,
input rows, output rows, table), and the openings of the trace units at the domains' points (the
global evaluation proof of a recursion layer is shared by all blocks and not included).

usage: python -m scripts.b3bench [log2 N] [reps]     (default 2^16 compressions = 2^22 G rows)"""
import sys
import time
import numpy as np
import torch
from zk.recur import b3ref as R, b3dom as B
from zk import domain as dm, gpcs
from zk.gkr import NBETA
from zk.transcript import Transcript

D, NT = 4, 18


def main():
    logN = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    N = 1 << logN
    rng = np.random.default_rng(0)
    cv, blk = rng.integers(0, 1 << 32, (N, 8)), rng.integers(0, 1 << 32, (N, 16))
    trace = []
    out = R.compress(cv, blk, 7, 64, 3, trace)
    s8 = out[:, 8:] ^ cv
    final = np.concatenate([out[:, :8] ^ s8, s8], axis=1)
    WG = B.g_witness_wired(trace)
    WI = B.in_witness(B.in_words(cv, blk, 7, 64, 3))
    WO = B.out_witness(final)
    mult = B.table_mult(B.exprs(WG, False)[1] + B.in_lookups(WI) + B.out_lookups(WO), NT)
    dev = lambda a: torch.from_numpy((np.asarray(a) % B.P).astype(np.int32)).cuda()
    units = {"g": {c: dev(WG[c]) for c in B.COLS}, "in": {k: dev(v) for k, v in WI.items()},
             "out": {k: dev(v) for k, v in WO.items()}, "tm": {"m": dev(mult)}}
    sizes = {"g": B.LOG_GPC + logN, "in": B.LOG_IN + logN, "out": B.LOG_OUT + logN, "tm": NT}
    cells = sum(len(t) << sizes[u] for u, t in units.items())
    print(f"{N} compressions: {cells} committed cells ({cells / N:.0f} per compression)", flush=True)
    for rep in range(reps):
        tr = Transcript(b"b3bench", D)
        pcs = gpcs.GPCS(tr, True)
        for u, t in units.items():
            pcs.declare(u, [(k, 1, sizes[u]) for k in t])
        # the domains' columns are uploaded before timing (in a recursion layer the witness is
        # generated on the device)
        doms = [("g", B.g_domain_wired(N, WG)), ("in", B.in_domain(N, WI)), ("out", B.out_domain(N, WO)),
                (None, B.expect_domain(out[:, :8])), ("tm", B.table_domain(NT, mult)[0])]
        torch.cuda.synchronize()
        t0 = time.time()
        for u, t in units.items():
            pcs.commit(u, t)
        torch.cuda.synchronize()
        t1 = time.time()
        Z, beta = tr.challenge(), tr.challenge(NBETA)
        claims = {u: [] for u in units}
        for u, d in doms:
            cl = dm.prove(tr, d, Z, beta)
            if u is None:
                continue
            for name, (pt, val) in cl.items():
                if name in units[u]:
                    claims[u].append((name, pt, {0: np.eye(D, dtype=np.int64)[0]}, val[:, 0]))
        torch.cuda.synchronize()
        t2 = time.time()
        for u, t in units.items():
            pcs.open(u, claims[u], t)
        torch.cuda.synchronize()
        t3 = time.time()
        tot = t3 - t0
        print(f"rep {rep}: commit {t1 - t0:.3f}s, domains {t2 - t1:.3f}s, openings {t3 - t2:.3f}s, "
              f"total {tot:.3f}s = {tot / N * 1e6:.2f} us per compression", flush=True)


if __name__ == "__main__":
    main()
