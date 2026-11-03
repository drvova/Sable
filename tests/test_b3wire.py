"""BLAKE3 wiring (zk/recur/b3dom.py): G rows, IN rows and OUT rows of several compressions joined by
the LogUp memory prove and verify against public expected outputs; a message word that differs
from the one the G rows read, and a wrong expected output, unbalance the sets."""
import numpy as np
from zk.gkr import NBETA
from zk.recur import b3ref as R, b3dom as B
from zk import domain as dm
from zk.field import P, Ext
from zk.fe import fe
from zk.transcript import Transcript

D, NT, N = 4, 18, 8


def run(words, final, out_exp, trace):
    WG, WI, WO = B.g_witness_wired(trace), B.in_witness(words), B.out_witness(final)
    mult = B.table_mult(B.exprs(WG, False)[1] + B.in_lookups(WI) + B.out_lookups(WO), NT)

    def doms(on):
        return [B.g_domain_wired(N, WG if on else None), B.in_domain(N, WI if on else None),
                B.out_domain(N, WO if on else None), B.expect_domain(out_exp, on),
                B.table_domain(NT, mult if on else None)[0]]
    tr = Transcript(b"b3w", D)
    Z, beta = tr.challenge(), tr.challenge(NBETA)
    pd = doms(True)
    for d in pd:
        dm.prove(tr, d, Z, beta)
    tv = Transcript(b"b3w", D, msgs=tr.msgs)
    Zv, bv = tv.challenge(), tv.challenge(NBETA)
    roots, claims = [], []
    try:
        for d in doms(False):
            c, r = dm.verify(tv, d, Zv, bv)
            roots += r
            claims.append(c)
    except ValueError:
        return False
    E = Ext(D)          # committed columns stand in for PCS openings; public ones are evaluated
    for d, c in zip(pd, claims):
        for col in d.cols:
            pt, v = c[col.name]
            assert E.eq(v[:, 0], E.mle_eval(col.data.cpu().numpy().astype(np.int64) % P, pt))
    return dm.frac_sum(fe(D), roots) == fe(D).zero


def test_b3_wiring():
    rng = np.random.default_rng(1)
    cv, blk = rng.integers(0, 1 << 32, (N, 8)), rng.integers(0, 1 << 32, (N, 16))
    trace = []
    out = R.compress(cv, blk, 7, 64, 3, trace)
    s8 = out[:, 8:] ^ cv
    final = np.concatenate([out[:, :8] ^ s8, s8], axis=1)
    words = B.in_words(cv, blk, 7, 64, 3)
    assert run(words, final, out[:, :8], trace)
    print("wired compressions: proof verifies")
    wb = words.copy()
    wb[1, 16 + 3] ^= 1 << 9
    assert not run(wb, final, out[:, :8], trace)
    print("message word not the one read: rejected")
    eb = out[:, :8].copy()
    eb[2, 5] ^= 1
    assert not run(words, final, eb, trace)
    print("wrong expected output: rejected")


if __name__ == "__main__":
    test_b3_wiring()
