"""BLAKE3 G-row gadget (zk/recur/b3dom.py): reference compression, witness constraints, an honest
proof that verifies with its table, and a forged XOR byte that is rejected."""
import numpy as np
import blake3
from zk.gkr import NBETA
from zk.recur import b3ref as R, b3dom as B
from zk import domain as dm
from zk.field import P, Ext
from zk.fe import fe
from zk.transcript import Transcript

D, NT = 4, 18


def prove_verify(W, n, mult):
    tr = Transcript(b"b3", D)
    Z, beta = tr.challenge(), tr.challenge(NBETA)
    cl = dm.prove(tr, B.g_domain(n, W), Z, beta)
    dm.prove(tr, B.table_domain(NT, mult)[0], Z, beta)
    tv = Transcript(b"b3", D, msgs=tr.msgs)
    Zv, bv = tv.challenge(), tv.challenge(NBETA)
    _, r1 = dm.verify(tv, B.g_domain(n, None), Zv, bv)
    _, r2 = dm.verify(tv, B.table_domain(NT, None)[0], Zv, bv)
    return cl, dm.frac_sum(fe(D), r1 + r2) == fe(D).zero


def test_b3_gadget():
    rng = np.random.default_rng(0)
    msgs = [rng.integers(0, 256, 64, dtype=np.uint8).tobytes() for _ in range(16)]
    out = R.compress(np.tile(R.IV, (16, 1)), np.stack([R.words_le(m) for m in msgs]), 0, 64, 11)
    assert all(out[k, :8].astype("<u4").tobytes() == blake3.blake3(msgs[k]).digest() for k in range(16))
    print("reference compression matches blake3")
    N = 64
    trace = []
    R.compress(rng.integers(0, 1 << 32, (N, 8)), rng.integers(0, 1 << 32, (N, 16)), 5, 64, 3, trace)
    n = (N * 56 - 1).bit_length()
    W = B.witness(trace, n)
    cons, _ = B.exprs(W)
    assert all(np.all(c == 0) for c in cons)
    mult = B.multiplicities(W, NT)
    cl, ok = prove_verify(W, n, mult)
    E = Ext(D)
    assert ok and all(E.eq(v[:, 0], E.mle_eval(W[k] % P, pt)) for k, (pt, v) in cl.items())
    print("honest G rows: proof verifies")
    Wb = {k: v.copy() for k, v in W.items()}
    Wb["x1_0"][5] ^= 1                 # forged XOR byte; the add that consumes it is repaired
    carry = 0
    for j in range(4):
        tot = Wb[f"c{j}"][5] + Wb[f"x1_{(j + 2) % 4}"][5] + carry
        Wb[f"s2_{j}"][5], carry = tot & 255, tot >> 8
        Wb[f"k2_{j + 1}"][5] = carry
    carry = 0                          # and s4 = s2 + d2
    for j in range(4):
        tot = Wb[f"s2_{j}"][5] + Wb[f"x3_{(j + 1) % 4}"][5] + carry
        Wb[f"s4_{j}"][5], carry = tot & 255, tot >> 8
        Wb[f"k4_{j + 1}"][5] = carry
    assert all(np.all(c == 0) for c in B.exprs(Wb)[0])
    try:
        _, ok = prove_verify(Wb, n, mult)
    except ValueError:
        ok = False
    assert not ok
    print("forged XOR byte: rejected")


if __name__ == "__main__":
    test_b3_gadget()
