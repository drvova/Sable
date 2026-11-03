"""GKR layer sumcheck uses the specialized per-lookup kernel."""
import numpy as np
import torch
from zk.field import P
from zk.ext import gops
from zk.transcript import Transcript
from zk import gkr
from zk.sumcheck import FULL, _gkr_lookups


def test_gkr_lookups():
    L = 5
    kinds = [FULL] * (4 * L)
    terms = [(None, [4 * l, 4 * l + 3]) for l in range(L)]
    terms += []  # keep pattern via list below
    terms = []
    for l in range(L):
        terms += [(None, [4 * l, 4 * l + 3]), (None, [4 * l + 1, 4 * l + 2]),
                  (None, [4 * l + 2, 4 * l + 3])]
    assert _gkr_lookups(terms, kinds) == L
    bw = 4 * L
    kinds3 = kinds + [FULL]
    terms3 = [(None, list(r) + [bw]) for _, r in terms]
    assert _gkr_lookups(terms3, kinds3) == L
    assert _gkr_lookups(terms, [FULL] * 3) == 0


def test_gkr_upper(D=4, n=6, K=5, L=5):
    G = gops(D)
    total = K << (n - 1)
    rng = np.random.default_rng(0)
    top = []
    for _ in range(L):
        p = torch.as_tensor(rng.integers(0, P, size=(D, total), dtype=np.int32), device="cuda")
        q = torch.as_tensor(rng.integers(0, P, size=(D, total), dtype=np.int32), device="cuda")
        top.append((p, q))
    layers, tt = gkr.build(G, top, K, n)
    tr = Transcript(b"gkr", D)
    ri, rb, cp, cq = gkr.prove_upper(tr, G, n, K, layers, tt, L)
    tv = Transcript(b"gkr", D, msgs=tr.msgs)
    roots, ri2, rb2, cp2, cq2 = gkr.verify_upper(tv, G.E, n, K, L)
    assert np.array_equal(ri, ri2) and np.array_equal(rb, rb2)
    for l in range(L):
        assert np.array_equal(cp[l], cp2[l]) and np.array_equal(cq[l], cq2[l])
    print("ok gkr_upper", D, n, K, L)


if __name__ == "__main__":
    test_gkr_lookups()
    test_gkr_upper()
    test_gkr_upper(4, 8, 9, 5)
    print("ok")
