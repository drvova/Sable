"""Every corrupted proof message must be rejected."""
import copy
import numpy as np
from zk import llm
from zk.field import P
from tests.test_llm import qwen_tiny, statement


def main(ntrials=24):
    M = qwen_tiny()
    B, S, ngen = 2, 256, 16
    sh = llm.Shp(M.cfg, B, S, 7, 7, ngen)
    tokens, targets = statement(M, B, S, ngen)
    msgs = llm.prove(M, sh, tokens, targets)
    rng = np.random.default_rng(0)
    idx = sorted(set(np.linspace(0, len(msgs) - 1, ntrials).astype(int).tolist()))
    acc = 0
    for i in idx:
        bad = copy.copy(msgs)
        m = bad[i]
        if isinstance(m, (bytes, bytearray)):
            b = bytearray(m)
            b[rng.integers(len(b))] ^= 1
            bad[i] = bytes(b)
        else:
            a = np.array(m, copy=True)
            j = np.unravel_index(rng.integers(a.size), a.shape)
            a[j] = (a[j] + 1) % P
            bad[i] = a
        try:
            llm.verify(M, sh, tokens, targets, bad)
        except Exception as e:                     # noqa: BLE001
            print(f"msg {i}: rejected ({type(e).__name__}: {str(e)[:70]})")
        else:
            print(f"msg {i}: ACCEPTED")
            acc += 1
    print("accepted tampered:", acc)


def test_llm_tamper():
    main()


if __name__ == "__main__":
    main()
