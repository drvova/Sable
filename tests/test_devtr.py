"""The device transcript (DevTr, b3w.cuh) matches the host Transcript byte for byte."""
import numpy as np
import torch
from zk.field import P
from zk.transcript import Transcript
from zk.devtr import DevTr


def dev(a):
    return torch.as_tensor(np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32), device="cuda")


def main(steps=400, seed=0, D=4):
    rng = np.random.default_rng(seed)
    H = Transcript(b"devtr-test", D)
    T = Transcript(b"devtr-test", D)
    dt = DevTr(T)
    checks = []
    for _ in range(steps):
        op = rng.integers(5)
        if op == 0:                                    # send of any length (crosses chunk ends)
            a = rng.integers(0, P, size=int(rng.integers(1, 400)))
            H.send(a)
            dt.send(dev(a))
        elif op == 1:                                  # base challenges, k <= 250
            k = int(rng.integers(1, 250))
            checks.append((H.challenge_base(k), dt.challenge(None, D=k)))
        elif op == 2:                                  # sumcheck round: npts x D sums, then a D-challenge
            npts = int(rng.integers(1, 9))
            s = rng.integers(0, P, size=npts * D)
            H.send(s.reshape(npts, D))
            checks.append((H.challenge(), dt.round(dev(s).data_ptr(), npts)))
        elif op == 3:                                  # columns of a (D, n) table, one send each
            n = int(rng.integers(1, 40))
            tab = rng.integers(0, P, size=(D, n))
            cols = sorted(rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False).tolist())
            for c in cols:
                H.send(tab[:, c])
            dt.send_cols(dev(tab), cols)
        else:                                          # D-challenges in rows
            k = int(rng.integers(1, 250 // D))
            checks.append((H.challenge(k).T.reshape(-1), dt.challenge(k).reshape(-1)))
    dt.finish()
    for i, (h, d) in enumerate(checks):
        d = d.cpu().numpy().view(np.uint32).astype(np.int64).reshape(-1)
        assert np.array_equal(np.asarray(h).reshape(-1), d), f"challenge {i} differs"
    assert H.h.copy().digest() == T.h.copy().digest(), "final state differs"
    assert len(H.msgs) == len(T.msgs) and all(np.array_equal(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))
                                              for a, b in zip(H.msgs, T.msgs)), "sent messages differ"
    print(f"devtr ok: {len(checks)} challenge draws, {len(H.msgs)} messages, "
          f"{H.n_absorbed} bytes absorbed")


def sumcheck_rounds(D, n):
    """The transcript step fused into the sumcheck round kernels (any extension degree)."""
    from zk import sumcheck as sc
    from zk.sumcheck import Tab, FULL
    rng = np.random.default_rng(n)
    A, B = dev(rng.integers(0, P, 1 << n)), dev(rng.integers(0, P, 1 << n))
    r = rng.integers(0, P, (D, n))
    T = Transcript(b"sc-rounds", D)
    rho, _, _ = sc.prove(T, D, n, [Tab(A, FULL), Tab(B, FULL)], [(np.eye(D, dtype=np.int64)[0], [0, 1])],
                         np.zeros(D, dtype=np.int64), eq_r=r)
    H = Transcript(b"sc-rounds", D)
    H.absorb(b"sc|" + n.to_bytes(2, "little") + (2).to_bytes(2, "little") + (1).to_bytes(4, "little"))
    for j, m in enumerate(T.msgs[:n]):
        H.send(np.asarray(m))
        assert np.array_equal(H.challenge(), rho[:, j]), f"D={D} n={n}: round {j} challenge differs"
    assert H.h.copy().digest() == T.h.copy().digest(), f"D={D} n={n}: final state differs"


def session(D=4, steps=300, seed=1):
    """Transcript.begin_device(): every operation queued on the device, challenges lazy."""
    rng = np.random.default_rng(seed)
    H = Transcript(b"session", D)
    T = Transcript(b"session", D)
    T.begin_device()
    checks = []
    for _ in range(steps):
        op = rng.integers(6)
        if op == 0:
            b = bytes(rng.integers(0, 256, int(rng.integers(0, 200)), dtype=np.uint8))
            H.absorb(b)
            T.absorb(b)
        elif op == 1:
            a = rng.integers(0, P, size=(int(rng.integers(1, 9)), D))
            H.send(a)
            T.send(a)
        elif op == 2:
            b = bytes(rng.integers(0, 256, int(rng.integers(1, 300)), dtype=np.uint8))
            H.send_bytes(b)
            T.send_bytes(b)
        elif op == 3:
            checks.append((H.challenge(), T.challenge()))
        elif op == 4:
            k = int(rng.integers(1, 40))
            checks.append((H.challenge(k), T.challenge(k)))
        else:
            k = int(rng.integers(1, 60))
            c = T.challenge_base(k)
            h = H.challenge_base(k)
            checks.append((h, c))
            H.send(h)
            T.send(c)                                      # a lazy (device) array sent back
    for i, (h, d) in enumerate(checks):
        assert np.array_equal(np.asarray(h), np.asarray(d)), f"session challenge {i} differs"
    bits = (H.challenge_bits(20, 7), T.challenge_bits(20, 7))      # ends the session
    assert T.dev is None and np.array_equal(*bits)
    assert H.h.copy().digest() == T.h.copy().digest(), "session state differs"
    assert len(H.msgs) == len(T.msgs)
    for a, b in zip(H.msgs, T.msgs):
        if isinstance(a, (bytes, bytearray)):
            assert a == b
        else:
            assert np.array_equal(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))
    print(f"device session ok (D={D}): {len(checks)} challenge draws, {len(H.msgs)} messages")


def test_device_transcript():
    for D in (4, 8):
        session(D)
    for D in (4, 8):
        main(D=D)
        for n in (1, 3, 12, 20):
            sumcheck_rounds(D, n)
        print(f"fused sumcheck transcript ok (D={D})")


if __name__ == "__main__":
    test_device_transcript()
