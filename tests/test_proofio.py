"""The proof byte format (zk/proofio.py): a round trip keeps every message (shapes, canonical values,
byte messages), and malformed input (truncated, trailing bytes, a value >= p, a wrong tag or magic, an
oversized shape) raises ValueError."""
import struct
import numpy as np
from zk import proofio
from zk.field import P


def main():
    rng = np.random.default_rng(0)
    msgs = [rng.integers(0, P, (3, 4)), b"\x00\x01cap bytes", rng.integers(0, P, 7), np.zeros((2, 0), np.int64),
            rng.integers(-5, 5, (2, 2, 2)), b""]
    out = proofio.loads(proofio.dumps(msgs))
    assert len(out) == len(msgs)
    for a, b in zip(msgs, out):
        if isinstance(a, bytes):
            assert a == b
        else:
            assert b.shape == np.shape(a) and np.array_equal(b, np.asarray(a) % P)
    print("round trip: ok")
    good = proofio.dumps(msgs)
    bad = {
        "truncated": good[:-3],
        "trailing bytes": good + b"\x00",
        "magic": b"X" + good[1:],
        "tag": good[:12] + b"\x07" + good[13:],
        "value >= p": proofio.MAGIC + struct.pack("<IBBQ", 1, 1, 1, 1) + struct.pack("<I", P),
        "oversized shape": proofio.MAGIC + struct.pack("<IBBQ", 1, 1, 1, 1 << 60),
    }
    for name, b in bad.items():
        try:
            proofio.loads(b)
        except ValueError:
            print(f"{name}: rejected")
            continue
        raise AssertionError(f"{name}: accepted")


def test_proofio():
    main()


if __name__ == "__main__":
    main()
