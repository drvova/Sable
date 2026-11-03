"""The transcript hash (zk/cuda/b3s.h) against a direct specification of its sequential mode over
the reference BLAKE3 compression of the recursion circuit (zk/recur/b3ref.py)."""
import numpy as np
from zk.b3s import Hasher
from zk.recur.b3ref import compress, IV, CHUNK_END, ROOT


def spec(data, nout):
    """cv <- compress(cv, block_i, i, 64, 0)[:8] for every block but the last (possibly empty or
    full) one; output block j = compress(cv, last block, j, its length, CHUNK_END | ROOT)."""
    nb = max(1, (len(data) + 63) // 64)
    blocks = [data[64 * i:64 * i + 64] for i in range(nb)]
    cv = IV.reshape(1, 8).copy()
    for i, b in enumerate(blocks[:-1]):
        cv = compress(cv, np.frombuffer(b, dtype="<u4").astype(np.int64).reshape(1, 16), i, 64, 0)[:, :8]
    last = blocks[-1]
    w = np.frombuffer(last + bytes(64 - len(last)), dtype="<u4").astype(np.int64).reshape(1, 16)
    out = b""
    for j in range((nout + 63) // 64):
        o = compress(cv, w, j, len(last), CHUNK_END | ROOT)[0]
        out += o.astype("<u4").tobytes()
    return out[:nout]


def main():
    rng = np.random.default_rng(0)
    for n in (0, 1, 7, 63, 64, 65, 127, 128, 129, 1000, 1024, 1025, 5000):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        for nout in (8, 32, 64, 100, 200):
            h = Hasher()
            # split the input into random pieces: the state must not depend on the split
            cut = sorted(rng.integers(0, n + 1, 3).tolist())
            for a, b in zip([0] + cut, cut + [n]):
                h.update(data[a:b])
            assert h.digest(nout) == spec(data, nout), (n, nout)
    print("transcript mode ok")


def test_transcript_mode():
    main()


if __name__ == "__main__":
    main()
