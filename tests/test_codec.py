import time
import numpy as np
import torch
import blake3
from zk import rs, codec, field
from zk.gpu import module
from zk.field import P


def to_t(a):
    return torch.from_numpy(np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32)).cuda()


def from_t(t):
    return t.cpu().numpy().view(np.uint32).astype(np.int64)


def test_field():
    for D in (4, 8):
        E = field.Ext(D)
        rng = np.random.default_rng(D)
        n = 1000
        a = rng.integers(0, P, size=(D, n))
        b = rng.integers(0, P, size=(D, n))
        c = torch.empty((D, n), dtype=torch.int32, device="cuda")
        module(D).ext_mul(to_t(a), to_t(b), c)
        assert np.array_equal(from_t(c), E.mul(a, b)), D
    print("field ok")


def test_encode():
    rng = np.random.default_rng(0)
    for n in (5, 8, 9, 10, 11, 14, 20):
        msg = rng.integers(0, P, size=(3, 1 << (n - 1)))
        cw = codec.encode(to_t(msg), n)
        ref = rs.encode_ref(msg, n)
        assert np.array_equal(from_t(cw), ref), n
    # line-level encode
    n = 12
    tw = rs.twiddles(n)
    for level in (1, 4):
        msg = rng.integers(0, P, size=(2, 1 << (n - level - 1)))
        cw = codec.encode(to_t(msg), n, level=level)
        ref = np.stack([rs.encode_line_ref(m, n, level, tw) for m in msg])
        assert np.array_equal(from_t(cw), ref), level
    print("encode ok")


def test_merkle():
    rng = np.random.default_rng(1)
    N = 1 << 10
    cw = rng.integers(0, P, size=(16, N))
    lh = codec.leaf_hashes(to_t(cw))
    leaf0 = b"".join(cw[k, 0:16].astype("<u4").tobytes() for k in range(16))
    assert codec.digest_bytes(lh[0]) == blake3.blake3(leaf0).digest()
    lv = codec.merkle_levels(lh)
    h0 = codec.digest_bytes(lv[0][0]) + codec.digest_bytes(lv[0][1])
    assert codec.digest_bytes(lv[1][0]) == blake3.blake3(h0).digest()
    print("merkle ok")


def bench_encode(n=24, ncols=4):
    msg = torch.randint(0, P, (ncols, 1 << (n - 1)), dtype=torch.int32, device="cuda")
    out = torch.empty((ncols, 1 << n), dtype=torch.int32, device="cuda")
    codec.encode(msg, n, out=out)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        codec.encode(msg, n, out=out)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 3 / ncols
    print(f"encode 2^{n - 1}->2^{n}: {dt * 1e3:.2f} ms/column")
    lh = codec.leaf_hashes(out[:16])
    torch.cuda.synchronize()
    t = time.time()
    lh = codec.leaf_hashes(out[:16])
    codec.merkle_levels(lh, keep=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"hash {min(16, ncols)} cols x 2^{n}: {dt * 1e3:.2f} ms "
          f"({min(16, ncols) * (4 << n) / dt / 1e9:.0f} GB/s)")


if __name__ == "__main__":
    test_field()
    test_encode()
    test_merkle()
    bench_encode()
