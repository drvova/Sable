"""Reed-Solomon code over KoalaBear on multiplicative subgroups (rate 1/2).

A message m of length 2^(n-1) is the coefficient vector of f(X) = sum_j m_j X^j.
Its codeword is f evaluated on the subgroup H_n of order 2^n, in bit-reversed
order: position t holds f(w_n^{bitrev_n(t)}).  Positions (2s, 2s+1) hold
f(x), f(-x) with x = w_n^{bitrev_{n-1}(s)}, so the 2^k symbols that fold to
one symbol after k folds are contiguous.

Folding with challenge r (MLE-compatible, LSB variable first: the folded
message is (1 - r) m_even + r m_odd):
    f0 = (a + b)/2, f1 = (a - b)/(2x), folded = f0 + r (f1 - f0)
The folded codeword is the encoding of the folded message on H_{n-1}.

tw[l][s] = x at level-l position 2s, for l = 0..n-2 (level l has 2^(n-l)
positions and 2^(n-l-1) twiddles); tw[l+1][s] = tw[l][2s]^2.
"""
import numpy as np
from .field import P, Ext

GEN = 3          # generator of F_p^*;  p - 1 = 2^24 * 127
TWO_ADICITY = 24


def root_of_unity(k):
    assert k <= TWO_ADICITY
    return pow(GEN, (P - 1) >> k, P)


def bitrev_perm(bits):
    n = 1 << bits
    idx = np.arange(n, dtype=np.int64)
    r = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        r |= ((idx >> b) & 1) << (bits - 1 - b)
    return r


def _powers(w, N):
    out = np.empty(N, dtype=np.int64)
    out[0] = 1
    filled, step = 1, w
    while filled < N:
        c = min(filled, N - filled)
        out[filled:filled + c] = out[:c] * step % P
        filled += c
        step = step * step % P
    return out


def twiddles(n):
    w = root_of_unity(n)
    t0 = _powers(w, 1 << (n - 1))[bitrev_perm(n - 1)]
    tw = [t0]
    for _ in range(1, n - 1):
        tw.append(tw[-1][0::2] * tw[-1][0::2] % P)
    return tw


def twiddles_dev(n):
    import torch
    N = 1 << (n - 1)
    pw = torch.ones(N, dtype=torch.int64, device="cuda")
    filled, step = 1, root_of_unity(n)
    while filled < N:
        c = min(filled, N - filled)
        pw[filled:filled + c] = pw[:c] * step % P
        filled += c
        step = step * step % P
    idx = torch.arange(N, dtype=torch.int64, device="cuda")
    rev = torch.zeros_like(idx)
    for b in range(n - 1):
        rev |= ((idx >> b) & 1) << (n - 2 - b)
    tw = [pw[rev]]
    for _ in range(1, n - 1):
        tw.append(tw[-1][0::2] * tw[-1][0::2] % P)
    return tw


def domain_points(n):
    return _powers(root_of_unity(n), 1 << n)[bitrev_perm(n)]


def _fft(X, tw, level, L):
    N = 1 << L
    for b in range(L - 2, -1, -1):
        X = X.reshape(X.shape[:-1] + (N >> (b + 1), 2, 1 << b))
        a, c = X[..., 0, :], X[..., 1, :]
        t = tw[level + b].reshape((-1, 1))
        tc = (t * c) % P
        X = np.stack([(a + tc) % P, (a - tc) % P], axis=-2).reshape(X.shape[:-3] + (N,))
    return X


def encode_ref(msg, n, tw=None):
    msg = np.asarray(msg, dtype=np.int64) % P
    assert msg.shape[-1] == 1 << (n - 1)
    if tw is None:
        tw = twiddles(n)
    return _fft(np.concatenate([msg, msg], axis=-1), tw, 0, n)


def encode_line_ref(msg, n, level, tw):
    """Encode msg (length 2^(n-level-1)) on level `level` of the 2^n tables."""
    msg = np.asarray(msg, dtype=np.int64) % P
    L = n - level
    assert msg.shape[-1] == 1 << (L - 1)
    return _fft(np.concatenate([msg, msg], axis=-1), tw, level, L)


def brute_eval(msg, n):
    msg = np.asarray(msg, dtype=np.int64) % P
    x = domain_points(n)
    out = np.zeros(1 << n, dtype=np.int64)
    for c in msg[::-1]:
        out = (out * x + c) % P
    return out


def inv_base(v):
    v = np.asarray(v, dtype=np.int64) % P
    return np.vectorize(lambda a: pow(int(a), P - 2, P), otypes=[np.int64])(v)


INV2 = pow(2, P - 2, P)


def fold_ref(E, vals, tw_level, r):
    a, b = vals[:, 0::2], vals[:, 1::2]
    f0 = E.mul_base(E.add(a, b), INV2)
    f1 = E.mul_base(E.sub(a, b), inv_base(2 * tw_level % P))
    return E.add(f0, E.mul(E.sub(f1, f0), r.reshape(-1, 1)))


def selftest():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 6, 9):
        msg = rng.integers(0, P, size=1 << (n - 1))
        assert np.array_equal(encode_ref(msg, n), brute_eval(msg, n)), n
    E = Ext(8)
    n = 7
    tw = twiddles(n)
    msg = rng.integers(0, P, size=1 << (n - 1))
    cw = E.from_base(encode_ref(msg, n, tw))
    m = E.from_base(msg)
    for lvl in range(5):
        r = rng.integers(0, P, size=8)
        cw = fold_ref(E, cw, tw[lvl], r)
        m = E.add(m[:, 0::2], E.mul(E.sub(m[:, 1::2], m[:, 0::2]), r.reshape(-1, 1)))
        enc = np.stack([encode_line_ref(m[k], n, lvl + 1, tw) for k in range(8)])
        assert np.array_equal(cw, enc), lvl
    return True
