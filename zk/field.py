"""Reference arithmetic for the KoalaBear field and its tower extensions.

  F_p    p = 2^31 - 2^24 + 1
  F_p2 = F_p[x]/(x^2 - 3)
  F_p4 = F_p2[y]/(y^2 - x)       (= F_p[Y]/(Y^4 - 3))
  F_p8 = F_p4[z]/(z^2 - y)       (= F_p[X]/(X^8 - 3))

Extension elements are numpy int64 arrays whose first axis holds the D base
coordinates (D = 4 or 8), so a vector of n elements has shape (D, n).  The
coordinate order is the tower order: F_p8 = (A, B) with A, B in F_p4,
F_p4 = (a, b) with a, b in F_p2, F_p2 = (a0, a1).  F_p4 embeds in F_p8 as (A, 0).
"""
import numpy as np

P = (1 << 31) - (1 << 24) + 1
NONRES = 3


def mod(x):
    return np.mod(x, P)


# ------------------------------------------------------------------ F_p2
def _f2_mul(a, b):
    a0, a1 = a[0], a[1]
    b0, b1 = b[0], b[1]
    c0 = mod(a0 * b0 + NONRES * mod(a1 * b1))
    c1 = mod(a0 * b1 + a1 * b0)
    return np.stack([c0, c1])


def _f2_mul_x(a):           # multiply by x
    return np.stack([mod(NONRES * a[1]), a[0]])


def _f2_inv(a):
    n = mod(a[0] * a[0] - NONRES * mod(a[1] * a[1]))
    ni = np.vectorize(lambda v: pow(int(v), P - 2, P), otypes=[np.int64])(n)
    return np.stack([mod(a[0] * ni), mod(-a[1] * ni)])


# ------------------------------------------------------------------ F_p4
def _f4_mul(a, b):
    A0, A1 = a[0:2], a[2:4]
    B0, B1 = b[0:2], b[2:4]
    c0 = mod(_f2_mul(A0, B0) + _f2_mul_x(_f2_mul(A1, B1)))
    c1 = mod(_f2_mul(A0, B1) + _f2_mul(A1, B0))
    return np.concatenate([c0, c1])


def _f4_mul_u(a):           # multiply by y
    A0, A1 = a[0:2], a[2:4]
    return np.concatenate([_f2_mul_x(A1), A0])


def _f4_inv(a):
    A0, A1 = a[0:2], a[2:4]
    n = mod(_f2_mul(A0, A0) - _f2_mul_x(_f2_mul(A1, A1)))
    ni = _f2_inv(n)
    return np.concatenate([_f2_mul(A0, ni), mod(-_f2_mul(A1, ni))])


# ------------------------------------------------------------------ F_p8
def _f8_mul(a, b):
    A0, A1 = a[0:4], a[4:8]
    B0, B1 = b[0:4], b[4:8]
    c0 = mod(_f4_mul(A0, B0) + _f4_mul_u(_f4_mul(A1, B1)))
    c1 = mod(_f4_mul(A0, B1) + _f4_mul(A1, B0))
    return np.concatenate([c0, c1])


def _f8_inv(a):
    A0, A1 = a[0:4], a[4:8]
    n = mod(_f4_mul(A0, A0) - _f4_mul_u(_f4_mul(A1, A1)))
    ni = _f4_inv(n)
    return np.concatenate([_f4_mul(A0, ni), mod(-_f4_mul(A1, ni))])


class Ext:
    """Operations on (D, ...) int64 arrays for D in {4, 8}."""

    def __init__(self, D):
        assert D in (4, 8)
        self.D = D
        self._mul = _f4_mul if D == 4 else _f8_mul
        self._inv = _f4_inv if D == 4 else _f8_inv

    def zero(self, *shape):
        return np.zeros((self.D,) + shape, dtype=np.int64)

    def one(self, *shape):
        z = self.zero(*shape)
        z[0] = 1
        return z

    def from_base(self, x):
        x = mod(np.asarray(x, dtype=np.int64))
        z = np.zeros((self.D,) + x.shape, dtype=np.int64)
        z[0] = x
        return z

    def add(self, a, b):
        return mod(a + b)

    def sub(self, a, b):
        return mod(a - b)

    def neg(self, a):
        return mod(-a)

    def mul(self, a, b):
        a, b = np.broadcast_arrays(np.asarray(a, np.int64), np.asarray(b, np.int64))
        return self._mul(a, b)

    def mul_base(self, a, s):
        return mod(a * mod(np.asarray(s, dtype=np.int64)))

    def inv(self, a):
        return self._inv(np.asarray(a, np.int64))

    def pow(self, a, e):
        r = self.one(*a.shape[1:])
        b = a.copy()
        while e:
            if e & 1:
                r = self.mul(r, b)
            b = self.mul(b, b)
            e >>= 1
        return r

    def eq(self, a, b):
        return np.array_equal(mod(a), mod(b))

    # Multilinear helpers -------------------------------------------------
    def eq_table(self, r):
        """eq(r, x) for x in {0,1}^n, x_1 = bit 0 (LSB).  r: (D, n)."""
        n = r.shape[1]
        t = self.one(1)
        for k in range(n):
            rk = r[:, k:k + 1]
            one_m = self.sub(self.one(1), rk)
            # new index = old index + 2^k * bit
            t = np.concatenate([self.mul(t, one_m), self.mul(t, rk)], axis=1)
        return t

    def eq_eval(self, a, b):
        """eq(a, b) = prod_k (a_k b_k + (1 - a_k)(1 - b_k)).  a, b: (D, n)."""
        r = self.one()
        for k in range(a.shape[1]):
            ak, bk = a[:, k], b[:, k]
            t = self.add(self.mul(ak, bk),
                         self.mul(self.sub(self.one(), ak), self.sub(self.one(), bk)))
            r = self.mul(r, t)
        return r

    def mle_eval(self, vals, r):
        """MLE of vals (base (n,) or ext (D, n)) at r (D, k), LSB first."""
        vals = np.asarray(vals, dtype=np.int64)
        if vals.ndim == 1:
            vals = self.from_base(vals)
        k = r.shape[1]
        assert vals.shape[1] == 1 << k, (vals.shape, k)
        cur = vals
        for j in range(k):
            lo, hi = cur[:, 0::2], cur[:, 1::2]
            cur = self.add(lo, self.mul(self.sub(hi, lo), r[:, j:j + 1]))
        return cur[:, 0]


def embed(a, D=8):
    """F_p4 element(s) (4, ...) -> F_p8 (8, ...) (zero upper half)."""
    a = np.asarray(a, dtype=np.int64)
    if a.shape[0] == D:
        return a
    return np.concatenate([a, np.zeros_like(a)], axis=0)


def check_tower():
    """Irreducibility facts behind the tower (spec section 'Fields')."""
    assert P % 4 == 1 and P - 1 == (1 << 24) * 127
    # 3 is a non-square, so X^(2^k) - 3 is irreducible for every k (p = 1 mod 4)
    assert pow(NONRES, (P - 1) // 2, P) == P - 1
    e = Ext(8)
    rng = np.random.default_rng(1)
    for _ in range(4):
        a = rng.integers(0, P, size=(8,), dtype=np.int64)
        ai = e.inv(a)
        assert e.eq(e.mul(a, ai), e.one())
    # z^8 = 3
    z = np.zeros(8, dtype=np.int64)
    z[4] = 1
    assert e.eq(e.pow(z, 8), e.from_base(3))
    return True


def to_signed(x):
    x = np.asarray(x, dtype=np.int64) % P
    return np.where(x > P // 2, x - P, x)
