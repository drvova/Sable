"""Scalar extension-field arithmetic on Python ints (fast for single values).

Elements are tuples of D ints in the tower order used by field.py."""
P = (1 << 31) - (1 << 24) + 1


def _f2m(a0, a1, b0, b1):
    return ((a0 * b0 + 3 * a1 * b1) % P, (a0 * b1 + a1 * b0) % P)


def _f4m(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    x0, x1 = _f2m(a0, a1, b0, b1)
    y0, y1 = _f2m(a2, a3, b2, b3)
    m0, m1 = _f2m(a0 + a2, a1 + a3, b0 + b2, b1 + b3)
    # y * x = 3 y1 + y0 x
    return ((x0 + 3 * y1) % P, (x1 + y0) % P, (m0 - x0 - y0) % P, (m1 - x1 - y1) % P)


def _f4mu(a):           # times y
    a0, a1, a2, a3 = a
    return ((3 * a3) % P, a2, a0, a1)


def _f4a(a, b):
    return tuple((x + y) % P for x, y in zip(a, b))


def _f4s(a, b):
    return tuple((x - y) % P for x, y in zip(a, b))


def _f8m(a, b):
    A0, A1 = a[:4], a[4:]
    B0, B1 = b[:4], b[4:]
    x = _f4m(A0, B0)
    y = _f4m(A1, B1)
    m = _f4m(_f4a(A0, A1), _f4a(B0, B1))
    c0 = _f4a(x, _f4mu(y))
    c1 = _f4s(_f4s(m, x), y)
    return c0 + c1


def _inv_base(v):
    return pow(v % P, P - 2, P)


def _f2inv(a0, a1):
    n = _inv_base(a0 * a0 - 3 * a1 * a1)
    return (a0 * n % P, (-a1) * n % P)


def _f4inv(a):
    A0, A1 = a[:2], a[2:]
    s0 = _f2m(A0[0], A0[1], A0[0], A0[1])
    t = _f2m(A1[0], A1[1], A1[0], A1[1])
    tx = (3 * t[1], t[0])
    n0, n1 = (s0[0] - tx[0]) % P, (s0[1] - tx[1]) % P
    i0, i1 = _f2inv(n0, n1)
    c0 = _f2m(A0[0], A0[1], i0, i1)
    c1 = _f2m(A1[0], A1[1], i0, i1)
    return (c0[0], c0[1], (-c1[0]) % P, (-c1[1]) % P)


def _f8inv(a):
    A0, A1 = a[:4], a[4:]
    n = _f4s(_f4m(A0, A0), _f4mu(_f4m(A1, A1)))
    ni = _f4inv(n)
    c0 = _f4m(A0, ni)
    c1 = _f4m(A1, ni)
    return c0 + tuple((-v) % P for v in c1)


class FE:
    def __init__(self, D):
        self.D = D
        self.mul = _f8m if D == 8 else _f4m
        self.inv = _f8inv if D == 8 else _f4inv
        self.zero = (0,) * D
        self.one = (1,) + (0,) * (D - 1)

    def c(self, a):
        """numpy/list -> tuple"""
        return tuple(int(v) % P for v in a)

    def base(self, v):
        return (int(v) % P,) + (0,) * (self.D - 1)

    def add(self, a, b):
        return tuple((x + y) % P for x, y in zip(a, b))

    def sub(self, a, b):
        return tuple((x - y) % P for x, y in zip(a, b))

    def neg(self, a):
        return tuple((-x) % P for x in a)

    def smul(self, a, s):
        s = int(s) % P
        return tuple(x * s % P for x in a)

    def eq1(self, a, b):
        ab = self.mul(a, b)
        # ab + (1-a)(1-b) = 2ab - a - b + 1
        r = tuple((2 * x - y - z) % P for x, y, z in zip(ab, a, b))
        return (r[0] + 1) % P, *r[1:]

    def eq(self, a_list, b_list):
        r = self.one
        for a, b in zip(a_list, b_list):
            r = self.mul(r, self.eq1(a, b))
        return r

    def lagrange(self, vals, x):
        """vals at 0..d -> value at x."""
        d = len(vals) - 1
        if d not in _LW:
            w = []
            for i in range(d + 1):
                den = 1
                for j in range(d + 1):
                    if j != i:
                        den = den * (i - j) % P
                w.append(_inv_base(den))
            _LW[d] = w
        w = _LW[d]
        mul = self.mul
        xm = [((x[0] - j) % P,) + tuple(x[1:]) for j in range(d + 1)]
        pre = [self.one]
        for j in range(d):
            pre.append(mul(pre[-1], xm[j]))
        res = self.zero
        suf = self.one
        for i in range(d, -1, -1):
            res = self.add(res, self.smul(mul(mul(pre[i], suf), vals[i]), w[i]))
            suf = mul(suf, xm[i])
        return res

    def dot(self, a_list, b_list):
        r = self.zero
        for a, b in zip(a_list, b_list):
            r = self.add(r, self.mul(a, b))
        return r


_FE = {}
_LW = {}


def fe(D):
    if D not in _FE:
        _FE[D] = FE(D)
    return _FE[D]


def emb(a):
    """F_p4 tuple -> F_p8 tuple."""
    return tuple(a) + (0,) * (8 - len(a)) if len(a) < 8 else tuple(a)
