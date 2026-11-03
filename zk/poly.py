"""Polynomials over column symbols with base-field integer coefficients.

A Poly is {monomial: coef}, monomial = sorted tuple of column ids (repeats
allowed), coef an int mod p.  Used for constraint and lookup-tuple
expressions; ext-field combinations are applied when converting to sumcheck
terms."""
from .field import P


class Poly:
    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = {k: v % P for k, v in (t or {}).items() if v % P}

    @staticmethod
    def var(i):
        return Poly({(i,): 1})

    @staticmethod
    def const(c):
        return Poly({(): c})

    @staticmethod
    def lift(x):
        return x if isinstance(x, Poly) else Poly.const(int(x))

    def __add__(self, o):
        o = Poly.lift(o)
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = (t.get(k, 0) + v) % P
        return Poly(t)

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-Poly.lift(o))

    def __rsub__(self, o):
        return Poly.lift(o) - self

    def __mul__(self, o):
        o = Poly.lift(o)
        t = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = tuple(sorted(k1 + k2))
                t[k] = (t.get(k, 0) + v1 * v2) % P
        return Poly(t)

    __rmul__ = __mul__

    def __pow__(self, e):
        r = Poly.const(1)
        for _ in range(e):
            r = r * self
        return r

    def degree(self):
        return max((len(k) for k in self.t), default=0)

    def vars(self):
        return sorted(set(i for k in self.t for i in k))

    def const_term(self):
        return self.t.get((), 0)

    def rename(self, m):
        """Substitute column ids via dict/callable m (layer parameters, ids < 0, are kept)."""
        f0 = m if callable(m) else (lambda i: m[i])
        f = lambda i: i if i < 0 else f0(i)
        t = {}
        for k, v in self.t.items():
            nk = tuple(sorted(f(i) for i in k))
            t[nk] = (t.get(nk, 0) + v) % P
        return Poly(t)

    def eval(self, vals):
        """Evaluate over ints mod p; vals: dict id -> int."""
        s = 0
        for k, v in self.t.items():
            m = v
            for i in k:
                m = m * vals[i] % P
            s += m
        return s % P

    def __repr__(self):
        return " + ".join(f"{v}*{'*'.join(f'c{i}' for i in k) or '1'}" for k, v in self.t.items()) or "0"


# Layer parameters (per-layer format constants such as shift offsets) appear in constraint constant
# terms as variables with reserved ids < 0: the prover then reads their values from a per-layer
# device buffer (domain.LPARAMS_DEV), so the same domain program serves every layer.
LPARAM_NAMES = ("Kco", "Kcd", "Dyn", "Dme")


def lparam(name):
    return Poly.var(-1 - LPARAM_NAMES.index(name))


def lparam_index(var_id):
    return -1 - var_id

