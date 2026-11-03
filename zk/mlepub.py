"""Closed-form multilinear extensions of structured public vectors (verifier).

The verifier used to evaluate every public column by folding the whole vector (mle_host), which
costs O(length) extension operations per evaluation; inside a recursive verifier that is the
dominant cost.  The structured columns used by the protocol have O(bits) closed forms:

  iota(i) = i, or (i >> s) << s           sum_j 2^j x_j over the kept bits
  [c <= r] on (c low bits, r high bits)    comparator from the top bit
  [v < y], [v == y] for constants y        comparator / eq against y's bits
  vocabulary indicators of the targets     the above per row, weighted by eq over the rows

Points are host arrays (D, n) with coordinate j the bit j of the index (little-endian), as in
mle_host.  Every function returns a (D,) int64 array."""
import numpy as np
from .field import P
from .fe import fe


def _col(pt, j):
    return tuple(int(v) % P for v in np.asarray(pt[:, j]))


def iota(pt, shift=0):
    """MLE of i -> (i >> shift) << shift over len(pt) bits."""
    D = pt.shape[0]
    F = fe(D)
    acc = F.zero
    for j in range(shift, pt.shape[1]):
        acc = F.add(acc, F.smul(_col(pt, j), pow(2, j, P)))
    return np.array(acc, dtype=np.int64)


def _cmp(F, xs, ys):
    """(lt, eq) MLEs for the predicates x < y and x == y, bitwise from the most significant bit;
    xs, ys: lists of D-tuples (bit j at index j)."""
    one = F.one
    eq, lt = one, F.zero
    for j in range(len(xs) - 1, -1, -1):
        a, b = xs[j], ys[j]
        ab = F.mul(a, b)
        # bit_lt = (1 - a) b ; bit_eq = ab + (1 - a)(1 - b) = 2ab - a - b + 1
        blt = F.sub(b, ab)
        beq = F.add(F.sub(F.sub(F.smul(ab, 2), a), b), one)
        lt = F.add(lt, F.mul(eq, blt))
        eq = F.mul(eq, beq)
    return lt, eq


def le_lowhigh(pt, k):
    """MLE of index -> [c <= r] with c the low k bits and r the next k bits (a k-bit square mask)."""
    D = pt.shape[0]
    F = fe(D)
    lo = [_col(pt, j) for j in range(k)]
    hi = [_col(pt, k + j) for j in range(k)]
    lt, eq = _cmp(F, lo, hi)
    return np.array(F.add(lt, eq), dtype=np.int64)


def _const_bits(F, y, n):
    return [F.base((y >> j) & 1) for j in range(n)]


def lt_const(F, xs, y):
    """MLE of v -> [v < y] over the bits xs (y an integer, may be outside [0, 2^n])."""
    n = len(xs)
    if y <= 0:
        return F.zero
    if y >= (1 << n):
        return F.one
    return _cmp(F, xs, _const_bits(F, y, n))[0]


def eq_const(F, xs, y):
    """MLE of v -> [v == y] over the bits xs."""
    n = len(xs)
    if y < 0 or y >= (1 << n):
        return F.zero
    e = F.one
    for j in range(n):
        x = xs[j]
        e = F.mul(e, x if (y >> j) & 1 else F.sub(F.one, x))
    return e


def eq_rows(F, xs, R):
    """eq(xs, r) for r < R as a list of D-tuples (R = 2^len(xs))."""
    t = [F.one]
    for j in range(len(xs)):
        x = xs[j]
        nx = F.sub(F.one, x)
        t = [F.mul(v, nx) for v in t] + [F.mul(v, x) for v in t]
    return t[:R]


def vocab_blocks(pt, ys, cb, kind):
    """MLEs of the Kv blocks of rm(X, cb) for X (R, V) the row indicators of targets ys:
    kind "lt": X[r, v] = [v < y_r]; "eq": X[r, v] = [v == y_r].  Block k has index
    (v_lo low cb bits, r high bits) with v = k 2^cb + v_lo.  Returns (D, Kv) for Kv = ceil(V/2^cb)
    given as len of the result by the caller (blocks beyond every y are 0 for "eq", and for "lt"
    the sum of eq weights of rows whose target lies beyond)."""
    D = pt.shape[0]
    F = fe(D)
    lo = [_col(pt, j) for j in range(cb)]
    rbits = [_col(pt, cb + j) for j in range(pt.shape[1] - cb)]
    R = len(ys)
    assert R == 1 << len(rbits), (R, len(rbits))
    er = eq_rows(F, rbits, R)
    return lo, er


def vocab_indicator(pt, ys, cb, Kv, kind):
    """(D, Kv) MLEs of the blocks of rm([v < y_r] or [v == y_r], cb), see vocab_blocks."""
    D = pt.shape[0]
    F = fe(D)
    lo, er = vocab_blocks(pt, ys, cb, kind)
    out = [F.zero] * Kv
    if kind == "eq":
        for r, y in enumerate(ys):
            k = y >> cb
            if k < Kv:
                out[k] = F.add(out[k], F.mul(er[r], eq_const(F, lo, y & ((1 << cb) - 1))))
    else:
        # rows whose target block is above k contribute their full eq weight to block k
        above = [F.zero] * (Kv + 1)
        for r, y in enumerate(ys):
            k = min(y >> cb, Kv)
            above[k] = F.add(above[k], er[r])
            if k < Kv:
                out[k] = F.add(out[k], F.mul(er[r], lt_const(F, lo, y & ((1 << cb) - 1))))
        run = F.zero
        for k in range(Kv - 1, -1, -1):
            run = F.add(run, above[k + 1])
            out[k] = F.add(out[k], run)
    return np.array(out, dtype=np.int64).T
