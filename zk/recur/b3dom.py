"""BLAKE3 G-function rows as a domain (one row per G call), for the recursive verifier.

Words are 4 little-endian bytes (32-bit words do not fit the 31-bit field).  Per row, for
inputs (a, b, c, d) and message words (mx, my):

  s1 = a + b + mx      x1 = d ^ s1    d1 = rotr16(x1)        (byte permutation)
  s2 = c + d1          x2 = b ^ s2    b1 = rotr12(x2)        (x2 bytes split into nibbles)
  s3 = s1 + b1 + my    x3 = d1 ^ s3   d2 = rotr8(x3)         (byte permutation)
  s4 = s2 + d2         x4 = b1 ^ s4   b2 = rotr7(x4)         (x4 bytes split into 1 + 7 bits)

outputs (s3, b2, s4, d2).  Additions: one linear constraint per byte with a carry cell, whose
range {0, 1(, 2)} is a polynomial constraint; XOR: lookups into 2^16-row tables that also bound
the bytes: (XOR_TAG, u, v, u ^ v), and for x2 / x4 the split parts directly (XORN_TAG: nibbles
of u ^ v, XOR7_TAG: top bit and low 7 bits), so the rotations by 12 and 7 need no other cells;
inputs not bounded by an XOR lookup get byte range lookups when checked (standalone).  The same expressions (exprs) build
the domain (Poly columns) and count the table multiplicities (numpy columns)."""
import numpy as np
import torch
from ..field import P
from .. import domain as dm
from ..sumcheck import ROW, COL
from .b3ref import M32

XOR_TAG, XORN_TAG, XOR7_TAG = 32, 33, 34
RANGES = (8,)                     # range tags used: k -> rows (k, v) for v < 2^k
WORDS = ("a", "b", "c", "d", "mx", "my")
COLS = ([f"{w}{j}" for w in WORDS for j in range(4)]
        + [f"s{i}_{j}" for i in range(1, 5) for j in range(4)]
        + [f"k{i}_{j}" for i in range(1, 5) for j in range(1, 5)]
        + [f"x{i}_{j}" for i in (1, 3) for j in range(4)]
        + [f"h{j}" for j in range(4)] + [f"l{j}" for j in range(4)]
        + [f"t{j}" for j in range(4)] + [f"r{j}" for j in range(4)])


def exprs(V, inputs_checked=True):
    """V: column name -> Poly or numpy array.  Returns (constraints, lookups [(tag, [vals])])."""
    byte = lambda w, j: V[f"{w}{j}"]
    s = lambda i, j: V[f"s{i}_{j}"]
    x = lambda i, j: V[f"x{i}_{j}"]
    k = lambda i, j: V[f"k{i}_{j}"] if j else 0          # k_i_0 = 0 (no carry into byte 0)
    d1 = lambda j: x(1, (j + 2) % 4)                      # rotr16
    b1 = lambda j: V[f"h{(j + 1) % 4}"] + 16 * V[f"l{(j + 2) % 4}"]   # rotr12 of x2
    d2 = lambda j: x(3, (j + 1) % 4)                      # rotr8
    cons, lks = [], []
    for j in range(4):
        cons.append(byte("a", j) + byte("b", j) + byte("mx", j) + k(1, j) - s(1, j) - 256 * k(1, j + 1))
        cons.append(byte("c", j) + d1(j) + k(2, j) - s(2, j) - 256 * k(2, j + 1))
        cons.append(s(1, j) + b1(j) + byte("my", j) + k(3, j) - s(3, j) - 256 * k(3, j + 1))
        cons.append(s(2, j) + d2(j) + k(4, j) - s(4, j) - 256 * k(4, j + 1))
        for i in range(1, 5):          # carries: {0, 1, 2} for the three-operand adds, else {0, 1}
            kk = V[f"k{i}_{j + 1}"]
            cons.append(kk * (kk - 1) * (kk - 2) if i in (1, 3) else kk * (kk - 1))
    for j in range(4):
        lks.append((XOR_TAG, [byte("d", j), s(1, j), x(1, j)]))
        lks.append((XORN_TAG, [byte("b", j), s(2, j), V[f"h{j}"], V[f"l{j}"]]))
        lks.append((XOR_TAG, [d1(j), s(3, j), x(3, j)]))
        lks.append((XOR7_TAG, [b1(j), s(4, j), V[f"t{j}"], V[f"r{j}"]]))
        if inputs_checked:              # inputs not bounded by an XOR lookup
            lks += [(8, [byte(w, j)]) for w in ("a", "c", "mx", "my")]
    return cons, lks


def outputs(V):
    """Output words' bytes of the row: (a', b', c', d') as lists of 4 byte expressions."""
    b2 = [2 * V[f"r{(j + 1) % 4}"] + V[f"t{j}"] for j in range(4)]       # rotr7 of x4
    return ([V[f"s3_{j}"] for j in range(4)], b2, [V[f"s4_{j}"] for j in range(4)],
            [V[f"x3_{(j + 1) % 4}"] for j in range(4)])


def witness(trace, n):
    """G trace (list of dicts of (N,) word arrays, see b3ref.compress) -> columns (2^n,) int64,
    rows in call order (compression-major), zero padded."""
    rows = {w: np.concatenate([np.stack([t[w] for t in trace], axis=1).reshape(-1)])
            for w in trace[0]}
    R = rows["a"].size
    assert R <= 1 << n

    def bytes_(w):
        return [(w >> (8 * j)) & 255 for j in range(4)]
    out = {}
    for w in WORDS:
        for j, v in enumerate(bytes_(rows[w])):
            out[f"{w}{j}"] = v
    adds = {1: ("a", "b", "mx"), 2: ("c", "d1", None), 3: ("s1", "b1", "my"), 4: ("s2", "d2", None)}
    rows["d1"] = ((rows["x1"] >> 16) | (rows["x1"] << 16)) & M32
    rows["b1"] = ((rows["x2"] >> 12) | (rows["x2"] << 20)) & M32
    rows["d2"] = ((rows["x3"] >> 8) | (rows["x3"] << 24)) & M32
    for i, (u, v, m) in adds.items():
        bu, bv = bytes_(rows[u]), bytes_(rows[v])
        bm = bytes_(rows[m]) if m else [0] * 4
        carry = 0
        for j in range(4):
            tot = bu[j] + bv[j] + bm[j] + carry
            out[f"s{i}_{j}"] = tot & 255
            carry = tot >> 8
            out[f"k{i}_{j + 1}"] = carry
        assert np.array_equal(sum(out[f"s{i}_{j}"] << (8 * j) for j in range(4)), rows[f"s{i}"])
    xb = {i: bytes_(rows[f"x{i}"]) for i in range(1, 5)}
    for i in (1, 3):
        for j, v in enumerate(xb[i]):
            out[f"x{i}_{j}"] = v
    for j in range(4):
        out[f"h{j}"], out[f"l{j}"] = xb[2][j] >> 4, xb[2][j] & 15
        out[f"t{j}"], out[f"r{j}"] = xb[4][j] >> 7, xb[4][j] & 127
    pad = (1 << n) - R
    return {c: np.concatenate([out[c], np.zeros(pad, dtype=np.int64)]) for c in COLS}


def g_domain(n, data=None, inputs_checked=True, name="b3g"):
    """The G-row domain over 2^n rows (data: witness columns, or None for the verifier)."""
    d = dm.Domain(name, n, 1)
    dev = (lambda c: torch.from_numpy((data[c] % P).astype(np.int32)).cuda()) if data else (lambda c: None)
    V = {c: d.col(c, dev(c)) for c in COLS}
    cons, lks = exprs(V, inputs_checked)
    for cp in cons:
        d.zero(cp)
    for tag, vals in lks:
        d.lookup(tag, *vals)
    return d


# ------------------------------------------------------------------ table
def table_rows():
    """Rows (tag, v0, v1, v2, v3): range tags then the XOR tables."""
    rows = []
    off = {}
    for k in RANGES:
        off[k] = len(rows)
        rows += [(k, v, 0, 0, 0) for v in range(1 << k)]
    arr = [np.array(rows, dtype=np.int64).reshape(-1, 5)] if rows else []
    u = np.repeat(np.arange(256), 256)
    v = np.tile(np.arange(256), 256)
    x = u ^ v
    n0 = len(rows)
    for tag, a, b in ((XOR_TAG, x, 0 * x), (XORN_TAG, x >> 4, x & 15), (XOR7_TAG, x >> 7, x & 127)):
        off[tag] = n0
        arr.append(np.stack([np.full(65536, tag), u, v, a, b], axis=1))
        n0 += 65536
    return np.concatenate(arr), off


def multiplicities(data, nT, inputs_checked=True):
    """Table multiplicities of the G rows' lookups (numpy columns)."""
    return table_mult(exprs(data, inputs_checked)[1], nT)


def table_domain(nT, mult=None):
    arr, off = table_rows()
    full = np.zeros((1 << nT, 5), dtype=np.int64)
    full[:len(arr)] = arr
    full[len(arr):] = arr[0]           # padding rows repeat the first row (multiplicity 0)
    d = dm.Domain("b3table", nT, 1)
    dev = lambda a: torch.from_numpy((a % P).astype(np.int32)).cuda()
    cols = [d.col(f"t{i}", dev(full[:, i]) if mult is not None else None, role="public") for i in range(5)]
    m = d.col("m", dev(mult) if mult is not None else None)
    d.lookup(cols[0], cols[1], cols[2], cols[3], cols[4], num=-m)
    return d, full


# ------------------------------------------------------------------ wiring (LogUp memory)
# Every word is a tuple (MEM_TAG, id, b0, b1, b2, b3) in a dynamic set: written by the row that
# produces it (numerator -mult) and read by each consumer (numerator +1), so the sets of all domains
# sum to zero exactly when every read finds the bytes written under its id.  All multiplicities are
# constants of the layout, so there are no multiplicity columns.  Ids of compression c: state slot
# s at version v: c * 512 + 16 s + v (each half round writes every slot once, versions 0..14);
# message word w: c * 512 + 256 + w (read 7 times).  Output words: (OUT_TAG, c * 8 + i, bytes).
# Written bytes are bounded (XOR lookups in the G rows, range lookups in the IN rows), so reads
# need no range checks of their own.
MEM_TAG, OUT_TAG = 64, 65
LOG_GPC, LOG_IN, LOG_OUT = 6, 5, 3   # rows per compression: G (56 used of 64), IN (32), OUT (8)
GPC = 1 << LOG_GPC
VFIN = 14                            # version of the final state


def _dev(a):
    return torch.from_numpy((np.asarray(a) % P).astype(np.int32)).cuda()


def g_ids():
    """Per position g < 64 of a compression: (sel, ids read for a, b, c, d, mx, my, ids written)."""
    from .b3ref import G_IDX, SCHED
    sel = np.zeros(GPC, dtype=np.int64)
    rd = np.zeros((6, GPC), dtype=np.int64)
    wr = np.zeros((4, GPC), dtype=np.int64)
    for g in range(56):
        r, i = divmod(g, 8)
        v = 2 * r + (i >= 4)
        sel[g] = 1
        for w, slot in enumerate(G_IDX[i]):
            rd[w, g] = 16 * slot + v
            wr[w, g] = 16 * slot + v + 1
        rd[4, g] = 256 + SCHED[r][2 * i]
        rd[5, g] = 256 + SCHED[r][2 * i + 1]
    return sel, rd, wr


def g_domain_wired(N, data=None, name="b3gw"):
    """G rows of N compressions (64 rows each, rows 56..63 inactive) with their memory traffic."""
    n = LOG_GPC + N.bit_length() - 1
    d = dm.Domain(name, n, 1)
    on = data is not None
    V = {c: d.col(c, _dev(data[c]) if on else None) for c in COLS}
    cons, lks = exprs(V, inputs_checked=False)
    for cp in cons:
        d.zero(cp)
    for tag, vals in lks:
        d.lookup(tag, *vals)
    sel, rd, wr = g_ids()
    pub = lambda nm, a, kind, c: d.col(nm, _dev(a) if on else None, kind, c, shared=kind == COL, role="public")
    base = pub("cbase", np.arange(N) * 512, ROW, LOG_GPC)
    sl = pub("sel", sel, COL, LOG_GPC)
    rid = [pub(f"rid{w}", rd[w], COL, LOG_GPC) for w in range(6)]
    wid = [pub(f"wid{w}", wr[w], COL, LOG_GPC) for w in range(4)]
    for w, nm in enumerate(WORDS):
        d.lookup(MEM_TAG, base + rid[w], *[V[f"{nm}{j}"] for j in range(4)], num=sl, dynamic=True)
    for w, bs in enumerate(outputs(V)):
        d.lookup(MEM_TAG, base + wid[w], *bs, num=-sl)
    return d


def g_witness_wired(trace):
    """Columns of g_domain_wired from a G trace (b3ref.compress) of N compressions."""
    N = trace[0]["a"].size
    W = witness(trace, max(2, (N * 56 - 1).bit_length()))
    pad = np.zeros((N, GPC - 56), dtype=np.int64)
    return {c: np.concatenate([W[c][:N * 56].reshape(N, 56), pad], axis=1).reshape(-1) for c in COLS}


def _bytes(w):
    return [(np.asarray(w) >> (8 * j)) & 255 for j in range(4)]


def in_words(cv, block, counter, blen, flags):
    """(N, 32) initial words of each compression: state (cv, IV, counter, blen, flags), block."""
    from .b3ref import IV
    N = cv.shape[0]
    ctr = np.broadcast_to(np.asarray(counter, dtype=np.int64), (N,))
    st = np.stack([cv[:, i] for i in range(8)] + [np.full(N, IV[i]) for i in range(4)]
                  + [ctr & M32, ctr >> 32, np.broadcast_to(blen, (N,)), np.broadcast_to(flags, (N,))], axis=1)
    return np.concatenate([st, block], axis=1).astype(np.int64)


def in_witness(words):
    w = words.reshape(-1)
    return {f"w{j}": v for j, v in enumerate(_bytes(w))}


def in_domain(N, data=None, name="b3in"):
    """IN rows (c, k): word k < 16 written as state slot k version 0, k >= 16 as message word k - 16
    (multiplicity 7); bytes range checked."""
    n = LOG_IN + N.bit_length() - 1
    d = dm.Domain(name, n, 1)
    on = data is not None
    k = np.arange(32)
    pub = lambda nm, a, kind, c: d.col(nm, _dev(a) if on else None, kind, c, shared=kind == COL, role="public")
    base = pub("cbase", np.arange(N) * 512, ROW, LOG_IN)
    off = pub("off", np.where(k < 16, 16 * k, 256 + k - 16), COL, LOG_IN)
    mult = pub("mult", np.where(k < 16, 1, 7), COL, LOG_IN)
    bs = [d.col(f"w{j}", _dev(data[f"w{j}"]) if on else None) for j in range(4)]
    for b in bs:
        d.lookup(8, b)
    d.lookup(MEM_TAG, base + off, *bs, num=-mult)
    return d


def in_lookups(data):
    return [(8, [data[f"w{j}"]]) for j in range(4)]


def out_witness(final):
    """final: (N, 16) final state words -> OUT rows (c, i): u = s_i, v = s_(i+8), o = u ^ v."""
    u, v = final[:, :8].reshape(-1), final[:, 8:].reshape(-1)
    W = {}
    for nm, a in (("u", u), ("v", v), ("o", u ^ v)):
        for j, b in enumerate(_bytes(a)):
            W[f"{nm}{j}"] = b
    return W


def out_domain(N, data=None, name="b3out"):
    """OUT rows (c, i): read slots i and i + 8 at the final version, XOR them, write output word
    (OUT_TAG, 8 c + i)."""
    n = LOG_OUT + N.bit_length() - 1
    d = dm.Domain(name, n, 1)
    on = data is not None
    i = np.arange(8)
    pub = lambda nm, a, kind, c: d.col(nm, _dev(a) if on else None, kind, c, shared=kind == COL, role="public")
    base = pub("cbase", np.arange(N) * 512, ROW, LOG_OUT)
    obase = pub("obase", np.arange(N) * 8, ROW, LOG_OUT)
    ii = pub("i", i, COL, LOG_OUT)
    V = {f"{nm}{j}": d.col(f"{nm}{j}", _dev(data[f"{nm}{j}"]) if on else None) for nm in "uvo" for j in range(4)}
    for nm, off in (("u", 0), ("v", 128)):
        d.lookup(MEM_TAG, base + 16 * ii + off + VFIN, *[V[f"{nm}{j}"] for j in range(4)], dynamic=True)
    for tag, vals in out_lookups(V):
        d.lookup(tag, *vals)
    d.lookup(OUT_TAG, obase + ii, *[V[f"o{j}"] for j in range(4)], num=-1)
    return d


def out_lookups(V):
    return [(XOR_TAG, [V[f"u{j}"], V[f"v{j}"], V[f"o{j}"]]) for j in range(4)]


def expect_domain(out, on=True, name="b3exp"):
    """Reads the output words (N, 8) as public values (stand-in for their consumers)."""
    N = out.shape[0]
    n = LOG_OUT + N.bit_length() - 1
    d = dm.Domain(name, n, 1)
    oid = d.col("oid", _dev(np.arange(8 * N)) if on else None, role="public")
    bs = [d.col(f"e{j}", _dev(b) if on else None, role="public") for j, b in enumerate(_bytes(out.reshape(-1)))]
    d.lookup(OUT_TAG, oid, *bs, dynamic=True)
    return d


def table_mult(lks, nT):
    """Table multiplicities of numpy lookups [(tag, vals)] (global-table tags only)."""
    _, off = table_rows()
    m = np.zeros(1 << nT, dtype=np.int64)
    for tag, vals in lks:
        if tag in (XOR_TAG, XORN_TAG, XOR7_TAG):
            idx = off[tag] + vals[0] * 256 + vals[1]
        else:
            idx = off[tag] + vals[0]
        m += np.bincount(np.asarray(idx, dtype=np.int64), minlength=1 << nT)
    return m
