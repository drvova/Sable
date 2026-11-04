"""Writes lisp/tests/vectors.sexp: reference values computed by the Python implementation (zk/), which
the Common Lisp port must reproduce bit for bit.  Needs numpy, blake3 and g++ (for zk/cuda/b3s.h);
torch is not needed (a stub stands in for torch.is_tensor).

    python3 lisp/tests/gen-vectors.py            # from the repository root
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
torch = types.ModuleType("torch")
torch.is_tensor = lambda x: False
sys.modules.setdefault("torch", torch)

import blake3  # noqa: E402
import numpy as np  # noqa: E402
from zk import field, fe as fem, rs, proofio  # noqa: E402
from zk.poly import Poly  # noqa: E402
from zk.b3s import Hasher  # noqa: E402
from zk.transcript import Transcript, label_name  # noqa: E402

P = field.P
rng = np.random.default_rng(20261104)
out = []


def sx(v):
    """Python value -> s-expression text."""
    if isinstance(v, (bytes, bytearray)):
        return '"' + bytes(v).hex() + '"'
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "t" if v else "nil"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, np.ndarray):
        return sx(v.tolist())
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(sx(x) for x in v) + ")"
    raise TypeError(type(v))


def emit(kind, **kw):
    out.append("(:" + kind + " " + " ".join(f":{k.replace('_', '-')} {sx(v)}" for k, v in kw.items()) + ")")


def rel(D):
    return [int(x) for x in rng.integers(0, P, size=D)]


# ------------------------------------------------------------------ fields
for D in (4, 8):
    F = fem.fe(D)
    for _ in range(12):
        a, b = rel(D), rel(D)
        emit("ext", d=D, a=a, b=b, add=F.add(a, b), sub=F.sub(a, b), mul=F.mul(a, b), inv=F.inv(a),
             eq1=F.eq1(a, b))
    for deg in (1, 2, 3, 5):
        vals = [rel(D) for _ in range(deg + 1)]
        x = rel(D)
        emit("lagrange", d=D, vals=vals, x=x, res=F.lagrange(vals, x))
    for k in (0, 1, 3, 5):
        a = [rel(D) for _ in range(k)]
        b = [rel(D) for _ in range(k)]
        emit("eq", d=D, a=a, b=b, res=F.eq(a, b))
        E = field.Ext(D)
        r = np.array(a, dtype=np.int64).T.reshape(D, k)
        tab = E.eq_table(r)
        base = rng.integers(0, P, size=1 << k)
        ext = rng.integers(0, P, size=(D, 1 << k))
        emit("mle", d=D, r=a, eqtab=tab.T, base=base, base_res=E.mle_eval(base, r), ext=ext.T,
             ext_res=E.mle_eval(ext, r))
emit("pow", a=123456789, e=P - 2, res=pow(123456789, P - 2, P))

# ------------------------------------------------------------------ BLAKE3
for n in (0, 1, 31, 63, 64, 65, 127, 128, 129, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 5000, 9999):
    data = bytes((i * 7 + n) & 255 for i in range(n))
    emit("blake3", input=data, digest=blake3.blake3(data).digest(), xof=blake3.blake3(data).digest(length=131))
for chunks in ([b""], [b"abc"], [b"x" * 64], [b"x" * 64, b"y"], [b"a" * 10, b"b" * 200, b"c" * 64], [bytes(range(256)) * 3]):
    h = Hasher()
    for c in chunks:
        h.update(c)
    emit("b3s", chunks=chunks, digest=h.digest(32), xof=h.digest(200))

# ------------------------------------------------------------------ transcript
tr = Transcript(b"vec", 8)
ops = []
tr.absorb(b"hello")
ops.append(["absorb", b"hello"])
ops.append(["challenge-base", 3, tr.challenge_base(3)])
arr = rng.integers(0, P, size=(3, 8))
tr.absorb_ext(arr)
ops.append(["absorb-ext", arr.reshape(-1)])
ops.append(["challenge", 8, tr.challenge(D=8)])
ops.append(["challenge", 4, tr.challenge(D=4)])
ops.append(["challenge-bits", 20, 5, tr.challenge_bits(20, 5)])
ops.append(["challenge-base", 70, tr.challenge_base(70)])
ops.append(["digest", tr.h.copy().digest()])
emit("transcript", label=b"vec", ops=ops)
for bits in (0, 4, 8):
    base = blake3.blake3(bytes([bits])).digest()
    nonce = next(i for i in range(1 << 20) if Transcript.check_grind(base, i, bits))
    emit("grind", base=base, bits=bits, nonce=nonce, ok=True)
    if nonce:
        emit("grind", base=base, bits=bits, nonce=nonce - 1, ok=Transcript.check_grind(base, nonce - 1, bits))
for name in ("12.mo", "3.oproj|3.rqo", "mo", "a.b", "0.x|y|17.z"):
    emit("label-name", name=name, res=label_name(name).decode())

# ------------------------------------------------------------------ proof byte format
msgs = [b"abc", np.array(5, dtype=np.int64), rng.integers(0, P, size=(2, 3)), b"", rng.integers(0, P, size=(4,)),
        np.zeros((0, 8), dtype=np.int64)]
blob = proofio.dumps(msgs)
assert all((isinstance(a, bytes) and a == b) or np.array_equal(a, b) for a, b in zip(msgs, proofio.loads(blob)))
emit("proofio", bytes=blob, msgs=[["bytes", m] if isinstance(m, bytes) else ["array", list(np.shape(m)), np.asarray(m).reshape(-1)] for m in msgs])
bad = {"magic": b"SABLEPF2" + blob[8:], "truncated": blob[:-1], "trailing": blob + b"\x00",
       "tag": blob[:12] + b"\x07" + blob[13:],
       "range": proofio.MAGIC + (1).to_bytes(4, "little") + b"\x01\x01" + (1).to_bytes(8, "little") + P.to_bytes(4, "little"),
       "rank": proofio.MAGIC + (1).to_bytes(4, "little") + b"\x01\x09" + bytes(72),
       "huge": proofio.MAGIC + (1).to_bytes(4, "little") + b"\x01\x02" + (1 << 40).to_bytes(8, "little") * 2}
for why, b in bad.items():
    try:
        proofio.loads(b)
        raise AssertionError(why)
    except ValueError as e:
        emit("proofio-bad", why=why, bytes=b, message=str(e))

# ------------------------------------------------------------------ polynomials
x, y, z = Poly.var(0), Poly.var(1), Poly.var(2)
p = (x + 3) * (y - x) * z + 7 - x ** 2
vals = {0: 11, 1: P - 5, 2: 1 << 30}
emit("poly", eval=p.eval(vals), degree=p.degree(), vars=p.vars(), const=p.const_term(),
     terms=sorted([list(k), v] for k, v in p.t.items()))

# ------------------------------------------------------------------ Reed-Solomon
assert rs.selftest()
for n in (2, 3, 5, 7):
    msg = rng.integers(0, P, size=1 << (n - 1))
    emit("rs-encode", n=n, root=rs.root_of_unity(n), msg=msg, cw=rs.encode_ref(msg, n), points=rs.domain_points(n),
         tw=[t for t in rs.twiddles(n)] if n > 1 else [])
n = 6
E = field.Ext(8)
tw = rs.twiddles(n)
msg = rng.integers(0, P, size=1 << (n - 1))
cw = E.from_base(rs.encode_ref(msg, n, tw))
rr = rng.integers(0, P, size=8)
emit("rs-fold", n=n, cw=cw.T, r=rr, res=rs.fold_ref(E, cw, tw[0], rr).T)


# ------------------------------------------------------------------ sumcheck (honest CPU prover + verify)
def sc_prove(tr, D, n, tables, eq_r, label):
    """sum_x w(x) prod_k T_k(x); tables: lists of FE tuples (length 2^n)."""
    F = fem.fe(D)
    d = len(tables)
    npts = max(d, 1)
    tr.absorb(label + b"|" + n.to_bytes(2, "little") + d.to_bytes(2, "little") + (1).to_bytes(4, "little"))
    T = [list(t) for t in tables]
    rho = []
    for j in range(n):
        if eq_r is not None:
            rest = [tuple(int(v) for v in eq_r[:, i]) for i in range(j + 1, n)]
            w = [F.one]
            for rk in rest:
                w = [F.mul(v, F.sub(F.one, rk)) for v in w] + [F.mul(v, rk) for v in w]
        else:
            w = [F.one] * (len(T[0]) // 2)
        g = []
        for X in [0] + list(range(2, npts + 1)):
            acc = F.zero
            for i in range(len(T[0]) // 2):
                m = w[i]
                for t in T:
                    lo, hi = t[2 * i], t[2 * i + 1]
                    m = F.mul(m, F.add(lo, F.smul(F.sub(hi, lo), X)))
                acc = F.add(acc, m)
            g.append(acc)
        tr.send(np.array(g, dtype=np.int64).reshape(npts, D))
        r = F.c(tr.challenge(D=D))
        rho.append(r)
        T = [[F.add(t[2 * i], F.mul(r, F.sub(t[2 * i + 1], t[2 * i]))) for i in range(len(t) // 2)] for t in T]
    return rho, [t[0] for t in T]


def sc_verify(tr, D, n, d, claim, eq_r, label):          # zk/sumcheck.py verify (Python path)
    F = fem.fe(D)
    tr.absorb(label + b"|" + n.to_bytes(2, "little") + d.to_bytes(2, "little") + (1).to_bytes(4, "little"))
    claim_p = np.asarray(claim, dtype=np.int64) % P
    npts = max(d, 1)
    rho = []
    for j in range(n):
        msg = tr.send(None)
        assert msg.shape == (npts, D)
        rj = tr.challenge(D=D)
        rho.append(rj)
        c = F.c(claim_p)
        g0 = F.c(msg[0])
        if eq_r is not None:
            r = F.c(eq_r[:, j])
            g1 = F.mul(F.sub(c, F.mul(F.sub(F.one, r), g0)), F.inv(r))
        else:
            g1 = F.sub(c, g0)
        vals = [g0, g1] + [F.c(m) for m in msg[1:]]
        claim_p = np.array(F.lagrange(vals[:max(d, 1) + 1], F.c(rj)), dtype=np.int64)
    return rho, claim_p


for D, n, ntab, eqm in ((4, 4, 2, False), (8, 5, 3, True), (8, 3, 1, False), (4, 6, 4, True)):
    F = fem.fe(D)
    tables = [[tuple(rel(D)) for _ in range(1 << n)] for _ in range(ntab)]
    eq_r = np.array([rel(D) for _ in range(n)], dtype=np.int64).T if eqm else None
    if eqm:
        w = [F.one]
        for k in range(n):
            rk = F.c(eq_r[:, k])
            w = [F.mul(v, F.sub(F.one, rk)) for v in w] + [F.mul(v, rk) for v in w]
    else:
        w = [F.one] * (1 << n)
    claim = F.zero
    for x in range(1 << n):
        m = w[x]
        for t in tables:
            m = F.mul(m, t[x])
        claim = F.add(claim, m)
    ptr = Transcript(b"sc-vec", D)
    rho, finals = sc_prove(ptr, D, n, tables, eq_r, b"sc")
    blob = proofio.dumps(ptr.msgs)
    vtr = Transcript(b"sc-vec", D, proofio.loads(blob))
    vrho, red = sc_verify(vtr, D, n, ntab, claim, eq_r, b"sc")
    prod = F.one
    for f in finals:
        prod = F.mul(prod, f)
    assert [F.c(r) for r in vrho] == rho and F.c(red) == prod
    emit("sumcheck", d=D, n=n, tables=tables, eq_r=None if eq_r is None else eq_r.T, claim=claim, proof=blob,
         rho=rho, reduced=F.c(red), finals=finals)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors.sexp")
with open(path, "w") as f:
    f.write(";;; Generated by lisp/tests/gen-vectors.py from the Python implementation; do not edit.\n(\n")
    f.write("\n".join(out))
    f.write("\n)\n")
print(f"{len(out)} vectors -> {path}")
