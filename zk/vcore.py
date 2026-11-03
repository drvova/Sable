"""Compiled verifier core (zk/cuda/vcore.cpp): the sumcheck and LogUp-GKR verifiers in C++, on the
transcript's own state buffer (b3s.h) and the proof's field messages packed once into one array.
Used by sumcheck.verify and gkr.verify_upper in verifier mode; ZK_PYVERIFY=1 keeps the Python
implementations (e.g. for operation counts)."""
import ctypes
import hashlib
import os
import subprocess
import numpy as np
from .field import P

_DIR = os.path.join(os.path.dirname(__file__), "cuda")
_LIB = None
ENABLED = not os.environ.get("ZK_PYVERIFY")
CALLS = {"sc": 0, "gkr": 0}
TIME = [0.0]                    # seconds inside the compiled core (wall)
BYFN = {}
_p = ctypes.c_void_p
_i64 = ctypes.c_int64
_i32 = ctypes.c_int


class _Timed:
    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        f = getattr(self._lib, name)

        def call(*a):
            import time
            t0 = time.perf_counter()
            try:
                return f(*a)
            finally:
                dt = time.perf_counter() - t0
                TIME[0] += dt
                BYFN[name] = BYFN.get(name, 0.0) + dt
        return call


def _lib():
    global _LIB
    if _LIB is None:
        srcs = [os.path.join(_DIR, f) for f in ("b3s.h", "vcore.cpp")]
        dig = hashlib.sha1(b"".join(open(f, "rb").read() for f in srcs)).hexdigest()[:16]
        cache = os.path.join(os.path.expanduser("~"), ".cache", "zk_b3s")
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"vcore_{dig}.so")
        if not os.path.exists(so):
            tmp = so + f".{os.getpid()}.tmp"
            subprocess.check_call(["g++", "-O3", "-march=native", "-shared", "-fPIC", "-I", _DIR, srcs[1], "-o", tmp])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.vc_sc_verify.restype = _i64
        lib.vc_sc_verify.argtypes = [_p, _i32, _p, _p, _i64, _i64, ctypes.c_char_p, _i32, _i32, _i32, _i32,
                                     _p, _p, _p, _p]
        lib.vc_gkr_upper.restype = _i64
        lib.vc_gkr_upper.argtypes = [_p, _i32, _p, _p, _i64, _i64, ctypes.c_char_p, _i32, _i32, _i32, _i32, _i32,
                                     _p, _p, _p, _p, _p]
        lib.vc_eqv.restype = None
        lib.vc_eqv.argtypes = [_i32, _i32, _p, _p, _p]
        lib.vc_eval_prog.restype = None
        lib.vc_eval_prog.argtypes = [_i32, _i32, _p, _i32, _p, _i32, _p, _p, _i64, _p, _p, _i32, _p, _p]
        lib.vc_eq_table.restype = None
        lib.vc_eq_table.argtypes = [_i32, _i32, _p, _p]
        lib.vc_mle.restype = None
        lib.vc_mle.argtypes = [_i32, _i32, _i32, _p, _p, _p]
        lib.vc_vmul.restype = None
        lib.vc_vmul.argtypes = [_i32, _i64, _i32, _p, _p, _p]
        lib.vc_dot.restype = None
        lib.vc_dot.argtypes = [_i32, _i64, _i32, _p, _p, _p]
        _LIB = _Timed(lib)
    return _LIB


def usable(tr):
    return ENABLED and getattr(tr, "verifier", False) and not tr._pending and tr.dev is None


def _packed(tr):
    """(flat int64, offsets) of the transcript's messages; byte messages get length 0 (never a field shape)."""
    pk = getattr(tr, "_packed", None)
    if pk is None or pk[2] != len(tr.msgs):
        lens = np.array([m.size if isinstance(m, np.ndarray) else 0 for m in tr.msgs], dtype=np.int64)
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        arrs = [np.ascontiguousarray(m, dtype=np.int64).reshape(-1) for m in tr.msgs if isinstance(m, np.ndarray)]
        flat = np.concatenate(arrs) if arrs else np.zeros(1, dtype=np.int64)
        pk = tr._packed = (flat, off, len(tr.msgs))
    return pk


def _ptr(a):
    return a.ctypes.data if a is not None else None


def _check(pos, what):
    if pos == -1:
        raise ValueError(f"{what}: message shape")
    if pos == -2:
        raise ValueError("gkr top layer check failed")
    if pos == -3:
        raise ValueError("gkr layer check failed")


def sc_verify(tr, D, n, d, claim, eq_r, K, label):
    CALLS["sc"] += 1
    flat, off, nm = _packed(tr)
    st = tr.h.st
    cl = np.ascontiguousarray(np.asarray(claim, dtype=np.int64).reshape(-1) % P)
    er = None if eq_r is None else np.ascontiguousarray((np.asarray(eq_r, dtype=np.int64) % P)[:, :n].T)
    rho = np.zeros((n, D), dtype=np.int64)
    red = np.zeros(D, dtype=np.int64)
    pos = _lib().vc_sc_verify(st.ctypes.data, D, flat.ctypes.data, off.ctypes.data, nm, tr.pos, bytes(label),
                              len(label), n, d, K, cl.ctypes.data, _ptr(er), rho.ctypes.data, red.ctypes.data)
    _check(pos, "sumcheck")
    tr.pos = pos
    return rho.T.copy(), red


def gkr_upper(tr, D, n, K, L, label, d2):
    CALLS["gkr"] += 1
    from .fe import fe
    F = fe(D)
    flat, off, nm = _packed(tr)
    lk = max(0, (K - 1).bit_length())
    roots = np.zeros((L, 2, D), dtype=np.int64)
    ri = np.zeros((max(n - 1, 0), D), dtype=np.int64)
    rb = np.zeros((lk, D), dtype=np.int64)
    cp = np.zeros((L, D), dtype=np.int64)
    cq = np.zeros((L, D), dtype=np.int64)
    pos = _lib().vc_gkr_upper(tr.h.st.ctypes.data, D, flat.ctypes.data, off.ctypes.data, nm, tr.pos, bytes(label),
                              len(label), n, K, L, d2, roots.ctypes.data, ri.ctypes.data, rb.ctypes.data,
                              cp.ctypes.data, cq.ctypes.data)
    _check(pos, "gkr")
    tr.pos = pos
    return ([(F.c(roots[l, 0]), F.c(roots[l, 1])) for l in range(L)], ri.T.copy(), rb.T.copy(),
            [cp[l] for l in range(L)], [cq[l] for l in range(L)])


def eqv(D, a, b):
    """prod_j eq(a[:, j], b[:, j]) for (D, n) arrays -> FE tuple."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.int64))
    out = np.zeros(D, dtype=np.int64)
    _lib().vc_eqv(D, a.shape[1], a.ctypes.data, b.ctypes.data, out.ctypes.data)
    return tuple(int(v) for v in out)


_OPS = {"+": 4, "-": 5, "*": 6}


def _compiled(prog):
    """(polys, code) arrays of a sumcheck.Prog, cached on it."""
    c = getattr(prog, "_vc", None)
    if c is None:
        pl = []
        for p in prog.polys:
            pl.append(len(p.t))
            for mono, cf in p.t.items():
                assert all(i >= 0 for i in mono), "layer parameters are resolved before evaluation"
                pl += [int(cf) % P, len(mono)] + list(mono)
        code = []

        def rec(e):
            if e[0] == "p":
                code.extend((0, e[1]))
            elif e[0] == "c":
                code.extend((1, e[1]))
            elif e[0] == "k":
                code.extend((2, int(e[1]) % P))
            elif e[0] == "kc":
                code.extend((3, e[1]))
            else:
                rec(e[1])
                rec(e[2])
                code.extend((_OPS[e[0]], 0))
        rec(prog.expr)
        c = prog._vc = (np.array(pl or [0], dtype=np.int64), np.array(code, dtype=np.int64))
    return c


def eval_prog(D, prog, finals, bw=None):
    """sumcheck.eval_prog in C++ -> numpy (D,)."""
    pl, code = _compiled(prog)
    K = max(f.shape[1] for f in finals) if finals else 1
    fw = np.array([f.shape[1] for f in finals], dtype=np.int64)
    fo = np.zeros(len(finals), dtype=np.int64)
    if len(finals) > 1:
        np.cumsum(fw[:-1], out=fo[1:])
    fin = np.ascontiguousarray(np.concatenate([np.asarray(f, dtype=np.int64) for f in finals], axis=1)) \
        if finals else np.zeros((D, 1), dtype=np.int64)
    co = np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.int64).reshape(-1) for c in prog.coefs])) \
        if prog.coefs else np.zeros((1, D), dtype=np.int64)
    bwa = None if bw is None else np.ascontiguousarray(np.asarray(bw, dtype=np.int64))
    out = np.zeros(D, dtype=np.int64)
    _lib().vc_eval_prog(D, K, pl.ctypes.data, len(prog.polys), code.ctypes.data, len(code), co.ctypes.data,
                        fin.ctypes.data, fin.shape[1], fo.ctypes.data, fw.ctypes.data, len(finals),
                        None if bwa is None else bwa.ctypes.data, out.ctypes.data)
    return out


def _c64(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64))


def eq_table(D, r):
    r = _c64(r)
    k = r.shape[1]
    out = np.zeros((D, 1 << k), dtype=np.int64)
    _lib().vc_eq_table(D, k, r.ctypes.data, out.ctypes.data)
    return out


def mle(D, t, r):
    """t: numpy base (2^k,) or ext (D, 2^k); r (D, k) -> (D,)"""
    t, r = _c64(t), _c64(r)
    out = np.zeros(D, dtype=np.int64)
    _lib().vc_mle(D, r.shape[1], int(t.ndim == 1), t.ctypes.data, r.ctypes.data, out.ctypes.data)
    return out


def vmul(D, a, b):
    a, b = _c64(a), _c64(b)
    n = b.shape[1]
    out = np.zeros((D, n), dtype=np.int64)
    _lib().vc_vmul(D, n, int(a.ndim == 1), a.ctypes.data, b.ctypes.data, out.ctypes.data)
    return out


def dot(D, a, b):
    a, b = _c64(a), _c64(b)
    out = np.zeros(D, dtype=np.int64)
    _lib().vc_dot(D, b.shape[1], int(a.ndim == 1), a.ctypes.data, b.ctypes.data, out.ctypes.data)
    return out
