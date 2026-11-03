"""Proof of F_O (profile O) for a Qwen2-style model.

Units per layer (forward order) and their commitment groups:
  in    Requant8 + NormScale of h, Q/K/V by ShiftQ8S, RoPE8          G_in
  att.g attention of KV group g: tiles of P, key gather, N blocks    G_att.g
  ao    softmax normalisation, Requant8(o), O-proj, SatAdd           G_ao
  mi    Requant8 + NormScale of h_mid, per-row MLP exponents         G_mi
  mlp.s MLP slice s of d_ff: ShiftLUT(SiLU), ShiftQ8(up), Requant8(m) G_mlp.s
  mo    down-projection shift, SatAdd                                G_mo
The prove pass visits units in reverse forward order, so every claim on a
committed tensor is known when its group is opened (a group is owned by the
forward-earliest unit that consumes it)."""
import contextlib
import re
import os
import time
import numpy as np
import torch
from .field import P
from .poly import Poly, lparam
from . import domain as dm
from .domain import Domain
from .sumcheck import Tab, FULL, ROW, COL, Prog
from . import model as mdl
from . import tables as tb
from .proto import Ctx, nbits, const_pt, cat, eq_point
from . import logup
from . import gkr
from . import kern
from . import mlepub
from .kern import ptr
from .ext import dev as _dev

# float32 matmuls compute exact integer products in this file (sums < 2^24): never TF32
torch.backends.cuda.matmul.allow_tf32 = False

INV2 = (P + 1) // 2
TILE = 7                      # 128 x 128 attention tiles
XMIN = (1 << 23) - 1          # tau_exp(XMIN) = 0: lookup input of masked entries
ATT_HIST_TPB = int(os.environ.get("ZK_ATT_HIST_TPB", "8"))  # tiles per block when counting
GRIND_LOOKUP = 11


def log2(x):
    l = int(x).bit_length() - 1
    assert 1 << l == x, x
    return l


_F32K = {}
_F32T = {torch.int64: "f32_i64", torch.int32: "f32_i32", torch.int16: "f32_i16", torch.int8: "f32_i8",
         torch.uint8: "f32_u8", torch.bool: "f32_u8", torch.float32: "f32_f32", torch.float64: "f32_f64"}


def f32(x):
    """x (any integer/float/bool dtype, any strides) -> flat int32 x mod P, in one pass on the GPU."""
    if not x.is_cuda or x.numel() == 0:
        return (x.to(torch.int64) % P).to(torch.int32).reshape(-1)
    if not _F32K:
        m = kern.file_module("f32.cu", 4)
        _F32K.update({t: m.get_function(n) for t, n in _F32T.items()})
    if x.dim() > 4:
        x = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
    if x.dtype == torch.bool:
        x = x.view(torch.uint8)
    sz = [1] * (4 - x.dim()) + list(x.shape)
    st = [0] * (4 - x.dim()) + list(x.stride())
    n = x.numel()
    out = torch.empty(n, dtype=torch.int32, device=x.device)
    kern.launch(_F32K[x.dtype], int(min((n + 255) // 256, 188 * 16)), 256,
                [ptr(x), np.int64(n), np.int64(sz[1]), np.int64(sz[2]), np.int64(sz[3]),
                 np.int64(st[0]), np.int64(st[1]), np.int64(st[2]), np.int64(st[3]), ptr(out)])
    return out


def rm(X, cb):
    """(T, N) -> field, block-major [N >> cb][T][2^cb]."""
    T, N = X.shape
    return f32(X.reshape(T, N >> cb, 1 << cb).permute(1, 0, 2))


def hexpr(s, o):
    """SatAdd output from the pre-saturation sum s and flag o."""
    o2 = o * o
    return (1 - o2) * s + (o2 + o) * (INV2 * ((1 << 23) - 1)) - (o2 - o) * (INV2 * (1 << 23))


def satlookup(s, o):
    o2 = o * o
    return (1 - o2) * (s + (1 << 23)) + o2 * (o * s - (1 << 23) - (o2 - o) * INV2)


class Shp:
    def __init__(self, cfg, B, S, cb, cbv, ngen):
        self.cfg = cfg
        self.B, self.S, self.T = B, S, B * S
        self.t, self.ls, self.lb = log2(B * S), log2(S), log2(B)
        self.cb, self.cbv = cb, cbv
        d, dh = cfg.d, cfg.dh
        self.dh, self.ldh = dh, log2(dh)
        assert dh == 1 << TILE and (1 << cb) % dh == 0
        self.nh, self.nkv = cfg.nh, cfg.nkv
        self.grp = cfg.nh // cfg.nkv
        self.hb = (1 << cb) // dh
        self.Kd = d >> cb
        self.Kq = (cfg.nh * dh) >> cb
        self.Kk = (cfg.nkv * dh) >> cb
        self.Kf = cfg.dff >> cb
        self.Kv = cfg.V >> cbv
        assert self.Kd << cb == d and self.Kf << cb == cfg.dff and self.Kv << cbv == cfg.V
        self.nt1 = S >> TILE
        self.ltile = log2(self.nt1)
        self.ntile = self.nt1 * (self.nt1 + 1) // 2
        self.nbb = max(0, self.ls - mdl.KEYBLK)
        self.ng, self.lng = ngen, log2(ngen)
        self.R = B * ngen
        # tile list per sequence: (a, b) with b <= a, index a(a+1)/2 + b
        self.tiles = [(a, b) for a in range(self.nt1) for b in range(a + 1)]


def block_bits(cfg, cb_max=10, cbv_max=9):
    """Column-block sizes of the paper's runs: the largest cb <= cb_max for which every width the
    layout splits (d, Q, K / V, d_ff) is a multiple of 2^cb, and the largest cbv <= cbv_max dividing V."""
    widths = [cfg.d, cfg.nh * cfg.dh, cfg.nkv * cfg.dh, cfg.dff]
    cb = max(c for c in range(TILE, cb_max + 1) if all(w % (1 << c) == 0 for w in widths))
    cbv = max(c for c in range(0, cbv_max + 1) if cfg.V % (1 << c) == 0)
    return cb, cbv


# =================================================================== witness
def _imm(A, B):
    """Exact integer product (|entries| sums < 2^53)."""
    return (A.double() @ B.double()).round().to(torch.int64)


def _pad2(x):
    n = 1 << nbits(x.numel())
    return x if n == x.numel() else torch.cat([x, torch.zeros(n - x.numel(), dtype=x.dtype, device=x.device)])


def qk_exact(A, B):
    """A B^T for |entries| <= 180, dim 128: bf16 inputs are exact (< 2^8) and every fp32
    partial sum is an integer below 2^22, so the tensor-core product is exact."""
    if A.is_cuda:
        mdl.check(torch.maximum(A.abs().max(), B.abs().max()) <= 256, "qk_exact: entries above 256")
        return torch.mm(A.to(torch.bfloat16), B.to(torch.bfloat16).T, out_dtype=torch.float32)
    return A.float() @ B.float().T


def pv_int(Pm, V):
    """Exact P (in [0, 2^10]) V (int8) by two int8 GEMMs (int64 result)."""
    m, k = Pm.shape
    kp = (k + 31) // 32 * 32
    Vt = V.to(torch.int8)
    if kp != k:
        Vt = torch.cat([Vt, torch.zeros((kp - k, V.shape[1]), dtype=torch.int8, device=V.device)])
    Vt = Vt.T.contiguous().T
    out = torch.empty((m, V.shape[1]), dtype=torch.int64, device=V.device)
    for r0 in range(0, m, 8192):
        Pr = Pm[r0:r0 + 8192]
        if kp != k:
            Pr = torch.cat([Pr, torch.zeros((Pr.shape[0], kp - k), dtype=Pr.dtype, device=Pr.device)], dim=1)
        mm = Pr.shape[0]
        if mm < 32:
            Pr = torch.cat([Pr, torch.zeros((32 - mm, kp), dtype=Pr.dtype, device=Pr.device)])
        hi = torch._int_mm((Pr >> 5).to(torch.int8), Vt)
        lo = torch._int_mm((Pr & 31).to(torch.int8), Vt)
        out[r0:r0 + mm] = (hi.to(torch.int64) * 32 + lo)[:mm]
    return out


def _limbs(f, sig, asel, s, quarter=False):
    """Minimality witness w >= 0.  Requant8 (|a| < 2^23): w = 4 a - 255 s (two 15-bit limbs).
    ShiftQ8 (quarter: |acc| < 2^29, s >= 4): the same inequality divided by 4, w = acc - 255 s / 4
    (two 14-bit limbs), so that the relation stays below p (4 acc alone reaches 2^31)."""
    if quarter:
        w = f * (sig * (asel - 255 * (s >> 2)) + (1 - sig) * (-asel - 255 * (s >> 2) - 1))
        mdl.check((w.min() >= 0) & (w.max() < (1 << 28)), "shiftq8: minimality witness range")
        return w & 16383, w >> 14
    w = f * (sig * (4 * asel - 255 * s) + (1 - sig) * (-4 * asel - 255 * s - 1))
    mdl.check((w.min() >= 0) & (w.max() < (1 << 30)), "requant: minimality witness range")
    return w & 32767, w >> 15


def _jbits(j, cb, K):
    return [(j >> i) & 1 for i in range(cb)] + [((j >> cb) >> i) & 1 for i in range(nbits(K))]


def _compact(v):
    """int64 tensors -> int32 (all witness values are below 2^31 in magnitude)."""
    if torch.is_tensor(v):
        return v.to(torch.int32) if v.dtype == torch.int64 else v
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_compact(x) for x in v]
    return v


def qk_headrows(X, KX, hb, dh):
    """(T, KX hb dh) in the column-block layout -> head rows (KX T hb, dh), ordered (hib, token, block)."""
    T = X.shape[0]
    return X.reshape(T, KX, hb, dh).permute(1, 0, 2, 3).reshape(-1, dh)


def qk_unheadrows(Y, T, KX, hb, dh):
    return Y.reshape(KX, T, hb, dh).permute(1, 0, 2, 3).reshape(T, KX * hb * dh)


def qk_headidx(sh, KX):
    """Head index of every head row (hib, token, block)."""
    hib = torch.arange(sh.hb, device=_dev()).repeat(KX * sh.T)
    kb = torch.arange(KX, device=_dev()).repeat_interleave(sh.T * sh.hb)
    return kb * sh.hb + hib


def rq_pack(a, cb, e0=0, emax=17):
    """Requant8 / ShiftQ8 row quantization with its per-row witnesses."""
    q, e, w = mdl._rowq(a, e0, emax)
    s = mdl.pow2(e)
    wl, wh = _limbs(w["f"], w["sigma"], w["asel"], s, quarter=(e0 == 2))
    K = a.shape[1] >> cb
    return dict(q=q, e=e, s=s, f=w["f"], sig=w["sigma"], wl=wl, wh=wh, asel=w["asel"],
                jb=_jbits(w["jstar"], cb, K))


def ns_pack(q):
    qp, Eh, w = mdl.normscale(q)
    return dict(qp=qp, Eh=Eh, Ep=w["Ep"], pEp=mdl.pow2(w["Ep"]), pEn=mdl.pow2(w["En"]),
                pEh=tb.p2t(Eh).to(torch.int64), delta=w["delta"], rho=w["rho"], mu=w["mu"], sig=w["sig"],
                ng=(Eh < 0).to(torch.int64))


_SRROWS = {}


def sr_pack(k):
    rows = _SRROWS.get(k.device)
    if rows is None:
        rows = _SRROWS[k.device] = torch.tensor([tb.sr_row(x) + tb.sra_row(x)[1:]
                                                 for x in range(tb.SR_KMIN, tb.SR_KMAX + 1)],
                                                dtype=torch.int64, device=k.device)
    r = rows[(k - tb.SR_KMIN).reshape(-1)]
    return dict(k=r[:, 0], s=r[:, 1], R=r[:, 2], tR=r[:, 3], tB=r[:, 4], oB=r[:, 5], tA=r[:, 6], oA=r[:, 7],
                f=r[:, 8])


SR_F = ("k", "s", "R", "tR", "tB", "oB")
# shifts that may be left shifts (k <= 0) also carry the input bound of tables.SRA
SRA_F = ("tA", "oA", "f")
SRX_F = SR_F + SRA_F
RQ_F = ("e", "s", "f", "sig", "wl", "wh")
QK_F = ("kq", "pkq", "fq")          # QK-norm head row: shift max(kn + E2, 1), 2^shift, [kn + E2 < 1]
EG_F = ("E", "pE", "f")             # gate row exponent max(E_h, K_gl), 2^E_g, [E_h < K_gl]
NS_F = ("Eh", "Ep", "pEp", "pEn", "pEh", "delta", "rho", "mu", "ng")
ND_F = ("delta", "rhoD", "recip", "fk")
# sampling (theta > 0), per generated token: sums Z, A = sum_{j<y} p, P = p_y; 10-bit limbs of Z,
# of the remainder q and of w = floor(u Z / 2^30) (q, w = limbs 0-2, 3-5 of u Z), carries of the
# limb product, and 15-bit limbs of w - A and A + P - 1 - w
SAMP_F = ("Z", "A", "P", "z0", "z1", "z2", "q0", "q1", "q2", "w0", "w1", "w2",
          "k0", "k1", "k2", "k3", "k4", "d1l", "d1h", "d2l", "d2h")


def _l10(x):
    return [(x >> (10 * i)) & 1023 for i in range(3)]


def samp_wit(z, zmax, jz, y, lt, u, theta, sh):
    """Witness of the inverse-CDF sampling gadget for logits z (R, V) and targets y."""
    p = mdl.samp_table(theta)[zmax.unsqueeze(1) - z]
    Z = p.sum(dim=1)
    A = (p * lt).sum(dim=1)
    Pp = p.gather(1, y.unsqueeze(1)).squeeze(1)
    u = u.to(_dev()).to(torch.int64)
    X = u * Z
    w, q = X >> mdl.U_BITS, X & ((1 << mdl.U_BITS) - 1)
    assert bool(torch.all((A <= w) & (w < A + Pp))), "targets are not the policy's sample"
    ul, zl = _l10(u), _l10(Z)
    t = _l10(q) + _l10(w)
    out = dict(p=p, jb=_jbits(jz, sh.cbv, sh.Kv), Z=Z, A=A, P=Pp)
    for i in range(3):
        out[f"z{i}"], out[f"q{i}"], out[f"w{i}"] = zl[i], t[i], t[3 + i]
    carry = torch.zeros_like(Z)
    for kk in range(5):
        c = sum(ul[i] * zl[kk - i] for i in range(3) if 0 <= kk - i < 3) + carry
        assert bool(torch.all((c & 1023) == t[kk]))
        carry = c >> 10
        out[f"k{kk}"] = carry
    assert bool(torch.all(carry == t[5]))
    d1, d2 = w - A, A + Pp - 1 - w
    out["d1l"], out["d1h"], out["d2l"], out["d2h"] = d1 & 32767, d1 >> 15, d2 & 32767, d2 >> 15
    return out


WITT = {}


def _witt(key=None, _st=[None]):
    """ZK_WITT=1: synchronized section timers of LayerWit (accumulated in WITT)."""
    if not os.environ.get("ZK_WITT"):
        return
    torch.cuda.synchronize()
    t = time.time()
    if key is not None and _st[0] is not None:
        WITT[key] = WITT.get(key, 0.0) + t - _st[0]
    _st[0] = t


def layer_lparams(M, l):
    """Layer l's constants in poly.LPARAM_NAMES order: format constants Kco, Kcd, the first tag of
    the layer's dynamic lookup sets (one per KV group) and the tag of an MoE layer's slot multiset."""
    Ly, cfg = M.layers[l], M.cfg
    Kcd = 0 if cfg.moe else -(Ly.eWu + M.eA + Ly.eWd + M.f_h)     # MoE: the down shift is per token
    return [Ly.Kco, Kcd, tb.DYN0 + l * cfg.nkv, tb.DYN0 + cfg.L * cfg.nkv + l]


def cap_col(ctx, db, name, like, getcap, cap_host):
    """A layer constant K as a constant public column of a per-row domain (shaped like the row column
    like()); its MLE is K at every point, and the verifier takes K from its own model (bound by h_circ)."""
    def fn(pt):
        v = np.zeros((ctx.D, 1), dtype=np.int64)
        v[0, 0] = cap_host() % P
        return v
    return db.col(name, lambda: f32(torch.zeros_like(like()) + getcap()), R_pub(ctx, fn))


def wit_lparams(M, l, out=None):
    """layer_lparams as a device int64 (2,), copied into out when given (no host sync)."""
    h = torch.tensor(layer_lparams(M, l)[:2], dtype=torch.int64).pin_memory()
    if out is None:
        return h.to("cuda", non_blocking=True)
    return out.copy_(h, non_blocking=True)


class _LayerSlot:
    """Fixed device inputs of a captured layer: one GPU copy of the layer weights, the boundary
    (s2, os2) of the layer below and the witness constants, refilled per layer (fill) before a
    replay.  Installs itself as M.layers[l] and Proto.boundary's source; close() restores."""

    def __init__(self, P, c):
        self.P, M = P, P.M
        assert len({Ly.E for Ly in M.layers}) == 1, "exp table differs per layer"
        self.orig = list(M.layers)
        hostL = P._host_layers
        self.src = (lambda l: hostL[l]) if hostL is not None else (lambda l: self.orig[l])
        if hostL is not None:
            for i in range(len(M.layers)):
                M.layers[i] = hostL[i]
        self.slot = slot = mdl.QLayer()
        Lc = self.src(c)
        for k, v in vars(Lc).items():
            if not k.startswith("_"):
                setattr(slot, k, torch.empty(v.shape, dtype=v.dtype, device=_dev()) if torch.is_tensor(v) else v)
        slot._vsh = [0] * Lc.vshift.numel()            # verifier-side constants: not used by the prover
        slot._vsh_dev = torch.empty(Lc.vshift.numel(), dtype=torch.int64, device=_dev())
        s2h, os2h = P.bnd[c - 1]
        self.gb = P._gbnd = (c - 1, torch.empty(s2h.shape, dtype=s2h.dtype, device=_dev()),
                             torch.empty(os2h.shape, dtype=os2h.dtype, device=_dev()))
        if getattr(P, "_wlp", None) is None:
            P._wlp = torch.zeros(2, dtype=torch.int64, device=_dev())
        mdl._WT.clear()                 # cached weight layouts of the slot are recomputed in the graph
        self.gok = torch.ones((), dtype=torch.bool, device=_dev())     # outside the capture

    def fill(self, l):
        P, slot, Ls = self.P, self.slot, self.src(l)
        for k, v in vars(Ls).items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                d = getattr(slot, k)
                assert d.shape == v.shape and d.dtype == v.dtype, k
                d.copy_(v, non_blocking=True)
            else:
                setattr(slot, k, v)
        slot._vsh_dev.copy_(Ls.vshift, non_blocking=True)
        P.M.layers[l] = slot
        s2h, os2h = P.bnd[l - 1]
        self.gb[1].copy_(s2h, non_blocking=True)
        self.gb[2].copy_(os2h, non_blocking=True)
        wit_lparams(P.M, l, P._wlp)

    @contextlib.contextmanager
    def checks(self):
        """Witness checks queued inside a capture (entered inside it): and-ed into one flag in the
        graph, so every replay's checks accumulate."""
        npend = len(mdl._PENDING)
        yield
        new = mdl._PENDING[npend:]
        del mdl._PENDING[npend:]
        if new:
            self.gok.logical_and_(torch.cat([x for x, _ in new]).all())
            mdl._PENDING.append((self.gok.reshape(1), "witness checks of the graph-replayed layers"))

    def close(self):
        self.P._gbnd = None
        for i, Ly in enumerate(self.orig):
            self.P.M.layers[i] = Ly


_LAYER_NAME = re.compile(r"^\d+\.")


def _tmap(o, fn, kf=None):
    """Copy of nested lists / tuples / dicts with their device leaves (tensors, lazy arrays) mapped
    by fn and lazy arrays' draw keys by kf."""
    from .lazy import Lz
    if isinstance(o, Lz):
        return Lz(fn(o.dev), keys=kf(o.keys) if kf is not None and o.keys is not None else o.keys)
    if torch.is_tensor(o):
        return fn(o)
    if isinstance(o, dict):
        return {k: _tmap(v, fn, kf) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_tmap(v, fn, kf) for v in o)
    return o


def _tleaves(o):
    """The device tensors of a nested structure, in _tmap order."""
    out = []
    _tmap(o, lambda t: out.append(t) or t)
    return out


def _tsig(o):
    """Structure of a nested claim list as the captured graph sees it: shapes and dtypes of device
    leaves, host leaves' values, which lazy points share draw keys."""
    from .lazy import Lz
    ids = {}

    def walk(o):
        if isinstance(o, Lz):
            ks = None if o.keys is None else tuple(ids.setdefault(k, len(ids)) for k in o.keys)
            return ("L", o.shape, ks)
        if torch.is_tensor(o):
            return ("T", tuple(o.shape), str(o.dtype))
        if isinstance(o, dict):
            return ("d",) + tuple((k, walk(v)) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            return ("l",) + tuple(walk(v) for v in o)
        if isinstance(o, np.ndarray):
            return ("n", o.shape, o.tobytes())
        return ("o", repr(o))
    return walk(o)


class LayerWit:
    """All integer intermediates of one layer (prover)."""

    def __init__(self, M, sh, l, hin, hist=False, lp=None):
        """hist: also count the softmax lookup inputs per head (commit pass, see step_att).
        lp: layer l's format constants (Kco, Kcd) as a device int64 (2,) (default: from M)."""
        self.want_hist = hist
        if lp is None:
            lp = wit_lparams(M, l)
        with mdl.deferred_checks():
            self._build(M, sh, l, hin, lp)

    def _build(self, M, sh, l, hin, lp):
        _witt()
        cfg = M.cfg
        Ly = M.layers[l]
        self.l = l
        T, nh, nkv, dh = sh.T, cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        self.hin = hin
        # attention input
        self.r1 = rq_pack(hin, sh.cb)
        self.n1 = ns_pack(self.r1["q"])
        mdl.check(self.n1["Eh"].max() <= Ly.kqs_t, "F_O: row exponent beyond the ShiftQ8S bound")
        qp = self.n1["qp"]
        self.acc_qkv = mdl.imm32(qp, Ly.Wqkv)
        if cfg.qk_norm:
            self.qk_norm_wit(Ly, sh)
        else:
            self.qkv, w = mdl.shiftq8s_fused(self.acc_qkv, self.n1["Eh"], Ly.kc, Ly.bias)
            self.oq, self.whq = w["o"], w["wh"]
        _witt("qkv")
        cos, sin = mdl.rope_tables(cfg, sh.S)
        pos = torch.arange(T, device=_dev()) % sh.S
        self.cos, self.sin = cos[pos], sin[pos]             # (T, 64)
        half = dh // 2
        Q = self.qkv[:, :nq + nk].reshape(T, nh + nkv, dh)
        r1, r2 = mdl.rope8(Q[..., :half], Q[..., half:], self.cos.unsqueeze(1), self.sin.unsqueeze(1))
        rQK = torch.cat([r1, r2], dim=-1)                  # (T, nh + nkv, dh)
        self.rQ = rQK[:, :nh].reshape(T, nq)
        self.rK = rQK[:, nh:].reshape(T, nk)
        self.V = self.qkv[:, nq + nk:]
        # exactness bound of the bf16 QK^T products (qk_exact)
        mdl.check(torch.maximum(self.rQ.abs().max(), self.rK.abs().max()) <= 256, "RoPE8 output above 256")
        # attention (per group, kept compact)
        _witt("rope")
        self.att = [None] * nkv
        self.texp = mdl.exp_table(Ly.E, dh)
        mdl.check(self.texp[XMIN] == 0, "tau_exp(XMIN) != 0")
        Ns = torch.zeros((T, nh, dh), dtype=torch.int64, device=_dev())
        Dd = torch.zeros((T, nh), dtype=torch.int64, device=_dev())
        for g in range(nkv):
            a = self.att_group(M, sh, g)
            for i in range(sh.grp):
                Ns[:, g * sh.grp + i] = a["Nr"][i].sum(dim=1)
                Dd[:, g * sh.grp + i] = a["D"][i]
            self.att[g] = a
        self.Ns, self.D = Ns, Dd
        _witt("att")
        w = mdl.norm_den(Dd)
        vsh = Ly.vshift[:nh].to(device=_dev(), dtype=torch.int64)
        o, w2 = mdl.norm_div(Ns, w, vsh)
        xh, tt = w2["xh"], w2["t"]
        w.update(recip=w2["recip"], kp=w2["kp"])
        self.nd = {key: v.T.contiguous() for key, v in w.items()}     # (nh, T)
        self.ndxh, self.ndt = xh.reshape(T, nq), tt.reshape(T, nq)
        self.o = o.reshape(T, nq)
        self.ro = rq_pack(self.o, sh.cb)
        self.acc_o = mdl.imm(self.ro["q"], Ly.Wo)
        self.ko = lp[0] - self.ro["e"]
        c1 = mdl.gshiftround(self.acc_o, self.ko.unsqueeze(-1))
        self.s1 = hin + c1
        self.hmid, w = mdl.satadd(hin, c1)
        self.os1 = w["o"]
        _witt("normdiv+oproj")
        # MLP
        self.r2 = rq_pack(self.hmid, sh.cb)
        self.n2 = ns_pack(self.r2["q"])
        if cfg.moe:
            from . import moe
            self.moe = moe.moe_wit(M, Ly, self.n2["qp"], self.n2["Eh"], moe.Geo(cfg, sh))
            self.accd, self.kd = self.moe["accd"], self.moe["kd"]
            c2 = mdl.gshiftround(self.accd, self.kd.unsqueeze(-1))
            self.s2 = self.hmid + c2
            self.hout, w = mdl.satadd(self.hmid, c2)
            self.os2 = w["o"]
            _witt("moe")
            for k, v in vars(self).items():
                if k not in ("hout", "moe"):
                    setattr(self, k, _compact(v))
            return
        mdl.check(self.n2["Eh"].max() <= Ly.kgm_t, "F_O: row exponent beyond the gate shift bound")
        qp2 = self.n2["qp"]
        self.accg = mdl.imm32(qp2, Ly.Wg)
        Eg = torch.maximum(self.n2["Eh"], Ly.kgl_t)           # E_g = max(E_h, K_gl): the gate's row exponent
        self.eg = dict(E=Eg, pE=tb.p2t(Eg).to(torch.int64), f=(self.n2["Eh"] < Ly.kgl_t).to(torch.int64))
        kg = (M.eG - Ly.eWg).unsqueeze(0) + Eg.unsqueeze(-1)
        mdl.check((kg.min() >= 1) & (kg.max() <= mdl.SILU_KMAX), "SiLU shift range")
        self.c, self.a = mdl.shiftround_silu(self.accg, Eg, M.eG - Ly.eWg, M.eG, M.eA)
        self.accu = mdl.imm32(qp2, Ly.Wu)
        self.ru = rq_pack(self.accu, sh.cb, 2, 22)
        self.m = self.a * self.ru["q"]
        if cfg.down_b == 127:
            self.rm = rq_pack(self.m, sh.cb)
            self.accd = mdl.imm(self.rm["q"], Ly.Wd)
        else:
            from .moe import rq_pack_b
            self.rm = rq_pack_b(self.m.to(torch.int64), sh.cb, 0, 17, cfg.down_b)
            self.accd = mdl.imm_wide(self.rm["q"], Ly.Wd)
        self.kd = lp[1] - (self.rm["e"] + self.ru["e"] - self.n2["Eh"])
        c2 = mdl.gshiftround(self.accd, self.kd.unsqueeze(-1))
        self.s2 = self.hmid + c2
        self.hout, w = mdl.satadd(self.hmid, c2)
        self.os2 = w["o"]
        _witt("mlp")
        for k, v in vars(self).items():
            if k != "hout":
                setattr(self, k, _compact(v))
        _witt("compact")

    def qk_norm_wit(self, Ly, sh):
        """QK-norm (Qwen3): per Q / K head row of the accumulator, ShiftQ8 (8-bit, row exponent),
        NormScale, then ShiftQ8S of q' G by kn + E2 (no bias); V by ShiftQ8S as without QK-norm.
        Head rows are in the order (head in block, token, block) of the column-block layout."""
        nq, nk, dh = sh.nh * sh.dh, sh.nkv * sh.dh, sh.dh
        acc = self.acc_qkv.to(torch.int64)
        self.qn, outs = {}, []
        for X, c0, n, KX in (("qn", 0, nq, sh.Kq), ("kn", nq, nk, sh.Kk)):
            H = qk_headrows(acc[:, c0:c0 + n], KX, sh.hb, dh)
            r = rq_pack(H, sh.ldh, 2, 22)
            ns = ns_pack(r["q"])
            head = qk_headidx(sh, KX)
            G = Ly.Gqk[c0:c0 + n].view(-1, dh).to(torch.int64)
            kn = Ly.kn[c0:c0 + n].view(-1, dh)[:, 0].to(torch.int64)
            knr = kn[head]
            kv = knr + ns["Eh"]
            mdl.check(kv.max() <= mdl.QKV_KMAX, "F_O: QK-norm shift above QKV_KMAX")
            k = kv.clamp(min=1)                                   # kn + E2 below 1: shifted by 1
            q, w = mdl.shiftq8s(ns["qp"] * G[head], k.unsqueeze(-1), torch.zeros_like(k).unsqueeze(-1))
            un = lambda Y: qk_unheadrows(Y, sh.T, KX, sh.hb, dh)
            self.qn[X] = dict(r=r, n=ns, knr=knr, q8=un(r["q"]), qp=un(ns["qp"]), o=un(w["o"]), wh=un(w["wh"]),
                              kq=k, pkq=tb.p2t(k).to(torch.int64), fq=(kv < 1).to(torch.int64))
            outs.append(un(q))
        v, w = mdl.shiftq8s_fused(self.acc_qkv[:, nq + nk:], self.n1["Eh"], Ly.kc[nq + nk:], Ly.bias[nq + nk:])
        self.oq, self.whq = w["o"], w["wh"]
        self.qkv = torch.cat(outs + [v.to(torch.int64)], dim=1)

    def att_group(self, M, sh, g):
        """P tiles, gathered keys and blocked numerators of KV group g."""
        T, S, B, dh = sh.T, sh.S, sh.B, sh.dh
        rK = self.rK[:, g * dh:(g + 1) * dh]
        V = self.V[:, g * dh:(g + 1) * dh]
        nt1 = sh.nt1
        ta, tbb = _tile_index(sh)
        out = dict(P=[], js=[], Kg=[], Nr=[], D=[], M=[])
        mk = torch.zeros(T, dtype=torch.int64, device=_dev())
        texp16 = self.texp.to(torch.int16).contiguous()
        if S >= 128 and not os.environ.get("ZK_ATT_OLD"):
            return self._att_group_fused(sh, g, rK, V, ta.to(torch.int32), tbb.to(torch.int32), texp16, out, mk)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=_dev()))
        for i in range(sh.grp):
            hh = g * sh.grp + i
            rQ = self.rQ[:, hh * dh:(hh + 1) * dh]
            Pt, js, Mx = [], [], []
            Nr = torch.zeros((T, max(1, 1 << sh.nbb), dh), dtype=torch.int64, device=_dev())
            Dh = torch.zeros(T, dtype=torch.int64, device=_dev())
            for b in range(B):
                rows = slice(b * S, (b + 1) * S)
                accf = qk_exact(rQ[rows], rK[rows])
                Mx_b, j_b = accf.masked_fill(~causal, float("-inf")).max(dim=1)
                x = (Mx_b.unsqueeze(1) - accf).to(torch.int32).clamp_(0, XMIN).masked_fill_(~causal, XMIN)
                del accf
                Pm = texp16[x]
                del x
                Dh[rows] = Pm.sum(dim=1, dtype=torch.int64)
                km = mdl.norm_den(Dh[rows])["km"].unsqueeze(-1)
                Vb = V[rows]
                for kb in range(max(1, 1 << sh.nbb)):
                    j0, j1 = kb << mdl.KEYBLK, min(S, (kb + 1) << mdl.KEYBLK)
                    Nr[rows, kb] = mdl.shiftround(pv_int(Pm[:, j0:j1], Vb[j0:j1]), km)
                Pt.append(Pm.reshape(nt1, 128, nt1, 128)[ta, :, tbb, :])   # (ntile, 128, 128) int16
                del Pm
                js.append(j_b + b * S)
                Mx.append(Mx_b.to(torch.int64))
            js = torch.cat(js)
            mk.scatter_add_(0, js, torch.ones_like(js))
            out["P"].append(torch.stack(Pt))              # (B, ntile, 128, 128)
            out["js"].append(js)
            out["Kg"].append(rK[js])
            out["Nr"].append(Nr)
            out["D"].append(Dh)
            out["M"].append(torch.cat(Mx))
        out["mk"] = mk
        return out

    def _att_group_fused(self, sh, g, rK, V, ta, tbb, texp16, out, mk):
        """All heads of KV group g, HB heads per pass (one score GEMM, one row-max and one tile
        launch per pass and sequence; PV limb GEMMs batched over the pass's heads)."""
        T, S, B, dh, grp = sh.T, sh.S, sh.B, sh.dh, sh.grp
        lkb = min(sh.ls, mdl.KEYBLK)
        KB, nkb = 1 << lkb, S >> lkb
        f = _attw()
        HB = max(1, min(grp, int(ATT_SCORE_BYTES // (4 * S * S))))
        nt = ta.numel()
        Pall = torch.empty((grp, B, nt, 128, 128), dtype=torch.int16, device=_dev())
        jsl = torch.empty((grp, T), dtype=torch.int64, device=_dev())
        Dall = torch.zeros((grp, T), dtype=torch.int64, device=_dev())
        Mall = torch.empty((grp, T), dtype=torch.int64, device=_dev())
        Nall = torch.empty((grp, T, nkb, dh), dtype=torch.int64, device=_dev())
        hist = torch.zeros((grp, 1 << 23), dtype=torch.int32, device=_dev()) if self.want_hist else None
        phi = torch.zeros((nkb, HB, S, KB), dtype=torch.int8, device=_dev())
        plo = torch.zeros((nkb, HB, S, KB), dtype=torch.int8, device=_dev())
        hib = torch.empty((nkb, HB * S, dh), dtype=torch.int32, device=_dev())
        lob = torch.empty((nkb, HB * S, dh), dtype=torch.int32, device=_dev())
        mx = torch.empty(HB * S, dtype=torch.float32, device=_dev())
        Kbf = rK.to(torch.bfloat16)
        for h0 in range(0, grp, HB):
            hb = min(HB, grp - h0)
            hh0 = g * grp + h0
            Qbf = self.rQ[:, hh0 * dh:(hh0 + hb) * dh].reshape(T, hb, dh).to(torch.bfloat16)
            for b in range(B):
                rows = slice(b * S, (b + 1) * S)
                # (hb, S, dh) x (dh, S) -> (hb * S, S): exact integers (qk_exact)
                q = Qbf[rows].transpose(0, 1).reshape(hb * S, dh)
                accf = torch.mm(q, Kbf[rows].T, out_dtype=torch.float32)
                kern.launch(f["att_rowmax_h"], (hb * S * 32 + 255) // 256, 256,
                            [ptr(accf), np.int32(S), np.int32(hb * S), ptr(mx),
                             np.uint64(jsl.data_ptr() + 8 * (h0 * T + b * S)), np.int64(T)])
                tpb = ATT_HIST_TPB if hist is not None else 1
                kern.launch(f["att_tiles_h"], ((nt + tpb - 1) // tpb, hb), 128,
                            [ptr(accf), np.int32(S), ptr(mx), ptr(ta), ptr(tbb), ptr(texp16),
                             np.uint64(Pall[h0, b].data_ptr()), np.int64(B * nt * 16384),
                             ptr(phi), ptr(plo), np.int32(lkb), np.int32(HB),
                             np.uint64(Dall.data_ptr() + 8 * (h0 * T + b * S)), np.int64(T),
                             np.uint64(hist[h0].data_ptr() if hist is not None else 0), np.int64(1 << 23),
                             np.int32(nt), np.int32(tpb)])
                Mall[h0:h0 + hb, rows] = mx[:hb * S].view(hb, S).to(torch.int64)
                del accf
                # k_m of norm_den(D) (its range checks run on the whole layer's D afterwards)
                Db = Dall[h0:h0 + hb, rows]
                km = (torch.frexp(Db.double()).exponent.to(torch.int64) - 16).clamp_(min=mdl.KM_MIN)
                Vk = V[rows].to(torch.int8).reshape(nkb, KB, dh)
                for kb in range(nkb):
                    vk = Vk[kb].T.contiguous().T
                    torch._int_mm(phi[kb, :hb].reshape(hb * S, KB), vk, out=hib[kb, :hb * S])
                    torch._int_mm(plo[kb, :hb].reshape(hb * S, KB), vk, out=lob[kb, :hb * S])
                hi_ = hib if hb == HB else hib[:, :hb * S].contiguous()
                lo_ = lob if hb == HB else lob[:, :hb * S].contiguous()
                kmc = km.contiguous()
                n = nkb * hb * S * dh
                kern.launch(f["att_nr_h"], int(min((n + 255) // 256, 188 * 16)), 256,
                            [ptr(hi_), ptr(lo_), np.int32(nkb), np.int32(hb * S), np.int32(S), np.int32(dh),
                             ptr(kmc), np.int64(S), np.int64(0),
                             np.uint64(Nall[h0].data_ptr() + 8 * b * S * nkb * dh), np.int64(T * nkb * dh)])
        for i in range(grp):
            out["P"].append(Pall[i])
            out["js"].append(jsl[i] + torch.arange(B, device=_dev()).repeat_interleave(S) * S)
            out["Kg"].append(rK[out["js"][-1]])
            out["Nr"].append(Nall[i])
            out["D"].append(Dall[i])
            out["M"].append(Mall[i])
        jall = torch.cat(out["js"])
        mk.scatter_add_(0, jall, torch.ones_like(jall))
        out["mk"] = mk
        if hist is not None:
            out["hist"] = hist
        return out


_ATTW = None


def _attw():
    global _ATTW
    if _ATTW is None:
        m = kern.file_module("attw.cu", 4)
        _ATTW = {n: m.get_function(n) for n in ("att_rowmax", "att_tiles", "att_split", "att_coldot",
                                                "att_rowmax_h", "att_tiles_h", "att_nr_h")}
    return _ATTW


def embed(M, tokens):
    return M.emb[tokens.reshape(-1).to(_dev())].to(torch.int64)


# =================================================================== builders
# committed zero padding of the attention probability columns, as a fraction of the tiles
P_PAD = 0.10
# prover: keep the transcript on the device through the prove pass, with claims, points and
# coefficients lazy on the device (nothing read back per layer); ZK_DEVSESSION=0: per-step device
# transcript segments with a host round trip each.
DEV_SESSION = bool(int(os.environ.get("ZK_DEVSESSION", "1")))
# prover: the middle layers' prove pass is captured once as a CUDA graph and replayed for every
# middle layer (Proto.graph_layers): no Python per layer, kernels launched back to back.  Qwen2.5-32B,
# 64 layers: 1x4K 133 -> 106 s, 4x4K 322 -> 310 s, 16K 503 -> 497 s; the capture costs ~1.5-2.2 s
# once (8-layer 4x4K/16K proofs lose ~1 s).  ZK_GRAPH=0 disables it; ZK_GRAPHLOG=1 times capture and
# replays (with syncs).  (Replaying the PCS passes P3-P5 the same way was tried: no gain, GPU-bound.)
GRAPH = bool(int(os.environ.get("ZK_GRAPH", "1")))
# one Merkle tree per size class for all of a layer's groups (gpcs.GPCS.merge): fewer paths per query
MERGE_TREES = bool(int(os.environ.get("ZK_MERGE", "1")))
# padding allowance when packing weight blocks into wide columns (fraction of a tensor)
W_PACK_PAD = float(os.environ.get("ZK_WPAD", "0.15"))
# softmax tiles: separate lower-degree domain for the off-diagonal tiles (see tile_sets)
ATT_SPLIT = bool(int(os.environ.get("ZK_ATT_SPLIT", "0")))
# fp32 score bytes per head-batched attention pass (witness)
ATT_SCORE_BYTES = float(os.environ.get("ZK_ATTB", str(4 << 30)))
# MLP slice size in units of 2^24 cells (GKR memory grows linearly with it)
MLP_SLICE = int(os.environ.get("ZK_MLPSL", "9"))
# KV groups whose small attention domains are batched together
ATT_MERGE = int(os.environ.get("ZK_ATTMG", "8"))
# softmax tile domains of several KV groups are merged up to this many lookup entries
TILE_MERGE_LEAVES = 1 << 28
# at most this many KV groups per merged tile domain (the domain kernel grows with its columns)
TILE_MERGE_G = int(os.environ.get("ZK_TILE_MERGE", "2"))
# CPU LRU of committed tensors so PCS P3–P5 skip LayerWit when they still fit.
# Off by default: a 16K layer is ~25 GB and host copies dominate prove time.
STASH_GB = int(os.environ.get("ZK_STASH_GB", "140"))
STASH_L = int(os.environ.get("ZK_STASH_L", "0"))

class DB:
    """Domain builder: columns with data (prover) and claim routes."""

    def __init__(self, ctx, name, n, K=1, merge=False):
        self.ctx, self.dom, self.routes = ctx, Domain(name, n, K, merge), {}

    def col(self, name, data, route, kind=FULL, c=0, shared=False):
        d = (data() if callable(data) else data) if self.ctx.P else None
        v = self.dom.col(name, d, kind, c, shared)
        self.routes[name] = route
        return v

    def run(self):
        q = getattr(self.ctx, "dom_batch", None)
        if q is not None:           # inside domain_batch(): proved when the batch closes
            q.append(self)
            return
        self.ctx.run_domain(self.dom, self.routes)


def merge_domains(dbs):
    """One Domain holding the columns, lookups and constraints of several domains of one shape
    (n, K, merge); column names are prefixed by the domain's position, routes follow them."""
    d0 = dbs[0].dom
    m = Domain("|".join(db.dom.name for db in dbs), d0.n, d0.K, d0.merge)
    routes = {}
    for j, db in enumerate(dbs):
        d = db.dom
        assert (d.n, d.K, d.merge) == (d0.n, d0.K, d0.merge)
        off = len(m.cols)
        for c in d.cols:
            m.col(f"{j}:{c.name}", c.data, c.kind, c.c, c.shared, c.role)
            routes[f"{j}:{c.name}"] = db.routes[c.name]

        def sh(poly, off=off):
            return poly.rename(lambda i: i + off)
        for tag, vals, num in d.lookups:
            m.lookups.append((sh(tag), [sh(v) for v in vals], None if num is None else sh(num)))
        m.dyn += d.dyn
        m.constraints += [sh(cp) for cp in d.constraints]
    return m, routes


class domain_batch:
    """Domains run inside the block are proved together when it closes: those of one shape
    (n, K, merge) as one merged domain (one GKR and one final sumcheck), in order of first
    appearance.  Nothing inside the block may read their claims."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        assert getattr(self.ctx, "dom_batch", None) is None
        self.ctx.dom_batch = []

    def __exit__(self, *exc):
        q, self.ctx.dom_batch = self.ctx.dom_batch, None
        if exc[0] is not None:
            return False
        groups, order = {}, []
        for db in q:
            k = (db.dom.n, db.dom.K, db.dom.merge)
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(db)
        for k in order:
            g = groups[k]
            if len(g) == 1:
                self.ctx.run_domain(g[0].dom, g[0].routes)
            else:
                dom, routes = merge_domains(g)
                self.ctx.run_domain(dom, routes)
        return False


def R_blk(ctx, name, bmap=None, pre=None, post=None):
    """Route domain blocks to tensor blocks; pre/post extend the point."""
    def f(pt, vals):
        p = pt
        if pre is not None:
            p = cat(pre, p)
        if post is not None:
            p = cat(p, post)
        if bmap is None:
            ctx.claim(name, p, vals)
        else:
            d = {}
            for b in range(vals.shape[1]):
                tb_ = bmap(b)
                if tb_ is not None:
                    d[tb_] = vals[:, b]
            ctx.claim(name, p, d)
    return f


def R_virt(ctx, key):
    def f(pt, vals):
        ctx.put(key, (pt, vals))
    return f


def R_pub(ctx, fn):
    """Verifier evaluates the public column: fn(pt) -> (D, K) or (D, 1)."""
    def f(pt, vals):
        if not ctx.P:
            exp = fn(pt)
            if not np.array_equal(np.asarray(exp) % P, vals % P):
                raise ValueError("public column mismatch")
    return f


def R_none(pt, vals):
    pass


def eq_host(D, r, n):
    """eq(r, b) for b < n as numpy (D, n), on the host (small tables: no device round trip)."""
    from .field import Ext
    E = Ext(D)
    r = np.asarray(r, dtype=np.int64) % P
    t = E.one(1)
    for j in range(r.shape[1]):
        rj = r[:, j:j + 1]
        hi = E.mul(t, rj)
        t = np.concatenate([E.sub(t, hi), hi], axis=1)
    return t[:, :n]


def mle_pub(ctx, closed, full):
    """A public column's MLE by its closed form (zk.mlepub); ZK_MLECHECK=1 also evaluates the full
    vector (full()) and checks they agree."""
    v = np.asarray(closed(), dtype=np.int64) % P
    if os.environ.get("ZK_MLECHECK"):
        ref = np.asarray(full(), dtype=np.int64) % P
        assert np.array_equal(v, ref), "closed-form MLE differs from the full evaluation"
    return v


def mle_host(ctx, data, pt):
    """MLE of a public base vector (torch, length 2^k) at pt -> numpy (D,)."""
    return np.asarray(ctx.G.mle(f32(data), pt), dtype=np.int64)


def pub_const_rows(ctx, vals_list):
    """Per-block constants -> (D, K)."""
    out = np.zeros((ctx.D, len(vals_list)), dtype=np.int64)
    out[0] = [int(v) % P for v in vals_list]
    return out


def check_eq(ctx, label, fin, a, b):
    """Verifier: the final of a shared eq(a, .) table (every block) equals eq(a, b)."""
    if ctx.P:
        return
    e = eq_point(ctx.F, a, b)
    for k in range(fin.shape[1]):
        if ctx.F.c(fin[:, k]) != e:
            raise ValueError(f"sumcheck {label!r}: eq table final")


def lz_scale(v, k):
    """A claimed value times a base constant: lazy device values stay on the device."""
    from .lazy import Lz
    if isinstance(v, Lz) or torch.is_tensor(v):
        d = v.dev if isinstance(v, Lz) else v
        return Lz((d.to(torch.int64) * (int(k) % P) % P).to(torch.int32))
    return (np.asarray(v, dtype=np.int64) * k) % P


def eqb_coefs(ctx, x):
    """(2x - 1, 1 - x) for eqb_expr: on the device for a lazy x (prover), else host values."""
    from .lazy import Lz
    if ctx.P and isinstance(x, Lz):
        d = x.dev.reshape(-1).to(torch.int64)
        one = torch.zeros_like(d)
        one[:1].fill_(1)
        return Lz(((2 * d - one) % P).to(torch.int32)), Lz(((one - d) % P).to(torch.int32))
    F = ctx.F
    a = F.sub(F.add(F.c(x), F.c(x)), F.one)
    b_ = F.sub(F.one, F.c(x))
    return np.array(a, dtype=np.int64), np.array(b_, dtype=np.int64)


def eqb_expr(cidx, a, b):
    """Prog fragment for b * x + (1 - b)(1 - x) given coef ids of (2x - 1) and (1 - x)."""
    return ("+", ("*", ("c", a), ("p", cidx)), ("c", b))


# =================================================================== sub-protocols
def oh_check(ctx, label, jb_name, nbit, claims_oh, t, data_bits):
    """claims_oh: list of (point (D, cb + t), vals (D, K)) on OH(m, i) = eq(bits(j*_m), i)
    over a row-major (T x K 2^cb) domain.  Reduces to claims on the bit planes."""
    cb = nbit[0]
    nbk = nbit[1]
    tau = [ctx.chal(nbk) for _ in claims_oh]
    # batch: each claim gives a separate product sumcheck sharing the row eq point? points differ
    for idx, (pt, vals) in enumerate(claims_oh):
        pj, pm = pt[:, :cb], pt[:, cb:]
        tv = tau[idx]
        claim = ctx.blockcomb(tv, vals)
        coefs, polys = [], []
        nb = cb + nbk
        expr = None
        xs = [pj[:, i] for i in range(cb)] + [tv[:, i] for i in range(nbk)]
        for i, x in enumerate(xs):
            coefs += list(eqb_coefs(ctx, x))
            polys.append(Poly.var(i))
            e = eqb_expr(i, 2 * i, 2 * i + 1)
            expr = e if expr is None else ("*", expr, e)
        prog = Prog(polys, expr, coefs)
        shapes = [(FULL, 0)] * nb
        rho, fin = ctx.sumcheck(label + b"-oh", t, 1, shapes, claim,
                                tabs=(lambda: [Tab(f32(b), FULL) for b in data_bits()]) if ctx.proving else None,
                                prog=prog, eq_r=pm)
        ctx.claim(jb_name, rho, {i: fin[i][:, 0] for i in range(nb)})


def matmul(ctx, label, A_name, A_data, Kin, Mrows, W_name, W_nat, Kout, cb, claims, cbo=None):
    """claims: list of (pt_o (D, cb + nbits(Kout)), pt_m (D, log Mrows), value) on
    acc = A W with A row-major (Mrows x Kin 2^cb) and W (Kin 2^cb x Kout 2^cb) natural.
    Weight tensor blocks (tb, ob), in-block (o' low, t' high)."""
    if not claims:
        return
    F, G = ctx.F, ctx.G
    cbo = cb if cbo is None else cbo
    Nin, Nout = Kin << cb, Kout << cbo
    gam = ctx.chal(len(claims))
    claim = ctx.claimsum(gam, [v for _, _, v in claims])
    terms = [(gam[:, i], [2 * i, 2 * i + 1]) for i in range(len(claims))]
    shapes = [(FULL, 0)] * (2 * len(claims))

    def tabs():
        out = []
        A = A_data()
        for po, pm, _ in claims:
            Af = G.coldot(A, G.eq_table(pm), Kin, Mrows, 1 << cb)
            eo = G.eq_table(po)[:, :Nout].contiguous()
            Wf = G.rowdot(W_nat.reshape(-1), eo, Nin, Nout)
            out += [Tab(Af, FULL), Tab(Wf, FULL)]
        return out
    rho, fin = ctx.sumcheck(label, cb, Kin, shapes, np.array(claim), tabs=tabs if ctx.proving else None,
                            terms=terms)
    for i, (po, pm, _) in enumerate(claims):
        ctx.claim(A_name, cat(rho, pm), fin[2 * i])
        if ctx.P:
            # block values W_{tb,ob}(po', rho)
            e1 = G.eq_table(po[:, :cbo])
            r1 = G.rowdot(W_nat.reshape(-1), e1, Nin * Kout, 1 << cbo)          # (D, Nin*Kout) [tb][t'][ob]
            et = G.eq_table(rho)                                                 # (D, 2^cb)
            etr = et.repeat_interleave(Kout, dim=1).repeat(1, Kin)
            pr = G.mul(r1, etr).to(torch.int64).reshape(ctx.D, Kin, 1 << cb, Kout).sum(dim=2) % P
            vals = pr.reshape(ctx.D, Kin * Kout).to(torch.int32)
        else:
            vals = None
        vals = ctx.send_vals(vals, (ctx.D, Kin * Kout))
        tau = po[:, cbo:]
        for tbk in range(Kin if not ctx.P else 0):
            v = ctx.blockcomb(tau, vals[:, tbk * Kout:(tbk + 1) * Kout])
            if tuple(int(x) for x in v) != F.c(fin[2 * i + 1][:, tbk]):
                raise ValueError(f"matmul {label!r}: weight block values inconsistent")
        ctx.claim(W_name, cat(po[:, :cbo], rho), vals)


def combine_blocks(ctx, pt_in, vals, nb):
    """Per-block claims at pt_in -> point claim (pt_in, tau) with fresh tau over nb bits."""
    tau = ctx.chal(nb) if nb else np.zeros((ctx.D, 0), dtype=np.int64)
    return cat(pt_in, tau), ctx.blockcomb(tau, vals)


# =================================================================== the protocol
class Bound:
    def __init__(self, s2, os2):
        self.s2, self.os2 = s2, os2


def h_from_sat(s, o):
    return torch.where(o == 0, s, torch.where(o > 0, torch.full_like(s, (1 << 23) - 1), torch.full_like(s, -(1 << 23))))


NVTX = bool(os.environ.get("ZK_NVTX"))


# step timers synchronize the GPU only with ZK_TIMESYNC=1 (exact per-step GPU attribution);
# otherwise they measure host time and leave the pipeline running
TIMESYNC = bool(os.environ.get("ZK_TIMESYNC"))


class Timer:
    def __init__(self, tr=None):
        self.t, self.b, self.tr, self.depth = {}, {}, tr, 0

    def __call__(self, key):
        tmr = self

        class _C:
            def __enter__(self):
                if TIMESYNC:
                    torch.cuda.synchronize()
                self.t0 = time.time()
                self.b0 = tmr.tr.sent_bytes if tmr.tr is not None else 0
                tmr.depth += 1
                if NVTX:
                    torch.cuda.nvtx.range_push(key)

            def __exit__(self, *a):
                if TIMESYNC:
                    torch.cuda.synchronize()
                if NVTX:
                    torch.cuda.nvtx.range_pop()
                tmr.depth -= 1
                tmr.t[key] = tmr.t.get(key, 0.0) + time.time() - self.t0
                if tmr.tr is not None and tmr.depth == 0:
                    tmr.b[key] = tmr.b.get(key, 0) + tmr.tr.sent_bytes - self.b0
        return _C()


class Proto:
    def __init__(self, ctx, M, sh, tokens, targets):
        self.ctx, self.M, self.sh = ctx, M, sh
        self.tokens = tokens            # (B, S) public
        self.targets = targets          # (B, ngen) public: next tokens for the last ngen rows
        self.theta = 0                  # temperature index of the policy (0: greedy)
        self.u = None                   # (B ngen,) public PRF values u_t (theta > 0)
        self.cfg = M.cfg
        self.L = self.cfg.L
        self.wit = {}
        self.tm = Timer(ctx.tr)
        self._host_layers = None
        self._stash, self._stash_n, self._stash_ord = {}, 0, []
        self._pdom = {}                 # (layer, group) -> P claims of the tile domains

    def memlog(self, where):
        if os.environ.get("ZK_MEMLOG"):
            print(f"[mem] {where}: {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
                  f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)

    def use_layer(self, l):
        """Keep only layer l's weights on the GPU when the model lives on the host."""
        M = self.M
        from . import ext
        if ext.CPU[0]:                  # host verifier: the layers stay where they are
            return
        if self._host_layers is None:
            if not any(Ly.Wqkv.device.type == "cpu" for Ly in M.layers):
                return
            M.layers = list(M.layers)
            self._host_layers = list(M.layers)
        for i, Ly in enumerate(M.layers):
            if i != l and Ly is not self._host_layers[i]:
                M.layers[i] = self._host_layers[i]
        if M.layers[l] is self._host_layers[l]:
            M.layers[l] = mdl.layer_to(self._host_layers[l], "cuda")

    # ------------------------------------------------------------ names / groups
    def declare(self):
        sh, ctx = self.sh, self.ctx
        t, cb = sh.t, sh.cb
        for l in range(self.L):
            p = f"{l}."
            rows = lambda pre, fields: [(pre + f, 1, t) for f in fields]
            jb = lambda pre, K: [(pre + "jb", cb + nbits(K), t)]
            Ko = sh.Kk if self.cfg.qk_norm else sh.Kq + 2 * sh.Kk         # blocks of oq, whq
            qkn = []
            for X, KX in (("qn", sh.Kq), ("kn", sh.Kk)) if self.cfg.qk_norm else ():
                q, nr = p + X + ".", (cb - sh.ldh) + t + nbits(KX)
                qkn += [(q + f, KX, cb + t) for f in ("q8", "qp", "o", "wh")]
                qkn += [(q + "r." + f, 1, nr) for f in RQ_F] + [(q + "r.jb", sh.ldh, nr)]
                qkn += [(q + "n." + f, 1, nr) for f in NS_F] + [(q + f, 1, nr) for f in QK_F]
            ctx.group(p + "in", [(p + "q1", sh.Kd, cb + t), (p + "qp1", sh.Kd, cb + t)] + rows(p + "r1.", RQ_F)
                      + jb(p + "r1.", sh.Kd) + rows(p + "n1.", NS_F)
                      + [(p + "Q", sh.Kq, cb + t), (p + "K", sh.Kk, cb + t), (p + "V", sh.Kk, cb + t),
                         (p + "oq", Ko, cb + t), (p + "whq", Ko, cb + t),
                         (p + "rQ", sh.nh, sh.ldh + t), (p + "rK", sh.nkv, sh.ldh + t)] + qkn)
            for g in range(sh.nkv):
                q = f"{l}.g{g}."
                npc = self.p_cols()
                ctx.group(p + f"att{g}", [(q + "P", sh.grp * npc[0], npc[1]), (q + "js", sh.grp, t),
                                          (q + "Kg", sh.grp, sh.ldh + t), (q + "mk", 1, t),
                                          (q + "Nr", sh.grp, sh.ldh + t + sh.nbb)]
                          + [(q + "nK." + f, sh.grp, t) for f in SR_F])
            ctx.group(p + "ao", [(p + "nE." + f, sh.nh, t) for f in SR_F] + [(p + "nM." + f, sh.nh, t) for f in SR_F]
                      + [(p + "nd." + f, sh.nh, t) for f in ND_F]
                      + [(p + "nd.xh", sh.nh, sh.ldh + t), (p + "nd.t", sh.nh, sh.ldh + t),
                         (p + "o", sh.Kq, cb + t), (p + "qo", sh.Kq, cb + t)]
                      + rows(p + "ro.", RQ_F) + jb(p + "ro.", sh.Kq) + rows(p + "so.", SRX_F)
                      + [(p + "s1", sh.Kd, cb + t), (p + "os1", sh.Kd, cb + t)])
            if self.cfg.moe:
                ctx.group(p + "mi", [(p + "q2", sh.Kd, cb + t), (p + "qp2", sh.Kd, cb + t)] + rows(p + "r2.", RQ_F)
                          + jb(p + "r2.", sh.Kd) + rows(p + "n2.", NS_F) + rows(p + "sd.", SRX_F))
                mp = self.moe_of(l)
                ctx.group(p + "moe", mp.items())
                for gname, items in mp.expert_groups().items():
                    ctx.group(gname, items)
            else:
                ctx.group(p + "mi", [(p + "q2", sh.Kd, cb + t), (p + "qp2", sh.Kd, cb + t)] + rows(p + "r2.", RQ_F)
                          + jb(p + "r2.", sh.Kd) + rows(p + "n2.", NS_F) + rows(p + "ru.", RQ_F) + jb(p + "ru.", sh.Kf)
                          + rows(p + "rm.", RQ_F) + jb(p + "rm.", sh.Kf) + rows(p + "sd.", SRX_F)
                          + rows(p + "g.", EG_F))
                for s, (b0, b1) in enumerate(self.slices()):
                    q = f"{l}.m{s}."
                    ctx.group(p + f"mlp{s}", [(q + x, b1 - b0, cb + t) for x in ("c", "a", "qu", "qm")])
            ctx.group(p + "mo", [(p + "s2", sh.Kd, cb + t), (p + "os2", sh.Kd, cb + t)])
        r = log2(sh.R)
        ctx.group("fin", [("f.q", sh.Kd, cb + r), ("f.qp", sh.Kd, cb + r)] + [("f.r." + f, 1, r) for f in RQ_F]
                  + [("f.r.jb", cb + nbits(sh.Kd), r)] + [("f.n." + f, 1, r) for f in NS_F]
                  + [("f.z", sh.Kv, sh.cbv + r)] + [("f.sz." + f, 1, r) for f in SRX_F]
                  + ([("f.p", sh.Kv, sh.cbv + r), ("f.zj", sh.cbv + nbits(sh.Kv), r)]
                     + [("f.s." + f, 1, r) for f in SAMP_F] if self.theta else []))
        for l in range(self.L):
            Kqkv = sh.Kq + 2 * sh.Kk
            # the weights' blocks are packed flat where a matmul claims them all at one point (qkv, o)
            # or in long runs (down: 9-row slices); padding is judged against the cached weight
            # commitment, not per tensor
            wqkv = [(f"{l}.Wq", sh.Kd * sh.Kq, 2 * cb), (f"{l}.Wk", sh.Kd * sh.Kk, 2 * cb),
                    (f"{l}.Wv", sh.Kd * sh.Kk, 2 * cb)] if self.cfg.qk_norm else \
                [(f"{l}.Wqkv", sh.Kd * Kqkv, 2 * cb)]
            mlpw = self.moe_of(l).weight_items() if self.cfg.moe else [
                (f"{l}.Wg", sh.Kd * sh.Kf, 2 * cb, (sh.Kd, sh.Kf)),
                (f"{l}.Wu", sh.Kd * sh.Kf, 2 * cb, (sh.Kd, sh.Kf)),
                (f"{l}.Wd", sh.Kf * sh.Kd, 2 * cb)]
            ctx.group(f"{l}.W", wqkv + [(f"{l}.Wo", sh.Kq * sh.Kd, 2 * cb)] + mlpw, pad=W_PACK_PAD)
        ctx.group("Wlm", [("Wlm", sh.Kd * sh.Kv, cb + sh.cbv, (sh.Kd, sh.Kv))])
        ctx.group("emb", [("emb", sh.Kv * sh.Kd, sh.cbv + cb, (sh.Kv, sh.Kd))])
        if MERGE_TREES:
            for l in range(self.L):
                ctx.pcs.merge(self.layer_groups(l))
        # each layer's groups and weights are reduced to one point per class at the end of the
        # layer's prove pass, while its witness is on the device (gpcs.SEG)
        for l in range(self.L):
            ctx.pcs.segment_decl(self.layer_groups(l) + self.layer_wgroups(l))
        # the fixed LogUp table (tag, v0..v5), committed once per (model, temperature): the verifier
        # takes its evaluations from the prover as claims on this unit (no O(2^26) evaluation)
        ctx.group("tab", [(f"tab{j}", 1, tb.LOG_ROWS) for j in range(tb.NV + 1)])
        # the RoPE tables at every position (cos, sin): one fixed unit for all layers, claimed by
        # every layer's RoPE domain (instead of a verifier evaluation per layer)
        ctx.group("rope", [("rope.cos", 1, sh.ldh - 1 + t), ("rope.sin", 1, sh.ldh - 1 + t)])

    def p_cols(self):
        """P columns per head: (count, log size)."""
        sh = self.sh
        tot = sh.B * sh.ntile
        # the largest column (fewest columns, narrowest Merkle leaves) whose zero padding stays
        # within P_PAD of the tiles: padding is committed like data
        for c in range(min(23, 14 + nbits(tot)), 13, -1):
            npc = ((tot << 14) + (1 << c) - 1) >> c
            if (npc << c) <= (tot << 14) * (1 + P_PAD):
                return npc, c
        return tot, 14

    def slices(self):
        sh = self.sh
        per = max(1, min(sh.Kf, (MLP_SLICE << 24) // (sh.T << sh.cb)))
        return [(b, min(sh.Kf, b + per)) for b in range(0, sh.Kf, per)]

    # ------------------------------------------------------------ tensor data (prover)
    def tensors(self, gname, W):
        sh, cb = self.sh, self.sh.cb
        out = {}

        def done(o):
            self._stash_put(gname, o)
            return o
        if gname.endswith("W") or gname in ("Wlm", "emb") or gname.split(".")[-1] in ("Wge", "Wue", "Wde"):
            return self.weight_tensors(gname)
        if gname == "mult":
            return {"mult": self.ctx.mult}
        if gname == "fin":
            F = self.fin
            out["f.q"], out["f.qp"] = rm(F["r"]["q"], cb), rm(F["n"]["qp"], cb)
            for f in RQ_F:
                out["f.r." + f] = f32(F["r"][f])
            out["f.r.jb"] = torch.cat([f32(b) for b in F["r"]["jb"]])
            for f in NS_F:
                out["f.n." + f] = f32(F["n"][f])
            out["f.z"] = rm(F["z"], sh.cbv)
            sz = sr_pack(F["k"])
            for f in SRX_F:
                out["f.sz." + f] = f32(sz[f])
            if F["samp"] is not None:
                S_ = F["samp"]
                out["f.p"] = rm(S_["p"], sh.cbv)
                out["f.zj"] = torch.cat([f32(b) for b in S_["jb"]])
                for f in SAMP_F:
                    out["f.s." + f] = f32(S_[f])
            return done(out)
        l = int(gname.split(".")[0])
        p = f"{l}."
        kind = gname.split(".")[1]
        if kind == "in":
            out[p + "q1"], out[p + "qp1"] = rm(W.r1["q"], cb), rm(W.n1["qp"], cb)
            for f in RQ_F:
                out[p + "r1." + f] = f32(W.r1[f])
            out[p + "r1.jb"] = torch.cat([f32(b) for b in W.r1["jb"]])
            for f in NS_F:
                out[p + "n1." + f] = f32(W.n1[f])
            nq, nk = sh.nh * sh.dh, sh.nkv * sh.dh
            out[p + "Q"], out[p + "K"], out[p + "V"] = rm(W.qkv[:, :nq], cb), rm(W.qkv[:, nq:nq + nk], cb), \
                rm(W.qkv[:, nq + nk:], cb)
            out[p + "oq"], out[p + "whq"] = rm(W.oq, cb), rm(W.whq, cb)
            out[p + "rQ"], out[p + "rK"] = rm(W.rQ, sh.ldh), rm(W.rK, sh.ldh)
            for X, Q in getattr(W, "qn", {}).items():
                q = p + X + "."
                for f in ("q8", "qp", "o", "wh"):
                    out[q + f] = rm(Q[f], cb)
                for f in RQ_F:
                    out[q + "r." + f] = f32(Q["r"][f])
                out[q + "r.jb"] = torch.cat([f32(b) for b in Q["r"]["jb"]])
                for f in NS_F:
                    out[q + "n." + f] = f32(Q["n"][f])
                for f in QK_F:
                    out[q + f] = f32(Q[f])
        elif kind.startswith("att"):
            g = int(kind[3:])
            a = W.att[g]
            q = f"{l}.g{g}."
            npc, pc = self.p_cols()
            Pf = torch.zeros((sh.grp, npc << pc), dtype=torch.int32, device=_dev())
            for i in range(sh.grp):
                flat = a["P"][i].reshape(-1)
                Pf[i, :flat.numel()] = flat.to(torch.int32)
            out[q + "P"] = Pf.reshape(-1)
            out[q + "js"] = torch.cat([f32(x) for x in a["js"]])
            out[q + "Kg"] = torch.cat([f32(x) for x in a["Kg"]])
            out[q + "mk"] = f32(a["mk"])
            out[q + "Nr"] = torch.cat([f32(x) for x in a["Nr"]])
            srK = sr_pack(W.nd["km"][g * sh.grp:(g + 1) * sh.grp])
            for f in SR_F:
                out[q + "nK." + f] = f32(srK[f])
        elif kind == "ao":
            for pre, key in (("nE.", "E"), ("nM.", "kp")):
                sr = sr_pack(W.nd[key])
                for f in SR_F:
                    out[p + pre + f] = f32(sr[f])
            for f in ND_F:
                out[p + "nd." + f] = f32(W.nd[f])
            out[p + "nd.xh"], out[p + "nd.t"] = rm(W.ndxh, sh.ldh), rm(W.ndt, sh.ldh)
            out[p + "o"], out[p + "qo"] = rm(W.o, cb), rm(W.ro["q"], cb)
            for f in RQ_F:
                out[p + "ro." + f] = f32(W.ro[f])
            out[p + "ro.jb"] = torch.cat([f32(b) for b in W.ro["jb"]])
            so = sr_pack(W.ko)
            for f in SRX_F:
                out[p + "so." + f] = f32(so[f])
            out[p + "s1"], out[p + "os1"] = rm(W.s1, cb), rm(W.os1, cb)
        elif kind == "mi":
            out[p + "q2"], out[p + "qp2"] = rm(W.r2["q"], cb), rm(W.n2["qp"], cb)
            for pre, r in (("r2.", W.r2),) + ((("ru.", W.ru), ("rm.", W.rm)) if not self.cfg.moe else ()):
                for f in RQ_F:
                    out[p + pre + f] = f32(r[f])
                out[p + pre + "jb"] = torch.cat([f32(b) for b in r["jb"]])
            for f in NS_F:
                out[p + "n2." + f] = f32(W.n2[f])
            sd = sr_pack(W.kd)
            for f in SRX_F:
                out[p + "sd." + f] = f32(sd[f])
            if not self.cfg.moe:
                for f in EG_F:
                    out[p + "g." + f] = f32(W.eg[f])
        elif kind.startswith("mlp"):
            s = int(kind[3:])
            b0, b1 = self.slices()[s]
            q = f"{l}.m{s}."
            cols = slice(b0 << cb, b1 << cb)
            out[q + "c"], out[q + "a"] = rm(W.c[:, cols], cb), rm(W.a[:, cols], cb)
            out[q + "qu"], out[q + "qm"] = rm(W.ru["q"][:, cols], cb), rm(W.rm["q"][:, cols], cb)
        elif kind == "mo":
            out[p + "s2"], out[p + "os2"] = rm(W.s2, cb), rm(W.os2, cb)
        elif kind == "moe":
            out.update(self.moe_of(l).tensors(W.moe))
        return done(out)

    def weight_tensors(self, gname):
        """Weight group '{l}.W', 'Wlm' or 'emb': blocks (tb, ob), in-block (o' low, t' high)."""
        sh, M, cb = self.sh, self.M, self.sh.cb

        def wl(Wn, cbi, cbo):
            Nin, Nout = Wn.shape
            Ki, Ko = Nin >> cbi, Nout >> cbo
            return f32(Wn.reshape(Ki, 1 << cbi, Ko, 1 << cbo).permute(0, 2, 1, 3))
        if gname == "Wlm":
            return {"Wlm": wl(M.Wlm, cb, sh.cbv)}
        if gname == "emb":
            return {"emb": wl(M.emb, sh.cbv, cb)}
        l = int(gname.split(".")[0])
        Ly = M.layers[l]
        if gname.split(".")[-1] in ("Wge", "Wue", "Wde"):
            return self.moe_of(l).expert_tensor(gname.split(".")[-1])
        if self.cfg.qk_norm:
            nq, nk = sh.nh * sh.dh, sh.nkv * sh.dh
            wq = {f"{l}.Wq": wl(Ly.Wqkv[:, :nq], cb, cb), f"{l}.Wk": wl(Ly.Wqkv[:, nq:nq + nk], cb, cb),
                  f"{l}.Wv": wl(Ly.Wqkv[:, nq + nk:], cb, cb)}
        else:
            wq = {f"{l}.Wqkv": wl(Ly.Wqkv, cb, cb)}
        if self.cfg.moe:
            return {**wq, f"{l}.Wo": wl(Ly.Wo, cb, cb), **self.moe_of(l).weight_tensors()}
        return {**wq, f"{l}.Wo": wl(Ly.Wo, cb, cb), f"{l}.Wg": wl(Ly.Wg, cb, cb),
                f"{l}.Wu": wl(Ly.Wu, cb, cb), f"{l}.Wd": wl(Ly.Wd, cb, cb)}

    # ------------------------------------------------------------ residual columns
    def hin_cols(self, db, l, getW, pre=None, post=None):
        """Adds the residual input of layer l as a Poly over columns of db."""
        ctx, sh = self.ctx, self.sh
        if l == 0:
            return db.col("h0", lambda: rm(getW().hin, sh.cb), R_virt(ctx, "h0"))
        q = f"{l - 1}."
        s = db.col(q + "s2", lambda: rm(self.wit_of(l - 1).s2, sh.cb), R_blk(ctx, q + "s2", pre=pre, post=post))
        o = db.col(q + "os2", lambda: rm(self.wit_of(l - 1).os2, sh.cb), R_blk(ctx, q + "os2", pre=pre, post=post))
        return hexpr(s, o)

    def wit_of(self, l):
        return self.wit[l]

    def _stash_pop(self, gname):
        old = self._stash.pop(gname, None)
        if old is None:
            return
        self._stash_ord = [x for x in self._stash_ord if x != gname]
        self._stash_n -= sum(t.numel() * t.element_size() for t in old.values())

    def _stash_put(self, gname, tensors):
        if not self.ctx.P or not tensors:
            return
        if self.L > STASH_L:
            return
        self._stash_pop(gname)
        cpu, n = {}, 0
        for k, v in tensors.items():
            if not torch.is_tensor(v):
                continue
            t = v.detach().cpu()
            cpu[k] = t
            n += t.numel() * t.element_size()
        lim = STASH_GB << 30
        while self._stash_n + n > lim and self._stash_ord:
            self._stash_pop(self._stash_ord[0])
        if self._stash_n + n > lim:
            return
        self._stash[gname] = cpu
        self._stash_ord.append(gname)
        self._stash_n += n

    def _stash_get(self, gname):
        torch.cuda.empty_cache()
        return {k: v.cuda(non_blocking=True) for k, v in self._stash[gname].items()}

    # ------------------------------------------------------------ gadget emitters
    def rq_rows(self, db, pre, getr, K, e0=0, emax=17, asel_key=None, cb=None, B=127):
        """Per-row Requant8/ShiftQ8 minimality (domain over rows of K 2^cb elements); B: the bound
        |q| <= B of a wider Requant (e0 = 0)."""
        ctx = self.ctx
        cb = self.sh.cb if cb is None else cb
        c = {f: db.col(pre + f, (lambda f=f: f32(getr()[f])), R_blk(ctx, pre + f)) for f in RQ_F}
        nb = cb + nbits(K)
        bits = [db.col(pre + f"jb{i}", (lambda i=i: f32(getr()["jb"][i])), R_blk(ctx, pre + "jb", bmap=lambda b, i=i: i))
                for i in range(nb)]
        asel = db.col(pre + "asel", lambda: f32(getr()["asel"]), R_virt(ctx, asel_key))
        d = db.dom
        e, s, f, sg, wl, wh = (c[x] for x in RQ_F)
        d.lookup(tb.POW, e + tb.POW_OFF, s)
        d.lookup(5, e - e0)
        d.lookup(5, emax - e)
        d.zero(f * (1 - f))
        d.zero((e - e0) * (1 - f))
        d.zero(sg * (1 - sg))
        if e0 == 2:
            # ShiftQ8: |asel| < 2^29, so the test "does not fit at e - 1" is asel >= 255 s / 4 (s >= 4),
            # with w < 2^28 in two 14-bit limbs: |asel| + 255 s / 4 + 2^28 < 2^30 < p (4 asel wraps)
            q4 = pow(4, P - 2, P)
            d.zero(f * (sg * (asel - 255 * q4 * s) + (1 - sg) * (-asel - 255 * q4 * s - 1)) - (1 << 14) * wh - wl)
            d.lookup(14, wl)
            d.lookup(14, wh)
        else:
            # Requant (products a q, |asel| < 2^23): 4 |asel| + W 2^emax + 2^30 < p needs a small emax;
            # a ShiftQ8 of an accumulator up to 2^22 must take the quartered branch above (e0 = 2)
            W = 2 * B + 1
            assert e0 == 0 or B == 127
            assert (1 << 25) + (W << emax) + (1 << 30) < P, (B, emax)
            d.zero(f * (sg * (4 * asel - W * s) + (1 - sg) * (-4 * asel - W * s - 1)) - (1 << 15) * wh - wl)
            d.lookup(15, wl)
            d.lookup(15, wh)
        for b in bits:
            d.zero(b * (1 - b))
        if K & (K - 1):
            js = bits[0]
            for i, b in enumerate(bits[1:], 1):
                js = js + (1 << i) * b
            d.lookup(24, (K << cb) - 1 - js)
        return c

    def ns_rows(self, db, pre, getn, sig_key):
        ctx = self.ctx
        c = {f: db.col(pre + f, (lambda f=f: f32(getn()[f])), R_blk(ctx, pre + f)) for f in NS_F}
        sig = db.col(pre + "sig", lambda: f32(getn()["sig"]), R_virt(ctx, sig_key))
        d = db.dom
        Eh, Ep, pEp, pEn, pEh, delta, rho, mu, ng = (c[x] for x in NS_F)
        d.lookup(tb.NPOW, 2 * Eh + 15, Ep, pEp, pEn, 1, ng)
        d.lookup(tb.POW, Eh + tb.POW_OFF, pEh)
        d.lookup(Ep, rho)
        d.lookup(16, ng * sig)
        d.lookup(tb.RSQ, delta - tb.RSQ_LO, mu)
        d.zero(pEn * sig - pEp * delta - rho)
        return c

    def sr_rows(self, db, pre, getk, kexpr, left=False):
        """SR regime tuple for a per-row shift k (kexpr: Poly over db columns); left: k may be <= 0, so
        the row also carries the SRA input bound, which the domain of its elements applies."""
        ctx = self.ctx
        fs = SRX_F if left else SR_F
        c = {f: db.col(pre + f, (lambda f=f: f32(sr_pack(getk())[f])), R_blk(ctx, pre + f)) for f in fs}
        db.dom.lookup(tb.SR, *[c[f] for f in SR_F])
        if left:
            db.dom.lookup(tb.SRA, c["k"], *[c[f] for f in SRA_F])
        db.dom.zero(c["k"] - (kexpr - tb.SR_KMIN))
        return c

    def gather_rowsum(self, label, l, n_rows, rowpt, K, parts, jb_name, getbits, cb=None):
        """Gather and row-sum sumcheck over a row-major (rows x K 2^cb) domain:
        sum_{m,x} eq(r, m) [OH(m,x) h(m,x) + gamma q(m,x)^2] = asel(r) + gamma (sig(r) - 1).
        parts: dict with 'cols' [(name, data_fn, route)], 'h' fn(polys) -> Poly, 'q' index or None,
        'asel' value, 'sig' value or None."""
        ctx, sh = self.ctx, self.sh
        cb = sh.cb if cb is None else cb
        F = ctx.F
        n = cb + n_rows
        gam = ctx.chal()
        if ctx.P:
            claim = np.zeros(ctx.D, dtype=np.int64)
        else:
            claim = F.c(parts["asel"])
            if parts.get("sig") is not None:
                claim = F.add(claim, F.mul(F.c(gam), F.sub(F.c(parts["sig"]), F.one)))
        cols = parts["cols"]
        # tables: 0 = eq rows (ROW shared), 1 = OH, 2.. = columns
        nc = len(cols)
        polys = [Poly.var(1) * parts["h"]([Poly.var(2 + i) for i in range(nc)])]
        expr = ("*", ("p", 1), ("p", 0))
        coefs = []
        polys.append(Poly.var(0))
        if parts.get("q") is not None:
            qv = Poly.var(2 + parts["q"])
            polys.append(qv * qv)
            coefs.append(gam if ctx.P else np.asarray(gam, dtype=np.int64))
            expr = ("*", ("p", 1), ("+", ("p", 0), ("*", ("c", 0), ("p", 2))))
        prog = Prog(polys, expr, coefs)
        shapes = [(ROW, cb), (FULL, 0)] + [(FULL, 0)] * nc

        def tabs():
            G = ctx.G
            eqr = G.eq_table(rowpt)
            bits = getbits()
            j = torch.zeros_like(bits[0])
            for i, b in enumerate(bits):
                j = j | (b << i)
            OH = torch.zeros(K << (n_rows + cb), dtype=torch.int32, device=_dev())
            m = torch.arange(1 << n_rows, device=_dev())
            OH.index_fill_(0, (((j >> cb) << n_rows) + m << cb) + (j & ((1 << cb) - 1)), 1)
            return [Tab(eqr, ROW, cb, shared=True), Tab(OH, FULL)] + [Tab(f(), FULL) for _, f, _ in cols]
        rho, fin = ctx.sumcheck(label, n, K, shapes, np.array(claim), tabs=tabs if ctx.proving else None, prog=prog)
        check_eq(ctx, label, fin[0], rowpt, rho[:, cb:])
        for i, (name, _, route) in enumerate(cols):
            route(rho, fin[2 + i])
        oh_check(ctx, label, jb_name, (cb, nbits(K)), [(rho, fin[1])], n_rows, getbits)

    # ------------------------------------------------------------ final logits
    def fin_wit(self, hL):
        sh, M = self.sh, self.M
        rows = torch.cat([torch.arange(b * sh.S + sh.S - sh.ng, (b + 1) * sh.S, device=_dev()) for b in range(sh.B)])
        h = hL[rows]
        r = rq_pack(h, sh.cb)
        n = ns_pack(r["q"])
        acc = mdl.imm(n["qp"], M.Wlm)
        k = M.Kz + n["Eh"]
        z = mdl.gshiftround(acc, k.unsqueeze(-1))
        y = self.targets.reshape(-1).to(_dev())
        lt = (torch.arange(self.cfg.V, device=_dev()).unsqueeze(0) < y.unsqueeze(1)).to(torch.int64)
        if self.theta == 0:
            zmax = z.gather(1, y.unsqueeze(1)).squeeze(1)
            assert int((zmax.unsqueeze(1) - z - lt).min()) >= 0, "targets are not the greedy argmax"
            samp = None
        else:
            zmax = z.max(dim=1).values
            samp = samp_wit(z, zmax, torch.argmax(z, dim=1), y, lt, self.u, self.theta, sh)
        self.fin = dict(h=h, r=r, n=n, acc=acc, k=k, z=z, zmax=zmax, lt=lt, rows=rows, samp=samp)

    def fin_steps(self):
        ctx, sh, M = self.ctx, self.sh, self.M
        cb, r = sh.cb, log2(sh.R)
        L = self.L - 1
        Fw = (lambda: self.fin) if ctx.P else None
        hi = const_pt(ctx.D, [1] * (sh.ls - sh.lng))

        def ins(pt):          # in-block point over (j', r, seq) -> (j', r, 1.., seq)
            return cat(pt[:, :cb + sh.lng], hi, pt[:, cb + sh.lng:])

        def Rlast(name):
            def f(pt, vals):
                ctx.claim(name, ins(pt), vals)
            return f
        rowsel = (lambda X: rm(X[self.fin["rows"]], cb))
        # z domain over (v', r) x Kv
        db = DB(ctx, "fin.z", sh.cbv + r, sh.Kv)
        acc = db.col("accz", lambda: rm(Fw()["acc"], sh.cbv), R_virt(ctx, "accz"))
        z = db.col("z", lambda: rm(Fw()["z"], sh.cbv), R_blk(ctx, "f.z"))
        srr = {f: db.col("sz." + f, (lambda f=f: f32(sr_pack(Fw()["k"])[f])), R_blk(ctx, "f.sz." + f),
                         kind=ROW, c=sh.cbv, shared=True) for f in ("s", "R", "tR", "tB", "oB") + SRA_F}
        zmax = db.col("zmax", lambda: f32(Fw()["zmax"]), R_virt(ctx, "zmax"), kind=ROW, c=sh.cbv, shared=True)
        y = self.targets.reshape(-1)

        def lt_eval(pt):
            def full():
                ltt = (torch.arange(self.cfg.V, device=_dev()).unsqueeze(0) < y.to(_dev()).unsqueeze(1)).to(torch.int64)
                X = rm(ltt, sh.cbv).reshape(sh.Kv, -1)
                return np.stack([mle_host(ctx, X[k], pt) for k in range(sh.Kv)], axis=1)
            return mle_pub(ctx, lambda: mlepub.vocab_indicator(pt, [int(v) for v in y.tolist()], sh.cbv, sh.Kv, "lt"),
                           full)
        d = db.dom
        d.lookup(srr["tR"], acc + srr["R"] - srr["s"] * z)
        d.lookup(srr["tB"], z + srr["oB"])
        d.lookup(srr["tA"], srr["f"] * acc + srr["oA"])
        if self.theta == 0:
            # greedy: zmax = z at the public target and zmax - z - [v < y] >= 0
            lt = db.col("lt", lambda: rm(Fw()["lt"], sh.cbv), R_pub(ctx, lt_eval))
            d.lookup(24, zmax - z - lt)
        else:
            # sampling: p = tau_exp,theta(zmax - z); the table domain forces z <= zmax
            pc = db.col("f.p", lambda: rm(Fw()["samp"]["p"], sh.cbv), R_blk(ctx, "f.p"))
            d.lookup(tb.EXPT, zmax - z, pc)
        db.run()
        if ctx.active:
            (pz, vz), = ctx.take("zmax")
            if self.theta == 0:
                # zmax gather at the public targets
                self.public_gather(pz, vz)
            else:
                # zmax = z at a committed index (Boolean bits, bounded in fin.rows)
                self.gather_rowsum(b"fin-zmax", L, r, pz, sh.Kv, dict(
                    cols=[("f.z", lambda: rm(Fw()["z"], sh.cbv), R_blk(ctx, "f.z"))],
                    h=lambda c: c[0], q=None, asel=vz[:, 0], sig=None),
                    "f.zj", lambda: Fw()["samp"]["jb"], cb=sh.cbv)
                self.samp_sums()
            # lm head product
            cl = []
            for pt, vals in ctx.take("accz"):
                po, v = combine_blocks(ctx, pt[:, :sh.cbv], vals, nbits(sh.Kv))
                cl.append((po, pt[:, sh.cbv:], v))
            matmul(ctx, b"lm", "f.qp", (lambda: rm(Fw()["n"]["qp"], cb)) if ctx.P else None, sh.Kd, sh.R,
                   "Wlm", M.Wlm if ctx.P else None, sh.Kv, cb, cl, cbo=sh.cbv)
        # per-row domain
        db = DB(ctx, "fin.rows", r)
        self.rq_rows(db, "f.r.", lambda: Fw()["r"], sh.Kd, asel_key="f.asel")
        nsc = self.ns_rows(db, "f.n.", lambda: Fw()["n"], "f.sig")
        self.sr_rows(db, "f.sz.", lambda: Fw()["k"], nsc["Eh"] + M.Kz, left=True)
        if self.theta:
            self.samp_rows(db)
        db.run()
        # Requant8 / NormScale domain over (j', r, seq) x Kd
        db = DB(ctx, "fin.rq", cb + r, sh.Kd)
        s = db.col(f"{L}.s2", lambda: rowsel(self.wit_of(L).s2), Rlast(f"{L}.s2"))
        o = db.col(f"{L}.os2", lambda: rowsel(self.wit_of(L).os2), Rlast(f"{L}.os2"))
        h = hexpr(s, o)
        self.rq_norm_dom(db, h, "f.q", "f.qp", "f.r.", "f.n.", lambda: Fw()["r"], lambda: Fw()["n"])
        db.run()
        if ctx.active:
            (pa, va), = ctx.take("f.asel")
            (ps, vs), = ctx.take("f.sig")
            assert np.array_equal(pa, ps)
            self.gather_rowsum(b"fin-g", L, r, pa, sh.Kd, dict(
                cols=[(f"{L}.s2", lambda: rowsel(self.wit_of(L).s2), Rlast(f"{L}.s2")),
                      (f"{L}.os2", lambda: rowsel(self.wit_of(L).os2), Rlast(f"{L}.os2")),
                      ("f.q", lambda: rm(Fw()["r"]["q"], cb), R_blk(ctx, "f.q"))],
                h=lambda c: hexpr(c[0], c[1]), q=2, asel=va[:, 0], sig=vs[:, 0]),
                "f.r.jb", lambda: Fw()["r"]["jb"])

    def samp_rows(self, db):
        """Per generated token: Boolean bits and bound of the row-maximum index, and the
        inverse-CDF condition A <= floor(u Z / 2^30) < A + P on 10-bit limbs (every relation
        stays below 2^22, so none wraps)."""
        ctx, sh = self.ctx, self.sh
        Fw = (lambda: self.fin["samp"]) if ctx.P else None
        d = db.dom
        nb = sh.cbv + nbits(sh.Kv)
        bits = [db.col(f"f.zj{i}", (lambda i=i: f32(Fw()["jb"][i])), R_blk(ctx, "f.zj", bmap=lambda b, i=i: i))
                for i in range(nb)]
        for b in bits:
            d.zero(b * (1 - b))
        if sh.Kv & (sh.Kv - 1):
            js = bits[0]
            for i, b in enumerate(bits[1:], 1):
                js = js + (1 << i) * b
            d.lookup(24, (sh.Kv << sh.cbv) - 1 - js)
        c = {f: db.col("f.s." + f, (lambda f=f: f32(Fw()[f])), R_blk(ctx, "f.s." + f)) for f in SAMP_F}
        ul = _l10(self.u.to(_dev()))
        u = [db.col(f"u{i}", (lambda i=i: f32(ul[i])),
                    R_pub(ctx, lambda pt, i=i: mle_host(ctx, ul[i], pt).reshape(-1, 1))) for i in range(3)]
        z = [c[f"z{i}"] for i in range(3)]
        t = [c["q0"], c["q1"], c["q2"], c["w0"], c["w1"], c["w2"]]
        k = [c[f"k{i}"] for i in range(5)]
        d.zero(c["Z"] - (z[0] + (1 << 10) * z[1] + (1 << 20) * z[2]))
        for kk in range(5):
            col = None
            for i in range(3):
                if 0 <= kk - i < 3:
                    col = u[i] * z[kk - i] if col is None else col + u[i] * z[kk - i]
            if kk:
                col = col + k[kk - 1]
            d.zero(col - t[kk] - (1 << 10) * k[kk])
        d.zero(k[4] - t[5])
        w = t[3] + (1 << 10) * t[4] + (1 << 20) * t[5]
        d.zero(w - c["A"] - (c["d1l"] + (1 << 15) * c["d1h"]))
        d.zero(c["A"] + c["P"] - 1 - w - (c["d2l"] + (1 << 15) * c["d2h"]))
        for x in z + t:
            d.lookup(10, x)
        for x in k:
            d.lookup(12, x)
        for f in ("d1l", "d1h", "d2l", "d2h"):
            d.lookup(15, c[f])

    def samp_sums(self):
        """Z, A = sum_{v<y} p, P = p_y per row from p: one sumcheck of
        sum_{r,v} eq(rho, r) (a + b [v < y_r] + g [v = y_r]) p(r, v) = a Z(rho) + b A(rho) + g P(rho);
        the verifier evaluates both public indicator tables itself."""
        ctx, sh = self.ctx, self.sh
        F, G = ctx.F, ctx.G
        r = log2(sh.R)
        rr = ctx.chal(r)
        co = [ctx.chal() for _ in range(3)]
        y = self.targets.reshape(-1).to(_dev())
        vlist = torch.arange(self.cfg.V, device=_dev()).unsqueeze(0)
        pub = [(vlist < y.unsqueeze(1)).to(torch.int64), (vlist == y.unsqueeze(1)).to(torch.int64)]
        vals = None
        if ctx.P:
            S_ = self.fin["samp"]
            vals = np.stack([mle_host(ctx, S_[f], rr) for f in ("Z", "A", "P")], axis=1)
        vals = ctx.send_vals(vals, (ctx.D, 3))
        claim = F.zero
        for i in range(3):
            claim = F.add(claim, F.mul(F.c(co[i]), F.c(vals[:, i])))
        polys = [Poly.var(i) for i in range(4)]
        expr = ("*", ("p", 0), ("*", ("+", ("c", 0), ("+", ("*", ("c", 1), ("p", 1)), ("*", ("c", 2), ("p", 2)))),
                                ("p", 3)))
        prog = Prog(polys, expr, [np.asarray(x, dtype=np.int64) for x in co])
        shapes = [(ROW, sh.cbv), (FULL, 0), (FULL, 0), (FULL, 0)]

        def tabs():
            return [Tab(G.eq_table(rr), ROW, sh.cbv, shared=True), Tab(rm(pub[0], sh.cbv), FULL),
                    Tab(rm(pub[1], sh.cbv), FULL), Tab(rm(self.fin["samp"]["p"], sh.cbv), FULL)]
        rho, fin = ctx.sumcheck(b"samp-sums", sh.cbv + r, sh.Kv, shapes, np.array(claim),
                                tabs=tabs if ctx.proving else None, prog=prog)
        check_eq(ctx, b"samp-sums", fin[0], rr, rho[:, sh.cbv:])
        if not ctx.P:
            ys = [int(v) for v in y.tolist()]
            for i in range(2):
                X = lambda i=i: rm(pub[i], sh.cbv).reshape(sh.Kv, -1)
                exp = mle_pub(ctx, lambda i=i: mlepub.vocab_indicator(rho, ys, sh.cbv, sh.Kv, ("lt", "eq")[i]),
                              lambda i=i: np.stack([mle_host(ctx, X(i)[kb], rho) for kb in range(sh.Kv)], axis=1))
                if not np.array_equal(exp, fin[1 + i] % P):
                    raise ValueError("sampling indicator tables")
        R_blk(ctx, "f.p")(rho, fin[3])
        for i, f in enumerate(("Z", "A", "P")):
            ctx.claim("f.s." + f, rr, {0: vals[:, i]})

    def public_gather(self, pz, vz):
        """zmax(r) = sum_{r, v} eq(pz, r) [v = y_r] z(r, v)."""
        ctx, sh = self.ctx, self.sh
        r = log2(sh.R)
        y = self.targets.reshape(-1)
        shapes = [(ROW, sh.cbv), (FULL, 0), (FULL, 0)]
        terms = [(np.array(ctx.F.one), [0, 1, 2])]

        def oh():
            OH = torch.zeros((sh.R, self.cfg.V), dtype=torch.int64, device=_dev())
            OH[torch.arange(sh.R, device=_dev()), y.to(_dev())] = 1
            return rm(OH, sh.cbv)

        def tabs():
            return [Tab(ctx.G.eq_table(pz), ROW, sh.cbv, shared=True), Tab(oh(), FULL),
                    Tab(rm(self.fin["z"], sh.cbv), FULL)]
        rho, fin = ctx.sumcheck(b"zmax", sh.cbv + r, sh.Kv, shapes, vz[:, 0], tabs=tabs if ctx.proving else None,
                                terms=terms)
        check_eq(ctx, b"zmax", fin[0], pz, rho[:, sh.cbv:])
        if not ctx.P:
            exp = mle_pub(ctx, lambda: mlepub.vocab_indicator(rho, [int(v) for v in y.tolist()], sh.cbv, sh.Kv, "eq"),
                          lambda: np.stack([mle_host(ctx, oh().reshape(sh.Kv, -1)[k], rho) for k in range(sh.Kv)], axis=1))
            if not np.array_equal(exp, fin[1] % P):
                raise ValueError("one-hot of targets")
        ctx.claim("f.z", rho, fin[2])

    def rq_norm_dom(self, db, h, qn, qpn, rpre, npre, getr, getn):
        """Requant8 + NormScale lookups over a row-major domain (columns added to db)."""
        ctx, sh = self.ctx, self.sh
        cb = sh.cb
        q = db.col(qn, lambda: rm(getr()["q"], cb), R_blk(ctx, qn))
        qp = db.col(qpn, lambda: rm(getn()["qp"], cb), R_blk(ctx, qpn))
        e = db.col(rpre + "e", lambda: f32(getr()["e"]), R_blk(ctx, rpre + "e"), kind=ROW, c=cb, shared=True)
        s = db.col(rpre + "s", lambda: f32(getr()["s"]), R_blk(ctx, rpre + "s"), kind=ROW, c=cb, shared=True)
        mu = db.col(npre + "mu", lambda: f32(getn()["mu"]), R_blk(ctx, npre + "mu"), kind=ROW, c=cb, shared=True)
        d = db.dom
        d.lookup(e + 1, 2 * h - 2 * s * q + s)
        d.lookup(tb.STAR8, q + 127)
        d.lookup(15, q * mu + (1 << 14) - (1 << 15) * qp)
        d.lookup(tb.STAR8, qp + 127)
        return q, qp

    # ------------------------------------------------------------ layer steps (reverse order)
    def step_mo(self, l):
        """Down-projection shift + SatAdd; per-row domain of G_mi; open G_mo."""
        ctx, sh, M = self.ctx, self.sh, self.M
        cb, t = sh.cb, sh.t
        p = f"{l}."
        Ly = M.layers[l]
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        db = DB(ctx, p + "mo", cb + t, sh.Kd)
        acc = db.col("accd", lambda: rm(Wt().accd, cb), R_virt(ctx, (l, "accd")))
        s2 = db.col(p + "s2", lambda: rm(Wt().s2, cb), R_blk(ctx, p + "s2"))
        o2 = db.col(p + "os2", lambda: rm(Wt().os2, cb), R_blk(ctx, p + "os2"))
        s1 = db.col(p + "s1", lambda: rm(Wt().s1, cb), R_blk(ctx, p + "s1"))
        o1 = db.col(p + "os1", lambda: rm(Wt().os1, cb), R_blk(ctx, p + "os1"))
        sr = {f: db.col(p + "sd." + f, (lambda f=f: f32(sr_pack(Wt().kd)[f])), R_blk(ctx, p + "sd." + f),
                        kind=ROW, c=cb, shared=True) for f in ("s", "R", "tR", "tB", "oB") + SRA_F}
        hm = hexpr(s1, o1)
        c2 = s2 - hm
        d = db.dom
        d.lookup(sr["tR"], acc + sr["R"] - sr["s"] * c2)
        d.lookup(sr["tB"], c2 + sr["oB"])
        d.lookup(sr["tA"], sr["f"] * acc + sr["oA"])
        d.lookup(24, satlookup(s2, o2))
        d.zero(o2 * o2 * o2 - o2)
        db.run()
        # per-row domain of G_mi
        db = DB(ctx, p + "mi.rows", t)
        self.rq_rows(db, p + "r2.", lambda: Wt().r2, sh.Kd, asel_key=(l, "asel2"))
        nsc = self.ns_rows(db, p + "n2.", lambda: Wt().n2, (l, "sig2"))
        if self.cfg.moe:
            q = p + "e."
            e2 = db.col(q + "r2.e", lambda: f32(Wt().moe["r2"]["e"]), R_blk(ctx, q + "r2.e"))
            Et = db.col(q + "Et", lambda: f32(Wt().moe["Et"]), R_blk(ctx, q + "Et"))
            fd = db.col(q + "fd", lambda: f32(Wt().moe["fd"]), R_blk(ctx, q + "fd"))
            Kc = -(M.eA - mdl.W_FRAC + M.f_h)
            kdv = Kc - e2 - Et + nsc["Eh"]                   # kd = min(kdv, 24): fd = [kdv > 24]
            db.dom.zero(fd * (1 - fd))
            db.dom.lookup(24, fd * (kdv - 25))
            self.sr_rows(db, p + "sd.", lambda: Wt().kd, fd * 24 + (1 - fd) * kdv, left=True)
            db.run()
            if ctx.active:
                self.ctx.open(p + "mo", self.tensors(p + "mo", Wt()) if ctx.P else None)
            return                                 # the down product's claims stay for the MoE (moe.down)
        kgm = cap_col(ctx, db, p + "kgm", lambda: Wt().n2["Eh"], lambda: self.M.layers[l].kgm_t, lambda: self.M.layers[l].kgm)
        db.dom.lookup(5, kgm - nsc["Eh"])                    # gate shift kcg_n + E_h <= 15
        # E_g = max(E_h, K_gl): f = [E_h < K_gl] (both ranges far below 2^24, so the lookups cannot wrap)
        kgl = cap_col(ctx, db, p + "kgl", lambda: Wt().n2["Eh"], lambda: self.M.layers[l].kgl_t, lambda: self.M.layers[l].kgl)
        eg = {f: db.col(p + "g." + f, (lambda f=f: f32(Wt().eg[f])), R_blk(ctx, p + "g." + f)) for f in EG_F}
        d = db.dom
        d.zero(eg["f"] * (1 - eg["f"]))
        d.zero(eg["E"] - eg["f"] * kgl - (1 - eg["f"]) * nsc["Eh"])
        d.lookup(24, (1 - eg["f"]) * (nsc["Eh"] - kgl))
        d.lookup(24, eg["f"] * (kgl - 1 - nsc["Eh"]))
        d.lookup(tb.POW, eg["E"] + tb.POW_OFF, eg["pE"])
        ru = self.rq_rows(db, p + "ru.", lambda: Wt().ru, sh.Kf, 2, 22, asel_key=(l, "aselu"))
        rmm = self.rq_rows(db, p + "rm.", lambda: Wt().rm, sh.Kf, asel_key=(l, "aselm"), B=self.cfg.down_b)
        self.sr_rows(db, p + "sd.", lambda: Wt().kd, lparam("Kcd") - (rmm["e"] + ru["e"] - nsc["Eh"]), left=True)
        db.run()
        if ctx.active:
            self.ctx.open(p + "mo", self.tensors(p + "mo", Wt()) if ctx.P else None)
            # down product claims -> per-slice partial values
            cl = []
            for pt, vals in ctx.take((l, "accd")):
                po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(sh.Kd))
                cl.append((po, pt[:, cb:], v))
            self.down_claims = []
            for po, pm, v in cl:
                parts = None
                if ctx.P:
                    G = ctx.G
                    eo = G.eq_table(po)[:, :M.cfg.d].contiguous()
                    Wf = G.rowdot(Ly.Wd.reshape(-1), eo, M.cfg.dff, M.cfg.d)              # (D, dff)
                    Af = G.coldot(rm(Wt().rm["q"], cb), G.eq_table(pm), sh.Kf, sh.T, 1 << cb)
                    prod = G.mul(Af, Wf).to(torch.int64)
                    parts = (torch.stack([prod[:, b0 << cb:b1 << cb].sum(dim=1) for b0, b1 in self.slices()],
                                         dim=1) % P).to(torch.int32)
                parts = ctx.send_vals(parts, (ctx.D, len(self.slices())))
                if not ctx.P:
                    tot = ctx.F.zero
                    for s_ in range(parts.shape[1]):
                        tot = ctx.F.add(tot, ctx.F.c(parts[:, s_]))
                    if tot != ctx.F.c(v):
                        raise ValueError("down projection partial sums")
                self.down_claims.append((po, pm, parts))
            # gather targets -> per-slice partials
            self.gath = {}
            for key in ("aselu", "aselm"):
                (pt, vals), = ctx.take((l, key))
                self.gath[key] = (pt, vals[:, 0])

    def step_mlp(self, l, s):
        ctx, sh, M = self.ctx, self.sh, self.M
        cb, t = sh.cb, sh.t
        p = f"{l}."
        q = f"{l}.m{s}."
        Ly = M.layers[l]
        b0, b1 = self.slices()[s]
        Ks = b1 - b0
        cols = slice(b0 << cb, b1 << cb)
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        db = DB(ctx, q + "dom", cb + t, Ks)
        accg = db.col("accg", lambda: rm(Wt().accg[:, cols], cb), R_virt(ctx, (l, s, "accg")))
        accu = db.col("accu", lambda: rm(Wt().accu[:, cols], cb), R_virt(ctx, (l, s, "accu")))
        c = db.col(q + "c", lambda: rm(Wt().c[:, cols], cb), R_blk(ctx, q + "c"))
        a = db.col(q + "a", lambda: rm(Wt().a[:, cols], cb), R_blk(ctx, q + "a"))
        qu = db.col(q + "qu", lambda: rm(Wt().ru["q"][:, cols], cb), R_blk(ctx, q + "qu"))
        qm = db.col(q + "qm", lambda: rm(Wt().rm["q"][:, cols], cb), R_blk(ctx, q + "qm"))
        rowc = lambda name, fn: db.col(name, fn, R_blk(ctx, name), kind=ROW, c=cb, shared=True)
        Eg = rowc(p + "g.E", lambda: f32(Wt().eg["E"]))
        pEg = rowc(p + "g.pE", lambda: f32(Wt().eg["pE"]))
        eu = rowc(p + "ru.e", lambda: f32(Wt().ru["e"]))
        su = rowc(p + "ru.s", lambda: f32(Wt().ru["s"]))
        em = rowc(p + "rm.e", lambda: f32(Wt().rm["e"]))
        sm = rowc(p + "rm.s", lambda: f32(Wt().rm["s"]))
        kgc = (M.eG - Ly.eWg)[cols]
        mdl.check((kgc >= 0) & (kgc < 31), "gate column shift outside [0, 31)")
        kgcol = lambda: rm(kgc.unsqueeze(0).expand(1, -1), cb)
        pkcol = lambda: rm(torch.bitwise_left_shift(torch.ones_like(kgc), kgc).unsqueeze(0), cb)

        def colpub(X):
            def fn(pt):
                Xr = X().reshape(Ks, -1)
                return np.stack([mle_host(ctx, Xr[k], pt) for k in range(Ks)], axis=1)
            return fn
        kg = db.col("kgc", kgcol, R_pub(ctx, colpub(kgcol)), kind=COL, c=cb)
        pk = db.col("pkg", pkcol, R_pub(ctx, colpub(pkcol)), kind=COL, c=cb)
        d = db.dom
        sg = pk * pEg
        d.lookup(kg + Eg, accg + sg * INV2 - sg * c)
        d.lookup(tb.SILU, c + (1 << 14), a)
        d.lookup(eu, accu + su * INV2 - su * qu)
        d.lookup(tb.STAR8, qu + 127)
        d.lookup(em + 1, 2 * a * qu - 2 * sm * qm + sm)
        Bd = self.cfg.down_b
        if Bd == 127:
            d.lookup(tb.STAR8, qm + 127)
        else:                                                   # |qm| <= Bd = 2^nb - 1
            nb = (Bd + 1).bit_length() - 1
            assert Bd == (1 << nb) - 1
            d.lookup(nb + 1, qm + Bd)
            d.lookup(nb + 1, Bd - qm)
        db.run()
        if not ctx.active:
            return
        F = ctx.F
        # minimality gathers over this slice: partial sums of the per-row targets
        (pu, _), (pm_, _) = self.gath["aselu"], self.gath["aselm"]
        if not ctx.P:
            assert np.array_equal(pu, pm_)
        nsl = len(self.slices())
        if s == 0:
            self.gparts = {}
            for key, (pt, v) in self.gath.items():
                parts = None
                if ctx.P:
                    W = Wt()
                    jb = W.ru["jb"] if key == "aselu" else W.rm["jb"]
                    j = sum(b << i for i, b in enumerate(jb))
                    X = W.accu if key == "aselu" else W.m
                    sel = X.gather(1, j.unsqueeze(1)).squeeze(1)
                    eqr = ctx.G.eq_table(pt)
                    pairs = []
                    for si, (c0, c1) in enumerate(self.slices()):
                        msk = ((j >> cb) >= c0) & ((j >> cb) < c1)
                        pairs.append((f32(torch.where(msk, sel, torch.zeros_like(sel))), eqr))
                    parts = ctx.G.dots_dev(pairs).T.contiguous()
                parts = ctx.send_vals(parts, (ctx.D, nsl))
                if not ctx.P:
                    tot = F.zero
                    for si in range(nsl):
                        tot = F.add(tot, F.c(parts[:, si]))
                    if tot != F.c(v):
                        raise ValueError("gather partial sums")
                self.gparts[key] = parts
        gam = ctx.chal()
        claim = np.zeros(ctx.D, dtype=np.int64) if ctx.P else \
            F.add(F.c(self.gparts["aselu"][:, s]), F.mul(F.c(gam), F.c(self.gparts["aselm"][:, s])))
        # tables: 0 eq rows, 1 OHu, 2 OHm, 3 accu, 4 a, 5 qu
        polys = [Poly.var(1) * Poly.var(3), Poly.var(2) * Poly.var(4) * Poly.var(5), Poly.var(0)]
        prog = Prog(polys, ("*", ("p", 2), ("+", ("p", 0), ("*", ("c", 0), ("p", 1)))),
                    [gam if ctx.P else np.asarray(gam, dtype=np.int64)])
        shapes = [(ROW, cb), (FULL, 0), (FULL, 0), (FULL, 0), (FULL, 0), (FULL, 0)]

        def ohs(jb):
            j = sum(b << i for i, b in enumerate(jb))
            n_ = Ks * sh.T << cb
            OH = torch.zeros(n_ + 1, dtype=torch.int32, device=_dev())     # last: sink of out-of-slice rows
            m = torch.arange(sh.T, device=_dev())
            blk = (j >> cb) - b0
            ok = (blk >= 0) & (blk < Ks)
            flat = torch.where(ok, ((blk * sh.T + m) << cb) + (j & ((1 << cb) - 1)), n_)
            OH.index_fill_(0, flat, 1)
            return OH[:n_]

        def tabs():
            W = Wt()
            return [Tab(ctx.G.eq_table(pu), ROW, cb, shared=True), Tab(ohs(W.ru["jb"]), FULL),
                    Tab(ohs(W.rm["jb"]), FULL), Tab(rm(W.accu[:, cols], cb), FULL),
                    Tab(rm(W.a[:, cols], cb), FULL), Tab(rm(W.ru["q"][:, cols], cb), FULL)]
        rho, fin = ctx.sumcheck(b"mlp-gather", cb + t, Ks, shapes, np.array(claim),
                                tabs=tabs if ctx.proving else None, prog=prog)
        check_eq(ctx, b"mlp-gather", fin[0], pu, rho[:, cb:])
        ctx.put((l, s, "accu"), (rho, fin[3]))
        ctx.claim(q + "a", rho, fin[4])
        ctx.claim(q + "qu", rho, fin[5])
        # one-hot claims on this slice's blocks: OH over the full row restricted to blocks [b0, b1)
        for key, jbn, idx in (("u", p + "ru.jb", 1), ("m", p + "rm.jb", 2)):
            self.oh_slice(jbn, rho, fin[idx], b0, Ks, (lambda key=key: Wt().ru["jb"] if key == "u" else Wt().rm["jb"]))
        # down projection partial over this slice
        cl = []
        for po, pm, parts in self.down_claims:
            cl.append((po, pm, parts[:, s]))
        self.matmul_rows(b"down", q + "qm", (lambda: rm(Wt().rm["q"][:, cols], cb)) if ctx.P else None, Ks, sh.T,
                         p + "Wd", Ly.Wd[b0 << cb:b1 << cb] if ctx.P else None, sh.Kd, b0, cl)
        # gate / up products
        clg = []
        for pt, vals in ctx.take((l, s, "accg")):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(Ks))
            clg.append((po, pt[:, cb:], v))
        clu = []
        for pt, vals in ctx.take((l, s, "accu")):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(Ks))
            clu.append((po, pt[:, cb:], v))
        self.matmul_cols(b"gate", p + "qp2", (lambda: rm(Wt().n2["qp"], cb)) if ctx.P else None, sh.Kd, sh.T,
                         p + "Wg", Ly.Wg if ctx.P else None, sh.Kf, b0, Ks, clg)
        self.matmul_cols(b"up", p + "qp2", (lambda: rm(Wt().n2["qp"], cb)) if ctx.P else None, sh.Kd, sh.T,
                         p + "Wu", Ly.Wu if ctx.P else None, sh.Kf, b0, Ks, clu)
        ctx.open(p + f"mlp{s}", self.tensors(p + f"mlp{s}", Wt()) if ctx.P else None)

    def oh_slice(self, jbn, rho, vals, b0, Ks, getbits):
        """OH claims on blocks b0.. b0+Ks of the full row (Kf blocks)."""
        ctx, sh = self.ctx, self.sh
        cb = sh.cb
        F = ctx.F
        # values of OH on every block of the full row; the slice's blocks are checked here
        full = None
        if ctx.P:
            bits = getbits()
            j = sum(b << i for i, b in enumerate(bits))
            G = ctx.G
            prod = G.mul(G.eq_table(rho[:, cb:]), G.eq_table(rho[:, :cb])[:, j & ((1 << cb) - 1)].contiguous())
            acc = torch.zeros((ctx.D, sh.Kf), dtype=torch.int64, device=_dev())
            acc.index_add_(1, j >> cb, prod.to(torch.int64))
            full = (acc % P).to(torch.int32)
        full = ctx.send_vals(full, (ctx.D, sh.Kf))
        for k in range(Ks if not ctx.P else 0):
            if F.c(full[:, b0 + k]) != F.c(vals[:, k]):
                raise ValueError("one-hot slice values")
        if not hasattr(self, "_ohpend"):
            self._ohpend = {}
        self._ohpend.setdefault(jbn, []).append((rho, full))

    def flush_oh(self, l):
        """Row checks of the slice one-hot claims (one per slice and kind)."""
        ctx, sh = self.ctx, self.sh
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        for jbn, lst in getattr(self, "_ohpend", {}).items():
            key = "ru" if ".ru." in jbn else "rm"
            # only claims of this layer
            mine = [x for x in lst if jbn.startswith(f"{l}.")]
            if not mine:
                continue
            oh_check(ctx, b"mlp-oh", jbn, (sh.cb, nbits(sh.Kf)), mine, sh.t,
                     lambda key=key: getattr(Wt(), key)["jb"])
            self._ohpend[jbn] = [x for x in lst if not jbn.startswith(f"{l}.")]

    def matmul_rows(self, label, A_name, A_data, Ks, Mrows, W_name, W_rows, Kout, b0, claims):
        """acc partial = A_slice (Mrows x Ks 2^cb) . W[rows of slice] for claims at (po, pm)."""
        ctx = self.ctx
        if not claims:
            return
        sh = self.sh
        cb = sh.cb
        # reuse matmul with a temporary weight-name mapping: block (tb_local, ob) -> (b0 + tb_local, ob)
        tmp = f"__tmp_{W_name}"
        info = ctx.tinfo[W_name]
        from .proto import TInfo
        ctx.tinfo[tmp] = TInfo(tmp, Ks * Kout, info.c, info.group)
        matmul(ctx, label, A_name, A_data, Ks, Mrows, tmp, W_rows, Kout, cb, claims)
        for pt, vals in ctx.claims.pop(tmp, []):
            ctx.claim(W_name, pt, {(b0 + b // Kout) * Kout + b % Kout: v for b, v in vals.items()})
        del ctx.tinfo[tmp]

    def matmul_cols(self, label, A_name, A_data, Kin, Mrows, W_name, W_full, Kf, b0, Ks, claims):
        """acc over output blocks [b0, b0+Ks) of W (Kin 2^cb x Kf 2^cb)."""
        ctx = self.ctx
        if not claims:
            return
        sh = self.sh
        cb = sh.cb
        tmp = f"__tmp_{W_name}"
        info = ctx.tinfo[W_name]
        from .proto import TInfo
        ctx.tinfo[tmp] = TInfo(tmp, Kin * Ks, info.c, info.group)
        Wsl = W_full[:, b0 << cb:(b0 + Ks) << cb].contiguous() if ctx.P else None
        matmul(ctx, label, A_name, A_data, Kin, Mrows, tmp, Wsl, Ks, cb, claims)
        for pt, vals in ctx.claims.pop(tmp, []):
            ctx.claim(W_name, pt, {(b // Ks) * Kf + b0 + b % Ks: v for b, v in vals.items()})
        del ctx.tinfo[tmp]

    def step_mi(self, l):
        ctx, sh = self.ctx, self.sh
        cb, t = sh.cb, sh.t
        p = f"{l}."
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        self.flush_oh(l)
        db = DB(ctx, p + "mi", cb + t, sh.Kd)
        s1 = db.col(p + "s1", lambda: rm(Wt().s1, cb), R_blk(ctx, p + "s1"))
        o1 = db.col(p + "os1", lambda: rm(Wt().os1, cb), R_blk(ctx, p + "os1"))
        self.rq_norm_dom(db, hexpr(s1, o1), p + "q2", p + "qp2", p + "r2.", p + "n2.", lambda: Wt().r2, lambda: Wt().n2)
        db.run()
        if not ctx.active:
            return
        (pa, va), = ctx.take((l, "asel2"))
        (ps, vs), = ctx.take((l, "sig2"))
        self.gather_rowsum(b"mi-g", l, t, pa, sh.Kd, dict(
            cols=[(p + "s1", lambda: rm(Wt().s1, cb), R_blk(ctx, p + "s1")),
                  (p + "os1", lambda: rm(Wt().os1, cb), R_blk(ctx, p + "os1")),
                  (p + "q2", lambda: rm(Wt().r2["q"], cb), R_blk(ctx, p + "q2"))],
            h=lambda c: hexpr(c[0], c[1]), q=2, asel=va[:, 0], sig=vs[:, 0]),
            p + "r2.jb", lambda: Wt().r2["jb"])
        ctx.open(p + "mi", self.tensors(p + "mi", Wt()) if ctx.P else None)

    def step_ao(self, l):
        ctx, sh, M = self.ctx, self.sh, self.M
        cb, t = sh.cb, sh.t
        p = f"{l}."
        Ly = M.layers[l]
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        # O-proj shift + SatAdd over T x d
        with domain_batch(ctx):
            db = DB(ctx, p + "oproj", cb + t, sh.Kd)
            acc = db.col("acco", lambda: rm(Wt().acc_o, cb), R_virt(ctx, (l, "acco")))
            s1 = db.col(p + "s1", lambda: rm(Wt().s1, cb), R_blk(ctx, p + "s1"))
            o1 = db.col(p + "os1", lambda: rm(Wt().os1, cb), R_blk(ctx, p + "os1"))
            h = self.hin_cols(db, l, Wt)
            sr = {f: db.col(p + "so." + f, (lambda f=f: f32(sr_pack(Wt().ko)[f])), R_blk(ctx, p + "so." + f),
                            kind=ROW, c=cb, shared=True) for f in ("s", "R", "tR", "tB", "oB") + SRA_F}
            c1 = s1 - h
            d = db.dom
            d.lookup(sr["tR"], acc + sr["R"] - sr["s"] * c1)
            d.lookup(sr["tB"], c1 + sr["oB"])
            d.lookup(sr["tA"], sr["f"] * acc + sr["oA"])
            d.lookup(24, satlookup(s1, o1))
            d.zero(o1 * o1 * o1 - o1)
            db.run()
            # per-row: Requant8(o) minimality + SR(k_o)
            db = DB(ctx, p + "ao.rows", t)
            ro = self.rq_rows(db, p + "ro.", lambda: Wt().ro, sh.Kq, asel_key=(l, "aselo"))
            self.sr_rows(db, p + "so.", lambda: Wt().ko, lparam("Kco") - ro["e"], left=True)
            db.run()
            # Requant8(o) over T x nq
            db = DB(ctx, p + "rqo", cb + t, sh.Kq)
            o = db.col(p + "o", lambda: rm(Wt().o, cb), R_blk(ctx, p + "o"))
            qo = db.col(p + "qo", lambda: rm(Wt().ro["q"], cb), R_blk(ctx, p + "qo"))
            e = db.col(p + "ro.e", lambda: f32(Wt().ro["e"]), R_blk(ctx, p + "ro.e"), kind=ROW, c=cb, shared=True)
            s = db.col(p + "ro.s", lambda: f32(Wt().ro["s"]), R_blk(ctx, p + "ro.s"), kind=ROW, c=cb, shared=True)
            db.dom.lookup(e + 1, 2 * o - 2 * s * qo + s)
            db.dom.lookup(tb.STAR8, qo + 127)
            db.run()
            # o = MulRound(N, recip, k) elementwise over (c, m) x nh
            db = DB(ctx, p + "nd", sh.ldh + t, sh.nh)
            N = db.col("Nsum", lambda: rm(Wt().Ns.reshape(sh.T, -1), sh.ldh), R_virt(ctx, (l, "Nsum")))
            xh = db.col(p + "nd.xh", lambda: rm(Wt().ndxh, sh.ldh), R_blk(ctx, p + "nd.xh"))
            tt = db.col(p + "nd.t", lambda: rm(Wt().ndt, sh.ldh), R_blk(ctx, p + "nd.t"))
            ov = db.col("o_view", lambda: rm(Wt().o, sh.ldh), R_virt(ctx, (l, "o_view")))
            sM = {f: db.col(p + "nM." + f, (lambda f=f: f32(sr_pack(Wt().nd["kp"])[f])), R_blk(ctx, p + "nM." + f),
                            kind=ROW, c=sh.ldh) for f in ("s", "R", "tR", "tB", "oB")}
            rc = db.col(p + "nd.recip", lambda: f32(Wt().nd["recip"]), R_blk(ctx, p + "nd.recip"), kind=ROW, c=sh.ldh)
            d = db.dom
            xl = N - (1 << 12) * xh
            d.lookup(12, xl)
            d.lookup(12, xh + (1 << 11))
            d.lookup(12, xl * rc - (1 << 12) * tt)
            d.lookup(18, tt + (1 << 17))
            d.lookup(sM["tR"], xh * rc + tt + sM["R"] - sM["s"] * ov)
            d.lookup(sM["tB"], ov + sM["oB"])
            db.run()
            # per-(row, head): D = 2^E delta + rhoD, recip, k_m = max(7, E), MulRound shift
            db = DB(ctx, p + "nd.rows", t, sh.nh)
            Dc = db.col("D", lambda: f32(Wt().D.T.contiguous()), R_virt(ctx, (l, "D")))
            srE = {f: db.col(p + "nE." + f, (lambda f=f: f32(sr_pack(Wt().nd["E"])[f])), R_blk(ctx, p + "nE." + f))
                   for f in SR_F}
            srM = {f: db.col(p + "nM." + f, (lambda f=f: f32(sr_pack(Wt().nd["kp"])[f])), R_blk(ctx, p + "nM." + f))
                   for f in SR_F}
            nc = {f: db.col(p + "nd." + f, (lambda f=f: f32(Wt().nd[f])), R_blk(ctx, p + "nd." + f)) for f in ND_F}

            def km_route(pt, vals):
                for g in range(sh.nkv):
                    ctx.claim(f"{l}.g{g}.nK.k", pt, {i: vals[:, g * sh.grp + i] for i in range(sh.grp)})
            kmc = db.col("nKk", lambda: f32(sr_pack(Wt().nd["km"])["k"]), km_route)
            if getattr(Ly, "_vsh", None) is None:
                Ly._vsh = [int(v) for v in Ly.vshift.tolist()]       # model constants, read once
            vsh = Ly._vsh
            if getattr(Ly, "_vsh_dev", None) is None:
                Ly._vsh_dev = torch.tensor(vsh, dtype=torch.int64, device=_dev())
            vs = db.col("vshift", lambda: f32(Ly._vsh_dev),
                        R_pub(ctx, lambda pt: pub_const_rows(ctx, vsh)), kind=ROW, c=t)
            dl, rD, rc, fk = (nc[f] for f in ND_F)
            E = srE["k"] + tb.SR_KMIN
            km = kmc + tb.SR_KMIN
            d = db.dom
            d.lookup(tb.SR, *[srE[f] for f in SR_F])
            d.lookup(tb.SR, *[srM[f] for f in SR_F])
            d.lookup(srE["tR"], rD)
            d.lookup(tb.RECIP, dl - (1 << 15), rc)
            d.zero(Dc - srE["s"] * dl - rD)
            d.zero(fk * (1 - fk))
            d.lookup(4, fk * (E - mdl.KM_MIN) + (1 - fk) * (mdl.KM_MIN - 1 - E))
            d.lookup(5, E + 5)
            d.lookup(5, 14 - E)
            d.lookup(22, (1 - fk) * Dc)
            d.zero(km - mdl.KM_MIN - fk * (E - mdl.KM_MIN))
            d.zero(srM["k"] + tb.SR_KMIN - (mdl.O_BASE - 12 + E - km + vs))
            db.run()
        if not ctx.active:
            return
        # o view claims -> point claim on o
        for pt, vals in ctx.take((l, "o_view")):
            # head block hh -> o column hh*dh + c; o layout [kb][m][(hib, c)]
            ptc, ptm = pt[:, :sh.ldh], pt[:, sh.ldh:]
            tau = ctx.chal(nbits(sh.nh))
            v = ctx.blockcomb(tau, vals)
            lh = log2(sh.hb)
            assert nbits(sh.nh) - lh == nbits(sh.Kq)
            ctx.point_claim(p + "o", cat(ptc, tau[:, :lh], ptm, tau[:, lh:]), v,
                            data=rm(Wt().o, cb) if ctx.P else None)
        # Nsum -> Nr (sum over key blocks)
        for pt, vals in ctx.take((l, "Nsum")):
            half = np.zeros((ctx.D, sh.nbb), dtype=np.int64)
            half[0, :] = INV2
            sc_ = pow(INV2, sh.nbb, P)
            for g in range(sh.nkv):
                ctx.claim(f"{l}.g{g}.Nr", cat(pt[:, :sh.ldh], half, pt[:, sh.ldh:]),
                          {i: lz_scale(vals[:, g * sh.grp + i], sc_) for i in range(sh.grp)})
        # D -> per group virtual
        for pt, vals in ctx.take((l, "D")):
            for g in range(sh.nkv):
                ctx.put((l, g, "D"), (pt, vals[:, g * sh.grp:(g + 1) * sh.grp]))
        # gather for Requant8(o)
        (pa, va), = ctx.take((l, "aselo"))
        self.gather_rowsum(b"ao-g", l, t, pa, sh.Kq, dict(
            cols=[(p + "o", lambda: rm(Wt().o, cb), R_blk(ctx, p + "o"))],
            h=lambda c: c[0], q=None, asel=va[:, 0], sig=None), p + "ro.jb", lambda: Wt().ro["jb"])
        # O-proj product
        cl = []
        for pt, vals in ctx.take((l, "acco")):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(sh.Kd))
            cl.append((po, pt[:, cb:], v))
        matmul(ctx, b"oproj", p + "qo", (lambda: rm(Wt().ro["q"], cb)) if ctx.P else None, sh.Kq, sh.T,
               p + "Wo", Ly.Wo if ctx.P else None, sh.Kd, cb, cl)
        ctx.open(p + "ao", self.tensors(p + "ao", Wt()) if ctx.P else None)

    def step_att_pre(self, l, gs):
        """Domains of KV groups gs batched as one lookup set each: ShiftRound of N, its row table,
        key gather, causal key range and the key table side."""
        ctx, sh = self.ctx, self.sh
        t, dh = sh.t, sh.dh
        p = f"{l}."
        nm = f"{l}.a{gs[0]}-{gs[-1]}."
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        grp = sh.grp
        # ---- Nr = ShiftRound(N^(b), k_m) over (c, kb, m) x grp
        with domain_batch(ctx):
            db = DB(ctx, nm + "nr", sh.ldh + sh.nbb + t, grp)
            for g in gs:
                q = f"{l}.g{g}."
                A = (lambda g=g: Wt().att[g]) if ctx.P else None
                kmw = (lambda g=g: Wt().nd["km"][g * grp:(g + 1) * grp]) if ctx.P else None
                Nb = db.col(f"Nb{g}", lambda g=g: self.nb_data(l, g), R_virt(ctx, (l, g, "Nb")))
                Nr = db.col(q + "Nr", lambda A=A: torch.cat([f32(x) for x in A()["Nr"]]), R_blk(ctx, q + "Nr"))
                sK = {f: db.col(q + "nK." + f, (lambda f=f, kmw=kmw: f32(sr_pack(kmw())[f])), R_blk(ctx, q + "nK." + f),
                                kind=ROW, c=sh.ldh + sh.nbb) for f in ("s", "R", "tR", "tB", "oB")}
                db.dom.lookup(sK["tR"], Nb + sK["R"] - sK["s"] * Nr)
                db.dom.lookup(sK["tB"], Nr + sK["oB"])
            with self.tm("a.nr"):
                db.run()
            db = DB(ctx, nm + "nK.rows", t, grp)
            for g in gs:
                q = f"{l}.g{g}."
                kmw = (lambda g=g: Wt().nd["km"][g * grp:(g + 1) * grp]) if ctx.P else None
                sK = {f: db.col(q + "nK." + f, (lambda f=f, kmw=kmw: f32(sr_pack(kmw())[f])), R_blk(ctx, q + "nK." + f))
                      for f in SR_F}
                db.dom.lookup(tb.SR, *[sK[f] for f in SR_F])
            db.run()
            # ---- key gather (witness side) over (c, m) x grp
            db = DB(ctx, nm + "kg", sh.ldh + t, grp)
            ci = db.col("ci", lambda: f32(torch.arange(dh, device=_dev())),
                        R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.iota(pt), lambda: mle_host(ctx, torch.arange(dh, device=_dev()), pt)).reshape(-1, 1)),
                        kind=COL, c=sh.ldh, shared=True)
            for g in gs:
                q = f"{l}.g{g}."
                A = (lambda g=g: Wt().att[g]) if ctx.P else None
                js = db.col(q + "js", lambda A=A: torch.cat([f32(x) for x in A()["js"]]), R_blk(ctx, q + "js"),
                            kind=ROW, c=sh.ldh)
                Kg = db.col(q + "Kg", lambda A=A: torch.cat([f32(x) for x in A()["Kg"]]), R_blk(ctx, q + "Kg"))
                db.dom.lookup(lparam("Dyn") + g, js, ci, Kg, dynamic=True)
            db.run()
            # ---- causal range of the selected keys, per (row, head)
            db = DB(ctx, nm + "js.rows", t, grp)
            mi = db.col("mi", lambda: f32(torch.arange(sh.T, device=_dev())),
                        R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.iota(pt), lambda: mle_host(ctx, torch.arange(sh.T, device=_dev()), pt)).reshape(-1, 1)),
                        kind=COL, c=t, shared=True)
            ss = db.col("ss", lambda: f32((torch.arange(sh.T, device=_dev()) >> sh.ls) << sh.ls),
                        R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.iota(pt, sh.ls), lambda: mle_host(
                            ctx, (torch.arange(sh.T, device=_dev()) >> sh.ls) << sh.ls, pt)).reshape(-1, 1)),
                        kind=COL, c=t, shared=True)
            for g in gs:
                q = f"{l}.g{g}."
                A = (lambda g=g: Wt().att[g]) if ctx.P else None
                js = db.col(q + "js", lambda A=A: torch.cat([f32(x) for x in A()["js"]]), R_blk(ctx, q + "js"))
                db.dom.lookup(24, mi - js)
                db.dom.lookup(24, js - ss)
            db.run()
            # ---- key table side over (c, j)
            db = DB(ctx, nm + "ktab", sh.ldh + t, 1)
            ci = db.col("ci", lambda: f32(torch.arange(dh, device=_dev())),
                        R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.iota(pt), lambda: mle_host(ctx, torch.arange(dh, device=_dev()), pt)).reshape(-1, 1)),
                        kind=COL, c=sh.ldh, shared=True)
            ji = db.col("ji", lambda: f32(torch.arange(sh.T, device=_dev())),
                        R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.iota(pt), lambda: mle_host(ctx, torch.arange(sh.T, device=_dev()), pt)).reshape(-1, 1)),
                        kind=ROW, c=sh.ldh, shared=True)
            for g in gs:
                q = f"{l}.g{g}."
                A = (lambda g=g: Wt().att[g]) if ctx.P else None
                rK = db.col(f"rK{g}", lambda g=g: rm(Wt().rK, sh.ldh)[g << (sh.ldh + t):(g + 1) << (sh.ldh + t)],
                            R_blk(ctx, p + "rK", bmap=lambda b, g=g: g))
                mkc = db.col(q + "mk", lambda A=A: f32(A()["mk"]), R_blk(ctx, q + "mk"), kind=ROW, c=sh.ldh, shared=True)
                db.dom.lookup(lparam("Dyn") + g, ji, ci, rK, num=-mkc)
            with self.tm("a.ktab"):
                db.run()

    def att_batches(self):
        """KV groups whose softmax tile domains are proved as one domain (one GKR tree and one
        final sumcheck instead of one per group): as many groups as keep the merged domain at
        most TILE_MERGE_LEAVES lookup entries.  A function of the shapes only."""
        sh = self.sh
        per = sh.grp * sh.B * sh.ntile << 14
        k = max(1, min(sh.nkv, TILE_MERGE_G, TILE_MERGE_LEAVES // per))
        return [list(range(g, min(sh.nkv, g + k))) for g in range(0, sh.nkv, k)]

    def step_att(self, l, gs):
        """Softmax tile domains of the KV groups gs (merged), then each group's reductions,
        N / D product and opening."""
        ctx, sh = self.ctx, self.sh
        p = f"{l}."
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        grp = sh.grp
        tag = f"{l}.g{gs[0]}." if len(gs) == 1 else f"{l}.g{gs[0]}-{gs[-1]}."
        # ---- tile domains: off-diagonal tiles read x = M - acc; on the S/128 diagonal tiles the
        # causal mask gates the input (masked entries read tau_exp(XMIN) = 0), so only those
        # carry the higher-degree gate (spec, Optimized profile: one fewer relation per entry)
        mask = (torch.arange(128, device=_dev()).unsqueeze(0) <= torch.arange(128, device=_dev()).unsqueeze(1))
        for dname, sel, gate in self.tile_sets():
            if not sel:
                continue
            idx = self.tile_sub(sel)
            if ctx.mode == "commit" and gate == "diag" and all("hist" in Wt().att[g] for g in gs):
                # multiplicities of the softmax lookups from the witness's histogram of x: the
                # same chunks, in the order tables.count takes them, without rebuilding the columns
                total = idx.numel() << 14
                base = tb.LAYOUT.ent[tb.EXP][0]
                for gi, g in enumerate(gs):
                    H = Wt().att[g]["hist"]
                    for i in range(grp):
                        m = ctx.chunks.take((tag + dname, gi * grp + i), total)
                        m[base:base + (1 << 23)] += H[i]
                continue
            db = DB(ctx, tag + dname, 14, idx.numel(), merge=True)
            if gate:
                mk = db.col("mask", lambda: f32(mask.to(torch.int64)),
                            R_pub(ctx, lambda pt: mle_pub(ctx, lambda: mlepub.le_lowhigh(pt, 7),
                                                    lambda: mle_host(ctx, mask.to(torch.int64).reshape(-1), pt)).reshape(-1, 1)),
                            kind=COL, c=14, shared=True)
            if gate == "diag":
                dgt = _dlist(sh, [1 if sh.tiles[k][0] == sh.tiles[k][1] else 0 for k in sel] * sh.B)
                dg = db.col("diag", lambda dgt=dgt: f32(dgt),
                            R_pub(ctx, lambda pt, dgt=dgt: mle_host(ctx, _pad2(dgt), pt).reshape(-1, 1)), kind=ROW, c=14)
                mk = 1 - dg * (1 - mk)
            for g in gs:
                q = f"{l}.g{g}."
                for i in range(grp):
                    Pc = db.col(q + f"P{i}", lambda g=g, i=i, idx=idx: self.p_sub(l, g, i, idx),
                                R_virt(ctx, (l, g, i, dname, "Pdom")))
                    acc = db.col(f"acc{g}.{i}", lambda g=g, i=i, sel=sel, gate=gate:
                                 self.acc_tiles(l, g, i, sel, gate == "mask"), R_virt(ctx, (l, g, i, dname, "acc")))
                    Mc = db.col(f"M{g}.{i}", lambda g=g, i=i, sel=sel: self.m_tiles(l, g, i, sel),
                                R_virt(ctx, (l, g, i, dname, "Mt")), kind=ROW, c=TILE)
                    x = mk * (Mc - acc) + (1 - mk) * XMIN if gate else Mc - acc
                    db.dom.lookup(tb.EXP, x, Pc)
            with self.tm("a.tdom"):
                db.run()
            if not ctx.active:
                continue
            for g in gs:
                # every head's columns are claimed at the domain's one point: batch the heads
                src, vms, vas = None, [], []
                for i in range(grp):
                    (pt_, vals_), = ctx.take((l, g, i, dname, "Pdom"))
                    (pm_, vm_), = ctx.take((l, g, i, dname, "Mt"))
                    (pa_, va_), = ctx.take((l, g, i, dname, "acc"))
                    if src is None:
                        src = [pt_[:, :14], self.sub_eq(idx, pt_[:, 14:]), []]
                        pm0, pa0 = pm_, pa_
                    src[2].append(vals_[:, 0])
                    vms.append(vm_[:, 0])
                    vas.append(va_[:, 0])
                with self.tm("a.rowmax"):
                    self.m_claims(l, g, pm0[:, :TILE], self.sub_eq(idx, pm0[:, TILE:]), vms)
                with self.tm("a.accred"):
                    self.acc_reduce(l, g, pa0[:, :2 * TILE], self.sub_eq(idx, pa0[:, 2 * TILE:]), vas)
                self._pdom.setdefault((l, g), []).append(src)
        if not ctx.active:
            return
        W = Wt() if ctx.P else None
        for g in gs:
            # ---- N / D product
            with self.tm("a.nd"):
                self.nd_product(l, g)
            with self.tm("a.open"):
                ctx.open(p + f"att{g}", self.tensors(p + f"att{g}", W) if ctx.P else None)
            if W is not None:
                W.att[g] = None

    # attention helpers ---------------------------------------------------
    def nb_data(self, l, g):
        """Blocked numerators (before rounding) per head, layout [i][m][kb][c]."""
        W = self.wit_of(l)
        sh = self.sh
        a = W.att[g]
        V = W.V[:, g * sh.dh:(g + 1) * sh.dh]
        out = []
        nbk = max(1, 1 << sh.nbb)
        S = sh.S
        if S >= 128 and not os.environ.get("ZK_ATT_OLD"):
            f = _attw()
            lkb = min(sh.ls, mdl.KEYBLK)
            KB, nkb = 1 << lkb, S >> lkb
            assert nkb == nbk
            ta = _dlist(sh, [x for x, _ in sh.tiles], torch.int32)
            tbb = _dlist(sh, [y for _, y in sh.tiles], torch.int32)
            phi = torch.zeros((nkb, S, KB), dtype=torch.int8, device=_dev())
            plo = torch.zeros((nkb, S, KB), dtype=torch.int8, device=_dev())
            hib = torch.empty((nkb, S, sh.dh), dtype=torch.int32, device=_dev())
            lob = torch.empty((nkb, S, sh.dh), dtype=torch.int32, device=_dev())
            for i in range(sh.grp):
                Nb = torch.empty((sh.T, nbk, sh.dh), dtype=torch.int64, device=_dev())
                for b in range(sh.B):
                    Pt = a["P"][i][b].contiguous()
                    kern.launch(f["att_split"], Pt.shape[0], 128,
                                [ptr(Pt), ptr(ta), ptr(tbb), np.int32(S), ptr(phi), ptr(plo), np.int32(lkb)])
                    rows = slice(b * S, (b + 1) * S)
                    Vb = V[rows].to(torch.int8)
                    for kb in range(nkb):
                        Vk = Vb[kb * KB:(kb + 1) * KB].T.contiguous().T
                        torch._int_mm(phi[kb], Vk, out=hib[kb])
                        torch._int_mm(plo[kb], Vk, out=lob[kb])
                    Nb[rows] = (hib.to(torch.int64) * 32 + lob).transpose(0, 1)
                out.append(f32(Nb))
            return torch.cat(out)
        for i in range(sh.grp):
            Nb = torch.zeros((sh.T, nbk, sh.dh), dtype=torch.int64, device=_dev())
            for b in range(sh.B):
                Pm = self.p_full(a["P"][i][b], sh)
                rows = slice(b * sh.S, (b + 1) * sh.S)
                for kb in range(nbk):
                    j0, j1 = kb << mdl.KEYBLK, min(sh.S, (kb + 1) << mdl.KEYBLK)
                    Nb[rows, kb] = pv_int(Pm[:, j0:j1], V[rows][j0:j1])
                del Pm
            out.append(f32(Nb))
        return torch.cat(out)

    @staticmethod
    def p_full(Pt, sh):
        nt1 = sh.nt1
        Pm = torch.zeros((nt1, 128, nt1, 128), dtype=torch.int32, device=_dev())
        ta = torch.tensor([a for a, b in sh.tiles], device=_dev())
        tbb = torch.tensor([b for a, b in sh.tiles], device=_dev())
        Pm[ta, :, tbb, :] = Pt.to(torch.int32)
        return Pm.reshape(sh.S, sh.S)

    def eqt(self, r):
        G = self.ctx.G
        return G.eq_table(r) if r.shape[1] else G.from_base(torch.ones(1, dtype=torch.int32, device=_dev()))

    def tile_ix(self):
        """Per head: tile k -> (seq, a, b) and rows dst = seq nt1 + a, src = seq nt1 + b (cuda int64)."""
        if getattr(self, "_tix", None) is None:
            sh = self.sh
            ts = torch.tensor([s for s in range(sh.B) for _ in sh.tiles], device=_dev())
            ta = torch.tensor([a for _ in range(sh.B) for a, b in sh.tiles], device=_dev())
            tb_ = torch.tensor([b for _ in range(sh.B) for a, b in sh.tiles], device=_dev())
            self._tix = (ts, ta, tb_, ts * sh.nt1 + ta, ts * sh.nt1 + tb_)
        return self._tix

    def tile_sets(self):
        """(domain, per-sequence tile list positions, gate) of the softmax tile domains.  gate:
        None (x = M - acc), "mask" (every tile diagonal: the causal mask gates x), "diag" (the
        mask gates x on the diagonal tiles only).  ATT_SPLIT proves the off-diagonal tiles in their
        own lower-degree domain; measured on Qwen2.5-32B shapes the second domain's fixed cost
        (its own GKR tree and reductions) outweighs the saving, so one domain is the default."""
        sh = self.sh
        if not ATT_SPLIT:
            return [("tiles", list(range(len(sh.tiles))), "diag")]
        return [("tiles", [k for k, (a, b) in enumerate(sh.tiles) if a != b], None),
                ("tdiag", [k for k, (a, b) in enumerate(sh.tiles) if a == b], "mask")]

    def tile_sub(self, sel):
        """Positions in the full tile list (B ntile) of a per-sequence subset, sequence-major (cuda int64)."""
        sh = self.sh
        s_ = _dlist(sh, sel)
        return (torch.arange(sh.B, device=_dev()).unsqueeze(1) * sh.ntile + s_.unsqueeze(0)).reshape(-1)

    def sub_eq(self, idx, rb):
        """Weights over the full tile list of a claim on a sub-domain: eq(rb, k) at position idx[k]."""
        K = self.tile_ix()[3].numel()
        w = torch.zeros((self.ctx.D, K), dtype=torch.int32, device=_dev())
        w[:, idx] = self.eqt(rb)[:, :idx.numel()]
        return w

    def p_sub(self, l, g, i, idx):
        return self.wit_of(l).att[g]["P"][i].reshape(-1, 1 << 14)[idx].reshape(-1).to(torch.int32)

    def acc_tiles(self, l, g, i, sel=None, diag=False):
        """Score tiles acc (int32 field) of head i for the per-sequence tile subset sel."""
        W = self.wit_of(l)
        sh = self.sh
        sel = range(len(sh.tiles)) if sel is None else sel
        rK = W.rK[:, g * sh.dh:(g + 1) * sh.dh]
        ta = _dlist(sh, [sh.tiles[k][0] for k in sel])
        tbb = _dlist(sh, [sh.tiles[k][1] for k in sel])
        hh = g * sh.grp + i
        rQ = W.rQ[:, hh * sh.dh:(hh + 1) * sh.dh]
        out = []
        for b in range(sh.B):
            rows = slice(b * sh.S, (b + 1) * sh.S)
            # one 128 x 128 block product per kept tile (exact, as qk_exact): entries <= 180 in
            # magnitude, sums < 2^22, exact in fp32 (TF32 is off for fp32 matmul); a full S x S
            # product would compute the masked half too and gather the tiles from it
            Qr = rQ[rows].reshape(sh.nt1, 128, sh.dh).to(torch.float32)
            Kr = rK[rows].reshape(sh.nt1, 128, sh.dh).to(torch.float32)
            nt = ta.numel()
            CHK = 2048
            for k0 in range(0, nt, CHK):
                acc = torch.bmm(Qr[ta[k0:k0 + CHK]], Kr[tbb[k0:k0 + CHK]].transpose(1, 2)).round().to(torch.int32)
                out.append(f32(acc))
                del acc
        return torch.cat(out)

    def m_tiles(self, l, g, i, sel=None):
        W = self.wit_of(l)
        sh = self.sh
        sel = range(len(sh.tiles)) if sel is None else sel
        ta = _dlist(sh, [sh.tiles[k][0] for k in sel])
        out = []
        for b in range(sh.B):
            Mx = W.att[g]["M"][i][b * sh.S:(b + 1) * sh.S].reshape(sh.nt1, 128)
            out.append(f32(Mx[ta]))
        return torch.cat(out)

    def p_claims(self, name, pt_in, rb, vs, tvs):
        """Claims of every head i of a group: sum_t eq(rb, t) P_{i,t}(pt_in) = vs[i] (tile list split
        into columns).  The values of every block (head i, column k) of the group's P at the shared
        point z = (pt_in, rb low bits) are sent and checked against vs with the weights eq(rb high, k);
        then they are claimed as ordinary block values, so P can be packed into wide columns."""
        ctx, sh = self.ctx, self.sh
        npc, pc = self.p_cols()
        nb = pc - 14
        grp, D = sh.grp, ctx.D
        z = cat(pt_in, rb[:, :nb])
        vals = None
        if ctx.P:
            # block (i, k) = sum over its tiles u of eq(rb low, u) * (head i's tile value at pt_in),
            # from the tile values tvs[i] (D, >= tiles) of the tile-list sumcheck
            G = ctx.G
            el = G.eq_table(rb[:, :nb]) if nb else torch.ones((D, 1), dtype=torch.int32, device=_dev())
            if not nb:
                el[1:] = 0
            n = npc << nb
            vals = torch.empty((D, grp * npc), dtype=torch.int32, device=_dev())
            for i in range(grp):
                tv = tvs[i]
                X = torch.zeros((D, n), dtype=torch.int32, device=_dev())
                m = min(n, tv.shape[1])
                X[:, :m] = tv[:, :m]
                prod = G.mul(X, el.repeat(1, npc).contiguous())
                vals[:, i * npc:(i + 1) * npc] = (prod.view(D, npc, 1 << nb).to(torch.int64).sum(dim=2) % P).to(torch.int32)
        vals = ctx.send_vals(vals, (D, grp * npc))
        if not ctx.P:
            eh = eq_host(D, rb[:, nb:], npc) if rb.shape[1] > nb else const_pt(D, [1] * npc)
            F = ctx.F
            for i in range(grp):
                s_ = F.zero
                for k in range(npc):
                    s_ = F.add(s_, F.mul(F.c(eh[:, k]), F.c(vals[:, i * npc + k])))
                if s_ != F.c(vs[i]):
                    raise ValueError("P block values do not match the tile claims")
        ctx.claim(name, z, vals)

    def tile_eq(self, rb):
        """eq(rb, t) over the tile list (D, K) on the device."""
        K = self.tile_ix()[3].numel()
        return self.eqt(rb)[:, :K].contiguous()

    def m_claims(self, l, g, px, wt, vms):
        """Row-max claims sum_t wt(t) M_i(row(t), px) = vms[i] of every head i of group g
        (wt: public weights over the tile list) -> one row-dot sumcheck against rQ, Kg with
        heads as blocks (random block weights) and public row weights W(m) = c_{m >> 7} eq(px, m & 127)."""
        ctx, sh = self.ctx, self.sh
        F, G = ctx.F, ctx.G
        ts, ta, tb_, dst, src = self.tile_ix()
        nr = sh.B * sh.nt1
        grp = sh.grp
        c = torch.zeros((ctx.D, nr), dtype=torch.int64, device=_dev())
        c.index_add_(1, dst, wt.to(torch.int64))
        c = (c % P).to(torch.int32)
        Wrow = G.mul(c.repeat_interleave(1 << TILE, dim=1).contiguous(), self.eqt(px).repeat(1, nr).contiguous())
        W = (lambda: self.wit_of(l)) if ctx.P else None
        bw = ctx.chal(grp)
        claim = ctx.claimsum(bw, vms)

        def tabs():
            Wl = W()
            rQ = rm(Wl.rQ[:, g * grp * sh.dh:(g + 1) * grp * sh.dh], sh.ldh)       # [head][row][c]
            Kg = torch.cat([f32(Wl.att[g]["Kg"][i]) for i in range(grp)])
            return [Tab(Wrow, ROW, sh.ldh, shared=True), Tab(rQ, FULL), Tab(Kg, FULL)]
        shapes = [(ROW, sh.ldh), (FULL, 0), (FULL, 0)]
        rho, fin = ctx.sumcheck(b"rowmax", sh.ldh + sh.t, grp, shapes, np.array(claim),
                                tabs=tabs if ctx.proving else None, terms=[(np.array(F.one), [0, 1, 2])], bw=bw)
        if not ctx.P:
            rm_ = rho[:, sh.ldh:]
            s_ = np.asarray(G.dot(c, self.eqt(rm_[:, TILE:])[:, :nr].contiguous()), dtype=np.int64)
            if F.mul(F.c(s_), eq_point(F, px, rm_[:, :TILE])) != F.c(fin[0][:, 0]):
                raise ValueError("row max weights")
        ctx.claim(f"{l}.rQ", rho, {g * grp + i: fin[1][:, i] for i in range(grp)})
        ctx.claim(f"{l}.g{g}.Kg", rho, {i: fin[2][:, i] for i in range(grp)})

    def acc_reduce(self, l, g, pa, wd, vas):
        """Score claims sum_t wd(t) acc_{i,t}(pa) = vas[i] of every head i of group g (wd: public
        weights over the tile list) -> claims on rQ and rK.  The key side is shared by the heads,
        so one product sumcheck runs on a random combination of the heads' query tables; the
        prover then sends each head's query value at the final point."""
        ctx, sh = self.ctx, self.sh
        F, G = ctx.F, ctx.G
        ts, ta, tb_, dst, src = self.tile_ix()
        K = dst.numel()
        grp = sh.grp
        gm = ctx.chal(grp)
        claim = ctx.claimsum(gm, vas)
        pj, pmm = pa[:, :TILE], pa[:, TILE:2 * TILE]
        n = TILE + sh.ltile + sh.lb            # (c, a, seq)
        W = (lambda: self.wit_of(l)) if ctx.P else None
        nrow = sh.T >> TILE
        Qfs = []

        def tabs():
            Wl = W()
            rK = Wl.rK[:, g * sh.dh:(g + 1) * sh.dh]
            Kf = G.coldot(f32(rK.reshape(nrow, 128, sh.dh)).reshape(-1), G.eq_table(pj), nrow, 128, sh.dh)
            Kf = Kf.reshape(ctx.D, nrow, sh.dh)
            eqm = G.eq_table(pmm)
            Qc = None
            for i in range(grp):
                hh = g * grp + i
                rQ = Wl.rQ[:, hh * sh.dh:(hh + 1) * sh.dh]
                Qf = G.coldot(f32(rQ.reshape(nrow, 128, sh.dh)).reshape(-1), eqm, nrow, 128, sh.dh)
                Qfs.append(Qf)
                t_ = G.scale_p(Qf, gm[:, i])
                Qc = t_ if Qc is None else G.add(Qc, t_)
            # G(seq, a, c) = sum_{b <= a} w_(seq, a, b) rKf(seq, b, c)
            prod = G.mul(Kf[:, src].reshape(ctx.D, -1).contiguous(),
                         wd.repeat_interleave(sh.dh, dim=1).contiguous()).to(torch.int64).reshape(ctx.D, K, sh.dh)
            acc = torch.zeros((ctx.D, nrow, sh.dh), dtype=torch.int64, device=_dev())
            acc.index_add_(1, dst, prod)
            return [Tab(Qc, FULL), Tab((acc % P).to(torch.int32).reshape(ctx.D, -1), FULL)]
        shapes = [(FULL, 0), (FULL, 0)]
        rho, fin = ctx.sumcheck(b"acc-red", n, 1, shapes, np.array(claim),
                                tabs=tabs if ctx.proving else None, terms=[(np.array(F.one), [0, 1])])
        rc, ra = rho[:, :TILE], rho[:, TILE:]
        # each head's query value at rho; their gm-combination is the combined table's final value
        qv = None
        if ctx.P:
            e = G.eq_table(rho)
            qv = G.dots_dev([(Qf, e) for Qf in Qfs]).T.contiguous()
        qv = ctx.send_vals(qv, (ctx.D, grp))
        if not ctx.P:
            s_ = F.zero
            for i in range(grp):
                s_ = F.add(s_, F.mul(F.c(gm[:, i]), F.c(qv[:, i])))
            if s_ != F.c(fin[0][:, 0]):
                raise ValueError("score reduction: head values")
        ctx.claim(f"{l}.rQ", cat(rc, pmm, ra), {g * grp + i: qv[:, i] for i in range(grp)})
        # rK values at (rc, pj) for every key tile (seq, b)
        kv = None
        if ctx.P:
            rK = W().rK[:, g * sh.dh:(g + 1) * sh.dh]
            e = G.eq_table(cat(rc, pj))                       # rK tile rows are [x_j][c], c minor
            kv = G.rowdot(f32(rK).reshape(-1), e, nrow, 128 * sh.dh)
        kv = ctx.send_vals(kv, (ctx.D, nrow))
        if not ctx.P:
            # G(rho) = sum_tiles eq(ra, (a, seq)) w kv(seq, b)
            Ea = self.eqt(ra)
            got = G.dot(G.mul(Ea[:, dst].contiguous(), wd), G.to_dev(kv)[:, src].contiguous())
            if not np.array_equal(got % P, fin[1][:, 0] % P):
                raise ValueError("score reduction: key values")
        tau = ctx.chal(sh.ltile + sh.lb) if sh.ltile + sh.lb else np.zeros((ctx.D, 0), dtype=np.int64)
        v = ctx.blockcomb(tau, kv)
        ctx.claim(f"{l}.rK", cat(rc, pj, tau), {g: v})

    def nd_product(self, l, g):
        """sum_j P_i(pm, j) eq(pb, kb(j)) V(j, pc) and sum_j P_i(pd, j)."""
        ctx, sh = self.ctx, self.sh
        F, G = ctx.F, ctx.G
        (pn, vn), = ctx.take((l, g, "Nb"))
        (pd, vd), = ctx.take((l, g, "D"))
        pc_, pb, pm = pn[:, :sh.ldh], pn[:, sh.ldh:sh.ldh + sh.nbb], pn[:, sh.ldh + sh.nbb:]
        gam = ctx.chal(2)
        bw = ctx.chal(sh.grp)
        claim = np.zeros(ctx.D, dtype=np.int64)
        if not ctx.P:
            claim = F.zero
            for i in range(sh.grp):
                v = F.add(F.mul(F.c(gam[:, 0]), F.c(vn[:, i])), F.mul(F.c(gam[:, 1]), F.c(vd[:, i])))
                claim = F.add(claim, F.mul(F.c(bw[:, i]), v))
        W = (lambda: self.wit_of(l)) if ctx.P else None
        T = sh.T

        def eqb_tab():
            jb = (torch.arange(T, device=_dev()) % sh.S) >> mdl.KEYBLK
            return self.eqt(pb)[:, jb].contiguous()

        def tabs():
            Wl = W()
            a = Wl.att[g]
            V = Wl.V[:, g * sh.dh:(g + 1) * sh.dh]
            Vf = G.rowdot(f32(V).reshape(-1), G.eq_table(pc_), T, sh.dh)
            e1, e2 = G.eq_table(pm), G.eq_table(pd)
            P1, P2 = [], []
            fast = sh.S >= 128 and not os.environ.get("ZK_ATT_OLD")
            if fast:
                fa = _attw()
                ta32 = _dlist(sh, [x for x, _ in sh.tiles], torch.int32)
                tb64 = _dlist(sh, [y for _, y in sh.tiles])
                ntl = ta32.numel()
                D = ctx.D
            for i in range(sh.grp):
                f1 = torch.zeros((ctx.D, T), dtype=torch.int32, device=_dev())
                f2 = torch.zeros((ctx.D, T), dtype=torch.int32, device=_dev())
                for b in range(sh.B):
                    if fast:
                        Pt = a["P"][i][b].contiguous()
                        pp = torch.empty((2, ntl, D, 128), dtype=torch.int32, device=_dev())
                        kern.launch(fa["att_coldot"], ntl, 128,
                                    [ptr(Pt), ptr(ta32), np.int32(D), np.uint64(e1.data_ptr() + 4 * b * sh.S),
                                     np.uint64(e2.data_ptr() + 4 * b * sh.S), np.int64(e1.stride(0)), np.int64(e2.stride(0)),
                                     ptr(pp[0]), ptr(pp[1])])
                        acc_ = torch.zeros((2, sh.nt1, D, 128), dtype=torch.int64, device=_dev())
                        acc_.index_add_(1, tb64, pp.to(torch.int64))
                        red = (acc_ % P).to(torch.int32).permute(0, 2, 1, 3).reshape(2, D, sh.S)
                        rows = slice(b * sh.S, (b + 1) * sh.S)
                        f1[:, rows], f2[:, rows] = red[0], red[1]
                        continue
                    Pm = f32(self.p_full(a["P"][i][b], sh)).reshape(-1)
                    rows = slice(b * sh.S, (b + 1) * sh.S)
                    f1[:, rows] = G.coldot(Pm, e1[:, rows].contiguous(), 1, sh.S, sh.S)
                    f2[:, rows] = G.coldot(Pm, e2[:, rows].contiguous(), 1, sh.S, sh.S)
                    del Pm
                P1.append(f1)
                P2.append(f2)
            return [Tab(torch.cat(P1, dim=1), FULL), Tab(torch.cat(P2, dim=1), FULL),
                    Tab(eqb_tab(), COL, sh.t, shared=True), Tab(Vf, COL, sh.t, shared=True)]
        shapes = [(FULL, 0), (FULL, 0), (COL, sh.t), (COL, sh.t)]
        terms = [(gam[:, 0], [0, 2, 3]), (gam[:, 1], [1])]
        rho, fin = ctx.sumcheck(b"nd", sh.t, sh.grp, shapes, np.array(claim), tabs=tabs if ctx.proving else None,
                                terms=terms, bw=bw)
        # eq(pb, kb(rho)) public
        if not ctx.P:
            ev = F.one
            for j in range(sh.nbb):
                ev = F.mul(ev, F.eq1(F.c(pb[:, j]), F.c(rho[:, mdl.KEYBLK + j])))
            if F.c(fin[2][:, 0]) != ev:
                raise ValueError("key block selector")
        # V claim
        lh = log2(sh.hb)
        gb = const_pt(ctx.D, [(g % sh.hb) >> j & 1 for j in range(lh)])
        kb = const_pt(ctx.D, [(g // sh.hb) >> j & 1 for j in range(nbits(sh.Kk))])
        ctx.point_claim(f"{l}.V", cat(pc_, gb, rho, kb), fin[3][:, 0],
                        data=(rm(W().V, sh.cb) if ctx.P else None))
        # P claims at full-square points (pm, rho) and (pd, rho): sum_t w(t) P_t(in-tile point),
        # w(t) = eq(row point, row(t)) eq(col point, col(t)) public -> one sumcheck over the tile list
        # with the tile domains' P claims: sources (in-tile point, public tile weights, value per head)
        ts, ta, tb_, dst, src = self.tile_ix()
        K = dst.numel()
        nt = nbits(K)
        srcs = []
        for idx_, rowp in enumerate((pm, pd)):
            w = G.mul(self.eqt(rowp[:, TILE:])[:, dst].contiguous(), self.eqt(rho[:, TILE:])[:, src].contiguous())
            srcs.append([cat(rho[:, :TILE], rowp[:, :TILE]), w, [fin[idx_][:, i] for i in range(sh.grp)]])
        srcs += self._pdom.pop((l, g), [])
        ns = len(srcs)
        pts, wts = [s_[0] for s_ in srcs], []
        for s_ in srcs:
            w = torch.zeros((ctx.D, 1 << nt), dtype=torch.int32, device=_dev())
            w[:, :K] = s_[1]
            wts.append(w)
        gm = ctx.chal(ns * sh.grp)
        claim = ctx.claimsum(gm, [srcs[idx_][2][i] for idx_ in range(ns) for i in range(sh.grp)])
        terms = []
        for idx_ in range(ns):
            for i in range(sh.grp):
                k = idx_ * sh.grp + i
                terms.append((gm[:, k], [idx_, ns + k]))

        ptv = {}                       # prover: per-tile P values at each source's in-tile point

        def ptabs():
            out = [Tab(w, FULL) for w in wts]
            es = [G.eq_table(pts[idx_]) for idx_ in range(ns)]
            for i in range(sh.grp):            # every source's tile values in one read of P (int16)
                Pi = W().att[g]["P"][i].reshape(-1)
                for idx_, v in enumerate(G.rowdot_multi(Pi, es, K, 1 << 14)):
                    tv = torch.zeros((ctx.D, 1 << nt), dtype=torch.int32, device=_dev())
                    tv[:, :K] = v
                    ptv[(idx_, i)] = tv
            for idx_ in range(ns):
                for i in range(sh.grp):
                    out.append(Tab(ptv[(idx_, i)], FULL))
            return out
        rt, fp = ctx.sumcheck(b"nd-tiles", nt, 1, [(FULL, 0)] * (ns + ns * sh.grp), np.array(claim),
                              tabs=ptabs if ctx.proving else None, terms=terms)
        for idx_ in range(ns):
            if not ctx.P:
                ev = np.asarray(G.dot(wts[idx_], self.eqt(rt)), dtype=np.int64)
                if not np.array_equal(ev % P, fp[idx_][:, 0] % P):
                    raise ValueError("P tile weights")
            self.p_claims(f"{l}.g{g}.P", pts[idx_], rt, [fp[ns + idx_ * sh.grp + i][:, 0] for i in range(sh.grp)],
                          [ptv[(idx_, i)] for i in range(sh.grp)] if ctx.P else None)

    def step_in(self, l):
        ctx, sh, M = self.ctx, self.sh, self.M
        cb, t, dh = sh.cb, sh.t, sh.dh
        p = f"{l}."
        Ly = M.layers[l]
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        nq, nk = sh.nh * dh, sh.nkv * dh
        Kqkv = sh.Kq + 2 * sh.Kk
        lh = log2(sh.hb)
        # RoPE over (c6, m) x (nh + nkv)
        db = DB(ctx, p + "rope", sh.ldh - 1 + t, sh.nh + sh.nkv)

        def halfview(X, h):        # (T, H*dh) -> [H][T][64] of half h
            Tn = X.shape[0]
            return f32(X.reshape(Tn, -1, 2, dh // 2)[:, :, h].permute(1, 0, 2))

        def qk_route(h):
            def f(pt, vals):
                pc, pm = pt[:, :sh.ldh - 1], pt[:, sh.ldh - 1:]
                hb = const_pt(ctx.D, [h])
                for name, H, off in ((p + "Q", sh.nh, 0), (p + "K", sh.nkv, sh.nh)):
                    nbh = nbits(H)
                    tau = ctx.chal(nbh) if nbh else np.zeros((ctx.D, 0), dtype=np.int64)
                    v = ctx.blockcomb(tau, vals[:, off:off + H])
                    K = sh.Kq if name.endswith("Q") else sh.Kk
                    nbk = nbits(K)
                    hib = tau[:, :lh]
                    kbp = tau[:, lh:]
                    if kbp.shape[1] < nbk:
                        kbp = cat(kbp, np.zeros((ctx.D, nbk - kbp.shape[1]), dtype=np.int64))
                    ctx.point_claim(name, cat(pc, hb, hib, pm, kbp[:, :nbk]), v,
                                    data=(rm(Wt().qkv[:, :nq] if name.endswith("Q") else Wt().qkv[:, nq:nq + nk], cb)
                                          if ctx.P else None))
            return f

        def r_route(h):
            def f(pt, vals):
                pc, pm = pt[:, :sh.ldh - 1], pt[:, sh.ldh - 1:]
                p2_ = cat(pc, const_pt(ctx.D, [h]), pm)
                ctx.claim(p + "rQ", p2_, {i: vals[:, i] for i in range(sh.nh)})
                ctx.claim(p + "rK", p2_, {i: vals[:, sh.nh + i] for i in range(sh.nkv)})
            return f
        qk = lambda: Wt().qkv[:, :nq + nk]
        rqk = lambda: torch.cat([Wt().rQ, Wt().rK], dim=1)
        q1 = db.col("q1", lambda: halfview(qk(), 0), qk_route(0))
        q2 = db.col("q2", lambda: halfview(qk(), 1), qk_route(1))
        r1 = db.col("r1", lambda: halfview(rqk(), 0), r_route(0))
        r2 = db.col("r2", lambda: halfview(rqk(), 1), r_route(1))
        T = sh.T
        cos, sin = mdl.rope_tables(self.cfg, sh.S)
        pos = torch.arange(T, device=_dev()) % sh.S
        cs = lambda X: f32(X[pos])                         # (T, 64): index m*64 + c  ->  (c low, m high)
        cos_c = db.col("cos", lambda: cs(cos), R_blk(ctx, "rope.cos"), kind=COL, c=sh.ldh - 1 + t, shared=True)
        sin_c = db.col("sin", lambda: cs(sin), R_blk(ctx, "rope.sin"), kind=COL, c=sh.ldh - 1 + t, shared=True)
        d = db.dom
        d.lookup(15, q1 * cos_c - q2 * sin_c + (1 << 14) - (1 << 15) * r1)
        d.lookup(tb.STAR9, r1 + 180)
        d.lookup(15, q2 * cos_c + q1 * sin_c + (1 << 14) - (1 << 15) * r2)
        d.lookup(tb.STAR9, r2 + 180)
        db.run()
        qkn = self.cfg.qk_norm
        if qkn:
            for X, c0, KX in (("qn", 0, sh.Kq), ("kn", nq, sh.Kk)):
                self.qkn_domains(l, X, c0, KX)
        # ShiftQ8S over T x (nq + 2 nk) (with QK-norm: over V only, T x nk)
        v0 = nq + nk if qkn else 0                       # first column of the domain
        Kqkv = sh.Kk if qkn else Kqkv
        db = DB(ctx, p + "qkv", cb + t, Kqkv)

        def qkv_bmap(b):
            if qkn:
                return (p + "V", b)
            if b < sh.Kq:
                return (p + "Q", b)
            if b < sh.Kq + sh.Kk:
                return (p + "K", b - sh.Kq)
            return (p + "V", b - sh.Kq - sh.Kk)

        def qkv_route(pt, vals):
            d_ = {}
            for b in range(vals.shape[1]):
                nm, bb = qkv_bmap(b)
                d_.setdefault(nm, {})[bb] = vals[:, b]
            for nm, dd in d_.items():
                ctx.claim(nm, pt, dd)
        acc = db.col("accqkv", lambda: rm(Wt().acc_qkv[:, v0:], cb), R_virt(ctx, (l, "accqkv")))
        qv = db.col("qkv", lambda: rm(Wt().qkv[:, v0:], cb), qkv_route)
        o = db.col(p + "oq", lambda: rm(Wt().oq, cb), R_blk(ctx, p + "oq"))
        wh = db.col(p + "whq", lambda: rm(Wt().whq, cb), R_blk(ctx, p + "whq"))
        Eh = db.col(p + "n1.Eh", lambda: f32(Wt().n1["Eh"]), R_blk(ctx, p + "n1.Eh"), kind=ROW, c=cb, shared=True)
        pEh = db.col(p + "n1.pEh", lambda: f32(Wt().n1["pEh"]), R_blk(ctx, p + "n1.pEh"), kind=ROW, c=cb, shared=True)

        def colvec(v):
            X = lambda: rm(v.unsqueeze(0), cb)

            def fn(pt):
                Xr = X().reshape(Kqkv, -1)
                return np.stack([mle_host(ctx, Xr[k], pt) for k in range(Kqkv)], axis=1)
            return X, fn
        kcX, kcF = colvec(Ly.kc[v0:])
        pkX, pkF = colvec(mdl.pow2(Ly.kc[v0:]))
        biX, biF = colvec(Ly.bias[v0:])
        kc = db.col("kc", kcX, R_pub(ctx, kcF), kind=COL, c=cb)
        pk = db.col("pkc", pkX, R_pub(ctx, pkF), kind=COL, c=cb)
        bi = db.col("bias", biX, R_pub(ctx, biF), kind=COL, c=cb)
        s = pk * pEh
        k = kc + Eh
        o2 = o * o
        u = acc + s * INV2 - s * (qv - bi)
        w = o * (acc + s * bi) - 255 * s * INV2 - (o2 - o) * INV2 - (1 << 15) * wh
        d = db.dom
        d.lookup((1 - o2) * k + o2 * 15, (1 - o2) * u + o2 * w)
        d.lookup((1 - o2) * tb.STAR8 + o2 * 15, (1 - o2) * (qv + 127) + o2 * wh)
        d.zero(o2 * o - o)
        d.zero(o2 * (qv - 127 * o))
        db.run()
        # per-row domain of G_in
        db = DB(ctx, p + "in.rows", t)
        self.rq_rows(db, p + "r1.", lambda: Wt().r1, sh.Kd, asel_key=(l, "asel1"))
        nsc1 = self.ns_rows(db, p + "n1.", lambda: Wt().n1, (l, "sig1"))
        kqs = cap_col(ctx, db, p + "kqs", lambda: Wt().n1["Eh"], lambda: self.M.layers[l].kqs_t, lambda: self.M.layers[l].kqs)
        db.dom.lookup(5, kqs - nsc1["Eh"])                   # ShiftQ8S wrap-free: kappa_n + E_h <= k_max(b_n)
        db.run()
        # Requant8 + NormScale over T x d
        db = DB(ctx, p + "in", cb + t, sh.Kd)
        h = self.hin_cols(db, l, Wt)
        self.rq_norm_dom(db, h, p + "q1", p + "qp1", p + "r1.", p + "n1.", lambda: Wt().r1, lambda: Wt().n1)
        db.run()
        if not ctx.active:
            return
        # qkv product (with QK-norm: Q, K and V products, Q and K after their head-row gathers)
        cl = []
        for pt, vals in ctx.take((l, "accqkv")):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(Kqkv))
            cl.append((po, pt[:, cb:], v))
        A = (lambda: rm(Wt().n1["qp"], cb)) if ctx.P else None
        if qkn:
            for X, c0, KX, wn in (("qn", 0, sh.Kq, "Wq"), ("kn", nq, sh.Kk, "Wk")):
                clx = self.qkn_gather(l, X, c0, KX)
                matmul(ctx, b"qkn-" + X.encode(), p + "qp1", A, sh.Kd, T, p + wn,
                       Ly.Wqkv[:, c0:c0 + (KX << cb)].contiguous() if ctx.P else None, KX, cb, clx)
            matmul(ctx, b"v", p + "qp1", A, sh.Kd, T, p + "Wv",
                   Ly.Wqkv[:, nq + nk:].contiguous() if ctx.P else None, Kqkv, cb, cl)
        else:
            matmul(ctx, b"qkv", p + "qp1", A, sh.Kd, T, p + "Wqkv", Ly.Wqkv if ctx.P else None, Kqkv, cb, cl)
        # gather + row sum
        (pa, va), = ctx.take((l, "asel1"))
        (ps, vs), = ctx.take((l, "sig1"))
        if l == 0:
            cols = [("h0", lambda: rm(Wt().hin, cb), R_virt(ctx, "h0"))]
            hf = lambda c: c[0]
            qi = 1
        else:
            q = f"{l - 1}."
            cols = [(q + "s2", lambda: rm(self.wit_of(l - 1).s2, cb), R_blk(ctx, q + "s2")),
                    (q + "os2", lambda: rm(self.wit_of(l - 1).os2, cb), R_blk(ctx, q + "os2"))]
            hf = lambda c: hexpr(c[0], c[1])
            qi = 2
        cols.append((p + "q1", lambda: rm(Wt().r1["q"], cb), R_blk(ctx, p + "q1")))
        self.gather_rowsum(b"in-g", l, t, pa, sh.Kd, dict(cols=cols, h=hf, q=qi, asel=va[:, 0], sig=vs[:, 0]),
                           p + "r1.jb", lambda: Wt().r1["jb"])
        ctx.open(p + "in", self.tensors(p + "in", Wt()) if ctx.P else None)

    # ------------------------------------------------------------ QK-norm (Qwen3)
    def qkn_domains(self, l, X, c0, KX):
        """Q or K under QK-norm, over the column-block layout (cb + t) x KX: ShiftQ8 of the accumulator
        to q8, NormScale q8 -> q', ShiftQ8S of q' G by kn + E2 into the committed Q / K; the head-row
        quantities (exponents, NormScale witnesses) are per-block ROW columns over (hib, token).  Then
        the per-head-row domain: ShiftQ8 minimality, NormScale, and 1 <= kn + E2 <= 22."""
        ctx, sh = self.ctx, self.sh
        cb, t, ldh = sh.cb, sh.t, sh.ldh
        p, q = f"{l}.", f"{l}.{X}."
        n = KX << cb
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        Wq = (lambda: self.wit_of(l).qn[X]) if ctx.P else None
        nbk = nbits(KX)

        def Rrow(name):                  # per-block row values -> the head-row tensor (1, nr)
            def f(pt, vals):
                tau = ctx.chal(nbk) if nbk else np.zeros((ctx.D, 0), dtype=np.int64)
                ctx.claim(name, cat(pt, tau), {0: ctx.blockcomb(tau, vals)})
            return f

        def pubcol(vfn):                 # a public per-column vector of the domain (COL, c = cb)
            Xf = lambda: rm(vfn().unsqueeze(0), cb)

            def fn(pt):
                Xr = Xf().reshape(KX, -1)
                return np.stack([mle_host(ctx, Xr[k], pt) for k in range(KX)], axis=1)
            return Xf, fn
        db = DB(ctx, q + "dom", cb + t, KX)
        acc = db.col("acc", lambda: rm(Wt().acc_qkv[:, c0:c0 + n], cb), R_virt(ctx, (l, "acc" + X)))
        q8 = db.col(q + "q8", lambda: rm(Wq()["q8"], cb), R_blk(ctx, q + "q8"))
        qp = db.col(q + "qp", lambda: rm(Wq()["qp"], cb), R_blk(ctx, q + "qp"))
        qv = db.col("out", lambda: rm(Wt().qkv[:, c0:c0 + n], cb), R_blk(ctx, p + ("Q" if X == "qn" else "K")))
        o = db.col(q + "o", lambda: rm(Wq()["o"], cb), R_blk(ctx, q + "o"))
        wh = db.col(q + "wh", lambda: rm(Wq()["wh"], cb), R_blk(ctx, q + "wh"))
        rowc = lambda name, fn: db.col(q + name, lambda: f32(fn()), Rrow(q + name), kind=ROW, c=ldh)
        e8 = rowc("r.e", lambda: Wq()["r"]["e"])
        s8 = rowc("r.s", lambda: Wq()["r"]["s"])
        mu = rowc("n.mu", lambda: Wq()["n"]["mu"])
        kq = rowc("kq", lambda: Wq()["kq"])
        pkq = rowc("pkq", lambda: Wq()["pkq"])
        Ly = lambda: self.M.layers[l]
        gX, gF = pubcol(lambda: Ly().Gqk[c0:c0 + n])
        G = db.col("G", gX, R_pub(ctx, gF), kind=COL, c=cb)
        d = db.dom
        d.lookup(e8, acc + s8 * INV2 - s8 * q8)                  # ShiftQ8: q8 = round(acc / s8)
        d.lookup(tb.STAR8, q8 + 127)
        d.lookup(15, q8 * mu + (1 << 14) - (1 << 15) * qp)      # NormScale
        d.lookup(tb.STAR8, qp + 127)
        x = qp * G                                                # ShiftQ8S of q' G by kq, bias 0
        s = pkq
        k = kq
        o2 = o * o
        u = x + s * INV2 - s * qv
        w = o * x - 255 * s * INV2 - (o2 - o) * INV2 - (1 << 15) * wh
        d.lookup((1 - o2) * k + o2 * 15, (1 - o2) * u + o2 * w)
        d.lookup((1 - o2) * tb.STAR8 + o2 * 15, (1 - o2) * (qv + 127) + o2 * wh)
        d.zero(o2 * o - o)
        d.zero(o2 * (qv - 127 * o))
        db.run()
        # per head row: (hib, token, block)
        nr = (cb - ldh) + t + nbk
        db = DB(ctx, q + "rows", nr)
        self.rq_rows(db, q + "r.", lambda: Wq()["r"], 1, e0=2, emax=22, asel_key=(l, X + "asel"), cb=ldh)
        nsc = self.ns_rows(db, q + "n.", lambda: Wq()["n"], (l, X + "sig"))

        def knr_eval(pt):                # kn depends on the head (hib, block) only
            kh = Ly().kn[c0:c0 + n].view(-1, sh.dh)[:, 0]     # head hib + hb block, the row bits' order
            ph = cat(pt[:, :cb - ldh], pt[:, cb - ldh + t:])
            if ph.shape[1] == 0:
                v = np.zeros((ctx.D, 1), dtype=np.int64)
                v[0, 0] = int(kh[0]) % P
                return v
            return mle_host(ctx, kh, ph).reshape(-1, 1)
        knr = db.col("knr", lambda: f32(Wq()["knr"]), R_pub(ctx, knr_eval))
        # kq = max(kn + E2, 1) <= 22 (the ShiftQ8S table range): fq = [kn + E2 < 1]
        qc = {f: db.col(q + f, (lambda f=f: f32(Wq()[f])), R_blk(ctx, q + f)) for f in QK_F}
        kv, fq = knr + nsc["Eh"], qc["fq"]
        d = db.dom
        d.zero(fq * (1 - fq))
        d.zero(qc["kq"] - fq - (1 - fq) * kv)
        d.lookup(5, (1 - fq) * (kv - 1))
        d.lookup(5, fq * (0 - kv))
        d.lookup(5, mdl.QKV_KMAX - qc["kq"])
        d.lookup(tb.POW, qc["kq"] + tb.POW_OFF, qc["pkq"])
        db.run()

    def qkn_gather(self, l, X, c0, KX):
        """Minimality (selected accumulator) and the NormScale sum sum q8^2 of every head row in one
        gather / row-sum sumcheck over (c7, hib, token, block); its claims on the accumulator are
        returned as point claims for the Q / K product."""
        ctx, sh = self.ctx, self.sh
        cb, t, ldh = sh.cb, sh.t, sh.ldh
        q = f"{l}.{X}."
        n = KX << cb
        Wt = (lambda: self.wit_of(l)) if ctx.P else None
        nr = (cb - ldh) + t + nbits(KX)
        (pa, va), = ctx.take((l, X + "asel"))
        (ps, vs), = ctx.take((l, X + "sig"))
        direct = []

        def acc_route(rho, vals):
            direct.append((cat(rho[:, :cb], rho[:, cb + t:]), rho[:, cb:cb + t], vals[:, 0]))

        def q8_route(rho, vals):
            ctx.point_claim(q + "q8", rho, vals[:, 0], data=rm(self.wit_of(l).qn[X]["q8"], cb) if ctx.P else None)
        self.gather_rowsum(b"qkn-g-" + X.encode(), l, nr, pa, 1, dict(
            cols=[("acc", lambda: rm(Wt().acc_qkv[:, c0:c0 + n], cb), acc_route),
                  (q + "q8", lambda: rm(Wt().qn[X]["q8"], cb), q8_route)],
            h=lambda c: c[0], q=1, asel=va[:, 0], sig=vs[:, 0]),
            q + "r.jb", lambda: self.wit_of(l).qn[X]["r"]["jb"], cb=ldh)
        cl = []
        for pt, vals in ctx.take((l, "acc" + X)):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(KX))
            cl.append((po, pt[:, cb:], v))
        return cl + direct

    # ------------------------------------------------------------ embedding
    def step_embed(self):
        """h0 claims -> claims on the embedding table (public token one-hots)."""
        ctx, sh, M = self.ctx, self.sh, self.M
        cb = sh.cb
        F, G = ctx.F, ctx.G
        cl = []
        for pt, vals in ctx.take("h0"):
            po, v = combine_blocks(ctx, pt[:, :cb], vals, nbits(sh.Kd))
            cl.append((po, pt[:, cb:], v))
        if not cl:
            return
        gam = ctx.chal(len(cl))
        claim = ctx.claimsum(gam, [v for _, _, v in cl])
        x = self.tokens.reshape(-1)
        V, cbv = self.cfg.V, sh.cbv

        def ohx(pm):
            e = G.eq_table(pm)                                  # (D, T)
            out = torch.zeros((ctx.D, V), dtype=torch.int64, device=_dev())
            out.index_add_(1, x.to(_dev()), e.to(torch.int64))
            return (out % P).to(torch.int32)

        def tabs():
            out = []
            for po, pm, _ in cl:
                ef = G.eq_table(po)[:, :M.cfg.d].contiguous()
                Ef = G.rowdot(M.emb.reshape(-1), ef, V, M.cfg.d) if M.emb.dtype == torch.int8 else \
                    G.rowdot(f32(M.emb).reshape(-1), ef, V, M.cfg.d)
                out += [Tab(ohx(pm), FULL), Tab(Ef, FULL)]
            return out
        terms = [(gam[:, i], [2 * i, 2 * i + 1]) for i in range(len(cl))]
        rho, fin = ctx.sumcheck(b"embed", cbv, sh.Kv, [(FULL, 0)] * (2 * len(cl)), np.array(claim),
                                tabs=tabs if ctx.proving else None, terms=terms)
        for i, (po, pm, _) in enumerate(cl):
            if not ctx.P:
                X = ohx(pm)
                exp = np.stack([G.to_host(G.fold_all(X[:, k << cbv:(k + 1) << cbv], rho)) for k in range(sh.Kv)], axis=1) \
                    if hasattr(G, "fold_all") else None
                if exp is None:
                    # evaluate per vocabulary block
                    ev = []
                    for k in range(sh.Kv):
                        blk = X[:, k << cbv:(k + 1) << cbv].contiguous()
                        cur = blk
                        for j in range(cbv):
                            cur = G.fold(cur, rho[:, j])
                        ev.append(G.to_host(cur[:, 0]))
                    exp = np.stack(ev, axis=1)
                if not np.array_equal(exp % P, fin[2 * i] % P):
                    raise ValueError("token one-hot")
            # emb blocks (vb, ob), in-block (o' low, v' high)
            vals = None
            if ctx.P:
                e1 = G.eq_table(po[:, :cb])
                Xe = M.emb.reshape(-1) if M.emb.dtype == torch.int8 else f32(M.emb)
                r1 = G.rowdot(Xe, e1, V * sh.Kd, 1 << cb)                    # [vb][v'][ob]
                et = G.eq_table(rho)
                etr = et.repeat_interleave(sh.Kd, dim=1).repeat(1, sh.Kv)
                pr = G.mul(r1, etr).to(torch.int64).reshape(ctx.D, sh.Kv, 1 << cbv, sh.Kd).sum(dim=2) % P
                vals = pr.reshape(ctx.D, -1).to(torch.int32)
            vals = ctx.send_vals(vals, (ctx.D, sh.Kv * sh.Kd))
            tau = po[:, cb:]
            for k in range(sh.Kv if not ctx.P else 0):
                v = ctx.blockcomb(tau, vals[:, k * sh.Kd:(k + 1) * sh.Kd])
                if tuple(int(a) for a in v) != F.c(fin[2 * i + 1][:, k]):
                    raise ValueError("embedding block values")
            ctx.claim("emb", cat(po[:, :cb], rho), vals)

    # ------------------------------------------------------------ table side
    def table_cols(self):
        if getattr(self, "_tcols", None) is None:
            M = self.M
            ex = mdl.exp_table(M.layers[0].E, self.cfg.dh)
            c = torch.arange(-(1 << 14), (1 << 14) + 1, device=_dev())
            si = mdl.silu_values(c, M.eG, M.eA)
            self._tcols = tb.build_table(ex, si, mdl.samp_table(self.theta) if self.theta else None,
                                         mdl.router_table() if self.cfg.moe else None)
        return self._tcols

    def table_tensors(self):
        return {f"tab{j}": c for j, c in enumerate(self.table_cols())}

    def rope_tensors(self):
        cos, sin = mdl.rope_tables(self.cfg, self.sh.S)
        pos = torch.arange(self.sh.T, device=_dev()) % self.sh.S
        return {"rope.cos": f32(cos[pos]), "rope.sin": f32(sin[pos])}     # index m * 64 + c

    def fixed_tensors(self, gname):
        return self.table_tensors() if gname == "tab" else self.rope_tensors()

    def step_table(self):
        """Batched table side of every LogUp chunk; opens the table and multiplicity units."""
        ctx = self.ctx
        if ctx.P:
            logup.table_side(ctx, self.table_cols(), ctx.chunks, None, None)
            n = ctx.chunks.n()
        else:
            n = self.nchunks
            S = [ctx.csum.get(i, ctx.F.zero) for i in range(n)]
            logup.table_side(ctx, None, None, S, None)
        for gname in FIXED_UNITS:
            ctx.open(gname, self.fixed_tensors(gname) if ctx.P else None)
        for i in range(n):
            ctx.open(f"mult{i}", {f"mult{i}": ctx.chunks.get(i)} if ctx.P else None)

    # ------------------------------------------------------------ passes
    def boundary(self, l):
        """Residual output of layer l: SatAdd (s2, os2) (prover, from host; in a captured layer
        from fixed device buffers, refilled per replay)."""
        gb = getattr(self, "_gbnd", None)
        if gb is not None and l == gb[0]:
            return Bound(gb[1].to(torch.int64), gb[2].to(torch.int64))
        s2, os2 = self.bnd[l]
        return Bound(s2.cuda(non_blocking=True).to(torch.int64), os2.cuda(non_blocking=True).to(torch.int64))

    def layer_input(self, l):
        if l == 0:
            return embed(self.M, self.tokens)
        b = self.boundary(l - 1)
        return h_from_sat(b.s2, b.os2)

    def wgroups(self):
        return [g for l in range(self.L) for g in self.layer_wgroups(l)] + ["Wlm", "emb"]

    def layer_wgroups(self, l):
        """Layer l's weight groups: '{l}.W' and, for an MoE, its expert stacks."""
        return [f"{l}.W"] + ([f"{l}.{nm}" for nm in ("Wge", "Wue", "Wde")] if self.cfg.moe else [])

    def moe_of(self, l):
        from . import moe
        cache = self.__dict__.setdefault("_moe", {})
        if l not in cache:
            cache[l] = moe.MoEProof(self, l)
        return cache[l]

    def provider(self, skip=None):
        """Every group's tensors. skip: unit names already satisfied (have g / stash)."""
        skip = skip or set()
        for gname in self.wgroups():
            if gname in skip:
                continue
            if gname[0].isdigit():
                self.use_layer(int(gname.split(".")[0]))
            yield gname, self.weight_tensors(gname)
        for l in range(self.L):
            gl = self.layer_groups(l)
            if all(g in skip or g in self._stash for g in gl):
                for gname in gl:
                    if gname not in skip:
                        yield gname, self._stash_get(gname)
                continue
            self.use_layer(l)
            with self.tm("wit"):
                W = LayerWit(self.M, self.sh, l, self.layer_input(l))
            for gname in gl:
                if gname not in skip:
                    yield gname, self.tensors(gname, W)
            del W
        for gname in FIXED_UNITS:
            if gname not in skip:
                yield gname, self.fixed_tensors(gname)
        if "fin" not in skip:
            yield "fin", (self._stash_get("fin") if "fin" in self._stash else self.tensors("fin", None))
        for i in range(self.ctx.chunks.n()):
            gn = f"mult{i}"
            if gn not in skip:
                yield gn, {gn: self.ctx.chunks.get(i)}

    def declare_mult(self, n):
        self.nchunks = n
        for i in range(n):
            self.ctx.group(f"mult{i}", [(f"mult{i}", 1, tb.LOG_ROWS)])

    def run(self):
        """Prover: all passes.  Verifier: replay.  Prover-side witness checks are verified once,
        at the end (mdl.deferred_checks), instead of synchronizing at each."""
        if self.ctx.P:
            with mdl.deferred_checks():
                return self._run()
        return self._run()

    def _run(self):
        ctx, sh = self.ctx, self.sh
        tr = ctx.tr
        self.declare()
        L = self.L
        tm = self.tm
        if ctx.P:
            ctx.mode = "commit"
            with tm("setup.W"):
                cached = self.M.__dict__.get("_cw_cache", {}).get(_cw_key(self.M, sh, ctx.pcs_args))
                for gname in self.wgroups():
                    if cached is not None and cached[0] == self.C_W:
                        ctx.pcs.commit_caps(gname, cached[1][gname])
                        continue
                    if gname[0].isdigit():
                        self.use_layer(int(gname.split(".")[0]))
                    ctx.commit(gname, self.weight_tensors(gname))
                tcache = self.M.__dict__.setdefault("_tab_cache", {})
                for gname in FIXED_UNITS:
                    tkey = _fixed_key(self.M, self.sh, self.theta, gname, ctx.pcs_args)
                    if tkey in tcache:
                        ctx.pcs.commit_caps(gname, tcache[tkey][1])
                    else:
                        ctx.commit(gname, self.fixed_tensors(gname))
                        tcache[tkey] = (_wdigest(ctx, [gname]), ctx.pcs.cap_bytes(gname))
            self.bnd = {}
            h = self.layer_input(0)
            for l in range(L):
                self.use_layer(l)
                with tm("wit"):
                    W = LayerWit(self.M, sh, l, h, hist=True)
                self.wit = {l: W}
                if l > 0:
                    self.wit[l - 1] = self.boundary(l - 1)
                with tm("count"):
                    self.layer_steps(l, commit=True)
                with tm("commit"):
                    for gname in self.layer_groups(l):
                        ctx.commit(gname, self.tensors(gname, W))
                self.bnd[l] = (_to_pinned(W.s2.to(torch.int32)), _to_pinned(W.os2.to(torch.int8)))
                h = W.hout
                del W
                self.wit = {}
            self.fin_wit(h)
            del h
            tb.check_counts()
            self.wit = {L - 1: self.boundary(L - 1)}
            self.fin_steps()
            with tm("commit"):
                ctx.commit("fin", self.tensors("fin", None))
                tb.check_counts()
                n = ctx.chunks.finish()
                tr.send_bytes(ctx.chunks.assign_bytes())
                self.declare_mult(n)
                for i in range(n):
                    ctx.commit(f"mult{i}", {f"mult{i}": ctx.chunks.get(i)})
            self.wit = {}
            ctx.mode = "prove"
            print(f"[run] commit done alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            if os.environ.get("ZK_DEBUG"):
                ctx.dbg_tcols = self.table_cols()
            with tm("grind"):
                tr.grind_bytes(GRIND_LOOKUP)
        else:
            for gname in self.wgroups():
                ctx.commit(gname)
            if _wdigest(ctx, self.wgroups()) != self.C_W:
                raise ValueError("weight commitment differs from C_W")
            for gname in FIXED_UNITS:
                ctx.commit(gname)
                if _wdigest(ctx, [gname]) != fixed_digest(self.M, self.sh, self.theta, gname, **ctx.pcs_args):
                    raise ValueError(f"fixed unit {gname}: commitment differs from the model's")
            for l in range(L):
                for gname in self.layer_groups(l):
                    ctx.commit(gname)
            ctx.commit("fin")
            ctx.assign = logup.parse_assign(tr.send_bytes(None))
            chunks = list(ctx.assign.values())
            if chunks and (min(chunks) < 0 or max(chunks) >= len(chunks)):
                raise ValueError("lookup chunk assignment: chunk index out of range")   # (at most one chunk per lookup)
            n = 1 + max(chunks, default=-1)
            self.declare_mult(n)
            for i in range(n):
                ctx.commit(f"mult{i}")
            if not tr.verify_grind_bytes(GRIND_LOOKUP):
                raise ValueError("lookup grinding")
        ctx.Z = ctx.chal()
        ctx.beta = ctx.chal(gkr.NBETA)
        if ctx.P and DEV_SESSION:
            # prove pass: transcript state on the device, results lazy (see lazy.py)
            tr.begin_device()
        # prove pass (reverse)
        if ctx.P:
            self.wit = {L - 1: self.boundary(L - 1)}
        if ctx.P:
            print("[run] prove fin", flush=True)
        with tm("fin"):
            self.fin_steps()
            ctx.open("fin", self.tensors("fin", None) if ctx.P else None)
            ctx.open("Wlm", self.weight_tensors("Wlm") if ctx.P else None)
        # graph mode: layers L-1 and 0 as usual, the middle ones replay one captured layer
        gmode = ctx.P and GRAPH and tr.dev is not None and L >= 3
        for l in range(L - 1, -1, -1):
            if gmode and 0 < l < L - 1:
                if l == L - 2:
                    self.graph_layers(l, 1)
                continue
            self.use_layer(l)
            if ctx.P:
                print(f"[run] prove layer {l} alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            w0 = tr.dev.cur if tr.dev is not None else 0
            self.prove_layer(l)
            if tr.dev is not None:
                self._lw = tr.dev.cur - w0              # transcript log words of one layer
            if ctx.P and os.environ.get("ZK_LAYERLOG"):
                print(f"[layer {l}] t={time.time():.1f} " + " ".join(f"{k}={v:.1f}" for k, v in sorted(self.tm.t.items())
                                                                    if k in ("mo", "a.ktab", "att", "mlp"))
                      + f" alloc={torch.cuda.memory_allocated() / 2**30:.1f} res={torch.cuda.memory_reserved() / 2**30:.1f}",
                      flush=True)
        self.wit = {}
        with tm("embed"):
            self.step_embed()
            ctx.open("emb", self.weight_tensors("emb") if ctx.P else None)
        with tm("table"):
            self.step_table()
        if not ctx.P:
            ctx.check_sets()
            left = [k for k, v in ctx.virt.items() if v]
            if left:                                # every virtual claim must reach its domain
                raise ValueError(f"unconsumed virtual claims: {left[:4]}")
        tr.end_device()
        with tm("pcs"):
            ctx.finish(self.provider if ctx.P else None)
        return True

    def prove_layer(self, l):
        """Prove pass of layer l: witness, steps, weight opening.  The layer index is absorbed
        here once; names absorbed inside the layer carry no layer prefix (transcript.label_name)."""
        self.layer_head(l)
        self.layer_body(l)

    def layer_head(self, l):
        """The layer-dependent inputs of layer_body: index absorbed, constants to the device."""
        ctx = self.ctx
        ctx.tr.absorb(b"layer|")
        ctx.tr.absorb(int(l).to_bytes(4, "little"))
        dm.set_lparams(layer_lparams(self.M, l))
        if ctx.P:
            if getattr(self, "_wlp", None) is None:
                self._wlp = torch.zeros(2, dtype=torch.int64, device=_dev())
            wit_lparams(self.M, l, self._wlp)

    def layer_body(self, l):
        """Layer l's prove work given layer_head(l): the same Python path for every layer (only
        names differ), so it can be captured once as a CUDA graph and replayed (graph_layers)."""
        ctx, tm = self.ctx, self.tm
        if ctx.P:
            self.wit = {}
            with tm("wit"):
                self.wit[l] = LayerWit(self.M, self.sh, l, self.layer_input(l), lp=self._wlp)
            if l > 0:
                self.wit[l - 1] = self.boundary(l - 1)
        self.layer_steps(l)
        with tm("open.W"):
            for gname in self.layer_wgroups(l):
                ctx.open(gname, self.weight_tensors(gname) if ctx.P else None)
        with tm("seg"):
            ctx.pcs.segment(l)

    def graph_layers(self, hi, lo):
        """Prover, device session: the prove pass of layers hi, hi-1, .., lo.  layer_body(hi) is
        captured once as a CUDA graph (its Python runs once) and replayed for every layer; per
        layer only its inputs are refilled (fixed device buffers: weights, boundary, constants,
        incoming claims) and its outputs re-labelled: claims on layer l-1, PCS opening states,
        transcript messages.  The proof is the one of prove_layer, layer by layer."""
        import copy
        from . import graph, gpcs
        ctx = self.ctx
        tr = ctx.tr
        dt, pcs = tr.dev, ctx.pcs
        c = hi
        pfx, opfx = f"{c}.", f"{c - 1}."
        slot = _LayerSlot(self, c)
        fill = slot.fill
        # incoming claims (on layer c's boundary): fixed device buffers the graph reads
        assert not any(n.startswith(pfx) for n in list(ctx.wclaims) + list(ctx.virt))
        inn = sorted(n[len(pfx):] for n in ctx.claims if n.startswith(pfx))
        inb = {k: _tmap(ctx.claims[pfx + k], lambda t: t.clone()) for k in inn}
        for k in inn:
            ctx.claims[pfx + k] = inb[k]
        ibuf, isig = _tleaves(inb), _tsig(inb)
        k0 = (set(ctx.claims), set(ctx.wclaims), set(ctx.virt), set(pcs.ucs))
        n0 = ({n: len(v) for n, v in ctx.claims.items()}, {n: len(v) for n, v in ctx.wclaims.items()})
        # the transcript messages of the layer go to a separate log, copied out per replay
        nd0 = dt.ndraw
        saved = (dt.log, dt.cur, dt.ents)
        cap = torch.empty(2 * getattr(self, "_lw", 1 << 22) + (1 << 16), dtype=torch.int32, device=_dev())
        dt.log, dt.cur, dt.ents = cap, 0, []
        fill(c)
        self.layer_head(c)
        nseg0, npend0 = len(pcs.segs), len(pcs.seg_pending)
        g = torch.cuda.CUDAGraph()
        print(f"[run] prove layers {hi}..{lo}: capture", flush=True)
        glog = os.environ.get("ZK_GRAPHLOG")

        def gt(what, t=[0.0]):
            if glog:
                torch.cuda.synchronize()
                now = time.time()
                if t[0]:
                    print(f"[graph] {what}: {now - t[0]:.3f}s", flush=True)
                t[0] = now
        gt("start")
        with graph.capture(g), slot.checks():
            self.layer_body(c)
        gt("capture")
        assert dt.log is cap, "transcript log grew during capture"
        cw, cents, per = dt.cur, dt.ents, dt.ndraw - nd0
        assert all(e[0] != "B" for e in cents)
        dt.log, dt.cur, dt.ents = saved
        self.wit = {}
        # outputs of the captured body (graph memory: copied out after each replay)
        assert not any(n.startswith(pfx) for n in ctx.claims), "incoming claims left unopened"
        assert set(ctx.virt) == k0[2]
        # claims the layer adds to global (fixed) units, e.g. its RoPE claims: templates, re-added
        # per replay with the layer's values
        glob = []
        for store, n0s in ((ctx.claims, n0[0]), (ctx.wclaims, n0[1])):
            for n in list(store):
                if _LAYER_NAME.match(n):
                    continue
                new = store[n][n0s.get(n, 0):]
                if new:
                    glob.append((store, n, new))
                    del store[n][n0s.get(n, 0):]
                    if not store[n]:
                        del store[n]
        assert set(ctx.wclaims) == k0[1]
        outn = sorted(set(ctx.claims) - k0[0])
        assert all(n.startswith(opfx) for n in outn), outn
        outb = {n[len(opfx):]: ctx.claims.pop(n) for n in outn}
        unew = [k for k in pcs.ucs if k not in k0[3]]
        assert all(n.startswith(pfx) for n, _ in unew), unew
        ucst = {k: pcs.ucs.pop(k) for k in unew}
        # the layer's segment: its record is cloned per replay and its combined columns (graph
        # memory) are copied to host memory before the next replay overwrites them
        csegs = pcs.segs[nseg0:]
        del pcs.segs[nseg0:]
        del pcs.seg_pending[npend0:]
        clone = lambda t: t.clone()
        for l in range(c, lo - 1, -1):
            if l != c:
                fill(l)
                self.layer_head(l)
                cur = {k: ctx.claims.pop(f"{l}.{k}") for k in inn}
                assert _tsig(cur) == isig, f"layer {l}: incoming claims differ from the captured layer's"
                for d, s_ in zip(ibuf, _tleaves(cur)):
                    d.copy_(s_)
            d0 = dt.ndraw
            gt(f"layer {l} inputs")
            g.replay()
            gt(f"layer {l} replay")
            # transcript: the layer's messages and draws
            dt.reserve(cw)
            off = dt.cur
            dt.log[off:off + cw].copy_(cap[:cw])
            dt.ents.extend((off + o, shp) for o, shp in cents)
            dt.cur += cw
            dt.ndraw = d0 + per

            def kf(keys, l=l, d0=d0):
                out = []
                for k in keys:
                    if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int):
                        k = (k[0] - nd0 + d0, k[1]) if k[0] >= nd0 else ("gk", l, k[0], k[1])
                    out.append(k)
                return tuple(out)
            for k, v in outb.items():
                ctx.claims[f"{l - 1}.{k}"] = _tmap(v, clone, kf)
            for store, n, new in glob:
                store.setdefault(n, []).extend(_tmap(new, clone, kf))
            for (n, cc), uc in ucst.items():
                n2 = f"{l}." + n[len(pfx):]
                u2 = copy.copy(uc)
                u2.unit = pcs.units[n2]
                for a in ("rho", "alpha", "y"):
                    setattr(u2, a, _tmap(getattr(uc, a), clone))
                u2.v = uc.v.clone() if uc.v is not None else None
                u2._keep = None
                pcs.ucs[(n2, cc)] = u2
            for sg in csegs:
                ren = lambda n, l=l: f"{l}." + n[len(pfx):]
                s2 = gpcs.SEG([ren(n) for n in sg.names], [(ren(n), cc) for n, cc in sg.keys])
                for pu in sg.pus:
                    p2 = gpcs.PU(s2, pu.c, _tmap(pu.rho, clone), _tmap(pu.y, clone), pu.v.clone())
                    p2.hd = pu.hd
                    s2.pus.append(p2)
                    pcs.seg_pending.append(p2)
                pcs.segs.append(s2)
            pcs.flush_segments()
            if gpcs._POOL.stream is not None:
                torch.cuda.current_stream().wait_stream(gpcs._POOL.stream)
        gt("outputs")
        # graph memory is released (what the copies above need is queued before)
        torch.cuda.current_stream().synchronize()
        if gpcs._POOL.stream is not None:
            gpcs._POOL.stream.synchronize()
        del g, inb, ibuf, outb, ucst, cap
        slot.close()

    def layer_groups(self, l):
        p = f"{l}."
        mlp = [p + "moe"] if self.cfg.moe else [p + f"mlp{s}" for s in range(len(self.slices()))]
        return [p + "in"] + [p + f"att{g}" for g in range(self.sh.nkv)] + [p + "ao", p + "mi"] + mlp + [p + "mo"]

    def layer_steps(self, l, commit=False):
        tm = (lambda k: self.tm("cnt." + k)) if commit else self.tm
        with tm("mo"):
            self.step_mo(l)
        with tm("mlp"):
            if self.cfg.moe:
                mp = self.moe_of(l)
                mp.steps()
                if self.ctx.active:
                    self.ctx.open(f"{l}.moe", mp.tensors() if self.ctx.P else None)
            else:
                for s in range(len(self.slices())):
                    self.step_mlp(l, s)
        W = self.wit.get(l) if self.ctx.proving else None
        if W is not None and not self.cfg.moe:
            for k in ("accg", "accu", "c", "a", "m"):
                setattr(W, k, None)
            W.ru["q"] = W.rm["q"] = None
        self.memlog(f"{l} after mlp")
        with tm("mi"):
            self.step_mi(l)
        with tm("ao"):
            self.step_ao(l)
        self.memlog(f"{l} after ao")
        with tm("att"):
            gs = list(range(self.sh.nkv))
            for i in range(0, len(gs), ATT_MERGE):
                self.step_att_pre(l, gs[i:i + ATT_MERGE])
            for gb in self.att_batches():
                self.step_att(l, gb)
                if W is not None:
                    for g in gb:
                        W.att[g] = None
        self.memlog(f"{l} after att")
        with tm("in"):
            self.step_in(l)


LAST = {}


def _dlist(sh, vals, dtype=torch.int64):
    """Cached device tensor of a host list (per shape object; uploads from lists block the host)."""
    cache = sh.__dict__.setdefault("_dlist", {})
    key = (dtype, tuple(vals))
    t = cache.get(key)
    if t is None:
        t = cache[key] = torch.tensor(list(vals), dtype=dtype, device=_dev())
    return t


def _tile_index(sh):
    """Device row / column tile indices of sh.tiles (cached: uploads from lists block the host)."""
    c = sh.__dict__.get("_tile_dev")
    if c is None:
        c = sh._tile_dev = (torch.tensor([a for a, b in sh.tiles], device=_dev()),
                            torch.tensor([b for a, b in sh.tiles], device=_dev()))
    return c


def _to_pinned(t):
    """Device tensor -> pinned host copy, queued on the current stream (stream-ordered with later
    uploads of it; the host must not read it before a sync)."""
    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    h.copy_(t, non_blocking=True)
    return h


class Policy:
    """Public policy of the statement (spec, Statement): context bound, temperature index,
    top-k, maximum output length G_max, stop set, tokenizer/template ids, server-prefix
    digest.  Temperature indices index mdl.THETA (0: greedy); top-k is not implemented.

    autoregressive: the generated tokens are fed back, i.e. targets[b, i] is the token at
    position S - ngen + 1 + i of the proved sequence for i < ngen - 1 (y_t is sampled from
    the logits at the position before it).  The prompt is tokens[b, :S - ngen + 1]."""

    def __init__(self, gmax=None, stop=(), theta=0, topk=None, ctx_bound=None, tokenizer=b"", template=b"",
                 hpre=bytes(32), autoregressive=True):
        if topk is not None:
            raise NotImplementedError("top-k sampling is not implemented")
        if not 0 <= theta < len(mdl.THETA):
            raise ValueError(f"temperature index {theta} not in Theta")
        self.gmax, self.stop, self.theta, self.topk = gmax, tuple(sorted(set(int(x) for x in stop))), theta, topk
        self.ctx_bound, self.tokenizer, self.template, self.hpre = ctx_bound, tokenizer, template, hpre
        self.autoregressive = autoregressive

    def encode(self):
        import hashlib
        parts = [f"{self.ctx_bound}|{self.theta}|{self.topk}|{self.gmax}|{int(self.autoregressive)}|".encode(),
                 np.asarray(self.stop, dtype="<i8").tobytes(), hashlib.sha256(self.tokenizer).digest(),
                 hashlib.sha256(self.template).digest(), bytes(self.hpre)]
        return b"".join(len(x).to_bytes(4, "little") + x for x in parts)

    def check(self, sh, tokens, targets):
        """Verifier checks on the public statement: termination rule and the fed-back tokens."""
        y = np.asarray(targets.cpu()).reshape(sh.B, sh.ng)
        x = np.asarray(tokens.cpu()).reshape(sh.B, sh.S)
        if min(x.min(), y.min()) < 0 or max(x.max(), y.max()) >= sh.cfg.V:
            raise ValueError("statement: token outside the vocabulary")
        gmax = sh.ng if self.gmax is None else self.gmax
        if self.ctx_bound is not None and sh.S > self.ctx_bound:
            raise ValueError("statement: context bound")
        for b in range(sh.B):
            stop = np.isin(y[b], self.stop)
            if stop[:-1].any():
                raise ValueError("statement: output continues after a stop token")
            if not (stop[-1] or sh.ng == gmax):
                raise ValueError("statement: output truncated before a stop token or G_max")
            if sh.ng > gmax:
                raise ValueError("statement: output longer than G_max")
            if self.autoregressive and not np.array_equal(x[b, sh.S - sh.ng + 1:], y[b, :-1]):
                raise ValueError("statement: generated tokens are not the fed-back sequence")


def _wdigest(ctx, gnames):
    import hashlib
    h = hashlib.sha256(b"C_W|")
    for g in gnames:
        for b in ctx.pcs.cap_bytes(g):
            h.update(b)
    return h.digest()


def sampling_seed(tokens, ngen, C_W, policy, nc=bytes(32)):
    """r = H(seed | n_c | C_W | H(x) | policy), x the prompt tokens[:, :S - ngen + 1]."""
    import hashlib
    x = np.asarray(tokens.cpu())[:, :tokens.shape[1] - ngen + 1].astype("<i8")
    return hashlib.sha256(b"seed" + bytes(nc) + C_W + hashlib.sha256(x.tobytes()).digest()
                          + policy.encode()).digest()


def decode(M, tokens, ngen, policy=None, u=None):
    """Statement for decoding under the policy: the prompt is tokens[:, :S - ngen + 1]; the rest
    of tokens is overwritten by the decoded sequence.  Returns (tokens, targets (B, ngen)).
    u: (B ngen,) PRF values (needed for theta > 0, see sampling_seed and mdl.prf_u).
    Causal attention makes each row's logits independent of later positions, so ngen full
    forward passes give the token-by-token decode exactly."""
    theta = (policy or Policy()).theta
    tokens = tokens.clone()
    B, S = tokens.shape
    for i in range(ngen):
        h = mdl.forward(M, tokens).reshape(B, S, -1)
        z = mdl.final_logits(M, h[:, S - ngen:].reshape(B * ngen, -1))
        y = mdl.sample(z, u if u is not None else torch.zeros(B * ngen, dtype=torch.int64), theta).reshape(B, ngen)
        del h, z
        if i == ngen - 1:
            return tokens, y
        tokens[:, S - ngen + 1 + i] = y[:, i]


def greedy_decode(M, tokens, ngen):
    return decode(M, tokens, ngen)


def _cw_key(M, sh, pcs_args):
    """Cache key of a weight commitment: block layout, PCS parameters and the weight objects
    (models made by copy.copy share the cache dict; a different layer list is another key)."""
    return (sh.cb, sh.cbv, tuple(sorted(pcs_args.items())), tuple(id(Ly) for Ly in M.layers), id(M.Wlm), id(M.emb))


# units of public data committed once and opened like witness units (the verifier checks their
# digest instead of evaluating them): the LogUp table, the RoPE tables
FIXED_UNITS = ("tab", "rope")


def _fixed_key(M, sh, theta, gname, pcs_args):
    pa = tuple(sorted(pcs_args.items()))
    if gname == "tab":
        return (gname, theta, pa, M.layers[0].E, M.eG, M.eA, M.cfg.dh)
    return (gname, pa, M.cfg.theta, M.cfg.dh, sh.S, sh.T)


def fixed_digest(M, sh, theta, gname, D=4, **pcs_args):
    """Digest of fixed unit gname's commitment for (model, shape, temperature): what the verifier
    expects the prover's caps to hash to (a constant of a recursive verifier).  Computed once and
    cached on M."""
    cache = M.__dict__.setdefault("_tab_cache", {})
    key = _fixed_key(M, sh, theta, gname, pcs_args)
    if key not in cache:
        from .transcript import Transcript
        ctx = Ctx("prove", Transcript(b"C_T", D), D, **pcs_args)
        pr = Proto(ctx, M, sh, None, None)
        pr.theta = theta
        pr.declare()
        ctx.mode = "commit"
        ctx.commit(gname, pr.fixed_tensors(gname))
        cache[key] = (_wdigest(ctx, [gname]), ctx.pcs.cap_bytes(gname))
    return cache[key][0]


def weights_digest(M, sh, D=4, **pcs_args):
    """C_W: digest of the weight commitment (Merkle caps of every weight group), computed
    once per model and block layout and cached on M.  Returns the digest."""
    key = _cw_key(M, sh, pcs_args)
    cache = M.__dict__.setdefault("_cw_cache", {})
    if key not in cache:
        from .transcript import Transcript
        ctx = Ctx("prove", Transcript(b"C_W", D), D, **pcs_args)
        pr = Proto(ctx, M, sh, None, None)
        pr.declare()
        ctx.mode = "commit"
        for gname in pr.wgroups():
            if gname[0].isdigit():
                pr.use_layer(int(gname.split(".")[0]))
            ctx.commit(gname, pr.weight_tensors(gname))
        if pr._host_layers is not None:
            M.layers = pr._host_layers
        cache[key] = (_wdigest(ctx, pr.wgroups()), {g: ctx.pcs.cap_bytes(g) for g in pr.wgroups()})
    return cache[key][0]


def statement(M, sh, tokens, targets, C_W, policy=None, nc=bytes(32), salt=bytes(32)):
    """mode | policy | C_W | h_circ | H(x) | H(y) | n_c | S | salt (spec, Transcript and phases)."""
    import hashlib
    policy = policy or Policy()
    hx = hashlib.sha256(np.asarray(tokens.cpu()).astype("<i8").tobytes()).digest()
    hy = hashlib.sha256(np.asarray(targets.cpu()).astype("<i8").tobytes()).digest()
    circ = f"{vars(M.cfg)}|{sh.B}|{sh.S}|{sh.cb}|{sh.cbv}|{sh.ng}"
    # h_circ: the configuration, the shape and every constant of F_O outside C_W (model.const_digest)
    hc = hashlib.sha256(circ.encode() + mdl.const_digest(M, sh.S, policy.theta)).digest()
    assert len(C_W) == 32 and len(nc) == 32 and len(salt) == 32
    pol = policy.encode()
    return (b"P|" + len(pol).to_bytes(4, "little") + pol + C_W + hc + hx + hy + bytes(nc)
            + int(sh.T).to_bytes(8, "little") + bytes(salt))


def _set_sampling(pr, sh, tokens, C_W, policy, nc):
    pr.theta = policy.theta
    if policy.theta:
        pr.u = mdl.prf_u(sampling_seed(tokens, sh.ng, C_W, policy, nc), sh.R)


def prove(M, sh, tokens, targets, D=4, C_W=None, policy=None, nc=bytes(32), salt=bytes(32), **pcs_args):
    from .transcript import Transcript
    policy = policy or Policy()
    policy.check(sh, tokens, targets)
    mdl.ensure_caps(M)
    t0 = time.time()
    if C_W is None:
        C_W = weights_digest(M, sh, D, **pcs_args)
    t_cw = time.time() - t0
    tr = Transcript(statement(M, sh, tokens, targets, C_W, policy, nc, salt), D)
    ctx = Ctx("prove", tr, D, **pcs_args)
    pr = Proto(ctx, M, sh, tokens, targets)
    pr.C_W = C_W
    _set_sampling(pr, sh, tokens, C_W, policy, nc)
    pr.run()
    tr.flush()
    LAST.clear()
    LAST.update(times=dict(pr.tm.t), bytes=tr.sent_bytes, bytes_by=dict(pr.tm.b), C_W=C_W, t_cw=t_cw,
                lookups=ctx.chunks.total, chunks=ctx.chunks.n())
    return tr.msgs


def verify(M, sh, tokens, targets, msgs, D=4, C_W=None, policy=None, nc=bytes(32), salt=bytes(32), cpu=False,
           **pcs_args):
    """C_W: the registry's weight commitment; for open weights the verifier may pass None
    and recompute it from M.  cpu: verify on the host only (no GPU; needs C_W)."""
    if cpu:
        from . import ext
        assert C_W is not None, "the host verifier takes C_W from the registry"
        ext.CPU[0] = True
        try:
            return verify(M, sh, tokens, targets, msgs, D, C_W, policy, nc, salt, **pcs_args)
        finally:
            ext.CPU[0] = False
    from .transcript import Transcript
    policy = policy or Policy()
    policy.check(sh, tokens, targets)
    mdl.ensure_caps(M)
    if C_W is None:
        C_W = weights_digest(M, sh, D, **pcs_args)
    tr = Transcript(statement(M, sh, tokens, targets, C_W, policy, nc, salt), D, msgs=msgs)
    ctx = Ctx("verify", tr, D, **pcs_args)
    pr = Proto(ctx, M, sh, tokens, targets)
    pr.C_W = C_W
    _set_sampling(pr, sh, tokens, C_W, policy, nc)
    return pr.run()
