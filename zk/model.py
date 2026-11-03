"""The integer function F_O for a Qwen2-style transformer, and the
quantization program that produces it from a BF16 checkpoint.

Every gadget is the exact integer definition proved by the constraint system
(gadgets.py) and returns its output and the witness values the prover
commits.  Tensors are torch int64 on the GPU.  Conventions fixed by C_W:

  residual h: f_h fractional bits, h in A = [-2^23, 2^23)
  matmul inputs: int8 rows with a per-row exponent (Requant8 + NormScale);
  weights int8 with a per-output-column exponent (per-tensor for W_up)
  Q, K, V: static per-head exponents with e^Q_h + e^K_g = E_attn (global),
  bias in output units; RoPE in the 8-bit domain (9-bit result)
  P = tau_exp(acc - M) in [0, 2^10]; D = 2^E delta + rho (signed Norm); N in
  aligned key blocks of 2^12, each shifted by k_m = max(7, E); o = MulRound(N,
  floor(2^32 / delta), 25 + E - k_m + vshift_h)
  MLP gate: SiLU by table on c = round(acc / 2^k) (global input units e_G,
  output units 2^-e_A, 16-bit); up: dynamic row exponent; m = a * q_up
  logits: z = round(acc / 2^k) in units 2^-8; greedy argmax.
"""
import math
import os
import numpy as np
import torch
from .ext import dev as _dev  # noqa: E402

A_BITS = 23
KN = 8
KEYBLK = 12
EXP_BITS = 10
RECIP_NUM = 1 << 32
O_BASE = 25
KM_MIN = 7
RSQ_C = 127 << 15
SILU_CMAX = 1 << 14
EZ = -8
KMIN_SR = -16
_CALIB = [False]
SILU_KMAX = 15


# ---------------------------------------------------------------- deferred checks
# Witness sanity checks (prover side) are device booleans; inside deferred_checks() they are
# collected and verified with one synchronization at the end instead of one each.
_DEFER = [0]
_PENDING = []


class WitnessError(AssertionError):
    """A prover-side check failed: the input is outside the domain of F_O (or the witness is wrong),
    and the honest prover aborts.  Raised explicitly, so it holds under python -O as well."""


def check(cond, msg="witness check"):
    """cond: a bool tensor (all entries must hold) or a Python bool."""
    if not torch.is_tensor(cond):
        if not cond:
            raise WitnessError(msg)
        return
    c = cond.all().reshape(1)
    if _DEFER[0]:
        _PENDING.append((c, msg))
        return
    if not bool(c):
        raise WitnessError(msg)


def flush_checks():
    if not _PENDING:
        return
    ok = torch.cat([c for c, _ in _PENDING]).cpu().tolist()
    msgs = [m for _, m in _PENDING]
    _PENDING.clear()
    for v, m in zip(ok, msgs):
        if not v:
            raise WitnessError(m)


class deferred_checks:
    def __enter__(self):
        _DEFER[0] += 1

    def __exit__(self, *exc):
        _DEFER[0] -= 1
        if _DEFER[0] == 0:
            if exc[0] is None:
                flush_checks()
            else:
                _PENDING.clear()
        return False


def fdiv(a, b):
    return torch.div(a, b, rounding_mode="floor")


def pow2(e):
    if torch.is_tensor(e):
        return torch.ones_like(e, dtype=torch.int64) << e.to(torch.int64)
    return 1 << e


_RQK = []


def _rowq(a, emin, emax):
    """least e in [emin, emax] with 2a in [-255 2^e, 255 2^e) for the whole row."""
    if a.is_cuda and a.dim() >= 1 and a.numel() and os.environ.get("ZK_ROWQ_OLD") is None:
        return _rowq_gpu(a, emin, emax)
    return _rowq_ref(a, emin, emax)


def _rowq_gpu(a, emin, emax):
    import numpy as np
    from . import kern
    if not _RQK:
        mod = kern.file_module("rowq.cu", 8)
        _RQK.extend([mod.get_function("rowq"), mod.get_function("rowq32")])
    shp = a.shape
    N = shp[-1]
    i32 = a.dtype == torch.int32
    x = a.reshape(-1, N).to(torch.int32 if i32 else torch.int64).contiguous()
    R = x.shape[0]
    q = torch.empty_like(x)
    v = torch.empty((5, R), dtype=torch.int64, device=a.device)
    err = torch.zeros(2, dtype=torch.int32, device=a.device)
    _RQK[1 if i32 else 0]((R,), (256,), tuple([kern.ptr(x), np.int64(N), np.int32(emin), np.int32(emax), kern.ptr(q)]
                                + [kern.ptr(v[i]) for i in range(5)] + [kern.ptr(err)]))
    check(err[0] == 0, "requant: |q| > 127")
    check(err[1] == 0, "requant: no witness column")
    rs = shp[:-1]
    e, js, sig, f, asel = (t.reshape(rs) for t in v)
    return q.reshape(shp), e, dict(jstar=js, sigma=sig, f=f, asel=asel)


def _rowq_ref(a, emin, emax):
    hi = a.max(dim=-1).values
    lo = a.min(dim=-1).values
    e = torch.full_like(hi, emax)
    for c in range(emax, emin - 1, -1):
        s = 1 << c
        ok = (2 * hi < 255 * s) & (2 * lo >= -255 * s)
        e = torch.where(ok, torch.full_like(e, c), e)
    s = pow2(e).unsqueeze(-1)
    q = fdiv(2 * a + s, 2 * s)
    assert int(q.abs().max()) <= 127
    f = e > emin
    cand = (4 * a >= 255 * s) | (4 * a < -255 * s)       # does not fit at e - 1
    jstar = torch.argmax(cand.to(torch.int8), dim=-1)
    jstar = torch.where(f, jstar, torch.zeros_like(jstar))
    asel = torch.gather(a, -1, jstar.unsqueeze(-1)).squeeze(-1)
    sigma = torch.where(f, (4 * asel >= 255 * s.squeeze(-1)).to(torch.int64), torch.zeros_like(e))
    assert bool(torch.all(~f | torch.gather(cand, -1, jstar.unsqueeze(-1)).squeeze(-1)))
    return q, e, dict(jstar=jstar, sigma=sigma, f=f.to(torch.int64), asel=asel)


def requant8(a, e0=0, emax=17):
    return _rowq(a, e0, emax)


def shiftq8(acc, emin=2, emax=22):
    return _rowq(acc, emin, emax)


_RSQ = None


def rsqrt_table():
    global _RSQ
    if _RSQ is None:
        out = np.zeros(1 << 16, dtype=np.int64)
        C2 = RSQ_C * RSQ_C
        for i in range(1, 1 << 16):
            out[i] = math.isqrt(C2 // i)
        _RSQ = torch.from_numpy(out).to(_dev())
    return _RSQ


def norm(X, E):
    """Signed Norm: 2^(E-) X = 2^(E+) delta + rho with rho in [0, 2^(E+))."""
    Ep, En = E.clamp(min=0), (-E).clamp(min=0)
    delta = (X >> Ep) << En
    return delta, X - ((X >> Ep) << Ep), Ep, En


def normscale(q):
    """Sigma = sum q^2 + 1 = 2^E delta + rho, E even in [-14, 14], delta in [2^14, 2^16)."""
    sig = (q * q).sum(dim=-1) + 1
    Eh = fdiv(_bitlen(sig) - 15, 2)
    delta, rho, Ep, En = norm(sig, 2 * Eh)
    check((delta.min() >= (1 << 14)) & (delta.max() < (1 << 16)), "normscale: mantissa range")
    mu = rsqrt_table()[delta]
    qp = fdiv(q * mu.unsqueeze(-1) + (1 << 14), 1 << 15)
    check(qp.abs().max() <= 127, "normscale: |q'| > 127")
    return qp, Eh, dict(sig=sig, delta=delta, rho=rho, mu=mu, Ep=Ep, En=En)


_SQK = []


def shiftq8s_fused(acc, kr, kc, beta):
    """shiftq8s with k = kr[row] + kc[col] (>= 1): (q, o, wh) as int32, range checks included."""
    import cupy as cp
    if not _SQK:
        _SQK.append(cp.ElementwiseKernel(
            "T x, int64 kr, int64 kc, int64 beta", "int32 q, int32 o, int32 wh, raw int32 err",
            "long long k = kr + kc; if (k < 1 || k > 22) { atomicAdd(&err[0], 1); k = 1; }"
            "long long s = 1ll << k, h = s >> 1, a = (long long)x;"
            "long long r = ((a + h) >> k) + beta;"
            "long long qq = r < -127 ? -127 : (r > 127 ? 127 : r);"
            "int oo = (r > qq) - (r < qq);"
            "long long w = oo != 0 ? oo * (a + s * beta) - 255 * h - (oo == -1 ? 1 : 0) : 0;"
            "if (w < 0 || w >= (1ll << 30)) atomicAdd(&err[1], 1);"
            "q = (int)qq; o = oo; wh = (int)(w >> 15);", "zk_shiftq8s"))
    shp = acc.shape
    ex = lambda v, s: v.reshape(s).to(torch.int64).expand(shp)
    q = torch.empty(shp, dtype=torch.int32, device=acc.device)
    o = torch.empty_like(q)
    wh = torch.empty_like(q)
    err = torch.zeros(2, dtype=torch.int32, device=acc.device)
    _SQK[0](cp.from_dlpack(acc.contiguous()), cp.from_dlpack(ex(kr, (-1, 1))), cp.from_dlpack(ex(kc, (1, -1))),
            cp.from_dlpack(ex(beta, (1, -1))), cp.from_dlpack(q), cp.from_dlpack(o), cp.from_dlpack(wh),
            cp.from_dlpack(err))
    check(err[0] == 0, "shiftq8s: shift out of range")
    check(err[1] == 0, "shiftq8s: w out of range")
    return q, dict(o=o, wh=wh)


# ShiftQ8S is wrap-free for k = kappa_n + E_h <= k_max(b_n): its saturation witness o (acc + 2^k b) -
# 255 2^(k-1) - [o = -1] must stay below p - 2^30 in absolute value with |acc| <= d 127^2, which also
# keeps the o = 0 branch (k <= 22) below p.  The gate shift kcg_n + E_h of ShiftRound into tau_SiLU is
# held to 15.  F_O is defined for the rows within these bounds; the prover refuses the others.
QKV_KMAX, GATE_KMAX = 22, 15
_PR = (1 << 31) - (1 << 24) + 1


def qkv_kcap(cfg, Ly):
    """The largest row exponent E_h for which every column's ShiftQ8S of layer Ly is wrap-free:
    min_n (k_max(b_n) - kappa_n)."""
    lim = _PR - (1 << 30) - cfg.d * 127 * 127
    b = Ly.bias.detach().abs().to(torch.float64).cpu().numpy() + 128
    kmax = np.minimum(np.floor(np.log2((lim - 1) / b)), QKV_KMAX).astype(np.int64)
    while True:                                  # exact integer check of the float estimate
        bad = (np.left_shift(np.int64(1), kmax) * b.astype(np.int64)) >= lim
        if not bad.any():
            break
        kmax = kmax - bad
    return int((kmax - Ly.kc.detach().cpu().numpy().astype(np.int64)).min())


def gate_kcap(M, Ly):
    """The largest row exponent E_h for the gate's ShiftRound (kcg_n + E_h <= 15)."""
    return GATE_KMAX - int((M.eG - (Ly.eWge if M.cfg.moe else Ly.eWg)).max())


def gate_kmin(M, Ly):
    """Dense layers: the least row exponent for the gate's ShiftRound (kcg_n + E_h >= 1); a row with a
    smaller E_h is shifted as if E_h were this bound (E_g = max(E_h, K_gl))."""
    if "kgl" in Ly.__dict__:
        return Ly.kgl
    return 1 - int((M.eG - Ly.eWg).min())


def ensure_caps(M):
    """Per layer: kqs, kgm (host ints, the layer parameters Kqs, Kgm) and kqs_t, kgm_t (device copies
    for the prover's checks, refilled per layer by a captured graph's slot); dense: kgl (the gate's
    lower clamp K_gl) and kgl_t; MoE: krc = -(eWr + R_FRAC) (the router shift is max(Eh + krc, 1)) and
    its device copy krc_t."""
    for Ly in M.layers:
        if "kqs" not in Ly.__dict__:
            Ly.kqs, Ly.kgm = qkv_kcap(M.cfg, Ly), gate_kcap(M, Ly)
            dv = Ly.bias.device
            Ly.kqs_t = torch.tensor(Ly.kqs, dtype=torch.int64, device=dv)
            Ly.kgm_t = torch.tensor(Ly.kgm, dtype=torch.int64, device=dv)
            if not M.cfg.moe:
                Ly.kgl = gate_kmin(M, Ly)
                Ly.kgl_t = torch.tensor(Ly.kgl, dtype=torch.int64, device=dv)
                # a clamped row's gate shift kcg_n + K_gl must stay <= 15 like an unclamped one's
                if Ly.kgl > Ly.kgm:
                    raise ValueError("model: gate exponents spread over more than 15 shifts (K_gl > K_gm)")
            if M.cfg.qk_norm and "kn" in Ly.__dict__:
                kh = Ly.kn.view(-1, M.cfg.dh)
                if not bool((kh == kh[:, :1]).all()):    # the proof reads kn per head (column 0)
                    raise ValueError("model: QK-norm shift kn not constant per head")
            if M.cfg.moe:
                Ly.krc = -(int(Ly.eWr) + R_FRAC)
                Ly.krc_t = torch.tensor(Ly.krc, dtype=torch.int64, device=dv)


_MODEL_SCALARS = ("f_h", "Kz", "E_attn", "eG", "eA", "eWlm")


def const_digest(M, S, theta=0):
    """SHA-256 of every constant of F_O that the weight commitment C_W does not cover: the model's
    and every layer's exponents, shifts and biases, and the contents of the tables computed from them
    in floating point (tau_exp, tau_SiLU, the RoPE tables for S positions, tau_exp,theta), so the
    statement pins the function whatever platform computed the tables."""
    import hashlib

    def blob(v):
        if torch.is_tensor(v):
            return v.detach().to(torch.int64).cpu().numpy().astype("<i8").tobytes()
        if isinstance(v, (list, tuple)):
            return repr([int(x) if not torch.is_tensor(x) else x.tolist() for x in v]).encode()
        return repr(int(v)).encode()
    h = hashlib.sha256(b"F_O constants v1|")
    for k in _MODEL_SCALARS:
        h.update(k.encode() + b"=" + blob(getattr(M, k)) + b"|")
    if M.cfg.down_b != 127:
        h.update(b"down_b=" + blob(M.cfg.down_b) + b"|")
    for l, Ly in enumerate(M.layers):
        for k in sorted(vars(Ly)):
            if k.startswith(("W", "_")) or k in ("bqkv",) + _CAPS:
                continue                      # weights (in C_W), the float bias source, derived bounds
            h.update(f"{l}.{k}=".encode() + blob(getattr(Ly, k)) + b"|")
    cos, sin = rope_tables(M.cfg, S)
    tables = [exp_table(M.E_attn, M.cfg.dh), _silu(torch.arange(-SILU_CMAX, SILU_CMAX + 1), M.eG, M.eA), cos, sin]
    if theta:
        tables.append(samp_table(theta))
    for t in tables:
        h.update(blob(t))
    return h.digest()


_CAPS = ("kqs", "kgm", "kqs_t", "kgm_t", "kgl", "kgl_t", "krc", "krc_t")


def drop_caps(M):
    for Ly in M.layers:
        for k in _CAPS:
            Ly.__dict__.pop(k, None)


def shiftq8s(acc, k, beta):
    s = pow2(k)
    r = fdiv(acc + s // 2, s) + beta
    q = r.clamp(-127, 127)
    o = torch.sign(r - q)
    w = o * (acc + s * beta) - 255 * (s // 2) - (o == -1).to(torch.int64)
    w = torch.where(o != 0, w, torch.zeros_like(w))
    check((w.min() >= 0) & (w.max() < (1 << 30)), "shiftq8s: w out of range")
    return q, dict(o=o, wh=w >> 15, w=w)


def rope8(q1, q2, cos, sin):
    r1 = fdiv(q1 * cos - q2 * sin + (1 << 14), 1 << 15)
    r2 = fdiv(q2 * cos + q1 * sin + (1 << 14), 1 << 15)
    return r1, r2


def shiftround(acc, k):
    s = pow2(k)
    return fdiv(acc + s // 2, s)


def gshiftround(acc, k):
    """Per-row shift k in [KMIN_SR, 24] (k <= 0 is an exact left shift);
    the output must lie in A (|c| < 2^23), the regime bound of GShiftRound."""
    if _CALIB[0]:
        k = k.clamp(KMIN_SR, 24)
    kk = k.clamp(min=0)
    c = torch.where(k > 0, fdiv(acc + (pow2(kk) >> 1), pow2(kk)), acc * pow2((-k).clamp(min=0)))
    if _CALIB[0]:
        _env("gsr", float(c.abs().max()) / (1 << A_BITS) if c.numel() else 0.0)
        return c.clamp(-(1 << A_BITS), (1 << A_BITS) - 1)
    check((k.min() >= KMIN_SR) & (k.max() <= 24), "gshiftround: shift out of range")
    check(c.abs().max() < (1 << A_BITS), "gshiftround: output outside A")
    return c


def satadd(h, c):
    s = h + c
    hp = s.clamp(-(1 << A_BITS), (1 << A_BITS) - 1)
    return hp, dict(o=torch.sign(s - hp))


def _bitlen(x):
    """bit length of x (0 for x <= 0), x < 2^40."""
    check(x.max() < (1 << 40), "bitlen: input >= 2^40")
    e = torch.frexp(x.double()).exponent.to(x.dtype)
    return torch.where(x > 0, e, torch.zeros_like(e))


def norm_den(D):
    """Softmax denominator D in [2^10, 2^30]: D = 2^E delta + rhoD, delta in [2^15, 2^16),
    E in [-5, 15]; block shift k_m = max(7, E)."""
    E = _bitlen(D) - 16
    delta, rhoD, _, _ = norm(D, E)
    check((delta.min() >= (1 << 15)) & (delta.max() < (1 << 16)), "norm_den: mantissa range")
    check((E.min() >= -5) & (E.max() <= 14), "norm_den: exponent range")
    km = E.clamp(min=KM_MIN)
    return dict(E=E, delta=delta, rhoD=rhoD, km=km, fk=(E >= KM_MIN).to(torch.int64))


def mulround(x, y, k):
    """round(x y / 2^k) for x in A, |y| <= 2^17, k in [13, 30], without forming x y."""
    xh = fdiv(x, 1 << 12)
    xl = x - (xh << 12)
    t = fdiv(xl * y, 1 << 12)
    H = xh * y + t
    kp = k - 12
    c = fdiv(H + (pow2(kp) >> 1), pow2(kp))
    return c, dict(xh=xh, t=t, kp=kp)


def norm_div(N, nd, vshift):
    """o = MulRound(N, recip(delta), O_BASE + E - k_m + vshift), N = sum_b ShiftRound(N^(b), k_m)."""
    check(N.abs().max() < (1 << A_BITS), "norm_div: N outside A")
    recip = RECIP_NUM // nd["delta"]
    k = O_BASE + nd["E"] - nd["km"] + vshift
    check((k.min() >= 13) & (k.max() <= 30), "norm_div: shift range")
    o, w = mulround(N, recip.unsqueeze(-1), k.unsqueeze(-1))
    return o, dict(recip=recip, xh=w["xh"], t=w["t"], kp=w["kp"].squeeze(-1))


# =================================================================== model
class Cfg:
    """qk_norm: per-head RMSNorm of the queries and keys before RoPE (Qwen3); qkv_bias: QKV biases
    (Qwen2); down_b: the bound B of the down projection's input |q| <= B (127: Requant8; 255: a 9-bit
    Requant, which needs dff 127 B + 2^30 < p for GShiftRound to stay wrap-free)."""
    def __init__(self, L, d, nh, nkv, dh, dff, V, theta=1e6, qk_norm=False, qkv_bias=True, moe=None, down_b=127):
        self.L, self.d, self.nh, self.nkv, self.dh, self.dff, self.V = L, d, nh, nkv, dh, dff, V
        self.theta = theta
        self.qk_norm, self.qkv_bias = qk_norm, qkv_bias
        self.moe = moe                 # MoE MLP in every layer: dict(E experts, k active, de width)
        # the down accumulator feeds GShiftRound, whose output bound needs |acc| + 2^30 + 2^23 < p (tables.sr_row)
        assert down_b in (127, 255) and (down_b == 127 or (not moe and dff * 127 * down_b + (1 << 30) + (1 << 23) < _PR))
        # MoE: the merged row (k de values of |q2| <= 1023) likewise; k <= 8 keeps the Div relation wrap-free
        assert not moe or (moe["k"] <= 8 and moe["k"] * moe["de"] * 127 * 1023 + (1 << 30) + (1 << 23) < _PR)
        self.down_b = down_b


QWEN32B = Cfg(64, 5120, 40, 8, 128, 27648, 152064)
QWEN3_32B = Cfg(64, 5120, 64, 8, 128, 25600, 151936, qk_norm=True, qkv_bias=False)
QWEN3_30B_A3B = Cfg(48, 2048, 32, 4, 128, 8 * 768, 151936, qk_norm=True, qkv_bias=False,
                   moe=dict(E=128, k=8, de=768))
ARCHS = {"Qwen/Qwen2.5-32B-Instruct": QWEN32B, "Qwen/Qwen3-32B": QWEN3_32B,
         "Qwen/Qwen3-30B-A3B": QWEN3_30B_A3B}


_TEXP = {}


def round_exact(v, exact, tol):
    """The integers nearest to the exact real values that v (float64) approximates within tol / 2,
    ties to even: the result does not depend on the platform's floating point.  Entries whose
    fractional part lies within tol of 1/2 are recomputed with exact(i) (an mpmath value at 60
    digits); elsewhere every approximation within tol / 2 rounds the same way."""
    import mpmath
    r = torch.round(v)
    near = ((v - torch.floor(v) - 0.5).abs() < tol).nonzero().flatten().cpu().tolist()
    if near:
        with mpmath.workdps(60):
            out = []
            for i in near:
                x = exact(i)
                n = int(mpmath.floor(x))
                f = x - n
                out.append(n + 1 if f > 0.5 or (f == 0.5 and n % 2) else n)
        r.view(-1)[torch.tensor(near, device=r.device)] = torch.tensor(out, dtype=r.dtype, device=r.device)
    return r


def exp_table(E_attn, dh):
    """tau_exp(x) = round(2^10 exp(-x 2^E_attn / sqrt(dh))), x in [0, 2^23), correctly rounded."""
    import mpmath
    key = (E_attn, dh)
    if key not in _TEXP:
        alpha = (2.0 ** E_attn) / math.sqrt(dh)
        x = torch.arange(0, 1 << 23, dtype=torch.float64, device=_dev())
        v = (1 << EXP_BITS) * torch.exp(-x * alpha)
        _TEXP[key] = round_exact(v, lambda i: (1 << EXP_BITS) * mpmath.exp(-i * mpmath.mpf(2) ** E_attn
                                                                          / mpmath.sqrt(dh)), 1e-4).to(torch.int64)
    return _TEXP[key]


_SILU = {}


def silu_values(c, eG, eA):
    """SiLU outputs; inputs in the table range go through a cached table of this same function."""
    if c.is_cuda and c.numel() > (1 << 20):
        key = (eG, eA)
        if key not in _SILU:
            _SILU[key] = _silu(torch.arange(-SILU_CMAX, SILU_CMAX + 1, device=c.device), eG, eA)
        cmax = int(c.abs().max())
        if cmax <= SILU_CMAX:
            return _SILU[key][(c + SILU_CMAX).to(torch.int64)]
    return _silu(c, eG, eA)


def silu_table(eG, eA, device):
    """The cached SiLU outputs of inputs -SILU_CMAX..SILU_CMAX."""
    key = (eG, eA)
    if key not in _SILU:
        _SILU[key] = _silu(torch.arange(-SILU_CMAX, SILU_CMAX + 1, device=device), eG, eA)
    return _SILU[key]


def _silu(c, eG, eA):
    """round(SiLU(c 2^eG) 2^-eA), correctly rounded, clipped to 16 bits."""
    import mpmath
    cf = c.reshape(-1)
    x = cf.double() * (2.0 ** eG)
    y = x / (1 + torch.exp(-x)) * (2.0 ** -eA)

    def exact(i):
        xe = int(cf[i]) * mpmath.mpf(2) ** eG
        return xe / (1 + mpmath.exp(-xe)) * mpmath.mpf(2) ** -eA
    return round_exact(y, exact, 1e-4).clamp(-(1 << 15), (1 << 15) - 1).to(torch.int64).reshape(c.shape)


_ROPE = {}


def rope_tables(cfg, S):
    """(cos, sin) int64 device tables (S, dh/2), cached per (theta, dh, S); callers must not modify them."""
    key = (cfg.theta, cfg.dh, S, _dev(), torch.cuda.current_device() if _dev() == "cuda" else -1)
    if key not in _ROPE:
        _ROPE.clear()
        _ROPE[key] = _rope_tables(cfg, S)
    return _ROPE[key]


def _rope_tables(cfg, S):
    """round(2^15 cos(t theta^(-2j/dh))) and the same for sin, correctly rounded."""
    import mpmath
    half = cfg.dh // 2
    inv = 1.0 / (cfg.theta ** (torch.arange(0, cfg.dh, 2, dtype=torch.float64) / cfg.dh))
    ang = torch.arange(S, dtype=torch.float64)[:, None] * inv[None, :]

    def exact(fn):
        return lambda i: (1 << 15) * fn((i // half) * mpmath.mpf(cfg.theta) ** (-mpmath.mpf(2 * (i % half)) / cfg.dh))
    cos = round_exact(torch.cos(ang) * (1 << 15), exact(mpmath.cos), 1e-4).to(torch.int64).to(_dev())
    sin = round_exact(torch.sin(ang) * (1 << 15), exact(mpmath.sin), 1e-4).to(torch.int64).to(_dev())
    return cos, sin


class QLayer:
    pass


class QModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.layers = []


def _qcols(W, per_tensor=False):
    m = W.abs().amax(dim=0) if not per_tensor else W.abs().max().expand(W.shape[1])
    m = torch.clamp(m, min=1e-12)
    e = torch.ceil(torch.log2(m / 127.0)).to(torch.int64)
    Wq = torch.round(W / torch.pow(2.0, e.to(W.dtype))).clamp(-127, 127).to(torch.int8)
    return Wq.contiguous(), e


def _rot(d, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    return (Q * torch.sign(torch.diagonal(R)).unsqueeze(0)).to(torch.float32).to(_dev())


def _e_fit(v):
    return int(math.ceil(math.log2(max(v, 1e-30) / 127.0)))


_WT = {}


def tn(W):
    """W with column-major storage: cuBLAS picks a much faster int8 kernel for A @ W^T layouts."""
    if W.stride(0) == 1:
        return W
    # keyed by identity; holding W keeps its storage from being reused by another tensor
    e = _WT.get(id(W))
    if e is None or e[0] is not W or e[1] != W._version:
        if len(_WT) >= 8:
            _WT.pop(next(iter(_WT)))
        e = _WT[id(W)] = (W, W._version, W.T.contiguous().T)
    return e[2]


def imm32(a, W):
    """int8 GEMM, int32 result (exact: |sum| < 2^31 for the model's inner dimensions)."""
    assert a.shape[0] >= 32 and a.shape[1] * 127 * 127 < (1 << 31)
    return torch._int_mm(a.to(torch.int8).contiguous(), tn(W))


_SRSK = {}


def shiftround_silu(acc, kr, kc, eG, eA):
    """(c, a) = (ShiftRound(acc, k), SiLU(c)) as int32 with k = kr[row] + kc[col] (>= 1);
    |c| <= SILU_CMAX is checked."""
    import cupy as cp
    key = (eG, eA)
    if key not in _SILU:
        _SILU[key] = _silu(torch.arange(-SILU_CMAX, SILU_CMAX + 1, device=acc.device), eG, eA)
    if "k" not in _SRSK:
        _SRSK["k"] = cp.ElementwiseKernel(
            "int32 x, int64 kr, int64 kc, raw int32 tab", "int32 c, int32 a, raw int32 err",
            f"long long k = kr + kc; if (k < 1 || k > 40) {{ atomicAdd(&err[0], 1); k = 1; }}"
            f"long long cc = ((long long)x + (1ll << (k - 1))) >> k; c = (int)cc;"
            f"if (cc < -{SILU_CMAX}ll || cc > {SILU_CMAX}ll) {{ atomicAdd(&err[0], 1); a = 0; }}"
            f"else a = tab[cc + {SILU_CMAX}];", "zk_srsilu")
    if key not in _SRSK:
        _SRSK[key] = _SILU[key].to(torch.int32).contiguous()
    assert acc.dtype == torch.int32 and acc.is_contiguous()
    kr = kr.reshape(-1, 1).to(torch.int64).expand(acc.shape)
    kc = kc.reshape(1, -1).to(torch.int64).expand(acc.shape)
    c = torch.empty_like(acc)
    a = torch.empty_like(acc)
    err = torch.zeros(1, dtype=torch.int32, device=acc.device)
    _SRSK["k"](cp.from_dlpack(acc), cp.from_dlpack(kr), cp.from_dlpack(kc), cp.from_dlpack(_SRSK[key]),
               cp.from_dlpack(c), cp.from_dlpack(a), cp.from_dlpack(err))
    check(err == 0, "SiLU input out of table")
    return c, a


_NRK = []


def nr_round(hi, lo, km):
    """ShiftRound(32 hi + lo, km) per row (km >= 1), int64."""
    import cupy as cp
    if not _NRK:
        _NRK.append(cp.ElementwiseKernel(
            "int32 h, int32 l, int64 k", "int64 y",
            "y = ((long long)h * 32 + (long long)l + (1ll << (k - 1))) >> k;", "zk_nrround"))
    kk = km.reshape(-1, 1).to(torch.int64).expand(hi.shape)
    y = torch.empty(hi.shape, dtype=torch.int64, device=hi.device)
    _NRK[0](cp.from_dlpack(hi.contiguous()), cp.from_dlpack(lo.contiguous()), cp.from_dlpack(kk), cp.from_dlpack(y))
    return y


def imm(a, W):
    """int8 GEMM, exact int64 result."""
    m = a.shape[0]
    Wt = tn(W)
    if m < 32:
        pad = torch.zeros((32, a.shape[1]), dtype=torch.int8, device=a.device)
        pad[:m] = a.to(torch.int8)
        return torch._int_mm(pad, Wt)[:m].to(torch.int64)
    return torch._int_mm(a.to(torch.int8).contiguous(), Wt).to(torch.int64)


def imm_wide(a, W):
    """Exact a W for |a| < 2^11 (int8 GEMMs on a = 16 hi + lo, lo in [0, 16))."""
    hi = fdiv(a, 16)
    return 16 * imm(hi, W) + imm(a - 16 * hi, W)


def _qkv_float(get, li, Qr, sq):
    """Layer li's fused QKV weight (d, nq + 2 nk) in floating point, rotated and with the norm gain."""
    p = f"model.layers.{li}."
    g1 = get(p + "input_layernorm.weight").float()
    W = torch.cat([get(p + f"self_attn.{n}_proj.weight").float().T for n in "qkv"], dim=1)
    return sq * (Qr.T @ (g1[:, None] * W))


def e_hmax(d):
    """The largest row exponent E_h of NormScale for rows of width d (Sigma <= d 127^2 + 1)."""
    return ((d * 127 * 127 + 1).bit_length() - 15) // 2


def fit_qkv_exponents(M, get=None, seed=1234):
    """After calibration: raise the weight exponent of every QKV column n whose ShiftQ8S bound
    k_max(b_n) - kappa_n is below e_hmax(d), by the missing amount, so that the per-row bound K_qs
    never binds and F_O is defined on every input.  The affected columns are requantized from the
    checkpoint (get) at the new exponent, or, without it, from their int8 values (round half to even).
    Returns the number of columns changed."""
    cfg, E = M.cfg, e_hmax(M.cfg.d)
    lim = _PR - (1 << 30) - cfg.d * 127 * 127
    Qr = _rot(cfg.d, seed) if get is not None else None
    sq = math.sqrt(cfg.d) / 127.0
    changed = 0
    for li, Ly in enumerate(M.layers):
        b = Ly.bias.detach().abs().to(torch.int64).cpu().tolist()
        kmax = torch.tensor([max(k for k in range(1, QKV_KMAX + 1) if (bi + 128) << k < lim) for bi in b])
        kc = Ly.kc.detach().cpu().to(torch.int64)
        delta = (kc - (kmax - E)).clamp(min=0)
        cols = torch.nonzero(delta).flatten()
        if cols.numel() == 0:
            continue
        dv, cd = Ly.Wqkv.device, cols.to(Ly.Wqkv.device)
        e_new = Ly.eWqkv.detach().cpu().to(torch.int64)[cols] + delta[cols]
        if get is not None:
            Wf = _qkv_float(get, li, Qr, sq)[:, cols.to(_dev())].to(torch.float64)
            Wq = torch.round(Wf / torch.pow(2.0, e_new.to(Wf.device, torch.float64))).clamp(-127, 127)
        else:
            Wq = torch.round(Ly.Wqkv[:, cd].to(torch.float64) / torch.pow(2.0, delta[cols].to(dv, torch.float64)))
        Ly.Wqkv[:, cd] = Wq.to(dv, torch.int8)
        Ly.eWqkv[cols.to(Ly.eWqkv.device)] = e_new.to(Ly.eWqkv.device, Ly.eWqkv.dtype)
        Ly.kc = Ly.eout.to(Ly.eWqkv.device) - Ly.eWqkv
        changed += cols.numel()
    drop_caps(M)
    return changed


def _qk_head_exponents(Ly, W, cfg):
    """QK-norm normalizes a head row of the accumulator, so the columns of a Q or K head must share
    one weight exponent (the largest of the head's per-column exponents)."""
    n, dh = (cfg.nh + cfg.nkv) * cfg.dh, cfg.dh
    eh = Ly.eWqkv[:n].view(-1, dh).amax(dim=1, keepdim=True).expand(-1, dh).reshape(-1)
    Wq = torch.round(W[:, :n] / torch.pow(2.0, eh.to(W.dtype)).unsqueeze(0)).clamp(-127, 127)
    Ly.Wqkv[:, :n] = Wq.to(torch.int8)
    Ly.eWqkv[:n] = eh


def qk_gains(Ly, cfg, gq, gk):
    """QK-norm gains as integers: the head row's NormScale output q' is 127 2^E q / |q|, so the normalized
    value times the gain g_c is q' g_c sqrt(dh) / 127 2^-E; G_c = round(g_c sqrt(dh) / 127 / 2^eG), one
    exponent eG per gain vector.  Ly.Gqk, Ly.eGqk: per column of the Q and K heads."""
    nq, nk, dh = cfg.nh * cfg.dh, cfg.nkv * cfg.dh, cfg.dh
    G, e = [], []
    for g, n in ((gq, nq // dh), (gk, nk // dh)):
        v = g.double() * math.sqrt(dh) / 127.0
        eg = _e_fit(v.abs().max().item())
        Gi = torch.round(v / 2.0 ** eg).clamp(-127, 127).to(torch.int64)
        G.append(Gi.repeat(n))
        e.append(torch.full((n * dh,), eg, dtype=torch.int64))
    Ly.Gqk = torch.cat(G).to(_dev())
    Ly.eGqk = torch.cat(e).to(_dev())


def qkv_out(cfg, Ly, acc, Eh, real=False):
    """Q, K, V rows (8-bit) from the fused QKV accumulator acc (R, nq + 2 nk) of rows with NormScale
    exponents Eh (R,).  Without QK-norm: ShiftQ8S by kc + Eh with the bias.  With QK-norm, each Q and K
    head row goes through ShiftQ8 (8-bit, row exponent), NormScale (exponent E2) and ShiftQ8S of q' G by
    kn + E2 (no bias); V as before.  real: also the float value of every column before the last
    rounding (calibration)."""
    nq, nk, dh = cfg.nh * cfg.dh, cfg.nkv * cfg.dh, cfg.dh
    if not cfg.qk_norm:
        if real and "kc" not in Ly.__dict__:      # calibration, before the shifts exist
            out = None
        else:
            out, _ = shiftq8s(acc, Ly.kc.unsqueeze(0) + Eh.unsqueeze(-1), Ly.bias.unsqueeze(0))
        if real:
            r = acc.double() * torch.pow(2.0, Ly.eWqkv.double()).unsqueeze(0) / torch.pow(2.0, Eh.double()).unsqueeze(-1)
            return out, r + Ly.bqkv
        return out
    R = acc.shape[0]
    a = acc[:, :nq + nk].reshape(-1, dh)
    q8, _, _ = shiftq8(a)                     # e >= 2: the proof's quartered minimality relation
    qp, E2, _ = normscale(q8)
    prod = qp.view(R, nq + nk) * Ly.Gqk.unsqueeze(0)
    E2c = E2.view(R, -1).repeat_interleave(dh, dim=1)
    qk = None
    if "kn" in Ly.__dict__:                       # not yet during the first calibration step
        k = Ly.kn.unsqueeze(0) + E2c
        if _CALIB[0]:
            k = k.clamp(1, QKV_KMAX)
        else:                                     # a head row whose kn + E2 is below 1 is shifted by 1
            check(k.max() <= QKV_KMAX, "F_O: QK-norm shift above QKV_KMAX")
            check(k.min() >= -31, "F_O: QK-norm shift below -31 (the proof's clamp range)")
            k = k.clamp(min=1)
        qk, _ = shiftq8s(prod, k, torch.zeros_like(Ly.kn).unsqueeze(0))
    kv = Ly.kc[nq + nk:].unsqueeze(0) + Eh.unsqueeze(-1)
    v, _ = shiftq8s(acc[:, nq + nk:], kv, Ly.bias[nq + nk:].unsqueeze(0))
    out = torch.cat([qk, v], dim=1) if qk is not None else None
    if real:
        rqk = prod.double() * torch.pow(2.0, (Ly.eGqk.unsqueeze(0) - E2c).double())
        rv = acc[:, nq + nk:].double() * torch.pow(2.0, Ly.eWqkv[nq + nk:].double()).unsqueeze(0) \
            / torch.pow(2.0, Eh.double()).unsqueeze(-1) + Ly.bqkv[nq + nk:]
        return out, torch.cat([rqk, rv], dim=1)
    return out


def _quantize_moe(Ly, cfg, get, p, Qr, g2, sq):
    """Router (one exponent) and the experts: gate per column, up and down one exponent per expert."""
    m = p + "mlp."
    Ly.Wr, e_ = _qcols(sq * (Qr.T @ (g2[:, None] * get(m + "gate.weight").float().T)), per_tensor=True)
    Ly.eWr = int(e_[0])
    Wg, eg, Wu, eu, Wd, ed = [], [], [], [], [], []
    for e in range(cfg.moe["E"]):
        x = m + f"experts.{e}."
        a, ea = _qcols(sq * (Qr.T @ (g2[:, None] * get(x + "gate_proj.weight").float().T)), per_tensor=True)
        b, eb = _qcols(sq * (Qr.T @ (g2[:, None] * get(x + "up_proj.weight").float().T)), per_tensor=True)
        c, ec = _qcols(get(x + "down_proj.weight").float().T @ Qr, per_tensor=True)
        Wg.append(a); eg.append(int(ea[0])); Wu.append(b); eu.append(int(eb[0])); Wd.append(c); ed.append(int(ec[0]))
    cm = lambda ws: torch.stack([w.T.contiguous() for w in ws]).transpose(1, 2)   # column-major per expert
    Ly.Wge, Ly.eWge = cm(Wg), torch.tensor(eg, device=_dev())          # (E, d, de), (E,)
    Ly.Wue, Ly.eWue = cm(Wu), torch.tensor(eu, device=_dev())          # (E, d, de), (E,)
    Ly.Wde, Ly.eWde = cm(Wd), torch.tensor(ed, device=_dev())          # (E, de, d), (E,)


def fit_gate_exponents(M, get, seed=1234):
    """After calibration: raise the gate weight exponent of every column (dense) or expert (MoE) whose
    shift eG - eW + E_h could exceed GATE_KMAX for some row (E_h <= e_hmax(d)), and requantize it from
    the checkpoint, so that the gate shift is in range on every input (MoE: most experts are never routed
    during calibration; dense: a first-token envelope can raise eG).  Returns the number changed."""
    cfg = M.cfg
    lo = M.eG + e_hmax(cfg.d) - GATE_KMAX                 # the least admissible gate exponent
    Qr, sq = _rot(cfg.d, seed), math.sqrt(cfg.d) / 127.0
    changed = 0
    for li, Ly in enumerate(M.layers):
        bad = ((Ly.eWge if cfg.moe else Ly.eWg) < lo).nonzero().flatten()
        if bad.numel() == 0:
            continue
        p = f"model.layers.{li}."
        g2 = get(p + "post_attention_layernorm.weight").float()
        if not cfg.moe:
            Wf = sq * (Qr.T @ (g2[:, None] * get(p + "mlp.gate_proj.weight").float().T))[:, bad]
            Ly.Wg[:, bad] = torch.round(Wf.double() / 2.0 ** lo).clamp(-127, 127).to(Ly.Wg.dtype)
            Ly.eWg[bad] = lo
            Ly.kcg = M.eG - Ly.eWg
            changed += bad.numel()
            continue
        for e in bad.tolist():
            Wf = sq * (Qr.T @ (g2[:, None] * get(p + f"mlp.experts.{e}.gate_proj.weight").float().T))
            Ly.Wge[e][:, :] = torch.round(Wf.double() / 2.0 ** lo).clamp(-127, 127).to(torch.int8)
            Ly.eWge[e] = lo
            changed += 1
        Ly.kcg = M.eG - Ly.eWge
    drop_caps(M)
    return changed


def set_f_h(M, get, f_h, seed=1234):
    """The residual's fractional bits f_h and the embedding rows in those units."""
    M.f_h = f_h
    emb = get("model.embed_tokens.weight").float() @ _rot(M.cfg.d, seed)
    M.emb = torch.round(emb * (1 << f_h)).clamp(-(1 << A_BITS), (1 << A_BITS) - 1).to(torch.int32)


def quantize_weights(cfg, get, f_h=10, seed=1234):
    """Rotation, norm-gain fusion and int8 weights (no activation statistics)."""
    d = cfg.d
    Qr = _rot(d, seed)
    M = QModel(cfg)
    set_f_h(M, get, f_h, seed)
    sq = math.sqrt(d) / 127.0
    for li in range(cfg.L):
        p = f"model.layers.{li}."
        Ly = QLayer()
        W = _qkv_float(get, li, Qr, sq)
        Ly.Wqkv, Ly.eWqkv = _qcols(W)
        if cfg.qk_norm:
            _qk_head_exponents(Ly, W, cfg)
        if cfg.qkv_bias:
            Ly.bqkv = torch.cat([get(p + f"self_attn.{n}_proj.bias").float() for n in "qkv"]).double()
        else:
            Ly.bqkv = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
        if cfg.qk_norm:
            qk_gains(Ly, cfg, get(p + "self_attn.q_norm.weight").float(), get(p + "self_attn.k_norm.weight").float())
        Ly.Wo, e_ = _qcols(get(p + "self_attn.o_proj.weight").float().T @ Qr, per_tensor=True)
        Ly.eWo = int(e_[0])
        g2 = get(p + "post_attention_layernorm.weight").float()
        if cfg.moe:
            _quantize_moe(Ly, cfg, get, p, Qr, g2, sq)
            M.layers.append(Ly)
            del W
            torch.cuda.empty_cache()
            continue
        Ly.Wg, Ly.eWg = _qcols(sq * (Qr.T @ (g2[:, None] * get(p + "mlp.gate_proj.weight").float().T)))
        Wu, eu = _qcols(sq * (Qr.T @ (g2[:, None] * get(p + "mlp.up_proj.weight").float().T)), per_tensor=True)
        Ly.Wu, Ly.eWu = Wu, int(eu[0])
        Ly.Wd, e_ = _qcols(get(p + "mlp.down_proj.weight").float().T @ Qr, per_tensor=True)
        Ly.eWd = int(e_[0])
        M.layers.append(Ly)
        del W
        torch.cuda.empty_cache()
    gF = get("model.norm.weight").float()
    M.Wlm, e_ = _qcols(sq * (Qr.T @ (gF[:, None] * get("lm_head.weight").float().T)), per_tensor=True)
    M.eWlm = int(e_[0])
    M.Kz = EZ - M.eWlm
    return M


def qk_split(realb, E, eQ, ekmax, g, grp, dh, nq):
    """K exponent of KV group g under the score exponent E = e_Q + e_K.  E is one constant for the model, so
    a group whose Q and K need less leaves slack; giving it all to K (ekmax = E - max e_Q, the coarsest K)
    leaves K a few bits on QK-norm models.  Pick e_K between the fit of 1.25 max|K| and ekmax that minimizes
    the scores' rounding noise sum_i (4^e_Q |K|^2 + 4^e_K |Q_i|^2) over the group's query heads."""
    rms2 = lambda a: realb[:, a:a + dh].pow(2).mean().item()
    k2 = rms2(nq + g * dh)
    q2 = sum(rms2(i * dh) for i in range(g * grp, (g + 1) * grp))
    lo = min(ekmax, _e_fit(1.25 * realb[:, nq + g * dh:nq + (g + 1) * dh].abs().max().item()))
    return min(range(lo, ekmax + 1), key=lambda k: 4.0 ** (E - k) * grp * k2 + 4.0 ** k * q2)


def calibrate(M, tokens, passes=2):
    """Sequential calibration of the static exponents on tokens (B, S)."""
    cfg = M.cfg
    nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
    nq, nk = nh * dh, nkv * dh
    grp = nh // nkv
    M.E_attn = None
    _CALIB[0] = True
    for ps in range(passes):
        need_attn, gate_max, silu_max = [], 0.0, 0.0
        B, S = tokens.shape
        cos, sin = rope_tables(cfg, S)
        h = M.emb[tokens.reshape(-1).to(_dev())].to(torch.int64)
        for li, Ly in enumerate(M.layers):
            q, e, _ = requant8(h)
            qp, Eh, _ = normscale(q)
            acc = imm(qp, Ly.Wqkv)
            if cfg.qk_norm:                       # V's shift needs kc before qkv_out; Q, K need none
                Ly.kc, Ly.bias = torch.ones_like(Ly.eWqkv), torch.zeros_like(Ly.eWqkv)
            _, realb = qkv_out(cfg, Ly, acc, Eh, real=True)
            eQ = [_e_fit(realb[:, i * dh:(i + 1) * dh].abs().max().item()) for i in range(nh)]
            eK = [_e_fit(realb[:, nq + i * dh:nq + (i + 1) * dh].abs().max().item()) for i in range(nkv)]
            eV = [_e_fit(realb[:, nq + nk + i * dh:nq + nk + (i + 1) * dh].abs().max().item()) for i in range(nkv)]
            need = [max(eQ[g * grp:(g + 1) * grp]) + eK[g] for g in range(nkv)]
            need_attn += need
            Ly.E = max(need) if M.E_attn is None else M.E_attn
            eKg = [max(eK[g], Ly.E - max(eQ[g * grp:(g + 1) * grp])) for g in range(nkv)]
            if cfg.qk_norm:
                eKg = [qk_split(realb, Ly.E, eQ, eKg[g], g, grp, dh, nq) for g in range(nkv)]
            eQh = [Ly.E - eKg[i // grp] for i in range(nh)]
            Ly.eQ, Ly.eK, Ly.eV = eQh, eKg, eV
            Ly.eVmax = max(eV)
            Ly.vshift = torch.tensor([Ly.eVmax - eV[i // grp] for i in range(nh)], device=_dev())
            eout = torch.tensor([eQh[i // dh] for i in range(nq)] + [eKg[i // dh] for i in range(nk)] +
                                [eV[i // dh] for i in range(nk)], dtype=torch.int64, device=_dev())
            Ly.eout = eout
            Ly.kc = eout - Ly.eWqkv
            if cfg.qk_norm:
                # Q, K: the QK-norm shift kn = e_out - eG (plus the head row's E2); kc unused there (1)
                Ly.kn = eout[:nq + nk] - Ly.eGqk
                Ly.kc[:nq + nk] = 1
            assert int(Ly.kc.min()) >= 1
            Ly.bias = torch.round(Ly.bqkv / torch.pow(2.0, eout.double())).to(torch.int64)
            Ly.Kco = -(Ly.eVmax - 7 + Ly.eWo + M.f_h)
            if M.E_attn is None and li == 0:
                M.eG, M.eA = -8, -10
            h, gm, sm = layer_forward(M, Ly, h, B, S, cos, sin, calib=True)
            gate_max, silu_max = max(gate_max, gm), max(silu_max, sm)
        M.E_attn = max(need_attn)
        for Ly in M.layers:
            Ly.E = M.E_attn
        M.eG = int(math.ceil(math.log2(gate_max * 1.25 / SILU_CMAX)))
        M.eA = int(math.ceil(math.log2(silu_max * 1.25 / 32767)))
        for Ly in M.layers:
            Ly.kcg = M.eG - (Ly.eWge if M.cfg.moe else Ly.eWg)
    _CALIB[0] = False
    drop_caps(M)                                 # the constants changed: recompute the bounds
    return M


STATS = {}
ENV = {}      # under _CALIB: the largest real gate input and SiLU output, and the largest |GShiftRound output| / 2^A_BITS


def _env(name, v):
    ENV[name] = max(ENV.get(name, 0.0), v)


def _stat(name, t):
    lo, hi = int(t.min()), int(t.max())
    a, b = STATS.get(name, (lo, hi))
    STATS[name] = (min(a, lo), max(b, hi))


def layer_to(Ly, device):
    """Copy of a layer with its tensors on device (pinned when on the host)."""
    out = QLayer()
    for k, v in vars(Ly).items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True) if device != "cpu" else v.cpu().pin_memory()
        setattr(out, k, v)
    return out


def load_model(path, layers_on="cuda"):
    st = torch.load(path, weights_only=False)
    M = QModel(Cfg(**st["cfg"]))
    for k in ("f_h", "E_attn", "eG", "eA"):
        setattr(M, k, st[k])
    for k in ("emb", "Wlm", "eWlm", "Kz"):
        setattr(M, k, st[k].to(_dev()) if torch.is_tensor(st[k]) else st[k])
    for d in st["layers"]:
        Ly = QLayer()
        for k, v in d.items():
            if torch.is_tensor(v):
                v = v.to(_dev()) if layers_on == "cuda" else (v.pin_memory() if _dev() == "cuda" else v)
            setattr(Ly, k, v)
        M.layers.append(Ly)
    return M


def layer_forward(M, Ly, h, B, S, cos, sin, calib=False, cache=None, pos0=0, wit=None):
    """One layer on rows h (B*S, d) at positions pos0.. ; cache: (rK, V) per layer
    of previous positions (decode).  Returns new h (and calibration stats)."""
    cfg = M.cfg
    nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
    nq, nk = nh * dh, nkv * dh
    q, e, wq = requant8(h)
    qp, Eh, wn = normscale(q)
    acc = imm(qp, Ly.Wqkv)
    k = Ly.kc.unsqueeze(0) + Eh.unsqueeze(-1)
    if not _CALIB[0]:
        cap = Ly.__dict__.get("kqs")
        check(Eh.max() <= (qkv_kcap(cfg, Ly) if cap is None else cap), "F_O: row exponent beyond the ShiftQ8S bound")
    _stat("k_qkv", k)
    _stat("e_h", e)
    qkv = qkv_out(cfg, Ly, acc, Eh)
    half = dh // 2
    Qh = qkv[:, :nq].reshape(B, S, nh, dh)
    Kh = qkv[:, nq:nq + nk].reshape(B, S, nkv, dh)
    Vh = qkv[:, nq + nk:].reshape(B, S, nkv, dh)
    c_ = cos[pos0:pos0 + S].view(1, S, 1, half)
    s_ = sin[pos0:pos0 + S].view(1, S, 1, half)
    rq1, rq2 = rope8(Qh[..., :half], Qh[..., half:], c_, s_)
    rk1, rk2 = rope8(Kh[..., :half], Kh[..., half:], c_, s_)
    rQ = torch.cat([rq1, rq2], dim=-1)
    rK = torch.cat([rk1, rk2], dim=-1)
    if cache is not None:
        if cache.get("rK") is not None:
            rK_all = torch.cat([cache["rK"], rK], dim=1)
            V_all = torch.cat([cache["V"], Vh], dim=1)
        else:
            rK_all, V_all = rK, Vh
        cache["rK"], cache["V"] = rK_all, V_all
    else:
        rK_all, V_all = rK, Vh
    texp = exp_table(Ly.E, dh)
    o = torch.empty((B, S, nh, dh), dtype=torch.int64, device=_dev())
    grp = nh // nkv
    for b in range(B):
        for hh in range(nh):
            g = hh // grp
            o[b, :, hh] = attn_rows(rQ[b, :, hh], rK_all[b, :, g], V_all[b, :, g], texp,
                                    int(Ly.vshift[hh]), pos0)
    o = o.reshape(B * S, nh * dh)
    qo, eo, _ = requant8(o)
    acc = imm(qo, Ly.Wo)
    k = (Ly.Kco - eo).unsqueeze(-1)
    _stat("k_o", k)
    _stat("e_o", eo)
    h, _ = satadd(h, gshiftround(acc, k))
    # MLP
    q, e, _ = requant8(h)
    qp, Eh, _ = normscale(q)
    y, gm, sm = mlp_out(M, Ly, qp, Eh, calib)
    h, _ = satadd(h, y)
    _stat("h", h)
    if calib:
        return h, gm, sm
    return h


def _gate_silu(M, accg, kg, calib):
    """SiLU input c = ShiftRound(accg, kg) and a = tau_SiLU(c); in calibration also the largest |gate|
    and |SiLU| seen (real values).  Under _CALIB (calibration, first_token_envelope) c is clamped instead
    of checked."""
    calib = calib or _CALIB[0]
    _stat("k_g", kg)
    if not calib:
        check((kg >= 1) & (kg <= SILU_KMAX), "F_O: gate shift outside [1, SILU_KMAX]")
    c = shiftround(accg, kg)
    gm = sm = 0.0
    if calib:
        xr = c.double() * 2.0 ** M.eG
        gm = xr.abs().max().item() if xr.numel() else 0.0
        sm = (xr / (1 + torch.exp(-xr))).abs().max().item() if xr.numel() else 0.0
        _env("gate", gm)
        _env("silu", sm)
        c = c.clamp(-SILU_CMAX, SILU_CMAX)
    check(c.numel() == 0 or int(c.abs().max()) <= SILU_CMAX, "F_O: SiLU input beyond SILU_CMAX")
    return silu_values(c, M.eG, M.eA), gm, sm


def mlp_out(M, Ly, qp, Eh, calib=False):
    """The MLP's contribution to the residual (in A) from the normalized rows qp with exponents Eh;
    also the calibration statistics (largest gate input, largest SiLU output)."""
    if M.cfg.moe:
        return moe_out(M, Ly, qp, Eh, calib)
    kg = (M.eG - Ly.eWg).unsqueeze(0) + Eh.clamp(min=gate_kmin(M, Ly)).unsqueeze(-1)
    a, gm, sm = _gate_silu(M, imm(qp, Ly.Wg), kg, calib)
    accu = imm(qp, Ly.Wu)
    qu, eup, _ = shiftq8(accu)
    m = a * qu
    if M.cfg.down_b == 127:
        qm, em, _ = requant8(m)
        accd = imm(qm, Ly.Wd)
    else:
        qm, em, _ = _rowq_b(m, M.cfg.down_b, 17)
        accd = imm_wide(qm, Ly.Wd)
    Kcd = -(Ly.eWu + M.eA + Ly.eWd + M.f_h)
    kd = (Kcd - (em + eup - Eh)).unsqueeze(-1)
    _stat("k_d", kd)
    _stat("e_m", em)
    _stat("e_up", eup)
    return gshiftround(accd, kd), gm, sm


# MoE (Qwen3-MoE): router logits z = GShiftRound(qp Wr, Eh - eWr - R_FRAC) in units 2^-R_FRAC; the top
# k logits (ties to the smaller index); p_s = tau_r(z_max - z_s) (12-bit); w_s = Div(p_s, sum p, 2^W_FRAC),
# Qwen3's softmax over the selected experts.  Each selected expert's path is the dense MLP with its own
# weights; slot s's path scale es = em + eup + eWu_e + eWd_e and its weight are folded into one row:
# omega_s = ShiftRound(w_s, Et - es) (Et = max_s es), r = qm omega, requantized over the whole token row
# to MOE_B; one down accumulator over the k slots and one GShiftRound into A.
# W_FRAC = 14 and the router shift <= R_KMAX = 14 keep the proof's Div and SatShift relations below p: at
# k = 15 a saturated row's relation reaches |acc| + 2^30 + 2^(k+15) > p (tests/test_router_wrap.py)
R_FRAC, R_TBITS, R_DOM, W_FRAC, MOE_B, R_SAT, R_KMAX = 8, 12, 1 << 16, 14, 1023, (1 << 15) - 1, 14
assert 2 * R_SAT < R_DOM                        # z_max - z_s < 2^16 for saturated logits


def satshift(acc, k, B):
    """round(acc / 2^k) (ties up), saturated to [-B, B]."""
    s = pow2(k)
    return fdiv(acc + s // 2, s).clamp(-B, B)
_TR = {}


def router_table():
    """tau_r(x) = round(2^12 exp(-x / 2^8)) for x in [0, 2^16), correctly rounded."""
    if "t" not in _TR:
        x = torch.arange(R_DOM, dtype=torch.float64)
        import mpmath

        def exact(i):
            return mpmath.mpf(1 << R_TBITS) * mpmath.exp(-mpmath.mpf(i) / (1 << R_FRAC))
        _TR["t"] = round_exact((1 << R_TBITS) * torch.exp(-x / (1 << R_FRAC)), exact, 1e-6).to(torch.int64).to(_dev())
    return _TR["t"]


def div_round(a, s, Fm):
    """Div(a, s, F) = floor((2 F a + s) / (2 s)), s >= 1: round(F a / s), ties up."""
    check(s >= 1, "div: divisor < 1")
    return fdiv(2 * Fm * a + s, 2 * s)


def moe_route(M, Ly, qp, Eh):
    """-> expert indices (T, k), weights (T, k) in units 2^-W_FRAC, and the router logits z."""
    k = M.cfg.moe["k"]
    E = Ly.Wr.shape[1]
    kr = (Eh - Ly.eWr - R_FRAC).unsqueeze(-1)
    if _CALIB[0]:
        kr = kr.clamp(1, R_KMAX)
    check(kr.max() <= R_KMAX, "F_O: router shift out of range")
    kr = kr.clamp(min=1)                      # a row of small norm (E_h near -7) is shifted by 1
    z = satshift(imm(qp, Ly.Wr), kr, R_SAT)
    key = z * E + (E - 1 - torch.arange(E, device=z.device)).unsqueeze(0)     # ties to the smaller index
    idx = torch.topk(key, k, dim=-1).indices
    zs = z.gather(1, idx)
    x = zs.max(dim=-1, keepdim=True).values - zs
    pr = router_table()[x]
    w = div_round(pr, pr.sum(dim=-1, keepdim=True), 1 << W_FRAC)
    return idx, w, z


def moe_out(M, Ly, qp, Eh, calib=False):
    cfg = M.cfg
    T, de = qp.shape[0], cfg.moe["de"]
    idx, w, _ = moe_route(M, Ly, qp, Eh)
    k = idx.shape[1]
    qm_all = torch.zeros((T, k, de), dtype=torch.int64, device=qp.device)
    es = torch.zeros((T, k), dtype=torch.int64, device=qp.device)
    gm = sm = 0.0
    for e in torch.unique(idx).tolist():
        rows, slot = torch.where(idx == e)
        q, E_ = qp[rows], Eh[rows]
        kg = ((M.eG - Ly.eWge[e]) + E_).clamp(min=1).unsqueeze(-1)
        a, g1, s1 = _gate_silu(M, imm(q, Ly.Wge[e]), kg, calib)
        gm, sm = max(gm, g1), max(sm, s1)
        qu, eup, _ = shiftq8(imm(q, Ly.Wue[e]))
        qm, em, _ = requant8(a * qu)
        qm_all[rows, slot] = qm
        es[rows, slot] = em + eup + Ly.eWue[e] + Ly.eWde[e]
    Et = es.max(dim=-1, keepdim=True).values
    so = Et - es
    check(so.max() <= 24, "F_O: MoE slot scale shift above 24")
    om = shiftround(w, so)                                      # omega_s in units 2^-W_FRAC 2^(es - Et)
    r = qm_all * om.unsqueeze(-1)
    q2, e2, _ = _rowq_b(r.reshape(T, k * de), MOE_B)
    q2 = q2.reshape(T, k, de)
    accd = torch.zeros((T, cfg.d), dtype=torch.int64, device=qp.device)
    for e in torch.unique(idx).tolist():
        rows, slot = torch.where(idx == e)
        accd.index_add_(0, rows, imm_wide(q2[rows, slot], Ly.Wde[e]))
    kd = (-(e2 + Et.squeeze(-1) + M.eA - Eh - W_FRAC + M.f_h)).clamp(max=24)
    _stat("k_moe", kd)
    return gshiftround(accd, kd.unsqueeze(-1)), gm, sm


MOE_EMAX = 17


def _rowq_b(a, B, emax=MOE_EMAX):
    """Least e in [0, emax] with every a in [-(B + 1/2) 2^e, (B + 1/2) 2^e) over the last dim;
    q = round(a / 2^e)."""
    m = torch.maximum(2 * a.max(dim=-1).values + 1, -2 * a.min(dim=-1).values)
    e = torch.zeros_like(m)
    for c in range(0, emax):
        e += ((2 * B + 1) * (1 << c) < m).to(torch.int64)
    s = pow2(e).unsqueeze(-1)
    q = fdiv(2 * a + s, 2 * s)
    check(q.abs().max() <= B, "requant: |q| > B")
    return q, e, None


def attn_rows(rQ, rK, V, texp, vshift, pos0, chunk=2048):
    """Causal attention for query rows at positions pos0 + i over keys [0, pos)."""
    S = rQ.shape[0]
    dh = rQ.shape[1]
    out = torch.empty((S, dh), dtype=torch.int64, device=_dev())
    Kf = rK.float()
    for r0 in range(0, S, chunk):
        r1 = min(S, r0 + chunk)
        p1 = pos0 + r1
        acc = (rQ[r0:r1].float() @ Kf[:p1].T).to(torch.int64)
        rows = torch.arange(pos0 + r0, pos0 + r1, device=_dev()).unsqueeze(1)
        cols = torch.arange(p1, device=_dev()).unsqueeze(0)
        mask = cols <= rows
        Mx = torch.where(mask, acc, torch.full_like(acc, -(1 << 40))).max(dim=1).values
        x = (Mx.unsqueeze(1) - acc).clamp(0, (1 << 23) - 1)
        Pm = torch.where(mask, texp[x], torch.zeros_like(x))
        nd = norm_den(Pm.sum(dim=1))
        km = nd["km"].unsqueeze(-1)
        Ns = torch.zeros((r1 - r0, dh), dtype=torch.int64, device=_dev())
        for kb in range(0, (p1 + (1 << KEYBLK) - 1) >> KEYBLK):
            j0, j1 = kb << KEYBLK, min(p1, (kb + 1) << KEYBLK)
            Ns += shiftround(pv_exact(Pm[:, j0:j1], V[j0:j1]), km)
        o, _ = norm_div(Ns, nd, vshift)
        out[r0:r1] = o
    return out


def pv_exact(P, V):
    """Exact P (<= 2^10) @ V (int8) via int8 GEMMs: P = 32 P_hi + P_lo."""
    m, kk = P.shape
    mp, kp = max(32, (m + 31) // 32 * 32), (kk + 31) // 32 * 32
    Ph = torch.zeros((mp, kp), dtype=torch.int8, device=_dev())
    Pl = torch.zeros((mp, kp), dtype=torch.int8, device=_dev())
    Ph[:m, :kk] = (P >> 5).to(torch.int8)
    Pl[:m, :kk] = (P & 31).to(torch.int8)
    Vt = torch.zeros((kp, V.shape[1]), dtype=torch.int8, device=_dev())
    Vt[:kk] = V.to(torch.int8)
    Vt = Vt.T.contiguous().T
    return (32 * torch._int_mm(Ph, Vt).to(torch.int64) + torch._int_mm(Pl, Vt).to(torch.int64))[:m]


def final_logits(M, h):
    q, e, _ = requant8(h)
    qp, Eh, _ = normscale(q)
    acc = imm(qp, M.Wlm)
    k = (M.Kz + Eh).unsqueeze(-1)
    _stat("k_z", k)
    return gshiftround(acc, k)


# ---------------------------------------------------------------- sampling (spec, Sampling)
# Temperature index theta -> temperature; index 0 is greedy decoding.
THETA = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
SAMP_BITS = 12          # p = tau_exp,theta(z - max) in [0, 2^12]
SAMP_DLOG = 24          # table domain: max - z in [0, 2^24)
U_BITS = 30             # u_t = PRF(r, t) in [0, 2^30)
_TSAMP = {}


def samp_table(theta):
    """tau_exp,theta(d) = round(2^12 exp(-d / (2^8 T))) for d = max - z in [0, 2^24) (logits are in
    units of 2^-8): int64 (2^24,) on the GPU."""
    import mpmath
    assert 0 < theta < len(THETA)
    if theta not in _TSAMP:
        d = torch.arange(0, 1 << SAMP_DLOG, dtype=torch.float64, device=_dev())
        v = (1 << SAMP_BITS) * torch.exp(-d / (256.0 * THETA[theta]))
        T = mpmath.mpf(THETA[theta])
        _TSAMP[theta] = round_exact(v, lambda i: (1 << SAMP_BITS) * mpmath.exp(-i / (256 * T)), 1e-4).to(torch.int64)
    return _TSAMP[theta]


def sample(z, u, theta):
    """Inverse-CDF sample of logits z (R, V) with u (R,) in [0, 2^30): the y with
    sum_{j<y} p_j <= w < sum_{j<=y} p_j, w = floor(u Z / 2^30), Z = sum_j p_j.
    theta = 0: the smallest index attaining the maximum."""
    if theta == 0:
        return torch.argmax(z, dim=1)
    p = samp_table(theta)[(z.max(dim=1, keepdim=True).values - z)]
    cum = torch.cumsum(p, dim=1)
    w = (u.to(torch.int64).to(_dev()) * cum[:, -1]) >> U_BITS
    return torch.searchsorted(cum, w.unsqueeze(1), right=True).squeeze(1)


def prf_u(r, n):
    """u_t = PRF(r, t) for t < n: the low 30 bits of BLAKE3(r || t)."""
    import blake3
    return torch.tensor([int.from_bytes(blake3.blake3(r + int(t).to_bytes(8, "little")).digest(length=4), "little")
                         & ((1 << U_BITS) - 1) for t in range(n)], dtype=torch.int64)


def forward(M, tokens):
    """Teacher-forced F_O over tokens (B, S): final hidden rows (B*S, d)."""
    B, S = tokens.shape
    cos, sin = rope_tables(M.cfg, S)
    h = M.emb[tokens.reshape(-1).to(_dev())].to(torch.int64)
    for Ly in M.layers:
        if Ly.Wqkv.device.type == "cpu":
            Ly = layer_to(Ly, "cuda")
        h = layer_forward(M, Ly, h, B, S, cos, sin)
    return h


def generate(M, prompt, ngen, Smax):
    """Greedy decode of F_O with integer KV caches.  prompt (P,) -> generated (ngen,)."""
    cos, sin = rope_tables(M.cfg, Smax)
    caches = [dict() for _ in M.layers]
    toks = prompt.view(1, -1).to(_dev())
    h = M.emb[toks.reshape(-1)].to(torch.int64)
    pos = 0
    out = []
    S = h.shape[0]
    for li, Ly in enumerate(M.layers):
        h = layer_forward(M, Ly, h, 1, S, cos, sin, cache=caches[li], pos0=0)
    z = final_logits(M, h[-1:])
    pos = S
    for t in range(ngen):
        y = int(torch.argmax(z[0]))           # smallest index attaining the max
        out.append(y)
        if t == ngen - 1:
            break
        h = M.emb[torch.tensor([y], device=_dev())].to(torch.int64)
        for li, Ly in enumerate(M.layers):
            h = layer_forward(M, Ly, h, 1, 1, cos, sin, cache=caches[li], pos0=pos)
        z = final_logits(M, h)
        pos += 1
    return torch.tensor(out)
