"""GPU extension-field vector ops on torch int32 tensors of shape (D, n)."""
import numpy as np
import torch
from .field import P, Ext
from . import kern
from .kern import ptr, i64, h2d

NT = 256


def _grid(m):
    return int(max(1, min((m + NT - 1) // NT, 188 * 16)))


class GOps:
    def __init__(self, D):
        self.D = D
        self.E = Ext(D)
        self.mod = kern.file_module("util.cu", D)
        self.f = {n: self.mod.get_function(n) for n in (
            "ext_fold", "ext_fold_many", "fold_list_p", "base_fold", "ext_fold_p", "ext_fold_many_p", "base_fold_p", "eh_build", "ext_fold_hi", "ext_fold_hi_p", "comb_coefs", "tail_expand", "eq_expand", "ext_mul_vec", "ext_scale",
            "ext_axpy", "base_axpy", "ext_dot_partial", "base_ext_dot_partial", "to_field",
            "eq_segments", "eq_outer", "gkr_pair", "gkr_pair_list", "base_inv", "cw_fold", "rowdot_b", "rowdot_i8", "rowdot_m16", "coldot_b", "coldot_i8")}

    # host <-> device ------------------------------------------------------
    def const(self, a):
        """numpy (D,) -> device int32 (D,)"""
        a = np.asarray(a, dtype=np.int64) % P
        return h2d(a.astype(np.uint32).view(np.int32))

    def r8(self, a):
        a = np.asarray(a, dtype=np.int64) % P
        out = [np.uint32(v) for v in a]
        return out + [np.uint32(0)] * (8 - len(out))

    def to_dev(self, a):
        from .lazy import Lz
        if isinstance(a, Lz):
            return a.dev.contiguous()
        a = np.asarray(a, dtype=np.int64) % P
        return h2d(np.ascontiguousarray(a.astype(np.uint32)).view(np.int32))

    def to_host(self, t):
        return t.detach().cpu().numpy().view(np.uint32).astype(np.int64)

    def zeros(self, n):
        return torch.zeros((self.D, n), dtype=torch.int32, device="cuda")

    def from_base(self, t):
        """int32 field values (n,) -> ext (D, n)."""
        out = self.zeros(t.numel())
        out[0] = t
        return out

    # kernels --------------------------------------------------------------
    def fold(self, t, r, out=None):
        m = t.shape[1] // 2
        if out is None:
            out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_fold"], _grid(m), NT,
                    [ptr(t), i64(t.stride(0)), ptr(out), i64(out.stride(0)), *self.r8(r), i64(m)])
        return out

    def fold_p(self, t, rp):
        """fold with the challenge at device pointer rp."""
        m = t.shape[1] // 2
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_fold_p"], _grid(m), NT,
                    [ptr(t), i64(t.stride(0)), ptr(out), i64(out.stride(0)), np.uint64(rp), i64(m)])
        return out

    def base_fold_p(self, t, rp):
        m = t.numel() // 2
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["base_fold_p"], _grid(m), NT,
                    [ptr(t), ptr(out), i64(out.stride(0)), np.uint64(rp), i64(m)])
        return out

    _FJ = np.dtype([("v", np.int64, 6 * 32)])

    def fold_list_p(self, ts, rp):
        """fold_p / base_fold_p of every table in ts (1-D base or (D, n) ext) with the challenge
        at device pointer rp, 32 tables per launch.  Returns the folded (D, n/2) tables."""
        outs = []
        for i in range(0, len(ts), 32):
            chunk = ts[i:i + 32]
            J = np.zeros((), dtype=self._FJ)
            v = J["v"].reshape(6, 32)
            mx = 1
            for j, t in enumerate(chunk):
                base = t.dim() == 1
                m = (t.numel() if base else t.shape[1]) // 2
                o = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
                v[:, j] = (t.data_ptr(), 0 if base else t.stride(0), o.data_ptr(), o.stride(0), m, int(base))
                outs.append(o)
                mx = max(mx, m)
            kern.launch(self.f["fold_list_p"], (_grid(mx), len(chunk)), NT, [J[()], np.int32(len(chunk)), np.uint64(rp)])
        return outs

    def fold_hi(self, t, r):
        m = t.shape[1] // 2
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_fold_hi"], _grid(m), NT,
                    [ptr(t), i64(t.stride(0)), ptr(out), i64(out.stride(0)), *self.r8(r), i64(m)])
        return out

    def fold_hi_p(self, t, rp):
        """fold_hi with the challenge at device pointer rp."""
        m = t.shape[1] // 2
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_fold_hi_p"], _grid(m), NT,
                    [ptr(t), i64(t.stride(0)), ptr(out), i64(out.stride(0)), np.uint64(rp), i64(m)])
        return out

    def comb_coefs(self, beta, C):
        """beta device rows (n, D), C numpy (n, nk, D) -> device (D, nk) = sum_i beta_i C_i."""
        if torch.is_tensor(C):          # device (n * nk, D), nk = C.shape[0] // beta rows
            n = beta.shape[0]
            nk = C.shape[0] // n
            Cd = C.contiguous()
        else:
            n, nk = C.shape[0], C.shape[1]
            Cd = self.to_dev(C.reshape(-1))
        out = torch.empty((self.D, nk), dtype=torch.int32, device="cuda")
        kern.launch(self.f["comb_coefs"], (nk + 127) // 128, 128,
                    [ptr(beta), ptr(Cd), np.int32(n), np.int32(nk), ptr(out)])
        return out

    def dot_dev(self, a, b):
        """dot on the device (no synchronization) -> int32 (D,)."""
        return self.dots_dev([(a, b)])[0]

    def scale_p(self, a, s):
        """a * s with s an ext value on the device (D,) (or Lz / host value)."""
        from .lazy import Lz
        if not (torch.is_tensor(s) or isinstance(s, Lz)):
            return self.scale(a, s)
        sd = s.dev if isinstance(s, Lz) else s
        m = a.shape[1]
        return self.mul(a, sd.reshape(self.D, 1).expand(self.D, m).contiguous())

    def dots_dev(self, pairs):
        """dots on the device, no synchronization -> int32 (len, D) canonical."""
        if not pairs:
            return torch.zeros((0, self.D), dtype=torch.int32, device="cuda")
        sums = []
        for a, b in pairs:
            m = b.shape[1]
            g = _grid(m)
            out = torch.empty((g, self.D), dtype=torch.int32, device="cuda")
            if a.dim() == 1:
                kern.launch(self.f["base_ext_dot_partial"], g, NT,
                            [ptr(a), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
            else:
                kern.launch(self.f["ext_dot_partial"], g, NT,
                            [ptr(a), i64(a.stride(0)), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
            sums.append(torch.sum(out.to(torch.int64) & 0xFFFFFFFF, dim=0))
        return (torch.stack(sums) % P).to(torch.int32)

    def base_fold(self, t, r):
        m = t.numel() // 2
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["base_fold"], _grid(m), NT,
                    [ptr(t), ptr(out), i64(out.stride(0)), *self.r8(r), i64(m)])
        return out

    def eq_table(self, r):
        """r: numpy (D, k) -> (D, 2^k) eq(r, x), x LSB = r[:,0]."""
        k = r.shape[1]
        n = 1 << k
        if k <= 14:
            from .sumcheck import eq_segments
            return eq_segments(self, r, [(0, k)], [0], n)
        from .sumcheck import eq_segments
        kl = k // 2
        buf = eq_segments(self, r, [(0, kl), (kl, k - kl)], [0, 1 << kl], (1 << kl) + (1 << (k - kl)))
        t = torch.empty((self.D, n), dtype=torch.int32, device="cuda")
        kern.launch(self.f["eq_outer"], _grid(n), NT,
                    [ptr(buf), i64(buf.stride(0)), np.int32(kl), i64(1 << kl), ptr(t), i64(n)])
        return t

    def mul(self, a, b, out=None):
        m = a.shape[1]
        if out is None:
            out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_mul_vec"], _grid(m), NT,
                    [ptr(a), i64(a.stride(0)), ptr(b), i64(b.stride(0)), ptr(out), i64(out.stride(0)), i64(m)])
        return out

    def scale(self, a, s):
        m = a.shape[1]
        out = torch.empty((self.D, m), dtype=torch.int32, device="cuda")
        kern.launch(self.f["ext_scale"], _grid(m), NT,
                    [ptr(a), i64(a.stride(0)), *self.r8(s), ptr(out), i64(out.stride(0)), i64(m)])
        return out

    def axpy(self, c, a, s):
        """c += a * s (in place); a ext or base."""
        if a.dim() == 1:
            m = a.numel()
            kern.launch(self.f["base_axpy"], _grid(m), NT,
                        [ptr(a), *self.r8(s), ptr(c), i64(c.stride(0)), i64(m)])
        else:
            m = a.shape[1]
            kern.launch(self.f["ext_axpy"], _grid(m), NT,
                        [ptr(a), i64(a.stride(0)), *self.r8(s), ptr(c), i64(c.stride(0)), i64(m)])
        return c

    def add(self, a, b):
        s = a.to(torch.int64) + b.to(torch.int64)
        return (s % P).to(torch.int32)

    def dot(self, a, b):
        """sum_i a_i * b_i.  a ext (D,m) or base (m,), b ext (D,m).  -> numpy (D,)"""
        m = b.shape[1]
        g = _grid(m)
        out = torch.empty((g, self.D), dtype=torch.int32, device="cuda")
        if a.dim() == 1:
            kern.launch(self.f["base_ext_dot_partial"], g, NT,
                        [ptr(a), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
        else:
            kern.launch(self.f["ext_dot_partial"], g, NT,
                        [ptr(a), i64(a.stride(0)), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
        return self.sum_rows(out)

    def dots(self, pairs):
        """[dot(a, b) for a, b in pairs] with one host transfer -> numpy (len, D)."""
        if not pairs:
            return np.zeros((0, self.D), dtype=np.int64)
        sums = []
        for a, b in pairs:
            m = b.shape[1]
            g = _grid(m)
            out = torch.empty((g, self.D), dtype=torch.int32, device="cuda")
            if a.dim() == 1:
                kern.launch(self.f["base_ext_dot_partial"], g, NT,
                            [ptr(a), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
            else:
                kern.launch(self.f["ext_dot_partial"], g, NT,
                            [ptr(a), i64(a.stride(0)), ptr(b), i64(b.stride(0)), i64(m), ptr(out)])
            sums.append(torch.sum(out.to(torch.int64) & 0xFFFFFFFF, dim=0))
        return (torch.stack(sums).cpu().numpy() % P).astype(np.int64)

    def sum_rows(self, out):
        """(g, D) int32 partials -> numpy (D,) sum mod p."""
        s = torch.sum(out.view(torch.int32).to(torch.int64) & 0xFFFFFFFF, dim=0)
        return (s.cpu().numpy() % P).astype(np.int64)

    def sum(self, t):
        """(D, m) -> numpy (D,)"""
        s = torch.sum(t.to(torch.int64) & 0xFFFFFFFF, dim=1)
        return (s.cpu().numpy() % P).astype(np.int64)

    def mle(self, t, r):
        """MLE of t (base (2^k,) or ext (D, 2^k)) at r numpy (D, k)."""
        k = r.shape[1]
        cur = t
        for j in range(k):
            cur = self.base_fold(cur, r[:, j]) if cur.dim() == 1 else self.fold(cur, r[:, j])
        if cur.dim() == 1:
            return self.E.from_base(int(cur[0]) & 0xFFFFFFFF)
        return self.to_host(cur[:, 0])

    def rowdot(self, X, e, R, L):
        """out[:, r] = sum_j e[:, j] X[r L + j]; X int32 field or int8 (R*L,), e (D, L) -> (D, R)."""
        CH = 1 << 16
        nch = (L + CH - 1) // CH
        part = torch.empty((R * nch, self.D), dtype=torch.int32, device="cuda")
        fn = self.f["rowdot_i8" if X.dtype == torch.int8 else "rowdot_b"]
        kern.launch(fn, R * nch, NT, [ptr(X), i64(L), ptr(e), i64(e.stride(0)), i64(nch), i64(CH), ptr(part)])
        s = (part.view(R, nch, self.D).to(torch.int64) & 0xFFFFFFFF).sum(dim=1) % P
        return s.T.contiguous().to(torch.int32)

    def rowdot_multi(self, X, es, R, L):
        """[rowdot(X, e, R, L) for e in es] in one read of X (int16 (R*L,)); es: (D, L) tables."""
        out = []
        for i in range(0, len(es), 8):
            grp = es[i:i + 8]
            E = torch.cat([e[:, :L].contiguous() for e in grp], dim=0).contiguous()     # (ns D, L)
            CH = 1 << 16
            nch = (L + CH - 1) // CH
            part = torch.empty((R * nch, len(grp), self.D), dtype=torch.int32, device="cuda")
            kern.launch(self.f["rowdot_m16"], R * nch, NT, [ptr(X), i64(L), ptr(E), np.int32(len(grp)), i64(nch),
                                                             i64(CH), ptr(part)])
            s = (part.view(R, nch, len(grp), self.D).to(torch.int64) & 0xFFFFFFFF).sum(dim=1) % P   # (R, ns, D)
            out += [s[:, k].T.contiguous().to(torch.int32) for k in range(len(grp))]
        return out

    def coldot(self, X, e, K, M, C):
        """out[:, k C + c] = sum_m e[:, m] X[(k M + m) C + c] -> (D, K C)."""
        MCH = max(1, min(M, (1 << 22) // max(1, K * C)))
        MCH = max(MCH, 64)
        nm = (M + MCH - 1) // MCH
        nct = (C + NT - 1) // NT
        part = torch.empty((nm, self.D, K * C), dtype=torch.int32, device="cuda")
        fn = self.f["coldot_i8" if X.dtype == torch.int8 else "coldot_b"]
        kern.launch(fn, K * nm * nct, NT, [ptr(X), i64(K), i64(M), i64(C), ptr(e), i64(e.stride(0)), i64(MCH),
                                           ptr(part), i64(K * C)])
        return ((part.to(torch.int64) & 0xFFFFFFFF).sum(dim=0) % P).to(torch.int32)

    def to_field(self, t):
        """signed int32 -> field int32 (new tensor)."""
        out = torch.empty_like(t, dtype=torch.int32)
        m = t.numel()
        kern.launch(self.f["to_field"], _grid(m), NT, [ptr(t), ptr(out), i64(m)])
        return out


_G = {}
# verifier on the host (llm.verify(..., cpu=True)): gops returns cpuops.CPUOps and dev() is "cpu"
CPU = [False]
_GC = {}


def dev():
    return "cpu" if CPU[0] else "cuda"


def gops(D):
    if CPU[0]:
        if D not in _GC:
            from .cpuops import CPUOps
            _GC[D] = CPUOps(D)
        return _GC[D]
    if D not in _G:
        _G[D] = GOps(D)
    return _G[D]
