"""CPU twin of ext.GOps for the verifier: the same methods and tensor conventions ((D, n) int32 torch
tensors of canonical values), computed exactly on the host with numpy (field.Ext), so a proof can be
verified without a GPU (llm.verify(..., cpu=True))."""
import numpy as np
import torch
from .field import P, Ext


def _np(t):
    """torch int32 field tensor -> numpy int64 (same shape)."""
    if isinstance(t, np.ndarray):
        return np.asarray(t, dtype=np.int64) % P
    return t.detach().cpu().contiguous().numpy().view(np.uint32).astype(np.int64)


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int64) % P).astype(np.int32))


class CPUOps:
    cpu = True

    def __init__(self, D):
        self.D = D
        self.E = Ext(D)

    # host <-> "device" -----------------------------------------------------
    def const(self, a):
        return _t(a)

    def r8(self, a):
        a = np.asarray(a, dtype=np.int64) % P
        out = [np.uint32(v) for v in a]
        return out + [np.uint32(0)] * (8 - len(out))

    def to_dev(self, a):
        from .lazy import Lz
        if isinstance(a, Lz):
            return a.dev.contiguous()
        return _t(a)

    def to_host(self, t):
        return _np(t)

    def zeros(self, n):
        return torch.zeros((self.D, n), dtype=torch.int32)

    def from_base(self, t):
        out = self.zeros(t.numel())
        out[0] = t
        return out

    # arithmetic ------------------------------------------------------------
    def fold(self, t, r, out=None):
        a = _np(t)
        lo, hi = a[:, 0::2], a[:, 1::2]
        rr = (np.asarray(r, dtype=np.int64) % P).reshape(self.D, 1)
        res = _t(self.E.add(lo, self.E.mul(rr, self.E.sub(hi, lo))))
        if out is not None:
            out.copy_(res)
            return out
        return res

    def base_fold(self, t, r):
        a = _np(t).reshape(-1)
        d = self.E.from_base(a[1::2] - a[0::2])
        rr = (np.asarray(r, dtype=np.int64) % P).reshape(self.D, 1)
        return _t(self.E.add(self.E.from_base(a[0::2]), self.E.mul(rr, d)))

    def eq_rows(self, r):
        """r numpy (D, k) -> numpy (D, 2^k) eq(r, x), x LSB = r[:, 0]."""
        from . import vcore
        return vcore.eq_table(self.D, np.asarray(r, dtype=np.int64) % P)

    def eq_table(self, r):
        return _t(self.eq_rows(r))

    def eq_segments(self, r, segs, offs, total):
        """sumcheck.eq_segments: r rows (k, D) or host (D, k); segment s = eq over rows st..st+ln-1."""
        if torch.is_tensor(r):
            rr = _np(r).T if r.shape[0] else np.zeros((self.D, 0), dtype=np.int64)    # (k, D) rows -> (D, k)
        else:
            rr = np.asarray(r, dtype=np.int64) % P
        buf = np.zeros((self.D, total), dtype=np.int64)
        for (st, ln), off in zip(segs, offs):
            buf[:, off:off + (1 << ln)] = self.eq_rows(rr[:, st:st + ln])
        return _t(buf)

    def mul(self, a, b, out=None):
        from . import vcore
        res = _t(vcore.vmul(self.D, _np(a), _np(b)))
        if out is not None:
            out.copy_(res)
            return out
        return res

    def scale(self, a, s):
        return _t(self.E.mul(_np(a), (np.asarray(s, dtype=np.int64) % P).reshape(self.D, 1)))

    def axpy(self, c, a, s):
        av = _np(a)
        if av.ndim == 1:
            av = self.E.from_base(av)
        c.copy_(_t(self.E.add(_np(c), self.E.mul(av, (np.asarray(s, dtype=np.int64) % P).reshape(self.D, 1)))))
        return c

    def add(self, a, b):
        return _t(_np(a) + _np(b))

    def dot(self, a, b):
        from . import vcore
        return vcore.dot(self.D, _np(a), _np(b))

    def dots(self, pairs):
        if not pairs:
            return np.zeros((0, self.D), dtype=np.int64)
        return np.stack([self.dot(a, b) for a, b in pairs])

    def sum_rows(self, out):
        return (_np(out).sum(axis=0) % P).astype(np.int64)

    def sum(self, t):
        return (_np(t).sum(axis=1) % P).astype(np.int64)

    def mle(self, t, r):
        """MLE of the first 2^k entries of t (as GOps.mle, which folds k times and reads entry 0)."""
        from . import vcore
        k = r.shape[1]
        a = _np(t)
        if a.shape[-1] >= (1 << k):
            v = vcore.mle(self.D, a[..., :1 << k], r)
            return v if a.ndim == 2 else v
        cur = t
        for j in range(k):
            cur = self.base_fold(cur, r[:, j]) if cur.dim() == 1 else self.fold(cur, r[:, j])
        if cur.dim() == 1:
            return self.E.from_base(int(cur[0]) & 0xFFFFFFFF)
        return self.to_host(cur[:, 0])

    def to_field(self, t):
        return _t(t.to(torch.int64).numpy())

    def __getattr__(self, name):
        raise AttributeError(f"CPUOps.{name}: a GPU-only operation (no host version)")
