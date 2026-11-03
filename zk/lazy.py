"""Lazy host views of device field arrays (prover only).

Lz wraps a device int32 tensor holding canonical field values in the layout the host code
expects of a numpy int64 array (e.g. a point (D, k) or values (D, K)).  The host copy is made
(one synchronization) only when the host actually reads the values; slicing, transposing and
concatenation stay on the device, and device consumers (eq tables, transcript sends) use .dev.
Every numpy-style use works unchanged, so a forgotten device path costs a sync, never a bug."""
import weakref
import numpy as np
import torch
from .field import P


class _IdMap:
    """Weak map by object identity (tensors compare elementwise, so no WeakKeyDictionary)."""

    def __init__(self):
        self.d = {}

    def __setitem__(self, t, v):
        i = id(t)
        if i not in self.d:
            weakref.finalize(t, self.d.pop, i, None)
        self.d[i] = v

    def get(self, t, default=None):
        return self.d.get(id(t), default)

    def __contains__(self, t):
        return id(t) in self.d

    def __getitem__(self, t):
        return self.d[id(t)]


KEYS = _IdMap()   # device challenge rows (k, D) or vector (D,) -> per-row draw keys


def rows_lz(t, keys=None):
    """Lz of host layout (D, k) from device rows (k, D), with the rows' draw keys (KEYS)."""
    return Lz(t.T, keys=tuple(keys) if keys is not None else KEYS.get(t))


def col_keys(p):
    """Per-column keys of a point (D, k): draw keys of a keyed Lz, values of host columns, else None."""
    if isinstance(p, Lz):
        return p.keys
    a = np.asarray(p, dtype=np.int64) % P
    return tuple(("v", tuple(int(x) for x in a[:, j])) for j in range(a.shape[1]))


class Lz:
    __slots__ = ("_d", "_h", "_p", "_i", "keys", "__weakref__")

    def __init__(self, d, parent=None, idx=None, keys=None):
        assert torch.is_tensor(d) and d.dtype == torch.int32
        self._d = d
        self._h = None
        self._p, self._i = parent, idx  # a view: its host copy is a slice of the parent's
        # points (D, k): one hashable per column naming the transcript draw it came from, so equal
        # points can be grouped without reading them (None: unknown)
        self.keys = keys

    # ---------------------------------------------------------------- device side
    @property
    def dev(self):
        return self._d

    @property
    def shape(self):
        return tuple(self._d.shape)

    @property
    def ndim(self):
        return self._d.dim()

    @property
    def size(self):
        return self._d.numel()

    @property
    def T(self):
        return Lz(self._d.T, self, "T")

    def __getitem__(self, idx):
        keys = None
        if self.keys is not None and isinstance(idx, tuple) and len(idx) == 2 and idx[0] == slice(None):
            s = idx[1]
            if isinstance(s, slice):
                keys = self.keys[s]
            elif isinstance(s, (int, np.integer)):
                keys = (self.keys[s],)
        return Lz(self._d[idx], self, idx, keys)

    def __len__(self):
        return self._d.shape[0]

    # ---------------------------------------------------------------- host side
    def np(self):
        if self._h is None:
            if self._p is not None:          # one transfer for the whole parent array
                ph = self._p.np()
                self._h = ph.T if self._i == "T" else ph[self._i]
            else:
                self._h = self._d.cpu().numpy().view(np.uint32).astype(np.int64)
        return self._h

    def __array__(self, dtype=None, copy=None):
        a = self.np()
        return a if dtype is None else a.astype(dtype)

    def __getattr__(self, name):        # anything else: the numpy array
        return getattr(self.np(), name)

    def __iter__(self):
        return iter(self.np())

    def __repr__(self):
        return f"Lz{self.shape}"


def _binop(name):
    def f(self, other):
        o = other.np() if isinstance(other, Lz) else other
        return getattr(self.np(), name)(o)
    return f


for _n in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__mod__", "__rmod__",
           "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__and__", "__or__", "__rshift__",
           "__lshift__", "__floordiv__", "__neg__", "__int__", "__bool__", "__index__"):
    if _n in ("__neg__", "__int__", "__bool__", "__index__"):
        setattr(Lz, _n, (lambda n: lambda self: getattr(self.np(), n)())(_n))
    else:
        setattr(Lz, _n, _binop(_n))


def is_lazy(x):
    return isinstance(x, Lz)


def to_dev(x):
    """numpy (canonical or not) / Lz -> device int32 tensor in the same layout."""
    if isinstance(x, Lz):
        return x.dev
    if torch.is_tensor(x):
        return x
    a = np.asarray(x, dtype=np.int64) % P
    from .kern import h2d
    return h2d(np.ascontiguousarray(a.astype(np.uint32)).view(np.int32))


def lcat(parts, axis=1):
    """np.concatenate along axis; a lazy result if any part is lazy."""
    nz = [p for p in parts if p.shape[axis]]
    if not nz:
        return parts[0]
    parts = nz
    if not any(isinstance(p, Lz) for p in parts):
        return np.concatenate(parts, axis=axis)
    keys = None
    if axis == 1:
        ks = [col_keys(p) for p in parts]
        if all(k is not None for k in ks):
            keys = tuple(x for k in ks for x in k)
    return Lz(torch.cat([to_dev(p) for p in parts], dim=axis), keys=keys)


def _root(x):
    while x._p is not None:
        x = x._p
    return x


def materialize(objs):
    """Host copies of every lazy array in objs (nested lists / tuples / dicts) with one transfer."""
    roots, seen = [], set()

    def walk(o):
        if isinstance(o, Lz):
            r = _root(o)
            if r._h is None and id(r) not in seen:
                seen.add(id(r))
                roots.append(r)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(objs)
    if not roots:
        return
    if len(roots) == 1:
        roots[0].np()
        return
    flat = torch.cat([r._d.reshape(-1) for r in roots])
    host = flat.cpu().numpy().view(np.uint32).astype(np.int64)
    off = 0
    for r in roots:
        n = r._d.numel()
        r._h = host[off:off + n].reshape(tuple(r._d.shape))
        off += n
