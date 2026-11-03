"""GPU Reed-Solomon encoding and BLAKE3 Merkle trees."""
import numpy as np
import torch
from . import rs
from .gpu import module

_TW = {}


class Twiddles:
    """All twiddle levels of the 2^n subgroup, concatenated on the GPU."""

    def __init__(self, n, device="cuda"):
        self.n = n
        tw = rs.twiddles_dev(n)
        offs = np.zeros(len(tw) + 1, dtype=np.int64)
        for i, t in enumerate(tw):
            offs[i + 1] = offs[i] + len(t)
        self.offs = offs
        self.dev = torch.cat(tw).to(torch.int32).to(device)
        self._host = None
        self.level_offs = {}

    @property
    def host(self):
        if self._host is None:
            self._host = [self.dev[a:b].cpu().numpy().view(np.uint32).astype(np.int64)
                          for a, b in zip(self.offs[:-1], self.offs[1:])]
        return self._host

    def offsets_for(self, level):
        if level not in self.level_offs:
            self.level_offs[level] = torch.from_numpy(self.offs[level:-1].copy())
        return self.level_offs[level]


def get_twiddles(n):
    if n not in _TW:
        _TW[n] = Twiddles(n)
    return _TW[n]


def encode(msg, n, level=0, D=8, out=None):
    """msg: int32 (ncols, 2^(n-level-1)) on GPU -> (ncols, 2^(n-level))."""
    m = module(D)
    tw = get_twiddles(n)
    L = n - level
    if out is None:
        out = torch.empty((msg.shape[0], 1 << L), dtype=torch.int32, device=msg.device)
    m.set_tw_offsets(tw.offsets_for(level))
    m.encode(msg, out, tw.dev, L, 0)
    return out


def encode_sel(msg, n, pos, level=0, D=8):
    """encode, with only the symbols at positions pos (int64 device tensor) computed in the last
    pass (the others in the result are undefined)."""
    m = module(D)
    tw = get_twiddles(n)
    out = torch.empty((msg.shape[0], 1 << (n - level)), dtype=torch.int32, device=msg.device)
    m.set_tw_offsets(tw.offsets_for(level))
    m.encode_sel(msg, out, tw.dev, n - level, 0, pos)
    return out


def leaf_hashes(cw, D=8):
    """cw: (<=16, N) int32 -> (N/16, 8) int32 leaf hashes."""
    m = module(D)
    out = torch.empty((cw.shape[1] // 16, 8), dtype=torch.int32, device=cw.device)
    m.leaf_hash(cw, out)
    return out


def merkle_levels(leaves, D=8, keep=True):
    """leaves: (N, 8) int32.  Returns list of levels, leaves first, root last."""
    m = module(D)
    levels = [leaves]
    cur = leaves
    while cur.shape[0] > 1:
        nxt = torch.empty((cur.shape[0] // 2, 8), dtype=torch.int32, device=cur.device)
        m.node_hash(cur, nxt)
        if keep:
            levels.append(nxt)
        else:
            levels = [nxt]
        cur = nxt
    return levels


def digest_bytes(h):
    """(8,) int32 tensor/array -> 32 bytes."""
    a = np.asarray(h.cpu() if torch.is_tensor(h) else h).astype(np.int32)
    return a.astype("<i4").tobytes()
