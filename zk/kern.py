"""NVRTC (CuPy) kernel compilation with caching, and launch helpers that take
torch tensors."""
import os
import hashlib
import numpy as np
import cupy as cp
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_INC = os.path.join(_HERE, "cuda")
_CACHE = {}


def _inc_digest():
    h = hashlib.sha1()
    for f in sorted(os.listdir(_INC)):
        if f.endswith(".cuh"):
            with open(os.path.join(_INC, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


_INC_TAG = f"// headers {_inc_digest()}\n"


def cuda_arch():
    """Target architecture of the compiled kernels: SABLE_CUDA_ARCH (e.g. sm_120), else the current
    device's compute capability.  Sable is developed and tested on sm_120 (RTX PRO 6000 Blackwell)."""
    arch = os.environ.get("SABLE_CUDA_ARCH")
    if not arch:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
    return arch


def compile_module(src, D, name_hint="k", fast=False):
    """fast: NVVM fast-compile mode (about 10x quicker to build, slower code); for kernels
    whose launches are small, e.g. sumchecks over per-row domains."""
    src = _INC_TAG + src
    key = (hashlib.sha1(src.encode()).hexdigest(), D, bool(os.environ.get("ZK_F4INLINE")), fast)
    if key in _CACHE:
        return _CACHE[key]
    opts = ("-std=c++17", f"--gpu-architecture={cuda_arch()}", f"-I{_INC}", f"-DEXT_D={D}",
            "--use_fast_math", "-default-device", "-DF8_NOINLINE") + \
        (("-DF4_INLINE",) if os.environ.get("ZK_F4INLINE") else ()) + ((f"-Ofc={os.environ.get('ZK_OFC', 'min')}",) if fast else ())
    m = cp.RawModule(code=src, options=opts, name_expressions=None)
    _CACHE[key] = m
    return m


class _Ring:
    """Pinned staging ring for small uploads.  Two halves; the copies from a half are queued
    on the current stream, an event is recorded when the ring leaves the half, and the half is
    reused only after that event has completed."""
    SIZE = 32 << 20

    def __init__(self):
        self.buf = torch.empty(self.SIZE, dtype=torch.uint8, pin_memory=True)
        self.np = self.buf.numpy()
        self.cur, self.off = 0, 0
        self.ev = [None, None]

    def put(self, a):
        n, half = a.nbytes, self.SIZE // 2
        if n == 0 or n > half:
            return None
        o = (self.off + 255) & ~255
        if o + n > (self.cur + 1) * half:
            self.ev[self.cur] = torch.cuda.Event()
            self.ev[self.cur].record()
            self.cur = 1 - self.cur
            if self.ev[self.cur] is not None:
                self.ev[self.cur].synchronize()
            o = self.cur * half
        self.np[o:o + n] = a.view(np.uint8).reshape(-1)
        self.off = o + n
        return self.buf[o:o + n]


_RING = []


# During a CUDA-graph capture (zk/graph.py): a list that keeps every staging buffer alive, since
# the captured copies read their pinned source again on every replay.
CAPTURE = None


def h2d(a):
    """numpy array -> CUDA tensor without a stream sync (pinned staging copy)."""
    a = np.ascontiguousarray(a)
    if CAPTURE is not None:
        src = torch.from_numpy(a.copy()).pin_memory()
        CAPTURE.append(src)
        return src.to("cuda", non_blocking=True)
    if not _RING:
        _RING.append(_Ring())
    src = _RING[0].put(a)
    if src is None:
        return torch.from_numpy(a).pin_memory().to("cuda", non_blocking=True)
    out = torch.empty(a.shape, dtype=torch.from_numpy(a[:0].reshape(-1)).dtype, device="cuda")
    out.view(torch.uint8).reshape(-1).copy_(src, non_blocking=True)
    return out


def ptr(t):
    """Raw device pointer of a torch tensor (or None -> 0)."""
    if t is None:
        return np.uint64(0)
    return np.uint64(t.data_ptr())


def i64(v):
    return np.int64(v)


def i32(v):
    return np.int32(v)


def launch(fn, grid, block, args, smem=0):
    fn(grid if isinstance(grid, tuple) else (grid,),
       block if isinstance(block, tuple) else (block,), tuple(args), shared_mem=smem)


def file_module(fname, D, extra=""):
    with open(os.path.join(_INC, fname)) as f:
        src = f.read()
    return compile_module(extra + src, D, fname)
