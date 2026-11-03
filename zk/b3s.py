"""Streaming BLAKE3 with an exportable flat state (zk/cuda/b3s.h), so the transcript
can move between the host and the GPU."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

_DIR = os.path.join(os.path.dirname(__file__), "cuda")
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        srcs = [os.path.join(_DIR, f) for f in ("b3s.h", "b3s_host.cpp")]
        dig = hashlib.sha1(b"".join(open(f, "rb").read() for f in srcs)).hexdigest()[:16]
        cache = os.path.join(os.path.expanduser("~"), ".cache", "zk_b3s")
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"b3s_{dig}.so")
        if not os.path.exists(so):
            tmp = so + f".{os.getpid()}.tmp"
            subprocess.check_call(["g++", "-O3", "-march=native", "-shared", "-fPIC", "-I", _DIR,
                                   srcs[1], "-o", tmp])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.b3s_size.restype = ctypes.c_uint64
        lib.b3s_init_c.argtypes = [ctypes.c_void_p]
        lib.b3s_update_c.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
        lib.b3s_digest_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        _LIB = lib
    return _LIB


def state_size():
    return int(_lib().b3s_size())


class Hasher:
    """blake3.blake3-compatible subset: update, copy, digest(length)."""

    def __init__(self, data=b"", state=None):
        lib = _lib()
        if state is None:
            self.st = np.zeros(state_size(), dtype=np.uint8)
            lib.b3s_init_c(self.st.ctypes.data)
        else:
            self.st = state
        if data:
            self.update(data)

    def update(self, data):
        data = bytes(data)
        _LIB.b3s_update_c(self.st.ctypes.data, data, len(data))
        return self

    def copy(self):
        return Hasher(state=self.st.copy())

    def digest(self, length=32):
        out = np.empty(length, dtype=np.uint8)
        _LIB.b3s_digest_c(self.st.ctypes.data, out.ctypes.data, length)
        return out.tobytes()
