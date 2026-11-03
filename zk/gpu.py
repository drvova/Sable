"""Builds and loads the CUDA extension (one build per extension degree)."""
import os
import torch
from torch.utils.cpp_extension import load
from .kern import cuda_arch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODS = {}


def _ninja_on_path():
    """torch's extension loader needs ninja, which the virtual environment installs next to the
    interpreter; put that directory on PATH when it is not there."""
    import shutil
    import sys
    if shutil.which("ninja") is None:
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")


def module(D=8):
    if D in _MODS:
        return _MODS[D]
    _ninja_on_path()
    srcs = [os.path.join(_HERE, "cuda", "pcs.cu"), os.path.join(_HERE, "cuda", "bind.cpp")]
    # SABLE_BUILD_DIR: where the extension is built (default: build/ next to the zk package, i.e. the
    # repository root of a source checkout; set it when the package directory is not writable)
    build = os.path.join(os.environ.get("SABLE_BUILD_DIR") or os.path.join(_HERE, "..", "build"), f"ext{D}")
    os.makedirs(build, exist_ok=True)
    m = load(
        name=f"zkgpu{D}",
        sources=srcs,
        extra_cflags=["-O3", f"-DEXT_D={D}"],
        extra_cuda_cflags=["-O3", f"-DEXT_D={D}", f"-arch={cuda_arch()}", "--expt-relaxed-constexpr",
                           "-lineinfo", "-Xptxas=-v" if os.environ.get("PTXAS_V") else "-w"],
        extra_include_paths=[os.path.join(_HERE, "cuda")],
        build_directory=build,
        verbose=bool(os.environ.get("ZK_VERBOSE")),
    )
    _MODS[D] = m
    return m


def u32(t):
    """View an int32 tensor's bits as values in [0, 2^32)."""
    return t.to(torch.int64) & 0xFFFFFFFF
