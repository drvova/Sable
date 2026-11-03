"""CUDA-graph capture of prover work (all launches on the capture stream, CuPy included).

capture(g) records the GPU work queued inside the block into g instead of running it; the
Python side effects happen as usual.  g.replay() then runs the recorded work."""
import contextlib
import cupy as cp
import torch
from . import kern


@contextlib.contextmanager
def capture(g, pool=None):
    keep = []
    prev = kern.CAPTURE
    with torch.cuda.graph(g, pool=pool):
        kern.CAPTURE = keep
        try:
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                yield keep
        finally:
            kern.CAPTURE = prev
    g._zk_keep = keep          # staging buffers read by the graph's copies
