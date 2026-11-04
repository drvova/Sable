"""Check GKR proofs made by the Lisp port with the original C++ verifier, zk/cuda/vcore.cpp's
vc_gkr_upper (which llm.verify uses on the host): it must accept each proof, consume every message,
and compute the same roots, points and claims as the Lisp verifier; and it must reject each proof
with one value changed.  Needs numpy and g++.

    sbcl ... --eval '(sable.tests::export-gkr-cases "gkr-cases.json")'   (or: make crosscheck)
    python3 lisp/tests/check-gkr-vcore.py gkr-cases.json
"""
import ctypes
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from zk import proofio  # noqa: E402  (numpy only)

CUDA = os.path.join(ROOT, "zk", "cuda")
P = (1 << 31) - (1 << 24) + 1


def library():
    so = os.path.join(tempfile.mkdtemp(), "vcore.so")
    subprocess.check_call(["g++", "-O2", "-shared", "-fPIC", "-I", CUDA, os.path.join(CUDA, "vcore.cpp"),
                           os.path.join(CUDA, "b3s_host.cpp"), "-o", so])
    lib = ctypes.CDLL(so)
    p, i32, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.b3s_size.restype = ctypes.c_uint64
    lib.b3s_init_c.argtypes = [p]
    lib.b3s_update_c.argtypes = [p, ctypes.c_char_p, ctypes.c_uint64]
    lib.vc_gkr_upper.restype = i64
    lib.vc_gkr_upper.argtypes = [p, i32, p, p, i64, i64, ctypes.c_char_p, i32, i32, i32, i32, i32,
                                 p, p, p, p, p]
    return lib


def run(lib, case, msgs):
    D, n, K, L = case["D"], case["n"], case["K"], case["L"]
    lk = max(0, (K - 1).bit_length())
    st = np.zeros(lib.b3s_size(), dtype=np.uint8)
    lib.b3s_init_c(st.ctypes.data)
    prefix = b"LLM-SUMCHECK-v6-O|" + case["label"].encode()
    lib.b3s_update_c(st.ctypes.data, prefix, len(prefix))
    lens = np.array([m.size if isinstance(m, np.ndarray) else 0 for m in msgs], dtype=np.int64)
    off = np.zeros(len(msgs) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    flat = np.concatenate([m.reshape(-1).astype(np.int64) for m in msgs if isinstance(m, np.ndarray)])
    roots = np.zeros((L, 2, D), dtype=np.int64)
    ri = np.zeros((max(n - 1, 1), D), dtype=np.int64)
    rb = np.zeros((max(lk, 1), D), dtype=np.int64)
    cp = np.zeros((L, D), dtype=np.int64)
    cq = np.zeros((L, D), dtype=np.int64)
    pos = lib.vc_gkr_upper(st.ctypes.data, D, flat.ctypes.data, off.ctypes.data, len(msgs), 0, b"gkr", 3,
                           n, K, L, 3 if lk else 0, roots.ctypes.data, ri.ctypes.data, rb.ctypes.data,
                           cp.ctypes.data, cq.ctypes.data)
    return pos, roots.reshape(-1), ri[:n - 1].reshape(-1), rb[:lk].reshape(-1), cp.reshape(-1), cq.reshape(-1)


def main(path):
    lib = library()
    cases = json.load(open(path))
    for case in cases:
        msgs = proofio.loads(bytes.fromhex(case["proof"]))
        pos, roots, ri, rb, cp, cq = run(lib, case, msgs)
        shape = (case["n"], case["K"], case["L"], case["D"])
        assert pos == len(msgs), (shape, "vc_gkr_upper rejected the proof", pos)
        for name, got in (("roots", roots), ("ri", ri), ("rb", rb), ("cp", cp), ("cq", cq)):
            assert list(got) == case[name], (shape, name, "differs from the Lisp verifier")
        rejected = 0
        for i, m in enumerate(msgs):
            if isinstance(m, np.ndarray) and i >= case["L"]:
                forged = list(msgs)
                forged[i] = m.copy()
                forged[i].reshape(-1)[0] = (forged[i].reshape(-1)[0] + 1) % P
                assert run(lib, case, forged)[0] < 0, (shape, "forgery accepted at message", i)
                rejected += 1
        print(f"n={case['n']} K={case['K']} L={case['L']} D={case['D']}: accepted, {len(msgs)} messages, "
              f"same outputs as Lisp; {rejected} forgeries rejected")
    print(f"{len(cases)} cases: vc_gkr_upper agrees with the Lisp port")


if __name__ == "__main__":
    main(sys.argv[1])
