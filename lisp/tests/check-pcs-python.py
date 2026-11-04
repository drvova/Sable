"""Check polynomial-commitment proofs made by the Lisp port with zk/gpcs.py's own verifier (on
the host, as llm.verify(..., cpu=True)): with the same units, merge set, segment and claims it must
accept the proof and consume every message; with any one value or byte changed it must reject it
(except the grinding nonce, which the tests' few bits of grinding may still accept).

Needs numpy, blake3, g++ and PyTorch (the CPU build is enough); CuPy is replaced by a stand-in.

    make crosscheck PYTHON=/path/to/python-with-torch
"""
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.modules.setdefault("cupy", types.ModuleType("cupy"))
from zk import ext, gpcs, proofio  # noqa: E402
from zk.transcript import Transcript  # noqa: E402

ext.CPU[0] = True


def run(case, msgs):
    gpcs.MAX_RED_POINTS = case["max_points"]          # more points are reduced in batches
    tr = Transcript(case["label"].encode(), 4, msgs)
    pcs = gpcs.GPCS(tr, False, cmax=case["cmax"], queries=case["queries"], grind=case["grind"])
    for u in case["units"]:
        pcs.declare(u["name"], [tuple(item) for item in u["items"]])
    if case["merge"]:
        pcs.merge(case["merge"])
    if case["segment"]:
        pcs.segment_decl(case["segment"])
    for u in case["units"]:
        pcs.commit(u["name"])
    for uc in case["claims"]:
        claims = [(c["tensor"], np.array(c["point"], dtype=np.int64).T.reshape(4, -1),
                   {int(b): np.array(w, dtype=np.int64) for b, w in c["coefficients"].items()},
                   np.array(c["value"], dtype=np.int64)) for c in uc["claims"]]
        pcs.open(uc["unit"], claims)
        if case["segment"] and uc["unit"] == case["segment"][-1]:
            pcs.segment(0)
    assert pcs.finish()
    return tr


def main(path):
    for n, case in enumerate(json.load(open(path))):
        msgs = proofio.loads(bytes.fromhex(case["proof"]))
        tr = run(case, msgs)
        assert tr.pos == len(msgs), "messages left over"
        rejected = skipped = 0
        for i, m in enumerate(msgs):
            if (isinstance(m, bytes) and len(m) in (0, 8)) or (not isinstance(m, bytes) and m.size == 0):
                skipped += 1             # an empty path message, or the grinding nonce
                continue
            forged = list(msgs)
            if isinstance(m, bytes):
                forged[i] = bytes([m[0] ^ 1]) + m[1:]
            else:
                forged[i] = m.copy()
                forged[i].reshape(-1)[0] = (forged[i].reshape(-1)[0] + 1) % proofio.P
            try:
                run(case, forged)
            except (ValueError, AssertionError):
                rejected += 1
                continue
            raise AssertionError(f"case {n}: forgery accepted at message {i}")
        print(f"case {n} (cmax {case['cmax']}, {len(case['units'])} units"
              f"{', a segment' if case['segment'] else ''}"
              f"{', points reduced in batches' if case['max_points'] < 64 else ''}): zk/gpcs.py accepts the Lisp proof, "
              f"{len(msgs)} messages, all consumed; rejects all {rejected} forgeries "
              f"({skipped} empty or nonce messages not changed)")


if __name__ == "__main__":
    main(sys.argv[1])
