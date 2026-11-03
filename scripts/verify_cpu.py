"""Host verification of a saved proof (scripts/bench_e2e.py with BENCH_SAVE): llm.verify(..., cpu=True)
with the compiled core and no GPU, as in the paper's verification times.

usage: CUDA_VISIBLE_DEVICES= python -m scripts.verify_cpu PREFIX [REPEAT]
Loads the model the proof was made for (on the host), the statement, C_W and the fixed-unit digests
(setup, not timed), then verifies REPEAT times (default 2) and prints each time."""
import json
import os
import pickle
import sys
import time
import torch
from zk import model as mdl, llm, proofio, ext


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES="
    pre, rep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
    d = pickle.load(open(pre + ".pkl", "rb"))
    msgs = proofio.loads(open(pre + ".proof", "rb").read())
    ext.CPU[0] = True                                  # load onto the host
    M = mdl.load_model(d["model"], layers_on="cpu")
    ext.CPU[0] = False
    M.layers = M.layers[:d["L"]]
    M.cfg.L = d["L"]
    M._tab_cache = d["tab_cache"]
    sh = llm.Shp(M.cfg, *d["shape"])
    policy = llm.Policy(theta=d["theta"])
    for i in range(rep):
        t0 = time.time()
        llm.verify(M, sh, d["tokens"], d["targets"], msgs, C_W=d["cw"], policy=policy, nc=d["nc"], cpu=True)
        print(json.dumps(dict(B=d["shape"][0], S=d["shape"][1], L=d["L"], verify_cpu=round(time.time() - t0, 1),
                              threads=torch.get_num_threads())), flush=True)


if __name__ == "__main__":
    main()
