"""Prove (and optionally verify) the F_O model of SABLE_ARCH (default Qwen2.5-32B) on B x S tokens.

usage: python -m scripts.bench L B S ngen [verify]
L limits the number of layers (the per-layer work is identical across layers); the prompt is random,
ngen tokens are decoded greedily.  Profiling aids: ZK_SCLOG=1 prints the slowest sumchecks, ZK_DOMT=1
the slowest domains, ZK_NVTX=1 adds NVTX ranges (for nsys)."""
import copy
import os
import sys
import time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from zk import model as mdl
from zk import llm
from scripts import paths


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    L, B, S, ngen = (int(x) for x in sys.argv[1:5])
    do_verify = len(sys.argv) > 5 and sys.argv[5] == "verify"
    t0 = time.time()
    M = mdl.load_model(paths.fo_model(), layers_on="cpu")
    M = copy.copy(M)
    M.layers = M.layers[:L]
    M.cfg = copy.copy(M.cfg)
    M.cfg.L = L
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    tokens = torch.randint(0, M.cfg.V, (B, S), generator=torch.Generator().manual_seed(0)).cuda()
    tokens, targets = llm.greedy_decode(M, tokens, ngen)
    torch.cuda.empty_cache()
    print(f"forward {time.time() - t1:.0f}s alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), ngen)
    t0 = time.time()
    cw = llm.weights_digest(M, sh)              # C_W: computed once per model release
    print(f"C_W {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    tp = time.time() - t0
    st = llm.LAST["times"]
    print(f"L={L} B={B} S={S}: prove {tp:.1f}s, "
          f"proof {llm.LAST['bytes'] / 2**20:.1f} MiB, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print({k: round(v, 1) for k, v in sorted(st.items())}, flush=True)
    print("MiB by step:", {k: round(v / 2**20, 2) for k, v in sorted(llm.LAST["bytes_by"].items()) if v}, flush=True)
    if os.environ.get("ZK_SCLOG"):
        from zk import sumcheck as _sc
        tot = [sum(v[i] for v in _sc.SCLOG.values()) for i in range(4)]
        print(f"sumchecks {tot[0]} rounds {tot[1]} time {tot[2]:.1f}s")
        for k, v in sorted(_sc.SCLOG.items(), key=lambda kv: -kv[1][2])[:40]:
            print(f"  {k:40s} calls {v[0]:6d} rounds {v[1]:7d} time {v[2]:7.2f}s  cells {v[3]:.2e}")
    if os.environ.get("ZK_DOMT"):
        from zk import domain as _dm
        for k, v in sorted(_dm.DOMT.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  dom {k[0]:28s} {k[1]:6s} {v:7.2f}s")
    if do_verify:
        t0 = time.time()
        llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
        print(f"verify ok {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
