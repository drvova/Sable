"""End-to-end proving benchmark of the F_O model of SABLE_ARCH (default Qwen2.5-32B-Instruct; the paper's Section "Evaluation").

usage: python -m scripts.bench_e2e B S RUNSPEC [RUNSPEC ...]
  B, S      batch (sequences) and sequence length; every sequence generates 16 tokens
  RUNSPEC   KIND:THETA:COUNT[:verify]
            KIND   rand (uniformly random token ids) or wiki (consecutive S-token windows of the
                   WikiText-2 test split)
            THETA  index into zk.model.THETA (0: greedy; > 0: temperature sampling, proved by the
                   sampling gadget, with u derived from the statement's seed)
            COUNT  timed proofs of this statement; the first proof of the process is an extra
                   warm-up run, recorded with warmup=true
            verify also verify the first proof (GPU verifier) and record its time
Environment: SABLE_FO_MODEL (required), BENCH_OUT (JSON-lines output, default bench_e2e.jsonl),
BENCH_L (prove only the first BENCH_L layers; the per-layer work is identical across layers),
SABLE_HF_SNAPSHOT / SABLE_DATA for the wiki prompts (see scripts/paths.py), BENCH_SAVE (with verify: also
save the verified proof, its statement and the fixed-unit digests to this path prefix, for the host
verifier scripts/verify_cpu.py).

One JSON record per proof: shape, prompt kind, theta, warmup flag, prove time (s), proof size (MiB),
peak GPU memory allocated by PyTorch (GiB), peak resident host memory of the process (GiB, VmHWM,
reset before each proof), static lookups and their chunks, and the per-phase times of llm.LAST.
Linux only (host memory is read from /proc)."""
import copy
import json
import os
import sys
import time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from zk import model as mdl
from zk import llm, proofio
from scripts import paths

NGEN = 16


def rss_reset():
    """Reset the peak resident set size (VmHWM) of this process."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def rss_peak():
    """Peak resident set size since the last reset, GiB."""
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 2**20


def wiki_tokens(B, S):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    ids = tok(paths.wikitext2("test"))["input_ids"]
    return torch.tensor([ids[b * S:(b + 1) * S] for b in range(B)])


def record(out, rec):
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def save(path, M, sh, tokens, targets, pb, cw, theta, nc):
    """The verifier's inputs: statement and setup (C_W, the fixed-unit digests a verifier holds as
    constants) as a pickle, the proof pb (byte format, as serialized when the prover returned it)."""
    import pickle
    cpu = lambda x: x.cpu() if torch.is_tensor(x) else x
    pickle.dump(dict(model=paths.fo_model(), shape=(sh.B, sh.S, sh.cb, sh.cbv, NGEN), tokens=cpu(tokens),
                     targets=cpu(targets), cw=cw, theta=theta, nc=nc, tab_cache=M.__dict__.get("_tab_cache", {}),
                     L=M.cfg.L, stmt=llm.statement(M, sh, tokens, targets, cw, llm.Policy(theta=theta), nc),
                     cfg=dict(vars(M.cfg)), cd=mdl.const_digest(M, sh.S, theta)), open(path + ".pkl", "wb"))
    open(path + ".proof", "wb").write(pb)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    B, S = int(sys.argv[1]), int(sys.argv[2])
    specs = sys.argv[3:]
    out = os.environ.get("BENCH_OUT", "bench_e2e.jsonl")
    t0 = time.time()
    M = mdl.load_model(paths.fo_model(), layers_on="cpu")
    if os.environ.get("BENCH_L"):
        M = copy.copy(M)
        M.layers = M.layers[:int(os.environ["BENCH_L"])]
        M.cfg = copy.copy(M.cfg)
        M.cfg.L = len(M.layers)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), NGEN)
    t0 = time.time()
    cw = llm.weights_digest(M, sh)                  # C_W: computed once per model release
    print(f"C_W {time.time() - t0:.1f}s", flush=True)
    nc = bytes(range(32))                           # the statement's nonce
    first = True
    for spec in specs:
        parts = spec.split(":")
        kind, theta, count = parts[0], int(parts[1]), int(parts[2])
        do_verify = len(parts) > 3 and parts[3] == "verify"
        if kind == "rand":
            base = torch.randint(0, M.cfg.V, (B, S), generator=torch.Generator().manual_seed(0))
        elif kind == "wiki":
            base = wiki_tokens(B, S)
        else:
            raise SystemExit(f"unknown prompt kind {kind!r}")
        policy = llm.Policy(theta=theta)
        u = mdl.prf_u(llm.sampling_seed(base, NGEN, cw, policy, nc), B * NGEN) if theta else None
        t1 = time.time()
        tokens, targets = llm.decode(M, base.cuda(), NGEN, policy, u)
        torch.cuda.empty_cache()
        print(f"{spec}: decode {time.time() - t1:.0f}s", flush=True)
        for i in range(count + (1 if first else 0)):
            torch.cuda.reset_peak_memory_stats()
            rss_reset()
            t1 = time.time()
            msgs = llm.prove(M, sh, tokens, targets, C_W=cw, policy=policy, nc=nc)
            tp = time.time() - t1
            record(out, dict(B=B, S=S, kind=kind, theta=theta, warmup=first, prove=round(tp, 2),
                             mib=round(llm.LAST["bytes"] / 2**20, 2),
                             gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                             host_gib=round(rss_peak(), 2), lookups=int(llm.LAST["lookups"]),
                             chunks=int(llm.LAST["chunks"]),
                             times={k: round(v, 2) for k, v in llm.LAST["times"].items()}))
            first = False
            if do_verify and i == 0:
                pb = proofio.dumps(msgs) if os.environ.get("BENCH_SAVE") else None
                t1 = time.time()
                llm.verify(M, sh, tokens, targets, msgs, C_W=cw, policy=policy, nc=nc)
                record(out, dict(B=B, S=S, kind=kind, theta=theta, verify_gpu=round(time.time() - t1, 1)))
                if pb is not None:
                    if proofio.dumps(msgs) != pb:
                        print("WARNING: proof messages changed between prove and the end of verify", flush=True)
                    save(os.environ["BENCH_SAVE"], M, sh, tokens, targets, pb, cw, theta, nc)
            del msgs
    print("done", flush=True)


if __name__ == "__main__":
    main()
