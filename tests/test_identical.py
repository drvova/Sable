"""The device transcript and the graph-replayed prover produce byte-identical proofs to the host
transcript and the plain prover.  Each configuration proves the same statement of a 4-layer toy model
(graph replay needs at least 3 layers) in its own process, since the switches are read at import:
  default: GPU-resident transcript session, middle layers replayed from one captured CUDA graph;
  ZK_GRAPH=0: every layer proved directly;
  host: ZK_DEVSESSION=0 ZK_HOSTTR=1 ZK_GRAPH=0, the sumcheck transcript on the host (the openings
  always run on the device transcript)."""
import hashlib
import os
import subprocess
import sys
import torch
from zk import model as mdl, llm, proofio
from tests.tiny import tiny_get
from tests.test_llm import statement

CONFIGS = {
    "default": {},
    "no graph": dict(ZK_GRAPH="0"),
    "host transcript": dict(ZK_DEVSESSION="0", ZK_HOSTTR="1", ZK_GRAPH="0"),
}


def child():
    B, S, ngen = 2, 256, 16
    if os.environ.get("IDENT_MOE"):             # Qwen3-MoE-shaped (tests/test_llm_moe.py), four layers
        from tests import test_llm_moe
        test_llm_moe.CFG["L"] = 4
        M = test_llm_moe.moe_tiny()
        sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), ngen)
    else:
        if os.environ.get("IDENT_QKNORM"):      # Qwen3-shaped: QK-norm, two heads per column block
            cfg = mdl.Cfg(L=4, d=256, nh=4, nkv=2, dh=128, dff=512, V=1024, qk_norm=True, qkv_bias=False)
            cb = 8
        else:
            cfg = mdl.Cfg(L=4, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024)
            cb = 7
        M = mdl.quantize_weights(cfg, tiny_get(cfg, 0))
        mdl.calibrate(M, torch.randint(0, cfg.V, (1, 256), generator=torch.Generator().manual_seed(1)))
        sh = llm.Shp(M.cfg, B, S, cb, 7, ngen)
    tokens, targets = statement(M, B, S, ngen)
    msgs = llm.prove(M, sh, tokens, targets)
    llm.verify(M, sh, tokens, targets, msgs)
    print("PROOF", hashlib.sha256(proofio.dumps(msgs)).hexdigest(), flush=True)


def test_identical_proofs(qknorm=False, moe=False):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digests = {}
    for name, extra in CONFIGS.items():
        env = {k: v for k, v in os.environ.items() if k not in ("ZK_GRAPH", "ZK_DEVSESSION", "ZK_HOSTTR")}
        if qknorm:
            env["IDENT_QKNORM"] = "1"
        if moe:
            env["IDENT_MOE"] = "1"
        env.update(extra, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
        r = subprocess.run([sys.executable, "-m", "tests.test_identical", "child"], env=env, cwd=root,
                           capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PROOF ")]
        assert r.returncode == 0 and lines, (name, r.stdout[-2000:], r.stderr[-2000:])
        digests[name] = lines[-1].split()[1]
        print(f"{name:16s} {digests[name]}", flush=True)
    assert len(set(digests.values())) == 1, digests
    print("proofs byte-identical")


def test_identical_proofs_qknorm():
    test_identical_proofs(qknorm=True)


def test_identical_proofs_moe():
    test_identical_proofs(moe=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child()
    else:
        test_identical_proofs()
