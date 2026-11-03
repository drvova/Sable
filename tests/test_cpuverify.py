"""The host verifier (llm.verify(..., cpu=True): compiled core zk/cuda/vcore.cpp, no GPU) accepts an
honest proof of a toy model and rejects tampered ones; the verification runs in a subprocess with
no visible GPU."""
import os
import pickle
import subprocess
import sys
import tempfile
import numpy as np
import torch
from zk import model as mdl, llm, proofio
from tests.tiny import tiny_get
from tests.test_llm import statement

CFG = dict(L=2, d=256, nh=4, nkv=2, dh=128, dff=384, V=1024)
SH = (2, 256, 7, 7, 16)
if os.environ.get("CPUV_QKNORM"):            # Qwen3-shaped: QK-norm, two heads per column block
    CFG = dict(L=2, d=256, nh=4, nkv=2, dh=128, dff=512, V=1024, qk_norm=True, qkv_bias=False)
    SH = (2, 256, 8, 7, 16)


def _cpu(o):
    for k, v in list(vars(o).items()):
        if torch.is_tensor(v):
            setattr(o, k, v.cpu())


def prove(path):
    cfg = mdl.Cfg(**CFG)
    M = mdl.quantize_weights(cfg, tiny_get(cfg, 3))
    mdl.calibrate(M, torch.randint(0, cfg.V, (1, 256), generator=torch.Generator().manual_seed(1)))
    sh = llm.Shp(M.cfg, *SH)
    tokens, targets = statement(M, SH[0], SH[1], SH[4])
    cw = llm.weights_digest(M, sh)
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw)         # fills the fixed-unit digests (setup)
    _cpu(M)
    for Ly in M.layers:
        _cpu(Ly)
    # the verifier's own setup (model, statement) as a pickle; the prover's proof in the byte format
    pickle.dump(dict(M=M, tokens=tokens.cpu(), targets=targets.cpu() if torch.is_tensor(targets) else targets,
                     cw=cw), open(path, "wb"))
    open(path + ".proof", "wb").write(proofio.dumps(msgs))


def verify_child(path):
    """Runs without a GPU: 0 honest accepted and all tampered rejected."""
    d = pickle.load(open(path, "rb"))
    d["msgs"] = proofio.loads(open(path + ".proof", "rb").read())
    M = d["M"]
    sh = llm.Shp(M.cfg, *SH)
    llm.verify(M, sh, d["tokens"], d["targets"], d["msgs"], C_W=d["cw"], cpu=True)
    rng = np.random.default_rng(0)
    idx = [i for i, m in enumerate(d["msgs"]) if isinstance(m, np.ndarray)]
    for i in rng.choice(idx, 8, replace=False):
        msgs = list(d["msgs"])
        m = np.ascontiguousarray(msgs[i]).copy()
        m.reshape(-1)[rng.integers(m.size)] += 1
        msgs[i] = m
        try:
            llm.verify(M, sh, d["tokens"], d["targets"], msgs, C_W=d["cw"], cpu=True)
        except Exception:
            continue
        raise AssertionError(f"tampered message {i} accepted")
    print("host verifier ok")


def test_cpu_verify_qknorm():
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=dict(os.environ, CPUV_QKNORM="1"),
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0 and "host verifier ok" in r.stdout


def test_cpu_verify():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "proof.pkl")
        prove(path)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "child", path], env=env, cwd=root,
                           capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
        assert r.returncode == 0 and "host verifier ok" in r.stdout
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        verify_child(sys.argv[2])
    else:
        test_cpu_verify()
