"""Several KV groups: their softmax tile domains are proved as one merged domain."""
import torch
from zk import model as mdl
from zk import llm
from tests.tiny import tiny_get
from tests.test_llm import statement


def main():
    llm.TILE_MERGE_G = 8
    cfg = mdl.Cfg(L=2, d=256, nh=4, nkv=2, dh=128, dff=384, V=1024)
    M = mdl.quantize_weights(cfg, tiny_get(cfg, 3))
    mdl.calibrate(M, torch.randint(0, cfg.V, (1, 256), generator=torch.Generator().manual_seed(1)))
    B, S, ngen = 2, 256, 16
    sh = llm.Shp(M.cfg, B, S, 7, 7, ngen)
    tokens, targets = statement(M, B, S, ngen)
    pr = llm.Proto.__new__(llm.Proto)
    pr.sh = sh
    batches = llm.Proto.att_batches(pr)
    assert any(len(b) > 1 for b in batches), batches
    msgs = llm.prove(M, sh, tokens, targets)
    llm.verify(M, sh, tokens, targets, msgs)
    print("merged tile domains", batches, "verify ok")
    bad = targets.clone()
    bad[1, -1] = (bad[1, -1] + 1) % cfg.V
    try:
        llm.verify(M, sh, tokens, bad, msgs)
    except Exception as e:                     # noqa: BLE001
        print("tampered target rejected:", type(e).__name__)
    else:
        raise SystemExit("tampered target accepted")


def test_llm_groups():
    g = llm.TILE_MERGE_G
    try:
        main()
    finally:
        llm.TILE_MERGE_G = g


if __name__ == "__main__":
    main()
