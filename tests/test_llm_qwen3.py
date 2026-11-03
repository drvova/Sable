"""End-to-end prove / verify of a tiny Qwen3-shaped model (QK-norm, no QKV biases): the prover's
witness equals F_O, the proof verifies, and a tampered target is rejected."""
import sys
import torch
from zk import model as mdl, llm
from tests.tiny import tiny_get
from tests.test_llm import statement


def qwen3_tiny(cfg, S=256, seed=0):
    M = mdl.quantize_weights(cfg, tiny_get(cfg, seed))
    mdl.calibrate(M, torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1)))
    return M


def run(cfg, cb, B=2, S=256, ngen=16):
    M = qwen3_tiny(cfg)
    sh = llm.Shp(M.cfg, B, S, cb, 7, ngen)
    tokens, targets = statement(M, B, S, ngen)
    # the prover's Q / K / V witness is F_O's
    l = 0
    hin = M.emb[tokens.reshape(-1)].to(torch.int64)
    mdl.ensure_caps(M)
    W = llm.LayerWit(M, sh, l, hin)
    q, _, _ = mdl.requant8(hin)
    qp, Eh, _ = mdl.normscale(q)
    ref = mdl.qkv_out(M.cfg, M.layers[l], mdl.imm(qp, M.layers[l].Wqkv), Eh)
    assert torch.equal(W.qkv.to(torch.int64), ref), "prover QKV witness differs from F_O"
    msgs = llm.prove(M, sh, tokens, targets)
    llm.verify(M, sh, tokens, targets, msgs)
    print(f"cb={cb}: proof of {llm.LAST['bytes'] / 2**20:.1f} MiB verifies")
    bad = targets.clone()
    bad[0, ngen - 1] = (bad[0, ngen - 1] + 1) % M.cfg.V
    try:
        llm.verify(M, sh, tokens, bad, msgs)
    except Exception as e:                         # noqa: BLE001
        print("tampered target rejected:", type(e).__name__)
    else:
        raise AssertionError("tampered target accepted")


def test_qwen3_cb7():
    run(mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True, qkv_bias=False), 7)


def test_qwen3_cb8():
    """two heads per column block (hib bits in the head rows)"""
    run(mdl.Cfg(L=2, d=256, nh=4, nkv=2, dh=128, dff=512, V=1024, qk_norm=True, qkv_bias=False), 8)


if __name__ == "__main__":
    test_qwen3_cb7() if len(sys.argv) < 2 or sys.argv[1] == "7" else test_qwen3_cb8()


def test_qwen3_forged_q8():
    """A prover whose q8 (ShiftQ8 of a head row) is off by one in one entry is rejected."""
    cfg = mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True, qkv_bias=False)
    M = qwen3_tiny(cfg)
    sh = llm.Shp(M.cfg, 2, 256, 7, 7, 16)
    tokens, targets = statement(M, 2, 256, 16)
    orig = llm.LayerWit.qk_norm_wit

    def forged(self, Ly, sh_):
        orig(self, Ly, sh_)
        if self.l == 1:
            Q = self.qn["qn"]
            Q["q8"] = Q["q8"].clone()
            Q["q8"][5, 3] += 1
    llm.LayerWit.qk_norm_wit = forged
    try:
        msgs = llm.prove(M, sh, tokens, targets)
        llm.verify(M, sh, tokens, targets, msgs)
    except Exception as e:                         # noqa: BLE001
        print("forged q8 rejected:", type(e).__name__, str(e)[:80])
    else:
        raise AssertionError("forged q8 accepted")
    finally:
        llm.LayerWit.qk_norm_wit = orig
