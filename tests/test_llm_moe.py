"""End-to-end prove / verify of a tiny Qwen3-MoE-shaped model: router, top-k, slots, experts of a width
that is not a power of two (3 blocks), gathered products; the prover's MoE output is F_O's."""
import torch
from zk import model as mdl, llm
from tests.tiny import tiny_get
from tests.test_llm import statement

CFG = dict(L=2, d=256, nh=2, nkv=1, dh=128, dff=4 * 96, V=1024, qk_norm=True, qkv_bias=False,
           moe=dict(E=8, k=4, de=96))


def moe_tiny(seed=0, S=256):
    cfg = mdl.Cfg(**CFG)
    get = tiny_get(cfg, seed)
    M = mdl.quantize_weights(cfg, get)
    mdl.calibrate(M, torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1)))
    mdl.fit_gate_exponents(M, get)
    return M


def test_moe_prove(B=2, S=256, ngen=16):
    M = moe_tiny()
    sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), ngen)
    tokens, targets = statement(M, B, S, ngen)
    # the prover's MoE output is F_O's
    mdl.ensure_caps(M)
    hin = M.emb[tokens.reshape(-1)].to(torch.int64)
    W = llm.LayerWit(M, sh, 0, hin)
    y, _, _ = mdl.mlp_out(M, M.layers[0], W.n2["qp"].to(torch.int64), W.n2["Eh"].to(torch.int64))
    assert torch.equal(mdl.gshiftround(W.accd.to(torch.int64), W.kd.to(torch.int64).unsqueeze(-1)), y)
    msgs = llm.prove(M, sh, tokens, targets)
    llm.verify(M, sh, tokens, targets, msgs)
    print(f"MoE proof of {llm.LAST['bytes'] / 2**20:.1f} MiB verifies")
    bad = targets.clone()
    bad[0, ngen - 1] = (bad[0, ngen - 1] + 1) % M.cfg.V
    try:
        llm.verify(M, sh, tokens, bad, msgs)
    except Exception as e:                         # noqa: BLE001
        print("tampered target rejected:", type(e).__name__)
    else:
        raise AssertionError("tampered target accepted")


if __name__ == "__main__":
    test_moe_prove()


def _forged(edit, label):
    from zk import moe
    M = moe_tiny()
    sh = llm.Shp(M.cfg, 2, 256, *llm.block_bits(M.cfg), 16)
    tokens, targets = statement(M, 2, 256, 16)
    orig = moe.moe_wit

    def forged(M_, Ly, qp, Eh, geo):
        w = orig(M_, Ly, qp, Eh, geo)
        if Ly is M_.layers[1]:
            edit(w, geo)
        return w
    moe.moe_wit = forged
    try:
        msgs = llm.prove(M, sh, tokens, targets)
        llm.verify(M, sh, tokens, targets, msgs)
    except Exception as e:                         # noqa: BLE001
        print(f"{label} rejected:", type(e).__name__, str(e)[:90])
    else:
        raise AssertionError(f"{label} accepted")
    finally:
        moe.moe_wit = orig


def test_moe_forged_weight():
    def edit(w, geo):
        w["w"] = w["w"].clone()
        w["w"][3, 1] += 1
    _forged(edit, "router weight off by one")


def test_moe_forged_expert():
    def edit(w, geo):
        w["idx"] = w["idx"].clone()
        t, s = 5, geo.k - 1
        used = set(w["idx"][t].tolist())
        w["idx"][t, s] = next(e for e in range(geo.E) if e not in used)
    _forged(edit, "slot routed to an unselected expert")
