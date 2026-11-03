"""The Q / K exponent split of QK-norm models (model.qk_split): every group keeps e_Q + e_K = E (the one score
exponent, so the exp table and the bounds of the attention relations are unchanged) and a QK-norm shift
kn + E2 in [1, QKV_KMAX]; K is never coarser than under the old rule (all slack to K) and, on a model with
slack, finer in some group.  test_llm_qwen3 proves and verifies the tiny model built this way."""
import torch
from zk import model as mdl
from tests.tiny import tiny_cfg3, tiny_get

S = 256


def build(split=True):
    cfg = tiny_cfg3()
    get = tiny_get(cfg)

    def get2(name):                          # layer 0: one large K-norm gain, as in the Qwen3 checkpoints, sets E;
        w = get(name)                        # layer 1: small K gains, so its groups have slack
        if name.endswith("k_norm.weight"):
            w = w.clone()
            if ".0." in name:
                w[3] = 8.0
            else:
                w = w / 8
        return w
    M = mdl.quantize_weights(cfg, get2)
    keep = mdl.qk_split
    if not split:
        mdl.qk_split = lambda realb, E, eQ, ekmax, *a: ekmax
    try:
        mdl.calibrate(M, torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1)))
    finally:
        mdl.qk_split = keep
    return M


def test_qk_split():
    M, M0 = build(), build(split=False)
    cfg = M.cfg
    grp, finer = cfg.nh // cfg.nkv, 0
    for Ly, L0 in zip(M.layers, M0.layers):
        for g in range(cfg.nkv):
            for i in range(g * grp, (g + 1) * grp):
                assert Ly.eQ[i] + Ly.eK[g] == M.E_attn
            assert Ly.eK[g] <= L0.eK[g]
            finer += Ly.eK[g] < L0.eK[g]
        assert int(Ly.kn.min()) >= 1 and int(Ly.kn.max()) + mdl.e_hmax(cfg.dh) <= mdl.QKV_KMAX
    assert finer > 0
