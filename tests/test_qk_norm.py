"""F_O with QK-norm (Qwen3): calibration gives in-range shifts, the batched engine computes the same
function as model.forward / model.generate, and the attention inputs track the float QK-norm."""
import torch
from zk import model as mdl, evalfo
from tests.tiny import tiny_cfg3, tiny_get, tiny_model


def test_qk_norm():
    M = tiny_model(cfg=tiny_cfg3())
    for Ly in M.layers:
        assert int(Ly.kn.min()) >= 1 - 14 and int(Ly.kc.min()) >= 1
    ok, ref, got = evalfo.check(M, n=2, S=120, ngen=6)
    assert ok, (ref, got)
    print("engine matches model.forward / generate")
    # the normalized, gained queries of layer 0 against floating point
    cfg, get = M.cfg, tiny_get(tiny_cfg3(), 0)
    toks = torch.randint(0, cfg.V, (1, 64), generator=torch.Generator().manual_seed(3)).cuda()
    h = M.emb[toks.reshape(-1)].to(torch.int64)
    Ly = M.layers[0]
    q, _, _ = mdl.requant8(h)
    qp, Eh, _ = mdl.normscale(q)
    acc = mdl.imm(qp, Ly.Wqkv)
    out = mdl.qkv_out(cfg, Ly, acc, Eh)
    nq, nk, dh = cfg.nh * cfg.dh, cfg.nkv * cfg.dh, cfg.dh
    x = acc.double() * torch.pow(2.0, Ly.eWqkv.double()).unsqueeze(0) / torch.pow(2.0, Eh.double()).unsqueeze(-1)
    for name, a, b in (("q", 0, nq), ("k", nq, nq + nk)):
        rows = x[:, a:b].reshape(-1, dh)
        g = get(f"model.layers.0.self_attn.{name}_norm.weight").double()
        ref = rows / rows.pow(2).mean(dim=1, keepdim=True).sqrt() * g        # Qwen3 RMSNorm (eps negligible)
        got = (out[:, a:b].double() * torch.pow(2.0, Ly.eout[a:b].double())).reshape(-1, dh)
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"{name}: cosine {cos:.5f}, relative error {rel:.4f} against float RMSNorm x gain")
        assert cos > 0.999 and rel < 0.05


if __name__ == "__main__":
    test_qk_norm()
