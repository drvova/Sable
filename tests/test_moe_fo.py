"""F_O with an MoE MLP (Qwen3-MoE): the batched engine computes the same function as model.forward /
model.generate, and the integer MoE tracks a floating-point MoE with the same (dequantized) weights."""
import torch
from zk import model as mdl, evalfo
from tests.tiny import tiny_cfg_moe, tiny_model


def float_moe(M, Ly, qp, Eh):
    """Qwen3-MoE in float64 on the normalized rows qp 2^-Eh, with the int8 weights dequantized."""
    k = M.cfg.moe["k"]
    x = qp.double() * torch.pow(2.0, -Eh.double()).unsqueeze(-1)
    z = x @ (Ly.Wr.double() * 2.0 ** Ly.eWr)
    top = torch.topk(z, k, dim=-1)
    w = torch.softmax(top.values, dim=-1)
    y = torch.zeros((x.shape[0], M.cfg.d), dtype=torch.float64, device=x.device)
    for s in range(k):
        for e in top.indices[:, s].unique().tolist():
            rows = (top.indices[:, s] == e).nonzero().flatten()
            g = x[rows] @ (Ly.Wge[e].double() * torch.pow(2.0, Ly.eWge[e].double()).unsqueeze(0))
            u = x[rows] @ (Ly.Wue[e].double() * 2.0 ** int(Ly.eWue[e]))
            y[rows] += w[rows, s].unsqueeze(-1) * ((g * torch.sigmoid(g) * u) @ (Ly.Wde[e].double() * 2.0 ** int(Ly.eWde[e])))
    return y * 2.0 ** M.f_h, top.indices


def test_moe_fo():
    M = tiny_model(cfg=tiny_cfg_moe())
    ok, ref, got = evalfo.check(M, n=2, S=120, ngen=6)
    assert ok, (ref, got)
    print("engine matches model.forward / generate")
    cfg = M.cfg
    toks = torch.randint(0, cfg.V, (1, 96), generator=torch.Generator().manual_seed(5)).cuda()
    h = M.emb[toks.reshape(-1)].to(torch.int64)
    Ly = M.layers[0]
    q, _, _ = mdl.requant8(h)
    qp, Eh, _ = mdl.normscale(q)
    y, _, _ = mdl.mlp_out(M, Ly, qp, Eh)
    yf, top = float_moe(M, Ly, qp, Eh)
    idx, w, _ = mdl.moe_route(M, Ly, qp, Eh)
    same = (idx.sort(dim=1).values == top.sort(dim=1).values).all(dim=1).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(y.double().flatten(), yf.flatten(), dim=0).item()
    rel = ((y.double() - yf).norm() / yf.norm()).item()
    print(f"routing equal to float on {same:.1%} of tokens; MoE output cosine {cos:.5f}, relative error {rel:.4f}")
    assert same > 0.9 and cos > 0.995


if __name__ == "__main__":
    test_moe_fo()
