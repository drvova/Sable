"""The serving engine (zk/fofast.py) equals evalfo.Engine, token for token, on tiny Qwen2, Qwen3 (QK-norm)
and Qwen3-MoE models, prefill and graph-replayed decode; a model whose rows leave F_O's domain (a
ShiftQ8S bound below every E_h) is flagged in the prefill and at every decode step."""
import torch
from zk import model as mdl, fofast
from tests.test_llm import qwen_tiny
from tests.test_llm_qwen3 import qwen3_tiny
from tests.test_llm_moe import moe_tiny


def main():
    models = {"qwen2": qwen_tiny(),
              "qwen3": qwen3_tiny(mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True,
                                          qkv_bias=False)),
              "moe": moe_tiny()}
    for name, M in models.items():
        for B in (2, 4):
            ok, ref, got = fofast.check(M, B=B, S=300, ngen=12)
            assert ok, (name, B, ref, got)
        print(f"{name}: engine equals evalfo (B = 2, 4), no row flagged")
    M = models["moe"]
    mdl.ensure_caps(M)
    M.layers[1].kqs = -8
    fe = fofast.FastEngine(M, 320, 2)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, M.cfg.V, (300 - 37 * i,), generator=g).tolist() for i in range(2)]
    _, _, _, fl = fe.generate(prompts, 8, flags=True)
    assert all(pf and all(st) for pf, st in fl), fl
    print("rows beyond the ShiftQ8S bound: flagged in the prefill and at every decode step")


if __name__ == "__main__":
    main()
