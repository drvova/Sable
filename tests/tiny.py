"""A tiny random Qwen2-shaped model for tests."""
import torch
from zk import model as mdl


def tiny_cfg():
    return mdl.Cfg(L=2, d=256, nh=4, nkv=2, dh=64, dff=512, V=1024)


def tiny_get(cfg, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = {}

    def r(*shape, s=0.05):
        return (torch.randn(*shape, generator=g) * s).cuda()
    W["model.embed_tokens.weight"] = r(cfg.V, cfg.d, s=1.0)
    for li in range(cfg.L):
        p = f"model.layers.{li}."
        W[p + "input_layernorm.weight"] = 1 + r(cfg.d, s=0.1)
        W[p + "post_attention_layernorm.weight"] = 1 + r(cfg.d, s=0.1)
        W[p + "self_attn.q_proj.weight"] = r(cfg.nh * cfg.dh, cfg.d)
        W[p + "self_attn.k_proj.weight"] = r(cfg.nkv * cfg.dh, cfg.d)
        W[p + "self_attn.v_proj.weight"] = r(cfg.nkv * cfg.dh, cfg.d)
        W[p + "self_attn.q_proj.bias"] = r(cfg.nh * cfg.dh, s=0.5)
        W[p + "self_attn.k_proj.bias"] = r(cfg.nkv * cfg.dh, s=0.5)
        W[p + "self_attn.v_proj.bias"] = r(cfg.nkv * cfg.dh, s=0.1)
        W[p + "self_attn.o_proj.weight"] = r(cfg.d, cfg.nh * cfg.dh)
        if cfg.moe:
            W[p + "mlp.gate.weight"] = r(cfg.moe["E"], cfg.d, s=0.2)
            for e in range(cfg.moe["E"]):
                x = p + f"mlp.experts.{e}."
                W[x + "gate_proj.weight"] = r(cfg.moe["de"], cfg.d)
                W[x + "up_proj.weight"] = r(cfg.moe["de"], cfg.d)
                W[x + "down_proj.weight"] = r(cfg.d, cfg.moe["de"])
        else:
            W[p + "mlp.gate_proj.weight"] = r(cfg.dff, cfg.d)
            W[p + "mlp.up_proj.weight"] = r(cfg.dff, cfg.d)
            W[p + "mlp.down_proj.weight"] = r(cfg.d, cfg.dff)
        if cfg.qk_norm:
            W[p + "self_attn.q_norm.weight"] = 1 + r(cfg.dh, s=0.3)
            W[p + "self_attn.k_norm.weight"] = 1 + r(cfg.dh, s=0.3)
    W["model.norm.weight"] = 1 + r(cfg.d, s=0.1)
    W["lm_head.weight"] = r(cfg.V, cfg.d)
    return lambda name: W[name]


def tiny_cfg3():
    """Qwen3-shaped: QK-norm, no QKV biases."""
    return mdl.Cfg(L=2, d=256, nh=4, nkv=2, dh=64, dff=512, V=1024, qk_norm=True, qkv_bias=False)


def tiny_cfg_moe():
    """Qwen3-MoE-shaped: QK-norm, no QKV biases, 8 experts of width 64, 2 active."""
    return mdl.Cfg(L=2, d=512, nh=4, nkv=2, dh=64, dff=2 * 64, V=1024, qk_norm=True, qkv_bias=False,
                   moe=dict(E=8, k=2, de=64))


def tiny_model(S=64, seed=0, cfg=None):
    cfg = cfg or tiny_cfg()
    M = mdl.quantize_weights(cfg, tiny_get(cfg, seed))
    toks = torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1))
    mdl.calibrate(M, toks)
    return M
