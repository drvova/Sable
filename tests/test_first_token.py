"""First-token calibration (scripts/quantize.first_token_calibrate).  A tiny model with a planted
'massive activation': token T at position 0 drives one MLP channel far past every value seen on the
calibration text (as "Question" does in layer 2 of Qwen3-32B).  Calibrated on text alone, T as a first
token leaves F_O's domain (SiLU input past SILU_CMAX, MLP output past A); with the first-token sweep,
f_h, e_G, e_A widen, no vocabulary token at position 0 leaves the domain, the serving engine still
equals evalfo, and a proof of a prompt starting with T verifies.  The gate refit keeps the gate shift in range for every row after e_G widens."""
import torch
from zk import model as mdl, evalfo, fofast, llm
from scripts.quantize import first_token_calibrate
from tests.tiny import tiny_get

T = 7


def planted(cfg, seed=0):
    get0 = tiny_get(cfg, seed)
    g = torch.Generator(device="cpu").manual_seed(5)
    u = torch.randn(cfg.d, generator=g)
    u = (u / u.norm()).cuda()
    W = {}
    emb = get0("model.embed_tokens.weight").clone()
    emb[T] = 4 * u
    W["model.embed_tokens.weight"] = emb
    p = "model.layers.0.mlp."
    for n, a in (("gate_proj", 4.0), ("up_proj", 4.0)):
        w = get0(p + n + ".weight").clone()
        w[0] = a * u
        W[p + n + ".weight"] = w
    w = get0(p + "down_proj.weight").clone()
    w[:, 0] = 200 * u
    W[p + "down_proj.weight"] = w
    return lambda name: W[name] if name in W else get0(name)


def text(cfg, S=256):
    toks = torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1))
    return torch.where(toks == T, T + 1, toks)             # the calibration text never contains T


def main():
    cfg = mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024)
    get = planted(cfg)
    M = mdl.quantize_weights(cfg, get)
    mdl.calibrate(M, text(cfg))
    bad = fofast.first_token_flags(M, chunk=512)
    env = evalfo.first_token_envelope(M)
    print("text calibration: f_h", M.f_h, "eG", M.eG, "eA", M.eA, "envelope", env, "flagged first tokens", bad)
    assert T in bad and env["gsr"] > 1, (bad, env)

    M = mdl.quantize_weights(cfg, get)
    first_token_calibrate(M, get, text(cfg))
    mdl.fit_qkv_exponents(M)
    mdl.fit_gate_exponents(M, get)
    assert all(mdl.gate_kcap(M, Ly) >= mdl.e_hmax(cfg.d) for Ly in M.layers)   # the gate shift never binds
    bad = fofast.first_token_flags(M, chunk=512)
    print("first-token calibration: f_h", M.f_h, "eG", M.eG, "eA", M.eA, "flagged first tokens", bad)
    assert M.f_h < 10 and bad == [], bad
    ok, ref, got = fofast.check(M, B=2, S=300, ngen=12)
    assert ok, (ref, got)
    print("serving engine equals evalfo, no row flagged")

    tokens = torch.randint(0, cfg.V, (2, 256), generator=torch.Generator().manual_seed(2)).cuda()
    tokens[:, 0] = T
    ngen = 16
    tokens, targets = llm.greedy_decode(M, tokens, ngen)
    sh = llm.Shp(cfg, 2, 256, 7, 7, ngen)
    msgs = llm.prove(M, sh, tokens, targets)
    llm.verify(M, sh, tokens, targets, msgs)
    print("proof of prompts starting with T verifies")


if __name__ == "__main__":
    main()
