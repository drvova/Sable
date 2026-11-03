"""Quantize the checkpoint of SABLE_ARCH (default Qwen2.5-32B-Instruct) into F_O and save the integer model (deterministic: two runs give
bit-identical files).  Calibration uses the first 1,024 tokens of docs/spec/corrected.tex, widened to every
vocabulary token at position 0 (first_token_calibrate).

usage: python -m scripts.quantize [OUT]        (OUT defaults to $SABLE_FO_MODEL)
Needs the checkpoint (SABLE_HF_SNAPSHOT or the Hugging Face cache, see scripts/paths.py)."""
import json
import math
import os
import sys
import time
import torch
from safetensors import safe_open
from zk import model as mdl
from scripts import paths


def checkpoint_reader(snap):
    """get(name) -> the checkpoint tensor `name` on the GPU."""
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(snap, f), framework="pt", device="cuda")
        return handles[f].get_tensor(name)
    return get


def first_token_calibrate(M, get, ids, margin=1.0):
    """Calibrate on ids, then widen the ranges to every vocabulary token at position 0 (a prompt's first
    row attends only to itself, so the sweep is exhaustive there; some first tokens carry a much larger
    activation than any row of the text): e_G, e_A cover the largest gate input and SiLU output, and the
    residual's fractional bits f_h are lowered until every GShiftRound output lies in A, each with the
    margin (none by default: the sweep is exhaustive at position 0, and every lost bit of f_h costs the
    rows of smallest norm).  Repeated until the sweep needs no change (clamping hides what lies past a
    range)."""
    from zk import evalfo
    eG = eA = -64
    while True:
        mdl.calibrate(M, ids)
        M.eG, M.eA = max(M.eG, eG), max(M.eA, eA)
        for Ly in M.layers:
            Ly.kcg = M.eG - (Ly.eWge if M.cfg.moe else Ly.eWg)
        mdl.drop_caps(M)
        env = evalfo.first_token_envelope(M)
        print("first-token envelope", {k: round(v, 4) for k, v in env.items()}, "f_h", M.f_h, "eG", M.eG,
              "eA", M.eA, flush=True)
        eG = max(M.eG, math.ceil(math.log2(env["gate"] * margin / mdl.SILU_CMAX)))
        eA = max(M.eA, math.ceil(math.log2(env["silu"] * margin / 32767)))
        f_h = M.f_h - max(0, math.ceil(math.log2(env["gsr"] * margin)))
        if (eG, eA, f_h) == (M.eG, M.eA, M.f_h):
            return
        if f_h != M.f_h:
            mdl.set_f_h(M, get, f_h)


def quantize(calib_ids, out):
    """Quantize the weights, calibrate on calib_ids(tokenizer) -> (B, S) token ids, refit the QKV
    exponents and save the model to out."""
    from transformers import AutoTokenizer
    snap = paths.hf_snapshot()
    get = checkpoint_reader(snap)
    t0 = time.time()
    M = mdl.quantize_weights(mdl.ARCHS[paths.HF_REPO], get)
    print("weights quantized", time.time() - t0, flush=True)
    tok = AutoTokenizer.from_pretrained(snap)
    first_token_calibrate(M, get, calib_ids(tok))
    print("calibrated", time.time() - t0, "f_h", M.f_h, "E_attn", M.E_attn, "eG", M.eG, "eA", M.eA, flush=True)
    n = mdl.fit_qkv_exponents(M, get)            # every input provable: K_qs >= e_hmax(d) in every layer
    print("QKV columns refitted", n, "K_qs", sorted({mdl.qkv_kcap(M.cfg, Ly) for Ly in M.layers}), flush=True)
    print("gate columns (dense) or experts (MoE) refitted", mdl.fit_gate_exponents(M, get), flush=True)
    if M.cfg.qk_norm:                            # the QK-norm shift kn + E2 never exceeds QKV_KMAX
        top = max(int(Ly.kn.max()) for Ly in M.layers) + mdl.e_hmax(M.cfg.dh)
        print("QK-norm: largest kn + e_hmax(dh)", top, flush=True)
        assert top <= mdl.QKV_KMAX
    state = dict(cfg=M.cfg.__dict__, f_h=M.f_h, emb=M.emb.cpu(), Wlm=M.Wlm.cpu(), eWlm=M.eWlm, Kz=M.Kz,
                 E_attn=M.E_attn, eG=M.eG, eA=M.eA, layers=[])
    for Ly in M.layers:
        state["layers"].append({k: v.cpu() if torch.is_tensor(v) else v for k, v in Ly.__dict__.items()})
    torch.save(state, out)
    print("saved", out, time.time() - t0, flush=True)


def spec_ids(tok):
    """The calibration batch of the paper's F_O: the first 1,024 tokens of the spec text."""
    text = open(paths.CALIB_CORPUS).read()[:20000]
    return torch.tensor(tok(text)["input_ids"][:1024]).view(1, -1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else paths.fo_model(must_exist=False)
    quantize(spec_ids, out)


if __name__ == "__main__":
    main()
