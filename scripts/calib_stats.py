"""Per-layer calibration statistics of F_O (shift and exponent ranges, residual magnitude) on 512
tokens of the spec text, as recorded by model.layer_forward(calib=True).

usage: python -m scripts.calib_stats"""
import torch
from transformers import AutoTokenizer
from zk import model as mdl
from scripts import paths


def main():
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    M = mdl.load_model(paths.fo_model())
    text = open(paths.CALIB_CORPUS).read()[30000:60000]
    ids = torch.tensor(tok(text)["input_ids"][:512]).view(1, -1)
    B, S = ids.shape
    cos, sin = mdl.rope_tables(M.cfg, S)
    h = M.emb[ids.reshape(-1).cuda()].to(torch.int64)
    for li, Ly in enumerate(M.layers):
        mdl.STATS.clear()
        h, _, _ = mdl.layer_forward(M, Ly, h, B, S, cos, sin, calib=True)
        st = mdl.STATS
        print(li, "k_o", st["k_o"], "e_o", st["e_o"], "k_d", st["k_d"], "e_m", st["e_m"], "e_up", st["e_up"],
              "k_qkv", st["k_qkv"], "k_g", st["k_g"], "eWd", (int(Ly.eWd.min()), int(Ly.eWd.max())),
              "eWo", (int(Ly.eWo.min()), int(Ly.eWo.max())), "hmax", int(h.abs().max()),
              "hrms", float(h.double().pow(2).mean().sqrt()), flush=True)


if __name__ == "__main__":
    main()
