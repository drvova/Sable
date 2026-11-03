"""WikiText-2 accuracy over the full test split: the reference variants and F_O with single steps made
more precise (the accuracy attribution of the paper's evaluation; the "+" variants are measurement
devices, not provable functions).

usage: python -m scripts.acc_attribution OUTDIR NAME [NWIN]
  NAME: bf16 | fp8 | sq8 (as in scripts/acc.py) | fo | fo+FLAG[+FLAG...], FLAG in
    P:    attention probabilities and the softmax normalization in float64 (instead of 10-bit P and
          16-bit denominator mantissas)
    QK:   queries and keys with 4 more bits (12-bit ShiftQ8S output, RoPE kept at that precision)
    SILU: SiLU outputs with 4 more fractional bits
    UP:   MLP up projection output with 8 more bits (16-bit instead of ShiftQ8's 8-bit)
  NWIN: only the first NWIN windows of 2,048 tokens (default: all, about 300K tokens)
F_O is read from SABLE_FO_MODEL (e.g. point it at a model from scripts/quantize_calib.py and use a
NAME such as fo_calwt).  Writes OUTDIR/NAME.json (per-window NLL, perplexity, top-1) and
OUTDIR/NAME.npy (greedy predictions, for the agreement column of scripts/acc_summary.py)."""
import json
import math
import os
import sys
import time
import numpy as np
import torch
from scripts import acc as sa
from scripts import paths
from zk import model as mdl, evalfo
from zk.model import (requant8, normscale, imm, shiftq8s, rope8, gshiftround, satadd, shiftround,
                      silu_values, shiftq8, norm_den, norm_div, fdiv, pow2, KEYBLK)

if __name__ == "__main__" and len(sys.argv) < 3:
    raise SystemExit(__doc__)
OUTD, NAME = sys.argv[1], sys.argv[2]
NWIN = int(sys.argv[3]) if len(sys.argv) > 3 else None
FLAGS = set(NAME.split("+")[1:]) if NAME.startswith("fo") else set()


class Eng(evalfo.Engine):
    def _attend(self, Ly, rQ, rK, V, qpos, texp, kmax):
        if not (FLAGS & {"P", "QK"}):
            return super()._attend(Ly, rQ, rK, V, qpos, texp, kmax)
        cfg = self.M.cfg
        nh, G, dh = cfg.nh, cfg.nkv, cfg.dh
        grp = nh // G
        B, Sq = rQ.shape[:2]
        Sk = rK.shape[1]
        sc = 2.0 ** (-8 if "QK" in FLAGS else 0)          # QK: scores are 2^8 times larger
        alpha = (2.0 ** Ly.E) / math.sqrt(dh) * sc
        out = torch.empty((B, Sq, nh, dh), dtype=torch.int64, device="cuda")
        jj = torch.arange(Sk, device="cuda")
        vsh = Ly.vshift.view(G, grp)
        step = max(1, int(1e8) // max(1, G * grp * Sq * Sk))
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            nb = b1 - b0
            q = rQ[b0:b1].view(nb, Sq, G, grp, dh).permute(0, 2, 3, 1, 4).reshape(nb * G, grp * Sq, dh)
            k = rK[b0:b1].permute(0, 2, 1, 3).reshape(nb * G, Sk, dh)
            acc = torch.bmm(q.double(), k.double().transpose(1, 2))
            pos = qpos[b0:b1].view(nb, 1, 1, Sq).expand(nb, G, grp, Sq).reshape(nb * G, grp * Sq)
            mask = jj.view(1, 1, -1) <= pos.unsqueeze(-1)
            Mx = torch.where(mask, acc, torch.full_like(acc, -1e30)).max(dim=2).values
            x = (Mx.unsqueeze(-1) - acc).clamp(min=0)
            del acc
            Vf = V[b0:b1].permute(0, 2, 1, 3).reshape(nb * G, Sk, dh).double()
            vs = vsh.view(1, G, grp, 1).expand(nb, G, grp, Sq).reshape(nb * G, grp * Sq, 1)
            if "P" in FLAGS:
                Pf = torch.where(mask, torch.exp(-x * alpha), torch.zeros_like(x))
                o = torch.bmm(Pf, Vf) / Pf.sum(dim=2, keepdim=True)
                o = torch.round(o * (2.0 ** 32) * torch.pow(2.0, -(mdl.O_BASE + vs).double())).to(torch.int64)
            else:
                Pm = torch.where(mask, torch.round((1 << mdl.EXP_BITS) * torch.exp(-x * alpha)),
                                 torch.zeros_like(x)).to(torch.int64)
                nd = norm_den(Pm.sum(dim=2).reshape(-1))
                km = nd["km"].view(nb * G, grp * Sq, 1)
                Ns = torch.zeros((nb * G, grp * Sq, dh), dtype=torch.int64, device="cuda")
                for kb in range((kmax + (1 << KEYBLK) - 1) >> KEYBLK):
                    j0, j1 = kb << KEYBLK, min(Sk, (kb + 1) << KEYBLK)
                    pv = torch.round(torch.bmm(Pm[:, :, j0:j1].double(), Vf[:, j0:j1])).to(torch.int64)
                    Ns += shiftround(pv, km)
                o, _ = norm_div(Ns.view(-1, dh), nd, vs.reshape(-1))
                o = o.view(nb * G, grp * Sq, dh)
            del x, mask
            out[b0:b1] = o.view(nb, G, grp, Sq, dh).permute(0, 3, 1, 2, 4).reshape(nb, Sq, nh, dh)
        return out

    def _layer(self, Ly, h, B, S, pos, cache, wpos, kmax):
        M, cfg = self.M, self.M.cfg
        nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        half = dh // 2
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        acc = imm(qp, Ly.Wqkv)
        k = Ly.kc.unsqueeze(0) + Eh.unsqueeze(-1)
        qkv, _ = shiftq8s(acc, k, Ly.bias.unsqueeze(0))
        Vh = qkv[:, nq + nk:].reshape(B, S, nkv, dh)
        if "QK" in FLAGS:                       # 12-bit Q, K: shift by 4 less, saturate at 16 * 127
            k4 = k[:, :nq + nk] - 4
            s = pow2(k4)
            r = fdiv(acc[:, :nq + nk] + s // 2, s) + 16 * Ly.bias[:nq + nk].unsqueeze(0)
            qk = r.clamp(-16 * 127, 16 * 127)
            Qh = qk[:, :nq].reshape(B, S, nh, dh)
            Kh = qk[:, nq:].reshape(B, S, nkv, dh)
        else:
            Qh = qkv[:, :nq].reshape(B, S, nh, dh)
            Kh = qkv[:, nq:nq + nk].reshape(B, S, nkv, dh)
        c_ = self.cos[pos].view(B, S, 1, half)
        s_ = self.sin[pos].view(B, S, 1, half)
        rq1, rq2 = rope8(Qh[..., :half], Qh[..., half:], c_, s_)
        rk1, rk2 = rope8(Kh[..., :half], Kh[..., half:], c_, s_)
        rQ = torch.cat([rq1, rq2], dim=-1)
        rK = torch.cat([rk1, rk2], dim=-1)
        bi = torch.arange(B, device="cuda").view(B, 1).expand(B, S)
        cache["rK"][bi, wpos] = rK.to(cache["rK"].dtype)
        cache["V"][bi, wpos] = Vh.to(torch.int8)
        o = self._attend(Ly, rQ, cache["rK"][:, :kmax], cache["V"][:, :kmax], pos, mdl.exp_table(Ly.E, dh), kmax)
        o = o.reshape(B * S, nq)
        qo, eo, _ = requant8(o)
        acc = imm(qo, Ly.Wo)
        h, _ = satadd(h, gshiftround(acc, (Ly.Kco - eo).unsqueeze(-1)))
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        kg = (M.eG - Ly.eWg).unsqueeze(0) + Eh.unsqueeze(-1)
        c = shiftround(imm(qp, Ly.Wg), kg)
        extra = 0
        if "SILU" in FLAGS:
            x = c.double() * (2.0 ** M.eG)
            a = torch.round(x / (1 + torch.exp(-x)) * (2.0 ** (4 - M.eA))).to(torch.int64)
            extra += 4
        else:
            a = silu_values(c, M.eG, M.eA)
        accu = imm(qp, Ly.Wu)
        qu, eup, _ = shiftq8(accu)
        if "UP" in FLAGS:                       # 8 more bits of the up projection
            k8 = (eup - 8).unsqueeze(-1)
            kk = k8.clamp(min=0)
            qu = torch.where(k8 > 0, fdiv(accu + (pow2(kk) >> 1), pow2(kk)), accu * pow2((-k8).clamp(min=0)))
            extra += 8
        qm, em, _ = requant8(a * qu, 0, 17 + extra)
        Kcd = -(Ly.eWu + M.eA + Ly.eWd + M.f_h)
        kd = (Kcd - (em - extra + eup - Eh)).unsqueeze(-1)
        h, _ = satadd(h, gshiftround(imm(qm, Ly.Wd), kd))
        return h

    def new_caches(self, B, L):
        cfg = self.M.cfg
        dt = torch.int32 if "QK" in FLAGS else torch.int16
        return [dict(rK=torch.zeros((B, L, cfg.nkv, cfg.dh), dtype=dt, device="cuda"),
                     V=torch.zeros((B, L, cfg.nkv, cfg.dh), dtype=torch.int8, device="cuda"))
                for _ in self.M.layers]


class FOm:
    def __init__(self):
        self.M = mdl.load_model(paths.fo_model())
        self.eng = Eng(self.M, 4096)

    def logits_all(self, ids):
        caches = self.eng.new_caches(1, len(ids))
        h = self.eng.prefill(torch.tensor([ids]).cuda(), caches)[0]
        out = []
        for i in range(0, h.shape[0], 512):
            out.append(self.eng.logits(h[i:i + 512]).to(torch.float32) * (2.0 ** mdl.EZ))
        return torch.cat(out)


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUTD, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    sa.MAXTOK = 10 ** 9                      # the whole test split
    wins = sa.wiki_ids(tok)[:NWIN]
    m = FOm() if NAME.startswith("fo") else sa.HFModel(NAME, tok)
    nll, n, ok, preds, wn = 0.0, 0, 0, [], []
    t0 = time.time()
    for w in wins:
        with torch.no_grad():
            lg = m.logits_all(w)[:-1]
        tgt = torch.tensor(w[1:], device=lg.device)
        lp = torch.log_softmax(lg, dim=-1)
        x = float(-lp.gather(1, tgt.view(-1, 1)).sum())
        nll += x
        wn.append(x)
        pr = lg.argmax(dim=-1)
        ok += int((pr == tgt).sum())
        n += tgt.numel()
        preds.append(pr.cpu().numpy())
        print(f"  {NAME} {len(preds)}/{len(wins)} ppl={math.exp(nll / n):.4f} {time.time() - t0:.0f}s", flush=True)
    np.save(f"{OUTD}/{NAME}.npy", np.concatenate(preds))
    res = dict(name=NAME, windows=len(wins), tokens=n, ppl=math.exp(nll / n), top1=ok / n, window_nll=wn,
               seconds=time.time() - t0)
    json.dump(res, open(f"{OUTD}/{NAME}.json", "w"))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "window_nll"}), flush=True)


if __name__ == "__main__":
    main()
