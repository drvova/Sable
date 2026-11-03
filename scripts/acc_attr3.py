"""Accuracy attribution for Qwen3 F_O models (QK-norm, dense MLP or mixture of experts): WikiText-2 perplexity of
F_O with single steps made more precise.  The "+" variants are measurement devices, not provable functions.

usage: python -m scripts.acc_attr3 OUTDIR NAME [NWIN]
  NAME: fo | fo+FLAG[+FLAG...], FLAG in
    P     attention probabilities and normalization in float64 (as scripts/acc_attribution.py)
    QK    queries and keys with 4 more bits (12-bit QK-norm output; RoPE and scores at that precision)
    QKN   QK-norm in float: the head row normalized without its 8-bit requantization, float gains
    SILU  SiLU outputs with 4 more fractional bits;  UP: up projection output with 8 more bits
    AATT, AMLP  the normalized input of QKV (AATT) or of gate and up (AMLP) unrounded (no Requant8/NormScale
                rounding: the same row direction and scale, every entry exact)
    AO, AD      the input of W_o (attention output) or of W_down (SiLU x up) unrounded (no Requant8 rounding)
    D1, D2, D3  the down input with 1-3 more bits (9-11-bit Requant);  CL1, CL2: its row scale 2^1 or 2^2
          below the row maximum, saturating at +-127
    SMD   power-of-two smoothing of the down input (provable): channel j of SiLU x up shifted by -k_j before its
          Requant8, row j of W_down scaled by 2^k_j and requantized to int8; k_j from SmoothQuant (alpha 0.5)
          statistics of windows 40-41 (outside the evaluated ones)
    WQKV, WO, WG, WU, WD  that weight matrix from the checkpoint (bf16, rotated and fused as by the
                quantizer) at F_O's exponents instead of int8
  Mixture-of-experts layers (Qwen3-30B-A3B) take AMLP, SILU and UP inside every expert and the router, and:
    RT    router logits, top-k selection and weights in float64 (no SatShift to 2^-8, tau_r table or Div to
          2^-14; without AD or MQ the weights are then rounded to 2^-14 for the merge)
    AD    everything after each expert's SiLU x up exact: no expert Requant8, weight shift or token-row
          requantization to MOE_B, the slots merged in float64
    MQ    the merge exact (weight shift and token-row requantization), each expert's Requant8 kept
  NWIN: the first NWIN windows of 2,048 tokens (default 48)
F_O is read from SABLE_FO_MODEL; the checkpoint (W*, QKN) from the Hugging Face cache.  Writes
OUTDIR/NAME.json (per-window NLL, perplexity, top-1) and OUTDIR/NAME.npy (greedy predictions)."""
import json
import math
import os
import sys
import time
import numpy as np
import torch
from scripts import acc as sa
from scripts import paths
from scripts import acc_attribution as a1
from zk import model as mdl, evalfo
from zk.model import (requant8, normscale, imm, shiftq8s, rope8, gshiftround, satadd, shiftround,
                      silu_values, shiftq8, fdiv, pow2)

if __name__ == "__main__" and len(sys.argv) < 3:
    raise SystemExit(__doc__)
OUTD, NAME = sys.argv[1], sys.argv[2]
NWIN = int(sys.argv[3]) if len(sys.argv) > 3 else 48
FLAGS = set(NAME.split("+")[1:]) if NAME.startswith("fo") else set()
a1.FLAGS = FLAGS                                  # the parent engine's attention and caches follow these flags
WFLAGS = {"WQKV": "Wqkv", "WO": "Wo", "WG": "Wg", "WU": "Wu", "WD": "Wd"}


def unrounded(x, q):
    """x (int, before rounding) scaled per row onto q (its rounded image): the exact value q stands for."""
    xd = x.double()
    c = (q.double() * xd).sum(-1) / (xd * xd).sum(-1).clamp(min=1.0)
    return xd * c.unsqueeze(-1)


def mm(x, Ly, name):
    """x @ W in W's integer units: int8 GEMM, or float64 when x is unrounded or W comes from the checkpoint."""
    F = Ly.__dict__.get("F" + name)
    if F is None and not x.is_floating_point():
        return imm(x, getattr(Ly, name))
    W = F if F is not None else getattr(Ly, name)
    out = torch.empty((x.shape[0], W.shape[1]), dtype=torch.int64, device="cuda")
    for i in range(0, x.shape[0], 1024):
        out[i:i + 1024] = torch.round(x[i:i + 1024].double() @ W.double()).to(torch.int64)
    return out


def load_float(M):
    """Checkpoint weights (and QK-norm gains) for the W* / QKN flags, in units of F_O's exponents."""
    need = [WFLAGS[f] for f in FLAGS if f in WFLAGS]
    if not need and "QKN" not in FLAGS:
        return
    from scripts.quantize import checkpoint_reader
    get = checkpoint_reader(paths.hf_snapshot())
    cfg = M.cfg
    Qr, sq = mdl._rot(cfg.d, 1234), math.sqrt(cfg.d) / 127.0
    for li, Ly in enumerate(M.layers):
        p = f"model.layers.{li}."
        g2 = get(p + "post_attention_layernorm.weight").float()

        def put(name, Wf, e):
            e = torch.as_tensor(e, device="cuda").double()
            setattr(Ly, "F" + name, (Wf.double() / torch.pow(2.0, e).reshape(1, -1)).to(torch.bfloat16))
        if "Wqkv" in need:
            put("Wqkv", mdl._qkv_float(get, li, Qr, sq), Ly.eWqkv)
        if "Wo" in need:
            put("Wo", get(p + "self_attn.o_proj.weight").float().T @ Qr, Ly.eWo)
        if "Wg" in need:
            put("Wg", sq * (Qr.T @ (g2[:, None] * get(p + "mlp.gate_proj.weight").float().T)), Ly.eWg)
        if "Wu" in need:
            put("Wu", sq * (Qr.T @ (g2[:, None] * get(p + "mlp.up_proj.weight").float().T)), Ly.eWu)
        if "Wd" in need:
            put("Wd", get(p + "mlp.down_proj.weight").float().T @ Qr, Ly.eWd)
        if "QKN" in FLAGS:
            gq, gk = (get(p + f"self_attn.{n}_norm.weight").double() * math.sqrt(cfg.dh) / 127.0 for n in "qk")
            g = torch.cat([gq.repeat(cfg.nh), gk.repeat(cfg.nkv)])
            Ly.Fg = g / torch.pow(2.0, Ly.eGqk.double())
        torch.cuda.empty_cache()


SMD_REC = None


def smooth_down(M, eng, wins):
    """SMD: per-channel power-of-two shifts k_j of the down input from SmoothQuant statistics on wins."""
    global SMD_REC
    from scripts.quantize import checkpoint_reader
    SMD_REC = {}
    for w in wins:
        with torch.no_grad():
            eng.prefill(torch.tensor([w]).cuda(), eng.new_caches(1, len(w)))
    rec, SMD_REC = SMD_REC, None
    get = checkpoint_reader(paths.hf_snapshot())
    Qr = mdl._rot(M.cfg.d, 1234)
    for li, Ly in enumerate(M.layers):
        Wf = (get(f"model.layers.{li}.mlp.down_proj.weight").float().T @ Qr).double() / 2.0 ** Ly.eWd
        s = 0.5 * (torch.log2(rec[id(Ly)].clamp(min=1e-30)) - torch.log2(Wf.abs().amax(1).clamp(min=1e-30)))
        k = torch.round(s - s.median()).to(torch.int64).clamp(-20, 20)
        Ly.smK = int(k.max())
        Ly.smsh = pow2(Ly.smK - k)
        W2 = Wf * torch.pow(2.0, k.double()).unsqueeze(1)
        Ly.smdE = max(0, math.ceil(math.log2(float(W2.abs().max()) / 127.0)))
        Ly.WdS = torch.round(W2 / 2.0 ** Ly.smdE).clamp(-127, 127).to(Ly.Wd.dtype)
        if li % 16 == 0:
            print(f"  SMD layer {li}: k in [{int(k.min())}, {int(k.max())}], weight exponent +{Ly.smdE}", flush=True)
    torch.cuda.empty_cache()


class Eng(a1.Eng):
    def _qkv(self, Ly, acc, Eh):
        cfg = self.M.cfg
        nq, nk, dh = cfg.nh * cfg.dh, cfg.nkv * cfg.dh, cfg.dh
        R = acc.shape[0]
        if not FLAGS & {"QK", "QKN"}:
            return mdl.qkv_out(cfg, Ly, acc, Eh)
        a = acc[:, :nq + nk].reshape(-1, dh)
        q8, _, _ = shiftq8(a, emin=0)
        qp, E2, _ = normscale(q8)
        E2c = E2.view(R, -1).repeat_interleave(dh, dim=1)
        k = Ly.kn.unsqueeze(0) + E2c
        if "QKN" in FLAGS:
            prod = unrounded(a, qp).view(R, nq + nk) * Ly.Fg.unsqueeze(0)
        else:
            prod = (qp.view(R, nq + nk) * Ly.Gqk.unsqueeze(0)).double()
        lim = 127
        if "QK" in FLAGS:
            k, lim = k - 4, 16 * 127
        qk = torch.round(prod / torch.pow(2.0, k.double())).clamp(-lim, lim).to(torch.int64)
        v, _ = shiftq8s(acc[:, nq + nk:], Ly.kc[nq + nk:].unsqueeze(0) + Eh.unsqueeze(-1),
                        Ly.bias[nq + nk:].unsqueeze(0))
        return torch.cat([qk, v], dim=1)

    def _layer(self, Ly, h, B, S, pos, cache, wpos, kmax):
        M, cfg = self.M, self.M.cfg
        nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        half = dh // 2
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        qkv = self._qkv(Ly, mm(unrounded(h, qp) if "AATT" in FLAGS else qp, Ly, "Wqkv"), Eh)
        Qh = qkv[:, :nq].reshape(B, S, nh, dh)
        Kh = qkv[:, nq:nq + nk].reshape(B, S, nkv, dh)
        Vh = qkv[:, nq + nk:].reshape(B, S, nkv, dh)
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
        acc = mm(unrounded(o, qo) if "AO" in FLAGS else qo, Ly, "Wo")
        h, _ = satadd(h, gshiftround(acc, (Ly.Kco - eo).unsqueeze(-1)))
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        xm = unrounded(h, qp) if "AMLP" in FLAGS else qp
        if cfg.moe:
            h, _ = satadd(h, self._moe(Ly, xm, qp, Eh))
            return h
        kg = (M.eG - Ly.eWg).unsqueeze(0) + Eh.clamp(min=mdl.gate_kmin(M, Ly)).unsqueeze(-1)
        c = shiftround(mm(xm, Ly, "Wg"), kg).clamp(-mdl.SILU_CMAX, mdl.SILU_CMAX)
        extra = 0
        if "SILU" in FLAGS:
            x = c.double() * (2.0 ** M.eG)
            a = torch.round(x / (1 + torch.exp(-x)) * (2.0 ** (4 - M.eA))).to(torch.int64)
            extra += 4
        else:
            a = silu_values(c, M.eG, M.eA)
        accu = mm(xm, Ly, "Wu")
        qu, eup, _ = shiftq8(accu)
        if "UP" in FLAGS:
            k8 = (eup - 8).unsqueeze(-1)
            kk = k8.clamp(min=0)
            qu = torch.where(k8 > 0, fdiv(accu + (pow2(kk) >> 1), pow2(kk)), accu * pow2((-k8).clamp(min=0)))
            extra += 8
        m = a * qu
        Kcd = -(Ly.eWu + M.eA + Ly.eWd + M.f_h)
        if SMD_REC is not None:
            r = (m.abs().double() * torch.pow(2.0, (eup - Eh).double()).unsqueeze(-1)).amax(0)
            SMD_REC[id(Ly)] = torch.maximum(SMD_REC[id(Ly)], r) if id(Ly) in SMD_REC else r
        if "SMD" in FLAGS and SMD_REC is None:
            m = m * Ly.smsh.unsqueeze(0)
            qm, em, _ = requant8(m, 0, 60)
            accd = imm(qm, Ly.WdS)
            kd = (Kcd - (em - extra + eup - Eh) - Ly.smdE + Ly.smK).unsqueeze(-1)
            h, _ = satadd(h, gshiftround(accd, kd))
            return h
        qm, em, _ = requant8(m, 0, 17 + extra)
        nb = sum(int(f[1:]) for f in FLAGS if f in ("D1", "D2", "D3"))
        nc = sum(int(f[2:]) for f in FLAGS if f in ("CL1", "CL2"))
        if nb or nc:
            em = (em - nb - nc).clamp(min=0)
            lim = 127 << nb
            qm = fdiv(2 * m + pow2(em).unsqueeze(-1), 2 * pow2(em).unsqueeze(-1)).clamp(-lim, lim)
            if nb:
                qm = qm.double()
        accd = mm(unrounded(m, qm) if "AD" in FLAGS else qm, Ly, "Wd")
        kd = (Kcd - (em - extra + eup - Eh)).unsqueeze(-1)
        h, _ = satadd(h, gshiftround(accd, kd))
        return h


MOE_FLAGS = {"RT", "AD", "MQ", "SILU", "UP", "AMLP"}
OWN_MOE = [False]                                 # selftest: this engine's MoE path without flags


def mmw(x, W):
    """x @ W for an expert or router matrix: int8 GEMM, or float64 when x is unrounded."""
    if not x.is_floating_point():
        return imm(x, W)
    return torch.round(x.double() @ W.double()).to(torch.int64)


def _moe(self, Ly, xm, qp, Eh):
    """The MoE MLP's contribution to the residual (mdl.moe_out) with the attribution flags."""
    M, cfg = self.M, self.M.cfg
    if not FLAGS & MOE_FLAGS and not OWN_MOE[0]:
        return mdl.moe_out(M, Ly, qp, Eh)[0]
    k, E, T = cfg.moe["k"], Ly.Wr.shape[1], xm.shape[0]
    accr = mmw(xm, Ly.Wr)
    if "RT" in FLAGS:
        zl, idx = torch.topk(accr.double() * torch.pow(2.0, (Ly.eWr - Eh).double()).unsqueeze(-1), k, dim=-1)
        wf = torch.softmax(zl, dim=-1)
        w = torch.round(wf * (1 << mdl.W_FRAC)).to(torch.int64)
    else:
        kr = (Eh - Ly.eWr - mdl.R_FRAC).clamp(min=1).unsqueeze(-1)
        z = mdl.satshift(accr, kr, mdl.R_SAT)
        key = z * E + (E - 1 - torch.arange(E, device=z.device)).unsqueeze(0)
        idx = torch.topk(key, k, dim=-1).indices
        zs = z.gather(1, idx)
        pr = mdl.router_table()[zs.max(dim=-1, keepdim=True).values - zs]
        w = mdl.div_round(pr, pr.sum(dim=-1, keepdim=True), 1 << mdl.W_FRAC)
        wf = w.double() / (1 << mdl.W_FRAC)
    exact = FLAGS & {"AD", "MQ"}
    y = torch.zeros((T, cfg.d), dtype=torch.float64, device=xm.device) if exact else None
    qm_all = torch.zeros((T, k, cfg.moe["de"]), dtype=torch.int64, device=xm.device)
    es = torch.zeros((T, k), dtype=torch.int64, device=xm.device)
    for e in torch.unique(idx).tolist():
        rows, slot = torch.where(idx == e)
        x, E_ = xm[rows], Eh[rows]
        kg = ((M.eG - Ly.eWge[e]) + E_).clamp(min=1).unsqueeze(-1)
        c = shiftround(mmw(x, Ly.Wge[e]), kg).clamp(-mdl.SILU_CMAX, mdl.SILU_CMAX)
        extra = 0
        if "SILU" in FLAGS:
            g = c.double() * (2.0 ** M.eG)
            a = torch.round(g / (1 + torch.exp(-g)) * (2.0 ** (4 - M.eA))).to(torch.int64)
            extra += 4
        else:
            a = silu_values(c, M.eG, M.eA)
        accu = mmw(x, Ly.Wue[e])
        qu, eup, _ = shiftq8(accu)
        if "UP" in FLAGS:
            k8 = (eup - 8).unsqueeze(-1)
            kk = k8.clamp(min=0)
            qu = torch.where(k8 > 0, fdiv(accu + (pow2(kk) >> 1), pow2(kk)), accu * pow2((-k8).clamp(min=0)))
            extra += 8
        m = a * qu
        qm, em, _ = requant8(m, 0, 17 + extra)
        if exact:
            # value of slot (row, s) in the residual's units: w_s (m 2^-extra) W_down 2^(eup + eWu + eWd + eA + f_h - Eh)
            v = m.double() if "AD" in FLAGS else qm.double() * torch.pow(2.0, em.double()).unsqueeze(-1)
            sc = torch.pow(2.0, (eup + Ly.eWue[e] + Ly.eWde[e] + M.eA + M.f_h - E_ - extra).double())
            y.index_add_(0, rows, (v @ Ly.Wde[e].double()) * (wf[rows, slot] * sc).unsqueeze(-1))
        else:
            qm_all[rows, slot] = qm
            es[rows, slot] = em - extra + eup + Ly.eWue[e] + Ly.eWde[e]
    if exact:
        return torch.round(y).to(torch.int64)
    Et = es.max(dim=-1, keepdim=True).values
    om = shiftround(w, Et - es)
    q2, e2, _ = mdl._rowq_b((qm_all * om.unsqueeze(-1)).reshape(T, -1), mdl.MOE_B)
    q2 = q2.reshape(T, k, -1)
    accd = torch.zeros((T, cfg.d), dtype=torch.int64, device=xm.device)
    for e in torch.unique(idx).tolist():
        rows, slot = torch.where(idx == e)
        accd.index_add_(0, rows, mdl.imm_wide(q2[rows, slot], Ly.Wde[e]))
    kd = (-(e2 + Et.squeeze(-1) + M.eA - Eh - mdl.W_FRAC + M.f_h)).clamp(max=24)
    return gshiftround(accd, kd.unsqueeze(-1))


Eng._moe = _moe


def selftest(M, eng):
    """Without flags this engine is F_O: the same logits as evalfo on a short random prompt (for a mixture of
    experts, also through this engine's own MoE path)."""
    ids = torch.randint(0, M.cfg.V, (1, 300), generator=torch.Generator().manual_seed(3)).cuda()
    ref = evalfo.Engine(M, 512)
    a = ref.logits(ref.prefill(ids, ref.new_caches(1, 300))[0])
    for own in ([False, True] if M.cfg.moe else [False]):
        OWN_MOE[0] = own
        b = eng.logits(eng.prefill(ids, eng.new_caches(1, 300))[0])
        assert torch.equal(a, b), "the attribution engine without flags differs from F_O"
    OWN_MOE[0] = False
    print("self-test: equals evalfo without flags", flush=True)


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUTD, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    sa.MAXTOK = 10 ** 9
    wins = sa.wiki_ids(tok)[:NWIN]
    if not NAME.startswith("fo"):
        m = sa.HFModel(NAME, tok)
        logits_all = m.logits_all
    else:
        M = mdl.load_model(paths.fo_model())
        mdl.check = lambda cond, msg="": None    # range conditions are not the subject here
        eng = Eng(M, 4096)
        if not FLAGS:
            selftest(M, eng)
        load_float(M)
        if "SMD" in FLAGS:
            smooth_down(M, eng, sa.wiki_ids(tok)[40:42])

        def logits_all(ids):
            h = eng.prefill(torch.tensor([ids]).cuda(), eng.new_caches(1, len(ids)))[0]
            return torch.cat([eng.logits(h[i:i + 512]).to(torch.float32) * (2.0 ** mdl.EZ)
                              for i in range(0, h.shape[0], 512)])
    nll, n, ok, preds, wn = 0.0, 0, 0, [], []
    t0 = time.time()
    for w in wins:
        with torch.no_grad():
            lg = logits_all(w)[:-1]
        tgt = torch.tensor(w[1:], device=lg.device)
        lp = torch.log_softmax(lg.float(), dim=-1)
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
