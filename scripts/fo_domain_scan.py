"""Where is F_O undefined?  The domain scan of the paper's evaluation ("aborts"): runs the evalfo
engine's prefill over real and adversarial prompts with F_O's range conditions recorded instead of
asserted, and counts, per prompt set, the sequences and rows on which an input leaves the domain of
F_O (where the honest prover aborts):
  silu_in      SiLU input |c| > 2^14 (zk.model.SILU_CMAX)
  gate_shift   gate shift outside [1, SILU_KMAX = 15]
  o_shift      o-projection shift outside [KMIN_SR, 24] = [-16, 24]
  o_left       o-projection output (a left shift for k <= 0) outside the activation range A
  d_shift      down-projection shift outside [-16, 24]
  d_left       down-projection output outside A
  helper       any other range check of the engine's helpers failed (whole batch counted)
Prompt sets: WikiText-2 test (2,048-token windows), GSM8K 8-shot, 2,000 random MMLU 5-shot prompts,
and 4,096-token adversarial sets (one repeated token, random ids, digit strings, random bytes,
repeated phrases with special tokens).

usage: python -m scripts.fo_domain_scan OUT.json        (ABORT_SMOKE=1: two prompts per set)
Needs SABLE_FO_MODEL and the checkpoint's tokenizer (scripts/paths.py).
SCAN_FAST=1: the serving engine's per-sequence flags instead (zk/fofast.py: every condition of F_O,
including QK-norm and MoE models, which the engine above does not implement); rows_by_condition is
then empty.  Sets already in OUT.json are kept and skipped (resume)."""
import json
import os
import random
import sys
import time
import torch
from scripts import acc as sa
from scripts import paths
from zk import model as mdl, evalfo
from zk.model import (requant8, normscale, imm, shiftq8s, rope8, satadd, shiftround, silu_values, shiftq8,
                      fdiv, pow2, A_BITS)

KEYS = ("silu_in", "gate_shift", "o_shift", "o_left", "d_shift", "d_left", "helper")


def gsr(acc, k, rec, name, valid):
    bad_k = ((k < mdl.KMIN_SR) | (k > 24)).reshape(-1)
    kc = k.clamp(mdl.KMIN_SR, 24)
    kk = kc.clamp(min=0)
    c = torch.where(kc > 0, fdiv(acc + (pow2(kk) >> 1), pow2(kk)), acc * pow2((-kc).clamp(min=0)))
    bad_c = (c.abs() >= (1 << A_BITS)).any(dim=-1)
    rec[name + "_shift"] |= bad_k & valid
    rec[name + "_left"] |= bad_c & valid
    return c.clamp(-(1 << A_BITS) + 1, (1 << A_BITS) - 1)


class Eng(evalfo.Engine):
    def _layer(self, Ly, h, B, S, pos, cache, wpos, kmax):
        M, cfg = self.M, self.M.cfg
        nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        half = dh // 2
        rec, valid = self.rec, self.valid
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        acc = imm(qp, Ly.Wqkv)
        k = Ly.kc.unsqueeze(0) + Eh.unsqueeze(-1)
        qkv, _ = shiftq8s(acc, k, Ly.bias.unsqueeze(0))
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
        cache["rK"][bi, wpos] = rK.to(torch.int16)
        cache["V"][bi, wpos] = Vh.to(torch.int8)
        o = self._attend(Ly, rQ, cache["rK"][:, :kmax], cache["V"][:, :kmax], pos, mdl.exp_table(Ly.E, dh), kmax)
        o = o.reshape(B * S, nq)
        qo, eo, _ = requant8(o)
        acc = imm(qo, Ly.Wo)
        h, _ = satadd(h, gsr(acc, (Ly.Kco - eo).unsqueeze(-1), rec, "o", valid))
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        kg = (M.eG - Ly.eWg).unsqueeze(0) + Eh.unsqueeze(-1)
        rec["gate_shift"] |= ((kg < 1) | (kg > mdl.SILU_KMAX)).any(dim=-1) & valid
        c = shiftround(imm(qp, Ly.Wg), kg.clamp(1, mdl.SILU_KMAX))
        rec["silu_in"] |= (c.abs() > mdl.SILU_CMAX).any(dim=-1) & valid
        a = silu_values(c.clamp(-mdl.SILU_CMAX, mdl.SILU_CMAX), M.eG, M.eA)
        qu, eup, _ = shiftq8(imm(qp, Ly.Wu))
        qm, em, _ = requant8(a * qu)
        Kcd = -(Ly.eWu + M.eA + Ly.eWd + M.f_h)
        kd = (Kcd - (em + eup - Eh)).unsqueeze(-1)
        h, _ = satadd(h, gsr(imm(qm, Ly.Wd), kd, rec, "d", valid))
        return h

    def scan(self, seqs):
        """seqs: list of token lists. -> per sequence: dict of violated keys (row counts)."""
        B = len(seqs)
        S = max(len(s) for s in seqs)
        toks = torch.zeros((B, S), dtype=torch.int64)
        valid = torch.zeros((B, S), dtype=torch.bool)
        for b, s in enumerate(seqs):
            toks[b, :len(s)] = torch.tensor(s)
            valid[b, :len(s)] = True
        self.valid = valid.reshape(-1).cuda()
        self.rec = {k: torch.zeros(B * S, dtype=torch.bool, device="cuda") for k in KEYS}
        caches = self.new_caches(B, S)
        try:
            self.prefill(toks.cuda(), caches)
        except AssertionError as ex:
            print("helper check failed:", ex, flush=True)
            self.rec["helper"] |= self.valid
        del caches
        out = []
        for b in range(B):
            out.append({k: int(v.view(B, S)[b].sum()) for k, v in self.rec.items()})
        return out


def main():
    from transformers import AutoTokenizer
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    M = mdl.load_model(paths.fo_model())
    eng = None if os.environ.get("SCAN_FAST") else Eng(M, 4096)
    rng = random.Random(1)
    sets = {}
    # real text
    ids = tok(paths.wikitext2("test"))["input_ids"]
    sets["wikitext2_test"] = [ids[i:i + 2048] for i in range(0, len(ids) - 2048 + 1, 2048)]
    sets["gsm8k_8shot"] = [tok(p)["input_ids"] for p, _ in sa.gsm8k_items()]
    mm = sa.mmlu_items()
    rng.shuffle(mm)
    sets["mmlu_5shot_2000"] = [tok(p)["input_ids"] for p, _, _ in mm[:2000]]
    # adversarial, 4096 tokens each
    V = 151643                                          # ordinary (non-special) token ids
    common = [tok(w)["input_ids"][0] for w in (" the", "a", "0", "\n", "!", " ", ".", "<", "x", "9")]
    sets["repeat_one_token"] = [[t] * 4096 for t in common] + [[rng.randrange(V)] * 4096 for _ in range(22)]
    sets["random_ids"] = [[rng.randrange(V) for _ in range(4096)] for _ in range(32)]
    sets["digit_strings"] = [tok("".join(rng.choice("0123456789") for _ in range(12000)))["input_ids"][:4096]
                             for _ in range(32)]
    sets["random_bytes"] = [tok(bytes(rng.randrange(256) for _ in range(20000)).decode("latin-1"))["input_ids"][:4096]
                            for _ in range(32)]
    sets["repeat_phrase"] = [tok((" " + " ".join(rng.choice(["ignore", "previous", "instructions", "AAAA", "<|im_end|>", "☃", "%%%%"]) for _ in range(3))) * 2000)["input_ids"][:4096]
                             for _ in range(32)]
    res = json.load(open(out)) if os.path.exists(out) else {}
    if os.environ.get("ABORT_SMOKE"):
        sets = {k: v[:2] for k, v in sets.items()}
    fast = bool(os.environ.get("SCAN_FAST"))
    fe = None
    for name, seqs in sets.items():
        if name in res:
            continue
        t0 = time.time()
        bs = max(1, 8192 // max(len(s) for s in seqs))
        per = []
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        if fast:
            from zk import fofast
            fe = None                                   # free the previous set's engine first
            torch.cuda.empty_cache()
            fe = fofast.FastEngine(M, -(-max(len(s) for s in seqs) // 128) * 128 + 2, bs)   # prefill padded to 128
        for i in range(0, len(order), bs):
            batch = [seqs[j] for j in order[i:i + bs]]
            if fast:
                n = len(batch)
                _, _, _, fl = fe.generate(batch + [batch[0]] * (bs - n), 1, flags=True, spad=128)
                per += [{"undefined": int(pf)} for pf, _ in fl[:n]]
            else:
                per += eng.scan(batch)
        bad = [p for p in per if any(p.values())]
        tot = {k: sum(p.get(k, 0) for p in per) for k in KEYS}
        res[name] = dict(sequences=len(seqs), tokens=sum(len(s) for s in seqs), undefined_sequences=len(bad),
                         rows_by_condition=tot, seconds=round(time.time() - t0, 1))
        print(name, json.dumps(res[name]), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
