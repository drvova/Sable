"""Accuracy of F_O against the BF16 checkpoint of SABLE_ARCH and two deployed quantizations.

usage: python -m scripts.acc VARIANT TASK [limit]
  VARIANT: fo (F_O, zk/evalfo.py), bf16, fp8 (W8A8 FP8 e4m3, per-tensor scales, activations
           dynamic), sq8 (SmoothQuant W8A8 INT8, alpha 0.5, per-channel weights, per-token
           activations); fp8 and sq8 are simulated by quantize-dequantize in BF16.
  TASK:    wiki (WikiText-2 test, windows of 2048 tokens over the first ACC_MAXTOK tokens, default
           32768, 0 for the whole split: perplexity,
           top-1 accuracy, greedy predictions saved for agreement), mmlu (5-shot, test split),
           gsm8k (8-shot, greedy, at most ACC_GEN new tokens (default 256), exact match of the final number).
Writes results/acc_{VARIANT}_{TASK}.json (and .npz with the wiki predictions and per-window NLL).
ACC_STRIDE=k evaluates every k-th MMLU or GSM8K item (all subjects at 1/k of the cost); ACC_DETAIL=1 also
saves the four answer logits of every MMLU item (nats, acc_{VARIANT}_mmlu_logits.npz) and the GSM8K
generations (texts in the json), for the analysis of where F_O's task accuracy goes.
GSM8K serving: ACC_SERVE=1 generates by continuous batching (F_O: zk.fofast.FastEngine.serve, exact per
sequence; baselines: transformers continuous batching), ACC_SMAX the serving context, ACC_ANSWER_STOP=1 also
stops a reply after its first complete '#### N' (the extracted answer cannot change)."""
import json
import math
import os
import re
import sys
import time
import numpy as np
import torch

from scripts import paths

OUT = os.environ.get("ACC_OUT", "results")
CTX, MAXTOK = 2048, int(os.environ.get("ACC_MAXTOK", "32768")) or None
B_FO = int(os.environ.get("ACC_BFO", "16"))
B_HF = int(os.environ.get("ACC_BHF", "32"))
STRIDE = int(os.environ.get("ACC_STRIDE", "1"))
DETAIL = bool(int(os.environ.get("ACC_DETAIL", "0")))
GEN = int(os.environ.get("ACC_GEN", "256"))          # GSM8K generation cap (tokens)
SERVE = bool(int(os.environ.get("ACC_SERVE", "0")))  # GSM8K: continuous batching (F_O: FastEngine.serve)
SMAX = int(os.environ.get("ACC_SMAX", "4096"))       # serving engine context (prompt + generation)
ANS_STOP = bool(int(os.environ.get("ACC_ANSWER_STOP", "0")))   # GSM8K served: also stop once '#### N' is complete


# ------------------------------------------------------------------ data
def wiki_ids(tok):
    ids = tok(paths.wikitext2("test"))["input_ids"][:MAXTOK]
    return [ids[i:i + CTX] for i in range(0, len(ids) - CTX + 1, CTX)]


def mmlu_items(limit=None):
    import datasets
    d = datasets.load_dataset("cais/mmlu", "all", cache_dir=paths.datasets_cache("mmlu"))
    dev = {}
    for ex in d["dev"]:
        dev.setdefault(ex["subject"], []).append(ex)

    def fmt(ex, ans=True):
        s = ex["question"].strip() + "\n"
        for L, c in zip("ABCD", ex["choices"]):
            s += f"{L}. {c}\n"
        return s + "Answer:" + (f" {'ABCD'[ex['answer']]}\n\n" if ans else "")
    items = []
    for ex in d["test"]:
        subj = ex["subject"]
        head = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
        prompt = head + "".join(fmt(e) for e in dev[subj][:5]) + fmt(ex, ans=False)
        items.append((prompt, ex["answer"], subj))
    items = items[::STRIDE]
    return items[:limit] if limit else items


def gsm8k_items(limit=None):
    import datasets
    d = datasets.load_dataset("openai/gsm8k", "main", cache_dir=paths.datasets_cache("gsm8k"))
    shots = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in list(d["train"])[:8])
    items = [(shots + f"Question: {e['question']}\nAnswer:", e["answer"].split("####")[-1].strip().replace(",", ""))
             for e in d["test"]]
    items = items[::STRIDE]
    return items[:limit] if limit else items


def gsm8k_pred(text):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        s = nums[-1] if nums else ""
    s = s.replace(",", "").rstrip(".")
    return s


# ------------------------------------------------------------------ models
class FOModel:
    def __init__(self):
        from zk import model as mdl, evalfo
        self.mdl = mdl
        self.M = mdl.load_model(paths.fo_model())
        self.eng = evalfo.Engine(self.M, 4096)
        self.fast = {}

    def logits_all(self, ids):
        """(S, V) float logits of one window."""
        caches = self.eng.new_caches(1, len(ids))
        h = self.eng.prefill(torch.tensor([ids]).cuda(), caches)[0]
        out = []
        for i in range(0, h.shape[0], 512):
            out.append(self.eng.logits(h[i:i + 512]).to(torch.float32) * (2.0 ** self.mdl.EZ))
        return torch.cat(out)

    def last_logits(self, prompts):
        lens = [len(p) for p in prompts]
        S = max(lens)
        toks = torch.zeros((len(prompts), S), dtype=torch.int64)
        for b, p in enumerate(prompts):
            toks[b, :len(p)] = torch.tensor(p)
        caches = self.eng.new_caches(len(prompts), S)
        h = self.eng.prefill(toks.cuda(), caches)
        rows = h[torch.arange(len(prompts), device="cuda"), torch.tensor(lens, device="cuda") - 1]
        return self.eng.logits(rows)                 # integers in units 2^-8

    def generate(self, prompts, ngen, stop):
        """Greedy generation; with the serving engine (ACC_FOFAST, default on) also self.undef: per
        prompt, whether F_O is undefined on it (a row of its prefill or of a decode step it used
        leaves F_O's domain; evalfo stops with an error instead)."""
        if os.environ.get("ACC_FOFAST", "1") == "0":
            self.undef = [False] * len(prompts)
            return self.eng.generate(prompts, ngen, stop)
        from zk import fofast
        if B_FO not in self.fast:
            self.fast[B_FO] = fofast.FastEngine(self.M, SMAX, B_FO)
        B = len(prompts)
        ids, _, _, fl = self.fast[B_FO].generate(prompts + [prompts[0]] * (B_FO - B), ngen, flags=True, spad=128)
        out, self.undef = [], []
        for b in range(B):
            g = ids[b]
            n = next((m for m in range(1, len(g) + 1) if stop(b, g[:m])), len(g))   # evalfo's early stop
            out.append(g[:n])
            pf, st = fl[b]
            self.undef.append(bool(pf or any(st[:n - 1])))
        return out

    def serve(self, prompts, ngen, stop, done):
        """Greedy generation of all prompts by the serving engine with continuous batching (B_FO slots);
        done(i, ids, F_O undefined on prompt i) as each sequence ends."""
        from zk import fofast
        if B_FO not in self.fast:
            self.fast[B_FO] = fofast.FastEngine(self.M, SMAX, B_FO)
        self.fast[B_FO].serve(prompts, ngen, stop, done=done)


def _fake_fp8(x):
    s = x.abs().amax().float().clamp(min=1e-12) / 448.0
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def _fake_int8_rows(x):
    s = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 127.0
    return (torch.round(x.float() / s).clamp(-127, 127) * s).to(x.dtype)


def _moe_experts_8bit(model, mode):
    """Fused MoE experts (Qwen3-MoE in transformers 5): 8-bit weights per expert and 8-bit expert inputs.
    fp8: per-tensor e4m3 weights of each expert matrix, dynamic per-tensor inputs; int8: per-channel
    weights, per-token inputs (no SmoothQuant smoothing, which would need per-expert calibration).
    The router stays in BF16, as in deployed 8-bit MoE kernels."""
    import types
    import torch.nn.functional as Fn
    wq = (lambda W: torch.stack([_fake_fp8(w) for w in W])) if mode == "fp8" else \
        (lambda W: torch.stack([_fake_int8_rows(w) for w in W]))
    aq = _fake_fp8 if mode == "fp8" else _fake_int8_rows

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = Fn.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            g, u = Fn.linear(aq(hidden_states[tok]), self.gate_up_proj[e]).chunk(2, dim=-1)
            h = Fn.linear(aq(self.act_fn(g) * u), self.down_proj[e])
            out.index_add_(0, tok, (h * top_k_weights[tok, pos, None]).to(out.dtype))
        return out
    n = 0
    for _, mod in model.named_modules():
        if type(mod).__name__.endswith("MoeExperts"):
            mod.gate_up_proj.data = wq(mod.gate_up_proj.data)
            mod.down_proj.data = wq(mod.down_proj.data)
            mod.forward = types.MethodType(forward, mod)
            n += 1
    return n


class HFModel:
    LIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, variant, tok):
        from transformers import AutoModelForCausalLM
        self.tok = tok
        self.model = AutoModelForCausalLM.from_pretrained(paths.hf_snapshot(), dtype=torch.bfloat16).cuda().eval()
        if variant == "fp8":
            self._fp8()
        elif variant == "sq8":
            self._sq8()

    def _linears(self):
        for name, mod in self.model.named_modules():
            if name.split(".")[-1] in self.LIN:
                yield name, mod

    def _fp8(self):
        for _, lin in self._linears():
            lin.weight.data = _fake_fp8(lin.weight.data)
            lin.register_forward_pre_hook(lambda m, a: (_fake_fp8(a[0]),))
        _moe_experts_8bit(self.model, "fp8")

    def _sq8(self, alpha=0.5):
        # calibration: per input channel amax of every linear's input on WikiText-2 train
        import datasets
        d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=paths.datasets_cache("wiki"))
        ids = self.tok("".join(d["train"]["text"][:4000]))["input_ids"][:64 * 512]
        amax, hooks = {}, []
        for name, lin in self._linears():
            def h(m, a, name=name):
                v = a[0].detach().abs().reshape(-1, a[0].shape[-1]).amax(dim=0).float()
                amax[name] = torch.maximum(amax[name], v) if name in amax else v
            hooks.append(lin.register_forward_pre_hook(h))
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                self.model(torch.tensor([ids[i:i + 512]]).cuda())
        for hk in hooks:
            hk.remove()
        for name, lin in self._linears():
            W = lin.weight.data.float()                               # (out, in)
            s = (amax[name].clamp(min=1e-5) ** alpha / W.abs().amax(dim=0).clamp(min=1e-5) ** (1 - alpha))
            s = s.clamp(min=1e-5)
            Ws = W * s.view(1, -1)
            sw = Ws.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
            lin.weight.data = (torch.round(Ws / sw).clamp(-127, 127) * sw).to(torch.bfloat16)
            inv = (1.0 / s).to(torch.bfloat16)
            lin.register_forward_pre_hook(lambda m, a, inv=inv: (_fake_int8_rows(a[0] * inv),))
        _moe_experts_8bit(self.model, "int8")

    @torch.no_grad()
    def logits_all(self, ids):
        return self.model(torch.tensor([ids]).cuda()).logits[0].float()

    @torch.no_grad()
    def last_logits(self, prompts):
        lens = [len(p) for p in prompts]
        S = max(lens)
        toks = torch.zeros((len(prompts), S), dtype=torch.int64)
        att = torch.zeros((len(prompts), S), dtype=torch.int64)
        for b, p in enumerate(prompts):                  # right padding: causal rows unaffected
            toks[b, :len(p)] = torch.tensor(p)
            att[b, :len(p)] = 1
        hs = self.model.model(toks.cuda(), attention_mask=att.cuda()).last_hidden_state
        rows = hs[torch.arange(len(prompts)), torch.tensor(lens) - 1]      # logits of the last real tokens only
        return self.model.lm_head(rows).float()

    @torch.no_grad()
    def generate(self, prompts, ngen, stop):
        S = max(len(p) for p in prompts)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        toks = torch.full((len(prompts), S), pad, dtype=torch.int64)
        att = torch.zeros((len(prompts), S), dtype=torch.int64)
        for b, p in enumerate(prompts):                  # left padding for generation
            toks[b, S - len(p):] = torch.tensor(p)
            att[b, S - len(p):] = 1
        out = self.model.generate(toks.cuda(), attention_mask=att.cuda(), max_new_tokens=ngen, do_sample=False,
                                  pad_token_id=pad, stop_strings=["Question:"], tokenizer=self.tok)
        return [o[S:].tolist() for o in out]

    @torch.no_grad()
    def serve(self, prompts, ngen, stop, done):
        """Greedy generation of all prompts with transformers' continuous batching (paged KV cache, shared
        prefix blocks).  Each request streams its tokens; at the first m with stop(i, ids[:m]) it is cut there
        and cancelled.  done(i, ids, False) as each sequence ends."""
        from transformers import GenerationConfig
        from transformers.generation.continuous_batching.utils import WorkloadHints
        gc = GenerationConfig(max_new_tokens=ngen, do_sample=False, eos_token_id=self.model.generation_config.eos_token_id)
        hints = WorkloadHints(max_prompt_length=max(len(q) for q in prompts), max_generated_length=ngen,
                              num_requests=len(prompts))
        ended, checked = set(), {}
        with self.model.continuous_batching_context_manager(generation_config=gc, block=True, timeout=5,
                                                            workload_hints=hints) as man:
            rid = {}
            for i, q in enumerate(prompts):
                rid[man.add_request(q, request_id=f"r{i}", max_new_tokens=ngen, streaming=True)] = i
            while len(ended) < len(prompts):
                r = man.get_result(timeout=1)
                if r is None:
                    if not man.is_running():
                        raise RuntimeError("continuous batching stopped with requests pending")
                    continue
                i = rid[r.request_id]
                if i in ended:
                    continue
                if r.error:
                    raise RuntimeError(f"request {i}: {r.error}")
                g = r.generated_tokens
                n = next((m for m in range(checked.get(i, 0) + 1, len(g) + 1) if stop(i, g[:m])), None)
                checked[i] = len(g)
                if n is None and not r.is_finished():
                    continue
                ended.add(i)
                if not r.is_finished():
                    man.cancel_request(r.request_id)
                done(i, g[:n or len(g)], False)


# ------------------------------------------------------------------ tasks
def run_wiki(m, tok, variant):
    wins = wiki_ids(tok)
    nll, n, ok, preds, wnll = 0.0, 0, 0, [], []
    t0 = time.time()
    for w in wins:
        lg = m.logits_all(w)[:-1]
        tgt = torch.tensor(w[1:], device=lg.device)
        lp = torch.log_softmax(lg, dim=-1)
        wnll.append(float(-lp.gather(1, tgt.view(-1, 1)).sum()))
        nll += wnll[-1]
        pr = lg.argmax(dim=-1)
        ok += int((pr == tgt).sum())
        n += tgt.numel()
        preds.append(pr.cpu().numpy())
        print(f"  wiki {len(preds)}/{len(wins)} ppl={math.exp(nll / n):.4f} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{OUT}/acc_{variant}_wiki.npz", preds=np.concatenate(preds), window_nll=np.array(wnll))
    return dict(tokens=n, ppl=math.exp(nll / n), top1=ok / n, seconds=time.time() - t0)


def run_mmlu(m, tok, variant, limit):
    items = mmlu_items(limit)
    cid = [tok(f" {L}")["input_ids"][-1] for L in "ABCD"]
    B = B_FO if variant == "fo" else B_HF
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))   # similar lengths per batch
    correct, by, t0 = 0, {}, time.time()
    preds, logit4 = {}, {}
    toks = {i: tok(items[i][0])["input_ids"] for i in order}
    batches, cur = [], []
    for i in order:                     # at most B prompts and B * S^2 <= 1.2e8 (attention memory)
        S = len(toks[i])
        if cur and (len(cur) == B or (len(cur) + 1) * S * S > float(os.environ.get("ACC_S2", "1.2e8"))):
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    k = 0
    ck = f"{OUT}/ckpt_{variant}_mmlu.json"          # resumable: predictions so far
    done = {int(i): p for i, p in json.load(open(ck)).items()} if os.path.exists(ck) else {}
    for idx in batches:
        if all(i in done for i in idx):
            pr = [done[i] for i in idx]
        else:
            prompts = [toks[i] for i in idx]
            lg = m.last_logits(prompts)[:, cid]
            pr = lg.argmax(dim=1).tolist()                               # ties: the first letter
            done.update(zip(idx, pr))
            if DETAIL:
                logit4.update(zip(idx, (lg.double() / (256.0 if variant == "fo" else 1.0)).cpu().numpy()))
            json.dump({str(i): p for i, p in done.items()}, open(ck, "w"))
        for i, p in zip(idx, pr):
            preds[i] = p
            ok = int(p == items[i][1])
            correct += ok
            s = by.setdefault(items[i][2], [0, 0])
            s[0] += ok
            s[1] += 1
        k += len(idx)
        print(f"  mmlu {k}/{len(order)} acc={correct / k:.4f} {time.time() - t0:.0f}s", flush=True)
    macro = float(np.mean([a / b for a, b in by.values()]))
    if DETAIL:
        ii = sorted(logit4)
        np.savez(f"{OUT}/acc_{variant}_mmlu_logits.npz", idx=np.array(ii), logits=np.stack([logit4[i] for i in ii]),
                 answer=np.array([items[i][1] for i in ii]))
    return dict(n=len(items), acc=correct / len(items), macro_acc=macro, seconds=time.time() - t0,
                preds=[preds[i] for i in range(len(items))])


ANSWER = re.compile(r"####\s*-?[\d,\.]+[^\d,\.]")   # a complete '#### N': gsm8k_pred no longer changes


def run_gsm8k(m, tok, variant, limit):
    items = gsm8k_items(limit)
    B = B_FO if variant == "fo" else B_HF
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))

    def stop(b, ids):
        if len(ids) % 8:
            return False
        if "Question:" in tok.decode(ids[-8:]):
            return True
        return ANS_STOP and bool(ANSWER.search(tok.decode(ids).split("Question:")[0]))
    correct, t0, outs, ntok = 0, time.time(), {}, 0

    def record(i, g, u):
        nonlocal correct, ntok
        text = tok.decode(g)
        ntok += len(g)
        p = gsm8k_pred(text)
        ok = int(p != "" and abs(float(p) - float(items[i][1])) < 1e-6) if re.fullmatch(r"-?\d+\.?\d*", p or "x") else 0
        correct += ok
        outs[i] = dict(pred=p, ok=ok, undef=u, text=text)
        if len(outs) % B == 0 or len(outs) == len(items):
            print(f"  gsm8k {len(outs)}/{len(order)} acc={correct / len(outs):.4f} {time.time() - t0:.0f}s "
                  f"{ntok / (time.time() - t0):.1f} gen tok/s, F_O undefined on {sum(o['undef'] for o in outs.values())}",
                  flush=True)
    if SERVE:
        m.serve([tok(items[i][0])["input_ids"] for i in range(len(items))], GEN, stop, record)
    else:
        for k in range(0, len(order), B):
            idx = order[k:k + B]
            prompts = [tok(items[i][0])["input_ids"] for i in idx]
            gens = m.generate(prompts, GEN, stop)
            undef = getattr(m, "undef", [False] * len(gens))
            for i, g, u in zip(idx, gens, undef):
                record(i, g, u)
    und = [i for i in range(len(items)) if outs[i]["undef"]]
    return dict(n=len(items), acc=correct / len(items), seconds=time.time() - t0, gen_tokens=ntok,
                undefined=len(und), undefined_ids=und, acc_defined=sum(outs[i]["ok"] for i in range(len(items))
                                                                      if not outs[i]["undef"]) / max(1, len(items) - len(und)),
                preds=[outs[i]["pred"] for i in range(len(items))],
                **(dict(texts=[outs[i]["text"] for i in range(len(items))]) if DETAIL else {}))


def main():
    from transformers import AutoTokenizer
    variant, task = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    m = FOModel() if variant == "fo" else HFModel(variant, tok)
    res = {"wiki": lambda: run_wiki(m, tok, variant), "mmlu": lambda: run_mmlu(m, tok, variant, limit),
           "gsm8k": lambda: run_gsm8k(m, tok, variant, limit)}[task]()
    res.update(variant=variant, task=task, limit=limit)
    with open(f"{OUT}/acc_{variant}_{task}{'' if limit is None else f'_{limit}'}.json", "w") as f:
        json.dump(res, f)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "preds"}), flush=True)


if __name__ == "__main__":
    main()
