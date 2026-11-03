"""Decoding throughput of F_O (zk/fofast.py: compiled, CUDA-graph decode; zk/evalfo.py: reference engine)
against BF16 Qwen2.5-32B-Instruct
(Hugging Face transformers, eager, greedy), same GPU, random prompts.

usage: python -m scripts.serve VARIANT B PROMPT NGEN      VARIANT in {fofast, fo, bf16}
Prints time to first token (prefill of the batch), decode tokens/s and total generated tokens/s."""
import sys
import time
import torch

from scripts import paths


def main():
    variant, B, S, G = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 100000, (S,), generator=g).tolist() for _ in range(B)]
    if variant == "fofast":
        from zk import model as mdl, fofast
        M = mdl.load_model(paths.fo_model())
        eng = fofast.FastEngine(M, S + G + 1, B)
        caches = eng.new_caches()                                   # one set: the graph is captured on it
        eng.generate(prompts, 2, caches)                            # compile at full shapes, capture, warm up
        out, ttft, dec = eng.generate(prompts, G, caches)
        step = dec / max(1, G - 1)
        print(f"RESULT {variant} B={B} prompt={S} gen={G}: ttft {ttft:.2f}s, decode step {step * 1e3:.1f} ms "
              f"({B / step:.1f} tok/s), end to end {B * G / (ttft + dec):.1f} gen tok/s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return
    if variant == "fo":
        from zk import model as mdl, evalfo
        M = mdl.load_model(paths.fo_model())
        eng = evalfo.Engine(M, S + G + 1)
        eng.generate([p[:64] for p in prompts[:1]], 2)              # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        caches = eng.new_caches(B, S + 1)
        h = eng.prefill(torch.tensor(prompts).cuda(), caches)
        eng.logits(h[:, -1])
        torch.cuda.synchronize()
        ttft = time.time() - t0
        del caches, h
        torch.cuda.synchronize()
        t0 = time.time()
        eng.generate(prompts, G)
        torch.cuda.synchronize()
        tot = time.time() - t0
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(paths.hf_snapshot(), dtype=torch.bfloat16).cuda().eval()
        x = torch.tensor(prompts).cuda()
        with torch.no_grad():
            model.generate(x[:1, :64], max_new_tokens=2, do_sample=False)
            torch.cuda.synchronize()
            t0 = time.time()
            model.lm_head(model.model(x).last_hidden_state[:, -1])
            torch.cuda.synchronize()
            ttft = time.time() - t0
            t0 = time.time()
            model.generate(x, max_new_tokens=G, min_new_tokens=G, do_sample=False)
            torch.cuda.synchronize()
            tot = time.time() - t0
    dec = (tot - ttft) / max(1, G - 1)
    print(f"RESULT {variant} B={B} prompt={S} gen={G}: ttft {ttft:.2f}s, decode step {dec * 1e3:.1f} ms "
          f"({B / dec:.1f} tok/s), end to end {B * G / tot:.1f} gen tok/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
