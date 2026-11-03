"""Generate answers to two chat prompts with F_O (model.generate, the reference implementation), as a
quick check that a quantized model file is sane.

usage: python -m scripts.chat_demo"""
import time
import torch
from transformers import AutoTokenizer
from zk import model as mdl
from scripts import paths


def main():
    tok = AutoTokenizer.from_pretrained(paths.hf_snapshot())
    t0 = time.time()
    M = mdl.load_model(paths.fo_model())
    print("loaded", time.time() - t0, flush=True)
    for q in ["Explain in two sentences why the sky is blue.",
              "Write a Python function that checks whether a number is prime."]:
        msgs = [{"role": "user", "content": q}]
        ids = torch.tensor(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))["input_ids"])
        t0 = time.time()
        y = mdl.generate(M, ids, 60, len(ids) + 64)
        print(repr(tok.decode(y)), time.time() - t0, flush=True)
    print(mdl.STATS)


if __name__ == "__main__":
    main()
