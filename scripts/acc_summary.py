"""Summary of scripts/acc_attribution.py outputs: perplexity with a bootstrap 95% confidence interval
over windows (2,000 resamples, seed 0), top-1 accuracy, and agreement of the greedy predictions with
BF16 (when DIR/bf16.npy exists).

usage: python -m scripts.acc_summary DIR"""
import json
import math
import os
import sys
import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    d = sys.argv[1]
    ref = np.load(f"{d}/bf16.npy") if os.path.exists(f"{d}/bf16.npy") else None
    rng = np.random.default_rng(0)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(f"{d}/{f}"))
        wn = np.array(r["window_nll"])
        bs = [math.exp(wn[rng.integers(0, len(wn), len(wn))].sum() / r["tokens"]) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        agree = ""
        if ref is not None and r["name"] != "bf16" and os.path.exists(f"{d}/{r['name']}.npy"):
            p = np.load(f"{d}/{r['name']}.npy")
            agree = f"agree={np.mean(p == ref) * 100:.2f}%"
        print(f"{r['name']:24s} tok={r['tokens']} ppl={r['ppl']:.4f} CI=[{lo:.4f},{hi:.4f}] "
              f"top1={r['top1'] * 100:.2f}% {agree}")


if __name__ == "__main__":
    main()
