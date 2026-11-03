"""F_O calibrated on WikiText-2 instead of the spec text (the calibration-corpus ablation of the paper's
evaluation): the same quantization as scripts/quantize.py, with the static exponents calibrated on the
first CAL_NB x 1,024 tokens of the WikiText-2 train split (default CAL_NB = 8).

usage: python -m scripts.quantize_calib OUT
Evaluate the result with SABLE_FO_MODEL=OUT python -m scripts.acc_attribution OUTDIR fo_calwt."""
import os
import sys
import torch
from scripts import paths
from scripts.quantize import quantize


def wikitext2_train_ids(tok):
    nb = int(os.environ.get("CAL_NB", "8"))
    ids = tok(paths.wikitext2("train"))["input_ids"][:nb * 1024]
    return torch.tensor(ids).view(nb, 1024)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    quantize(wikitext2_train_ids, sys.argv[1])


if __name__ == "__main__":
    main()
