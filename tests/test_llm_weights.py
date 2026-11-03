"""A proof made with different weights must not verify against the committed model C_W."""
import copy
import torch
from zk import llm
from tests.test_llm import qwen_tiny, statement


def main():
    M = qwen_tiny()
    B, S, ngen = 1, 256, 16
    sh = llm.Shp(M.cfg, B, S, 7, 7, ngen)
    cw = llm.weights_digest(M, sh)
    # the cheating prover changes one MLP weight (same shapes and exponents)
    M2 = copy.copy(M)
    M2.layers = list(M.layers)
    Ly = copy.copy(M.layers[0])
    Ly.Wd = Ly.Wd.clone()
    Ly.Wd.view(-1)[0] = 127 if int(Ly.Wd.view(-1)[0]) != 127 else -127
    M2.layers[0] = Ly
    tokens, targets = statement(M2, B, S, ngen)
    msgs = llm.prove(M2, sh, tokens, targets, C_W=cw)
    try:
        llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
    except Exception as e:                     # noqa: BLE001
        print("proof with other weights rejected:", type(e).__name__, str(e)[:100])
    else:
        raise SystemExit("proof with other weights ACCEPTED")
    # the honest proof with the committed weights verifies
    tokens, targets = statement(M, B, S, ngen)
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
    print("honest proof verifies")


def test_llm_weights():
    torch.manual_seed(0)
    main()


if __name__ == "__main__":
    test_llm_weights()
