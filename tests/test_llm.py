"""End-to-end prove/verify of the layer protocol on a tiny Qwen-shaped model."""
import sys
import time
import torch
from zk import model as mdl
from zk import llm
from tests.tiny import tiny_get


def qwen_tiny(S=256, seed=0):
    cfg = mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024)
    M = mdl.quantize_weights(cfg, tiny_get(cfg, seed))
    toks = torch.randint(0, cfg.V, (1, S), generator=torch.Generator().manual_seed(1))
    mdl.calibrate(M, toks)
    return M


def statement(M, B, S, ngen, seed=2):
    """Random prompt of S - ngen + 1 tokens, then greedy decoding (Policy(): G = G_max = ngen)."""
    tokens = torch.randint(0, M.cfg.V, (B, S), generator=torch.Generator().manual_seed(seed)).cuda()
    return llm.greedy_decode(M, tokens, ngen)


def main(B=2, S=256, ngen=16):
    M = qwen_tiny()
    sh = llm.Shp(M.cfg, B, S, 7, 7, ngen)
    tokens, targets = statement(M, B, S, ngen)
    t0 = time.time()
    msgs = llm.prove(M, sh, tokens, targets)
    t1 = time.time()
    print(f"prove {t1 - t0:.1f}s, {len(msgs)} msgs, {llm.LAST['bytes'] / 2**20:.1f} MiB")
    print({k: round(v, 2) for k, v in sorted(llm.LAST["times"].items())})
    llm.verify(M, sh, tokens, targets, msgs)
    print(f"verify ok {time.time() - t1:.1f}s")
    # a fed-back token (public check) and the last token (only the proof binds it)
    for i in (3, ngen - 1):
        bad = targets.clone()
        bad[0, i] = (bad[0, i] + 1) % M.cfg.V
        try:
            llm.verify(M, sh, tokens, bad, msgs)
        except Exception as e:                     # noqa: BLE001
            print(f"tampered target {i} rejected:", type(e).__name__, str(e)[:100])
        else:
            raise SystemExit(f"tampered target {i} accepted")


def test_llm():
    main()


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:4]))
