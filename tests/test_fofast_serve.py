"""Continuous batching in the serving engine (FastEngine.serve) is F_O sequence by sequence: on tiny Qwen3
(QK-norm) and Qwen3-MoE models, more prompts than slots, with and without a shared prefix (prefilled once
and copied into each slot), and with per-prompt stops, give evalfo.Engine's greedy tokens for each prompt
alone, cut at its stop, with no row flagged."""
import torch
from zk import model as mdl, fofast
from zk.evalfo import Engine
from tests.test_llm_qwen3 import qwen3_tiny
from tests.test_llm_moe import moe_tiny

NGEN = 12


def prompts_for(V, shared, seed=0):
    g = torch.Generator().manual_seed(seed)
    pre = torch.randint(0, V, (200,), generator=g).tolist() if shared else []
    return [pre + torch.randint(0, V, (60 + 23 * i - (0 if shared else -150),), generator=g).tolist()
            for i in range(5)]


def stop(i, ids):
    return i % 2 == 1 and len(ids) == 3 + i          # prompts 1 and 3 stop early; the others run to NGEN


def run(M):
    for shared in (True, False):
        prompts = prompts_for(M.cfg.V, shared)
        ref = Engine(M, 512)
        want = []
        for i, q in enumerate(prompts):
            g = ref.generate([q], NGEN)[0]
            n = next((m for m in range(1, NGEN + 1) if stop(i, g[:m])), NGEN)
            want.append(g[:n])
        fe = fofast.FastEngine(M, 512, 2)
        got = fe.serve(prompts, NGEN, stop, every=4)
        assert [g for g, _ in got] == want, (shared, want, got)
        assert not any(u for _, u in got)


def test_serve():
    run(qwen3_tiny(mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True, qkv_bias=False)))
    run(moe_tiny())


if __name__ == "__main__":
    test_serve()
    print("serve equals evalfo per prompt (shared prefix and none, stops, 5 prompts on 2 slots)")
