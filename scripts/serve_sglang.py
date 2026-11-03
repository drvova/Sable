"""BF16 and FP8 serving of Qwen2.5-32B-Instruct in SGLang (offline engine), for comparison with
scripts/serve.py: the same random prompts, greedy decoding of exactly NGEN tokens.

usage: <sglang venv>/bin/python -m scripts.serve_sglang VARIANT B PROMPT NGEN     VARIANT in {bf16, fp8}
fp8: SGLang's FP8 W8A8 (per-tensor weight scales computed at load, dynamic activation scales).
Prints the time to first token of the batch (prefill and one token), the decode step and tokens/s."""
import sys
import time
import torch

from scripts import paths


def main():
    import sglang as sgl
    variant, B, S, G = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 100000, (S,), generator=g).tolist() for _ in range(B)]
    eng = sgl.Engine(model_path=paths.hf_snapshot(), quantization="fp8" if variant == "fp8" else None, dtype="bfloat16",
                     mem_fraction_static=0.88, max_running_requests=B, context_length=S + G + 16,
                     disable_radix_cache=True, chunked_prefill_size=S, log_level="error")

    def run(ps, n):
        t0 = time.time()
        out = eng.generate(input_ids=ps, sampling_params=dict(max_new_tokens=n, min_new_tokens=n, temperature=0.0,
                                                              ignore_eos=True))
        dt = time.time() - t0
        assert all(len(o["output_ids"]) == n for o in out)
        return dt
    info = eng.get_server_info()
    print("SGLANG max_total_num_tokens", info.get("max_total_num_tokens"), "max_running_requests",
          info.get("max_running_requests"), flush=True)
    run([p[:64] for p in prompts], 4)                    # warm-up (CUDA graphs, kernels)
    run(prompts, 1)
    ttft = run(prompts, 1)
    tot = run(prompts, G)
    step = (tot - ttft) / max(1, G - 1)
    print(f"RESULT sglang-{variant} B={B} prompt={S} gen={G}: ttft {ttft:.2f}s, decode step {step * 1e3:.1f} ms "
          f"({B / step:.1f} tok/s), end to end {B * G / tot:.1f} gen tok/s", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
