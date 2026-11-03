# Sable

Sable (Small-field Argument for Bit-exact LLM Execution) proves that a language model with committed
weights produced a given reply to a given prompt, bit for bit. A client that receives the reply and the
proof can check, without the GPU that ran the model, that every token of the reply, including where it
stopped, is what the committed model computes.

Sable has two parts:

- **F_O, an exact integer version of the model.** Every rounding, normalization, softmax and sampling
  decision is an integer operation fixed by its inputs, so the model has exactly one output for each input
  and every GPU computes the same one.
- **A proof system for F_O.** It is built on sumcheck and LogUp-GKR lookups over the 31-bit KoalaBear field
  (p = 2^31 - 2^24 + 1), with a hash-based polynomial commitment (rate-1/2 Reed-Solomon code, BLAKE3
  Merkle trees and a BaseFold opening over F_p^8). Its constraints provably never wrap modulo p, and its
  prover streams the model layer by layer on a single GPU.

The paper in [`paper/`](paper/) (`paper/sable.pdf`) describes the integer function, the constraint system,
the protocol and its soundness analysis, the GPU prover and the measurements in full.

## Results

All numbers are for one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB). Each proof covers the whole prompt and
16 generated tokens, and the cost depends on the total length.

| Model | 1 x 4,096 tokens | 4 x 4,096 tokens | 1 x 16,384 tokens |
|---|---|---|---|
| Qwen2.5-32B-Instruct (64 layers) | 92.5 s, 135 MiB | 253.1 s, 239 MiB | 397.5 s, 346 MiB |
| Qwen3-32B (64 layers, QK-norm) | 121.8 s, 165 MiB | 485.7 s, 298 MiB | 668.7 s, 477 MiB |
| Qwen3-30B-A3B (48 layers, 128 experts, 8 per token) | 84.7 s, 106 MiB | 181.9 s, 153 MiB | 265.7 s, 220 MiB |

Each entry gives the proving time (mean of three runs after a warm-up) and the proof size.

- **Verification** takes 19 to 56 s on a single CPU core, with no GPU.
- **Soundness error** is 2^-103 per round in the random-oracle model, with proven proximity bounds and a
  30-bit proof of work.
- **Cost.** A proof takes about 41 to 50 times as long as computing the same tokens with the serving engine
  for F_O in this repository. For a 4,096-token request on Qwen2.5-32B that comes to about 6.4 cents at
  public GPU prices.

Exactness costs some accuracy against the BF16 checkpoint:

| Model | Variant | WikiText-2 perplexity | MMLU | GSM8K |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | BF16 / F_O | 5.151 / 5.580 | 83.2% / 81.8% | 88.6% / 88.0% |
| Qwen3-32B | BF16 / F_O | 7.444 / 7.933 | 81.8% / 80.4% | 93.9% / 92.2% |
| Qwen3-30B-A3B | BF16 / F_O | 8.455 / 8.618 | 79.6% / 78.0% | 92.7% / 91.4% |

The evaluations use these sets:

- **WikiText-2:** the full test split.
- **MMLU:** all 14,042 questions, 5-shot.
- **GSM8K:** all 1,319 problems, 8-shot and greedy, with replies of up to 256 tokens for Qwen2.5 and 1,024
  for Qwen3.

The paper also compares F_O with FP8 and SmoothQuant INT8, attributes the loss to individual integer steps,
and measures where F_O is undefined.

## What a proof states

A statement is `(C_W, policy, (x_b, y_b) for b < B, n_c)`, with these parts:

- `C_W`, the commitment to the model's weights;
- a public policy: temperature, stop set, maximum output length, context bound, and tokenizer and template
  digests;
- `B` prompts `x_b` of equal length, with their generated replies `y_b`;
- a client nonce `n_c`.

The proof shows that each `y_b` is the autoregressive output of F_O with the committed weights on `x_b`,
including the stopping rule. Decoding is greedy, or samples at a positive temperature from a seed derived
from the nonce, the commitment, the prompts and the policy, so the server cannot choose the randomness.

`C_W` is a Merkle root over the quantized weights. Anyone can recompute it from the public checkpoint with
the deterministic quantization program in `scripts/quantize.py`.

## Limitations

- **Not zero-knowledge.** A proof reveals evaluations of the trace at random positions, and about 2 x 10^4
  proofs together reveal the weights. The system as released is meant for open-weight models and for
  verifiers entitled to see the prompts in a batch.
- **Large proofs.** Proofs are 100 to 500 MiB, which suits auditors more than every client. Compressing them
  by proving the verifier recursively is future work; `zk/recur/` holds its first component, a BLAKE3
  circuit.
- **Accuracy.** F_O is a different function from the BF16 model, and loses some accuracy against it (see
  the table above).
- **One GPU size.** The prover targets one large GPU. The measured shapes fit in 96 GB, and larger batches
  or contexts run out of memory.

## Requirements

- **GPU.** An NVIDIA GPU. Sable is developed and measured on an RTX PRO 6000 Blackwell (sm_120). Kernels
  compile for the current device (`SABLE_CUDA_ARCH` overrides this); other architectures are untested.
  Full-model proofs use 18 to 82 GB of GPU memory and 72 to 175 GB of host memory, depending on the model
  and shape.
- **System.** Linux, a CUDA 13 driver, and a host C++ compiler (`g++`) for the verifier core and the
  transcript hash.
- **Python.** Python 3.12 or later, with PyTorch (CUDA build), CuPy (`cupy-cuda13x`), NumPy, blake3,
  mpmath and ninja. `requirements.txt` pins the exact environment of the paper's measurements.

## Install

    python -m venv .venv && . .venv/bin/activate
    pip install -e .            # the zk package (prover and verifier)
    pip install -e .[eval]      # plus transformers, datasets, ... for scripts/ (quantization, accuracy)
    pip install -e .[dev]       # plus pytest and ruff

Run all commands from the repository root.

## Quickstart

Prove and verify a tiny random two-layer model; no checkpoint is needed:

    python -m tests.test_llm

Run the full test suite on tiny models, including forgery tests that must be rejected:

    pytest                      # or one process per file: python -m tests.<name>

## Proving a real model

1. Download a supported checkpoint, then quantize it into an F_O model file. The quantization is
   deterministic: two runs give bit-identical files.

       hf download Qwen/Qwen3-32B
       export SABLE_ARCH=Qwen/Qwen3-32B SABLE_FO_MODEL=/data/qwen3_32b_FO.pt
       python -m scripts.quantize

   Supported values of `SABLE_ARCH`: `Qwen/Qwen2.5-32B-Instruct` (the default), `Qwen/Qwen3-32B` and
   `Qwen/Qwen3-30B-A3B`.

2. Prove, verify and time. The prompt is random; the run makes a warm-up proof, then three timed proofs,
   and verifies the first one:

       BENCH_OUT=bench.jsonl BENCH_SAVE=/data/proof_1x4096 python -m scripts.bench_e2e 1 4096 rand:0:3:verify

   `BENCH_SAVE` writes the proof (`.proof`, in the byte format of `zk/proofio.py`) and the verifier's
   inputs (`.pkl`).

3. Verify the saved proof on the host, with no GPU:

       CUDA_VISIBLE_DEVICES= python -m scripts.verify_cpu /data/proof_1x4096

From Python, the steps are those of `scripts/bench_e2e.py`:

```python
import torch
from scripts import paths
from zk import model as mdl, llm, proofio

M = mdl.load_model(paths.fo_model(), layers_on="cpu")       # the F_O model
B, S, ngen = 1, 4096, 16
sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), ngen)     # proof shape
cw = llm.weights_digest(M, sh)                             # C_W, once per model release
policy, nonce = llm.Policy(theta=0), bytes(32)             # greedy; the client's nonce
prompt = torch.randint(0, M.cfg.V, (B, S))                  # tokens[:, :S - ngen + 1] is the prompt
tokens, targets = llm.decode(M, prompt.cuda(), ngen, policy)
msgs = llm.prove(M, sh, tokens, targets, C_W=cw, policy=policy, nc=nonce)
blob = proofio.dumps(msgs)                                   # the proof, as bytes
llm.verify(M, sh, tokens, targets, proofio.loads(blob), C_W=cw, policy=policy, nc=nonce)
```

The kernels are generated per shape and compiled on first use. With an empty cache, the first proof
spends about 25 minutes compiling. Compiled kernels are cached in `~/.cupy/kernel_cache` (or
`CUPY_CACHE_DIR`; several GB) and reused by later runs.

## Reproducing the paper

| Paper section | Command |
|---|---|
| End-to-end proving (all three models) | `python -m scripts.bench_e2e B S rand:0:3[:verify]` with `SABLE_ARCH`, `SABLE_FO_MODEL` (`ZK_GRAPH=0` for Qwen3-32B at 4 x 4K and 16K) |
| Host verification | `python -m scripts.verify_cpu PREFIX` |
| Per-layer profiles | `python -m scripts.bench L B S 16 [verify]` (the first L layers) |
| Accuracy | `python -m scripts.acc {fo,bf16,fp8,sq8} {wiki,mmlu,gsm8k}` (`ACC_MAXTOK=0`: the whole WikiText-2 test split) |
| GSM8K for Qwen3 | `ACC_GEN=1024 ACC_SERVE=1 ACC_ANSWER_STOP=1 ACC_SMAX=2560 python -m scripts.acc VARIANT gsm8k` |
| Accuracy attribution | `python -m scripts.acc_attribution` (Qwen2.5), `python -m scripts.acc_attr3` (Qwen3), `python -m scripts.acc_summary` |
| Calibration ablation | `python -m scripts.quantize_calib OUT` |
| Where F_O is undefined | `python -m scripts.fo_domain_scan OUT.json` |
| Serving throughput | `python -m scripts.serve fofast B PROMPT GEN`; SGLang baselines: `scripts/serve_sglang.py` |
| BLAKE3 circuit | `python -m scripts.b3bench` |
| Cost model | `python costmodel/costmodel.py`, `python costmodel/costmodel_blackwell.py` |

Each script's docstring gives its options.

## Environment

| Variable | Meaning |
|---|---|
| `SABLE_ARCH` | the model, as its Hugging Face repository (default `Qwen/Qwen2.5-32B-Instruct`) |
| `SABLE_FO_MODEL` | the F_O model file (written by `scripts.quantize`, read by the other scripts) |
| `SABLE_HF_SNAPSHOT` | checkpoint directory. If unset, the checkpoint of `SABLE_ARCH` is looked up in the local Hugging Face cache (`HF_HOME`) and never downloaded implicitly |
| `SABLE_DATA` | optional cache root for the evaluation datasets (`$SABLE_DATA/datasets_<name>`) |
| `SABLE_CUDA_ARCH` | kernel target, for example `sm_120` (default: the current device) |
| `SABLE_BUILD_DIR` | build directory of the PyTorch extension (default `build/`) |
| `CUPY_CACHE_DIR` | NVRTC kernel cache (default `~/.cupy/kernel_cache`) |
| `ZK_GRAPH=0` | do not replay the middle layers from a captured CUDA graph (same proof, more time, less memory) |
| `ZK_DEVSESSION=0` | keep the prover's Fiat-Shamir transcript on the host (same proof, slower) |
| `ZK_NVTX=1`, `ZK_SCLOG=1`, `ZK_DOMT=1` | profiling: NVTX ranges, per-sumcheck and per-domain times |

The prover's other `ZK_*` switches select memory and performance trade-offs. Their defaults are the
configuration measured in the paper.

## Security notes

- **Exchanging proofs.** Use `zk.proofio` (`dumps` and `loads`), which parses a strict byte format and
  rejects malformed input. Some scripts and tests write pickles for local experiments; never unpickle data
  from an untrusted source.
- **The model registry entry.** This is the weight commitment `C_W`, the model configuration and the F_O
  constants. The verifier must get it from the model release, by recomputing it from the published
  checkpoint and quantization program, and never from the server whose proof it checks. `llm.verify`
  checks the statement and policy, then compares `C_W` and the digests of the fixed tables with these
  expected values before any sumcheck.
- **Batches.** A batch proof opens values of every sequence, so a batch should hold one client's sequences,
  or be verified only by a party entitled to see all of them.
- **Where F_O is undefined.** The honest prover raises `zk.model.WitnessError` there: for example, an
  activation shift outside its range. `scripts/fo_domain_scan.py` measures how often that happens; it never
  did on the paper's prompt sets.
- **No `python -O`.** Some checks are `assert` statements, so the package refuses to import under it.

## Layout

| Path | Contents |
|---|---|
| `zk/llm.py` | the protocol for a transformer: statement, policy, witness, per-layer steps, both prover passes, verifier |
| `zk/model.py` | the integer model F_O, its quantization and calibration |
| `zk/moe.py` | mixture-of-experts layers (router, top-k selection, experts) |
| `zk/evalfo.py`, `zk/fofast.py`, `zk/foattn.py` | evaluation and serving engines for F_O (exact, not part of the prover) |
| `zk/domain.py`, `zk/gkr.py`, `zk/logup.py`, `zk/sumcheck.py`, `zk/proto.py` | domains, LogUp-GKR, chunked lookups, sumchecks (kernels generated and compiled with NVRTC), protocol context |
| `zk/gpcs.py`, `zk/codec.py`, `zk/rs.py` | the polynomial commitment, encoding and Merkle trees |
| `zk/transcript.py`, `zk/devtr.py`, `zk/b3s.py` | the Fiat-Shamir transcript, on the host and resident on the GPU |
| `zk/vcore.py`, `zk/cpuops.py` | the compiled host verifier core (`llm.verify(..., cpu=True)`) |
| `zk/proofio.py`, `zk/tables.py` | the proof byte format; the LogUp table |
| `zk/recur/` | the BLAKE3 circuit for a recursive verifier |
| `zk/cuda/` | CUDA and C++ sources |
| `tests/` | unit, end-to-end and forgery tests on tiny models |
| `scripts/` | quantization, benchmarks and accuracy (`scripts/paths.py` resolves the inputs) |
| `costmodel/` | the analytical cost model the prover is measured against |
| `docs/spec/corrected.tex` | an early design specification of the system; its first 1,024 tokens are the F_O calibration text |
| `paper/sable.pdf` | the paper |

## Citation

    @misc{sable2026,
      author = {Tomer Solberg},
      title  = {Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models},
      year   = {2026},
      note   = {\url{https://github.com/moonmath-ai/Sable}}
    }

## Acknowledgements

We thank Omer Shlomovits, Miki Asa, Karthik Inbasekar and Suyash Bagad for useful discussions and comments.

## License

MIT (see `LICENSE`), copyright 2026 MoonMath.ai. Contact: research@moonmath.ai.
