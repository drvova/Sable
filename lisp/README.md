# Sable in Common Lisp (SBCL)

A port of Sable's host-side verifier from Python (`zk/`) to Common Lisp, built into an SBCL
image, with a CPU reference prover for every protocol it verifies. It has no dependencies beyond
SBCL and the ASDF and `sb-introspect` that ship with it.

The interface is Lisp code itself:

- **Polynomials and sumcheck statements are ordinary forms.** Constraints and statements are
  written as s-expressions, such as `(* (+ c0 3) (- c1 c0) c2)` or `(* t0 t1 t2)`. The code
  normalises, measures, renames and compiles them; there is no polynomial class.
- **The field tower is generated, not hand-written.** It is defined once, recursively, and turned
  into compiled code for any degree.
- **The executable evaluates forms.** There is no table of commands; the exported functions are
  the commands.

Everything is checked bit for bit against the Python implementation. `tests/gen-vectors.py` runs
the Python code and writes `tests/vectors.sexp`, and the Lisp tests reproduce every value in it.
That covers field arithmetic, BLAKE3 in both modes, transcript challenges, proof bytes,
polynomials, Reed-Solomon codewords and folds, and whole sumcheck proofs. For the sumcheck, the
Lisp prover writes the same proof bytes as the Python prover, and the Lisp verifier accepts the
Python proofs. The larger protocols are checked the other way round: `make crosscheck` has the
original verifiers check the Lisp prover's proofs, and every forgery of them.

## Status

| Part | State | Checked against |
|---|---|---|
| Field tower, BLAKE3, transcript, proof format, Reed-Solomon | ported | `tests/vectors.sexp` from the Python code |
| Sumcheck (verifier and prover) | ported, optimised (steps 1–6 of the plan) | Python's proofs, byte for byte |
| LogUp-GKR | ported | `vc_gkr_upper` in `zk/cuda/vcore.cpp`: 8 shapes, 231 forgeries |
| Domains | ported | `zk/domain.py`'s verifier: 16 domains |
| Polynomial commitment (BaseFold) | ported | `zk/gpcs.py`'s verifier: 3 scenarios, 287 forgeries |
| Statement, policy, sampling seed, C_W, commitment layout | ported | `zk/llm.py`: 18 layouts, 9 statements, 10 checks |
| Per-layer steps of `llm.verify` | not yet | a host-verifier trace from `tests/gpu-trace.py` (needs a GPU run) |
| GPU prover | not planned | |

So the Lisp code cannot yet verify a whole proof of a model: every protocol that proof is made
of is ported and checked, but not the per-layer steps that put them together. The plan behind the
optimisations, and the papers it draws on, is in
[`reports/Sable Lisp optimization research.md`](../reports/Sable%20Lisp%20optimization%20research.md).

## Forms

A form is an integer, a variable (any symbol), `(+ f ...)`, `(- f ...)`, `(* f ...)` or
`(expt f k)`.

```lisp
(normalize '(* (+ c0 3) (- c1 c0) c2))
;; => (+ (* -3 c0 c2) (* 3 c1 c2) (* -1 c0 c0 c2) (* c0 c1 c2))
(form-degree '(* eq t0 t1) '(t0 t1))       ; => 2, the degree in the tables
(form= '(expt (+ x 1) 2) '(+ (* x x) (* 2 x) 1))   ; => T
(sublis '((c0 . c5)) form)                 ; renaming is SUBLIS
(evaluate '(* a (+ b 1)) `((a . ,x) (b . ,y)) :degree 8)   ; over F_p^8
(arithmetic-form '(* a (expt b 2)) :degree 8)   ; => (EXT* A (EXT-EXPT B 2)), the code it runs
```

`compile-form` turns a form into a compiled function, once per form. A sumcheck statement is a
form over table variables, with optional constants such as challenges. That one form drives the
prover, the verifier and the final check:

```lisp
(let ((tables `((t0 . ,table0) (t1 . ,table1)))
      (statement '(+ (* alpha t0 t1) (* 3 t1))))
  (multiple-value-bind (challenges finals)
      (prove prover n statement tables :eq r :constants `((alpha . ,alpha)))
    ...)
  (check-final (nth-value 1 (verify verifier n statement claim :eq r :constants ...))
               statement finals :constants ...))
```

A table is a vector of 2^n base elements (integers) or extension elements. The prover stores
tables flat. A base table stays in the base field until its first fold, which then costs D base
products per entry instead of a full extension product.

Each round runs a kernel compiled for the statement. `round-kernel-code` returns it as a `lambda`
form you can read:

- **Unrolled:** the loop over table pairs has the statement inlined, unrolled over the tables and
  the evaluation points.
- **Line values by addition:** the line values at X = 0, 2, 3, ... each come from the previous one
  with a single addition.
- **No allocation in the loop:** every intermediate value has a buffer allocated once per round
  (`in-place-code` in `forms.lisp`).
- **Base-field first round:** when every table is still in the base field and there are no
  constants, the statement runs in base-field arithmetic and each term costs one base-times-extension
  product by its eq weight.

The eq weights of the remaining variables are split in two (`split-weights`, after Dao and Thaler,
ePrint 2024/1210). The weight of pair x = h·|low| + l is low[l]·high[h], each table over half of
the variables. So a round builds two tables of about √N entries instead of one of N/2. The kernel
sums the inner loop over l with the low weights, and multiplies each block by its high weight once
per point.

Compiling a kernel takes 8–20 ms, once per statement. A statement whose kernel would exceed
`*kernel-size-limit*` uses the generic loop instead, and a test checks that both give the same
proofs.

## Performance

The sumcheck prover after each step of the plan. Best of 7 runs, F_p^8:

| Prover run | Field rewrite (steps 1–2) | Flat tables (step 3) | Round kernels (step 4) | Split eq (step 5) |
|---|---|---|---|---|
| 3 extension tables, 14 variables | 0.040 s, 50.6 MB allocated | 0.028 s, 10.1 MB | 0.024 s, 5.4 MB | 0.024 s, 3.4 MB |
| 3 base tables, 14 variables | not supported | 0.028 s, 8.7 MB | 0.016 s, 4.0 MB | 0.012 s, 2.0 MB |
| 2 extension tables, 16 variables | 0.092 s, 106 MB | 0.072 s, 23.2 MB | 0.064–0.068 s, 16.9 MB | 0.052–0.056 s, 8.7 MB |
| 2 base tables, 16 variables | not supported | 0.060 s, 19.5 MB | 0.044–0.048 s, 13.2 MB | 0.036 s, 5.0 MB |

These runs are close to the timer's 4 ms resolution. On 18 variables (best of 3), the split eq
takes 3 extension tables from 0.452 s to 0.424 s, and 3 base tables from 0.292 s to 0.244 s.

**Small values (step 6).** While every table is in the base field and the statement has no
constants, the first ℓ0 rounds (`*small-value-rounds*`, 3 by default) don't fold the tables.
This follows Bagad, Dao, Domb and Thaler, ePrint 2025/1117.

1. One pass over the base tables computes accumulators A[z] = Σ_y eq(r_tail, y) F(tables(z, y))
   for every z on the grid {0..d}^ℓ0, where d is the statement's degree in the tables. The
   statement is evaluated in base-field arithmetic, and only the eq weighting is in the extension.
2. Each of the ℓ0 round messages is interpolated from A with Lagrange weights of the challenges
   so far.
3. The tables are then bound to all ℓ0 challenges at once, and the round kernels take over.

The messages are the same field elements as the folding rounds produce, and a test checks that the
proofs are identical byte for byte. `small-value-code` returns the pass as a `lambda` form. The
grid extension is unrolled into in-place `setf`s, so a kernel compiles in about 0.5 s (degree 3,
ℓ0 = 3). Kernels above `*small-value-size-limit*` step down to a smaller ℓ0, or to folding.

Base tables, best of 3:

| Prover run | Folding | Small values, ℓ0 = 3 |
|---|---|---|
| 3 tables, 16 variables, F_p^8 | 0.064 s, 7.4 MB | 0.040 s, 2.6 MB |
| 3 tables, 18 variables, F_p^8 | 0.240 s, 28.8 MB | 0.172 s, 9.8 MB |
| 2 tables, 18 variables, F_p^8 | 0.132 s, 19.4 MB | 0.080 s, 6.6 MB |
| 3 tables, 18 variables, F_p^4 | 0.120 s, 16.1 MB | 0.104 s, 6.6 MB |

That is 1.4–1.65× in F_p^8 and 1.15× in F_p^4. The cost model of ePrint 2024/1046 predicts 1.85×
and 1.31× at degree 2.

## The field tower

The field tower is generated the same way as the sumcheck kernels. `mul*` and `inv*` define
multiplication and inversion once, on symbolic coordinates:

- **`mul*`** uses the fact that the tower of degree D is F_p[w]/(w^D - 3) for its top generator w,
  with the coordinates in bit-reversed order. So it is the schoolbook product. Each output
  coefficient is one sum of D products, reduced once per four products, since four products of
  values below p fit in an unsigned 64-bit word (lazy reduction).
- **`inv*`** recurses over the levels, taking the norm one level down.

`(tower-code 'mul* 8)` returns the resulting straight-line `lambda` form, and `ext*` and `ext-inv`
compile it on first use for whatever degree they are given. Field addition and subtraction have no
branches. Every operation takes an optional `out` vector to write into, so a caller that reuses
buffers allocates nothing.

On this machine (SBCL 2.2.9, a noisy virtual machine):

| Measurement | Karatsuba tower | Now |
|---|---|---|
| F_p8 multiply | 520–550 ns | 77–96 ns, 0 bytes allocated when writing into `out` |
| F_p4 multiply | about 183 ns | about 40 ns |
| Sumcheck demo, 14 variables | 0.19 s | 0.06–0.09 s |
| Sumcheck demo, 16 variables | 0.43 s | 0.23 s |

## What is ported

| Lisp (`src/`) | Python | Contents |
|---|---|---|
| `util.lisp` | | octet vectors, little-endian integers |
| `conditions.lisp` | (`ValueError`, `assert`) | the condition hierarchy, `check-parameter` |
| `field.lisp` | `zk/field.py`, `zk/fe.py` | KoalaBear F_p with branchless addition, the tower (`mul*` as a lazily reduced schoolbook product, `inv*`, `tower-code`), Lagrange interpolation, eq tables, MLE evaluation |
| `forms.lisp` | `zk/poly.py`, the expressions of `zk/sumcheck.py` | arithmetic forms: `expand`, `normalize`, `form-degree`, `arithmetic-form`, `in-place-code`, `compile-form`, `evaluate` |
| `blake3.lisp` | `zk/cuda/b3s.h`, `blake3` | the sequential BLAKE3 mode of the transcript, and standard BLAKE3 |
| `transcript.lisp` | `zk/transcript.py` | the Fiat-Shamir transcript |
| `proofio.lisp` | `zk/proofio.py` | the proof byte format (`SABLEPF1`) |
| `rs.lisp` | `zk/rs.py` | the rate-1/2 Reed-Solomon code |
| `sumcheck.lisp` | `zk/sumcheck.py`, `zk/cuda/vcore.cpp` | the sumcheck of any statement form: verifier, final check, and a reference CPU prover on flat tables, with one compiled round kernel per statement (`round-kernel-code`), split eq weights, and the first rounds of base tables answered from small-value accumulators (`small-value-code`) |
| `gkr.lisp` | `zk/gkr.py`, `vc_gkr_upper` in `zk/cuda/vcore.cpp` | batched LogUp-GKR fractional sums: `verify-upper`, a CPU reference prover `prove-upper`, the layer statement as a form |
| `domain.lisp` | `zk/domain.py` | a domain (columns, zerocheck constraints, lookups) as data; its leaf-layer and zerocheck statement as one form; `verify-domain` and a CPU reference prover `prove-domain` |
| `pcs.lisp` | `zk/gpcs.py` | the polynomial commitment: units and size classes, merged mixed-height Merkle trees with caps, the per-class point reductions, segments, the cross-unit sumcheck, BaseFold with oracles, grinding and the query checks; a CPU reference prover |
| `sha256.lisp` | `hashlib.sha256` | SHA-256 (FIPS 180-4), for the statement's digests |
| `llm.lisp` | the outer layer of `zk/llm.py`, `zk/proto.py` | the transformer protocol's public side: `make-cfg`, `make-shape`, `column-block-bits`, the commitment layout (`commitment-layout`: groups, packing, merge sets, segments, weight groups; `declare-units`), the policy and its encoding, `check-statement`, `statement`, `sampling-seed`, `weights-digest` (C_W) |
| `toplevel.lisp` | | the executable: evaluates forms in `sable-user`; `(help)` |

Python's `Poly` gives layer parameters negative column ids. With symbols as variables that
workaround is not needed: a layer parameter is just another variable.

### LogUp-GKR

`verify-upper` checks the upper layers of the fraction trees, batched over the
lookups:

- **Roots and top tree:** the proof sends each lookup's root fraction, then proves each layer from
  the one above, first over the top tree of block roots, then in-block.
- **Layer statement:** each layer is a sumcheck of one statement form, Σ_l a_l (p0 q1 + p1 q0) +
  a_l λ q0 q1, built by `layer-statement`.
- **Block phase:** each in-block layer sums over the blocks with eq weights, then binds the block
  index in a second sumcheck.

`prove-upper` is a CPU reference prover for the same layers, and the sumcheck prover gained what
it needs for this:
- an in-block phase with given block weights (`:block-weights`)
- an explicit header degree (`:degree`; zk/ writes 0 for a sumcheck of no rounds)
- the folded tables as a third value

**Cross-check against the original C++ verifier.** No GPU is needed for this check.
`make crosscheck` has the Lisp prover make proofs for 8 shapes: 1 to 6 blocks (including
non-powers of two), 1 to 3 lookups, and F_p^4 and F_p^8. It then checks each with
`vc_gkr_upper` from `zk/cuda/vcore.cpp`, the verifier that `llm.verify` uses on the host:

- the C++ verifier accepts every proof and consumes every message
- it computes the same roots, points and claims as the Lisp verifier
- it rejects every proof with one value changed (231 forgeries)

The Lisp tests add LogUp's meaning: each root fraction equals the sum of its leaf fractions, and
the claims equal the multilinear extension of the leaves' parent layer.

### Domains

A domain is K blocks of 2^n elements with base-field columns, zerocheck constraints
and lookups, written as data. The constraints and the lookups' tags, values and numerators are
forms over the column names:

```lisp
(make-domain :name "12.toy" :n 3 :blocks 3
             :columns '((a) (b) (c) (m :kind :row :c 1 :shared t) (kb :kind :row :c 3))
             :constraints '((* a (- a 1)) (- c (* a b)))
             :lookups '((:tag 5 :values (a b)) (:tag 9 :values (kb) :num m)))
```

`verify-domain` replays `domain.verify`:

1. LogUp-GKR over the lookups down to layer n−1.
2. One sumcheck over the column halves, whose statement (`domain-statement`) is a single form: the
   leaf layer of every lookup's tree, plus the zerocheck of every constraint.
3. The column claims at the final point, with the block index bound inside the protocol when the
   domain is merged.

The header's degree follows zk/'s per-round rule (`domain-degree`): a `:row` or `:col` table only
counts in the rounds where it varies.

**Cross-check against Python's verifier.** `make crosscheck PYTHON=...`, given a Python with
PyTorch (the CPU build is enough), checks 16 domain proofs made by `prove-domain` with Python's
own `domain.verify`. It runs in `llm.verify(cpu=True)`'s host mode, on Python's factored form of
the leaf statement and the C++ verifier core. The proofs cover merged and unmerged blocks, 1 to 5
blocks, every column kind, no lookups or no constraints, and F_p^4 and F_p^8.

- Python accepts each proof, consumes every message, and computes the same claims and roots.
- It rejects every changed value except changes to the final of a column that no constraint or
  lookup uses. Such a final only becomes that column's claim, which the commitment's opening
  checks, and the Lisp verifier treats it the same way.

### The polynomial commitment

`pcs.lisp` ports `zk/gpcs.py`, the mixed-length BaseFold opening
over rate-1/2 Reed-Solomon codes and BLAKE3 Merkle trees, with the same calls:
`pcs-declare`, `pcs-merge`, `pcs-segment-decl`, `pcs-commit`, `pcs-open`, `pcs-segment` and
`pcs-finish`.

- **Units and trees:** a unit's columns are grouped by size class, and the classes of a unit or a
  merge set share a mixed-height Merkle tree. Leaf digests are chained 64 columns at a time and
  across the units of a merge set, and each tree sends its cap.
- **Opening:** a sumcheck in F_p^4 reduces each class's claims to one point, in batches when there
  are many points. A segment's units are brought to one point per class.
- **`pcs-finish`:**
  1. a cross-unit sumcheck in F_p^8
  2. BaseFold, with each class joining the folded codeword at its level and oracle trees at even
     levels
  3. grinding
  4. the query openings: every leaf against its tree's cap, then folded to the final table's
     codeword

`make-pcs :prover t` gives a CPU reference prover with the same messages.

**Cross-check against Python's verifier.** `make crosscheck PYTHON=...` checks the Lisp prover's
proofs with zk/gpcs.py's own verifier, on the host, in 3 scenarios:

- a merge set and a segment, 5 size classes, columns split at `cmax`, a tensor below the smallest
  class, and claims sharing points
- `cmax` 10, with two oracle levels and depth-2 paths
- point reductions in batches

Python accepts each proof, consumes every message, and rejects every changed value or byte (287
forgeries). The only messages left unchanged are the grinding nonce, which a few test bits of
grinding may still accept, and the empty paths of trees whose cap is their leaves. The tests use
6 queries and 4 bits of grinding instead of 153 and 30: grinding 30 bits on the CPU takes hours.
Python's verifier takes both as parameters too.

### The transformer protocol

`llm.lisp` ports the parts of `llm.verify` that
come before and around the per-layer steps, with the same bytes:

- the statement (`P|` policy, C_W, h_circ from the configuration's `repr`, the shape and the
  constants digest, H(x), H(y), the nonce, T, the salt), the policy encoding, and the verifier's
  checks on the public statement (`check-statement`, which signals `verification-failure` where
  `Policy.check` raises)
- the sampling seed, and C_W from the weight groups' caps
- the commitment layout: every group of every layer with Python's packing of small blocks into
  columns, the MoE expert groups, the merge sets, the segments and the weight groups;
  `declare-units` declares them on a `make-pcs` verifier in Python's order

`make crosscheck` compares them with `zk/llm.py`: 18 layouts (the tiny test models, and
Qwen2.5-32B, Qwen3-32B and Qwen3-30B-A3B at the paper's shapes, up to 1029 groups and 16830
items), 9 statements, policies and sampling seeds, 10 statement checks and C_W are all equal.

**A trace of the host verifier, for the rest.** `tests/gpu-trace.py` runs on a machine with an
NVIDIA GPU: `prove OUT --model tiny|qwen3|moe` proves a tiny model and saves the proof, the
verifier's setup and the statement inputs; `trace OUT` (CPU only, no GPU needed) reruns
`llm.verify(..., cpu=True)` and writes `trace.jsonl` with every sub-protocol call (domains,
LogUp-GKR, sumchecks, the commitment's calls) with its inputs, outputs and transcript position,
and which protocol step moved the transcript. That trace is what the step bodies will be ported
and checked against. The script has not been run yet: this environment has no GPU.

### Not ported yet

- **The per-layer steps of `llm.verify` (`Proto.run`):** the F_O model's domains of each unit,
  their claims and how they reach the commitment. Everything they are built from is ported: the
  sumcheck, LogUp-GKR, domains, the commitment and the outer layer. `read-proof-file` parses a
  real proof, but a real proof cannot be verified yet.
- **The GPU prover:** the kernels generated for NVRTC, CuPy and PyTorch. It is not a candidate for
  a direct port.

## Build

```bash
make            # ./sable, a standalone executable (runs the tests first; HEAP=8GB by default)
make core       # sable.core, for `sbcl --core sable.core' (starts in SABLE-USER)
make test       # compile and run the tests (asdf:test-system "sable")
make vectors    # regenerate tests/vectors.sexp from the Python code (numpy, blake3, g++)
make crosscheck # check Lisp proofs with the original verifiers: GKR with vcore.cpp (numpy, g++);
                #   domains, the commitment and the statement with zk/domain.py, zk/gpcs.py
                #   and zk/llm.py (also PyTorch: make crosscheck PYTHON=...)
```

On a machine with an NVIDIA GPU, from the repository root, prove a tiny model and trace the host
verifier on it (`--model tiny`, `qwen3` or `moe`; see the script's docstring for the files):

```bash
python lisp/tests/gpu-trace.py all OUT --model tiny
```

The `trace` half needs no GPU, so `OUT` can be traced again anywhere:

```bash
python lisp/tests/gpu-trace.py trace OUT
```

## Use

`sable` reads Lisp forms and evaluates them in `sable-user`, which uses every Sable package:

```bash
sable '(describe-proof (read-proof-file "/data/proof_1x4096.proof"))'
sable '(normalize (quote (* (+ c0 3) (- c1 c0))))'
sable '(demo :statement (quote (* t0 t1 t2)) :variables 14 :base t)'
sable '(let ((c (make-cfg :l 2 :d 256 :nh 4 :nkv 2 :dh 128 :dff 384 :v 1024)))
         (length (getf (commitment-layout c (make-shape c 2 256 7 7 16)) :groups)))'
sable '(run-tests)'
sable - < script.lisp        # forms from standard input
sable                        # a REPL, on a terminal
sable --help                 # (help): every exported function, from its lambda list and docstring
```

In the REPL, an error enters the debugger with the restarts on offer. For example, `(finv 0)`
offers `use-value`, `return-zero` and `abort`, which returns to the REPL.

| Exit status | Meaning |
|---|---|
| 0 | success |
| 1 | proof rejected (`malformed-proof`, `verification-failure`) |
| 2 | bad input: an unreadable form, an unknown function or variable, a missing file |
| 3 | internal error (`SABLE_DEBUG=1` enters the debugger instead, with the stack intact) |
| 130 | interrupted |
| 141 | standard output closed (e.g. piped into `head`) |

## Conditions

Errors use the condition system, following the
[Lisp Docs conditions tutorial](https://lisp-docs.github.io/docs/tutorial/conditions).
Every condition inherits from `sable-condition`:

```
sable-condition
  sable-error                     (also error)
    malformed-proof               reason: :magic :truncated :trailing-bytes :unknown-tag
                                          :value-range :rank :dimension, with the byte offset
    verification-failure          check: the keyword of the failed check
      message-shape-error         a message of the wrong kind or shape
      transcript-exhausted        the proof has too few messages
    invalid-parameter             a value outside a function's domain, or a malformed form
    field-division-by-zero        an inverse of zero (also cl:division-by-zero)
```

Restarts appear only where continuing is sound:

- **No restarts on rejection.** `malformed-proof` and `verification-failure` offer no restart that
  continues: a handler can only exit, so no handler can make the verifier accept a proof.
- **`finv` and `ext-inv` of zero** offer `use-value` and `return-zero`. `return-zero` gives the
  Python and C++ behaviour, where 0^(p-2) = 0.
- **`check-parameter`** works like `check-type` and signals `invalid-parameter` with `use-value`.
  The replacement value is checked again.
- **The tests** are the functions named `test-*` in `sable.tests`, and their checks are `assert`
  forms. `run-tests` wraps each in `skip-test` and `retry-test`. `(run-tests :debug t)` sends
  failures to the debugger with those restarts on offer.

The executable's exit status shows the difference between `handler-bind` and `handler-case`.
A `handler-bind` sees an internal error before the stack unwinds, so `SABLE_DEBUG=1` can open the
debugger on the failing frames. A `handler-case` then maps each condition class to an exit status.

## A difference from the Python code

`zk/proofio.py` computes a field array's size with `np.prod` over its declared shape, which wraps
at 2^64. A shape such as (2^40, 2^40) therefore passes its length check as size 0, and the proof is
rejected only when `reshape` fails afterwards. The Lisp parser multiplies the dimensions exactly
and rejects that input as `:truncated`.
