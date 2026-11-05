# Exacta

Exacta verifies proofs that a language model with committed weights produced a given reply to a
given prompt, bit for bit, and makes those proofs' parts on the CPU. It is written in Common Lisp
and built into an SBCL image, with no dependencies beyond SBCL and the ASDF and `sb-introspect`
that ship with it.

Exacta started as a port of [Sable](#origin) (Small-field Argument for Bit-exact LLM Execution),
whose paper is in [`paper/sable.pdf`](paper/sable.pdf). The proofs are Sable's: sumchecks and
LogUp-GKR lookups over the 31-bit KoalaBear field (p = 2^31 − 2^24 + 1), with a BaseFold
commitment over rate-1/2 Reed-Solomon codes and BLAKE3 Merkle trees. Sable's Python and CUDA
implementation has been removed from this repository. Comments that cite `zk/...` refer to it as of
commit 81807ef, the last one that has it.

The interface is Lisp code itself:

- **Polynomials and sumcheck statements are ordinary forms.** Constraints and statements are
  written as s-expressions, such as `(* (+ c0 3) (- c1 c0) c2)` or `(* t0 t1 t2)`. The code
  normalises, measures, renames and compiles them; there is no polynomial class.
- **The field tower is generated, not hand-written.** It is defined once, recursively, and turned
  into compiled code for any degree.
- **The executable evaluates forms.** There is no table of commands; the exported functions are
  the commands.

Everything was checked bit for bit against Sable's Python implementation, and the tests keep it
that way without Python:

- **Vectors.** `tests/vectors.sexp` was written by the Python code, and the tests reproduce every
  value in it: field arithmetic, BLAKE3 in both modes, transcript challenges, proof bytes,
  polynomials, Reed-Solomon codewords and folds, and whole sumcheck proofs. The Lisp sumcheck
  prover writes the same proof bytes as the Python prover, and the Lisp verifier accepts the Python
  proofs.
- **Cases the original verifiers accepted.** For the larger protocols it went the other way: the
  original verifiers checked the Lisp prover's proofs, and every forgery of them.
  `test-frozen-cases` pins those exact cases by their SHA-256, so any change to what Exacta
  proves or computes there fails the tests.

## Status

| Part | State | Checked against (Sable's code) |
|---|---|---|
| Field tower, BLAKE3, transcript, proof format, Reed-Solomon | done | the vectors |
| Sumcheck (verifier and prover) | done, optimised (steps 1–6 of the plan) | the vectors: Python's proofs, byte for byte |
| LogUp-GKR | done | the C++ verifier `vc_gkr_upper`: 8 shapes, 231 forgeries |
| Domains | done | `zk/domain.py`'s verifier: 16 domains |
| Polynomial commitment (BaseFold) | done | `zk/gpcs.py`'s verifier: 3 scenarios, 287 forgeries |
| Statement, policy, sampling seed, C_W, commitment layout | done | `zk/llm.py`: 18 layouts, 9 statements, 10 checks |
| Protocol context (claims, packing, domains and sumchecks in a step, LogUp sets) | done (stage 8a) | `zk/proto.py`'s `Ctx`: one scenario, 95 forgeries |
| The lookup table: its 2^26 rows, the model's value tables, multiplicities, the LogUp table side; public MLEs | done (stage 8b) | `zk/tables.py` and `zk/model.py`: every column and value table equal; `logup.table_side`: one scenario, 172 forgeries |
| Commitments and GKR at the table's size (2^26 rows) | not yet (stage 8b2) | |
| F_O: the model file, every gadget, the forward pass (dense layers, QK-norm) | done (stage 8c) | `zk/model.py`: all 54 intermediate values of two tiny models' passes equal |
| Per-layer steps of the model's proof | in progress (stage 8d): the MLP (output, slices, input) | `Proto.step_mo`, `step_mlp` and `step_mi` on the tiny model's layer 0: same claims, all 2749 forgeries rejected |
| MoE layers; quantizing a BF16 checkpoint into F_O | not yet | |
| Prover for whole models | not yet | |

So Exacta cannot yet verify a whole proof of a model. Every protocol that proof is made of is
done and checked, but not the per-layer steps that put them together. The plan behind the
optimisations, and the papers it draws on, is in
[`reports/Exacta optimization research.md`](reports/Exacta%20optimization%20research.md).

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

## The protocols

The second column names the Sable code that each file reimplements.

| `src/` | Origin (Sable) | Contents |
|---|---|---|
| `util.lisp` | | octet vectors, little-endian integers |
| `conditions.lisp` | (`ValueError`, `assert`) | the condition hierarchy, `check-parameter` |
| `field.lisp` | `zk/field.py`, `zk/fe.py` | KoalaBear F_p with branchless addition, the tower (`mul*` as a lazily reduced schoolbook product, `inv*`, `tower-code`), Lagrange interpolation, eq tables, MLE evaluation |
| `forms.lisp` | `zk/poly.py`, the expressions of `zk/sumcheck.py` | arithmetic forms: `expand`, `normalize`, `form-degree`, `arithmetic-form`, `in-place-code`, `compile-form`, `evaluate` |
| `blake3.lisp` | `zk/cuda/b3s.h`, `blake3` | the sequential BLAKE3 mode of the transcript, and standard BLAKE3 |
| `transcript.lisp` | `zk/transcript.py` | the Fiat-Shamir transcript |
| `proofio.lisp` | `zk/proofio.py` | the proof byte format: writes the magic `EXACTPF1`, and also reads Sable's `SABLEPF1` |
| `rs.lisp` | `zk/rs.py` | the rate-1/2 Reed-Solomon code |
| `sumcheck.lisp` | `zk/sumcheck.py`, `zk/cuda/vcore.cpp` | the sumcheck of any statement form: verifier, final check, and a reference CPU prover on flat tables, with one compiled round kernel per statement (`round-kernel-code`), split eq weights, and the first rounds of base tables answered from small-value accumulators (`small-value-code`) |
| `gkr.lisp` | `zk/gkr.py`, `vc_gkr_upper` in `zk/cuda/vcore.cpp` | batched LogUp-GKR fractional sums: `verify-upper`, a CPU reference prover `prove-upper`, the layer statement as a form |
| `domain.lisp` | `zk/domain.py` | a domain (columns, zerocheck constraints, lookups) as data; its leaf-layer and zerocheck statement as one form; `verify-domain` and a CPU reference prover `prove-domain` |
| `pcs.lisp` | `zk/gpcs.py` | the polynomial commitment: units and size classes, merged mixed-height Merkle trees with caps, the per-class point reductions, segments, the cross-unit sumcheck, BaseFold with oracles, grinding and the query checks; a CPU reference prover |
| `sha256.lisp` | `hashlib.sha256` | SHA-256 (FIPS 180-4), for the statement's digests |
| `proto.lisp` | `zk/proto.py` (`Ctx`, `Pack`), `logup.parse_assign` | the protocol context the steps share, in `:commit`, `:prove` and `:verify` mode: tensors and their packing, block, point and weighted claims, group openings, `run-domain` with the LogUp root sums and `check-lookup-sets`, `run-sumcheck` with zk/'s header-degree rules (`piece`, `sumcheck-degree`), virtual claims |
| `exact.lisp` | `round_exact` in `zk/model.py`, mpmath | correctly rounded values: fixed-point exp, ln, sin, cos and sqrt at 320 bits, and Sable's rule for when to use them |
| `tables.lisp` | `zk/tables.py`, `zk/logup.py`, the table functions of `zk/model.py` | the global lookup table: tags, layout and rows, the model's value tables (`exp-values`, `silu-values`, `samp-values`, `router-values`, `rope-tables`), `build-table`; chunks and `count-lookups`; `table-side`, the table's half of LogUp |
| `public.lisp` | `zk/mlepub.py` | closed-form MLEs of public vectors: `iota`, `le-low-high`, `lt-const`, `eq-const`, `vocab-indicator` |
| `model.lisp` | `zk/model.py` | F_O: the model file (`read-model`), the gadgets with their witness values (`requant8`, `normscale`, `shiftq8s`, `qkv-out`, `rope8`, `gshiftround`, `satadd`, `norm-den`, `mulround`, `norm-div`, …), `attention-rows`, `mlp-out`, `layer-forward`, `final-logits`, `forward` |
| `steps.lisp` | the steps of `zk/llm.py`'s `Proto`, `LayerWit` | the per-layer steps: the layer witness (`layer-witness`, `rq-pack`, `ns-pack`, `sr-pack`, `rm-layout`), the domain builder (`make-db`, `db-column`, `db-zero`, `db-lookup`, `db-run`) with its routes, the per-row gadgets (`rq-rows`, `ns-rows`, `sr-rows`, `cap-col`), `layer-head`, `step-mo`, `step-mlp`, `step-mi`, with `matmul`, `oh-check` and `gather-rowsum` |
| `llm.lisp` | the outer layer of `zk/llm.py`, `zk/proto.py` | the transformer protocol's public side: `make-cfg`, `make-shape`, `column-block-bits`, the commitment layout (`commitment-layout`: groups, packing, merge sets, segments, weight groups; `declare-units`), the policy and its encoding, `check-statement`, `statement`, `sampling-seed`, `weights-digest` (C_W) |
| `toplevel.lisp` | | the executable: evaluates forms in `exacta-user`; `(help)` |

Sable's `Poly` gives layer parameters negative column ids. With symbols as variables that
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

**Checked by Sable's C++ verifier.** The prover made proofs for 8 shapes: 1 to 6 blocks
(including non-powers of two), 1 to 3 lookups, and F_p^4 and F_p^8. Sable's host verifier,
`vc_gkr_upper`, checked each (the cases are pinned in `test-frozen-cases`):

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

**Checked by Sable's verifier.** Sable's `domain.verify` checked 16 domain proofs made by
`prove-domain`, in its host mode, on its own factored form of the leaf statement and its C++
verifier core (the cases are pinned in `test-frozen-cases`). The proofs cover merged and unmerged blocks, 1 to 5
blocks, every column kind, no lookups or no constraints, and F_p^4 and F_p^8.

- Sable's verifier accepts each proof, consumes every message, and computes the same claims and
  roots.
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

**Checked by Sable's verifier.** Sable's `zk/gpcs.py` verifier checked the prover's proofs on the
host in 3 scenarios (the cases are pinned in `test-frozen-cases`):

- a merge set and a segment, 5 size classes, columns split at `cmax`, a tensor below the smallest
  class, and claims sharing points
- `cmax` 10, with two oracle levels and depth-2 paths
- point reductions in batches

Sable's verifier accepts each proof, consumes every message, and rejects every changed value or
byte (287 forgeries). The only messages left unchanged are the grinding nonce, which a few test bits of
grinding may still accept, and the empty paths of trees whose cap is their leaves. The tests use
6 queries and 4 bits of grinding instead of 153 and 30: grinding 30 bits on the CPU takes hours.
Sable's verifier takes both as parameters too.

### The transformer protocol

`llm.lisp` has the parts of Sable's `llm.verify` that come before and around the per-layer
steps, with the same bytes:

- the statement (`P|` policy, C_W, h_circ from the configuration's `repr`, the shape and the
  constants digest, H(x), H(y), the nonce, T, the salt), the policy encoding, and the verifier's
  checks on the public statement (`check-statement`, which signals `verification-failure` where
  `Policy.check` raises)
- the sampling seed, and C_W from the weight groups' caps
- the commitment layout: every group of every layer with Sable's packing of small blocks into
  columns, the MoE expert groups, the merge sets, the segments and the weight groups;
  `declare-units` declares them on a `make-pcs` verifier in Sable's order

Sable's `zk/llm.py` computed the same values (pinned in `test-frozen-cases`): 18 layouts (the
tiny test models, and Qwen2.5-32B, Qwen3-32B and Qwen3-30B-A3B at the paper's shapes, up to 1029
groups and 16830 items), 9 statements, policies and sampling seeds, 10 statement checks and C_W.

### The protocol context

Every step of the model's proof works through one context (`proto.lisp`, Sable's `Ctx`). It
holds the declared tensors and the claims made on them until their group is opened, runs the
step's domains and sumchecks, and keeps the LogUp sums the verifier checks at the end:

- **Packing:** a tensor of many small blocks is committed as wider columns. A claim on it is
  rewritten to claims on those columns: one random combination per column whose blocks are all
  claimed, and per-block claims for the others.
- **Point claims:** a claim on a whole tensor sends each block's value at the in-block point, or
  becomes one weighted claim when the tensor is packed in a single row.
- **Lookups:** each table lookup's root fraction joins the sum of the chunk it was assigned to.
  The roots of dynamic lookups, and of lookups with a numerator, join their tag's set, which must
  sum to zero.
- **Sumcheck degrees:** the header states Sable's degree, not the true one. A `piece` (one of
  Sable's `Prog` polynomials, or a whole sum of terms) counts per round only the tables that vary
  in that round, maximised over the rounds; products of pieces add their degrees.

Domains may use the layer parameters (Kco, Kcd, Dyn, Dme), which `make-domain` substitutes.

**Checked by Sable's `Ctx`.** One scenario covers every operation: packed and unpacked tensors,
full and partial claims on packed columns, a sumcheck with ROW, COL and shared tables over three
weighted blocks, both degree rules, and a domain with a layer parameter, a table lookup and a
dynamic pair. Sable's `Ctx` replays Exacta's proof, consumes all 103 messages, computes the same
values and rejects all 95 forgeries; so does Exacta's verifier (`test-proto-roundtrip`).

### The lookup table

Every lookup into a fixed table goes to one global table of 2^26 rows (`tables.lisp`). Each row
is a tuple (tag, v0, …, v5), and each tag owns a segment of rows: ranges, shift regimes, powers of
two, reciprocals, square roots, and the model's value tables. The value tables hold exp for
attention, SiLU, the sampling temperature's exp, and the MoE router's exp. RoPE's cos and sin are
a second fixed unit.

The value tables are correctly rounded values of transcendental functions. Like Sable, Exacta
computes them in double floats and recomputes exactly any entry within 10⁻⁴ of a tie, using
fixed-point arithmetic at 320 bits (`exact.lisp`) where Sable used mpmath. The result is
independent of the platform. For a tiny model, every value table and all seven columns of the
2^26-row table are byte for byte Sable's (`test-table-values` pins the value tables). Building
the table takes about 4 s.

The table side of LogUp (`table-side`) proves that the lookups assigned to each chunk sum to that
chunk's multiplicities over the table. It uses one GKR over the table's rows, a leaf sumcheck,
claims on the table's columns and claims on the multiplicities. The scenario at 2^9 rows counts a
domain's lookups into two chunks; Sable's `logup.table_side` accepts the proof with the same
claims and sums and rejects all 172 forgeries (`test-table-side`, pinned in
`test-frozen-cases`).

### F_O

The proof is about F_O, an exact integer version of the model: every rounding, normalisation,
softmax and sampling decision is an integer operation, so each input has exactly one output.
`model.lisp` holds it gadget by gadget, each returning its output and the witness values the
prover commits, and signals `witness-error` when an input falls outside F_O's domain (an honest
prover then refuses to prove).

A model is a file of named integer tensors (`EXACTFO1`): the configuration, the model's exponents
and shifts, and each layer's int8 weights, per-column exponents, shifts and biases:

    magic "EXACTFO1" | u32 count | entry*
    entry = u16 name length | name | u8 width (1, 2, 4, 8) | u8 rank | u64 dimension * rank
            | signed little-endian integers of that width, row-major

`tests/models/` holds two of Sable's quantized tiny models (2 layers, d = 256, 4 heads of 128, a
vocabulary of 1024): a dense one and a Qwen3-shaped one with QK-norm. On the same tokens, every
intermediate value of both passes equals Sable's, 54 per model, from the row quantizations to the
logits (`test-model-forward`). Quantizing a BF16 checkpoint into F_O (Sable's calibration program)
is not ported yet, so models come from Sable's quantizer.

### Not done yet

- **The per-layer steps of the model's proof (Sable's `Proto.run`):** the F_O model's domains of
  each unit, their claims and how they reach the commitment. Everything they are built from is
  done: the sumcheck, LogUp-GKR, domains, the commitment, the outer layer and the protocol
  context, the lookup table and its side of LogUp. The remaining stages are 8b2 (commitments and
  GKR fast enough for the table's 2^26 rows: today a commitment costs about 6 µs and 2 KB of
  allocation per element, so the table alone would take most of an hour), 8d (each unit's steps,
  with the witness values only the proof needs: attention tiles, lookup histograms) and 8e (whole
  proofs, then MoE). `read-proof-file`
  parses a real Sable proof, but cannot verify it yet. The reference for this work is Sable's code
  at commit 81807ef. That commit also has `lisp/tests/gpu-trace.py`, which runs Sable's GPU
  prover on a tiny model and records every sub-protocol call of its host verifier, so each step
  can be checked message by message.
- **A prover for whole models:** Sable proves on an NVIDIA GPU (kernels generated for NVRTC,
  CuPy and PyTorch). Exacta has CPU reference provers for each protocol, but not yet for a model.

## Build

```bash
make            # ./exacta, a standalone executable (runs the tests first; HEAP=8GB by default)
make core       # exacta.core, for `sbcl --core exacta.core' (starts in EXACTA-USER)
make test       # compile and run the tests (asdf:test-system "exacta")
```

## Use

`exacta` reads Lisp forms and evaluates them in `exacta-user`, which uses every Exacta package:

```bash
exacta '(describe-proof (read-proof-file "/data/proof_1x4096.proof"))'
exacta '(normalize (quote (* (+ c0 3) (- c1 c0))))'
exacta '(demo :statement (quote (* t0 t1 t2)) :variables 14 :base t)'
exacta '(let ((c (make-cfg :l 2 :d 256 :nh 4 :nkv 2 :dh 128 :dff 384 :v 1024)))
          (length (getf (commitment-layout c (make-shape c 2 256 7 7 16)) :groups)))'
exacta '(run-tests)'
exacta - < script.lisp       # forms from standard input
exacta                       # a REPL, on a terminal
exacta --help                # (help): every exported function, from its lambda list and docstring
```

In the REPL, an error enters the debugger with the restarts on offer. For example, `(finv 0)`
offers `use-value`, `return-zero` and `abort`, which returns to the REPL.

| Exit status | Meaning |
|---|---|
| 0 | success |
| 1 | proof rejected (`malformed-proof`, `verification-failure`) |
| 2 | bad input: an unreadable form, an unknown function or variable, a missing file |
| 3 | internal error (`EXACTA_DEBUG=1` enters the debugger instead, with the stack intact) |
| 130 | interrupted |
| 141 | standard output closed (e.g. piped into `head`) |

## Conditions

Errors use the condition system, following the
[Lisp Docs conditions tutorial](https://lisp-docs.github.io/docs/tutorial/conditions).
Every condition inherits from `exacta-condition`:

```
exacta-condition
  exacta-error                    (also error)
    malformed-proof               reason: :magic :truncated :trailing-bytes :unknown-tag
                                          :value-range :rank :dimension, with the byte offset
    verification-failure          check: the keyword of the failed check
      message-shape-error         a message of the wrong kind or shape
      transcript-exhausted        the proof has too few messages
    invalid-parameter             a value outside a function's domain, or a malformed form
    witness-error                 an input outside F_O's domain: the honest prover refuses it
    field-division-by-zero        an inverse of zero (also cl:division-by-zero)
```

Restarts appear only where continuing is sound:

- **No restarts on rejection.** `malformed-proof` and `verification-failure` offer no restart that
  continues: a handler can only exit, so no handler can make the verifier accept a proof.
- **`finv` and `ext-inv` of zero** offer `use-value` and `return-zero`. `return-zero` gives the
  behaviour of Sable's Python and C++ code, where 0^(p-2) = 0.
- **`check-parameter`** works like `check-type` and signals `invalid-parameter` with `use-value`.
  The replacement value is checked again.
- **The tests** are the functions named `test-*` in `exacta.tests`, and their checks are `assert`
  forms. `run-tests` wraps each in `skip-test` and `retry-test`. `(run-tests :debug t)` sends
  failures to the debugger with those restarts on offer.

The executable's exit status shows the difference between `handler-bind` and `handler-case`.
A `handler-bind` sees an internal error before the stack unwinds, so `EXACTA_DEBUG=1` can open the
debugger on the failing frames. A `handler-case` then maps each condition class to an exit status.

## A difference from Sable's code

Sable's `zk/proofio.py` computes a field array's size with `np.prod` over its declared shape,
which wraps at 2^64. A shape such as (2^40, 2^40) therefore passes its length check as size 0, and the proof is
rejected only when `reshape` fails afterwards. Exacta's parser multiplies the dimensions exactly
and rejects that input as `:truncated`.

## Repository

| Path | Contents |
|---|---|
| `src/` | the Lisp sources (see The protocols) |
| `tests/` | the tests, `vectors.sexp` and the test framework |
| `exacta.asd`, `build.lisp`, `Makefile` | the ASDF systems and the image build |
| `paper/sable.pdf` | Sable's paper: the integer model F_O, the constraint system, the protocol and its soundness analysis, the GPU prover and its measurements |
| `docs/spec/corrected.tex` | an early design specification of Sable; its first 1,024 tokens are the F_O calibration text |
| `reports/`, `research_notes/` | the research behind the optimisation plan |

## Origin

Exacta is a Common Lisp reimplementation of Sable by Tomer Solberg (MoonMath.ai), whose Python and
CUDA code this repository held up to commit 81807ef:

    @misc{sable2026,
      author = {Tomer Solberg},
      title  = {Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models},
      year   = {2026},
      note   = {\url{https://github.com/moonmath-ai/Sable}}
    }

## License

MIT (see `LICENSE`), copyright 2026 MoonMath.ai.
