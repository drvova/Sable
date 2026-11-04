# Sable in Common Lisp (SBCL)

A port of Sable's host-side core from Python (`zk/`) to Common Lisp, built into an SBCL image.
It has no dependencies beyond SBCL and the ASDF that ships with it.

Everything here is checked bit for bit against the Python implementation. `tests/gen-vectors.py`
runs the Python code and writes `tests/vectors.sexp`, and the Lisp tests reproduce every value in it.
That covers field arithmetic, BLAKE3 in both modes, transcript challenges, proof bytes, Reed-Solomon
codewords and folds, and whole sumcheck proofs. For the sumcheck, the Lisp prover writes the same
proof bytes as the Python prover, and the Lisp verifier accepts the Python proofs.

## What is ported

| Lisp (`src/`) | Python | Contents |
|---|---|---|
| `util.lisp` | | octet vectors, little-endian integers, `define-constant` |
| `conditions.lisp` | (`ValueError`, `assert`) | the condition hierarchy, see below |
| `field.lisp` | `zk/field.py`, `zk/fe.py` | KoalaBear F_p, the F_p4 / F_p8 tower (multiplication and inversion written once, recursively, and expanded by a macro into straight-line code per degree), Lagrange interpolation, eq tables, MLE evaluation |
| `blake3.lisp` | `zk/cuda/b3s.h`, `blake3` | the sequential BLAKE3 mode of the transcript, and standard (tree-mode) BLAKE3 |
| `transcript.lisp` | `zk/transcript.py` | the Fiat-Shamir transcript: absorb, send / record, challenges, proof of work, label names |
| `proofio.lisp` | `zk/proofio.py` | the proof byte format (`SABLEPF1`), strict parser and writer |
| `poly.lisp` | `zk/poly.py` | polynomials over column symbols |
| `rs.lisp` | `zk/rs.py` | the rate-1/2 Reed-Solomon code: twiddles, encoding, folding |
| `sumcheck.lisp` | `zk/sumcheck.py` (`verify`, `eval_terms`), `zk/cuda/vcore.cpp` | the sumcheck verifier, and a reference CPU prover for products of multilinear tables |
| `cli.lisp` | | the command line of the executable (`define-command`, which also writes the help) |

Not ported yet:

- The rest of the verifier: LogUp-GKR (`gkr.verify_upper`), the BaseFold opening and Merkle
  query checks of the polynomial commitment (`gpcs.py`), and the transformer protocol (`llm.verify`).
  So `sable inspect` can parse and summarise a real proof, but cannot yet verify one.
- The GPU prover: the kernels generated for NVRTC, CuPy and PyTorch, and CUDA graphs. It is not a
  candidate for a direct port.

## Build

```bash
make            # ./sable, a standalone executable (runs the tests first; HEAP=8GB by default)
make core       # sable.core, the same system in a core for the REPL
make test       # compile and run the tests (asdf:test-system "sable")
make vectors    # regenerate tests/vectors.sexp from the Python code (numpy, blake3, g++)
```

The executable saves SBCL's runtime options. It parses no SBCL options of its own, so
`sable --help` is Sable's help. Set the heap size at build time: `make HEAP=16GB`. Real proofs are
100 to 500 MiB.

```bash
sbcl --core sable.core          # a REPL with the sable.* packages loaded
```

## Use

```
sable inspect PROOF...           parse proof files and summarise their messages
sable hash FILE...               BLAKE3 digests, in b3sum's format
sable sumcheck-demo [N] [D] [T]  prove and verify a random sumcheck on the CPU, then a tampered one
sable selftest                   the test suite, with its vectors built into the image
```

| Exit status | Meaning |
|---|---|
| 0 | success |
| 1 | proof rejected (`malformed-proof`, `verification-failure`) |
| 2 | usage or I/O error |
| 3 | internal error (`SABLE_DEBUG=1` enters the debugger instead, with the stack intact) |
| 141 | standard output closed (e.g. piped into `head`) |

From Lisp:

```lisp
(let* ((msgs (sable.proofio:read-proof-file "/data/proof_1x4096.proof"))
       (tr (sable.transcript:make-transcript "label" 8 :messages msgs)))
  (multiple-value-bind (rho reduced) (sable.sumcheck:verify tr n degree claim :eq-r eq-r)
    (sable.sumcheck:check-reduced reduced expected)))
```

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
    invalid-parameter             a value outside a function's domain
    field-division-by-zero        an inverse of zero (also cl:division-by-zero)
```

Restarts appear only where continuing is sound:

- **No restarts on rejection.** `malformed-proof` and `verification-failure` offer no restart that
  continues: a handler can only exit, so no handler can make the verifier accept a proof.
- **`finv` and `ext-inv` of zero** offer `use-value` and `return-zero`. `return-zero` gives the
  Python and C++ behaviour, where 0^(p-2) = 0:
  `(handler-bind ((field-division-by-zero #'return-zero)) (finv 0))` returns 0.
- **`invalid-parameter`** is signalled by `check-parameter`, which works like `check-type`.
  It offers `use-value`, and the replacement value is checked again. `root-of-unity`,
  extension-degree checks and the Reed-Solomon and MLE length checks use it.
- **The test runner** (`tests/framework.lisp`) wraps each test in `skip-test` and `retry-test`.
  `(sable.tests:run-tests :debug t)` sends failures to the debugger with those restarts on offer.

The CLI shows the difference between `handler-bind` and `handler-case`. A `handler-bind` sees an
internal error before the stack unwinds, so `SABLE_DEBUG=1` can open the debugger on the failing
frames. A `handler-case` then maps each condition class to an exit status.

## A difference from the Python code

`zk/proofio.py` computes a field array's size with `np.prod` over its declared shape, which wraps
at 2^64. A shape such as (2^40, 2^40) therefore passes its length check as size 0, and the proof is
rejected only when `reshape` fails afterwards. The Lisp parser multiplies the dimensions exactly
and rejects that input as `:truncated`. The test `proofio-rejects-malformed` covers this case.
