# Sable in Common Lisp (SBCL)

A port of Sable's host-side core from Python (`zk/`) to Common Lisp, built into an SBCL image.
It has no dependencies beyond SBCL and the ASDF and `sb-introspect` that ship with it.

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
Python proofs.

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

The field tower works the same way. `mul*` and `inv*` define multiplication and inversion once,
recursively over the levels (Karatsuba, and the norm one level down), on symbolic coordinates.
`(tower-code 'mul* 8)` returns the resulting straight-line `lambda` form. `ext*` and `ext-inv`
compile that code on first use for whatever degree they are given. On the sumcheck demo with 14
variables, proving takes 0.19 s, against 0.26 s with the earlier hand-written routines.

## What is ported

| Lisp (`src/`) | Python | Contents |
|---|---|---|
| `util.lisp` | | octet vectors, little-endian integers |
| `conditions.lisp` | (`ValueError`, `assert`) | the condition hierarchy, `check-parameter` |
| `field.lisp` | `zk/field.py`, `zk/fe.py` | KoalaBear F_p, the tower (`mul*`, `inv*`, `tower-code`), Lagrange interpolation, eq tables, MLE evaluation |
| `forms.lisp` | `zk/poly.py`, the expressions of `zk/sumcheck.py` | arithmetic forms: `expand`, `normalize`, `form-degree`, `arithmetic-form`, `compile-form`, `evaluate` |
| `blake3.lisp` | `zk/cuda/b3s.h`, `blake3` | the sequential BLAKE3 mode of the transcript, and standard BLAKE3 |
| `transcript.lisp` | `zk/transcript.py` | the Fiat-Shamir transcript |
| `proofio.lisp` | `zk/proofio.py` | the proof byte format (`SABLEPF1`) |
| `rs.lisp` | `zk/rs.py` | the rate-1/2 Reed-Solomon code |
| `sumcheck.lisp` | `zk/sumcheck.py`, `zk/cuda/vcore.cpp` | the sumcheck of any statement form: verifier, final check, reference CPU prover |
| `toplevel.lisp` | | the executable: evaluates forms in `sable-user`; `(help)` |

Python's `Poly` gives layer parameters negative column ids. With symbols as variables that
workaround is not needed: a layer parameter is just another variable.

Not ported yet:

- The rest of the verifier: LogUp-GKR (`gkr.verify_upper`), the BaseFold opening and Merkle
  query checks of the polynomial commitment (`gpcs.py`), and the transformer protocol (`llm.verify`).
  `read-proof-file` parses a real proof, but a real proof cannot be verified yet.
- The GPU prover: the kernels generated for NVRTC, CuPy and PyTorch. It is not a candidate for a
  direct port.

## Build

```bash
make            # ./sable, a standalone executable (runs the tests first; HEAP=8GB by default)
make core       # sable.core, for `sbcl --core sable.core' (starts in SABLE-USER)
make test       # compile and run the tests (asdf:test-system "sable")
make vectors    # regenerate tests/vectors.sexp from the Python code (numpy, blake3, g++)
```

## Use

`sable` reads Lisp forms and evaluates them in `sable-user`, which uses every Sable package:

```bash
sable '(describe-proof (read-proof-file "/data/proof_1x4096.proof"))'
sable '(normalize (quote (* (+ c0 3) (- c1 c0))))'
sable '(demo :statement (quote (* t0 t1 t2)) :variables 14)'
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
