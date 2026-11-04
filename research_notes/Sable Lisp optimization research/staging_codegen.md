# Staging, Partial Evaluation and Program Generation for Finite-Field / Cryptographic Arithmetic, and how they map onto Common Lisp (SBCL) macros + run-time COMPILE

Scope note: Verification status is marked on every entry.
- **[V]**: I opened the primary page or PDF in this session (arXiv abstract, the paper PDF, or the venue page).
- **[S]**: confirmed only through search-result snippets or a venue/catalog listing.
- **[PK]**: bibliographic data comes from prior knowledge and was not re-fetched this session. The URL given is the canonical one, but the report writer should treat these details as unconfirmed.

Lisp "sketch" bullets are my own inferences. None of them is a reported result.

Codebase context assumed: polynomials and sumcheck statements are s-expressions that are normalized and then turned into lambda forms with `COMPILE`. The F_p4/F_p8 tower over p = 2^31 - 2^24 + 1 (KoalaBear) is defined recursively on symbolic coordinates and expanded into straight-line code.

---

## Q1. Fiat-Cryptography and follow-ups (CryptOpt etc.): what they generate, how, and speedups

### Takeaway
Fiat-Cryptography shows that the architecture this Lisp codebase already uses is enough to get the fastest-known C, with proofs: generic high-level arithmetic, partial evaluation per parameter set, a rewriting/simplification pass, interval bounds analysis, then straight-line code. CryptOpt adds a randomized search over instruction schedules and instruction selection, benchmarked on the real CPU and checked by a verified equivalence checker. It reports a mean 1.74x speedup over GCC and 1.40x over Clang on Fiat's output. Two techniques transfer directly to Lisp: (a) interval bounds analysis to drop or delay modular reductions, and (b) search with benchmarking over equivalent emitted forms.

### Cited Findings
- **Fiat-Cryptography.** Erbsen, Philipoom, Gross, Sloan, Chlipala, "Simple High-Level Code For Cryptographic Arithmetic – With Proofs, Without Compromises", IEEE S&P 2019. Not on arXiv. PDF: http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf. Canonical IEEE DOI is 10.1109/SP.2019.00005 **[PK, DOI unconfirmed]**. Paper text **[V]**.
  - Abstract: "simple partial evaluation is sufficient to transform such initial code into the fastest-known C code". The library "achieves competitive performance for 80 prime fields and multiple CPU architectures", including the first verified high-performance P-256, and was deployed in BoringSSL for Chrome, Android and CloudFlare — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Mechanism, step 1: field arithmetic is written over lists of limbs, then partially evaluated per prime and limb configuration. To make Coq's reduction emit straight-line let-chains, operations are first CPS-converted (`add0 ... k`), and "Reduction turns this function into straightline code" — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Mechanism, step 2: the output is reified into a first-order AST (PHOAS). A bounds-inference pass then "works by standard abstract interpretation with intervals" to pick word sizes and prove that no overflow occurs, combined with algebraic simplification — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Templates: unsaturated Solinas / pseudo-Mersenne representations and word-by-word Montgomery. For each prime, a "champion" is chosen from two unsaturated candidates and one Montgomery candidate — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Result: on 1000 sequential 256-bit Montgomery-ladder computations over primes scraped from curves@moderncrypto.org, the generated code beats GMP (both `mpn_sec` and the variable-time `mpn`), with "Speedups range between 1.25X and 10X". Tested on Haswell (64-bit) and ARMv7 Krait (32-bit) — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Related work: the authors say their strategy "is most similar to the one from LMS" (Lightweight Modular Staging), and they also cite FFTW and Spiral — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
  - Press coverage (secondary): generated code matched the best handwritten code — [MIT Tech Review](https://www.technologyreview.com/2019/08/21/539/automatic-cryptocoder/)
- **CryptOpt.** Kuepper, Erbsen, Gross, Conoly, Sun, Tian, Wu, Chlipala, Chuengsatiansup, Genkin, Wagner, Yarom, "CryptOpt: Verified Compilation with Randomized Program Search for Cryptographic Primitives", PLDI 2023, arXiv **2211.10665** — https://arxiv.org/abs/2211.10665 **[V]**
  - Input is Fiat IR. A randomized search over x86-64 assembly benchmarks each candidate on the target CPU. A new equivalence checker, proven in Coq, verifies the result, so the search itself does not need to be trusted — [arXiv 2211.10665](https://arxiv.org/abs/2211.10665)
  - Search: Random Local Search plus "Bet-and-Run" (several independent runs, then the best one continues). The search state is a random topological order of the dataflow DAG (scheduling), per-op instruction templates (selection), and register/memory operand choices. Mutations revisit any of these decisions — [CryptOpt PDF](https://arxiv.org/pdf/2211.10665v3)
  - Results: across 9 prime moduli (mul and square) and 10 x86-64 machines (4 AMD, 6 Intel), "a mean speedup of 1.74 compared to GCC 12.1.0 (speedup 1.40 against Clang 15.0.6)". As a stand-alone optimizer it averages a 1.04x speedup over libsecp256k1's hand-optimized assembly, though that case is not verified because no Fiat IR exists for it — [CryptOpt PDF](https://arxiv.org/pdf/2211.10665v3)
  - Baseline: STOKE mostly timed out or emitted wrong code on these kernels. Where it worked (secp256k1 square), it was about 6% faster than GCC but still 73% slower than CryptOpt — [CryptOpt PDF](https://arxiv.org/pdf/2211.10665v3)
  - Secondary sources: a later thesis/journal version reports up to 2.96x over off-the-shelf compilers — [gp-bibliography Kuepper24](https://gpbib.cs.ucl.ac.uk/gp-html/Kuepper24.html) **[S]**. Third-party notes give 1.2–2.5x over gcc/clang — [dustri notes](https://dustri.org/b/paper-notes-cryptopt.html) **[S]**
- **Fiat follow-up (rewriting engine).** Gross, Erbsen, Philipoom, Agarwal, Chlipala, "Accelerating Verified-Compiler Development with a Verified Rewriting Engine", ITP 2022, arXiv 2205.00862 — https://arxiv.org/abs/2205.00862 **[PK, unconfirmed this session]**. It replaced Coq-tactic partial evaluation with a reflective, proven rewriter, which removed the Coq compile-time blowups. The S&P paper itself notes that "compilation of saturated arithmetic in Coq times out for a handful of larger primes" — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)
- **GPU follow-on in the same spirit.** Naifeng Zhang, Franz Franchetti, "Code Generation for Cryptographic Kernels using Multi-word Modular Arithmetic on GPU", CGO 2025, arXiv **2501.07535** — https://arxiv.org/abs/2501.07535 **[S]**
  - A rewrite system applies "MoMA" (multi-word modular arithmetic) recursively to data types, producing single-word operations for any code generator (built on SPIRAL)
  - Reported: BLAS kernels "orders of magnitude" faster than multi-precision libraries. A 256-bit NTT on an RTX 4090 beats a state-of-the-art library running on an H100 by 14x on average — [CGO'25 page](https://2025.cgo.org/details/cgo-2025-papers/35/Code-Generation-for-Cryptographic-Kernels-Using-Multi-word-Modular-Arithmetic-on-GPU), [arXiv 2501.07535](https://arxiv.org/abs/2501.07535)

### Inferences
- **Lisp sketch (Fiat bounds pass).** Add an interval abstract interpretation over the normalized s-expression DAG (each node carries `[lo, hi]` as integers) before emitting the lambda.
  - Use it to (i) choose `(unsigned-byte 32)` or `(unsigned-byte 64)` declarations, and (ii) delay modular reduction.
  - For KoalaBear, (p-1)^2 = (2^31 - 2^24)^2 = 2^62·(1-2^-7)^2. So 4·(p-1)^2 < 2^64, and a dot product of up to 4 unreduced products fits in a `(unsigned-byte 64)` accumulator. That covers exactly one F_p4 coordinate of a schoolbook product, plus or minus the non-residue multiply, which must be counted in the bound.
  - The pass would emit `(mod (+ a b c d) p)` once instead of 4 reductions. In SBCL this stays unboxed when the bounds are declared with `(the (unsigned-byte 64) ...)` or `(ldb (byte 64 0) ...)`.
  - The bounds-check arithmetic above is my own derivation, not a published result.
- **Lisp sketch (Fiat "template plus champion").** Keep several symbolic templates per field op: Montgomery vs Barrett vs plain `mod`, and schoolbook vs Karatsuba for F_p4. Partially evaluate each one, `COMPILE` each, time them on the target, and cache the winner per (op, degree, host). This mirrors Fiat's per-prime champion choice.
- **Lisp sketch (CryptOpt-lite).** CryptOpt's search space is exactly what a Lisp emitter controls: topological order of `let*` bindings, the choice of formula per node, and where reductions go.
  - A random local search with bet-and-run over those choices is cheap to build in Lisp: mutate the order, re-emit, `COMPILE`, benchmark.
  - Equivalence checking is easier here than for CryptOpt. All generated code is a polynomial map over F_p, so evaluating candidate and reference at random points is a probabilistic equivalence check by Schwartz–Zippel (error ≤ deg/p per trial).
  - Alternatively, normalize both symbolic expansions to a canonical polynomial, which the codebase can already do, for an exact check.
  - SBCL's native backend (register allocation and scheduling are its own) limits the gains from reordering more than assembly-level search does. Expect smaller wins than 1.4–1.7x. That expectation is unmeasured.

### Gaps
- I could not fetch the IEEE Xplore page to confirm the S&P'19 DOI.
- I did not verify per-curve numbers for CryptOpt's journal or thesis follow-up (2.96x).
- I found no work applying Fiat- or CryptOpt-style pipelines to 31-bit "small" STARK fields (BabyBear/KoalaBear/Mersenne31). All published targets are ≥ 255-bit ECC primes (plus P-448/P-521/SIKE).

---

## Q2. Equality saturation / e-graphs for arithmetic expression optimization, CSE, polynomial evaluation; any application to ZK circuits or finite-field code

### Takeaway
E-graphs are mature: egg (2021), egglog (2023), and rule inference with Ruler. They have strong results on DSP and vector kernels (Diospyros, 3.1x over vendor libraries), datapath and RTL arithmetic, linear algebra, and FHE arithmetic circuits (Lobster, CHEHAB). These are the closest neighbors to ZK and finite-field code. I found **no published work applying equality saturation to ZK constraint systems or to sumcheck/finite-field prover kernels**. There is a cautionary result: on large Boolean FHE circuits, plain equality saturation scaled poorly against greedy methods.

### Cited Findings
- **egg.** Willsey, Nandi, Wang, Flatt, Tatlock, Panchekha, "egg: Fast and Extensible Equality Saturation", POPL 2021, arXiv **2004.03082** — https://arxiv.org/abs/2004.03082 **[ID supplied by task; PK for content]**. It introduced "rebuilding" (deferred congruence closure) and e-class analyses. Constant folding modulo p is a natural e-class analysis.
- **egglog.** Zhang, Wang, Flatt, Cao, Zucker, Rosenthal, Tatlock, Willsey, "Better Together: Unifying Datalog and Equality Saturation", PLDI 2023, arXiv **2304.04332**, DOI 10.1145/3591239 — [arXiv](https://arxiv.org/abs/2304.04332) **[S]**. Datalog plus EqSat with composable analyses and incremental e-matching. It reimplements a floating-point rewriter (Herbie's) more simply and faster, and runs a points-to analysis 4.67x faster than Soufflé.
- **Diospyros.** VanHattum, Nigam, Lee, Bornholt, Sampson, "Vectorization for Digital Signal Processors via Equality Saturation", ASPLOS 2021, DOI 10.1145/3445814.3446707. Not on arXiv as far as I found — [NSF PAR](https://par.nsf.gov/biblio/10232895) **[S]**
  - Method: symbolically evaluate a small kernel into a straight-line scalar DAG, saturate with vector rewrite rules, extract by cost, and emit C++ intrinsics
  - Result: kernels 3.1x faster on average than vendor DSP libraries
  - This is the closest precedent to "symbolically expand the tower, then e-graph, then emit"
- **Isaria.** Thomas & Bornholt, "Automatic Generation of Vectorizing Compilers for Customizable DSPs", ASPLOS 2024. It synthesizes the Diospyros rewrite rules from an ISA specification — [Isaria paper PDF](https://cs.utexas.edu/~sgt/papers/compgen.pdf) **[S]**
- **Datapath and RTL arithmetic.**
  - Coward, Constantinides, Drane, "Automatic Datapath Optimization using E-Graphs", arXiv **2204.11478** (ARITH 2022) — https://arxiv.org/abs/2204.11478 **[S]**
  - "Automating Constraint-Aware Datapath Optimization using E-Graphs", arXiv **2303.01839** (DAC 2023) — https://arxiv.org/abs/2303.01839 **[S]**. Uses interval-style constraints inside the e-graph.
  - ROVER, "RTL Optimization via Verified E-Graph Rewriting", arXiv **2406.12421** — https://arxiv.org/abs/2406.12421 **[S]**
  - Relevance: integer bit-width and range reasoning combined with e-graphs is what a lazy-reduction pass needs
- **FHE arithmetic circuits (finite-ring arithmetic; the nearest analogue to ZK).**
  - Lobster (TOPLAS 2023): learns rewrite rules from small circuits by synthesis, applies them with EqSat, and extracts the circuit with the lowest multiplicative depth — [Lobster PDF](https://prl.korea.ac.kr/papers/toplas23-lobster.pdf) **[S]**
  - CHEHAB (CC 2026): an EqSat-based term rewriting system that vectorizes code and reduces instruction count and depth for FHE. It claims better scaling and faster code than Coyote — [CC 2026 page](https://conf.researchr.org/details/CC-2026/calls/3/CHEHAB-Automatic-Compiler-Code-Optimization-for-Fully-Homomorphic-Encryption) **[S]**
  - Oraqle (ePrint 2024/1409): depth-aware arithmetization that emits multiple circuits trading multiplications against depth — [ePrint 2024/1409](https://eprint.iacr.org/2024/1409) **[S]**
  - Porcupine: Cowan et al., PLDI 2021, arXiv 2101.07841 **[PK, unconfirmed this session]**. Synthesizes vectorized HE kernels.
- **Negative or scaling result.** "Cut Tracing with E-Graphs for Boolean FHE Circuit Synthesis", arXiv **2506.12883**. It reports that equality saturation "struggles to scale" on FHE circuit optimization, "often producing considerably worse results than state-of-the-art greedy methods". Its fix is to keep e-graphs small (cut-based) so ILP extraction stays tractable — [arXiv 2506.12883](https://arxiv.org/abs/2506.12883) **[S]**
- **Extraction and scaling tools.**
  - e-boost (extraction heuristics plus exact solving), arXiv **2508.13020** — https://arxiv.org/abs/2508.13020 **[S]**
  - E-morphic (scalable EqSat for logic synthesis), arXiv **2504.11574** — https://arxiv.org/abs/2504.11574 **[S]**
  - Quantum-circuit EqSat (Quasar), PLDI 2026 — [PLDI'26 page](https://pldi26.sigplan.org/details/pldi-2026-papers/11/Equality-Saturation-for-Quantum-Circuit-Optimization) **[S]**
- **Older e-graph arithmetic precedents [PK].**
  - Herbie: Panchekha et al., "Automatically Improving Accuracy for Floating Point Expressions", PLDI 2015, DOI 10.1145/2737924.2737959
  - SPORES: Wang et al., VLDB 2020, arXiv 2002.07951 (linear-algebra rewriting)
  - Ruler: Nandi et al., OOPSLA 2021, arXiv 2108.10436. Infers rewrite rules by enumeration plus EqSat. It would be useful to auto-derive F_p-specific identities.
- **Non-e-graph polynomial CSE.** Hosangadi, Fallah, Kastner, "Optimizing polynomial expressions by algebraic factorization and common subexpression elimination", IEEE TCAD 2006 **[PK, pre-2015, unconfirmed]**. Kernel/co-kernel extraction for multivariate polynomials. It is a cheaper deterministic alternative to EqSat for the sumcheck statement DAG.

### Inferences
- **Lisp sketch (tower e-graph).** Before emitting the F_p4/F_p8 lambda, insert the fully expanded symbolic coordinates into an e-graph. Options are a small hash-consed e-graph in Lisp, or serializing the s-expressions to egglog, which reads s-expressions natively.
  - Rules: commutativity and associativity of `+` and `*`; distributivity in both directions (factoring is what recovers Karatsuba from schoolbook); the Karatsuba identity `a0*b1 + a1*b0 = (a0+a1)*(b0+b1) - a0*b0 - a1*b1`; multiplication by the binomial non-residue W as shift/add when W is small; and constant folding mod p as an e-class analysis.
  - Extraction cost: base mult about 3–5, add/sub 1, modular reduction counted separately.
  - Expected effect: recover or beat Karatsuba operation counts automatically for each tower level and composite op, such as `a*b + c*d` in F_p4 sharing products.
  - To bound blowup, apply the cut-tracing lesson: saturate per tower level (F_p2 inside F_p4) rather than on the fully flattened F_p8 expression, and use rule schedules and node limits.
- **Lisp sketch (sumcheck statement CSE).** A sumcheck round computes g(t) = f(P_1(t),...,P_k(t)) for t = 0..d. If all d+1 evaluations are placed in one e-graph (or one hash-consed DAG), subterms that are linear in t can be shared. Extraction picks the cheapest combined program, for example a product evaluated through incremental differences.
- **Lisp sketch (cheaper fallback).** Much of the gain may already come from hash-consing at construction time ("smart constructors" in LMS terms; see Q4) plus Hosangadi-style co-kernel factoring. Full EqSat should be tried only if those plateau.
- **Open niche.** Applying EqSat to ZK statement and sumcheck-composition optimization appears unexplored in the literature I could find. It is an open niche, not a solved problem.

### Gaps
- No paper found on e-graphs for R1CS/Plonkish/AIR constraint reduction or sumcheck composition polynomials. Two searches ("e-graph equality saturation zero-knowledge circuit", "e-graphs finite field ... R1CS") returned none.
- No quantitative data found on EqSat recovering Karatsuba/Toom formulas automatically.
- I did not re-verify egg's own reported numbers.

---

## Q3. Superoptimization and search-based optimization for arithmetic kernels

### Takeaway
Classic stochastic superoptimization (STOKE) did not cope with multi-limb crypto kernels. Search over a structured space, as in CryptOpt (schedule, selection, registers), did, with verified results. Synthesizing superoptimizers (Souper, Minotaur) find peephole rules that are then folded into compilers, verified with SMT/Alive2. For a Lisp generator, the transferable pattern is benchmark-driven search over formula and schedule choices, plus verification by polynomial identity.

### Cited Findings
- **CryptOpt** (see Q1): structured random local search plus bet-and-run, 1.74x over GCC and 1.40x over Clang — [arXiv 2211.10665](https://arxiv.org/pdf/2211.10665v3) **[V]**
- **STOKE on these kernels.** Most attempts timed out or produced incorrect or unassemblable code. On secp256k1 square it was about 6% faster than GCC but 73% slower than CryptOpt — [CryptOpt PDF](https://arxiv.org/pdf/2211.10665v3) **[V]**. Original STOKE paper: Schkufza, Sharma, Aiken, "Stochastic Superoptimization", ASPLOS 2013, DOI 10.1145/2451116.2451150 **[PK, pre-2015]**
- **Souper.** Sasnauskas, Chen, Collingbourne, Ketema, Lup, Taneja, Regehr, "Souper: A Synthesizing Superoptimizer", arXiv **1711.04422**. As an automated pass it shrinks a Clang binary by about 3 MB (4.4%) — [arXiv 1711.04422](https://arxiv.org/abs/1711.04422) **[S]**
- **Minotaur.** Liu, Mada, Regehr, "Minotaur: A SIMD-Oriented Synthesizing Superoptimizer", OOPSLA 2024 (Distinguished Paper), arXiv **2306.00229**. Synthesis-based and verified with Alive2. Several discovered optimizations were upstreamed into LLVM — [arXiv 2306.00229](https://arxiv.org/abs/2306.00229) **[S]**
- **Diospyros/Isaria** (Q2) are search plus EqSat compilers for small numeric kernels, with a 3.1x average gain over vendor libraries — [NSF PAR](https://par.nsf.gov/biblio/10232895) **[S]**
- **Spiral/FFTW (Q4)** are the canonical "search over algebraic breakdown rules, with benchmarking" systems for transforms.

### Inferences
- **Lisp sketch.** Run a two-level search per kernel, such as an F_p4 mul, an F_p8 mul, or a sumcheck round evaluator for a given statement.
  - Outer level: discrete formula choices. These include Karatsuba depth, Toom-k vs schoolbook, reduction strategy, lazy-reduction batching width, and whether to unroll over evaluation points.
  - Inner level: random topological orders of the emitted `let*`.
  - Each candidate is built with `COMPILE` and timed. Correctness is checked by random-point evaluation against the reference symbolic polynomial.
  - Because SBCL's `COMPILE` takes milliseconds to tens of milliseconds for straight-line forms of a few hundred nodes, a few hundred candidates per kernel is feasible at build time (an estimate, not measured).
  - Persist winners in a cache keyed by (statement hash, CPU) so the search is not repeated.
- **Lisp sketch (Souper-style).** Mine peephole identities on emitted code, for example multiplication by small constants of the non-residue as shift-add. Feed them back as normalizer rules or SBCL `deftransform`s.

### Gaps
- No published superoptimization results specific to 31-bit STARK fields or to Lisp/SBCL targets.
- STOKE's original numbers were not re-verified.

---

## Q4. Multi-stage programming / generative programming for numerical code, and Lisp-specific precedents

### Takeaway
LMS, Spiral/FFTW, Terra, AnyDSL and MetaOCaml all express the same pattern as the Lisp codebase: a high-level generic program is specialized by staging into a straight-line or loop-nest program. Fiat-Crypto names LMS as its closest relative. Lisp's quasiquote plus `COMPILE` is staging without types. Its key gaps compared with LMS-style systems are guaranteed hash-consing/CSE at construction time, hygiene, and binding-time separation. All of these can be added with smart constructors and an explicit "staged value" struct.

### Cited Findings
- **Fiat-Crypto on its lineage.** Its strategy is "most similar to the one from LMS". It also cites FFTW and Spiral. It notes that verified compilers (CompCert, CakeML) cannot perform "abstraction-collapsing synthesis as done in this work or LMS" — [Fiat S&P'19 PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf) **[V]**
- **LMS.** Rompf & Odersky, "Lightweight Modular Staging: A Pragmatic Approach to Runtime Code Generation and Compiled DSLs", GPCE 2010, DOI 10.1145/1868294.1868314 **[PK, pre-2015]**. Its core ideas are a type-directed `Rep[T]` staging annotation and a graph IR built by smart constructors that perform CSE and constant folding on the fly ("hash-consing"), followed by code motion and scheduling — https://doi.org/10.1145/1868294.1868314
- **Spiral.** Püschel et al., "SPIRAL: Code Generation for DSP Transforms", Proc. IEEE 93(2), 2005 **[PK, pre-2015]**. Rule-based breakdown of transforms (SPL/Σ-SPL) plus search with runtime feedback. Spiral is still the backend of 2025 crypto-GPU codegen (MoMA, Q1) — [arXiv 2501.07535](https://arxiv.org/abs/2501.07535)
- **FFTW's genfft.** Frigo, "A Fast Fourier Transform Compiler", PLDI 1999 **[PK, pre-2015]**. An OCaml generator that builds a DAG of the straight-line transform, then runs an algebraic simplifier: constant folding, canonicalizing negated constants, and a "network transposition" trick. It schedules the result for register pressure. This is the most direct historical precedent for "expand symbolically, simplify, emit straight-line code".
- **Terra.** DeVito, Hegarty, Aiken, Hanrahan, Vitek, "Terra: A Multi-Stage Language for High-Performance Computing", PLDI 2013 **[PK]**. Lua is the meta-language for a low-level staged language, the same split as Lisp-as-meta with emitted typed straight-line Lisp.
- **AnyDSL / Thorin.** Leißa et al., OOPSLA 2018 **[PK]**. Partial evaluation driven by annotations in a single language.
- **BER MetaOCaml.** Kiselyov, FLOPS 2014 **[PK]**. Typed brackets and escapes. Kiselyov's later "generating optimal code with confidence" line of work covers staging numeric code such as Gaussian elimination and FIR filters **[PK, details unconfirmed]**.
- **Lisp-family staging.** Amin & Rompf, "Collapsing Towers of Interpreters", POPL 2018, DOI 10.1145/3158140 **[PK]**. It uses Pink, a small Lisp-like language, with a multi-level lambda calculus and stage polymorphism. It shows how to derive compilers from interpreters by staging, which is exactly the "normalize the s-expression, then emit a lambda" move.
- **Common Lisp facilities [PK].**
  - `define-compiler-macro` (CLHS) gives source-level partial evaluation at call sites with constant arguments
  - SBCL's `sb-c:deftransform` / `defknown` give type-directed transforms inside the compiler
  - `(declare (optimize (speed 3) (safety 0)))` with `(unsigned-byte 64)` modular arithmetic keeps code unboxed
  - CLHS: http://www.lispworks.com/documentation/HyperSpec/Body/m_define.htm

### Inferences
- **Lisp sketch (LMS smart constructors).** Replace raw `(list '* a b)` construction with `(staged-mul a b)`. This would:
  - hash-cons (look up an `equal` hash table of already-built nodes), which gives global CSE for free
  - canonicalize commutative operands by node id
  - constant-fold mod p
  - drop multiplication by 1 and addition of 0
  - The expansion of the F_p8 tower would then never materialize duplicate subtrees. This probably accounts for a large part of the "smaller expression DAGs" goal before any e-graph is needed. This is an inference, but it is LMS's core mechanism.
- **Lisp sketch (genfft-style scheduling).** After building the DAG, emit `let*` in an order that minimizes live values, using a Sethi–Ullman-like or "schedule for register pressure" heuristic as genfft does. SBCL's register allocator then sees fewer simultaneously live unboxed temporaries.
- **Lisp sketch (binding-time separation).** Represent statement-dependent quantities (the composition polynomial, degree, constant coefficients) as "static" values and table entries as "dynamic" values. Make the generator partially evaluate everything static. That gives the first Futamura projection applied to the generic sumcheck interpreter, which is exactly the Collapsing-Towers idea.
- **Lisp sketch (compiler macros as a cheap tier).** For the common ops `fp4-mul`, `fp4-add` and `fp8-mul`, compiler macros can inline the pre-generated straight-line body when the call site is in already-compiled code. Run-time `COMPILE` is then needed only for per-statement kernels.

### Gaps
- No recent (2015+) paper specifically on staging finite-field arithmetic in Common Lisp or Racket was found. Searches were not dedicated to this. The Racket/Scheme staging literature (e.g. Racket's `syntax-parse`-based DSLs) is not covered here.
- LMS, Spiral, FFTW, Terra, AnyDSL and MetaOCaml bibliographic data are [PK] and were not re-fetched.

---

## Q5. Code generation for ZK provers: specializing sumcheck/GKR provers to a statement; GPU/CPU kernel generation from expression trees

### Takeaway
Production ZK stacks compile front-end circuits into a prover-friendly IR: Expander's ECC produces layered GKR circuits, Circom produces R1CS, and Binius exposes a `CompositionPoly` `expression()`. The algorithmic literature (Gruen; Dao–Thaler; Bagad–Dao–Domb–Thaler) supplies the specializations a generator should bake in: small-value rounds, delayed reduction, eq-factor tricks, and fewer multiplications for products. Hardware and GPU work (zkPHIRE, zkDuel, zkSpeed, BatchZK) decompose sumcheck into "updates, extensions, products" and show that per-constraint kernels written in Triton can come within 8% of CUDA. Fully unrolled large constraints hit compiler-scalability limits, which matters for `COMPILE` too. I found **no paper describing an automatic statement-to-specialized-sumcheck code generator as its main contribution**.

### Cited Findings
- **zkDuel.** Narang, Daftardar, Reagen, "zkDuel: Characterizing SumCheck Prover Kernels across GPU Programming Models", arXiv **2610.37710** (Oct 2026; HASP'26 at MICRO 2026) — [arXiv 2610.37710](https://arxiv.org/abs/2610.37710) **[V]**
  - Setup: 25 zkPHIRE-derived constraints across 32–256-bit prime fields, written in both CUDA and Triton, checked against a CPU reference
  - Gap: CUDA's median advantage falls from 3.5x at 32 bits to 1.4x at 256 bits on H100, and from 4.4x to 1.6x on A100
  - The small-field gap comes from host launch overhead and redundant modular additions in the block reduction. With `tl.reduce` plus specialized point interpolation, Triton is within 8% of CUDA. Specialized interpolation gives 1.55–1.75x on 256-bit evaluation kernels
  - "the largest fully unrolled constraints still hit compiler-scalability limits"
  - The kernels are hand-matched, not auto-generated
- **zkPHIRE.** Daftardar, Mo et al., "zkPHIRE: A Programmable Accelerator for ZKPs over HIgh-degRee, Expressive Gates", arXiv **2508.16738** (HPCA 2026) — [arXiv 2508.16738](https://arxiv.org/abs/2508.16738) **[S]**
  - A programmable SumCheck unit for arbitrary high-degree multilinear compositions, reducing SumCheck to "updates, extensions, and products"
  - Results: 1486x geomean over CPU, 11.87x over the prior state of the art (zkSpeed) at iso-area, and the SumCheck unit alone >1000x over CPU
- **zkSpeed.** "Need for zkSpeed: Accelerating HyperPlonk for Zero-Knowledge Proofs", ePrint **2025/620** — [ePrint 2025/620](https://eprint.iacr.org/2025/620) **[S]**. Fixed-function, for vanilla gates.
- **BatchZK.** A fully pipelined GPU system with a pipelined sum-check module, ePrint **2024/1862** — [ePrint 2024/1862](https://eprint.iacr.org/2024/1862) **[S]**
- **NTT vs SumCheck on hardware.** "When Proofs Meet Hardware: Comparing NTT and SumCheck in Zero-Knowledge Systems", arXiv **2606.16146** — [arXiv PDF](https://arxiv.org/pdf/2606.16146) **[S, title only; content not read]**
- **Sumcheck algorithmic specializations (what a generator should emit).**
  - Bagad, Dao, Domb, Thaler, "Speeding Up Sum-Check Proving", ePrint **2025/1117**. Targets small-valued evaluations (u32 or base-subfield values inside an extension field). Reports 2–3x for the first Spartan sum-check in Jolt, and 20x or more when baselines near memory limits — [ePrint 2025/1117](https://eprint.iacr.org/2025/1117) **[S]**
  - Extended version: Dao, DeStefano, Bagad, Domb, Thaler, ePrint **2026/587**. Adds a technique for products of many multilinear polynomials that cuts field multiplications, plus a "small-value sum-check prover" — [ePrint 2026/587](https://eprint.iacr.org/2026/587) **[S]**
  - Related: Bagad, Domb, Thaler, "The Sum-Check Protocol over Fields of Small Characteristic", ePrint **2024/1046** — [ePrint](https://eprint.iacr.org/2024/1046); Dao & Thaler, "More Optimizations to Sum-Check Proving", ePrint **2024/1210** — [ePrint](https://eprint.iacr.org/2024/1210) **[S]**
  - Gruen, "Some Improvements for the PIOP for ZeroCheck", ePrint 2024/108 **[PK, unconfirmed]**. Factors out the eq polynomial so the round polynomial has one less degree to evaluate.
  - Tutorials: [zkSecurity "Faster Sumchecks"](https://blog.zksecurity.xyz/posts/faster-sumchecks.md), [LambdaClass "Optimizing sumcheck"](https://blog.lambdaclass.com/optimizing-sumcheck/), [Jolt book sumcheck appendix](https://jolt.a16zcrypto.com/how/appendix/sumcheck.html) **[S]**
- **Expander Compiler Collection (Polyhedra).** Compiles gnark-compatible Go, Rust, or experimental Circom front-ends into layered circuits for the Expander GKR prover. The prover uses a Libra-derived SIMD variant that reorders sumcheck variables for SIMD, plus Virgo++-style cross-layer relays — [ECC announcement](https://blog.polyhedra.network/announcing-the-expander-compiler-collection-ecc/), [Expander GKR docs](https://docs.polyhedra.network/expander/prover_internals/gkr) **[S]**. Libra: Xie et al., ePrint 2019/317 **[PK]**
- **Binius.** The `CompositionPolyOS` trait exposes `expression()` (an `ArithExpr`), degree, and a batch evaluator, so composition polynomials are data that the prover can inspect and evaluate in SIMD fashion — [binius_math docs](https://binius-ulvetannaoss-16d81a0876eb016acc581aa502e37b77aaa54acc5fb.gitlab.io/binius_math/trait.CompositionPolyOS.html) **[S]**. Binius paper: Diamond & Posen, "Succinct Arguments over Towers of Binary Fields", ePrint 2023/1784 **[PK]**. Binius is also the main tower-field precedent: binary towers are defined recursively, as the Lisp F_p4/F_p8 tower is.
- **Other systems named in the task [PK, not verified this session].**
  - Jolt: Arun, Setty, Thaler, ePrint 2023/1217
  - Ceno: ePrint 2024/387
  - Circom: Bellés-Muñoz et al., IEEE TDSC (circuit DSL compiled to R1CS plus C++/WASM witness generators)
  - zkLLVM: LLVM-based circuit compiler, no peer-reviewed paper found
  - None of these, as far as I know, generates a per-statement sumcheck kernel by symbolic specialization in the way the Python original's per-expression CUDA kernel generator does

### Inferences
- **Lisp sketch (per-statement round kernel).** For a statement g = f(P_1..P_k) of degree d, generate one lambda per statement that loops over pairs (lo_i, hi_i) and computes the d+1 (or d with Gruen) round evaluations. Its features:
  - incremental extension: P_i(t+1) = P_i(t) + (hi_i - lo_i), costing additions only, which is zkPHIRE's "extension" primitive
  - the composition f fully inlined as straight-line code produced by the hash-consed or e-graph DAG (Q2/Q4)
  - accumulators in unreduced 64-bit form with periodic reduction, sized by the Q1 bounds pass
  - optionally, rounds where all P_i are base-field values (before the first challenge), so the kernel uses F_p × F_p mults and lifts to F_p4 only at accumulation. This is the Bagad–Dao–Domb–Thaler small-value idea expressed as a binding-time choice: the generator emits a different kernel for round 1 than for later rounds.
- **Lisp sketch (fold kernel).** Generate a separate specialized "bind challenge r" kernel: P'(x) = lo + r·(hi - lo), with r in F_p4 and lo/hi possibly in F_p. Mixed-field multiplication (F_p × F_p4 = 4 base mults) should be emitted, not promoted to full F_p4 × F_p4 (9–16 base mults).
- **Lisp sketch (avoid unroll blowup).** zkDuel's finding that "fully unrolled constraints hit compiler-scalability limits" applies to SBCL `COMPILE` too, which is superlinear on very large functions in my experience (unmeasured). Cap straight-line size per lambda. Above the cap, split the composition into sub-lambdas, or keep a loop over terms with only the inner products unrolled.
- **GPU path.** The Python original's per-expression CUDA generation matches the zkDuel/zkPHIRE decomposition. A Lisp port could emit CUDA C or Triton source strings from the same DAG, using the same smart-constructor IR with a different printer.

### Gaps
- I found no paper whose primary contribution is "compile a sumcheck statement to a specialized prover kernel", for CPU or GPU. The closest are zkPHIRE (programmable hardware), zkDuel (hand-matched kernels) and Binius/Expander (IR plus generic evaluators).
- Jolt, Ceno, Circom, zkLLVM and Gruen details were not re-verified this session.
- No numbers found comparing a generated per-statement sumcheck kernel against an interpreted generic one on CPU.

---

## Q6. Optimal or near-optimal evaluation schemes a generator could choose (Horner, Estrin, Karatsuba/Toom thresholds) for extension-field and product arithmetic

### Takeaway
For small binomial extensions the choice is between schoolbook (n² mults), recursive Karatsuba (3 per halving: 9 for degree 4, 27 for degree 8), and Toom/interpolation-style or Montgomery's Karatsuba-like formulas, which need fewer mults but more additions and constant multiplications. With a 31-bit p and lazy 64-bit accumulation, additions and reductions dominate. The right choice is empirical per CPU, which argues for a generator with a search or benchmark step (Q1/Q3) instead of a fixed formula.

### Cited Findings
- **Montgomery's Karatsuba-like formulae.** P. L. Montgomery, "Five, Six, and Seven-Term Karatsuba-Like Formulae", IEEE Trans. Computers 54(3):362–369, 2005, DOI 10.1109/TC.2005.49 **[S, pre-2015]** — [dblp](https://dblp.dagstuhl.de/rec/bibtex/journals/tc/Montgomery05)
  - Division-free formulas multiply 5-term polynomials with 13 scalar mults, 6-term with 17, and 7-term with 22, in any characteristic
  - They can be mixed with 2- and 3-term Karatsuba recursively
  - A follow-up by Fan & Hasan (IEEE TC 2007) uses CRT to improve some sizes over GF(2)
- **Extension-field multiplication and squaring formulas** (Karatsuba, Toom-Cook, Chung–Hasan squaring for F_p2/F_p3/F_p6/F_p12 towers): Devegili, Ó hÉigeartaigh, Scott, Dahab, "Multiplication and Squaring on Pairing-Friendly Fields", ePrint 2006/471 **[PK, pre-2015, unconfirmed this session]** — https://eprint.iacr.org/2006/471
- **Product of many multilinear polynomials in sumcheck.** ePrint 2026/587 adds a technique to "cut field multiplications" for products of many multilinear polynomials — [ePrint 2026/587](https://eprint.iacr.org/2026/587) **[S]**. Specialized point interpolation gives 1.55–1.75x on GPU evaluation kernels — [arXiv 2610.37710](https://arxiv.org/abs/2610.37710) **[V]**
- **Horner/Estrin.** No 2015+ primary source found in this session. The standard references are Knuth TAOCP Vol. 2 §4.6.4 and Estrin (1960) **[PK]**.

### Inferences
- **Lisp sketch (formula menu for the tower).** For F_p4 = F_p[x]/(x^4 - W) and F_p8 as a quadratic or binomial extension over F_p4, the generator should offer at least the following, and pick per CPU by benchmark (Q3):
  - (a) schoolbook: 16 base mults for F_p4, with lazy accumulation of 4 products per coordinate in u64 (Q1 bound)
  - (b) 2-level Karatsuba: 9 mults, but more adds and intermediate values above p
  - (c) Toom/interpolation-style (7 mults for 4-term), which requires multiplication by inverse small constants mod p. That costs field multiplications too, so it is likely a loss at 31-bit p.
  - (d) for F_p8 over F_p4: Karatsuba at the top level (3 F_p4 mults plus the W-multiply)
  - Hypothesis (unmeasured): with 31-bit limbs and u64 accumulators, (a) or a hybrid (Karatsuba at the top level only) wins, because one reduction per output coordinate is cheaper than Karatsuba's extra subtractions, which need either signed handling or added p-multiples.
- **Lisp sketch (sumcheck products).** For the product of k multilinears evaluated at t = 0..d, choose between (i) direct per-point evaluation of the product (k-1 mults per point), and (ii) Toom-like divide-and-conquer: evaluate pair products as degree-2 polynomials at fewer points, then extend. This is a generator-level choice keyed on k and on the field of the inputs. The extended Bagad et al. paper is the reference for the multiplication-count optimum.
- **Lisp sketch (Horner vs Estrin).** These matter only if the codebase evaluates univariate round polynomials at challenges, which is cheap. Estrin trades multiplications for instruction-level parallelism and is usually preferred for degree ≥ 4 on out-of-order CPUs. A generator can emit either one from the same coefficient list.

### Gaps
- No recent, peer-reviewed measured comparison of Karatsuba vs schoolbook for degree-4/8 binomial extensions of 31-bit primes (BabyBear/KoalaBear/Mersenne31) was found. Plonky3 and similar code bases make this choice in code, but I did not find a paper.
- Devegili et al. and the Horner/Estrin references were not re-verified this session.
