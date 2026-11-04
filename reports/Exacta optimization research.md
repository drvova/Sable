# Lazy arithmetic and staging accelerate Sable's Lisp port

> Written when Exacta was the Lisp port of Sable. Paths under `zk/` refer to Sable's Python
> implementation, which was removed from this repository after commit 81807ef. The plan's steps 1–7
> have since been carried out (see `README.md`).

The largest and cheapest speedups for the SBCL port do not come from a new paper. They come from three changes to `field.lisp`: making modular add and subtract branchless, rewriting the tower multiply as a lazy-reduction schoolbook over X^8 − 3 (the same field as the tower, with the basis bit-reversed), and writing results into a caller-supplied buffer. Measured locally on SBCL 2.2.9, these take one F_p8 multiply from **about 500 ns to about 55 ns, with zero allocation**. Once the field is fast, the sumcheck literature adds smaller but real gains, almost all of it from the Thaler line on IACR ePrint rather than arXiv. The Dao–Thaler split-eq tables (ePrint 2024/1210) save about 10% on top of Gruen's trick, which Sable already uses. Keeping tables in the base field until the first fold, and then the Bagad–Dao–Domb–Thaler small-value prover (ePrint 2025/1117), are predicted to give **1.31× for F_p4 and 1.85× for F_p8** by its predecessor's cost model. For the unported verifier, Papini–Haböck LogUp-GKR (ePrint 2023/1284) and BaseFold analysed in Haböck's Johnson-bound regime (ePrint 2024/1571) fix what to port. WHIR (ePrint 2024/1586) is the one protocol upgrade worth coordinating with the Python prover. Three 2025–26 papers have disproved the "up-to-capacity" conjectures, so no port should adopt query counts that rely on them. arXiv contributes the compiler side (CryptOpt, egg/egglog, zkDuel), and together with Fiat-Cryptography (IEEE S&P 2019, not on arXiv) it points to "partial evaluation plus interval bounds analysis", which is close to what `field.lisp` and `forms.lisp` already do. On the SBCL side, the FFI and sb-simd pay off only for batched kernels that SBCL cannot vectorize itself, chiefly BLAKE3 Merkle hashing and AVX2 Montgomery folds. Everything else should be fixed by type declarations, `dynamic-extent` and destination-passing.

Throughout, "local measurement" means a microbenchmark run in this session with SBCL 2.2.9 on a noisy 4-vCPU Xeon VM at 2.1 GHz (AVX2 and AVX-512). Treat those numbers as ±20%, and up to 1.6× run-to-run on in-cache kernels. They are not published results.

## One F_p8 multiply drops from 500 ns to 55 ns without new math

The generated tower multiply wastes most of its time on branches and allocation, not on multiplications. Counting the code that `mul*` emits gives **27 general base multiplies**, **13 multiplies by the non-residue 3** (each emitted as a full `(fmul +nonres+ x)` with a `mod`), and **about 95 modular adds and subtracts**. Each of those adds is the branchy `fadd`/`fsub` from `field.lisp`, and on random field data the branches mispredict about half the time. The disassembly shows two more costs. The result vector comes from a full call through `#<FDEFN MAKE-EXT>`, and the eight stores go through `HAIRY-DATA-VECTOR-SET/CHECK-BOUNDS`. A misprediction estimate of roughly 95 × 0.5 × 15–20 cycles predicts 350–450 ns, which matches the measured cost.

Fixing these one at a time, in local measurements on F_p8 with chained multiplies:

| Variant of the F_p8 multiply | ns per multiply | Allocation |
|---|---|---|
| Current `TOWER-FUNCTION 'mul* 8` (Karatsuba, fresh vector) | 502–568 | 48 B/op |
| Same, with `make-ext` inlined as a constant-size `make-array` | 447–473 | 48 B/op |
| Same Karatsuba, branchless add/sub `s = a+b−p; s + (p & (s>>63))` | 138–162 | 48 B/op |
| Schoolbook X^8 − 3, lazy: two 4-product u64 sums per coefficient, each reduced once | 129–193 | 48 B/op |
| Same schoolbook, writing into a caller-supplied destination | **53–63** | **0 B/op** |

The combination of branchless add/sub, lazy schoolbook and a destination buffer was not measured as one variant. The best measured schoolbook still used the branchy `fadd` to join its two half-sums, so the combined version is probably faster still.

### The tower is X^8 − 3 in disguise

`field.lisp` builds F_p2 = F_p[x]/(x² − 3), F_p4 = F_p2[y]/(y² − x) and F_p8 = F_p4[z]/(z² − y), so z^8 = 3. The tower coordinate with bits (x, y, z) is the power-basis coefficient of z^(4i+2j+k), which is a 3-bit bit-reversal of the index (verified locally). This matters because Plonky3, the reference implementation for KoalaBear, writes every binomial extension multiply as one dot product per output coefficient. KoalaBear uses W = 3 for degrees 4 and 8, and implements `mul_w(a) = a.double() + a`, so multiplying by W costs two adds and no multiply ([Plonky3 koala_bear.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/koala-bear/src/koala_bear.rs); [binomial_extension.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)).

The arithmetic fits a 64-bit word. Each base product of canonical elements is below 2^62, so **four products fit in an unsigned 64-bit word**. An F_p4 coefficient has exactly four products and needs **one reduction**, so an F_p4 multiply needs 4 reductions against 9 + 4 now. An F_p8 coefficient has eight products and needs two, for 16 per multiply. Plonky3's own comments describe this headroom: sums of three or four products go to a single `large_monty_reduce`, and 5–8 terms are split into two 4-sums ([monty_31.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs)). Lazy reduction across whole towers is an old idea in pairing arithmetic. Aranha, Karabina, Longa, Gebotys and López generalized it from quadratic extensions to full towers and reported **27–33% pairing speedups** on 64-bit platforms ("Faster Explicit Formulas for Computing Pairings over Ordinary Curves", Eurocrypt 2011, ePrint 2010/526 — [ePrint](https://eprint.iacr.org/2010/526)).

### Keep canonical form and SBCL's Barrett for scalar code

SBCL already compiles a well-typed `(mod (* a b) p)` into a Granlund–Montgomery multiply-high sequence: `IMUL`, `MUL` by 2323999252290013197, `SAR 28`, `IMUL` by p, `SUB`. That is three multiplies and no `DIV` ([SBCL srctran.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/srctran.lisp)). Locally it ran at **5.2 ns of dependent latency, against 6.8–7.0 ns for a hand-written Montgomery reduction**.

Plonky3 stores KoalaBear in Montgomery form because AVX2 and AVX-512 have only 32×32→64 lane multiplies (`vpmuludq`) and no multiply-high. Its packed AVX2 multiply reaches "throughput: 4 cyc/vec (2 els/cyc)" ([avx2/packing.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)). Montgomery is therefore worth adopting only together with SIMD (see the last section).

The special form of KoalaBear does not help. Since 2^31 ≡ 2^24 − 1, one fold gains only 7 bits. Only Mersenne-31 gets a one- or two-fold reduction, and its local scalar multiply was about twice as fast (2.7–2.8 ns). The cost is circle-FFT machinery and a CM31/QM31 extension ([Circle STARKs, Haböck–Levit–Papini, ePrint 2024/278](https://eprint.iacr.org/2024/278)), so M31 is not a drop-in replacement.

A July 2026 survey of field choice explicitly names the lack of a controlled cross-field comparison as the field's empirical gap (Skatharoudis, ePrint 2026/1371 — [ePrint](https://eprint.iacr.org/2026/1371)).

### The type declarations decide whether any of this happens

A tight range type is the prerequisite for all of the above. In a local fold, declaring `h` and `l` only as `(unsigned-byte 32)` let SBCL derive `h − l + p < 2^33`. That made `(* r d)` possibly wider than 64 bits, and the compiled code fell back to signed `IMUL` with an overflow check, bignum allocation via `ALLOC-TRAMP`, and a full call to `TRUNCATE`: **15.6 ns per element**. Declaring the field range `(integer 0 2130706432)`, with `sb-ext:truly-the` on the intermediates, gave **2.2–4.6 ns per element, equal to gcc -O3**.

Only one product fits in a fixnum (below 2^62), so every lazy sum must be declared `(unsigned-byte 64)`.

### Inversion and the remaining arithmetic

`finv` is `fpow a (p−2)`, a generic square-and-multiply loop with an untyped exponent: about 60 dependent multiplies, roughly 300 ns. Plonky3 instead uses Pornin's optimized binary-GCD inversion (ePrint 2020/972, cited from Plonky3's code; the paper was not fetched in this research), which returns 2^60·a^−1 and corrects the power of two with one Montgomery step ([monty_31.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs)). A simpler alternative is a fixed addition chain for p − 2 = 0x7EFFFFFF with about 48 squarings and 6 multiplies, an unpublished derivation from the research notes. Extension inverses already use norms down the tower, as Plonky3's `octic_inv` and `quartic_inv` do, so each costs one base inversion. The bigger lever for bulk work is Montgomery batch inversion: about three multiplies per element plus a single inversion. Plonky3 implements it in chunked four-lane form ([batch_inverse.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/batch_inverse.rs)).

`lagrange-weight` is already memoized per degree, so the remaining inverse in the sumcheck verifier hot path is the `ext-inv eq-rj` in `next-claim`, once per round. That is negligible next to Merkle hashing.

## Thaler-line sumcheck tricks add 1.3–1.85× once data stays in the base field

Every primary small-field sumcheck paper is on IACR ePrint. The research found no arXiv versions of any of them.

| Ref | Paper | Authors, year | Source (venue) | Core idea for Sable |
|---|---|---|---|---|
| P1 | Speeding Up Sum-Check Proving | Bagad, Dao, Domb, Thaler, 2025 | [ePrint 2025/1117](https://eprint.iacr.org/2025/1117) ([full-text mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)) | Small-value Toom-Cook accumulators for the first ℓ0 rounds; Algorithm 6 combines this with eq splitting |
| P2 | Speeding Up Sum-Check Proving (Extended) | Dao, DeStefano, Bagad, Domb, Thaler, 2026 | [ePrint 2026/587](https://eprint.iacr.org/2026/587) | New algorithm for products of many multilinears (abstract only) |
| P3 | The Sum-Check Protocol over Fields of Small Characteristic | Bagad, Domb, Thaler, 2024 | [ePrint 2024/1046](https://eprint.iacr.org/2024/1046) ([author copy](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)) | Stay in the base field for the first rounds; cost model for degree-k extensions |
| P4 | More Optimizations to Sum-Check Proving | Dao, Thaler, 2024 | [ePrint 2024/1210](https://eprint.iacr.org/2024/1210) | Split eq into two √N tables |
| P5 | Some Improvements for the PIOP for ZeroCheck | Gruen, 2024 | [ePrint 2024/108](https://eprint.iacr.org/2024/108) | Factor out the round variable's eq; univariate skip |
| P6/P7 | Blendy; Time-Space Trade-Offs for Sumcheck | Chiesa, Fedele, Fenzi, Zitek-Estrada, 2024; Baweja et al., 2025 (TCC) | [ePrint 2024/524](https://eprint.iacr.org/2024/524); [ePrint 2025/1473](https://eprint.iacr.org/2025/1473) | Streaming sumcheck with O(N^(1/k)) space |
| P8 | The Sum-Check Protocol over the Monomial Basis, and Other Optimizations | Dao, Biswas, Eagen, Milson, Papini, Thaler, 2026 | [ePrint 2026/762](https://eprint.iacr.org/2026/762) | Sum over {0,∞}^n so that binding needs no subtraction |

### Sable already has Gruen; the eq table is the next target

The Lisp `prove` follows Sable's protocol. The round message g_j excludes the eq factor of the round variable, is sent at X = 0, 2, …, d, and g_j(1) is implied by the claim. That is Gruen's "send t_i" variant. P1 recaps it as s_i(X) = l_i(X)·t_i(X), with t_i(1) recovered from the running claim ([P1 §5.1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)). The degree reduction and the skipped X = 1 evaluation are already realized.

What the Lisp code still does is rebuild a full eq table over the remaining variables every round (`(weights eq (1+ j) n d)`). Each entry is a separate EXT vector produced by `ext*`, and `eq-table` builds each level by `concatenate`.

Dao and Thaler's split keeps only two √N-sized tables, eq_L and eq_R. The prover accumulates Σ_{xL} eq_L·∏p_k in an inner loop and multiplies by eq_R once per outer index. Their abstract reports that this eliminates about 2^(n−1) multiplications on top of Gruen, "about a 25% (respectively 10%) end-to-end prover speedup" over a prover without and with Gruen ([ePrint 2024/1210](https://eprint.iacr.org/2024/1210)). The inner loop should run over the low-order contiguous bits for locality ([P1 §5.2](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)).

This change needs no protocol change. In SBCL it also removes the largest source of short-lived EXT allocation in the prover.

### The first fold is where small fields pay

P1's cost model sorts multiplications into ss (base × base), sl (base × extension) and ll (extension × extension). In a tower basis, multiplying a subfield element by an extension element costs only k subfield multiplies ([P1 §2.1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)). The current Lisp code misses this twice: `fold-table` computes `(ext* r (ext- hi lo))`, a full ll multiply, even when `hi` and `lo` are base values, and `line-value` calls `ext-scale` for every table, pair and evaluation point.

The fix comes in three graded steps. The lowest-risk step keeps base tables as `(unsigned-byte 32)` arrays and emits an sl fold for round 1: 8 base products and 8 reductions for F_p8, instead of a 27-multiply Karatsuba. The second step delays the first fold, computing round 2 directly from base tables with sl multiplies and folding two variables at once. P3 notes that this "from scratch" algorithm "is actually cheaper" than the linear-time prover when base and mixed multiplies are much cheaper than extension multiplies ([P3](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)).

The third step is P1's full Algorithm 6. For the first ℓ0 rounds, one pass over the base tables fills accumulators A_i(v,u) on a (d+1)-ary Lagrange grid, and each round message becomes a short inner product with tensored Lagrange weights of the earlier challenges. P1 gives the cost as about (((d+1)/2)^ℓ0 + d)·N sl plus d²/2^(ℓ0+1)·N ll multiplies. Because eq lives in the extension, the relevant ratio is ll/sl rather than ll/ss, which makes the speedup "a square-root factor worse" than in the eq-free case ([P1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)). P3's table for d = 2 under Karatsuba costs predicts **1.31× at k = 4 (switch over at round 3) and 1.85× at k = 8 (round 4)**. It notes that "most the savings come from the first few rounds" ([P3 Table 1](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)).

The published measurements are on other fields: about 3× over original Jolt and up to 20× when baselines become memory-bound on BN254, and 20–40% for Algorithm 6 over Algorithm 5 on binary towers. No paper benchmarks 31-bit prime fields with degree-4 or degree-8 extensions. LambdaClass merged Algorithm 6 into whir-p3, which supports KoalaBear, with ℓ0 = 3, but published no timings ([LambdaClass](https://blog.lambdaclass.com/speeding-up-sumcheck-an-in-depth-walkthrough-of-our-implementation/)). The 2026 extended version (P2) claims 1.7–2.2× on a high-degree Jolt sumcheck through a new product algorithm, but only its abstract was available ([ePrint 2026/587](https://eprint.iacr.org/2026/587)).

### Smaller kernel tricks and the protocol-changing options

Inside each round, every table restricted to the round variable is linear. The kernel should therefore compute slope = hi − lo once and reach X = 2…d by repeated addition, which zkPHIRE calls the "extension" primitive. The current code instead recomputes `(ext-scale (ext- hi lo) x)` at every point. The kernel should also keep sl accumulations in unreduced u64 lanes.

The evidence on delayed reduction inside sumcheck is mixed. Plonky3 uses delayed-reduction kernels in its WHIR sumcheck ([Plonky3 #1890](https://github.com/Plonky3/Plonky3/issues/1890)). However, a mirrored PR fusing length-5 to length-8 dot products gained 8–27% in microbenchmarks while "the sumcheck benchmark does not move measurably" ([mirror PR #545](https://github.com/Tkelley325/Plonky3/pull/545)).

Three changes would alter the protocol, so they should wait until the Python prover and CUDA kernels can change in lockstep. Evaluating at ∞ means sending the leading coefficient instead of g(d), which is a plain product of slopes ([P1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)). P8's projective monomial basis gave about 10% end-to-end on BN254, but Plonky3 landed only the prover kernel, and its benefit on 31-bit fields remained "an open question" ([Plonky3 #1890](https://github.com/Plonky3/Plonky3/issues/1890)). Univariate skip cut prover time by 11.8% in a KoalaBear implementation in leanVM, and the PR was still closed unmerged over complexity ([leanVM PR #251](https://github.com/leanEthereum/leanVM/pull/251)). Streaming provers (P7: up to 120× less memory at under 2× time — [ePrint 2025/1473](https://eprint.iacr.org/2025/1473)) matter only if tables exceed RAM. SVO already reduces peak extension-table memory by 2^ℓ0.

## Port the verifier to proven bounds; WHIR is the upgrade

The LogUp-GKR and BaseFold verifiers that still need porting are defined by papers from 2023–24. None of the follow-ups makes the GKR verifier structurally cheaper, so the port order is clear. The research found no arXiv versions of the protocol papers. Some positive proximity-gap results appear on arXiv, but those were seen only as search snippets.

| Paper | Authors, year | Source | Role for the Lisp port |
|---|---|---|---|
| Improving logarithmic derivative lookups using GKR | Papini, Haböck, 2023 | [ePrint 2023/1284](https://eprint.iacr.org/2023/1284) | Defines the LogUp-GKR verifier to port |
| Multivariate lookups based on logarithmic derivatives | Haböck, 2022 | [ePrint 2022/1530](https://eprint.iacr.org/2022/1530) | Original LogUp baseline |
| Bypassing the characteristic bound in logUp | Eagen, Haböck, 2024 | [ePrint 2024/2067](https://eprint.iacr.org/2024/2067) | Lifts the requirement that multiplicities stay below p |
| Logup* | Soukhanov, 2025 | [ePrint 2025/946](https://eprint.iacr.org/2025/946.pdf) | Cheaper indexed lookups on the same fractional-GKR verifier |
| BaseFold | Zeilberger, Chen, Fisch, 2023 (CRYPTO 2024) | [ePrint 2023/1705](https://eprint.iacr.org/2023/1705) | The PCS to port |
| Basefold in the List Decoding Regime | Haböck, 2024 | [ePrint 2024/1571](https://eprint.iacr.org/2024/1571) | Proven soundness up to the Johnson bound, covering WHIR too |
| WHIR | Arnon, Chiesa, Fenzi, Yogev, 2024 (EUROCRYPT 2025) | [ePrint 2024/1586](https://eprint.iacr.org/2024/1586) | Successor PCS with a few-hundred-µs verifier |
| STIR | Arnon, Chiesa, Fenzi, Yogev, 2024 (CRYPTO 2024) | [ePrint 2024/390](https://eprint.iacr.org/2024/390) | Univariate rate-shifting; arguments 1.25–2.46× smaller than FRI |
| On Proximity Gaps for Reed–Solomon Codes | Ben-Sasson, Carmon, Haböck, Kopparty, Saraf, 2025 | [ePrint 2025/2055](https://eprint.iacr.org/2025/2055) | Tighter proven error terms at the unique-decoding and Johnson radii |
| On RS Proximity Gaps Conjectures; distances of random words; failure near capacity | Crites–Stewart 2025; Diamond–Gruen 2025; Krachun–Kazanin–Haböck 2026 | [2025/2046](https://eprint.iacr.org/2025/2046); [2025/2010](https://eprint.iacr.org/2025/2010); [2026/782](https://eprint.iacr.org/2026/782) | Disprove the up-to-capacity conjectures |

### LogUp-GKR is one more statement form

The Papini–Haböck verifier is a layered GKR over (numerator, denominator) pairs, with one sumcheck per layer, a random combination of the two child claims, and one final PCS claim ([ePrint 2023/1284](https://eprint.iacr.org/2023/1284)). In this codebase each layer's identity, p = p0·q1 + p1·q0 and q = q0·q1 combined with a challenge λ, is an ordinary statement form of degree 2 in the tables (degree 3 with eq). The existing `verify`, `next-claim` and `check-final` can therefore run it unchanged, and `forms.lisp` compiles the final check.

The characteristic bound needs attention. At p ≈ 2.1·10^9, a hot table entry in a 16k-token, 64-layer forward pass could approach p in total multiplicity. If the Python prover adopts Eagen–Haböck's generalized pole multiplicities, the verifier change is small, but it is a protocol change the port must follow ([ePrint 2024/2067](https://eprint.iacr.org/2024/2067)).

### Sable's parameters only add up under Haböck's Johnson-regime analysis

Sable uses rate 1/2, 153 queries and 30 grinding bits (`zk/gpcs.py`), and claims 2^−103 soundness with proven bounds (`README.md`). Unique decoding gives about 0.415 bits per query, roughly 93.5 bits in total. The Johnson regime gives about 0.5 bits per query, roughly 106 bits. Only the latter matches the claim, which suggests Sable already relies on Haböck's list-decoding analysis of BaseFold ([ePrint 2024/1571](https://eprint.iacr.org/2024/1571)). This is the research notes' reconstruction and should be confirmed against the paper's soundness section.

That conservatism is now justified, because three papers disprove the up-to-capacity conjectures that "conjectured security" parameter sets assume. Crites and Stewart disprove BCIKS correlated agreement up to capacity, WHIR's mutual-correlated-agreement conjecture, and the DEEP-FRI list-decoding conjecture ([ePrint 2025/2046](https://eprint.iacr.org/2025/2046)). Diamond and Gruen warn that "many deployed SNARKs may be less secure than previously assumed" ([ePrint 2025/2010](https://eprint.iacr.org/2025/2010)). Krachun, Kazanin and Haböck give a counterexample for RS codes over multiplicative subgroups of prime fields, which is exactly Sable's setting ([ePrint 2026/782](https://eprint.iacr.org/2026/782)). The Lisp port should therefore hard-code proven-regime parameters and reject any configuration that halves queries by appeal to capacity.

### WHIR is the verifier-speed upgrade, but it does not fix proof size

WHIR folds several variables per round, recommits on a smaller, lower-rate domain, and uses out-of-domain samples, so later rounds need far fewer queries. As a PCS at degree 2^22 and 100-bit security, its authors report **1.2 s to commit and open, 63 KiB, and a 360 µs verifier**, and they present it as a drop-in replacement for FRI, STIR and BaseFold ([ePrint 2024/1586](https://eprint.iacr.org/2024/1586)). It natively proves multilinear evaluation claims, which is what the sumcheck/GKR stack produces, so it suits Sable better than STIR. A Lisp WHIR verifier needs per-round Merkle roots, out-of-domain samples absorbed into the transcript, k-ary fold checks, and a final weight-polynomial evaluation, which for evaluation claims is the eq product Sable already computes. WHIR's 63 KiB scale also shows that the PCS does not explain Sable's 100–500 MiB proofs (`README.md`). The split between PCS and GKR/sumcheck bytes was not measured and should be checked with `proofio` first.

Recursive compression through accumulation (ARC, [ePrint 2024/1731](https://eprint.iacr.org/2024/1731); WARP, [ePrint 2025/753](https://eprint.iacr.org/2025/753)) would redesign the protocol, so it is out of scope for this port.

### Transcript hygiene is part of the port

Weak Fiat-Shamir, meaning not hashing the statement, broke 36 implementations across 12 proof systems ([Dao et al., ePrint 2023/691](https://eprint.iacr.org/2023/691)). Khovratovich, Rothblum and Soukhanov build false-but-accepting GKR proofs for crafted circuits, under any hash ([ePrint 2025/118](https://eprint.iacr.org/2025/118)).

Sable's fixed LLM circuit does not compute its own transcript hash. A future recursion layer that verifies BLAKE3 in-circuit (`zk/recur/`) does, and falls in the at-risk class. Round-by-round soundness is what justifies FS-compiled FRI-like protocols ([Block et al., ePrint 2023/1071](https://eprint.iacr.org/2023/1071)). Chiesa and Orrù give concrete bounds for duplex-sponge FS ([ePrint 2025/536](https://eprint.iacr.org/2025/536)).

For `transcript.lisp` this means three things. It must absorb the same tag, parameters, commitment roots and statement as the Python transcript, byte for byte. It should assert that nothing is squeezed after an unabsorbed send. And it should hide the hash behind leaf, node and permutation hooks, so that an outer-layer hash swap is a parameter change.

## Fiat-Cryptography's recipe already fits forms.lisp; e-graphs remain an open bet

The arXiv half of the source base is programming-languages and hardware work. The pattern that transfers best is the one Sable's Lisp already half-implements: write the arithmetic generically, partially evaluate it to straight-line code, and run a bounds analysis that decides word sizes and where reductions go.

| Paper | Authors, year | Source | What transfers |
|---|---|---|---|
| Simple High-Level Code for Cryptographic Arithmetic (Fiat-Cryptography) | Erbsen, Philipoom, Gross, Sloan, Chlipala, 2019 (IEEE S&P) | Not on arXiv; [author PDF](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf) | Partial evaluation, interval bounds, per-prime "champion" |
| CryptOpt: Verified Compilation with Randomized Program Search | Kuepper et al., 2023 (PLDI) | [arXiv 2211.10665](https://arxiv.org/abs/2211.10665) | Benchmark-driven search over schedule and instruction choice |
| Code Generation for Cryptographic Kernels using Multi-word Modular Arithmetic on GPU | Zhang, Franchetti, 2025 (CGO) | [arXiv 2501.07535](https://arxiv.org/abs/2501.07535) | Rewrite-system lowering of modular arithmetic |
| egg; egglog | Willsey et al., 2021 (POPL); Zhang et al., 2023 (PLDI) | [arXiv 2004.03082](https://arxiv.org/abs/2004.03082); [arXiv 2304.04332](https://arxiv.org/abs/2304.04332) | Equality saturation, e-class analyses (egglog reads s-expressions) |
| Vectorization for DSPs via Equality Saturation (Diospyros) | VanHattum et al., 2021 (ASPLOS) | Not on arXiv; [NSF PAR](https://par.nsf.gov/biblio/10232895) | Symbolic expansion, e-graph, then emit: 3.1× over vendor libraries |
| Datapath e-graphs; ROVER | Coward et al., 2022–23; 2024 | [arXiv 2204.11478](https://arxiv.org/abs/2204.11478); [arXiv 2303.01839](https://arxiv.org/abs/2303.01839); [arXiv 2406.12421](https://arxiv.org/abs/2406.12421) | Bit-width and range reasoning inside e-graphs |
| Cut Tracing with E-Graphs for Boolean FHE Circuit Synthesis | 2025 | [arXiv 2506.12883](https://arxiv.org/abs/2506.12883) | Negative result: whole-circuit EqSat scales poorly |
| zkPHIRE; zkDuel | Daftardar et al., 2025 (HPCA 2026); Narang, Daftardar, Reagen, 2026 | [arXiv 2508.16738](https://arxiv.org/abs/2508.16738); [arXiv 2610.37710](https://arxiv.org/abs/2610.37710) | Sumcheck as "updates, extensions, products"; limits of unrolling |
| Souper; Minotaur | Sasnauskas et al., 2017; Liu, Mada, Regehr, 2024 (OOPSLA) | [arXiv 1711.04422](https://arxiv.org/abs/1711.04422); [arXiv 2306.00229](https://arxiv.org/abs/2306.00229) | Mining peephole rules |

### Bounds analysis turns the tower generator into a lazy-reduction generator

Fiat-Cryptography found that "simple partial evaluation is sufficient to transform such initial code into the fastest-known C code". It then runs a bounds pass that "works by standard abstract interpretation with intervals" to choose word sizes and prove the absence of overflow. The result beat GMP by **1.25× to 10×** and shipped in BoringSSL ([Fiat S&P'19](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)).

`mul*` and `inv*` are already the partial-evaluation half. The missing half is an interval pass over the symbolic coordinates, with each node carrying `[lo, hi]`. It would emit unreduced `(unsigned-byte 64)` sums and insert a `mod` only where `hi` would pass 2^64. That pass automatically produces the lazy schoolbook from the first section, with its two 4-product sums per F_p8 coefficient. Applied to `forms.lisp` statement bodies, it would also size the accumulators in the per-statement round kernels.

Fiat also generates several candidates per prime and keeps a "champion". The Lisp analogue is to keep schoolbook-lazy, Karatsuba-at-the-top and Montgomery templates, `COMPILE` each one, time them on the host, and cache the winner.

CryptOpt adds a search over orderings and instruction choices, with each candidate benchmarked on the real CPU. It reports **a mean 1.74× over GCC 12 and 1.40× over Clang 15** on Fiat's output ([arXiv 2211.10665](https://arxiv.org/abs/2211.10665)). In Lisp the search space is the order of `let*` bindings and the formula chosen per node. Equivalence is easier to check than in CryptOpt: every candidate is a polynomial map over F_p, so random-point evaluation gives a Schwartz–Zippel check, and `normalize` in `forms.lisp` gives an exact one. SBCL does its own register allocation and scheduling, so expect less than CryptOpt's gain. That expectation is unmeasured.

### Concrete staging changes in the Lisp code

There are four changes, from most to least concrete.

**Generate the tower at load time.** The tower is fixed, so nothing requires compiling it on first use at run time. A `define-tower-ops` macro can expand `tower-code` at macroexpansion time into ordinary `defun`s, declaimed `inline` and destination-passing. Compiler macros on `ext*` and `ext+` can then splice them into typed callers. Run-time `COMPILE` stays for statement forms only.

**Hash-cons the IR.** Lightweight Modular Staging's smart constructors give common-subexpression elimination and constant folding during construction (Rompf & Odersky, GPCE 2010, [DOI](https://doi.org/10.1145/1868294.1868314); bibliographic data not re-verified). Fiat says its strategy "is most similar to the one from LMS" ([Fiat S&P'19](http://adam.chlipala.net/papers/FiatCryptoSP19/FiatCryptoSP19.pdf)).

**Compile whole round kernels, not per-point functions.** Today `statement-function` wraps the compiled form in `(lambda (&rest table-values) (apply f (append …)))`, and `round-message` calls it per pair and per point, then `reduce`s a freshly consed list. A staged kernel would be `(lambda (tables eq-l eq-r out) …)`. It would loop over pairs with the slope and incremental line values, inline F, accumulate into unreduced lanes, and write d values into `out`. The binding-time split then becomes structural: one kernel for base-field rounds (ss and sl arithmetic), one for extension rounds, and a separate fold kernel with the sl multiply. zkPHIRE breaks programmable sumcheck into exactly these "updates, extensions, and products" ([arXiv 2508.16738](https://arxiv.org/abs/2508.16738)).

**Cap the unrolling.** zkDuel found that "the largest fully unrolled constraints still hit compiler-scalability limits". It also found that specialized point interpolation gave 1.55–1.75× on evaluation kernels ([arXiv 2610.37710](https://arxiv.org/abs/2610.37710)). Above a size threshold, the generator should split F into sub-lambdas rather than hand `COMPILE` one enormous form.

### E-graphs are untested territory for ZK

Diospyros (symbolic expansion, then equality saturation, then emit) is the closest precedent, at 3.1× over vendor DSP libraries ([NSF PAR](https://par.nsf.gov/biblio/10232895)). The research found **no published application of equality saturation to ZK constraints or sumcheck composition polynomials**. Applied to large Boolean FHE circuits, equality saturation "struggles to scale" and loses to greedy methods ([arXiv 2506.12883](https://arxiv.org/abs/2506.12883)).

For Sable, hash-consing plus kernel/co-kernel factoring come first. An egglog pass is worth trying only per tower level, or on round-kernel bodies, with node limits. It is a research bet that this s-expression codebase is unusually well placed to try. It is not a planned speedup.

## Cross the FFI only in batches, and only for SIMD

Every SBCL mechanism below works in 2.2.9. The question is when each one beats well-typed Lisp.

| Mechanism | Pays off when | Does not pay off when | Evidence |
|---|---|---|---|
| Tight types (`(integer 0 2130706432)`, `truly-the`, u64 lazy sums) | Always: prerequisite for the rest | — | Local: fold 15.6 → 2.2–4.6 ns/element |
| `dynamic-extent` temporaries, destination-passing | Any per-element EXT temporary | Large arrays at `(safety 0)` (no stack-overflow check) | Local: 48 → 0 B/iteration, 12.4 → 4.7 ns ([SBCL manual §6.7](http://www.sbcl.org/manual/#stack-allocation)) |
| Inline `define-alien-routine` + `vector-sap` + `with-pinned-objects` | Arrays of ~1,000+ elements, and the C side uses SIMD SBCL won't emit | Per element or per EXT (2–3× slower); scalar C (no gain) | Local: 2.4 ns no-op call; pinning +0.9 ns; `without-gcing` 18.9 ns; break-even about 1,024 elements |
| sb-simd AVX2 (`u64.4-mul` = `vpmuludq`) | Compute-bound bulk kernels in structure-of-arrays layout | Single EXT ops; memory-bound folds at 2^22 | Heisig & Köstler: up to 94% of C ([ELS'22, Zenodo](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)) |
| `sb-kernel:%multiply-high`, `sb-bignum:%multiply` | 64×64→128 needs (not KoalaBear) | 31-bit fields: products fit in 62 bits | Exist in 2.0.0 and 2.2.9 ([arith.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/x86-64/arith.lisp)) |
| Custom `define-vop` / `inline-vop` | A BLAKE3 G function, if disassembly shows SBCL missing `ROR` | Anything SBCL already emits well | Internal API; template in Ironclad's BLAKE2s VOP |

### FFI: fixed costs dominate small calls

SBCL's manual says efficiency-critical foreign routines should be declaimed inline, so that word integers and pointers pass unboxed ([SBCL manual](http://www.sbcl.org/manual/)). On x86-64, `with-pinned-objects` only keeps objects live on the conservatively scanned stack, through an empty-bodied `touch-object` VOP ([pinned-objects.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/generic/pinned-objects.lisp)).

Locally, three results stood out. A C `mulmod` called per element cost 7.0 ns, against 5.3 ns for inline Lisp. A fold over 8,192 elements in cache ran at 2.2–3.7 ns per element in typed Lisp and 2.4–3.4 ns in gcc `-O3 -march=native` C. At 2^22 elements both ran at 4.58 ns per element, the memory-bandwidth limit.

| Array length | Typed Lisp | One FFI call (3 arrays, pinned) |
|---|---|---|
| 4 | 10 ns | 32 ns |
| 8 | 17 ns | 38 ns |
| 64 | 151 ns | 233 ns |
| 1024 | 2,949 ns | 2,089 ns |

The conclusion is that a scalar C kernel buys nothing. The FFI is worth it only when the C side brings SIMD.

### BLAKE3 is the strongest FFI candidate

Ironclad has BLAKE2s, with an SBCL x86-64 VOP for its mixing function, but no BLAKE3 ([Ironclad x86oid-vm.lisp](https://github.com/sharplispers/ironclad/blob/master/src/opt/sbcl/x86oid-vm.lisp)). BLAKE3's G function is the same 32-bit add/xor/rotate (16, 12, 8, 7). The first step for `blake3.lisp` is therefore to check `disassemble` for real `ROR` instructions from `sb-rotate-byte` and `ldb (byte 32 0)` arithmetic ([sb-rotate-byte](https://github.com/sbcl/sbcl/tree/master/contrib/sb-rotate-byte)), and to borrow Ironclad's VOP only if they are missing.

Merkle leaves and query paths are many independent compressions. That is what the official C library's SIMD backends parallelize, and it is what dominates a BaseFold verifier with 153 query paths. One `alien-funcall` per batch of leaves, or an 8-lane `u32.8` G function in sb-simd, is the path to C-class throughput. Lisp-versus-C BLAKE3 throughput was not measured.

### sb-simd brings Plonky3's AVX2 Montgomery within reach

Heisig and Köstler's sb-simd (ELS'22, published on Zenodo, not arXiv) compiles most calls "to a single machine instruction". It reached up to 94% of C on a Jacobi stencil. The remaining 6–33% gap comes from SBCL being unable to fold loads into arithmetic instructions ([ELS'22 PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)). It became a contrib in SBCL 2.2.6.

Locally, `(u64.4-mul …)` performs exact 32×32→64 lane products. That is enough to express Plonky3's AVX2 Montgomery multiply (even and odd `vpmuludq`, quotient by μ, times p, subtract, `min` correction), at about 2 elements per cycle ([avx2/packing.rs](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)). Scalar SBCL manages about 0.1 multiplies per cycle.

The potential gain of more than 10× applies only to compute-bound kernels: extension-heavy round accumulation, and eq tables in cache. It requires vectorizing across elements in structure-of-arrays layout, not within one EXT element. This is an inference; no sb-simd modular kernel was measured.

Several sb-simd bugs were fixed after 2.2.9: shuffles in 2.4.2, AVX2 shifts in 2.6.5, and AVX-512 support arrived only in 2.6.9 ([SBCL NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)). On 2.2.9, every SIMD kernel therefore needs randomized differential tests against the scalar path. A version of at least 2.4.2, preferably 2.6.9 or later, is advisable if the version can be chosen.

VOPs are documented mainly by Paul Khuong's 2014 tutorial ([pvk.ca](https://pvk.ca/Blog/2014/08/16/how-to-define-new-intrinsics-in-sbcl/)) and SBCL's source. They use unexported APIs that change between releases, so keep any custom VOP in one `#+(and sbcl x86-64)` file with a portable fallback.

## A prioritized upgrade plan for the Lisp port

The order follows evidence per unit of effort. Local measurements come first, then published results on comparable settings, then protocol changes that need the Python prover and CUDA kernels to move with them.

| # | Change | Where | Evidence and expected gain | Protocol change? |
|---|---|---|---|---|
| 0 | Profile with `sb-sprof`; add randomized differential tests of every new kernel against the current code and `tests/vectors.sexp` | tests | Prerequisite; no prover profile was taken in this research | No |
| 1 | `deftype fe (integer 0 2130706432)`; branchless `fadd`/`fsub`; mul-by-3 as adds; destination-passing `ext+`/`ext-`/`ext*` (typed loops instead of `MAP`); inline constant-size allocation or `dynamic-extent` | field.lisp | Local: about 3× from branchless alone; 48 → 0 B/op | No |
| 2 | Generate F_p4/F_p8 multiply as lazy schoolbook over X^4 − 3 / X^8 − 3, with the bit-reversal permutation applied at codegen time; expand at macroexpansion time, not first-use `COMPILE` | field.lisp | Local: F_p8 multiply ~500 → 53–63 ns (8–10×); Plonky3 uses the same form | No |
| 3 | Flat `(unsigned-byte 32)` table storage (stride k for extension data); base tables stay base until folded; dedicated sl multiply in fold and line values; slope computed once with incremental additions | sumcheck.lisp, field.lisp | P1 §2.1: sl costs k base multiplies against 27 now; removes per-element EXT boxing | No |
| 4 | Staged per-statement round kernel (loop over pairs, F inlined, u64 accumulators, `out` buffer) and fold kernel; separate base-round and extension-round kernels; size cap | forms.lisp, sumcheck.lisp | Fiat/LMS staging; zkPHIRE decomposition; removes `&rest`/`apply`/list consing per point | No |
| 5 | Dao–Thaler split eq (two √N tables, no per-round full eq table) | sumcheck.lisp | ePrint 2024/1210: about 10% on top of Gruen, plus memory | No |
| 6 | Delayed first fold, then SVO Algorithm 6 with ℓ0 = 2–3, choosing ℓ0 from the measured κ after steps 1–3 | sumcheck.lisp | P3 model: 1.31× (F_p4), 1.85× (F_p8) for d = 2 | No |
| 7 | Port the LogUp-GKR verifier (Papini–Haböck) as statement forms on the existing sumcheck verifier; port the BaseFold opening with proven Johnson-regime parameters; transcript assertions; hash abstraction | new verifier files, transcript.lisp | ePrint 2023/1284, 2023/1705, 2024/1571; FS: 2023/691, 2025/118 | No (follows Python) |
| 8 | GCD or addition-chain `finv`; batch inversion for bulk denominators | field.lisp | Fermat about 300 ns; batch about 3 multiplies per element | No |
| 9 | BLAKE3: verify `ROR` emission, type the compression; then batched FFI to the C library or 8-lane sb-simd for Merkle leaves | blake3.lisp | Local FFI break-even about 1k elements; verifier dominated by 153 query paths | No |
| 10 | sb-simd AVX2 Montgomery for compute-bound bulk kernels, structure-of-arrays, behind `instruction-set-case` with a scalar fallback | new simd file | Plonky3: 2 elements/cycle against about 0.1 scalar; pin SBCL ≥ 2.4.2 or test exhaustively | No |
| 11 | Template "champion" autotuning per host, cached; egglog experiments per tower level | codegen | CryptOpt 1.40–1.74× (assembly-level, not SBCL); EqSat for ZK unexplored | No |
| 12 | Coordinated protocol upgrades: X = ∞ evaluation point; WHIR in place of BaseFold; only if justified, Eagen–Haböck multiplicities; skip univariate skip and projective sumcheck for now | all ports | WHIR: 63 KiB, 360 µs at 2^22; leanVM skip −11.8% and rejected; P8 about 10% on BN254 only | Yes |

Steps 1–4 are pure engineering, with local evidence that the field layer alone gains close to an order of magnitude. They should land before any algorithmic change is benchmarked, because the published sumcheck gains are ratios of ll to sl to ss costs, and the current Lisp code distorts those ratios with branch mispredictions and allocation. Steps 5 and 6 are paper-backed and leave the protocol unchanged. Step 7 is the outstanding functional work. Steps 9 and 10 are where "alien engineering" earns its keep. Step 12 is the only part that needs cross-implementation agreement.

## Conclusion

The research changes the order of the port more than its contents. The algorithmic papers are worth implementing, but they assume a field layer that already runs at hardware speed. In Sable's Lisp, most of the cost of an extension multiply comes from mispredicted branches and allocation. Until that is fixed, measured gains from small-value sumcheck or split eq would mostly reflect Lisp overheads rather than the algorithms. The cost ratio κ that sets P1's optimal ℓ0 should be measured after the field rewrite, not before. Plugging in today's ~100:1 extension-to-base cost would overstate SVO's value. With lazy schoolbook (64 products and 16 reductions against 8 and 8 for sl), the ratio lands near P3's Karatsuba model, so the 1.3–1.85× predictions are the right expectation.

Two strategic points follow. Sable's choice of proven bounds has become an asset: the 2025–26 counterexamples close off the cheap route of cutting query counts, which leaves WHIR as the one verifier upgrade that is worth renegotiating the protocol for. And because Sable's statements are already s-expressions that `forms.lisp` normalizes and compiles, the codebase is better placed than most to test two ideas no published work has tried for ZK: bounds-driven automatic lazy reduction across whole sumcheck kernels, and equality saturation of composition polynomials.
