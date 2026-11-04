# Fast arithmetic for 31-bit prime fields (KoalaBear, BabyBear, M31) and their degree-4/8 extensions: what applies to the Sable SBCL implementation

Scope: KoalaBear p = 2^31 - 2^24 + 1 = 0x7f000001 = 127·2^24 + 1; BabyBear p = 2^31 - 2^27 + 1 = 15·2^27 + 1; Mersenne-31 p = 2^31 - 1. Extensions are degree 4 and 8 towers or binomial extensions (X^4 - W, X^8 - W).

Sources used:
- Plonky3: shallow clone of github.com/Plonky3/Plonky3 at commit `0d803737b681774186cd283b1fd2d8e493655c82` (2026-10-26). Code links below are pinned to that commit, and line numbers refer to it.
- SBCL source: github.com/sbcl/sbcl master, 2026-11-04.
- The Sable code: `/home/user/Sable/lisp/src/field.lisp`.
- Microbenchmarks I ran with the local SBCL 2.2.9 on a 4-vCPU "Intel Xeon @ 2.10GHz" VM (flags include avx2, avx512f and avx512ifma). Timings in a VM are noisy, so treat them as ±20%. The scripts are in the session scratchpad (`bench/base.lisp`, `bench/ext2.lisp`, `bench/bl.lisp`).
- Direct PDF downloads from eprint.iacr.org were blocked (each returned a 5 KB challenge page). ePrint facts below therefore come from the abstract pages, fetched with WebFetch.

## Q1. Montgomery vs Barrett vs special-form reduction for KoalaBear on 64-bit CPUs; what Plonky3 uses

### Takeaway
- **Scalar x86-64:** Barrett-style reduction (a constant "magic" multiply-high) and Montgomery reduction both cost about 3 multiplies. SBCL already compiles `(mod (* a b) p)` to the Barrett form, with no DIV instruction, and in my measurements it beat a hand-written Montgomery reduction (about 5.2 vs 6.8 ns latency).
- **Special form:** a Solinas-style fold gains only 7 bits per step for KoalaBear (4 bits for BabyBear), so it is not competitive. Only M31 gets a cheap reduction from its special form: shift, add, conditional subtract.
- **Plonky3:** it stores KoalaBear and BabyBear in Montgomery form (R = 2^32) everywhere. Montgomery's real advantage shows up in SIMD (AVX2/AVX-512 `vpmuludq`, NEON `vqdmulh`), where no 64-bit multiply-high exists, and in the larger lazy-accumulation range it allows.

### Cited Findings
- **Plonky3 scalar Montgomery reduction (`monty_reduce`).**
  - Steps: `t = (x * MONTY_MU) mod 2^32`; `u = t*P`; `(x - u) >> 32`; add P if the subtraction borrowed.
  - Input must be in [0, 2^32·P); output is in [0, P).
  - Cost: 2 multiplies plus a sub, shift and conditional add. The full product a·b adds a third multiply.
  - Constants: `MONTY_MU = 0x81000001` for KoalaBear and `0x88000001` for BabyBear.
  - Source: [Plonky3 monty-31/src/utils.rs L99-125 and test constants L244-258](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/utils.rs).
- **`large_monty_reduce`.** It accepts [0, 2·2^32·P) by first subtracting P·2^32 once with a conditional. It is "slower than `monty_reduce` but has a larger input range". — [utils.rs L127-158](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/utils.rs)
- **Scalar multiply.** `MontyField31::mul` is `monty_reduce(self.value as u64 * rhs.value as u64)`. — [monty-31/src/monty_31.rs L792-796](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs)
- **One implementation for both fields.** BabyBear and KoalaBear share the generic `monty-31` crate. A HackMD write-up on Plonky3 small fields says Monty-31 add/mul speed is "identical regardless of the prime", which leaves the prime free to be chosen for other reasons. — [Small Fields in Plonky3 (HackMD, @Syxton, numbers as of 18-Apr-2025)](https://hackmd.io/@Syxton/small_fields_in_plonky3)
- **Mersenne-31 reduction in Plonky3.** `from_u62(x)` is `lo31(x) + (x >> 31)` followed by a modular add, with no multiply. — [mersenne-31/src/mersenne_31.rs L636-639, L692-697](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/mersenne-31/src/mersenne_31.rs)
- **AVX2 packed Montgomery multiply (8 lanes of 32 bits).** Per the code comment, the sequence is:
  - 2× `vmovshdup`, then 6× `vpmuludq` (product even/odd, q = prod·MU, q·P), then 2× `vpsubq`, `vmovshdup`, `vpblendd`.
  - Code comment: "throughput: 4 cyc/vec (2 els/cyc), latency: 19 cyc".
  - The output is left signed in (-P, P) to save a correction.
  - Source: [monty-31/src/x86_64_avx2/packing.rs L425-450](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)
- **Fused dot products amortize the reduction (AVX2, per code comments).**
  - `dot_product_2`: "6.67 cyc/vec (1.20 els/cyc)", 4 vectors of inputs.
  - `dot_product_4`: "11.67 cyc/vec", i.e. 8 `vpmuludq` for the products plus only 4 for one shared reduction.
  - Source: [avx2/packing.rs L510-536, L577-624](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)
- **AVX-512 Montgomery multiply.** Code comments give "throughput: 6.5 cyc/vec (2.46 els/cyc), latency: 21 cyc" for 16 lanes. — [monty-31/src/x86_64_avx512/packing.rs ~L539-540](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx512/packing.rs)
- **NEON Montgomery multiply.**
  - Uses `vqdmulhq_s32` (signed doubling multiply returning the high half).
  - Mul comment: "throughput: 1.75 cyc/vec (2.29 els/cyc), latency: (lhs->) 11 cyc, (rhs->) 14 cyc".
  - Add: 0.25 cyc/vec (16 els/cyc).
  - Source: [monty-31/src/aarch64_neon/packing.rs L424-481](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/aarch64_neon/packing.rs)
- **Measured SIMD throughput table (elements/cycle; latency in cycles).** From the HackMD write-up:

  | ISA | M31 mul | Monty-31 mul | Add |
  |---|---|---|---|
  | NEON | 3.2 (10) | 2.29 (11, 14) | 5.33 (6) |
  | AVX2 | 2 (13) | 1.71 (21) | 8 (3) |
  | AVX-512 | 2.91 (15, 14) | 2.46 (21) | 10.67 (3) |

  The AVX2 Monty figure (1.71) differs from the code comment (2 els/cyc); the write-up does not explain why. — [HackMD Small Fields in Plonky3](https://hackmd.io/@Syxton/small_fields_in_plonky3)
- **NEON Barrett multiplication in the post-quantum literature.** Becker, Hwang, Kannwischer, B.-Y. Yang and S.-Y. Yang combine Montgomery multiplication and Barrett reduction into "Barrett multiplication" for one-known-factor modular multiplication on Armv8-A NEON. It is the same `sqdmulh`-style idea Plonky3 uses on NEON. — ["Neon NTT: Faster Dilithium, Kyber, and Saber on Cortex-A72 and Apple M1", TCHES 2022, IACR ePrint 2021/986](https://eprint.iacr.org/2021/986)
- **SBCL already applies Barrett-like division by a constant.** With `(declare (type fp a b))`, `(mod (* a b) +p+)` compiles to:
  - `IMUL`, `MOV RCX, 2323999252290013197`, `MUL` (high half), `SAR RDX, 28`, `IMUL RDX, 2130706433`, `SUB`.
  - That is 3 multiplies and no DIV. Verified by `disassemble` under SBCL 2.2.9 (my run).
  - The transform is the Granlund–Montgomery "choose-multiplier" with `%multiply-high`, plus `fastrem-32/64`. — [sbcl/src/compiler/srctran.lisp ~L5280-5330, L5445-5471](https://github.com/sbcl/sbcl/blob/master/src/compiler/srctran.lisp)
- **My SBCL 2.2.9 microbenchmark (ns per base multiply, KoalaBear, x86-64 VM @ 2.1 GHz).**

  | Variant | Dependent-chain latency | 4 independent chains (per 4 muls) |
  |---|---|---|
  | `(mod (* a b) p)` (current) | 5.2–5.3 | 5.9–6.1 |
  | Hand-written Montgomery, `(unsigned-byte 64)` words, branchy final correction | 6.8–7.0 | 8.2–9.1 |
  | M31 fold (for comparison) | 2.7–2.8 | 3.6–5.3 |

  Source: my measurement, `bench/base.lisp`. The Montgomery result was checked on 100,000 random pairs.

### Inferences
- **Special form for KoalaBear:**
  - 2^31 ≡ 2^24 - 1 (mod p). One fold maps a 62-bit x = h·2^31 + l to h·(2^24-1) + l < 2^55, gaining only 7 bits.
  - It takes about 5 folds plus a final correction, so it loses to the 3-multiply Barrett/Montgomery forms.
  - For BabyBear (2^31 ≡ 2^27 - 1) each fold gains only 4 bits.
  - Only M31 has a 1–2-fold reduction, and that is why it was the fastest scalar variant measured (about 2× faster than KoalaBear `mod`).
- **For scalar SBCL code, keep canonical representation and SBCL's `mod`-by-constant (Barrett).** Switching to Montgomery form buys nothing scalar. It adds to/from-Montgomery conversions at I/O boundaries and complicates comparison with the Python/C++ reference (zk/field.py, zk/cuda/vcore.cpp per the field.lisp header).
- **Montgomery becomes the right choice only if the code goes SIMD**, because:
  - AVX2/AVX-512 have only 32×32→64 lane multiplies (`vpmuludq`) and no 64-bit multiply-high, so Barrett's mulhi is unavailable.
  - Montgomery needs only low halves.
  - sb-simd exposes `vpmuludq` as `sb-simd-avx2:u64.4-mul` (see Q4).
- **The magic-constant path depends on the declared types.** SBCL's 1-product fixnum fits are what let it emit the 3-multiply sequence. Without declarations, or with an integer type wider than 64 bits, it falls back to generic arithmetic.

### Gaps
- I found no published scalar (non-SIMD) cycle comparison of Montgomery vs Barrett specifically for 31-bit primes on x86-64. Plonky3 documents only the SIMD costs.
- My Montgomery implementation used a branch for the final correction. A branchless version might close part of the gap, but I did not test it.

## Q2. Delayed (lazy) reduction: how many products fit in 64 bits; dot products and extension multiplication with one final reduction

### Takeaway
- Four products of canonical elements (each < p^2 < 2^62) fit in an unsigned 64-bit word for all three primes. Plonky3's Montgomery variant reduces sums of up to 4 products with a single `large_monty_reduce`, and 8-term sums with two 4-sums plus a cheap conditional subtract.
- In SBCL, only one product fits in a fixnum (< 2^62), so lazy sums must use `(unsigned-byte 64)` declared words.
- My benchmark: an SBCL-compiled X^8 - 3 schoolbook multiply with lazy 4-product sums reduced by `mod` ran 3–4× faster than the current Karatsuba tower.

### Cited Findings
- **Plonky3 `dot_product<N>` (Monty-31).** Its comments:
  - "As all values are < P < 2^31, the products are < P^2 < 2^31P." Two products stay below MONTY·P and go to `monty_reduce`.
  - 3 or 4 products are "less than 2·MONTY·P" and go to `large_monty_reduce`.
  - For 5 to 8 terms: two partial sums of at most 4 products each. Each is brought below MONTY·P with `min(s, s - P·2^32)` (a branchless conditional subtract via wrapping sub and min), then they are added and passed to one `large_monty_reduce`.
  - For larger N: u128 accumulation in chunks of 4.
  - Source: [monty-31/src/monty_31.rs L316-~420](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs)
- **Plonky3 `Sum`.** It accumulates u64 values and applies one `% P` at the end; the comment says this "will not overflow so long as iter.len() < 2^33". For N ≤ 7, direct modular adds were faster per benchmarking ("for N <= 7 it's faster to sum the elements directly"). — [monty_31.rs L297-313, L805-813](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs)
- **u128 reduction.** `monty_reduce_u128` reduces x < 2^96 via a 64-bit Montgomery step on the low limb plus `to_monty(hi)`, with "no division instruction at all". — [utils.rs L160-~225](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/utils.rs)
- **Extension multiplication is written as dot products.** Each output coefficient of `quartic_mul` and `octic_mul` is one `R::dot_product`, so the delayed reduction above applies per coefficient. For the octic case: `res[k] = dot_product::<8>(a, row_k)`, where row_k is b reversed with the wrapped tail pre-multiplied by W. — [field/src/extension/binomial_extension.rs `quartic_mul` L779-821, `octic_mul` L1142-~1185](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
- **Lazy reduction in extension towers (classic reference).** Aranha, Karabina, Longa, Gebotys and López "generalize lazy reduction techniques, previously considered only for arithmetic in quadratic extensions" to whole towers. With other improvements this gave 27%–33% speedups for pairings on 64-bit platforms. — ["Faster Explicit Formulas for Computing Pairings over Ordinary Curves", Eurocrypt 2011, IACR ePrint 2010/526](https://eprint.iacr.org/2010/526)
- **Headroom drives field choice in newer work.** MamaBearZKP (ePrint 2026/1698, ACM CCS 2026, Zhang, Guo, Lu, Zhang, Cheng):
  - Proposes p = 2^49 - 2^34 + 1 for AVX-512 IFMA, because "the 49-bit field's headroom enables efficient lazy reduction".
  - It also says BabyBear leaves the 52-bit IFMA multiplier width underused.
  - Reports up to 18× single-thread end-to-end over Plonky3's AVX-512 BabyBear backend. That is a whole-prover number.
  - Source: [IACR ePrint 2026/1698](https://eprint.iacr.org/2026/1698)
- **My SBCL measurement (`bench/ext2.lisp`, ns per F_p8 multiply, chained, KoalaBear).**
  - Correctness: the schoolbook version matched the existing Karatsuba code on 20,000 random pairs, after mapping between bases (see Q3).

  | Variant | ns/mul | Allocation |
  |---|---|---|
  | Current `TOWER-FUNCTION 'mul* 8` (Karatsuba, fresh vector) | 502–568 | 48 B/op |
  | Same, with `make-ext` replaced by inline `(make-array 8 :element-type '(unsigned-byte 32))` | 447–473 | 48 B/op |
  | Schoolbook X^8 - 3, lazy: per coefficient two 4-product `(unsigned-byte 64)` sums, each `mod p`, then `fadd`; w·b_j = 3b_j via 2 `fadd` | 129–193 | 48 B/op |
  | Same schoolbook, writing into a caller-supplied destination | 53–63 | 0 B/op |

  The disassembly of the lazy version contains no DIV; SBCL applies its magic-constant path to the u64 `mod`.

### Inferences
- **Headroom per prime.** (p-1)^2 is about 2^61.98 (KoalaBear), 0.879·2^62 (BabyBear) and about 2^62 (M31), so:
  - floor((2^64 - 1)/(p-1)^2) = 4 for all three. Four lazy products per u64 word; 2 if the sum must stay below 2^63 (signed).
  - In SBCL fixnums (most-positive-fixnum = 2^62 - 1) only 1 product fits, so a lazy sum must be declared `(unsigned-byte 64)`. SBCL keeps such words unboxed in registers inside one function, but boxes them if they escape.
  - An 8-term coefficient therefore needs either (a) two 4-sums each reduced separately (my benchmark), (b) Plonky3's trick of a conditional subtract of P·2^32 on each 4-sum, then one Montgomery reduction (valid only with Montgomery inputs), or (c) a 128-bit accumulator, which SBCL handles poorly.
- **Barrett variant of (b).** With canonical inputs, a 4-sum S < 4p^2 can have p·2^33 conditionally subtracted (still < 2^64 headroom), but there is no cheap "one reduction for 8 products" without Montgomery. Two `mod`s per coefficient (16 per F_p8 multiply) is the practical scalar minimum in canonical form.
- **A Montgomery-form variant would need** 8 reductions plus conditional subtracts. That saves about 8 multiplies per F_p8 multiply but costs conversions. This is a modest gain; prioritize allocation and branches first (Q3).

### Gaps
- I did not benchmark a Montgomery-form lazy octic multiply in SBCL, nor a branchless `fadd` inside the lazy schoolbook. Both are likely further gains.
- The reason the fresh-vector lazy version is about 2.5× slower than the destination version is not fully diagnosed. Observations and hypothesis:
  - 48 B/op allocation; `ALLOC-TRAMP` appears in the disassembly.
  - Suspected GC/nursery or page-touch costs, not measured.

## Q3. Extension-field multiplication cost: schoolbook vs Karatsuba vs direct binomial formulas for X^4 - W and X^8 - W; effect of W = 3

### Takeaway
- **The Sable tower is the same field as X^8 - 3, in a permuted basis.** F_p8 = F_p4[z]/(z^2 - y), F_p4 = F_p2[y]/(y^2 - x), F_p2 = F_p[x]/(x^2 - 3), so z^8 = 3. Tower coordinate index (bit0 = x, bit1 = y, bit2 = z) maps to the power of z by 3-bit bit-reversal: x^i·y^j·z^k = z^(4i+2j+k). I verified this in code.
- **Multiplication counts.** Karatsuba over the 2-2-2 tower needs 27 general multiplications. Direct binomial schoolbook needs 64 products, but in exchange it has only 8 reductions and 7 multiplications by W. Plonky3 uses the direct schoolbook-as-dot-product form for both degree 4 and 8.
- **In SBCL the multiplication count is not the bottleneck.** The current Karatsuba code executes about 95 data-dependent branches (one per modular add/sub). Making add/sub branchless alone cut the cost about 3×, from about 450 ns to about 140–160 ns.

### Cited Findings
- **Plonky3 extension constants and implementations.**
  - KoalaBear uses `BinomialExtensionData<4>` and `<8>` with `W = 3`, and implements `mul_w(a) = a.double() + a` (two adds, no multiply). — [koala-bear/src/koala_bear.rs L93-112](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/koala-bear/src/koala_bear.rs)
  - BabyBear uses W = 11 for degree 4 and 8, and W = 2 for degree 5 (`mul_w = a.double()`). — [baby-bear/src/baby_bear.rs L60-100](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/baby-bear/src/baby_bear.rs)
  - M31 uses CM31 = F_p[i]/(i^2 + 1) and QM31 = CM31[u]/(u^2 - (2 + i)), where multiplying by W = 2 + i is "using only additions": (a+bi)(2+i) = (2a-b) + (a+2b)i. — [mersenne-31/src/qm31.rs L1-45](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/mersenne-31/src/qm31.rs)
  - The degree-4/8 multiply dispatches to `quartic_mul_packed` / `octic_mul_packed`, which are architecture-specific. — [monty-31/src/extension.rs L22-37](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/extension.rs)
  - Without SIMD these fall back to the generic `quartic_mul` / `octic_mul`. — [monty-31/src/no_packing/mod.rs L12-46](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/no_packing/mod.rs)
- **Generic quartic and octic formulas.**
  - Quartic: `res[0] = a0·b0 + W(a1b3 + a2b2 + a3b1)`, …, `res[3] = a0b3 + a1b2 + a2b1 + a3b0`. That is 16 products plus 1–2 multiplies by W, in 4–6 small dot products.
  - Octic: "Constant coefficient = a0*b0 + w(a1*b7 + ... + a7*b1)", … with `res[6]`/`res[7]` as 8-term dot products. That is 64 products and 8 reductions.
  - Source: [field/src/extension/binomial_extension.rs L779-821, L1142-~1185](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
- **AVX2 octic multiply.**
  - Puts `b` in one 8-lane vector, computes `w_b = mul_w(packed_b)`, and builds 8 rotated rows `[w_b[7], b0..b6]`, etc. The product is "an 8-term dot product" of broadcast `a_i` against the rows: one SIMD dot product computes all 8 output coefficients.
  - The code carries "TODO: It's plausible that this could be improved by a custom AVX2 implementation".
  - Source: [monty-31/src/x86_64_avx2/packing.rs L1565-~1610](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)
- **Multiplying by a fixed extension element.** `binomial_mul_by_scalar` folds W into the precomputed row of the fixed operand, so "a SIMD left operand pays one reduction per coefficient, and nothing for W". — [binomial_extension.rs L599-656](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
- **Squaring.** Dedicated `quartic_square` and `octic_square` exist through `binomial_square`. — [binomial_extension.rs L658-680](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
- **Current Sable code** (`/home/user/Sable/lisp/src/field.lisp`, `MUL*`, `MUL-BY-GENERATOR*`, `TOWER-CODE`):
  - Every multiply by the non-residue is emitted as `(fmul +nonres+ x)`, a full `mod`.
  - Every add/sub is a branchy `fadd`/`fsub`.
  - The result vector comes from `make-ext`, which is not declaimed inline and takes a variable size.
  - Disassembly of the compiled degree-8 multiplier (my run) shows a full call through `#<FDEFN MAKE-EXT>`, and the 8 result stores go through `SB-KERNEL:HAIRY-DATA-VECTOR-SET/CHECK-BOUNDS`, i.e. generic, not open-coded. The compiled function is about 1,570 disassembly lines.
- **Branch removal (my measurement, `bench/bl.lisp`).**
  - Change: replaced `fadd`/`fsub` in the generated Karatsuba code by branchless versions, `s = a + b - p; s + (p & (s >> 63))`, plus inline allocation.
  - Result: F_p8 multiply went from 447–473 ns to 138–162 ns, i.e. about 3× faster, with the same 27 base multiplies. Output equality was checked on 2,000 random inputs.

### Inferences
- **Counting the current generated code (from `MUL*`):**
  - F_p2 multiply = 3 general `fmul` + 1 `fmul` by 3 + 5 add/sub.
  - F_p4 = 3·F_p2 + 1 by-3 + 10 add/sub → 9 general, 4 by-3, 25 add/sub.
  - F_p8 = 3·F_p4 + 1 by-3 + 20 add/sub → **27 general multiplies, 13 multiplies by 3 done as full `mod` multiplies, and about 95 modular add/subs**, each a conditional branch.
  - With random field data those branches mispredict about half the time. About 95 × 0.5 × 15–20 cycles ≈ 700–950 cycles ≈ 350–450 ns at 2.1 GHz, which matches the measured ~450 ns. Removing the branches recovered about 300 ns.
- **Candidate formulations (operation counts, F_p8):**

  | Form | General products | Reductions | Notes |
  |---|---|---|---|
  | (i) Karatsuba tower, reduce everything | 27 | 27 + 13 (by-3) | 95 add/sub |
  | (ii) Karatsuba tower, 3x as `(+ (+ x x) x)` and branchless add/sub | 27 | 27 | — |
  | (iii) Schoolbook X^8 - 3, lazy | 64 imul | 16 `mod` (2 per coefficient) | 7 times 3·b_j via adds; 8 adds — **measured ~53–63 ns with a destination buffer** |
  | (iv) Plonky3-style Montgomery schoolbook | 64 | 8 Montgomery | — |
  | (v) Lazy Karatsuba: 27 unreduced products combined in u64/signed words, then 8 final reductions | 27 | 8 | Sums of up to 8 coordinates times sums of up to 8 coordinates reach about 64p^2 ≈ 2^68, overflowing 64 bits; needs partial reductions of the pre-sums (cheap adds) or 128-bit words. More complex; probably a modest gain over (iii) on a CPU with 1 multiply/cycle throughput. |

- **F_p4 (X^4 - 3).** Schoolbook is 16 products with 4 coefficients of at most 4 products, so every coefficient fits one u64 lazy sum with **one** reduction each (4 reductions), plus 3 by-3 adds. Karatsuba gives 9 products with 9 + 4 reductions. Schoolbook-lazy should win clearly in SBCL.
- **W = 3 in the lazy schoolbook.** Fold 3 into b (3b_j mod p by two adds, or one unreduced `(* 3 b)` < 2^33, which keeps a·3b < 2^64 only if that coefficient's 4-sum is split). The alternative is to multiply the wrapped partial sum by 3 before reduction: 3·(4-sum) can reach 12p^2 > 2^64, so that needs splitting. Precomputing 3b_j (as Plonky3 does: `w_b = mul_w(packed_b)`) is simplest.
- **Basis change.** The tower basis and the X^8 - 3 power basis differ only by the bit-reversal permutation of the 8 coordinates (verified). Sable could keep its external coordinate order and permute indices at code-generation time, at zero runtime cost. Alternatively it could switch storage order, which affects compatibility with zk/field.py ordering.
- **Recommendation order (estimated from the measurements above):**
  1. Write into caller-provided destination vectors, or at least inline a constant-size `make-array` (about 2–3× on the lazy version).
  2. Branchless add/sub (about 3× on the existing Karatsuba code).
  3. Switch to direct lazy schoolbook for X^4 - 3 / X^8 - 3 (another 2–3× over branchless Karatsuba with allocation).
  4. Consider Montgomery form only together with SIMD.

### Gaps
- I did not find (and did not fetch) a primary paper giving optimal multiplication counts for 8-term binomial multiplication in this specific regime. Classic Karatsuba-like formula papers (e.g. P. L. Montgomery's five-, six- and seven-term formulas, IEEE TC 2005) were not verified in this session.
- The 27 / 64 counts above are elementary arithmetic, not sourced claims.
- The combined effect of branchless add/sub plus lazy schoolbook plus a destination buffer was not measured as one variant.

## Q4. Vectorized / SIMD arithmetic for small primes (AVX2, AVX-512, IFMA, NEON) and how it maps to SBCL

### Takeaway
- **State of the art for 31-bit fields is Plonky3's packed Monty-31 and M31 code.** It uses 32-bit lanes, `vpmuludq` even/odd splitting with Montgomery reduction (or a shift-fold for M31), and keeps outputs in a signed range to skip corrections. It reaches roughly 2–2.5 multiplies per cycle on AVX2/AVX-512.
- **AVX-512 IFMA (52-bit) suits larger (about 49-bit) primes better than 31-bit ones** (MamaBearZKP).
- **SBCL's `sb-simd` contrib exposes `vpmuludq`** (AVX2) and, since SBCL 2.6.9, some AVX-512, so the Plonky3 AVX2 sequence is expressible in Lisp. The local SBCL 2.2.9 lacks AVX-512 in sb-simd. No IFMA (`vpmadd52*`) wrapper was found.

### Cited Findings
- **Plonky3 AVX2/AVX-512/NEON costs and instruction sequences:** see Q1 for code paths and per-op cycle comments. The AVX2 `mul` takes `lhs`/`rhs` as 8×u32 lanes; `vmovshdup` moves odd lanes to even, since `vpmuludq` "reads the low 32 bits of each 64-bit lane". — [avx2/packing.rs L425-450, L688](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/x86_64_avx2/packing.rs)
- **Plonky3 sets `DELAYED_MIXED_DOT_PRODUCT = !cfg!(target_arch = "aarch64")`.** The comment: "On NEON the vectorized `ext_base_mul` and `ext_add` beat one delayed reduction per coordinate". So the best strategy for delayed reduction differs by ISA. — [monty-31/src/extension.rs L25-26](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/extension.rs)
- **Poseidon2 timings (from the HackMD write-up).**

  | Field | AVX2 width 16 | AVX2 width 24 | AVX-512 width 16 | AVX-512 width 24 |
  |---|---|---|---|---|
  | M31 | 0.71 µs | 1.3 µs | 1 µs | 1.7 µs |
  | KoalaBear | 0.78 µs | 1.3 µs | 1.1 µs | 1.7 µs |
  | BabyBear | 1 µs | 1.7 µs | 1.3 µs | 2.3 µs |

  These are as published; the AVX-512 numbers being slower than AVX2 is likely a different machine or setting, and the source does not explain it. — [HackMD Small Fields in Plonky3](https://hackmd.io/@Syxton/small_fields_in_plonky3)
- **MamaBearZKP (Zhang, Guo, Lu, Zhang, Cheng; ACM CCS 2026; IACR ePrint 2026/1698).**
  - Proposes a 49-bit prime 2^49 - 2^34 + 1 and an AVX-512 IFMA vectorization framework.
  - States that common ZKP fields "do not align with the AVX-512IFMA datapath".
  - Reports up to 42× (ZeroCheck) and up to 21× end-to-end single-thread over a Goldilocks baseline, and up to 18× over Plonky3's AVX-512 BabyBear backend.
  - Source: [eprint.iacr.org/2026/1698](https://eprint.iacr.org/2026/1698)
- **SBCL `sb-simd`.**
  - `two-arg-u64.4-mul` maps to `#:vpmuludq` (exported as `u64.4-mul`), and `s64.4-mul` to `vpmuldq`. — [sbcl contrib/sb-simd/code/instruction-sets/avx2.lisp L203, L356, L436](https://github.com/sbcl/sbcl/blob/master/contrib/sb-simd/code/instruction-sets/avx2.lisp)
  - Instruction-set files exist for avx512f/bw/dq/fp16, avx10, and neon.
  - SBCL NEWS for 2.6.9: "addition of some AVX512 support to SB-SIMD". — [sbcl NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)
  - Locally, `(require :sb-simd)` loads in SBCL 2.2.9 and exports `SB-SIMD-AVX2:U64.4-MUL` (verified).
  - `grep` for `madd52` in sb-simd found nothing, so no IFMA intrinsics.
- **Prior AVX2 Montgomery technique for small moduli.** The even/odd `vpmuludq` + signed Montgomery approach is standard in lattice cryptography. Seiler's AVX2 NTT paper (IACR ePrint 2018/039) is the usual reference. I did not fetch it in this session, so treat the attribution as unverified.

### Inferences
- **What SIMD would buy Sable.** If Sable's hot loops are bulk (sum-check folds, MLE evaluation, `eq-table`, `fold-table` over many field elements), a vectorized Monty-31 kernel via sb-simd AVX2 could approach Plonky3's about 2 multiplies per cycle, versus about 1 multiply per 5 ns (about 0.1/cycle at 2.1 GHz) for the scalar `mod`.
  - That is potentially more than 10× on the multiply-bound kernels.
  - It requires Montgomery form (or a separate lane-wise Barrett with 32-bit operations), 32-bit lane packing in `(unsigned-byte 32)` simple arrays (already Sable's storage type), and careful avoidance of boxing.
- **Single extension multiplies are a poor SIMD target in SBCL.** For one F_p8 multiply, Plonky3's AVX2 trick (b in one vector, 8 rotated rows, 8 broadcast `a_i`) is possible, but shuffle and broadcast overheads in sb-simd are unknown. Vectorizing across many independent extension operations (structure-of-arrays) is the more reliable route.

### Gaps
- No measured sb-simd field multiply in this session.
- I found no published paper specifically benchmarking BabyBear/KoalaBear on AVX-512 IFMA scalar-vs-SIMD in isolation; MamaBearZKP reports only prover-level numbers versus Plonky3.
- I found no GPU paper with clearly transferable CPU ideas within the time budget.

## Q5. Published KoalaBear vs BabyBear vs M31 comparisons; why KoalaBear

### Takeaway
- **Why KoalaBear:** p - 1 = 127·2^24 is coprime to 3, so x → x^3 is a permutation. This allows degree-3 S-boxes in Poseidon2 (BabyBear needs x^7, M31 x^5). The cost is two-adicity 24 instead of BabyBear's 27.
- **Raw add/mul speed:** identical for KoalaBear and BabyBear (same Monty-31 code). M31 has faster multiplies.
- **Benchmarks:** published comparisons are vendor or engineering write-ups, not controlled studies. A 2026 survey (ePrint 2026/1371) explicitly identifies "the lack of a controlled cross-field comparison" as a gap.

### Cited Findings
- **The HackMD write-up on Plonky3 small fields:**
  - BabyBear's smallest S-box exponent is d = 7. KoalaBear 127·2^24 + 1 "allows d = 3" at two-adicity 24.
  - Verifying x^5 or x^7 with constraint degree 3 needs an intermediate x^3, making them "twice as expensive to verify". The arithmetic-hash trace is "roughly 50% smaller" with KoalaBear.
  - KoalaBear is "not the unique solution". It also names 2^31 - 2^30 + 2^27 + 2^24 + 1 and 2^29 - 2^26 + 1.
  - End-to-end (i9 Raptor Lake, AVX2), 2^19 Poseidon2 permutations with a Keccak Merkle hash: KoalaBear 480 ms, M31 1.8 s, BabyBear 1.1 s (trace widths 1320 vs 2408 vs 2392).
  - 1365 Keccak permutations: KoalaBear 570 ms, M31 960 ms, BabyBear 570 ms.
  - Source: [hackmd.io/@Syxton/small_fields_in_plonky3](https://hackmd.io/@Syxton/small_fields_in_plonky3)
- **Plonky3 code confirms the cube-root map.** KoalaBear's `exp_root_d` uses `exp_1420470955`, with the comment "3 * 1420470955 = 2*(2^31 - 2^24) + 1 = 1 mod (p - 1)". — [koala-bear/src/koala_bear.rs L1, L63-64](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/koala-bear/src/koala_bear.rs); [field/src/exponentiation.rs L37-42](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/exponentiation.rs)
- **Extension two-adicity.** KoalaBear's extension data gives `EXT_TWO_ADICITY` 26 (degree 4) and 27 (degree 8). BabyBear's gives 29 (degree 4) and 30 (degree 8). — [koala_bear.rs L93-112](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/koala-bear/src/koala_bear.rs); [baby_bear.rs L66-100](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/baby-bear/src/baby_bear.rs)
- **Circle STARKs** (Haböck, Levit, Papini; IACR ePrint 2024/278). "Preliminary benchmarks indicate a speed-up by a factor of 1.4 compared to a traditional STARK using the Babybear prime" for M31. This is a whole-STARK number. — [eprint.iacr.org/2024/278](https://eprint.iacr.org/2024/278)
- **The M31 circle-group code foundation** is "Reed-Solomon Codes over the Circle Group" (Haböck, Lubarov, Nabaglo; IACR ePrint 2023/824). Its abstract has no benchmark numbers. — [eprint.iacr.org/2023/824](https://eprint.iacr.org/2023/824)
- **Survey.**
  - Skatharoudis, "The Small-Field Turn in Succinct Proofs: A Systematization of Finite-Field Choice in Modern SNARKs and STARKs" (IACR ePrint 2026/1371, July 2026). It systematizes fields by word fit, two-adicity, reduction cost and S-box structure, and states that a controlled cross-field comparison is the main empirical gap. — [eprint.iacr.org/2026/1371](https://eprint.iacr.org/2026/1371)
  - A search-result summary said it reports M31's vendor-claimed ~1.3× speedup over BabyBear measuring only ~1.07×. The abstract page I fetched did not show this figure, so it is **unconfirmed**.
- **Adoption claims.** A Ceno GitHub issue (secondary) claims KoalaBear multiplication is "measurably faster" than BabyBear and that SP1 adopted it, but gives no numbers. Treat as unverified. — [scroll-tech/ceno issue 1305 (mirror)](https://gh.r-mj.com/scroll-tech/ceno/issues/1305)

### Inferences
- **For Sable,** KoalaBear vs BabyBear makes no difference to base-field speed with either Barrett or Montgomery. The W = 3 non-residue for degree 4/8 is cheaper than BabyBear's W = 11: 3x is 2 adds, while 11x needs about 4–5 adds/shifts or a multiply.
- **M31** would give about 2× faster scalar multiplies (measured 2.8 vs 5.2 ns in SBCL). But it needs circle-FFT machinery and a different extension (CM31/QM31), so it is not a drop-in replacement.

### Gaps
- I found no primary Polygon/Succinct engineering blog post introducing KoalaBear with benchmarks. The HackMD note and Plonky3 code are the best primary-ish sources found.
- SP1 Hypercube's default field was not confirmed from a primary source.

## Q6. Inversion: exponentiation chains for p - 2, gcd-based inversion, extension inversion via norms, batch inversion

### Takeaway
- **Base field:** Plonky3 does not use Fermat. It uses a constant-time-style binary-GCD routine (Pornin's optimized binary GCD, IACR ePrint 2020/972) returning 2^60·a^-1, then fixes the power of 2 with one Montgomery reduction and a shift-multiply.
- **Extensions:** it inverts via norms down the 2-2-2 tower, so one base inversion per extension inversion. Sable already does this.
- **Bulk inversion:** use Montgomery's batch-inversion trick, which Plonky3 implements in chunked four-lane form.

### Cited Findings
- **Plonky3 base-field inverse.**
  - `MontyField31::try_inverse` calls `gcd_inversion_prime_field_32::<31>(value, P)` ("maps val mod P -> 2^{60}val mod P"), adds `P<<30` to make it positive, applies `monty_reduce` ("We could do a % operation here, but monty reduction is faster"), then multiplies by 2^36.
  - The output is `|v| <= 2^{2*FIELD_BITS-2}`.
  - The helper cites "https://eprint.iacr.org/2020/972.pdf". (I did not fetch that paper in this session; it is Pornin's binary-GCD inversion paper per the code reference.)
  - M31 does the same, then calls `div_2exp_u64(60)`.
  - Sources: [monty_31.rs L481-521](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/monty-31/src/monty_31.rs); [util/src/lib.rs L778-837](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/util/src/lib.rs); [mersenne_31.rs L422-433](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/mersenne-31/src/mersenne_31.rs)
- **Plonky3 octic inverse.**
  - Splits a = evens + odds·X, computes norm = evens^2 - odds^2·X^2 in F[X]/(X^4 - w) ("2 multiplications in the quartic extension field"), inverts the norm with `quartic_inv`, then does 2 quartic multiplies.
  - `quartic_inv` does the same down to the quadratic level, and `quadratic_inv` uses one base inversion of a0^2 - w·a1^2.
  - Source: [binomial_extension.rs L825-860, L1190-~1240](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
  - For other degrees there is a Frobenius-based `pseudo_inv`, computing a^(q + … + q^(D-1)) via Frobenius maps. — [binomial_extension.rs L111-135](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/extension/binomial_extension.rs)
- **Plonky3 batch inversion.** `batch_multiplicative_inverse` uses "chunked, four-lane Montgomery inversion", with 1024-element chunks per Rayon task. — [field/src/batch_inverse.rs L29-55](https://github.com/Plonky3/Plonky3/blob/0d803737b681774186cd283b1fd2d8e493655c82/field/src/batch_inverse.rs)
  - Montgomery's trick originates in P. L. Montgomery, "Speeding the Pollard and elliptic curve methods of factorization", Math. Comp. 48 (1987). I did not re-fetch it in this session.
- **Current Sable code.** `finv` uses `fpow a (p-2)` by generic square-and-multiply in a `loop`. `inv*` recursively applies the norm method, so one `finv` per extension inverse. — `/home/user/Sable/lisp/src/field.lisp` (`fpow`, `finv`, `inv*`)

### Inferences
- **Addition chains for p - 2 (derived here, not sourced):**
  - **KoalaBear:** p - 2 = 0x7EFFFFFF = (2^6 - 1)·2^25 + (2^24 - 1). Build t2 = x^3, t4 = x^(2^4-1), t6, t12 = t6^(2^6)·t6, t24 = t12^(2^12)·t12, then result = (t6^2)^(2^24)·t24 (t6^2 is the zero bit 24). That is about 48 squarings and 6 multiplies, versus generic binary (30 squarings + 29 multiplies ≈ 59 operations). With about 5 ns per operation, either is about 0.3 µs. The gain is small, and the generic `loop` has extra overhead from the untyped exponent.
  - **BabyBear:** p - 2 = 0x77FFFFFF = (2^3 - 1)·2^28 + (2^27 - 1).
  - **M31:** p - 2 = 2^31 - 3 = (2^29 - 1)·2^2 + 1.
- **A binary-GCD inverse** (Pornin-style, 31-bit) typically beats Fermat (about 60 dependent multiplies, about 300 ns at the measured 5 ns latency) on scalar CPUs. Plonky3 chose it for that reason. It is a straightforward port to SBCL fixnum/`(signed-byte 64)` code.
- **For bulk inversions** (e.g. Lagrange denominators, batch evaluation), batch inversion costs about 3 multiplies per element plus one inversion. It dominates any single-inversion optimization; implement it over extension elements too.

### Gaps
- I found no published optimal addition chain for KoalaBear's p - 2 (the chain above is my derivation).
- No measured SBCL timing of `finv` vs a GCD inverse in this session.
