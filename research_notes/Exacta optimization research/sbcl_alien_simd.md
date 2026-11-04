# SBCL-specific means of making numeric code fast: sb-alien FFI, sb-simd, custom VOPs, allocation control

Scope: SBCL 2.2+ on x86-64. The target workload is a Common Lisp port of a proof system: 31-bit prime field (p = 2^31 - 2^24 + 1 = 2130706433, "KoalaBear") on `(simple-array (unsigned-byte 32) (*))`, pure-Lisp BLAKE3, extension-field multiply/add on length-4/8 vectors, and fold loops `lo + r*(hi-lo)` over large arrays.

Note on "LOCAL MEASUREMENT" items: I ran microbenchmarks in this sandbox, because I found no published SBCL FFI call-cost numbers. Setup: SBCL 2.2.9.debian (/usr/bin/sbcl); gcc with `-O3 -march=native -shared -fPIC`; a VM reporting "Intel(R) Xeon(R) Processor @ 2.10GHz" with 4 vCPUs and avx2, avx512f and bmi2 flags. Every timing is wall clock over 3 to 2000 repetitions, `(optimize speed (safety 0))`. The VM is noisy: reruns of the same in-cache kernel varied by up to about 1.6x (2.2 vs 3.7 ns/element). Treat the numbers as order-of-magnitude only. Scripts are in /tmp/claude-0/-home-user-Sable/a0bd13ad-bed9-598f-9b64-b24dcfef3202/scratchpad/bench/ (b.lisp, d.lisp, f.c) and are not part of the repo.

## Q1. sb-alien: per-call overhead, pinning, and when batching is needed

### Takeaway
An inlined `define-alien-routine` call costs a few nanoseconds: about 2.4 ns for a trivial C function, measured locally on SBCL 2.2.9. Wrapping each call in `with-pinned-objects` adds under 1 ns. `without-gcing` per call is about 8x more expensive (about 19 ns), so don't use it. For this workload the FFI pays off only when one call covers a whole array AND the C code does something SBCL cannot, such as hand-written AVX2/AVX-512 Montgomery. A plain scalar C fold ran at the same speed as well-typed Lisp. On length-4/8 vectors the FFI was 2-3x slower than inline Lisp.

### Cited Findings
- SBCL manual, section 9 "Foreign Function Interface": "Any efficiency-critical foreign interface function should be inline expanded, which can be done by preceding the define-alien-routine call with: (declaim (inline lisp-name)). In addition to avoiding the Lisp call overhead, this allows pointers, word-integers and floats to be passed using unboxed representations, avoiding consing." — [SBCL 2.6.10 User Manual](http://www.sbcl.org/manual/)
- SBCL manual, same chapter: the GC moves objects. `sb-sys:with-pinned-objects` "arranges for some set of objects to be pinned in memory for the dynamic extent of its body forms. On ports which use the generational GC (most…) this affects exactly the specified objects. On other ports it is implemented by turning off GC for the duration". The third option is `SB-EXT:WITHOUT-GCING`. — [SBCL Manual](http://www.sbcl.org/manual/)
- Why pinning is cheap on x86-64: in SBCL source `src/compiler/generic/pinned-objects.lisp`, on `#+conservative-stack-scan` platforms (x86-64) `with-pinned-objects` just keeps the objects live on the stack. It uses `dx-let` plus `touch-object`, a VOP "with an empty body … we don't have to pay any extra beyond that". On non-conservative platforms it binds the special `*pinned-objects*` to a stack-allocated list. — [sbcl/sbcl pinned-objects.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/generic/pinned-objects.lisp)
- Historical overhead bug: a foreign call bound `*alien-stack*` unnecessarily ("a lot of overhead for a foreign function call"). It was later fixed. — [Launchpad bug 645121](https://bugs.launchpad.net/bugs/645121)
- I found no published nanosecond figure for `alien-funcall` or CFFI call overhead in SBCL. The only third-party call-cost figure I found is for Lisp DEFUN calls: 2.38 ns plus 0.57 ns per argument on SBCL 1.3.4 x86-64. — [CLiki "the cost of nothing"](https://www.cliki.net/the-cost-of-nothing) (seen only as a search snippet; not fetched)
- On Windows, foreign callbacks (C→Lisp) on foreign threads cost about 10,000x a plain C call in one synthetic benchmark. This is a callback number, not a Lisp→C number. — [sbcl-devel patch thread](https://groups.google.com/g/sbcl-devel/c/3dSxGvF5oGo/)
- CFFI's SBCL backend is built on `sb-alien`/`alien-funcall`, so it is not a different low-level path. — [cffi-sbcl.lisp](https://github.com/cffi/cffi/blob/master/src/cffi-sbcl.lisp)
- LOCAL MEASUREMENT (SBCL 2.2.9; inline `define-alien-routine`; unsigned-32 arguments):
  - `uint32_t noop32(uint32_t)`: 2.40 ns/call.
  - The same call wrapped per call in `(with-pinned-objects (v) …)`: 3.27 ns/call.
  - The same call wrapped per call in `(without-gcing …)`: 18.9 ns/call.
  - C `mulmod(a,b) = (uint64)a*b % p` called per element through the FFI: 7.0 ns per op, against 5.3 ns for inline Lisp `(mod (* a b) p)`.
- LOCAL MEASUREMENT, fold `out[i] = lo[i] + r*(hi[i]-lo[i]) mod p`:
  - 8192 elements (in cache): Lisp with tight type declarations ran at 2.2–3.7 ns/element; one FFI call into gcc -O3 -march=native C ran at 2.4–3.4 ns/element. These are statistically equal.
  - 2^22 elements (out of cache): both ran at 4.58 ns/element, consistent with a memory-bandwidth limit.
- LOCAL MEASUREMENT, small-array break-even (per call, with pinning of three vectors and `vector-sap`):

  | Array length | Lisp | FFI |
  |---|---|---|
  | 4 | 10 ns | 32 ns |
  | 8 | 17 ns | 38 ns |
  | 64 | 151 ns | 233 ns |
  | 1024 | 2949 ns | 2089 ns |

  The fixed cost of a multi-argument FFI call with three `vector-sap`s and pinning is about 20-30 ns. That is much more than the 2.4 ns noop. The measurement does not separate how much of that is argument marshalling and how much is C-side loop setup.

### Inferences
- Never cross the FFI per field element or per extension-field multiply (length 4/8). Inline Lisp wins by 2-3x.
- Batch at the granularity of whole fold layers, whole Merkle-leaf hashing, or whole NTT passes. Even then, gains come only if the C side uses SIMD that SBCL won't emit itself, for example AVX2/AVX-512 Montgomery multiply over 8/16 lanes, or the official BLAKE3 C library with its AVX2/AVX-512 hash_many. A scalar C kernel buys nothing over well-typed SBCL code.
- Recommended pattern: `(declaim (inline f))` before `define-alien-routine`; pass `system-area-pointer` arguments obtained with `sb-sys:vector-sap` on `(simple-array (unsigned-byte 32) (*))`; wrap the call in `sb-sys:with-pinned-objects` (on x86-64 the pin is effectively free). Avoid `without-gcing`: it blocks GC across threads and cost about 16 ns extra per entry locally.
- Large arrays (≥ about 2^20 elements × 3 arrays × 4 bytes) are memory-bound, so neither Lisp nor C SIMD can help much there beyond fusing passes.

### Gaps
- No published, peer-reviewed or blog measurement of SBCL `alien-funcall` cost was found. The numbers above come from a single noisy VM.
- Not measured: the cost of a call into the official BLAKE3 C library (`blake3_hasher_update`) against pure Lisp. This is likely the strongest FFI candidate.

## Q2. sb-simd: venue, instruction sets, versions, speedups, and 32x32→64 multiply (vpmuludq)

### Takeaway
sb-simd was presented in Marco Heisig and Harald Köstler, "Closing the Performance Gap Between Lisp and C", ELS'22 (March 21–22, 2022, Porto). It became an SBCL contrib in SBCL 2.2.6, so SBCL 2.2.9 ships it. It exposes SSE through AVX2 and FMA, and newer SBCL (2.6.x) adds AVX-512 pieces and ARM64. AVX2 `u64.4-mul` is `vpmuludq`, an exact 4-lane 32x32→64 unsigned multiply, which is the building block for vectorized Montgomery/Barrett arithmetic. I verified this locally on SBCL 2.2.9.

### Cited Findings
- **Paper.** Heisig, M. & Köstler, H., "Closing the Performance Gap Between Lisp and C", ELS'22, March 21–22, 2022, Porto, Portugal (the venue string appears in the page headers of the PDF). PDF on Zenodo record 6335627, "Published March 21, 2022 | Version v1". — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf); [Zenodo record](https://zenodo.org/record/6335627); [FAU CRIS entry](https://cris.fau.de/publications/275132889); [researchr bibtex](https://researchr.org/publication/HeisigK22/bibtex)
  - The DOI is not confirmed. Zenodo records normally have DOI 10.5281/zenodo.6335627, but I did not see it stated, so flag it as unverified.
  - I found no arXiv version.
- **Abstract.** Lisp code using sb-simd and Loopus reaches "up to 94% of the performance of highly optimized C code". The conclusion states the gap narrowed "to something between 6% and 33%, depending on the target hardware". — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Benchmark (Fig. 8).** Jacobi's method in GFlop/s, SBCL 2.2 + Loopus against GCC 9.2, problem fitting in L1. Each row is Lisp / C / ratio:

  | CPU | Lisp | C | Ratio |
  |---|---|---|---|
  | EPYC 7451 | 8.1 | 8.6 | 0.94 |
  | EPYC 7452 | 8.2 | 10.0 | 0.82 |
  | Xeon Gold 6148 | 10.8 | 13.9 | 0.78 |
  | Xeon Gold 6248 | 7.2 | 9.3 | 0.77 |
  | EPYC 7543 | 12.8 | 18.1 | 0.71 |
  | Xeon E5-2695 v3 | 6.9 | 9.8 | 0.70 |
  | Icelake 8360Y | 11.3 | 16.1 | 0.70 |
  | Icelake 8358 | 7.9 | 11.3 | 0.70 |
  | Broadwell E5-2697 v4 | 5.0 | 7.5 | 0.67 |

  Source: [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Remaining gap.** The paper attributes it to SBCL's VOP-based code generation: "it is very hard to combine loads with subsequent instructions". It also observes that SBCL kept the loop bound on the stack (`cmp [rsp+0x20], rax`), while GCC folded loads into `vaddpd`. — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Design.**
  - There is one package per instruction set, with inheritance by re-export.
  - Scalar types are f32, f64, sN and uN for N ∈ {8,16,32,64}. Pack types are X.Y, for example u32.8 or u64.4.
  - "Most calls to functions in sb-simd are compiled to a single machine instruction."
  - Packs have boxed and unboxed representations. Boxing (heap spill) "can usually be avoided via inlining, or by writing their values to specialized arrays … instead of passing them around as function arguments".
  - All code is generated from per-instruction-set metadata tables.
  - `instruction-set-case` gives run-time dispatch, for example AVX2 vs SSE2.
  - Source: [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Credits.** Paul Khuong wrote the initial SIMD-pack support in SBCL. Stas Boukarev added "the low-level machinery required for AVX, AVX2, and FMA instructions to SBCL". The work was funded by KONWIHR. — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf); [KONWIHR blog](https://blogs.fau.de/konwihr/?p=6298)
- **Contrib proposal.** Heisig proposed sb-simd as an SBCL contrib in late 2021. — [sbcl-devel "Potential contrib: sb-simd"](https://groups.google.com/g/sbcl-devel/c/d1A9AJHEHE4)
- **Version history (SBCL NEWS, master branch; from a local clone):**
  - "changes in sbcl-2.2.6 relative to sbcl-2.2.5: new contrib: sb-simd, to provide a convenient interface for SIMD programming on x86-64".
  - 2.2.9: build fix.
  - 2.3.3: tests can run without AVX2.
  - 2.4.2: several fixes, including AVX f64.4-reverse (lp#2012986), shuffles on AVX/SSE2 (lp#2012990), and bounds checking (lp#2012010).
  - 2.5.0: s16.8-maddubs fix.
  - 2.6.5: fix of some AVX2 shift instructions (lp#2152791).
  - 2.6.7: "the SB-SIMD contrib now supports ARM64".
  - 2.6.8: enhancements.
  - 2.6.9: "addition of some AVX512 support", u8.16-shuffle (SSE3), "address a number of issues".
  - 2.6.10: AVX2 four-way 64-bit permutation.
  - Source: [SBCL NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS); [lp#2012986](https://bugs.launchpad.net/bugs/2012986)
- **Instruction sets on master.** `contrib/sb-simd/code/instruction-sets/` contains sse, sse2, sse3, ssse3, sse4-1, sse4-2, avx, avx2, fma, avx512f, avx512bw, avx512dq, avx512fp16, avx10-1, avx10-2, arm64, neon and x86-64. — [sbcl contrib/sb-simd](https://github.com/sbcl/sbcl/tree/master/contrib/sb-simd)
- **Multiply instructions in the tables (master `avx2.lisp`):**
  - `(two-arg-u64.4-mul #:vpmuludq (u64.4) (u64.4 u64.4) …)`
  - `(two-arg-s64.4-mul #:vpmuldq …)`
  - `s32.8-mullo` → `vpmulld`
  - `s16.16-mulhi/mullo/mulhrs`
  - SSE4.1 has `s64.2-mul` → `pmuldq` (signed). SSE2 has no unsigned `pmuludq` wrapper in sb-simd, although the SBCL assembler defines `pmuludq` in `src/compiler/x86-64/insts.lisp`.
  - Source: [sb-simd avx2.lisp](https://github.com/sbcl/sbcl/blob/master/contrib/sb-simd/code/instruction-sets/avx2.lisp); [insts.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/x86-64/insts.lisp)
- **LOCAL VERIFICATION (SBCL 2.2.9).**
  - `(require :sb-simd)` works; the contrib fasl ships with the Debian 2.2.9 build.
  - `(u64.4-mul (u64.4 (+ (ash 1 32) 3)) (u64.4 #xffffffff))` returns 12884901885 = 3 × 0xFFFFFFFF in every lane. This confirms vpmuludq semantics: only the low 32 bits of each 64-bit lane are multiplied, giving a full 64-bit product.
  - The exported AVX2 MUL ops are S16.16-MULHI, S16.16-MULHRS, S16.16-MULLO, S16.8-MULLO, S32.4-MULLO, S32.8-MULLO and U64.4-MUL. There is no `u32.8-mul` and no unsigned 32-bit mul-high.
  - Available helpers include U32.8-AREF/ROW-MAJOR-AREF, U32.8-SHIFTL/SHIFTR, U64.4-SHIFTL/SHIFTR, U32.8-MIN, U32.8-IF, U32.8-PERMUTE, U32.8-UNPACKLO/HI, U64.4-PERMUTE and the non-temporal loads/stores. (local run)
- **Loopus.** Loopus is Heisig's portable loop-optimization and auto-vectorization library. On SBCL it uses sb-simd metadata. It rewrote the Jacobi loop into a 4-wide SIMD loop plus a remainder loop, replaced array accesses by pointer arithmetic, and hoisted index calculations. The paper calls it "still in its infancy". — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)

### Inferences
- A Plonky3-style AVX2 Montgomery multiply for a 31-bit field can be expressed in sb-simd on SBCL ≥ 2.2.6:
  1. `u64.4-mul` on the even lanes.
  2. `u64.4-shiftr` by 32 to bring the odd lanes down, then a second `u64.4-mul`.
  3. A `u64.4-mul` by μ for the Montgomery quotient, and another `u64.4-mul` by p.
  4. Shifts, subtract, `u32.8-min` or `u32.8-if` for the conditional correction, and an unpack or permute to recombine.

  This is inferred from the available op list; I did not find an existing sb-simd Montgomery implementation.
- For extension-field vectors of length 4/8 with 31-bit coefficients, a single u32.8 pack or two u64.4 packs holds a whole element. Keep the packs unboxed: write `(declaim (inline …))` helpers and load/store via `u32.8-aref` into specialized arrays, as the paper advises.
- Pin a minimum of SBCL ≥ 2.4.2 if relying on sb-simd shuffles/permutes, because of the shuffle fixes there. SBCL ≥ 2.6.5 fixes some AVX2 shifts (lp#2152791). Those are exactly the ops Montgomery needs. On 2.2.9, test shift results against scalar references.

### Gaps
- The exact set of AVX2 shift instructions that were broken before 2.6.5 (lp#2152791) was not checked.
- No benchmark of sb-simd on integer modular arithmetic was found. All published numbers are double-float stencils.

## Q3. Custom VOPs (define-vop): documentation, 64x64→128 multiply, libraries using them, stability

### Takeaway
You probably don't need to write a 64x64→128 VOP. SBCL already ships `sb-kernel:%multiply-high` (the high word via `mul` into rdx), `sb-kernel:%signed-multiply-high`, and `sb-bignum:%multiply`, which returns both hi and lo words as two values. All three exist in SBCL 2.2.9 and 2.0.0. Hand-written VOPs are documented mainly by Paul Khuong's 2014 blog post and by SBCL's own source. Ironclad shows production use, including a BLAKE2s G-function VOP. VOPs use internal, unexported APIs (`sb-c`, `sb-vm`, `sb-assem`) that change between releases.

### Cited Findings
- **Multiply VOPs in SBCL source** (`src/compiler/x86-64/arith.lisp`, master):
  - `(define-vop (bignum-mult) (:translate sb-bignum:%multiply) …)` emits `(move eax x) (inst mul y) (move hi edx) (move lo eax)` and returns `(hi lo)`.
  - `(define-vop (mulhi) (:translate %multiply-high) …)` keeps only rdx.
  - `mulhi/fx` is a fixnum variant.
  - An unnamed VOP translates `%signed-multiply-high`.
  - `defknown %multiply-high (word word) word` is in `src/compiler/fndb.lisp`. The symbol is exported from SB-KERNEL (`src/cold/exports.lisp`).
  - Sources: [arith.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/x86-64/arith.lisp); [fndb.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/fndb.lisp); [exports.lisp](https://github.com/sbcl/sbcl/blob/master/src/cold/exports.lisp)
- **Version check (via raw GitHub at tags).** `sbcl-2.2.9` `arith.lisp` has the `%multiply`, `%multiply-high` (×2) and `%signed-multiply-high` VOPs. `sbcl-2.0.0` has `%multiply` and `%multiply-high`. — [arith.lisp @ sbcl-2.2.9](https://github.com/sbcl/sbcl/blob/sbcl-2.2.9/src/compiler/x86-64/arith.lisp)
- **Division by constant.** SBCL's `srctran.lisp` uses `%multiply-high` to strength-reduce `truncate`/`mod` by constant divisors (multiply-by-reciprocal). — [srctran.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/srctran.lisp)
- LOCAL VERIFICATION (SBCL 2.2.9):
  - `(sb-kernel:%multiply-high #xFFFFFFFFFFFFFFFF #xFFFFFFFFFFFFFFFF)` returns `#xFFFFFFFFFFFFFFFE`.
  - `(sb-bignum:%multiply …)` on the same inputs returns `#xFFFFFFFFFFFFFFFE` and `1`, i.e. (hi, lo).
  - A disassembly of a well-typed `(mod (* r d) p)` with p = 2130706433 showed `MUL` + `IMUL RDX, RDX, 2130706433`. That is reciprocal multiplication with no DIV instruction.
- **`inline-vop`.** SBCL also has an internal macro `inline-vop` (`src/compiler/meta-vmdef.lisp`) for embedding an anonymous VOP with register/SC specs and assembly directly inside a function body. It is used, for example, in `src/code/arm64-simd.lisp`. It is present at tag sbcl-2.2.9 (`meta-vmdef.lisp` line 1731) but is internal and undocumented. — [meta-vmdef.lisp](https://github.com/sbcl/sbcl/blob/master/src/compiler/meta-vmdef.lisp)
- **Tutorial: Paul Khuong, "How to define new intrinsics in SBCL" (Aug 16, 2014).** Steps:
  1. `sb-c:defknown` with type and attributes (foldable, flushable, movable).
  2. `define-vop` with `:translate`, `:policy :fast-safe`, `:args`/`:results` storage classes (`unsigned-reg` for untagged and `any-reg` for tagged), and `:arg-types`/`:result-types`.
  3. A `:generator` cost; the compiler picks the cheapest applicable VOP.
  4. A wrapper `defun` for full calls.

  Example: a `popcnt` intrinsic. — [pvk.ca](https://pvk.ca/Blog/2014/08/16/how-to-define-new-intrinsics-in-sbcl/)
- **Gist example (Oct 2021) of a low-64 `modular*` VOP.** It allocates rax/rdx temporaries, uses `mul`, and discards rdx. A `:constant` variant uses `register-inline-constant`. — [privet-kitty gist](https://gist.github.com/privet-kitty/12d301145803faf175490e33b397fa4e)
- **Assembler syntax churn.** For example, "movzx (:dword :qword) dest source is no longer allowed on SBCL > 2.1.4.134" (search snippet from a cl-nibbles VOP file). — [cl-nibbles x86-64-vm.lisp (Debian)](https://sources.debian.org/src/cl-nibbles/20210520.gitdad2524-1/sbcl-opt/x86-64-vm.lisp)
- **Ironclad** (sharplispers/ironclad, last commit Oct 11 2024 in my clone) has `src/opt/sbcl/x86oid-vm.lisp` with VOPs:
  - fill-block-ub8 (LE/BE, 32/64) and expand-block
  - sha256-expand-block and update-sha1-block
  - salsa-core-fast and chacha-core-fast
  - pclmulqdq/AES-NI detection, AES-NI key schedule, encrypt and decrypt
  - **fast-blake2s-mixing**, the BLAKE2s G function as 32-bit `add/xor/ror 16,12,8,7` on four registers
  - gmac-mul-fast, xor128, mov128, swap32 and swap64

  They are enabled by the `:ironclad-assembly` feature, which `src/package.lisp` pushes by default. — [ironclad x86oid-vm.lisp](https://github.com/sharplispers/ironclad/blob/master/src/opt/sbcl/x86oid-vm.lisp); [ironclad README](https://github.com/sharplispers/ironclad)
- **sb-rotate-byte contrib.** It provides `rotate-byte` with x86-64 VOPs emitting `rol`/`ror` on `:dword` (32-bit) and on 64-bit registers (`contrib/sb-rotate-byte/x86-64-vm.lisp`). — [sb-rotate-byte](https://github.com/sbcl/sbcl/tree/master/contrib/sb-rotate-byte)

### Inferences
- **Field arithmetic.** For p < 2^31, products fit in 62 bits, so no 128-bit multiply is needed. Use `(mod (* a b) p)` with tight integer-range types: SBCL emits a multiply-high reciprocal sequence. Alternatively, write Montgomery with `(ldb (byte 32 0) …)` modular arithmetic. For 64-bit Goldilocks-style or 128-bit products, use `sb-kernel:%multiply-high` or `sb-bignum:%multiply` instead of writing a VOP. These are internal but have been stable since at least 2.0.0.
- **BLAKE3.** Its compression G function is the same as BLAKE2s (32-bit words, rotations 16/12/8/7). Ironclad's `fast-blake2s-mixing` VOP is directly reusable as a template. In pure Lisp, `(ldb (byte 32 0) (+ …))` plus `sb-rotate-byte:rotate-byte` (or ironclad's `ror32`) usually compiles to the same instructions, because SBCL recognizes 32-bit rotations in modular arithmetic. Check this with `disassemble` before writing a VOP.
- **Stability.** Keep any custom VOPs in one file guarded by `#+(and sbcl x86-64)`, with a portable fallback. Test it against every SBCL version you support, because `define-vop` options, SC names and `inst` syntax are internal.

### Gaps
- No ELS paper or SBCL manual chapter documents `define-vop`. There is no official VOP documentation beyond source comments and the Khuong blog.
- No measured speedup was found for ironclad's VOPs against its pure-Lisp fallback.

## Q4. Ironclad hashes and BLAKE3 in Lisp

### Takeaway
Ironclad has BLAKE2b (`blake2.lisp`) and BLAKE2s (`blake2s.lisp`), with an SBCL x86-64 VOP for the BLAKE2s mixing function. It has no BLAKE3 as of the Oct 2024 master. I found no Common Lisp BLAKE3 implementation, and no published speed comparison of ironclad against C for BLAKE2.

### Cited Findings
- Ironclad `src/digests/` contains:
  - adler32, blake2, blake2s, crc24, crc32
  - groestl, jh, kupyna
  - md2, md4, md5, ripemd-128/160
  - sha1, sha256, sha3, sha512
  - skein, sm3, streebog, tiger, tree-hash and whirlpool

  It has no blake3; `grep -i blake3` over the repo returned nothing (local clone). — [ironclad digests](https://github.com/sharplispers/ironclad/tree/master/src/digests)
- `blake2s.lisp` calls `fast-blake2s-mixing` (line 83) when the assembly feature is on. — [ironclad blake2s.lisp](https://github.com/sharplispers/ironclad/blob/master/src/digests/blake2s.lisp)
- Ironclad has `benchmark/benchmark.lisp`, but the repo publishes no results. — [ironclad](https://github.com/sharplispers/ironclad)
- A web search for a Common Lisp BLAKE3 found only Rust/C/packaging results. The official BLAKE3 repository lists Rust and C implementations with SIMD. — [BLAKE3 repo mirror](https://github.com/k0001/BLAKE3); [blake3.c](https://docs.rs/crate/blake3/1.8.0/source/c/blake3.c)

### Inferences
- BLAKE3 in pure Lisp can reach roughly ironclad-BLAKE2s-level scalar speed by porting ironclad's G VOP. It cannot match the official C/Rust code, whose speed comes from SIMD across multiple chunks/leaves (AVX2: 8 compressions in parallel; AVX-512: 16).
- Merkle-tree leaf hashing is many independent compressions, so it is the best FFI candidate in this codebase. Make one `alien-funcall` per batch of leaves into libblake3's `hash_many`-style API, or write an 8-way u32.8 G function in sb-simd (add, xor, shiftl/shiftr-or for rotations).

### Gaps
- No measured Lisp-vs-C BLAKE2/BLAKE3 throughput (cycles/byte) was found.

## Q5. Allocation reduction and numeric-performance techniques; related papers

### Takeaway
On SBCL, `dynamic-extent` on simple specialized `make-array`, inline structs and closures eliminates heap allocation. Locally, a length-8 ub32 temp array went from 48 bytes consed per iteration to 0, and from 12.4 to 4.7 ns per iteration. The biggest pitfall measured locally was imprecise integer-range types, which silently produce bignum arithmetic and generic `TRUNCATE` calls. Fixing them made the fold 3-7x faster. Specialized `(unsigned-byte 32)` arrays, `(ldb (byte 32 0) …)` modular arithmetic and inlining are the core tools.

### Cited Findings
- **SBCL manual §6.7 "Stack Allocation".** It covers dynamic-extent on:
  - &rest lists
  - the results of `cons`, `list`, `list*` and `vector`
  - simple `make-array` ("only if the resulting array is known to be both simple and one-dimensional, and it has a constant :element-type")
  - flet/labels and lambda closures
  - defstruct instances "when the structure constructor … has been declared inline"

  Raw-slot structs (word, double-float, etc.) can be stack-allocated "only on x86 and x86-64". The manual also warns: "Stack overflow checks are done except in 0 safety policies." — [SBCL Manual §6.7](http://www.sbcl.org/manual/#stack-allocation)
- SBCL NEWS dynamic-extent history:
  - 2.1.7: DX arrays of numeric/character types without an initial-element are no longer zeroed and "will reflect previous contents of the stack".
  - 2.3.8: DX works for conditionals even when not all branches are stack-allocatable.
  - 2.3.9: "stack allocation via DYNAMIC-EXTENT now applies to all values that a variable can take on (for example via SETQ)".
  - 2.5.2–2.5.7 and 2.6.9: DX bug fixes, including "don't improperly inherit dynamic extent nature of transformations of lists" (2.6.9) and an lp#2156347 fix for "dynamic-extent-heavy code".
  - Source: [SBCL NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)
- **SBCL manual §6.8 "Modular Arithmetic".** With `(logand <expr> <mask>)` or `ldb` of width 32/64 over `+ - logand logior logxor lognot` and positive `ash`, the compiler cuts intermediates to machine width "(but it is not done for variables and constants!)". — [SBCL Manual §6.8](http://www.sbcl.org/manual/#modular-arithmetic)
- LOCAL MEASUREMENT, dynamic extent (SBCL 2.2.9, 20M iterations of creating and filling a length-8 ub32 array):
  - with `(declare (dynamic-extent v))`: 0 bytes consed, 4.7 ns/iter
  - without it: 959,983,200 bytes consed (48 B/iter), 12.4 ns/iter
- LOCAL MEASUREMENT, type-derivation pitfall. In the fold `d = (if (>= h l) (- h l) (- (+ h p) l))` with h and l declared only `(unsigned-byte 32)`, SBCL derives d < about 2^33, so `(* r d)` can exceed 64 bits. The disassembly shows a signed `IMUL` with overflow check, bignum allocation via `ALLOC-TRAMP`, and a full call to `#<FDEFN TRUNCATE>`. Result: 15.6 ns/element, 0 bytes consed only when there is no overflow.

  Declaring the field range `(integer 0 2130706432)` with `sb-ext:truly-the` on l, h and d removes all calls and allocation: 42 instructions, an `IMUL`/`MUL`/`IMUL rdx, rdx, p` reciprocal sequence. The fold then ran at 2.2–4.6 ns/element, equal to gcc -O3.
- Heisig & Köstler (ELS'22) on boxing: SIMD packs are boxed when passed across non-inlined function boundaries; avoid this by inlining or by writing to specialized arrays. — [Zenodo PDF](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Older Lisp-vs-C work.** Didier Verna, "Beating C in Scientific Computing Applications" (ECOOP 2006 Lisp workshop) and "How to make Lisp go faster than C" (IMECS 2006) found typed, optimized Lisp matching C on image-processing loops without SIMD. — [LRDE publication page](https://www.lrde.epita.fr/wiki/Publications/verna.06.ecoop); cited in [Heisig & Köstler 2022](https://zenodo.org/records/6335627/files/2022-ELS-SIMD.pdf)
- **Petalisp.** Heisig's lazy, functional array-programming library; Loopus is its companion loop optimizer. I did not fetch the Petalisp repository or papers in this session.

### Inferences
- Concrete guidance for the codebase:
  1. Define `(deftype fe () '(integer 0 2130706432))` and declare every intermediate with it, using `truly-the` where invariants guarantee reduction. `(unsigned-byte 32)` is not tight enough to keep `r*(h-l)` in 62 bits.
  2. Build with `(optimize speed)` and read every compiler note. Notes about "*" / "TRUNCATE" "unable to … due to type uncertainty" or "doing … (cost N) to bignum" mean boxing.
  3. Spot-check hot functions with `disassemble` for `ALLOC-TRAMP`, `GENERIC-` and `FDEFN` calls.
  4. For the run-time-generated straight-line extension-field multiply, emit fixnum `let*` chains with `fe`/`(unsigned-byte 62)` declarations. Return results by writing into a caller-supplied `(simple-array (unsigned-byte 32) (8))` rather than returning a fresh vector or multiple values over 8 elements.
  5. Allocate per-call temporaries with `make-array … :element-type '(unsigned-byte 32)` under `dynamic-extent`, or preallocate per-thread scratch buffers. Remember that DX arrays are not zeroed (since 2.1.7) unless `:initial-element` is given.
  6. Prefer struct-of-arrays layouts: one `(unsigned-byte 32)` array per coefficient, or interleaved length-4/8 blocks in one flat ub32 array. Avoid vectors of boxed objects, because SBCL can only keep raw ub32 data unboxed inside specialized arrays (or raw struct slots).
  7. Avoid `(safety 0)` on code that does DX allocation of large arrays, because stack-overflow checks disappear at safety 0.
- The 2.3.9 SETQ-DX enhancement does not matter for fixed-size temporaries, so SBCL 2.2.9 is sufficient for the allocation techniques above.

### Gaps
- No 2023–2026 ELS paper benchmarking SBCL integer/cryptographic code against C was found by search. The ELS proceedings for those years were not exhaustively checked.
- Petalisp and Loopus are not characterized: their current maintenance status and applicability to modular-integer arithmetic were not checked.

## Q6. Is SBCL 2.2.9 sufficient, or are newer versions needed?

### Takeaway
SBCL 2.2.9 supports everything needed here: sb-simd as a bundled contrib (since 2.2.6), AVX2 `vpmuludq` via `u64.4-mul`, `%multiply-high`, `%multiply`, `inline-vop`/`define-vop`, `with-pinned-objects`/`vector-sap`, DX specialized arrays and inline-struct DX. Newer versions add sb-simd bug fixes (2.4.2 shuffles; 2.6.5 AVX2 shifts), AVX-512 sb-simd support (2.6.9), ARM64 sb-simd (2.6.7) and DX improvements (2.3.8/2.3.9). The current release is 2.6.10.

### Cited Findings
- sb-simd was added in 2.2.6, with a build fix in 2.2.9 ("make sb-simd build in compiler-only SBCL"). — [SBCL NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)
- LOCAL: the Debian SBCL 2.2.9 package ships `/usr/lib/sbcl/contrib/sb-simd.fasl`, along with sb-rotate-byte, sb-sprof, sb-gmp and sb-mpfr. `(require :sb-simd)` loads, and AVX2 functions execute on this AVX2/AVX-512 machine.
- The SBCL manual at sbcl.org is currently titled "SBCL 2.6.10 User Manual", and the top NEWS entry is "changes in sbcl-2.6.10 relative to sbcl-2.6.9". — [SBCL Manual](http://www.sbcl.org/manual/); [NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)
- The sb-simd fixes after 2.2.9 that touch ops relevant to Montgomery or BLAKE3 are listed in Q2: 2.4.2 for shuffles, f64.4-reverse and bounds checks; 2.6.5 for AVX2 shifts (lp#2152791); 2.6.9 for "a number of issues". — [SBCL NEWS](https://github.com/sbcl/sbcl/blob/master/NEWS)

### Inferences
- Develop on 2.2.9 if that is the deployment target. Keep a scalar reference implementation, and run randomized differential tests of every sb-simd kernel against it, because shift and shuffle bugs were fixed only in 2.4.2 and 2.6.5. If the version can be chosen, ≥ 2.4.2 is the minimum for sb-simd work, and ≥ 2.6.9 is preferable for AVX-512 and the accumulated fixes.

### Gaps
- I did not determine whether the specific AVX2 shift bug fixed in 2.6.5 affects `u64.4-shiftr`/`u32.8-shiftl` as used for Montgomery. The Launchpad bug lp#2152791 was not fetched.
