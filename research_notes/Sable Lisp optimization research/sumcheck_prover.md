# Prover-side sumcheck optimizations for 31-bit fields with extension-field challenges (KoalaBear, F_p4/F_p8)

Scope: techniques to speed up a CPU prover for `sum_x eq(r,x) * F(t0(x), t1(x), ...)`, with multilinear base-field tables, LSB-first binding, round messages at X = 0, 2, ..., d, and tables folded each round (lo + r(hi - lo)).

Where the papers live: every primary paper below is on IACR ePrint. I found no arXiv versions and confirmed no arXiv IDs, so none are given. eprint.iacr.org served abstract pages to WebFetch but blocked direct PDF downloads (HTTP 403 or a challenge page). The full-text detail for 2025/1117 comes from the authors' mirror (a16z-hosted PDF) linked below. The full-text detail for 2024/1046 comes from Thaler's Georgetown copy.

Paper index (IDs confirmed from ePrint abstract pages unless noted):

| # | Paper | Authors | Year | ID / URL |
|---|---|---|---|---|
| P1 | Speeding Up Sum-Check Proving | S. Bagad, Q. Dao, Y. Domb, J. Thaler | 2025 | ePrint 2025/1117, https://eprint.iacr.org/2025/1117 (full-text mirror: https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf) |
| P2 | Speeding Up Sum-Check Proving (Extended Version) | Q. Dao, Z. DeStefano, S. Bagad, Y. Domb, J. Thaler | 2026 (Mar) | ePrint 2026/587, https://eprint.iacr.org/2026/587 (only the abstract was obtained; PDF blocked) |
| P3 | The Sum-Check Protocol over Fields of Small Characteristic | ePrint version lists S. Bagad, Y. Domb, J. Thaler; the Georgetown copy lists J. Thaler only | 2024 | ePrint 2024/1046, https://eprint.iacr.org/2024/1046 ; https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf (I did not fetch the ePrint abstract page; the authors come from the search-result PDF title) |
| P4 | More Optimizations to Sum-Check Proving | Q. Dao, J. Thaler | 2024 | ePrint 2024/1210, https://eprint.iacr.org/2024/1210 |
| P5 | Some Improvements for the PIOP for ZeroCheck | A. Gruen (Polygon Zero) | 2024 | ePrint 2024/108, https://eprint.iacr.org/2024/108 |
| P6 | A Time-Space Tradeoff for the Sumcheck Prover ("Blendy") | A. Chiesa, E. Fedele, G. Fenzi, A. Zitek-Estrada | 2024 | ePrint 2024/524, https://eprint.iacr.org/2024/524 |
| P7 | Time-Space Trade-Offs for Sumcheck (TCC 2025) | A. Baweja, A. Chiesa, E. Fedele, G. Fenzi, P. Mishra, T. Mopuri, A. Zitek-Estrada | 2025 | ePrint 2025/1473, https://eprint.iacr.org/2025/1473 |
| P8 | The Sum-Check Protocol over the Monomial Basis, and Other Optimizations | Q. Dao, A. Biswas, L. Eagen, A. Milson, S. Papini, J. Thaler | 2026 (Apr) | ePrint 2026/762, https://eprint.iacr.org/2026/762 |

---

## Q1. Bagad–Dao–Domb–Thaler "Speeding Up Sum-Check Proving" and predecessors: small-value optimization, delayed extension-field multiplication, Toom-Cook style evaluation

### Takeaway
P1 merges P3 and P4. It replaces the large-field (ll) multiplications of the first ℓ0 rounds with base-field (ss) or base-by-extension (sl) multiplications. To do this it precomputes accumulators of products of base-field table values on a (d+1)-ary grid via Lagrange (Toom-Cook) interpolation, then takes inner products with tensor-product Lagrange coefficients of the challenges. With an eq factor, as in our statements, the accumulators need sl multiplications, so the gain is roughly the square root of the eq-free gain. P3's cost model for degree-4/8 extensions predicts about 1.3× (k=4) and 1.85× (k=8) for d=2. Our statements always carry an eq factor, so the benefit is real but moderate.

### Cited Findings
**Setting and cost model**
- Multiplications are split into three kinds: ss (small×small), sl (small×large) and ll (large×large), with "ss ≪ sl ≪ ll". The optimization cuts ll multiplications from O(N) to O(N/poly(κ)), where κ = ll/ss — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Case 2 of P1, "base field B = GF[p] inside degree-k extension F", is our setting. Extension×extension (ll) multiplication uses Karatsuba variants at O(k^1.585) base multiplications. In a tower basis, multiplying a subfield element by a larger-field element costs only 2^(j'-j) subfield multiplications, which makes base×extension cheap — [P1 mirror, §2.1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)

**Algorithms and their costs**
- **Algorithm 1 (standard linear-time, linear-space prover).** From round 2 on it uses only ll multiplications. With small inputs its dominant cost is d²·2^(ℓ−1) ll mults (Lemma 3.2) — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Algorithm 2 (quasilinear-time, square-root-space prover).** Recomputes the products "from scratch" each round from the original small values, using many sl multiplications — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf). P3 observes that Algorithm 2, although slower in the large-field setting, "is actually cheaper than the first when base field multiplications (and base-field-times-extension-field multiplications) are much cheaper than extension-field multiplications" — [P3, Georgetown copy](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)
- **Algorithm 3 (small-value, unoptimized).** Expands the product so the round-i message is an inner product of a challenge-dependent coefficient vector of length (d+1)^(i−1) with small-value accumulators. Its bottleneck is (d−1)·2^(d·ℓ0)·2^(ℓ−ℓ0) ss mults — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Algorithm 4 (Toom-Cook small-value prover).**
  - It views ∏_k p_k(Y1..Y_{i−1}, u, x′) as a polynomial F of individual degree d in the already-bound variables and Lagrange-interpolates it over a grid U_d (d+1 points per variable).
  - The message becomes s_i(u) = ⟨⊗_{j<i}(L_{U_d,v}(r_j))_v , A_i(·,u)⟩.
  - The accumulators A_i(v,u) = Σ_{x′} ∏_k p_k(v,u,x′) are computed once over all v ∈ U_d^{i−1} from small values.
  - Cost (Theorem 4.1): (d−1)((d+1)/2)^ℓ0·N ss mults, plus d·N sl mults, plus d²/2^(ℓ0+1)·N ll mults. It also saves a factor 2^ℓ0 in space.
  - [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Choosing ℓ0 and the resulting speedup.**
  - Optimal ℓ0 = log(κ·d²/(2(d−1)))/log(d+1). The speedup is about 2^ℓ0 ≈ κ^(1/log(d+1)): κ^0.63 for d=2 and κ^0.5 for d=3.
  - Example: κ≈30 (u64 vs BN254) with d=2 gives ℓ0≈3.72 and about a 4× speedup.
  - [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Switching back to Algorithm 1.** Round ℓ0+1 runs Algorithm 2 for one round, storing p_k(r_{1..ℓ0−1}, b, x′). Those arrays are then folded, and the remaining rounds use the linear-time algorithm — [P1 mirror, Fig. 7](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Algorithm 6 (combined eq-poly + small-value).**
  - The accumulators are A_i(v,u) = Σ_{xR} eq(wR,xR) Σ_{xL} eq(wL,xL) ∏_k p_k(v,u,xL,xR).
  - Because eq is in the large field, these need sl rather than ss multiplications, and the additions happen in the large field.
  - Cost (Theorem 6.1): ((d+1)/2)^ℓ0 + d)·N sl mults plus d²/2^(ℓ0+1)·N ll mults. The relevant κ is now ll/sl, about the square root of ll/ss, "making our asymptotic speedups a square-root factor worse."
  - [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- **Worked example (Spartan, d=2, ℓ0=3).** Algorithm 5 (eq-split only) costs 3N sl + 4.5N ll. Algorithm 6 costs 4.875N sl + 1.1875N ll — [P1 mirror, App. B](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)

**Measured results**
- Jolt (BN254, M4 Pro):
  - About 3× over original Jolt and 1.5–2× over Gruen-only for SHA-2 chains of 8–256 iterations.
  - About 20× at 1024 iterations, where the baselines become memory-bound and then run out of memory.
  - [P1 mirror, §8](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Standalone implementation (binary tower fields, GF(2^32) inside GF(2^128), described as not fully optimized):
  - Algorithm 6 (ℓ0=3) is 20–40% faster than Algorithm 5 for n=16..28, with degree 2 gaining more than degree 3.
  - Algorithm 4 (ℓ0=4) beats Algorithm 1 by "roughly the same amount."
  - [P1 mirror, §8](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Also from P1:
  - Optimized small×large multiplication (single Barrett round) is 2.5–3× faster than ll for u64×BN254, and gave a 5–10% end-to-end Jolt speedup.
  - Evaluating at "infinity" (the leading coefficient) lets the prover drop lower-degree terms in that evaluation.
  - [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- P3, the predecessor, for d=2 with Karatsuba costs (be = k·bb, ee = k^1.585·bb):

  | Extension degree k | Optimal switchover round | Speedup over Algorithm 1 |
  |---|---|---|
  | 4 | 3 | 1.31× |
  | 8 | 4 | 1.85× |
  | 16 | 4 | 2.83× |

  The optimal switchover is roughly round 0.8·log k as k grows, and "most the savings come from the first few rounds." When base multiplications are close to free, the speedup is "multiple orders of magnitude"; under Karatsuba relative costs it is "close to five" at most — [P3, Table 1](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)
- P2 (2026 extended version) adds "a new algorithm that substantially reduces the number of field multiplications" for products of many multilinear polynomials. It reports more than an order of magnitude runtime speedup and memory reduction on Spartan, and 1.7–2.2× on a high-degree sumcheck in Shout (Jolt) — [ePrint 2026/587 abstract](https://eprint.iacr.org/2026/587)
- An implementation for 31-bit fields exists. LambdaClass (28 Nov 2025) implemented P1's Algorithm 6 in whir-p3 (Plonky3 ecosystem: BabyBear, KoalaBear, degree-4 binomial extensions), upstream PR #322 merged. It uses ℓ0=3 (`NUM_SVO_ROUNDS`) for a single multilinear (d=1) times eq, and sends 2 field elements per round. The post gives no timings — [LambdaClass blog](https://blog.lambdaclass.com/speeding-up-sumcheck-an-in-depth-walkthrough-of-our-implementation/)

### Inferences
**Estimated gains for KoalaBear**
- Base×ext costs about k base mults. Ext×ext costs about 9 base mults (k=4, Karatsuba) or 16 (schoolbook), and about 27 or 64 for k=8, before reduction costs.
- So for F_p4, κ = ll/ss ≈ 9–16 and ll/sl ≈ 2–4.
- Plugging into P1's formula for d=2:
  - Eq-free case (κ≈9–16): ℓ0 ≈ 2.6–3.2, so ll work drops about 6–9×.
  - With eq (Algorithm 6, ratio ll/sl ≈ 2–4): ℓ0 is only about 1–2, consistent with P3's 1.31× (k=4) and 1.85× (k=8).
- Expect the gain to be larger for F_p8 than for F_p4.

**How it would change our prover loop**
- Replace the first ℓ0 rounds (ℓ0 ≈ 2–3) with:
  - (a) A single pass over the base-field tables. For every suffix x′ and every grid point (v,u) ∈ U_d^{ℓ0}, evaluate each table's multilinear restriction at v ∈ {0,1,...,d−1,∞} in the first ℓ0 coordinates (base-field adds and subtracts only), multiply the d values in the base field (ss), weight by eq_L/eq_R (sl), and add into the accumulators A_i(v,u).
  - (b) For rounds i ≤ ℓ0, form each message as a (d+1)^(i−1)-term inner product of tensored Lagrange weights of r_1..r_{i−1} with the accumulators. This is negligible work.
  - (c) At round ℓ0+1, perform the first materialized fold directly from base values with tensored eq-like weights (sl), not ℓ0 successive extension folds.
- This also removes ℓ0 rounds of extension-field folding of all tables. In our current loop the first fold makes every table an extension table, so this is a large memory-traffic saving in Lisp, where extension elements are probably boxed or stored as multi-word structs.
- Lowest-risk first step, before full SVO: delay the first fold (Algorithm 2 for round 2, i.e. P3's insight). Compute round 2 from base tables with sl mults and fold two variables at once. This keeps everything in the base field one round longer.

### Gaps
- I could not obtain P2's (2026/587) full text, so the exact "products of many multilinears" algorithm and its multiplication counts are unconfirmed.
- No paper reports SVO benchmarks on 31-bit prime fields with degree-4/8 extensions. P1 benchmarks BN254 and binary towers; LambdaClass/whir-p3 publishes no numbers.

---

## Q2. Gruen's eq trick (ePrint 2024/108) and the Dao–Thaler eq split, and what remains when g_j already excludes the round variable's eq factor

### Takeaway
Gruen factors eq(w,(r_<i, X, x′)) = eq(w_<i, r_<i)·eq(w_i, X)·eq(w_>i, x′). The prover then computes only t_i(X), which has degree d (not d+1), and recovers t_i(1) from the running claim. That saves one evaluation per round and removes eq from the per-point product. If our g_j already excludes the round variable's eq factor, we are already doing Gruen's variant. The remaining gains are:
- Dao–Thaler's split of eq(w_>i, ·) into two √N tables, which removes the 2^n eq table and saves about N ll multiplications.
- Using the point at infinity instead of X=d.

### Cited Findings
**Gruen's factoring (P5)**
- Gruen's abstract targets zerocheck over a small field. It reduces the number of extension-field constraint evaluations and "shifts some computation from the extension field back into the base field."
- Prover cost is roughly 2^n(1 + C_G/(2^k C_F))(d−1)C_F, where k trades prover against verifier cost (one interpolation of degree 2^k·d plus n of degree d).
- [ePrint 2024/108](https://eprint.iacr.org/2024/108)
- P1's recap of the trick:
  - s_i(X) = l_i(X)·t_i(X), with l_i(X) = eq(w_<i, r_<i)·eq(w_i, X) and t_i(X) = Σ_{x′} eq(w_>i, x′)·∏_k p_k(r_<i, X, x′).
  - The prover computes t_i at d points and gets t_i(1) = l_i(1)^(−1)·(C_i − l_i(0)·t_i(0)).
  - [P1 mirror, §5.1](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Two ways to send it. Gruen's writeup sends t_i directly, which changes the protocol and adds verifier work. The Setty–Thaler variant computes t_i, derives s_i, and sends s_i, so the verifier is unchanged — [P1 mirror, footnote 18](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)

**Dao–Thaler eq split (P4, Algorithm 5 in P1)**
- During the first ℓ/2 rounds, write t_i(u) = Σ_{xR} eq(wR,xR) Σ_{xL} eq(wL,xL)·∏_k p_k(r_<i, u, xL, xR).
- Precompute only the √N-sized tables eq(wR,·) and eq(w_[<i],·). With memoization this costs 2^(ℓ/2) ll mults in total, instead of a 2^ℓ eq table.
- Put the xL part, the low-order contiguous bits, in the inner sum for memory locality.
- [P1 mirror, §5.2](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Algorithm 5's dominant cost is d(d+1)/2·N ll mults, about N fewer than Algorithm 1 (Lemma 5.1) — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- P4's abstract:
  - The split eliminates about 2^(n+1) field multiplications against a standard linear-time prover, or about 2^(n−1) on top of Gruen.
  - That is "about a 25% (respectively 10%) end-to-end prover speedup in common use cases."
  - [ePrint 2024/1210](https://eprint.iacr.org/2024/1210)

**Evaluation at infinity**
- s(∞) is defined as the leading coefficient. For a product of linear factors it ignores lower-degree terms — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- P8 takes this further. It sums over {0,∞}^n, so binding becomes p(r,x′) = p(0,x′) + r·p(∞,x′), with no subtraction. That saves d(2^n−1) subtractions for a product of d multilinears and gives about 10% end-to-end on BN254 and a 128-bit field. It has not been measured on 31-bit fields — [ePrint 2026/762](https://eprint.iacr.org/2026/762); [Plonky3 issue #1890](https://github.com/Plonky3/Plonky3/issues/1890)

### Inferences
- **What we already do.** Our message format (g_j excludes the eq factor of the round variable; sent at X = 0, 2, ..., d) appears to be Gruen's original "send t_i" variant, with t(1) implied by the claim. The degree reduction and the skipped X=1 evaluation are already realized.
- **What remains, in order of value for us:**
  1. **Replace X=d with X=∞.** Each factor's value at ∞ is just its slope (hi − lo), so the X=∞ evaluation is a plain product of the d slopes. That saves the repeated additions used to reach X=d, and the lower-degree terms drop out. Changing from 0,2,..,d to 0,2,..,d−1,∞ is a verifier-side interpolation change only.
  2. **Never materialize or fold the full eq(r,x) table.** Keep eq_L (2^(n/2)) and eq_R (2^(n/2)) tables and accumulate Σ_{xL} eq_L·∏(...) in an inner loop, then multiply by eq_R once per xR. This removes the per-round eq fold of a 2^(n−i) extension table, about N ll mults overall, and its memory. For d=2 the eq work is a meaningful fraction of the total (P4 reports about 25% end-to-end vs. no-Gruen and about 10% on top of Gruen).
  3. **Inner sum in base field.** Inner sums over xL with an extension-valued eq_L weight are sl products accumulated in the extension. These are a natural place for delayed reduction (Q4).

### Gaps
- I could not fetch Gruen's PDF (ePrint PDF access was blocked), so the exact section numbering and his own operation counts for the eq trick are not verified here. The description above comes from P1's recap.

---

## Q3. Univariate skip and its relevance to 31-bit fields

### Takeaway
Univariate skip (Gruen, ePrint 2024/108) replaces the first k rounds with one round over a univariate domain of size 2^k (degree about d·2^k). All of that round's work stays in the base field, and it applies to small-characteristic and extension settings like ours. It changes the protocol and the PCS opening statement. The one direct KoalaBear measurement (leanVM) showed about 11–13% end-to-end, and the team rejected it as not worth the complexity.

### Cited Findings
- Per P1's related-work section:
  - The univariate skip applies sum-check to a g with degree about d·2^i in the first variable, summed over a set of size d·2^i, "In this setting, all of the prover's work in round 1 can occur over the base field."
  - Downsides: it applies only when values lie in a small subfield; proofs are larger; and SNARKs would need commitments that support the high-degree first variable.
  - [P1 mirror, §1.2](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Binius's implementation docs:
  - When witness values embed in a small field, "we naturally want to refrain from folding for skip_rounds."
  - The smallest field with cardinality at least the round polynomial's degree should be used for the domain.
  - [Binius docs](https://binius-ulvetannaoss-16d81a0876eb016acc581aa502e37b77aaa54acc5fb.gitlab.io/binius_core/protocols/sumcheck/prove/univariate/fn.zerocheck_univariate_evals.html)
- leanVM PR #251 (KoalaBear with a quintic extension, June 2026). Setup:
  - K=4 rounds were replaced by one univariate round over {0..15}, with a combined polynomial of degree ≤150.
  - The extension-field round tail shrank 16×.
  - A Newton forward-difference kernel replaced Lagrange dot products: 125.9 ms vs. 212.4 ms.
  - [leanVM PR #251](https://github.com/leanEthereum/leanVM/pull/251)
- leanVM PR #251 results and outcome:
  - Native prover −11.8%; recursion −6.3%; proof size unchanged.
  - WHIR openings became tensor-weight statements (Lagrange_16(r0) ⊗ eq).
  - The PR was closed without merging over complexity. An earlier skip had already been removed because its roughly 10% gain was not worth it.
  - [leanVM PR #251](https://github.com/leanEthereum/leanVM/pull/251)

### Inferences
- **What it would change for us.** Our tables are KoalaBear values and the challenges are in F_p4/F_p8, so univariate skip would apply. It would remove k rounds of extension-field folding of all tables. That is the same goal as SVO (Q1), but SVO keeps the protocol and opening statement unchanged.
- **Recommendation.** Since this is a reference prover that presumably must match a fixed protocol/verifier, prefer SVO plus delayed first fold. Consider univariate skip only if the protocol can be changed and the PCS can take a Lagrange ⊗ eq opening weight.

### Gaps
- Plonky3's own implementation, if one exists, was not located. The search found no Plonky3 source or docs describing univariate skip.

---

## Q4. Delayed / lazy modular reduction in inner-product-heavy sumcheck loops

### Takeaway
For 31-bit primes, a base×base product is below 2^62, so about 4 products can be summed in an unsigned 64-bit word before one reduction. This is the standard Plonky3 approach for 31-bit fields, and Plonky3's WHIR sumcheck uses delayed-reduction kernels. The measured effect on sumcheck is mixed: micro-benchmark wins sometimes do not move end-to-end sumcheck time. No peer-reviewed paper quantifies it for sumcheck.

### Cited Findings
- Plonky3 issue #252 on optimized delayed-reduction loops: for Mersenne-31/BabyBear, "Since our inputs are 31-bit, the products are 62-bit. We can add four of these together without overflowing a 64-bit integer" — [Plonky3 #252](https://github.com/Plonky3/Plonky3/issues/252)
- Plonky3's WHIR sumcheck on 31-bit Monty fields (BabyBear, KoalaBear) uses delayed-reduction kernels. The projective sumcheck PR reuses a "K-tiled delayed-reduction scaffold" — [Plonky3 #1890](https://github.com/Plonky3/Plonky3/issues/1890); [Plonky3 PR #1900](https://github.com/Plonky3/Plonky3/pull/1900)
- A change fusing x86 dot products of length 5–8 into one reduction reported 8–27% micro-benchmark gains, but "the sumcheck benchmark does not move measurably." This comes from a mirror of a Plonky3 PR, so it is a secondary source — [mirror PR #545](https://github.com/Tkelley325/Plonky3/pull/545)
- Overflow bounds matter. A proof-based review found Sum::sum breaks the 31-bit invariant past 2^31 terms, fixed upstream in Plonky3#2536 — [Plonky3-lean PR #7](https://github.com/Plonky3/Plonky3-lean/pull/7)
- P1 reports a related gain for small×large multiplication: a single Barrett reduction instead of full Montgomery. It is 2.5–3× faster than ll and gave 5–10% end-to-end in Jolt — [P1 mirror, §2.1/§7, App. B.3](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)

### Inferences
**Where to apply it in our loop**
- (a) Inside each extension multiplication. For F_p4 = F_p[x]/(x^4 − W), each output coefficient is a sum of 4 base products, some scaled by the small constant W. Accumulate unreduced and reduce once per coefficient: 4 reductions per extension mult instead of 16. Check the W-scaled terms against the u64 bound; split them or use a carry word if needed.
- (b) Inside the Gruen/eq inner sums and SVO accumulators. Σ_x eq_L(x)·prod(x) is a sum of base×ext (sl) products, so keep k independent u64 (or 2-word) accumulators per output coefficient.
- (c) In the first round, where everything is base field. Sums of products of 2 base values can be accumulated 4 at a time in u64. A double-word accumulator (hi:lo) allows 2^64 terms with one final reduction.

**SBCL notes** (not from a source)
- Fixnums on x86-64 are 62-bit signed. (p−1)^2 ≈ 2^61.97 fits a fixnum, but sums of two do not.
- Use declared (unsigned-byte 64) locals with `(ldb (byte 64 0) ...)` / `logand` modular arithmetic in fully typed, `(safety 0)` code so SBCL keeps them unboxed.
- Store tables in specialized `(simple-array (unsigned-byte 32) (*))` arrays, with extension elements as k-strided entries, to avoid boxing.

### Gaps
- I found no paper quantifying delayed reduction's end-to-end effect on a sumcheck prover over 31-bit fields. The evidence is GitHub issues/PRs, and one of them says sumcheck time did not move measurably.

---

## Q5. Evaluating a product of d linear polynomials at d+1 points cheaply

### Takeaway
Each factor restricted to the round variable is linear, p_k(X) = lo + X·(hi − lo). The standard approach evaluates every factor at the needed points by repeated addition of the slope (no multiplications), then multiplies pointwise, about (d−1) mults per point. P1 adds two improvements: evaluating at ∞ (the product of slopes) and Toom-Cook interpolation across rounds (Q1). P2 (2026) claims a new algorithm for products of many multilinears that substantially cuts multiplications, but I could not access its details.

### Cited Findings
- The round-i message needs s_i(u) = Σ_x ∏_k p_k(r_1..r_{i−1}, u, x) for u in a set of size d+1, e.g. {0,...,d} — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Evaluation "at infinity" means the leading coefficient. It is usable as one of the d+1 points and lets lower-degree terms be omitted — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- Toom-Cook / Lagrange interpolation over a grid U_d reduces the small-value overhead from 2^(d·i) to (d+1)^i products per round. This is about 5× fewer ss mults for d=2, ℓ0=5 — [P1 mirror, §4.2](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)
- P2 adds "a new algorithm that substantially reduces the number of field multiplications" for products of many multilinear polynomials. It reports 1.7–2.2× on Jolt's high-degree Shout sumcheck — [ePrint 2026/587](https://eprint.iacr.org/2026/587)
- In the leanVM univariate-skip PR, a Newton forward-difference kernel (extrapolating evaluations by finite differences) replaced Lagrange dot products: 125.9 ms vs. 212.4 ms for that kernel — [leanVM PR #251](https://github.com/leanEthereum/leanVM/pull/251)

### Inferences
- **Concrete per-x′ kernel for our loop**, for each pair (lo_k, hi_k):
  - slope_k = hi_k − lo_k.
  - Values at X=2..d−1 come by repeated addition: v_k(2) = hi_k + slope_k, and so on.
  - Value at ∞ = slope_k.
  - Then multiply across k: (d−1) mults per point for d points (0, 2, …, d−1, ∞). That is d(d−1) mults per x′ before the eq weight.
- **Product-tree alternative** (my arithmetic, not from a paper): for d ≥ 4, multiply the linear factors in coefficient form with Karatsuba (linear×linear = 3 mults, quadratic×quadratic = 6). For d=4 that is 3+3+6 = 12 vs. 4·3 = 12 pointwise, so no gain at d=4. Gains appear only at larger d. For our likely small d (2–4), the pointwise kernel plus ∞ is the right choice.
- **Bigger lever:** in round 1 every lo/hi is a base-field value, so this whole kernel is ss and should use base-field code paths, not generic extension arithmetic.

### Gaps
- The exact P2 algorithm and its counts for products of many multilinears are not obtained, because the ePrint PDF was blocked.

---

## Q6. Streaming / memory-efficient sumcheck (Blendy and successors), and its relevance to a CPU prover

### Takeaway
Blendy (2024/524) and its TCC 2025 successor (2025/1473) give O(N^(1/k)) space with O(kN), or O(N(log log N + k)), time. They matter only when tables exceed RAM or cache. P1 shows that the memory-bound regime is where the largest wall-clock wins (about 20×) appear, and that SVO itself enables streaming.

### Cited Findings
- Blendy proposes a family of prover algorithms for multilinear sumcheck with time-space tradeoffs, with the CTY11 log-space and linear-time algorithms as special cases. The abstract page gives no figures — [ePrint 2024/524](https://eprint.iacr.org/2024/524)
- Time-Space Trade-Offs for Sumcheck (TCC 2025):
  - Single multilinear: O(kN) time, O(N^(1/k)) space, optimal for non-adaptive provers.
  - Products: O(N(log log N + k)) time.
  - Measured: up to 120× less memory than the linear-time prover, with under 2× time overhead.
  - [ePrint 2025/1473](https://eprint.iacr.org/2025/1473)
- P1 positions Blendy as rediscovering a special case of sparse-dense sumcheck (a single multilinear only). SVO lets a streaming prover stream for O(log ℓ) rounds instead of O(ℓ). For 2^30 cycles under 2 GB that means "stream the input 3 times instead of 10 or more" — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf)

### Inferences
- For a reference Lisp prover at moderate n, full streaming is low priority.
- The cheap part to take is "skip the materialized extension tables for the first rounds" (SVO / delayed first fold). It halves to eighths the peak extension-table memory: after ℓ0 rounds the tables are 2^(n−ℓ0) extension elements instead of 2^(n−1).
- This matters doubly in SBCL, where GC pressure from large arrays of extension elements can dominate.

### Gaps
- No CPU benchmark of Blendy-style provers on 31-bit fields was found.

---

## Q7. Other 2024–2026 work on sumcheck for small fields

### Takeaway
The main 2025–2026 additions beyond P1, P3 and P4 are:
- Projective / monomial-basis sumcheck (P8, 2026/762): removes subtractions in folding.
- The extended BDDT paper (P2, 2026/587): high-degree product algorithm.
- Engineering in the Plonky3/whir-p3/leanVM ecosystem, which is KoalaBear-based: SVO Algorithm 6 merged; univariate skip tried and rejected.

### Cited Findings
- P8 benefits:
  - No subtractions in binding.
  - Cheaper eq/less-than evaluation.
  - Alignment with WHIR's basis.
  - About 10% end-to-end prover speedup on BN254 and a 128-bit pseudo-Mersenne field.
  - Montgomery-friendly challenge subsets give 1.92× field mult and 1.82× binding, but only for about 256-bit fields.
  - [ePrint 2026/762](https://eprint.iacr.org/2026/762)
- Plonky3 tried P8 on its WHIR sumcheck. Only PR #1900 (prover-side subtraction-free kernel) landed, and the issue was closed in Sep 2026. Benefit on 31-bit Monty fields was "still an open question" — [Plonky3 #1890](https://github.com/Plonky3/Plonky3/issues/1890); [Plonky3 PR #1900](https://github.com/Plonky3/Plonky3/pull/1900)
- P1 cites more small-characteristic work: Liu–Zhang (binary towers, Boolean-valued products), asymptotically slower than P1 for d ∈ {2,3} with no implementation; and Wei et al. (parallel repetition + NeutronNova sumfold). P3 "strongly recommend[s] not to combine parallel repetition with the Fiat-Shamir transformation" — [P1 mirror](https://cdn.prod.website-files.com/63970f25aa7b42e284492d52/6851e118b388a7d5dbf83fda_2025-1117.pdf); [P3](https://people.cs.georgetown.edu/jthaler/small-sumcheck.pdf)
- LambdaClass has a three-part series on the subject: Part I "optimizing-sumcheck", Part II "how-factoring-equality-polynomials-optimizes-sumcheck", Part III the Algorithm 6 walkthrough — [LambdaClass blog](https://blog.lambdaclass.com/speeding-up-sumcheck-an-in-depth-walkthrough-of-our-implementation/); [Part II](https://blog.lambdaclass.com/how-factoring-equality-polynomials-optimizes-sumcheck/)

### Inferences — suggested implementation order for the SBCL prover

| Rank | Change | Protocol change? | Expected effect |
|---|---|---|---|
| 1 | Specialized base×ext (sl) multiply for the first fold and the eq weights (k base mults, not a generic ext×ext) | No | Large constant factor in round 1–2 |
| 2 | Typed, unboxed u64 delayed-reduction accumulation in ext mult and inner sums | No | Constant factor; verify with benchmarks (Plonky3 evidence mixed) |
| 3 | Split-eq tables (√N each), no full eq table, no eq folding | No | About N ll mults saved; about 10% on top of Gruen per P4; less memory |
| 4 | Evaluate at ∞ instead of X=d | Interpolation points only | Small; removes adds and makes the top coefficient a pure slope product |
| 5 | Delayed first fold / SVO Algorithm 6 with ℓ0 = 2–3 | No | About 1.3× (k=4), 1.85× (k=8) by P3's cost model for d=2; plus memory savings |
| 6 | Univariate skip, projective sumcheck | Yes | About 6–13% in KoalaBear/BN254 reports; not worth it for a reference prover tied to a fixed verifier |

### Gaps
- I did not open the HackMD notes (tcoratger "Optimizing the Sum-Check Prover: Small Values and Equality Polynomials"; levs57; benediamond) or the Ingonyama standalone repo (github.com/ingonyama-zk/smallfield-super-sumcheck, branch sb/eq-optimized, cited in P1).
- No Succinct or Polyhedra primary papers on small-field sumcheck turned up in these searches.
- I found no arXiv versions of any of the cited ePrint papers.
