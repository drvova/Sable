"""Cost model for the corrected spec (corrected.tex).

Every projection in the spec is printed by this script. Units: H100-seconds
(one H100 SXM at the stated efficiencies). Run: python3 costmodel.py

Two prover profiles: BASE (the gadgets of the "Gadgets" section) and OPT
(the "Optimized profile" section: multi-column range tables, fused gadgets,
one relation per softmax entry, rate-1/2 code, small-value sumcheck and GKR).

Field: KoalaBear, p = 2^31 - 2^24 + 1.
"""
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- unit costs
# H100-seconds per unit of work, central values. Band is [0.5x, 2x].
# KoalaBear multiplies (Montgomery) are charged 10% above Mersenne-31.
MODES = {
    # Mode P: provable bounds. IOP challenges in F_{p^4} with one LogUp
    # instance behind LOGUP_GRIND bits of grinding; PCS batching, folding
    # and queries in F_{p^8}.
    "P": dict(c_commit=3.9e-10, c_lookup=1.5e-10, c_rel=5.5e-11),
    # Mode P8 (revision 3): every challenge in F_{p^8}, one LogUp instance.
    "P8": dict(c_commit=3.9e-10, c_lookup=4.1e-10, c_rel=1.4e-10),
    # Mode C: conjectured proximity gaps, everything in F_{p^4}.
    "C": dict(c_commit=3.5e-10, c_lookup=1.5e-10, c_rel=5.5e-11),
}
EXT_BYTES = {"P": 16, "P8": 32, "C": 16}
KAPPA = {"P": 1, "P8": 1, "C": 1}
# Per chunk, a false lookup survives one instance over F_{p^4} with
# probability (2 * distinct forms) / |F| ~ 2^-92.9; grinding before (beta, Z)
# brings it to 2^-103.9. Tight terms target 103 bits so that the main proof
# and up to 5 recursion layers sum to at most 2^-100.
LOGUP_GRIND = 11
TARGET_BITS = 103
C_MAC = 8e-15       # four integer forward passes on tensor cores, per MAC
# Weight opening per stored weight: each proof reduces its own claims to one
# point (C_WRED); fold, encode and open (C_WOPEN) can be shared by a batch.
C_WRED = 2e-11
C_WOPEN = 2e-11
TABLE_FIXED = 6 * 2**24  # table-side LogUp leaves and multiplicity commits
# KDA tables: tau_qd, tau_lambda, tau_sig, tau_mul8, tau_dec, tau_SiLU8.
TABLE_KDA = 2**24 + 2 * 2**23 + 2**21 + 2**20 + 2**17
# LogUp is sound only if no tuple can occur p times: lookups are split into
# chunks of at most 2^30, each with its own multiplicities and table side.
LOGUP_CHUNK = 2**30
MAX_TABLE = {False: 2**24, True: 2**25.6}   # largest table, BASE / tuple

# Optimization switches. BASE = none, OPT = all.
FLAGS = ("tuple", "fused", "score1", "rate_half", "small_value")
BASE = dict.fromkeys(FLAGS, False)
OPT = dict.fromkeys(FLAGS, True)
RATE_HALF_COMMIT = 0.6       # hash 2 symbols per element instead of 4
SMALL_VALUE_LOOKUP = 0.7     # base-field first GKR layers
SMALL_VALUE_REL = 0.5        # base-field first sumcheck rounds
TUPLE_TABLE_ROWS = 8         # multi-column tables are ~8x the R24 rows

# ------------------------------------------------------------ gadget costs
# (committed base-field elements, lookups, relation terms) per element.
G_BASE = {
    "req":   (2, 3, 2),  # dynamic Requant8 of a 24-bit row (spec: C=1)
    "reqs":  (3, 3, 3),  # static saturating Requant8 (spec: C=2)
    "norm":  (2, 3, 1),  # NormScale (spec: C=1)
    "sr":    (1, 4, 1),  # ShiftRound: 29-bit accumulator -> c
    "mul":   (3, 8, 3),  # MulRound: 24-bit x 17-bit, limb split, t bound
    "add":   (2, 1, 2),  # saturating residual add
    "lut":   (1, 1, 0),  # table lookup
    "score": (1, 1, 3),  # softmax entry: exp output P, max, PV/D product
    "relu":  (2, 5, 2),  # indexer ShiftRound + ReLU flag, per (query, key, head)
    "cmp":   (2, 2, 4),  # top-k: sel, tie flag, gated bound, multiset leaf
    "slot_h": (1, 1, 3),  # softmax entry per (query, slot, head)
    "slot_i": (1, 1, 1),  # slot index per (query, slot)
    "commit": (1, 0, 0),  # plain committed value (KDA state H)
    "samp":  (2, 5, 3),  # per (generated token, vocab id): logit + p
    "idx":   (15, 0, 15),  # selected index: 15 bits, each with b(1-b) = 0
    "bit":   (1, 0, 1),    # one Boolean address bit (expert or slot index)
}
# Multi-column range table T_R: one lookup per bound instead of two.
G_TUPLE = dict(G_BASE, req=(1, 2, 2), reqs=(2, 2, 3), norm=(1, 2, 1),
               sr=(1, 2, 1), mul=(3, 5, 3), relu=(2, 3, 2), cmp=(2, 2, 4),
               samp=(2, 3, 2))
# Fused gadgets (Section 11.2).
G_FUSED = dict(G_TUPLE,
               srq=(3, 2, 3),    # accumulator -> static saturating 8-bit
               rope8=(1, 2, 1),  # RoPE on 8-bit Q/K -> 9-bit
               srlut=(2, 2, 1),  # accumulator -> (k, c, tau(c)) in one row
               srreq=(1, 2, 2))  # accumulator -> dynamic 8-bit

G = G_BASE   # active table, set by proof_work


def acc(tot, g, n):
    c, l, r = G[g]
    tot[0] += c * n
    tot[1] += l * n
    tot[2] += r * n


@dataclass
class Model:
    name: str
    N_total: float
    N_active: float
    L: int
    d: int
    d_ff_active: int        # summed width of active FFN paths per token
    n_h: int
    n_kv: int
    d_h: int = 128
    V: int = 128256
    kind: str = "dense"     # dense | glm
    extra: dict = field(default_factory=dict)


PV_BLOCK = 4096   # keys per rounded PV partial sum (10-bit probabilities)

# KDA (Section 3.5): chunks of 128, sub-chunks of 8, head dimension 128.
KDA_C, KDA_SUB, KDA_D = 128, 8, 128
_nsub = KDA_C // KDA_SUB
KDA_PAIRS = (KDA_C + 1) / 2                      # (t, s) pairs per query
KDA_LUT_PER_HEAD = KDA_D * (
    2                                            # q~, k~ to sub-chunk bounds
    + (KDA_SUB + 1) / 2                          # same sub-chunk pair terms
    + sum(b + 1 for b in range(1, _nsub)) / _nsub  # q^(E'), q^(0) versions
    + 1                                          # k^end
) + (sum(b + 1 for b in range(1, _nsub)) + _nsub) * KDA_D / KDA_C  # decays G


def pv_blocks(S, P0=0):
    """Rounded PV partials per row, averaged over the causal rows P0..P0+S-1.

    Row m has ceil((m+1)/PV_BLOCK) blocks, plus the total when it has more
    than one.
    """
    tot, r, end = 0, P0, P0 + S
    while r < end:
        nb = r // PV_BLOCK + 1
        n = min(end, nb * PV_BLOCK) - r
        tot += n * (nb + (1 if nb > 1 else 0))
        r += n
    return tot / S


def dense_layer_per_token(m: Model, S: int, fused=False, P0=0):
    t = [0, 0, 0]
    d, dq, dkv, f = m.d, m.n_h * m.d_h, m.n_kv * m.d_h, m.d_ff_active
    acc(t, "req", d); acc(t, "norm", d)            # attention RMSNorm
    if fused:
        acc(t, "srq", dq + 2 * dkv)                # Q, K, V straight to 8-bit
        acc(t, "rope8", dq + dkv)                  # RoPE in the 8-bit domain
    else:
        acc(t, "sr", dq + 2 * dkv)                 # QKV outputs
        acc(t, "mul", dq + dkv)                    # RoPE
        acc(t, "reqs", dq + 2 * dkv)               # static Q, K, V
    acc(t, "sr", dq * pv_blocks(S, P0))            # PV block partials + total
    acc(t, "mul", dq)                              # division by row sum
    acc(t, "req", dq); acc(t, "sr", d)             # O projection
    acc(t, "add", d)                               # residual
    acc(t, "req", d); acc(t, "norm", d)            # MLP RMSNorm
    if fused:
        acc(t, "srlut", f)                         # gate -> SiLU in one row
        acc(t, "srreq", f)                         # up -> 8-bit
        acc(t, "req", f)                           # down input: SiLU(g) * up8
    else:
        acc(t, "sr", 2 * f)                        # gate, up
        acc(t, "lut", f)                           # SiLU
        acc(t, "mul", f)                           # SiLU(g) * up
        acc(t, "req", f)
    acc(t, "sr", d)                                # down projection
    acc(t, "add", d)
    if "experts" in m.extra:                       # (active, total) experts
        k, n = m.extra["experts"]
        acc(t, "bit", k * math.ceil(math.log2(n)))  # expert index bits
    return t


def glm_mixer_per_token(m: Model, layer: str, fused=False):
    t = [0, 0, 0]
    d = m.d
    w = 8192                                       # 64 heads x 128
    acc(t, "req", d); acc(t, "norm", d)
    if layer == "kda":                             # Section 3.5, per token
        acc(t, "sr", 4 * w)                        # q, k, v, output gate
        acc(t, "sr", 128); acc(t, "req", 128)      # rank-128 decay projection
        acc(t, "sr", w); acc(t, "lut", w)          # decay logits -> lambda
        acc(t, "reqs", 3 * w)                      # convolution inputs
        acc(t, "lut", 3 * w)                       # convolution -> SiLU
        acc(t, "sr", 2 * w)                        # L2 normalization of q, k
        H = 64
        acc(t, "lut", H * KDA_LUT_PER_HEAD)        # decayed q, k, pair terms
        acc(t, "req", H * (KDA_PAIRS + 128))       # scores, output row
        acc(t, "sr", H * 2 * 128)                  # output alignment
        acc(t, "commit", H * 128)                  # state H, per token
        acc(t, "commit", H * KDA_D / KDA_SUB)      # Lambda at sub-chunk ends
        acc(t, "sr", w)                            # per-head output L2 norm
        acc(t, "mul", H * 128); acc(t, "sr", H * 128); acc(t, "req", H * 128)
        acc(t, "lut", 2 * w)                       # sigmoid gate, gate product
    else:                                          # sparse (DSA-style) layer
        if fused:
            acc(t, "srq", w + 512 + 64)
            acc(t, "rope8", w // 2 + 64)
            acc(t, "srq", 4096 + 128 + 32)         # indexer q, k, head weights
            acc(t, "rope8", 32 * 64 + 64)          # indexer RoPE
        else:
            acc(t, "sr", w + 512 + 64)             # q, kv latent, rope key
            acc(t, "mul", w // 2 + 64)             # RoPE parts
            acc(t, "reqs", w + 512)
            acc(t, "sr", 4096 + 128 + 32)          # indexer q, k, head weights
            acc(t, "mul", 32 * 64 + 64)            # indexer RoPE
            acc(t, "reqs", 4096 + 128 + 32)
        acc(t, "sr", w); acc(t, "mul", w)          # PV numerators, division
        acc(t, "req", w)                           # O projection input
    acc(t, "sr", d)                                # O projection
    acc(t, "add", d)
    # MoE: router, then 9 active experts of width d_e
    acc(t, "req", d); acc(t, "norm", d)
    acc(t, "sr", 288); acc(t, "cmp", 288)
    acc(t, "bit", 8 * 9)                           # 8 routed indices of 288, 9 bits
    f = m.d_ff_active
    if fused:
        acc(t, "srlut", f); acc(t, "srreq", f); acc(t, "req", f)
    else:
        acc(t, "sr", 2 * f); acc(t, "lut", f); acc(t, "mul", f)
        acc(t, "req", f)
    acc(t, "sr", 9 * d)                            # 9 down outputs
    acc(t, "mul", 9 * d)                           # router-weighted combine
    acc(t, "add", d)
    return t


def units_for(mode, prof):
    u = dict(MODES[mode])
    if prof["rate_half"]:
        u["c_commit"] *= RATE_HALF_COMMIT
    if prof["small_value"]:
        u["c_lookup"] *= SMALL_VALUE_LOOKUP
        u["c_rel"] *= SMALL_VALUE_REL
    return u


def proof_work(m: Model, S: int, G_out: int, B: int = 1, mode="P",
               prof=BASE, relu_per_head=True, wopen_share=1.0,
               breakdown=False, window=None, global_every=1, P0=0):
    """H100-seconds for one proof of B sequences of length S (G_out generated).

    window/global_every model a hypothetical dense model in which only every
    global_every-th layer attends to the whole prefix and the rest use a
    sliding window.

    P0 > 0 is a session turn (Section "Session mode"): S new positions after
    P0 positions whose carried state earlier proofs committed.
    """
    assert P0 == 0 or global_every == 1
    global G
    G = G_FUSED if prof["fused"] else (G_TUPLE if prof["tuple"] else G_BASE)
    if prof["score1"]:
        G = dict(G, score=(1, 1, 2), slot_h=(1, 1, 2))
    u = units_for(mode, prof)
    fused = prof["fused"]
    tot = [0.0, 0.0, 0.0]
    macs = 0.0
    if m.kind == "dense":
        per_tok = dense_layer_per_token(m, S, fused, P0)
        for i in range(3):
            tot[i] += per_tok[i] * m.L * S * B
        n_glob = m.L / global_every
        ents = n_glob * m.n_h * (S * P0 + S * S / 2 + S * 64) * B   # causal tiles
        if global_every > 1:
            wl = min(window, S)
            ents += (m.L - n_glob) * m.n_h * (S * wl - wl * wl / 2 + 64 * S) * B
        acc(tot, "score", ents)
        acc(tot, "idx", m.L * m.n_h * S * B)            # row-maximum index
        macs += 2 * m.N_active * S * B + 2 * m.d_h * ents * 2
    else:
        n_kda, n_sp = m.extra["n_kda"], m.extra["n_sparse"]
        for layer, n in (("kda", n_kda), ("sparse", n_sp)):
            per_tok = glm_mixer_per_token(m, layer, fused)
            for i in range(3):
                tot[i] += per_tok[i] * n * S * B
        cand = (S * P0 + S * S / 2) / 4                 # causal pooled keys
        if relu_per_head:
            acc(tot, "relu", n_sp * 32 * cand * B)
        acc(tot, "cmp", n_sp * cand * B)
        slots = sum_min_slots(P0 + S) - sum_min_slots(P0)
        acc(tot, "slot_h", n_sp * 64 * slots * B)
        acc(tot, "slot_i", n_sp * slots * B)
        acc(tot, "bit", n_sp * slots * B
            * math.ceil(math.log2(max(2, (P0 + S) / 4))))
        acc(tot, "idx", n_sp * 65 * S * B)               # 64 maxima + top-k boundary
        macs += 2 * m.N_active * S * B + n_sp * 4096 * cand * B \
            + n_sp * 64 * slots * 1024 * B
    acc(tot, "samp", m.V * G_out * B)
    C, Lk, R = tot
    table_rows = (TABLE_FIXED + (TABLE_KDA if m.kind == "glm" else 0)) \
        * (TUPLE_TABLE_ROWS if prof["tuple"] else 1)
    # Upper bound: every chunk beyond the first pays the largest table side.
    table_rows += MAX_TABLE[prof["tuple"]] * max(0, math.ceil(Lk / LOGUP_CHUNK) - 1)
    units = n_units(C, Lk, mode)
    parts = dict(commit=C * u["c_commit"]
                 + units * (2**24 / COSET - 2**20) * 64 / NODE_HASH_BPS,
                 lookup=Lk * u["c_lookup"],
                 rel=R * u["c_rel"], forward=macs * C_MAC,
                 fixed=table_rows * (u["c_commit"] + u["c_lookup"])
                 + m.N_total * (C_WRED + C_WOPEN * wopen_share)
                 + carried_state(m, P0) * B * (u["c_commit"] + C_WRED + C_WOPEN))
    work = sum(parts.values())
    info = dict(committed=C, lookups=Lk, rel=R, chunks=math.ceil(Lk / LOGUP_CHUNK),
                units=units)
    if breakdown:
        info.update({k: v / work for k, v in parts.items()})
    G = G_BASE
    return work, info


UNIT_BYTES = 16e9          # resident working set of one work unit
TREES = {False: 4, True: 3}  # LogUp tables touched per unit, BASE / tuple
COSET = 4                  # symbols per column per leaf (2 folds before oracle)
DEPTH = 24 - 2             # unit subtree depth: codeword 2^24, cosets of 4
NODE_HASH_BPS = 120e9 * 0.6  # Merkle nodes beyond the 2^20 per unit in c_commit
GLOBAL_BYTES = 6 * 2**20   # fold oracles, table side, claims
BLAKE3_S = 1e-7            # CPU seconds per BLAKE3 compression (64 B)
EXT_ROUND_S = 5e-7         # CPU seconds per GKR round check
QUERY_GRIND = 30           # bits of grinding before query positions


def n_units(C, Lk, mode):
    return max(1, math.ceil((8 * C + EXT_BYTES[mode] * Lk) / UNIT_BYTES))


# Recursion: an outer STARK (same field, BLAKE3, Mode P) proves the inner
# verifier. Cells are trace area of the outer proof.
CELLS_BLAKE3 = 4096        # per compression (band 2k-12k)
CELLS_EXT = 16             # per extension-field multiply or fold step
C_CELL = 4e-10             # H100-s per outer trace cell (commit+constraints)
OUTER_ZK = 1.02            # salted leaves and random rows in every outer layer
SEGMENT_CELLS = 2**30      # outer proofs are split into segments of this size
SEG_VERIFY_COMP = 153 * 80  # compressions to verify one segment (q x path+leaf)
FINAL_PROOF_BYTES = 400 * 2**10   # last layer at rate 1/16, 44 queries
FINAL_VERIFY_S = 0.05


def queries(mode, prof):
    """Johnson-radius query count for TARGET_BITS with QUERY_GRIND bits of grinding."""
    rho = 0.5 if prof["rate_half"] else 0.25
    bits = -math.log2(math.sqrt(rho) + 0.01)
    return math.ceil((TARGET_BITS - QUERY_GRIND) / bits)


def unit_shape(m: Model, S: int, G_out: int, B: int, mode, prof, P0=0):
    _, info = proof_work(m, S, G_out, B=B, mode=mode, prof=prof, P0=P0)
    C, Lk = info["committed"], info["lookups"]
    units = info["units"]
    trees = TREES[prof["tuple"]] * KAPPA[mode]
    n = math.log2(max(2, Lk / units / TREES[prof["tuple"]]))
    rounds = n * (n + 1) / 2                     # GKR layer j has j rounds
    col = 2**23 if prof["rate_half"] else 2**22
    cols = max(1, math.ceil(C / units / col))   # columns per unit
    q = queries(mode, prof)
    cap = int(math.log2(q))                      # Merkle cap levels per unit
    return dict(units=units, trees=trees, rounds=rounds, q=q, cols=cols,
                path=DEPTH - cap, cap_nodes=2**cap)


def proof_size(m: Model, S: int, G_out: int, B: int = 1, mode="P", prof=BASE):
    """(work units, proof bytes, single-core verification seconds), uncompressed.

    One leaf per unit per query holds all of the unit's columns on a coset;
    each unit sends a Merkle cap once, and paths stop at the cap.
    """
    u = unit_shape(m, S, G_out, B, mode, prof)
    leaf = u["cols"] * COSET * 4
    per_uq = leaf + u["path"] * 32
    size = u["units"] * (u["trees"] * u["rounds"] * 4 * EXT_BYTES[mode]
                         + u["q"] * per_uq + u["cap_nodes"] * 32) + GLOBAL_BYTES
    comp = u["units"] * (u["q"] * (math.ceil(leaf / 64) + u["path"]) + u["cap_nodes"])
    verify = comp * BLAKE3_S + u["units"] * u["trees"] * u["rounds"] * EXT_ROUND_S + 1
    return u["units"], size, verify


def recursion_work(m: Model, S: int, G_out: int, B: int = 1, mode="P", prof=BASE,
                   P0=0, n_prev=0):
    """H100-seconds for the outer proof that verifies one inner proof.

    n_prev earlier state trees (session mode) each add one opening.
    """
    u = unit_shape(m, S, G_out, B, mode, prof, P0)
    leaf_blocks = math.ceil(u["cols"] * COSET * 4 / 64)
    comp = (u["units"] * (u["q"] * (leaf_blocks + u["path"]) + u["cap_nodes"])  # Merkle
            + n_prev * u["q"] * (DEPTH + 8)                     # state-tree openings
            + u["q"] * 2 * 24                                   # fold oracles
            + u["units"] * u["trees"] * u["rounds"]            # transcript
            + G_out * B + S * B / 16 + 10**5)                  # PRF, H(x), H(y)
    ext = (u["units"] * u["trees"] * u["rounds"] * 12          # round checks
           + u["units"] * u["q"] * u["cols"] * COSET)          # query folds
    cells = layer = comp * CELLS_BLAKE3 + ext * CELLS_EXT
    layers = 1
    while layer > SEGMENT_CELLS:             # aggregate segments until one remains
        layer = math.ceil(layer / SEGMENT_CELLS) * SEG_VERIFY_COMP * CELLS_BLAKE3
        cells += layer
        layers += 1
    return cells * C_CELL * OUTER_ZK, cells, layers


def carried_state(m: Model, P0: int):
    """Elements of the earlier turns' state trees that a session turn opens.

    Dense: K^ and V^ of every layer. GLM (Lemma state-sufficiency): per
    sparse layer and position the latent, the RoPE key and the per-token
    indexer key; per KDA layer H and H^ at the last chunk boundary, k^, v^
    and lambda of up to 127 open-chunk positions, and the convolution
    inputs of the last 3 positions.
    """
    if P0 == 0:
        return 0
    if m.kind == "dense":
        return m.L * 2 * m.n_kv * m.d_h * P0
    return state_per_position(m) * P0 + m.extra["n_kda"] * KDA_CARRY


KDA_CARRY = 2 * 64 * 128**2 + 64 * 128 + (127 * 3 + 3 * 3) * 8192


def sum_min_slots(S, k=2048):
    # sum over positions of min(position, k)
    if S <= k:
        return S * (S + 1) / 2
    return k * (k + 1) / 2 + (S - k) * k


MODELS = [
    Model("Llama 3.1 8B", 8.03e9, 8.03e9, 32, 4096, 14336, 32, 8),
    Model("Qwen2.5-32B", 32.5e9, 32.5e9, 64, 5120, 27648, 40, 8, V=152064),
    Model("Mixtral-8x7B", 46.7e9, 12.9e9, 32, 4096, 2 * 14336, 32, 8, V=32000,
          extra=dict(experts=(2, 8))),
    Model("Llama 3.1 70B", 70.6e9, 70.6e9, 80, 8192, 28672, 64, 8),
    Model("Llama 3.1 405B", 405e9, 405e9, 126, 16384, 53248, 128, 8),
    Model("GLM-5.3-Flash", 320e9, 18e9, 45, 4096, 9 * 2048, 64, 64, V=151552,
          kind="glm", extra=dict(n_kda=34, n_sparse=11)),
]

# ------------------------------------------------------------- machines
# H100-equivalents of proving throughput.
H100 = 1.0
RTX_PER_CARD = 0.75          # band 0.5-1.2: bandwidth 0.53x, int ALU higher
EPS_8 = {"dense": 0.80, "glm": 0.75}
USD_PER_H100_HOUR = 2.50


def node8(kind):
    return 8 * RTX_PER_CARD * EPS_8[kind]


def fmt_t(s):
    if s < 120:
        return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"
    if s < 7200:
        return f"{s/60:.1f} min" if s < 600 else f"{s/60:.0f} min"
    return f"{s/3600:.1f} h"


def check_fields():
    p = 2**31 - 2**24 + 1                          # KoalaBear
    assert p % 4 == 1
    # X^(2^k) - 3 is irreducible for every k when p = 1 mod 4 and 3 is a
    # non-square (Capelli): gives F_{p^4} = F_p[X]/(X^4 - 3) and
    # F_{p^8} = F_p[X]/(X^8 - 3).
    assert pow(3, (p - 1) // 2, p) == p - 1
    v = p - 1
    two_adicity = 0
    while v % 2 == 0:
        v //= 2
        two_adicity += 1
    assert two_adicity == 24                       # RS domains up to 2^24
    assert 2**30.1 < p                             # no-wrap margin
    # BCIKS Johnson-regime term, codeword 2^24, Mode P field.
    for rho in (0.25, 0.5):
        mm = max(math.ceil(math.sqrt(rho) / 0.02), 3)
        eps = (mm + 0.5)**7 * 2**48 / (3 * rho**1.5 * float(p)**8)
        print(f"proximity term rate {rho}: 2^{math.log2(eps):.1f}")
    D = 2**30 + 2**25.6                         # distinct forms per LogUp chunk
    print(f"LogUp per chunk, one instance: 2^{math.log2(2 * D / float(p)**4):.1f}, "
          f"with {LOGUP_GRIND} bits grinding: "
          f"2^{math.log2(2 * D / float(p)**4) - LOGUP_GRIND:.1f}")
    for rho, q, g in ((0.5, queries('P', OPT), QUERY_GRIND),
                      (0.25, queries('P', BASE), QUERY_GRIND), (1 / 16, 44, 20)):
        b = -math.log2(math.sqrt(rho) + 0.01)
        print(f"queries rate {rho}: {q} x {b:.3f} + {g} = {q * b + g:.1f} bits")
    print("field checks passed")


def total_work(m, S, G_out, B=1, prof=OPT, mode="P", recurse=True, wshare=1.0):
    w = proof_work(m, S, G_out, B=B, mode=mode, prof=prof, wopen_share=wshare)[0]
    return w + (recursion_work(m, S, G_out, B, mode, prof)[0] if recurse else 0)


SESSION_BITS = 113         # tight terms in session mode: 2^10 turns x 7 proofs
SESSION_MAX_TURNS = 2**10


SESSION_WSHARE = 1 / 16     # session turns join the shared weight opening
SESSION_FINAL_BYTES = 436 * 2**10   # final layer at 48 queries


def state_per_position(m: Model):
    if m.kind == "dense":
        return m.L * 2 * m.n_kv * m.d_h
    return m.extra["n_sparse"] * (576 + 128)


UDR_DELTA = 0.25            # state trees: unique decoding at rate 1/2
# One batched unique-decoding test per turn, at 2^-116 so that T of them
# fit beside the (7T+1) 2^-113 of the turns' proofs (Theorem session).
UDR_BITS = 116
UDR_QUERIES = math.ceil((UDR_BITS - QUERY_GRIND) / -math.log2(1 - UDR_DELTA))


def session_work(m, turns, n, G_out, prof=OPT, prefix=0, wshare=SESSION_WSHARE,
                 merge=True, stateless=False):
    """H100-seconds for `turns` session turns of n positions each (G_out
    generated), after an optional public prefix committed once; recursion
    included, at session-mode query counts.

    With merge, the session's state trees behave like a binary counter: a
    turn whose tree equals the previous one in size merges them, paying a
    re-commitment and one opening of each merged tree, so a turn opens
    popcount(k) <= log2(k) + 1 earlier trees instead of k. The public
    prefix tree is shared across sessions and never merged.

    Every committed or opened state tree also gets its own proximity test in
    the unique-decoding regime (UDR_QUERIES queries). stateless re-runs F's
    forward pass over the earlier positions and re-encodes the state each
    turn, charged at the four-pass witness rate C_MAC (an upper bound).
    """
    global TARGET_BITS
    saved, TARGET_BITS = TARGET_BITS, SESSION_BITS
    u = units_for("P", prof)
    udr_path = UDR_QUERIES * (DEPTH + 8) * CELLS_BLAKE3 * C_CELL * OUTER_ZK
    w, trees = 0.0, []
    for k in range(turns):
        P0 = prefix + n * k
        n_prev = (len(trees) if merge else k) + (1 if prefix else 0)
        w += proof_work(m, n, G_out, prof=prof, P0=P0, wopen_share=wshare)[0]
        merged_opens = 0
        if merge:
            trees.append(1)
            while len(trees) > 1 and trees[-1] == trees[-2]:
                size = trees.pop() + trees.pop()
                trees.append(size)
                w += (state_per_position(m) * n * size
                      * (u["c_commit"] + C_WRED + C_WOPEN))
                merged_opens += 2
        w += recursion_work(m, n, G_out, 1, "P", prof, P0=P0,
                            n_prev=n_prev + merged_opens)[0]
        w += state_per_position(m) * n * u["c_commit"] * 0.5     # UDR fold
        w += (n_prev + merged_opens + 1) * udr_path
        if stateless and P0 > 0:
            att = (m.L * m.n_h * P0 * P0 / 2 * 2 * m.d_h * 2 if m.kind == "dense"
                   else m.extra["n_sparse"] * 4096 * P0 * P0 / 8)
            w += (2 * m.N_active * P0 + att) * C_MAC
            w += state_per_position(m) * P0 * u["c_commit"]
    TARGET_BITS = saved
    return w


def whole_session_work(m, turns, n, G_out, prof=OPT, wshare=SESSION_WSHARE):
    """One default-mode proof of the final context that checks the sampling
    rule at every turn's generated positions."""
    return total_work(m, n * turns, G_out * turns, prof=prof, wshare=wshare)


def checkpoint_work(m, turns, n, G_out, prof=OPT, wshare=SESSION_WSHARE):
    """Whole-session proofs at turns 1, 2, 4, ..., and at the last turn."""
    cps = sorted({2**i for i in range(int(math.log2(turns)) + 1)} | {turns})
    return sum(whole_session_work(m, c, n, G_out, prof, wshare) for c in cps)


def sessions(prof=OPT):
    usd = lambda w, tok: w / tok * 1e6 / 3600 * USD_PER_H100_HOUR
    print("\n== session mode: turns of 2K positions (512 generated), OPT, "
          "incl. recursion ==")
    M = {m.name: m for m in MODELS}
    print("(every proof, from 0 or not, in a shared weight opening of 16)")
    for name in ("Llama 3.1 8B", "Qwen2.5-32B", "GLM-5.3-Flash"):
        m = M[name]
        for T in (4, 8, 16):
            full = sum(total_work(m, 2048 * k, 512, prof=prof, wshare=SESSION_WSHARE)
                       for k in range(1, T + 1))
            inc = session_work(m, T, 2048, 512, prof)
            last_f = total_work(m, 2048 * T, 512, prof=prof, wshare=SESSION_WSHARE)
            last_i = session_work(m, T, 2048, 512, prof) - session_work(m, T - 1, 2048, 512, prof)
            one = last_f
            print(f"{name:14s} {T:2d} turns ({2*T:2d}K): full {full:7.0f} session {inc:6.0f} "
                  f"H100-s ({full/inc:4.1f}x; one {2*T}K proof {one:5.0f}); last turn "
                  f"{last_f:5.0f} -> {last_i:4.0f}; $/M {usd(full, 2048*T):5.1f} -> "
                  f"{usd(inc, 2048*T):4.1f}")
    print("\n== public prefix committed once, request of 1024 + 512 generated ==")
    for name in ("Llama 3.1 8B", "Qwen2.5-32B", "GLM-5.3-Flash"):
        m = M[name]
        for pre in (2048, 4096, 8192):
            full = total_work(m, pre + 1536, 512, prof=prof)
            inc = session_work(m, 1, 1536, 512, prof, prefix=pre, wshare=1.0)
            print(f"{name:14s} prefix {pre//1024}K: {full:6.1f} -> {inc:5.1f} H100-s "
                  f"({full/inc:.1f}x)")
    m = M["Llama 3.1 8B"]
    print("session-mode queries: rate 1/2",
          math.ceil((SESSION_BITS - QUERY_GRIND) / -math.log2(math.sqrt(0.5) + 0.01)),
          "rate 1/4",
          math.ceil((SESSION_BITS - QUERY_GRIND) / -math.log2(math.sqrt(0.25) + 0.01)),
          "final rate 1/16", math.ceil((SESSION_BITS - 20) / -math.log2(0.25 + 0.01)),
          "| LogUp grinding", LOGUP_GRIND + SESSION_BITS - TARGET_BITS, "bits")
    T = SESSION_MAX_TURNS
    udr = UDR_QUERIES * -math.log2(1 - UDR_DELTA) + QUERY_GRIND
    merge = T * 25 / float(2**31 - 2**24 + 1)**4
    print("session bound: (7T+1) 2^-113 + T 2^-udr + W + merges at T=2^10:",
          f"2^{math.log2((7 * T + 1) * 2.0**-SESSION_BITS + T * 2**-udr + 2**-116 + merge):.2f}",
          f"| final proof ~{SESSION_FINAL_BYTES / 2**10:.0f} KiB")
    print(f"UDR state-tree queries: {UDR_QUERIES} (2^-{udr:.1f} per batched test)")
    print("\n== multi-turn options, 16 turns of 2K (512 generated), OPT, shared opening ==")
    for name in ("Llama 3.1 8B", "Qwen2.5-32B", "GLM-5.3-Flash"):
        mm = M[name]
        fs = sum(total_work(mm, 2048 * k, 512, prof=prof, wshare=SESSION_WSHARE)
                 for k in range(1, 17))
        wh = whole_session_work(mm, 16, 2048, 512, prof)
        cp = checkpoint_work(mm, 16, 2048, 512, prof)
        se = session_work(mm, 16, 2048, 512, prof)
        sl = session_work(mm, 16, 2048, 512, prof, stateless=True)
        print(f"{name:14s} every turn from 0 {fs:6.0f} | whole-session {wh:5.0f} | "
              f"checkpoints {cp:5.0f} | session {se:5.0f} | stateless session {sl:5.0f} "
              f"| session + whole-session backstop {se + wh:5.0f}")
    print("\n== many short turns, 8B, OPT: tree openings without / with merging ==")
    for T, n in ((64, 512), (256, 512), (1024, 256)):
        a = session_work(m, T, n, 128, prof, merge=False)
        b = session_work(m, T, n, 128, prof)
        print(f"T={T:4d} x {n}: {a:7.0f} -> {b:7.0f} H100-s ({(b/a-1)*100:+.1f}%)")


def sizes():
    print("\n== uncompressed proof, Mode P ==")
    M = {m.name: m for m in MODELS}
    for name, S in (("Llama 3.1 8B", 4096), ("Llama 3.1 8B", 16384),
                    ("Qwen2.5-32B", 16384), ("Llama 3.1 405B", 16384),
                    ("GLM-5.3-Flash", 16384), ("GLM-5.3-Flash", 1 << 17),
                    ("GLM-5.3-Flash", 1 << 20)):
        row = []
        for prof in (BASE, OPT):
            u, b, v = proof_size(M[name], S, 512, prof=prof)
            row.append(f"units {u:6d} size {b/2**20:8.0f} MiB verify {fmt_t(v):>8s}")
        print(f"{name:15s} {S:>8d} | " + " | ".join(row))
    print("\n== recursion (outer proof of the inner verifier), Mode P, OPT ==")
    for name, S in (("Llama 3.1 8B", 4096), ("Llama 3.1 8B", 16384),
                    ("Qwen2.5-32B", 16384), ("Llama 3.1 405B", 16384),
                    ("GLM-5.3-Flash", 16384), ("GLM-5.3-Flash", 1 << 17),
                    ("GLM-5.3-Flash", 1 << 20)):
        inner = proof_work(M[name], S, 512, prof=OPT)[0]
        rw, cells, layers = recursion_work(M[name], S, 512, prof=OPT)
        print(f"{name:15s} {S:>8d} inner {inner:9.0f} H100-s  outer {rw:8.1f} H100-s "
              f"({rw/inner*100:4.1f}%)  cells {cells:.1e}  layers {layers}")
    print(f"final proof ~{FINAL_PROOF_BYTES/2**10:.0f} KiB, verify ~{FINAL_VERIFY_S*1e3:.0f} ms")


def projections(prof, label):
    print(f"\n== per-proof work, Mode P, {label} ==")
    for m in MODELS:
        w4, _ = proof_work(m, 4096, 512, prof=prof)
        w, _ = proof_work(m, 16384, 512, prof=prof)
        wb, _ = proof_work(m, 4096, 512, B=4, prof=prof)
        n = node8(m.kind)
        print(f"{m.name:16s} work 4K={w4:6.0f} 16K={w:6.0f} | 1xH100 16K "
              f"{fmt_t(w):>8s} ({16384/w:5.1f} t/s) 4x4K {16384/wb:5.1f} t/s | "
              f"8xRTX 16K {fmt_t(w/n):>8s} ({16384*n/w:5.0f} t/s) "
              f"4x4K {16384*n/wb:5.0f} t/s | $/Mtok "
              f"{w/16384*1e6/3600*USD_PER_H100_HOUR:5.1f}")


def glm_sweep(prof, label):
    glm = MODELS[-1]
    n = node8("glm")
    print(f"\n== GLM context sweep, 8x RTX, Mode P, {label} ==")
    for S in (4096, 16384, 65536, 131072, 262144, 1 << 20):
        w, _ = proof_work(glm, S, 512, prof=prof)
        w2, _ = proof_work(glm, S, 512, prof=prof, relu_per_head=False)
        print(f"S={S:8d} {fmt_t(w/n):>8s} {S*n/w:6.0f} tok/s | no per-head ReLU: "
              f"{fmt_t(w2/n):>8s} {S*n/w2:6.0f} tok/s")


def bills(prof, label):
    glm = MODELS[-1]
    n = node8("glm")
    reqs = [(1024, 256), (8192, 512), (32768, 512), (131072, 1024)]
    print(f"\n== serving bills (with recursion), {label} ==")
    for P, Gn in reqs:
        tp = total_work(glm, P + Gn, Gn, prof=prof) / n
        tw = total_work(glm, P + Gn, Gn, prof=prof, wshare=1 / 16) / n
        single = [P / R + Gn / 200 for R in (6000, 15000)]
        busy = [P / R + Gn / Rd for R, Rd in ((6000, 2000), (15000, 5000))]
        print(f"GLM {P//1024}K/{Gn}: prove {fmt_t(tp)} (shared opening {fmt_t(tw)}); single "
              f"{(single[0]+tp)/single[0]:.0f}-{(single[1]+tp)/single[1]:.0f}x; "
              f"busy {(busy[0]+tp)/busy[0]:.0f}-{(busy[1]+tp)/busy[1]:.0f}x")
    m8 = MODELS[0]
    for P, Gn in reqs[:3]:
        w = total_work(m8, P + Gn, Gn, prof=prof)
        ww = total_work(m8, P + Gn, Gn, prof=prof, wshare=1 / 16)
        single = [P / R + Gn / 150 for R in (10000, 20000)]
        busy = [P / R + Gn / Rd for R, Rd in ((10000, 2500), (20000, 6000))]
        print(f"8B  {P//1024}K/{Gn}: prove {fmt_t(w)} (shared opening {fmt_t(ww)}); single "
              f"{(single[0]+w)/single[0]:.0f}-{(single[1]+w)/single[1]:.0f}x; "
              f"busy {(busy[0]+w)/busy[0]:.0f}-{(busy[1]+w)/busy[1]:.0f}x")
    for m in (m8, glm):
        w = total_work(m, 16384, 512, prof=prof)
        print(f"{m.name} $/M proven tokens incl. recursion at 16K: "
              f"{w/16384*1e6/3600*USD_PER_H100_HOUR:.1f}")


def ladder():
    """Cumulative optimizations, Qwen2.5-32B shape, one H100."""
    qwen = MODELS[1]
    print("\n== optimization ladder, Qwen2.5-32B shape, one H100 (tok/s) ==")
    print(f"{'':40s} {'16K x1':>7s} {'4K x4':>7s}")
    prof = dict(BASE)
    rows = [("spec gadgets (Mode P8)", None)] + [
        ("+ multi-column range tables", "tuple"),
        ("+ fused gadgets", "fused"),
        ("+ one relation per softmax entry", "score1"),
        ("+ rate-1/2 code", "rate_half"),
        ("+ small-value sumcheck and GKR", "small_value")]
    for name, flag in rows:
        if flag:
            prof[flag] = True
        a = 16384 / proof_work(qwen, 16384, 512, prof=prof, mode="P8")[0]
        b = 16384 / proof_work(qwen, 4096, 512, B=4, prof=prof, mode="P8")[0]
        print(f"{name:40s} {a:7.1f} {b:7.1f}")
    saved = dict(MODES["P"]), KAPPA["P"]
    MODES["P"]["c_lookup"], KAPPA["P"] = 2.75e-10, 2
    a = 16384 / proof_work(qwen, 16384, 512, prof=OPT)[0]
    b = 16384 / proof_work(qwen, 4096, 512, B=4, prof=OPT)[0]
    print(f"{'+ IOP challenges in F_p^4, two LogUp':40s} {a:7.1f} {b:7.1f}")
    MODES["P"], KAPPA["P"] = saved
    a = 16384 / proof_work(qwen, 16384, 512, prof=OPT)[0]
    b = 16384 / proof_work(qwen, 4096, 512, B=4, prof=OPT)[0]
    print(f"{'+ one LogUp instance, 11-bit grinding':40s} {a:7.1f} {b:7.1f}")
    a = 16384 / total_work(qwen, 16384, 512)
    b = 16384 / total_work(qwen, 4096, 512, B=4)
    print(f"{'+ recursion (included in cost)':40s} {a:7.1f} {b:7.1f}")
    for mode in ("C",):
        a = 16384 / proof_work(qwen, 16384, 512, prof=OPT, mode=mode)[0]
        b = 16384 / proof_work(qwen, 4096, 512, B=4, prof=OPT, mode=mode)[0]
        print(f"{'OPT, Mode C':40s} {a:7.1f} {b:7.1f}")
    for mode in ("P", "C"):
        kw = dict(prof=OPT, mode=mode, window=1024, global_every=6)
        a = 16384 / proof_work(qwen, 16384, 512, **kw)[0]
        b = 16384 / proof_work(qwen, 4096, 512, B=4, **kw)[0]
        print(f"{'OPT, 5:1 local attention, Mode ' + mode:40s} {a:7.1f} {b:7.1f}")
    print(f"{'Attestable (Gemma 4 31B)':40s} {53:7d} {77:7d}")
    _, info = proof_work(qwen, 16384, 512, prof=OPT)
    _, noatt = proof_work(qwen, 16384, 512, prof=OPT, window=0, global_every=10**9)
    print("Qwen OPT 16K per-element totals:",
          {k: f"{v:.2e}" for k, v in info.items()})
    print("Mode C / Mode P under OPT, 16K:",
          round(proof_work(qwen, 16384, 512, prof=OPT, mode="C")[0]
                / proof_work(qwen, 16384, 512, prof=OPT)[0], 2))


if __name__ == "__main__":
    check_fields()
    projections(BASE, "BASE")
    projections(OPT, "OPT")
    print("\n== Mode C / Mode P ratio at 16K, BASE ==")
    for m in MODELS:
        print(m.name, round(proof_work(m, 16384, 512, mode='C')[0] /
                            proof_work(m, 16384, 512)[0], 2))
    glm_sweep(BASE, "BASE")
    glm_sweep(OPT, "OPT")
    bills(BASE, "BASE")
    bills(OPT, "OPT")
    print("\n== breakdown at 16K, Mode P ==")
    for prof, label in ((BASE, "BASE"), (OPT, "OPT")):
        for m in (MODELS[0], MODELS[1], MODELS[-1]):
            _, info = proof_work(m, 16384, 512, prof=prof, breakdown=True)
            print(label, m.name, {k: round(v, 3) for k, v in info.items()
                                  if k in ("commit", "lookup", "rel", "forward",
                                           "fixed")})
    ladder()
    sizes()
    print("\nLogUp chunks at 16K (OPT):",
          {m.name: proof_work(m, 16384, 512, prof=OPT)[1]["chunks"] for m in MODELS})
    n = node8("glm")
    print("\nweight claim reduction per proof + opening, 8xRTX, GLM:",
          fmt_t(MODELS[-1].N_total * C_WRED / n), "+",
          fmt_t(MODELS[-1].N_total * C_WOPEN / n),
          " 405B:", fmt_t(405e9 * (C_WRED + C_WOPEN) / node8('dense')))
    for mode, prof in (("P", BASE), ("P", OPT)):
        print("queries", "rate 1/2" if prof["rate_half"] else "rate 1/4",
              queries(mode, prof))
    print("\n== 12-bit attention probabilities (PV blocks of 1024 keys), OPT ==")
    for m in (MODELS[0], MODELS[1], MODELS[3]):
        for S in (4096, 16384, 65536):
            w10 = total_work(m, S, 512)
            PV_BLOCK = 1024
            w12 = total_work(m, S, 512)
            PV_BLOCK = 4096
            print(f"{m.name:15s} {S:6d}: x{w12 / w10:.3f}")
    sessions()
