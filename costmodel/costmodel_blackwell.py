"""costmodel.py re-priced for one NVIDIA RTX PRO 6000 Blackwell (GB202).

Derived only from costmodel.py's H100 constants and from published
architecture figures, with no measurements of the prover. Every H100 unit
cost is split into the resources it uses (32-bit IMAD-class multiplies,
other integer ALU ops, DRAM bytes, INT8 tensor ops, PCIe host traffic),
and each share is scaled by the RTX/H100 throughput ratio of that resource:

    cost_rtx = cost_h100 * sum_r share_r * (thr_h100_r / thr_rtx_r)

The shares come from rough op and byte counts of the algorithm that each
constant prices (see WORK). The H100 time that the counts do not account for
(occupancy, latency, launches) is assumed to split across resources in the
same proportions. Units below: RTX-seconds. Run: python3 costmodel_blackwell.py
"""
import copy
from contextlib import contextmanager

import costmodel as cm

# ------------------------------------------------------------ architectures
# H100 SXM5: 132 SMs at 1.98 GHz boost; per SM per clock 64 INT32 lanes and
# 64 IMAD (FMA-heavy pipe); HBM3 3.35 TB/s; 50 MB L2; INT8 dense 1979 TOPS;
# 80 GB; PCIe Gen5 x16; NVLink 900 GB/s.
# RTX PRO 6000 Blackwell (Workstation/Server Ed.): 188 SMs at 2.617 GHz
# boost; 128 unified FP32/INT32 lanes per SM; GDDR7 512-bit 1.792 TB/s;
# 128 MB L2; INT8 dense ~1008 TOPS (5th-gen tensor cores, scaled from GB202
# per-SM rate); 96 GB; PCIe Gen5 x16; no NVLink; 600 W.
H100_SM, H100_CLK = 132, 1.98e9
RTX_SM, RTX_CLK = 188, 2.617e9

H100_THR = dict(imad=H100_SM * 64 * H100_CLK,
                alu=H100_SM * 64 * H100_CLK,
                dram=3.35e12, tensor=1979e12, pcie=64e9)

# Scenarios for the uncertain parts of GB202. The int lane counts are the
# open question: Blackwell unified the FP32/INT32 lanes, but it is not
# documented whether IMAD.HI and SHF/PRMT (BLAKE3 rotations, about 1/3 of
# its ops) issue on all 128 lanes.
#   clk:    sustained-clock factor vs spec boost ratio (600 W integer load)
#   imad:   IMAD lanes per SM per clock
#   alu:    effective int ALU lanes per SM per clock for the op mix
#   dram:   effective bandwidth ratio (GDDR7 efficiency; 128 MB L2 capture)
#   tensor: INT8 dense ratio vs H100
SCENARIOS = {
    "low":     dict(clk=0.88, imad=64, alu=64, dram=0.50, tensor=0.45),
    "central": dict(clk=1.00, imad=64, alu=96, dram=1.792 / 3.35,
                    tensor=1008 / 1979),
    "high":    dict(clk=1.10, imad=128, alu=128, dram=0.65, tensor=0.55),
}


def rtx_thr(s):
    f = RTX_SM * RTX_CLK * s["clk"]
    return dict(imad=f * s["imad"], alu=f * s["alu"],
                dram=H100_THR["dram"] * s["dram"],
                tensor=H100_THR["tensor"] * s["tensor"],
                pcie=H100_THR["pcie"])


# --------------------------------------------------------- work composition
# Per unit of work: (IMAD-class ops, other int ALU ops, DRAM bytes).
# Base-field Montgomery multiply: 4 IMAD (lo, hi, m = lo*mu, hi(m, p)) + 2 ALU.
# F_{p^4} multiply (X^4 - 3, lazy 64-bit accumulation, 4 reductions): ~(44, 44).
# F_{p^8} multiply: ~(140, 150). BLAKE3 compression: ~770 ALU (IADD3, LOP3,
# SHF/PRMT), no multiplies.
WORK = {
    # Per committed base element, rate 1/4, codeword 2^24 in cosets of 4:
    # iNTT + 4 coset NTTs of 2^22 = 55 butterflies (220 IMAD, 440 ALU); leaf
    # hashing of 16 B = 0.25 compressions (190 ALU); F_{p^8} batching of 4
    # symbols with lazy accumulation (100 IMAD, 120 ALU). DRAM: input, two
    # passes per transform, hash and batching reads of the codeword.
    "commit": (320, 750, 100),
    # Per lookup, F_{p^4}: leaf denominator (2 base x ext), fraction tree
    # (3 ext mul per node), GKR sumcheck (~10 ext mul per leaf over all
    # rounds and layers, with eq folding). DRAM: tree write 32 B plus round
    # reads/writes over the geometric layer sizes.
    "lookup4": (600, 750, 190),
    "lookup8": (1880, 2300, 370),
    # Per relation term, F_{p^4}: ~4 ext mul for round evaluations + 2
    # base x ext column folds; ~1.5 columns per term read once in the base
    # field then folded in the extension.
    "rel4": (210, 250, 80),
    "rel8": (620, 750, 150),
}
# These counts reproduce the H100 constants' P8/P ratios (lookup 2.6 vs 2.7,
# rel 2.5 vs 2.5) and land within 1.1-1.3x of the absolute c_lookup and
# c_rel, so the shares are taken as the H100 time composition.

# Terms with no clean op count, given as H100 time shares directly.
MIX_DIRECT = {
    # Four INT8 forward passes: GEMMs on tensor cores (C_MAC is ~2x the
    # ideal 4 passes at 1979 TOPS), epilogues/requant and attention traffic.
    "forward": dict(tensor=0.6, dram=0.3, alu=0.1),
    # Weight claim reduction: C_WRED = 2e-11 s is 50 G weights/s, which is
    # PCIe Gen5 x16 at 1 B per weight when the weights do not fit in device
    # memory (GLM, 405B), so half the share is host link.
    "wred": dict(pcie=0.5, dram=0.25, imad=0.15, alu=0.1),
    # Merkle nodes beyond the 2^20 per unit: pure BLAKE3.
    "hash": dict(alu=1.0),
}


def shares(name):
    if name in MIX_DIRECT:
        return MIX_DIRECT[name]
    imad, alu, b = WORK[name]
    t = dict(imad=imad / H100_THR["imad"], alu=alu / H100_THR["alu"],
             dram=b / H100_THR["dram"])
    s = sum(t.values())
    return {k: v / s for k, v in t.items()}


def mult(name, scn="central"):
    """RTX time / H100 time for one unit of `name`."""
    thr = rtx_thr(SCENARIOS[scn])
    return sum(f * H100_THR[r] / thr[r] for r, f in shares(name).items())


# ----------------------------------------------------- re-pricing costmodel
_GLOBALS = ("MODES", "C_MAC", "C_WRED", "C_WOPEN", "NODE_HASH_BPS", "C_CELL")


@contextmanager
def blackwell(scn="central"):
    """Reprice costmodel's unit costs so that its functions return RTX-seconds.

    Unchanged: protocol parameters (queries, unit size, tables, chunking),
    SMALL_VALUE_* and RATE_HALF_COMMIT (fractions of work), and the CPU
    verifier constants BLAKE3_S and EXT_ROUND_S. UNIT_BYTES stays at 16 GB
    so that proofs are identical; 96 GB would allow larger units.
    """
    saved = {k: copy.deepcopy(getattr(cm, k)) for k in _GLOBALS}
    try:
        for mode, u in cm.MODES.items():
            e = "8" if mode == "P8" else "4"
            u["c_commit"] *= mult("commit", scn)
            u["c_lookup"] *= mult("lookup" + e, scn)
            u["c_rel"] *= mult("rel" + e, scn)
        cm.C_MAC *= mult("forward", scn)
        cm.C_WRED *= mult("wred", scn)
        cm.C_WOPEN *= mult("commit", scn)          # fold, encode, open
        cm.NODE_HASH_BPS /= mult("hash", scn)
        cm.C_CELL *= mult("commit", scn)           # outer commit + constraints
        yield
    finally:
        for k, v in saved.items():
            setattr(cm, k, v)


# 8 cards share a PCIe Gen5 fabric (no NVLink); costmodel's EPS_8 already
# describes the 8x RTX node, so it carries over as is.
def node8(kind):
    return 8 * cm.EPS_8[kind]


USD_PER_RTX_HOUR = 1.30     # placeholder rental rate; replace with a quote


def usd_per_mtok(w, tok):
    return w / tok * 1e6 / 3600 * USD_PER_RTX_HOUR


# ------------------------------------------------------------------ reports
def unit_table():
    print("== resource ratios RTX / H100 (throughput) ==")
    for scn in SCENARIOS:
        thr = rtx_thr(SCENARIOS[scn])
        print(f"{scn:8s} " + "  ".join(f"{r} {thr[r] / H100_THR[r]:.2f}"
                                        for r in H100_THR))
    print("\n== H100 time shares and RTX/H100 time multiplier (low|central|high"
          " = pessimistic|central|optimistic for RTX) ==")
    for name in list(WORK) + list(MIX_DIRECT):
        sh = " ".join(f"{r}={f:.2f}" for r, f in shares(name).items())
        m = [mult(name, s) for s in ("low", "central", "high")]
        print(f"{name:8s} {sh:42s} x{m[1]:.2f}  [{m[0]:.2f}, {m[2]:.2f}]")
    print("\n== unit costs, Mode P (s per unit): H100 -> RTX central ==")
    h = dict(cm.MODES["P"], C_MAC=cm.C_MAC, C_WRED=cm.C_WRED,
             C_WOPEN=cm.C_WOPEN, NODE_HASH_BPS=cm.NODE_HASH_BPS,
             C_CELL=cm.C_CELL)
    with blackwell():
        r = dict(cm.MODES["P"], C_MAC=cm.C_MAC, C_WRED=cm.C_WRED,
                 C_WOPEN=cm.C_WOPEN, NODE_HASH_BPS=cm.NODE_HASH_BPS,
                 C_CELL=cm.C_CELL)
    for k in h:
        print(f"{k:14s} {h[k]:.3g} -> {r[k]:.3g}")


def per_card(prof, label):
    print(f"\n== per-proof work, Mode P, {label}: H100-s vs RTX-s, "
          "RTX in H100-equivalents (band) ==")
    for m in cm.MODELS:
        h = cm.proof_work(m, 16384, 512, prof=prof)[0]
        r = {}
        for scn in SCENARIOS:
            with blackwell(scn):
                r[scn] = cm.proof_work(m, 16384, 512, prof=prof)[0]
        with blackwell():
            wb = cm.proof_work(m, 4096, 512, B=4, prof=prof)[0]
            _, info = cm.proof_work(m, 16384, 512, prof=prof, breakdown=True)
        n = node8(m.kind)
        w = r["central"]
        print(f"{m.name:16s} 16K H100 {h:6.0f} RTX {w:6.0f} -> "
              f"{h / w:.2f} [{h / r['low']:.2f}, {h / r['high']:.2f}] | "
              f"1xRTX {fmt(w):>8s} ({16384 / w:5.1f} t/s) 4x4K {16384 / wb:5.1f} t/s | "
              f"8xRTX {fmt(w / n):>8s} ({16384 * n / w:5.0f} t/s) | "
              f"$/Mtok {usd_per_mtok(w, 16384):5.1f}")
        print(f"{'':16s} breakdown " + " ".join(
            f"{k} {info[k]:.2f}" for k in ("commit", "lookup", "rel", "forward",
                                          "fixed")))


def fmt(s):
    return cm.fmt_t(s)


def glm_sweep(prof, label):
    glm = cm.MODELS[-1]
    n = node8("glm")
    n_h = cm.node8("glm")
    print(f"\n== GLM context sweep, 8x RTX, Mode P, {label} "
          "(costmodel.py's RTX_PER_CARD=0.75 projection in brackets) ==")
    for S in (4096, 16384, 65536, 131072, 262144, 1 << 20):
        h = cm.proof_work(glm, S, 512, prof=prof)[0]
        with blackwell():
            w = cm.proof_work(glm, S, 512, prof=prof)[0]
        print(f"S={S:8d} {fmt(w / n):>8s} {S * n / w:6.0f} tok/s  "
              f"[{fmt(h / n_h):>8s} {S * n_h / h:6.0f} tok/s]")


def recursion():
    print("\n== total incl. recursion, OPT, Mode P, 1x RTX ==")
    M = {m.name: m for m in cm.MODELS}
    for name, S in (("Llama 3.1 8B", 4096), ("Llama 3.1 8B", 16384),
                    ("Qwen2.5-32B", 16384), ("GLM-5.3-Flash", 16384),
                    ("GLM-5.3-Flash", 1 << 17)):
        h = cm.total_work(M[name], S, 512)
        with blackwell():
            inner = cm.proof_work(M[name], S, 512, prof=cm.OPT)[0]
            rw = cm.recursion_work(M[name], S, 512, prof=cm.OPT)[0]
        print(f"{name:15s} {S:>8d} inner {inner:8.0f} outer {rw:7.1f} RTX-s "
              f"({rw / inner * 100:4.1f}%) | H100 total {h:8.0f} -> "
              f"{h / (inner + rw):.2f} H100-eq/card | $/Mtok "
              f"{usd_per_mtok(inner + rw, S):5.1f} (H100 "
              f"{h / S * 1e6 / 3600 * cm.USD_PER_H100_HOUR:5.1f})")


def ladder():
    """Qwen2.5-32B, cumulative optimizations, tok/s on one H100 vs one RTX."""
    qwen = cm.MODELS[1]
    print("\n== optimization ladder, Qwen2.5-32B, Mode P8 then P "
          "(tok/s 16K x1: H100 -> RTX) ==")
    prof = dict(cm.BASE)
    rows = [("spec gadgets (Mode P8)", None, "P8"),
            ("+ multi-column range tables", "tuple", "P8"),
            ("+ fused gadgets", "fused", "P8"),
            ("+ one relation per softmax entry", "score1", "P8"),
            ("+ rate-1/2 code", "rate_half", "P8"),
            ("+ small-value sumcheck and GKR", "small_value", "P8"),
            ("Mode P (F_p^4 IOP, one LogUp)", None, "P")]
    for name, flag, mode in rows:
        if flag:
            prof[flag] = True
        h = 16384 / cm.proof_work(qwen, 16384, 512, prof=prof, mode=mode)[0]
        with blackwell():
            r = 16384 / cm.proof_work(qwen, 16384, 512, prof=prof, mode=mode)[0]
        print(f"{name:36s} {h:6.1f} -> {r:6.1f}  ({r / h:.2f}x)")


if __name__ == "__main__":
    unit_table()
    per_card(cm.BASE, "BASE")
    per_card(cm.OPT, "OPT")
    glm_sweep(cm.OPT, "OPT")
    recursion()
    ladder()
