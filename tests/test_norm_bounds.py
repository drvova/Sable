"""Signed normalizations with a left shift (E < 0), as the gadgets use them:
NormScale   (NPOW, 2 E_h + 15, E+, 2^E+, 2^E-, 1, ng), (E+, rho), (16, ng Sigma), (RSQ, delta - 2^14, mu),
            2^E- Sigma - 2^E+ delta - rho = 0;
attention   (SR, E, s, ...), (tR, rhoD), (RECIP, delta - 2^15, recip), D - s delta - rhoD = 0,
denominator fk Boolean, (4, fk (E - 7) + (1 - fk) (6 - E)), (5, E + 5), (5, 14 - E), (22, (1 - fk) D).
For E < 0 the relation is 2^E- X = delta mod p. Without the bounds (16, ng Sigma) and (22, (1 - fk) D),
an X with 2^E- X >= p whose residue lands in the mantissa interval has a second witness (a review of
the MLSys submission found Sigma = 130050, E = -14, delta = 32767). With them, 2^E- X < 2^30 < p, the
relation holds over the integers, and every X has exactly one witness."""
import random
from zk.field import P
from zk import tables as tb
from zk import model as mdl


def in_range(t, v):
    return 0 <= t <= 24 and 0 <= v % P < (1 << t)


def ns_ok(sig, E, delta, rho, bound=True):
    """The NormScale lookups and relation for one row; the prover chooses E, delta, rho."""
    if E % 2 or not -14 <= E <= 14:                     # NPOW rows have E in [-15, 15], [E even] = 1
        return False
    _, Ep, pEp, pEn, _, ng = tb.npow_row(E)
    return (in_range(Ep, rho) and (not bound or in_range(16, ng * sig))
            and (1 << 14) <= delta % P < (1 << 16) and (pEn * sig - pEp * delta - rho) % P == 0)


def ns_witnesses(sig, bound=True):
    """Every (E, delta, rho) the lookups accept for Sigma: for E >= 0 the relation is below p, so delta
    and rho are Sigma's quotient and remainder; for E < 0, rho = 0 and delta = 2^-E Sigma mod p."""
    out = []
    for E in range(-14, 15, 2):
        delta, rho = ((sig >> E), sig & ((1 << E) - 1)) if E >= 0 else ((sig << -E) % P, 0)
        if ns_ok(sig, E, delta, rho, bound):
            out.append((E, delta, rho))
    return out


def den_ok(D, E, delta, rho, fk, bound=True):
    """The attention denominator's lookups and relation for one (row, head)."""
    if not tb.SR_KMIN <= E <= tb.SR_KMAX or fk not in (0, 1):
        return False
    _, s, _, tR, _, _ = tb.sr_row(E)
    return (in_range(tR, rho) and (1 << 15) <= delta % P < (1 << 16) and (D - s * delta - rho) % P == 0
            and in_range(4, fk * (E - mdl.KM_MIN) + (1 - fk) * (mdl.KM_MIN - 1 - E))
            and in_range(5, E + 5) and in_range(5, 14 - E) and (not bound or in_range(22, (1 - fk) * D)))


def den_witnesses(D, bound=True):
    out = []
    for E in range(-5, 15):
        delta, rho = ((D >> E), D & ((1 << E) - 1)) if E >= 0 else ((D << -E) % P, 0)
        fk = int(E >= mdl.KM_MIN)
        if den_ok(D, E, delta, rho, fk, bound):
            out.append((E, delta, rho))
    return out


def wrapped(lo, hi, En, half, rng, n):
    """n integers X in [lo, hi) with 2^En X >= p and 2^En X mod p in [half, 2^16): X = delta 2^-En mod p."""
    inv, out = pow(2, P - 1 - En, P), []
    while len(out) < n:
        X = rng.randrange(half, 1 << 16) * inv % P
        if lo <= X < hi and (X << En) >= P:
            out.append(X)
    return out


if __name__ == "__main__":
    rng = random.Random(1)
    # NormScale: Sigma = sum q^2 + 1 < 2^29 for rows of up to 2^15 int8 values
    sigs = list(range(1, 1 << 12)) + [rng.randrange(1, 1 << 29) for _ in range(20000)]
    for sig in sigs:
        (E, delta, rho), = ns_witnesses(sig)
        assert E == 2 * mdl.fdiv(sig.bit_length() - 15, 2) and (sig << max(-E, 0)) == (delta << max(E, 0)) + rho
    print(f"NormScale: {len(sigs)} honest Sigma, each with exactly one witness")
    for sig in (130050, 130051, 130052):                 # the review's instance
        assert ns_ok(sig, -14, (sig << 14) % P, 0, bound=False)
        assert not ns_ok(sig, -14, (sig << 14) % P, 0)
    forged = 0
    for En in range(2, 15, 2):
        for sig in wrapped(1, 1 << 29, En, 1 << 14, rng, 300):
            assert (-En, (sig << En) % P, 0) in ns_witnesses(sig, bound=False)
            assert len(ns_witnesses(sig, bound=False)) == 2
            assert len(ns_witnesses(sig)) == 1 and ns_witnesses(sig)[0][0] >= 0
            forged += 1
    print(f"NormScale: {forged} wrapped left shifts (E in [-14, -2]) accepted without (16, ng Sigma), "
          "all rejected with it")
    # attention denominator: D in [2^10, 2^30] (S <= 2^20 keys)
    Ds = list(range(1 << 10, 1 << 13)) + [rng.randrange(1 << 10, (1 << 30) + 1) for _ in range(20000)]
    for D in Ds:
        (E, delta, rho), = den_witnesses(D)
        assert E == D.bit_length() - 16 and (D << max(-E, 0)) == (delta << max(E, 0)) + rho
    print(f"attention denominator: {len(Ds)} honest D, each with exactly one witness")
    forged = 0
    for En in range(1, 6):
        for D in wrapped(1 << 10, (1 << 30) + 1, En, 1 << 15, rng, 300):
            assert (-En, (D << En) % P, 0) in den_witnesses(D, bound=False)
            assert len(den_witnesses(D)) == 1 and den_witnesses(D)[0][0] >= 0
            forged += 1
    print(f"attention denominator: {forged} wrapped left shifts (E in [-5, -1]) accepted without "
          "(22, (1 - fk) D), all rejected with it")
