"""ShiftRound regimes (tables.sr_row, tables.sra_row) as the gadget uses them: lookups
(tR, acc + R - s c), (tB, c + oB) and, for shifts that may be left shifts, (tA, f acc + oA).
Honest outputs pass for every k; a left shift of a large accumulator whose residue mod p lands in
A (which the first two lookups alone accept) fails the input bound."""
import random
from zk.field import P
from zk import tables as tb


def in_range(t, v):
    return 0 <= t <= 24 and 0 <= v % P < (1 << t)


def lookups_ok(k, acc, c):
    kk, s, R, tR, tB, oB = tb.sr_row(k)
    _, tA, oA, f = tb.sra_row(k)
    return (in_range(tR, acc + R - s * c) and in_range(tB, c + oB)
            and in_range(tA, f * acc + oA))


def shift(acc, k):
    return (acc + (1 << (k - 1))) >> k if k > 0 else acc << -k


def test_shift_regimes():
    rng = random.Random(1)
    A = 1 << 23
    ACC = (127 * 127) << 15                    # largest |acc| of an 8-bit product of 2^15 terms
    for k in range(tb.SR_KMIN, tb.SR_KMAX + 1):
        lim = min(ACC + 1, (A << k) if k >= 0 else A >> -k)
        for _ in range(200):
            acc = rng.randrange(-lim, lim)
            c = shift(acc, k)
            if -A <= c < A:
                assert lookups_ok(k, acc, c), (k, acc, c)
                assert not lookups_ok(k, acc, c + 1), (k, acc)
    print("honest shifts pass, off-by-one outputs fail, k in [-16, 24]")
    forged = 0
    for k in range(tb.SR_KMIN, 0):
        for _ in range(200):
            acc = rng.randrange(A >> -k, ACC + 1) * rng.choice((-1, 1))   # true left shift leaves A
            c = (acc * pow(2, -k, P)) % P
            c = c - P if c > P // 2 else c
            if -A <= c < A:                    # the residue lands in A: accepted without the bound
                forged += 1
                kk, s, R, tR, tB, oB = tb.sr_row(k)
                assert in_range(tR, acc + R - s * c) and in_range(tB, c + oB)
                assert not lookups_ok(k, acc, c), (k, acc, c)
    assert forged > 0
    print(f"left shifts of out-of-range accumulators: {forged} residues in A, all rejected by the input bound")


if __name__ == "__main__":
    test_shift_regimes()
