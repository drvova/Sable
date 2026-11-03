"""Regression for the router SatShift wrap (found while writing the bounds table): in the saturated case
the relation o acc - (2 R_SAT + 1) 2^(k-1) - [o = -1] = wl + 2^15 wh reaches |acc| + 2^(k+15) + 2^30, above p
at k = 15, so with the former bound k_r <= 15 a prover could claim the saturated logit +R_SAT for any
expert of a row with k_r = 15 (limbs from the residue mod p) and route the token to it.  The bound is
now k_r <= 14 (model.R_KMAX), where the residue never fits the limbs.  On a tiny MoE whose layer-0 router
is re-scaled so that k_r reaches the bound: (a) with the former bound 15 the forged proof is accepted
(the forgery is real and changes the generated tokens); (b) with the bound 14, the same forgery at
k_r = 14 is rejected."""
import torch
from zk import model as mdl, moe, llm
from zk import tables as tb
from zk.field import P
from tests.test_llm import statement
from tests.test_llm_moe import moe_tiny
from tests.test_wrap_bounds import rejects

B, S, NG = 2, 256, 16
_sat, _pack, RK = mdl.satshift, moe.satshift_pack, mdl.R_KMAX
EVIL, KR = [False], [0]


def forge_rows(acc, k):
    """Rows at the bound k_r = KR and, per row, the expert with the smallest router accumulator."""
    rows = (k.reshape(-1) == KR[0])
    return rows, acc.argmin(dim=-1)


def evil_satshift(acc, k, Bd):
    q = _sat(acc, k, Bd)
    if EVIL[0] and Bd == mdl.R_SAT:
        rows, j = forge_rows(acc, k)
        q = q.clone()
        q[rows, j[rows]] = Bd
    return q


def evil_pack(acc, k, Bd):
    if not (EVIL[0] and Bd == mdl.R_SAT):
        return _pack(acc, k, Bd)
    r = _pack(acc, k, Bd)
    rows, j = forge_rows(acc, k)
    s = mdl.pow2(k).expand_as(acc)
    a = acc[rows, j[rows]]
    res = (a - (2 * Bd + 1) * (s[rows, j[rows]] >> 1)) % P       # the residue a cheating prover splits
    assert bool((res < (1 << 30)).all()) or KR[0] < 15
    r = {n: v.clone() for n, v in r.items()}
    r["q"][rows, j[rows]] = Bd
    r["o"][rows, j[rows]] = 1
    r["wh"][rows, j[rows]] = res >> 15
    return r


def model_at(kr):
    """moe_tiny with layer 0's router exponent lowered so that the largest router shift is kr (the router
    weights rounded down by as many bits, so their real values shrink; the function stays defined)."""
    M = moe_tiny()
    sh = llm.Shp(M.cfg, B, S, *llm.block_bits(M.cfg), NG)
    tokens, _ = statement(M, B, S, NG)
    mdl.drop_caps(M)
    mdl.ensure_caps(M)
    W = llm.LayerWit(M, sh, 0, M.emb[tokens.reshape(-1)].to(torch.int64))
    d = kr - int(W.moe["kr"].max())
    Ly = M.layers[0]
    Ly.Wr = torch.round(Ly.Wr.double() / 2.0 ** d).to(Ly.Wr.dtype)
    Ly.eWr = Ly.eWr - d
    mdl.drop_caps(M)
    return M, sh


def forged(kr, rkmax):
    mdl.R_KMAX = rkmax
    KR[0] = kr
    try:
        M, sh = model_at(kr)
        tok_h, tgt_h = statement(M, B, S, NG)
        mdl.satshift, moe.satshift_pack = evil_satshift, evil_pack
        cc, chk = tb.check_counts, mdl.check
        tb.check_counts = lambda *a, **k: None
        mdl.check = lambda *a, **k: None
        EVIL[0] = True
        try:
            tok_e, tgt_e = statement(M, B, S, NG)
            msgs = llm.prove(M, sh, tok_e, tgt_e)
        finally:
            EVIL[0] = False
            mdl.satshift, moe.satshift_pack = _sat, _pack
            tb.check_counts, mdl.check = cc, chk
            tb._ERRS.clear()
        n = int((tgt_h != tgt_e).sum())
        print(f"k_r = {kr}, bound {rkmax}: forged routing changes {n} of {tgt_h.numel()} generated tokens")
        return n, rejects(lambda: llm.verify(M, sh, tok_e, tgt_e, msgs))
    finally:
        mdl.R_KMAX = RK


def main():
    n, rej = forged(15, 15)
    assert n > 0 and not rej, "the former bound: the forgery should be accepted"
    print("(a) former bound k_r <= 15: forged saturated router logit accepted")
    n, rej = forged(14, 14)
    assert rej
    print("(b) bound k_r <= 14: the same forgery is rejected")


def test_router_wrap():
    main()


if __name__ == "__main__":
    main()
