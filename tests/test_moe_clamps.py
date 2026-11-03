"""The MoE row clamps on a tiny Qwen3-MoE model: (a) layer 0 is re-quantized so that the router shift
Eh - eWr - 8 falls below 1 for some rows (router weights halved d_r times, eWr raised by d_r), the expert
gate shift eG - eWg_e + Eh below 1 for some slots (likewise), and the down shift kd above 24 for some rows
(eWd_e lowered): F_O shifts those rows by max(., 1) and min(kd, 24), the serving engine equals evalfo,
and the proof verifies; (b) a prover that marks an unclamped row or slot as clamped (shift exactly 1,
or kd exactly 24: the same shift either way) is rejected by the range lookup of the flag's branch."""
import torch
from zk import model as mdl, fofast, llm, moe
from zk import tables as tb
from tests.test_llm import statement
from tests.test_llm_moe import moe_tiny
from tests.test_wrap_bounds import rejects

B, S, NGEN = 2, 256, 16


def halve(W, d):
    """W 2^-d, rounded (an int8 weight with its exponent raised by d)."""
    return torch.round(W.double() / 2.0 ** d).clamp(-127, 127).to(W.dtype)


def wit(M, sh, tokens):
    mdl.drop_caps(M)
    mdl.ensure_caps(M)
    return llm.LayerWit(M, sh, 0, M.emb[tokens.reshape(-1)].to(torch.int64)).moe


def pair(v, up=1):
    """A value a of v with a + up also in v (the most common such a)."""
    u, c = torch.unique(v, return_counts=True)
    ok = torch.isin(u + up, u)
    assert bool(ok.any())
    return int(u[ok][c[ok].argmax()])


def clamped_model(sh, tokens):
    """Layer 0 re-quantized so that each clamp has clamped rows and unclamped rows at the edge shift."""
    M = moe_tiny()
    Ly = M.layers[0]
    w = wit(M, sh, tokens)                        # nothing clamps yet: kr, kg, kd are the raw shifts
    dr = pair(w["kr"])                            # a -> 0 (clamped), a + 1 -> 1 (the edge)
    Ly.Wr, Ly.eWr = halve(Ly.Wr, dr), Ly.eWr + dr
    dg = pair(w["kg"])
    Ly.Wge = halve(Ly.Wge, dg)
    Ly.eWge = Ly.eWge + dg
    Ly.kcg = M.eG - Ly.eWge
    w = wit(M, sh, tokens)
    dd = 25 - pair(w["kd"], -1)                   # b -> 25 (clamped), b - 1 -> 24 (the edge)
    Ly.eWde = Ly.eWde - dd
    return M


def main():
    M0 = moe_tiny()
    sh = llm.Shp(M0.cfg, B, S, *llm.block_bits(M0.cfg), NGEN)
    tokens, targets = statement(M0, B, S, NGEN)
    M = clamped_model(sh, tokens)
    tokens, targets = statement(M, B, S, NGEN)         # greedy under the re-quantized layer
    w = wit(M, sh, tokens)
    for f in ("fr", "fg", "fd"):
        n, N = int(w[f].sum()), w[f].numel()
        print(f"layer 0: {f} = 1 on {n} of {N}")
        assert 0 < n < N, f
    ok, ref, got = fofast.check(M, B=2, S=300, ngen=12)
    assert ok, (ref, got)
    print("(a) serving engine equals evalfo, no row flagged")
    cw = llm.weights_digest(M, sh)
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
    print("(a) proof verifies")

    orig = moe.moe_wit
    counts = tb.check_counts

    def forge(kind):
        def f(*a, **k):
            r = orig(*a, **k)
            if a[1] is not M.layers[0]:
                return r
            flag, sh_, edge = {"router": ("fr", "kr", 1), "gate": ("fg", "kg", 1), "down": ("fd", "kd", 24)}[kind]
            rows = (r[sh_] == edge) & (r[flag] == 0)      # honest: unclamped with the edge shift
            assert bool(rows.any()), kind
            r[flag] = torch.where(rows, torch.ones_like(r[flag]), r[flag])
            return r
        return f
    for kind in ("router", "gate", "down"):
        moe.moe_wit = forge(kind)
        tb.check_counts = lambda *a, **k: None
        try:
            bad = llm.prove(M, sh, tokens, targets, C_W=cw)
        finally:
            moe.moe_wit, tb.check_counts = orig, counts
            tb._ERRS.clear()
        assert rejects(lambda: llm.verify(M, sh, tokens, targets, bad, C_W=cw))
        print(f"(b) unclamped {kind} row marked as clamped: rejected")


def test_moe_clamps():
    main()


if __name__ == "__main__":
    main()
