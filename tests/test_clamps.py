"""Row clamps and the 9-bit down input on a tiny Qwen3 model (QK-norm, down_b = 255):
(a) the down input reaches |q_m| > 127; layer 0's QK-norm shifts kn + E2 fall below 1 for some head rows (kn lowered) and its gate bound K_gl
    exceeds some rows' E_h (K_gl raised): F_O shifts those rows by max(kn + E2, 1) and max(E_h, K_gl), the
    serving engine equals evalfo, and the proof verifies;
(b) a prover that marks an unclamped row as clamped (kn + E2 = 1 with fq = 1, or E_h = K_gl with f = 1,
    the same shift either way) is rejected by the range lookup of the flag's branch."""
import torch
from zk import model as mdl, fofast, llm
from zk import tables as tb
from tests.test_llm import statement
from tests.test_llm_qwen3 import qwen3_tiny
from tests.test_wrap_bounds import rejects

B, S, NGEN = 2, 256, 16
CFG = mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True, qkv_bias=False, down_b=255)


def clamped_model():
    M = qwen3_tiny(CFG)
    mdl.ensure_caps(M)
    Ly = M.layers[0]
    Ly.kn = torch.zeros_like(Ly.kn)                 # kn + E2 = E2 (0, 1 or 2 here): rows with E2 = 0 clamp
    Ly.kgl = 2                                      # E_h is 0, 1 or 2 here: rows below 2 are raised to it
    Ly.kgl_t = torch.tensor(Ly.kgl, dtype=torch.int64, device=Ly.kgl_t.device)
    return M


def witness_flags(M, sh, tokens):
    W = llm.LayerWit(M, sh, 0, M.emb[tokens.reshape(-1)].to(torch.int64))
    fq = torch.cat([W.qn[X]["fq"].reshape(-1) for X in ("qn", "kn")])
    return W, fq, W.eg["f"].reshape(-1)


def main():
    M = clamped_model()
    sh = llm.Shp(M.cfg, B, S, 7, 7, NGEN)
    cw = llm.weights_digest(M, sh)
    tokens, targets = statement(M, B, S, NGEN)
    W, fq, fg = witness_flags(M, sh, tokens)
    print(f"layer 0: {int(fq.sum())} of {fq.numel()} head rows with kn + E2 < 1, "
          f"{int(fg.sum())} of {fg.numel()} rows with E_h < K_gl = {M.layers[0].kgl}")
    assert 0 < int(fq.sum()) < fq.numel() and 0 < int(fg.sum()) < fg.numel()
    qm = int(W.rm["q"].abs().max())
    assert 127 < qm <= 255, qm
    print(f"layer 0: largest |q_m| of the down input {qm} (9-bit)")
    ok, ref, got = fofast.check(M, B=2, S=300, ngen=12)
    assert ok, (ref, got)
    print("(a) serving engine equals evalfo, no row flagged")
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
    print("(a) proof verifies")

    orig = llm.LayerWit.__init__
    check, counts = mdl.check, tb.check_counts

    def forge(kind):
        def init(self, M_, sh_, l, hin, *a, **k):
            orig(self, M_, sh_, l, hin, *a, **k)
            if l != 0:
                return
            if kind == "qk":
                Q = self.qn["qn"]
                rows = (Q["knr"] + Q["n"]["Eh"]) == 1      # honest: fq = 0, kq = 1
                assert bool(rows.any())
                Q["fq"] = torch.where(rows, torch.ones_like(Q["fq"]), Q["fq"])
            else:
                rows = self.n2["Eh"] == M_.layers[0].kgl     # honest: f = 0, E_g = K_gl
                assert bool(rows.any())
                self.eg["f"] = torch.where(rows, torch.ones_like(self.eg["f"]), self.eg["f"])
        return init
    for kind in ("qk", "gate"):
        llm.LayerWit.__init__ = forge(kind)
        tb.check_counts = lambda *a, **k: None
        try:
            bad = llm.prove(M, sh, tokens, targets, C_W=cw)
        finally:
            llm.LayerWit.__init__, mdl.check, tb.check_counts = orig, check, counts
            tb._ERRS.clear()
        assert rejects(lambda: llm.verify(M, sh, tokens, targets, bad, C_W=cw))
        print(f"(b) unclamped {kind} row marked as clamped: rejected")


def test_clamps():
    main()


if __name__ == "__main__":
    main()
