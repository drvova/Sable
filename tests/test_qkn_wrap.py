"""Regression for the QK-norm head-row ShiftQ8 wrap (found in the Qwen3 audit): with the unscaled minimality
relation 4 asel - 255 s, an exponent up to 22 let the relation reach 2^30.4, so a prover could claim e = 22
for any head row with limbs (4 asel - 255 2^22) mod p and the verifier accepted.  The head rows now take
the quartered relation (e >= 2, asel - 255 s / 4 in two 14-bit limbs).  The same forgery (every Q / K head
row at e = 22, limbs from the residue mod p) is rejected, and changes the generated tokens, so it is not a
vacuous rejection."""
import torch
from zk import llm, model as mdl, tables as tb
from zk.field import P
from tests.test_llm_qwen3 import qwen3_tiny
from tests.test_llm import statement
from tests.test_wrap_bounds import rejects

B, S, NG = 2, 256, 16
_rowq, _limbs = mdl._rowq, llm._limbs
_qkv_out, _qkw = mdl.qkv_out, llm.LayerWit.qk_norm_wit
EVIL, HEAD = [False], [False]


def evil_rowq(a, emin, emax):
    q, e, w = _rowq(a, emin, emax)
    if not (EVIL[0] and HEAD[0]):
        return q, e, w
    a2 = a.reshape(-1, a.shape[-1]).to(torch.int64)
    s = 1 << emax
    q2 = torch.div(2 * a2 + s, 2 * s, rounding_mode="floor")
    e2 = torch.full((a2.shape[0],), emax, dtype=torch.int64, device=a.device)
    ones = torch.ones_like(e2)
    w2 = dict(jstar=torch.zeros_like(e2), sigma=ones, f=ones, asel=a2[:, 0])
    return q2.reshape(q.shape).to(q.dtype), e2.reshape(e.shape), w2


def evil_limbs(f, sig, asel, s, quarter=False):
    if not (EVIL[0] and HEAD[0]):
        return _limbs(f, sig, asel, s, quarter)
    w = (f * (sig * (asel - 255 * (s >> 2)) + (1 - sig) * (-asel - 255 * (s >> 2) - 1))) % P
    return w & 16383, w >> 14


def head(fn):
    def g(*a, **k):
        HEAD[0] = True
        try:
            return fn(*a, **k)
        finally:
            HEAD[0] = False
    return g


def main():
    cfg = mdl.Cfg(L=2, d=256, nh=2, nkv=1, dh=128, dff=384, V=1024, qk_norm=True, qkv_bias=False)
    M = qwen3_tiny(cfg)
    sh = llm.Shp(M.cfg, B, S, 7, 7, NG)
    tok_h, tgt_h = statement(M, B, S, NG)
    mdl._rowq, llm._limbs = evil_rowq, evil_limbs
    mdl.qkv_out, llm.LayerWit.qk_norm_wit = head(_qkv_out), head(_qkw)
    cc, chk = tb.check_counts, mdl.check
    tb.check_counts = lambda *a, **k: None
    mdl.check = lambda *a, **k: None
    EVIL[0] = True
    try:
        tok_e, tgt_e = statement(M, B, S, NG)
        msgs = llm.prove(M, sh, tok_e, tgt_e)
    finally:
        EVIL[0] = False
        mdl._rowq, llm._limbs = _rowq, _limbs
        mdl.qkv_out, llm.LayerWit.qk_norm_wit = _qkv_out, _qkw
        tb.check_counts, mdl.check = cc, chk
        tb._ERRS.clear()
    n = int((tgt_h != tgt_e).sum())
    print(f"forged head rows at e = 22: generated tokens differ from F_O in {n} of {tgt_h.numel()}")
    assert n > 0
    assert rejects(lambda: llm.verify(M, sh, tok_e, tgt_e, msgs))
    print("forged QK-norm head-row exponents: rejected")


def test_qkn_wrap():
    main()


if __name__ == "__main__":
    main()
