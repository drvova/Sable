"""Regressions for the audit of 2026-10-26 on the tiny model:
(a) ShiftQ8's minimality witness no longer wraps modulo p: a prover that quantizes up-projection rows
    at exponent 22 (with a wrapped limb witness) is rejected;
(b) the per-row bounds Kqs (ShiftQ8S) and Kgm (gate shift) are enforced: a verifier whose bound is
    below the honest E_h rejects;
(c) the model constants are bound by the statement: a verifier model that differs only in one bias
    (same C_W) rejects the proof;
(d) tokens outside the vocabulary are refused;
(e) fit_qkv_exponents lifts every layer's K_qs to e_hmax(d), so the bound never binds, and the
    refitted model still proves and verifies;
(f) NormScale's left shift is bounded by (16, [E < 0] Sigma) (a review of the MLSys submission): a
    prover that normalizes a row with Sigma = 260100 at E = -14 (2^14 Sigma = 2p + 65534, so the
    field relation holds with delta = 65534) is rejected, and the same proof verifies once the
    bound is removed from both sides, so the bound is what rejects it.
    tests/test_norm_bounds.py covers this and the attention denominator gadget by gadget."""
import copy
import torch
from zk import llm
from zk import model as mdl
from zk import tables as tb
from zk.field import P
from zk.llm import NS_F, f32, R_blk, R_virt
from tests.test_llm import qwen_tiny

B, S, NGEN = 1, 256, 16
_rowq, _limbs = mdl._rowq, llm._limbs
_normscale, _ns_rows = mdl.normscale, llm.Proto.ns_rows
SIG, EN = 260100, 14                          # a Sigma of layer 1's attention input at seed 598
FORGED = []


def evil_rowq(a, emin, emax):
    """ShiftQ8 of the up projection: every second row at exponent 22, the most negative element
    selected (the audit's forgery)."""
    q, e, w = _rowq(a, emin, emax)
    if emin != 2:
        return q, e, w
    a2 = a.reshape(-1, a.shape[-1])
    q2, e2 = q.reshape(a2.shape), e.reshape(-1)
    rows = torch.arange(0, a2.shape[0], 2, device=a.device)
    s = 1 << 22
    q2[rows] = torch.div(2 * a2[rows].to(torch.int64) + s, 2 * s, rounding_mode="floor").to(q2.dtype)
    e2[rows] = 22
    js = torch.argmin(a2[rows], dim=1)
    w["jstar"].reshape(-1)[rows] = js.to(w["jstar"].dtype)
    w["asel"].reshape(-1)[rows] = a2[rows].gather(1, js.unsqueeze(1)).squeeze(1).to(w["asel"].dtype)
    w["sigma"].reshape(-1)[rows] = 1
    w["f"].reshape(-1)[rows] = 1
    return q2.reshape(q.shape), e2.reshape(e.shape), w


def evil_limbs(f, sig, asel, s, quarter=False):
    """The cheater's limbs: w mod p, clamped to the limb range (the fixed gadget has no valid ones)."""
    if quarter:
        w = (f * (sig * (asel - 255 * (s >> 2)) + (1 - sig) * (-asel - 255 * (s >> 2) - 1))) % P
        w = w.clamp(0, (1 << 28) - 1)
        return w & 16383, w >> 14
    w = (f * (sig * (4 * asel - 255 * s) + (1 - sig) * (-4 * asel - 255 * s - 1))) % P
    return w & 32767, w >> 15


def evil_normscale(q):
    """NormScale with the rows of sum q^2 + 1 = SIG left-shifted by EN: delta = 2^EN SIG mod p."""
    qp, Eh, w = _normscale(q)
    m = w["sig"] == SIG
    if m.any():
        d = (w["sig"][m] << EN) % P
        Eh[m], w["delta"][m], w["rho"][m], w["Ep"][m], w["En"][m] = -EN // 2, d, 0, 0, EN
        w["mu"][m] = mu = mdl.rsqrt_table()[d]
        qp[m] = mdl.fdiv(q[m] * mu.unsqueeze(-1) + (1 << 14), 1 << 15)
        FORGED.append(int(m.sum()))
    return qp, Eh, w


def ns_rows_unbounded(self, db, pre, getn, sig_key):
    """Proto.ns_rows without the lookup (16, ng Sigma)."""
    ctx = self.ctx
    c = {f: db.col(pre + f, (lambda f=f: f32(getn()[f])), R_blk(ctx, pre + f)) for f in NS_F}
    sig = db.col(pre + "sig", lambda: f32(getn()["sig"]), R_virt(ctx, sig_key))
    d = db.dom
    Eh, Ep, pEp, pEn, pEh, delta, rho, mu, ng = (c[x] for x in NS_F)
    d.lookup(tb.NPOW, 2 * Eh + 15, Ep, pEp, pEn, 1, ng)
    d.lookup(tb.POW, Eh + tb.POW_OFF, pEh)
    d.lookup(Ep, rho)
    d.lookup(tb.RSQ, delta - tb.RSQ_LO, mu)
    d.zero(pEn * sig - pEp * delta - rho)
    return c


def rejects(fn):
    try:
        fn()
    except (ValueError, AssertionError) as e:
        print("  rejected:", str(e)[:90])
        return True
    return False


def main():
    M = qwen_tiny()
    sh = llm.Shp(M.cfg, B, S, 7, 7, NGEN)
    cw = llm.weights_digest(M, sh)
    tokens0 = torch.randint(0, M.cfg.V, (B, S), generator=torch.Generator().manual_seed(2)).cuda()
    # (a) the ShiftQ8 forgery
    mdl._rowq, llm._limbs = evil_rowq, evil_limbs
    try:
        tok_e, tgt_e = llm.greedy_decode(M, tokens0, NGEN)
        try:
            msgs = llm.prove(M, sh, tok_e, tgt_e, C_W=cw)
        except Exception as e:                  # noqa: BLE001  (the prover's own checks may stop it)
            msgs = None
            print("  forging prover stopped:", str(e)[:90])
    finally:
        mdl._rowq, llm._limbs = _rowq, _limbs
    assert msgs is None or rejects(lambda: llm.verify(M, sh, tok_e, tgt_e, msgs, C_W=cw))
    print("(a) ShiftQ8 forgery at exponent 22: rejected")
    # an honest proof for the other checks
    tokens, targets = llm.greedy_decode(M, tokens0, NGEN)
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw)
    print("honest proof verifies")
    # (b) bounds enforced: prover and verifier both use a cap below every E_h, the prover's own
    # checks (witness bound, table membership) are bypassed, and the range lookup (5, K - E_h) must reject
    mdl.ensure_caps(M)
    check, counts = mdl.check, tb.check_counts
    for attr in ("kqs", "kgm"):
        saved = [(getattr(Ly, attr), getattr(Ly, attr + "_t")) for Ly in M.layers]
        for Ly in M.layers:
            setattr(Ly, attr, -8)
            setattr(Ly, attr + "_t", torch.full_like(getattr(Ly, attr + "_t"), -8))
        mdl.check = lambda c, msg="witness check": None if "bound" in msg else check(c, msg)
        tb.check_counts = lambda *a, **k: None
        try:
            bad = llm.prove(M, sh, tokens, targets, C_W=cw)
            mdl.check, tb.check_counts = check, counts
            assert rejects(lambda: llm.verify(M, sh, tokens, targets, bad, C_W=cw))
        finally:
            mdl.check, tb.check_counts = check, counts
            tb._ERRS.clear()                     # the bypassed proofs' out-of-table records
            for Ly, (v, vt) in zip(M.layers, saved):
                setattr(Ly, attr, v)
                setattr(Ly, attr + "_t", vt)
        print(f"(b) per-row bound {attr} below E_h: rejected")
    # (c) a verifier model with one bias changed (same C_W)
    M2 = copy.copy(M)
    M2.layers = list(M.layers)
    L0 = copy.copy(M.layers[0])
    L0.bias = M.layers[0].bias.clone()
    L0.bias[0] += 1
    M2.layers[0] = L0
    mdl.drop_caps(M2)
    assert rejects(lambda: llm.verify(M2, sh, tokens, targets, msgs, C_W=cw))
    print("(c) model constant changed (same C_W): rejected")
    # (d) a token outside the vocabulary
    bad = tokens.clone()
    bad[0, 0] = M.cfg.V
    assert rejects(lambda: llm.verify(M, sh, bad, targets, msgs, C_W=cw))
    print("(d) token outside the vocabulary: rejected")
    # (e) columns whose bound is below e_hmax: lower their weight exponents by 12, then refit
    M3 = qwen_tiny()
    E = mdl.e_hmax(M3.cfg.d)
    L1 = M3.layers[1]
    cols = torch.topk(L1.kc, 12).indices.to(L1.eWqkv.device)     # the columns with the largest kappa
    L1.eWqkv[cols] -= 12
    L1.kc = L1.eout.to(L1.eWqkv.device) - L1.eWqkv
    assert mdl.qkv_kcap(M3.cfg, L1) < E
    assert mdl.fit_qkv_exponents(M3) == cols.numel()
    assert all(mdl.qkv_kcap(M3.cfg, Ly) >= E for Ly in M3.layers)
    sh3 = llm.Shp(M3.cfg, B, S, 7, 7, NGEN)
    tok3, tgt3 = llm.greedy_decode(M3, tokens0, NGEN)
    cw3 = llm.weights_digest(M3, sh3)
    llm.verify(M3, sh3, tok3, tgt3, llm.prove(M3, sh3, tok3, tgt3, C_W=cw3), C_W=cw3)
    print(f"(e) refit: K_qs >= e_hmax = {E} in every layer, proof verifies")
    # (f) the NormScale forgery; layer 1's smallest static QKV shift is raised from 7 to 8 so that
    # E_h = -7 keeps every shift kappa_n + E_h >= 1 (ShiftQ8S's range), as in Qwen2.5-32B
    M4 = qwen_tiny()
    L1 = M4.layers[1]
    L1.eWqkv[(L1.kc == 7).nonzero().flatten().to(L1.eWqkv.device)] -= 1
    L1.kc = L1.eout.to(L1.eWqkv.device) - L1.eWqkv
    mdl.drop_caps(M4)
    cw4 = llm.weights_digest(M4, sh)
    tokens4 = torch.randint(0, M4.cfg.V, (B, S), generator=torch.Generator().manual_seed(598)).cuda()
    mdl.normscale = evil_normscale
    try:
        tok4, tgt4 = llm.greedy_decode(M4, tokens4, NGEN)
        assert FORGED, "no row with the forged Sigma"
        for rows, bounded in ((ns_rows_unbounded, False), (_ns_rows, True)):
            llm.Proto.ns_rows = rows
            mdl.check, tb.check_counts = (lambda c, msg="": None), (lambda *a, **k: None)
            try:
                bad = llm.prove(M4, sh, tok4, tgt4, C_W=cw4)
            finally:
                mdl.check, tb.check_counts = check, counts
                tb._ERRS.clear()
            if bounded:
                assert rejects(lambda: llm.verify(M4, sh, tok4, tgt4, bad, C_W=cw4))
            else:
                llm.verify(M4, sh, tok4, tgt4, bad, C_W=cw4)
    finally:
        mdl.normscale, llm.Proto.ns_rows = _normscale, _ns_rows
    print(f"(f) NormScale left shift of Sigma = {SIG} at E = -{EN}: verifies without (16, ng Sigma), "
          "rejected with it")


def test_wrap_bounds():
    main()


if __name__ == "__main__":
    main()
