"""Exact integer causal attention of F_O for prefill, as a Triton kernel (two passes over the keys,
no score matrix in memory).  Same function as evalfo.Engine._attend:

  acc_ij = q_i . k_j (9-bit by 9-bit, dh = 128: |acc| < 2^22), M_i = max_{j <= p_i} acc_ij,
  P_ij = tau_exp(min(M_i - acc_ij, 2^23 - 1)) for j <= p_i (else 0), D_i = sum_j P_ij,
  N_i = sum_blocks ShiftRound(sum_{j in block} P_ij v_j, k_m(D_i)) over key blocks of 2^12,
  o_i = NormDiv(N_i, D_i).

Products are exact in fp16 x fp16 -> fp32: |q|, |k| <= 2^9, and every fp32 sum stays below 2^24
(scores < 2^22; P = 32 P_hi + P_lo with 5-bit halves, so a block of 2^12 keys sums to at most
2^12 * 31 * 127 < 2^24 per half)."""
import torch
import triton
import triton.language as tl
from .model import KEYBLK, RECIP_NUM, O_BASE, KM_MIN

BM, BN = 64, 64


@triton.jit
def _rowmax(Q, K, POS, MX, Sq, Sk, sqb, sqs, skb, sks, nh: tl.constexpr, G: tl.constexpr, DH: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // nh, bh % nh
    g = h // (nh // G)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < Sq
    d = tl.arange(0, DH)
    q = tl.load(Q + b * sqb + rows[:, None] * sqs + h * DH + d[None, :], mask=rmask[:, None], other=0).to(tl.float16)
    pos = tl.load(POS + b * Sq + rows, mask=rmask, other=-1)
    hi = tl.minimum(tl.max(pos, axis=0) + 1, Sk)
    m = tl.full((BM,), -1.0e30, tl.float32)
    for j0 in range(0, hi, BN):
        cols = j0 + tl.arange(0, BN)
        k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=(cols < Sk)[:, None],
                    other=0).to(tl.float16)
        s = tl.dot(q, tl.trans(k))
        valid = (cols[None, :] <= pos[:, None]) & (cols[None, :] < Sk)
        m = tl.maximum(m, tl.max(tl.where(valid, s, -1.0e30), axis=1))
    tl.store(MX + bh * Sq + rows, m, mask=rmask)


@triton.jit
def _pv(Q, K, V, POS, MX, TEXP, PVO, DO, Sq, Sk, NB, sqb, sqs, skb, sks, svb, svs, nh: tl.constexpr,
        G: tl.constexpr, DH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // nh, bh % nh
    g = h // (nh // G)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < Sq
    d = tl.arange(0, DH)
    q = tl.load(Q + b * sqb + rows[:, None] * sqs + h * DH + d[None, :], mask=rmask[:, None], other=0).to(tl.float16)
    pos = tl.load(POS + b * Sq + rows, mask=rmask, other=-1)
    m = tl.load(MX + bh * Sq + rows, mask=rmask, other=0.0)
    hi = tl.minimum(tl.max(pos, axis=0) + 1, Sk)
    D = tl.zeros((BM,), tl.int32)
    for kb in range(0, NB):
        ah = tl.zeros((BM, DH), tl.float32)
        al = tl.zeros((BM, DH), tl.float32)
        for j0 in range(kb * KB, tl.minimum((kb + 1) * KB, hi), BN):
            cols = j0 + tl.arange(0, BN)
            cm = cols < Sk
            k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=cm[:, None],
                        other=0).to(tl.float16)
            v = tl.load(V + b * svb + cols[:, None] * svs + g * DH + d[None, :], mask=cm[:, None],
                        other=0).to(tl.float16)
            s = tl.dot(q, tl.trans(k))
            valid = (cols[None, :] <= pos[:, None]) & cm[None, :]
            x = tl.minimum(m[:, None] - s, 8388607.0).to(tl.int32)
            x = tl.where(valid, x, 0)
            p = tl.load(TEXP + x)
            p = tl.where(valid, p, 0)
            D += tl.sum(p, axis=1)
            ah += tl.dot((p >> 5).to(tl.float16), v)
            al += tl.dot((p & 31).to(tl.float16), v)
        pvb = ah.to(tl.int32) * 32 + al.to(tl.int32)
        tl.store(PVO + (((bh * Sq + rows[:, None]) * NB + kb) * DH + d[None, :]), pvb, mask=rmask[:, None])
    tl.store(DO + bh * Sq + rows, D, mask=rmask)


@triton.jit
def _dec(Q, K, V, POS, TEXP, PVO, DO, Sk, NB, sqb, skb, sks, svb, svs, nh: tl.constexpr, G: tl.constexpr,
         DH: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """Decode: one query per sequence; program (b, g) takes the group's heads as rows (R >= heads per
    group, padded), so the cache is read once per group.  Both passes in one program."""
    bg = tl.program_id(0)
    b, g = bg // G, bg % G
    grp: tl.constexpr = nh // G
    r = tl.arange(0, R)
    rmask = r < grp
    d = tl.arange(0, DH)
    q = tl.load(Q + b * sqb + (g * grp + r[:, None]) * DH + d[None, :], mask=rmask[:, None], other=0).to(tl.float16)
    p0 = tl.load(POS + b)
    hi = tl.minimum(p0 + 1, Sk)
    m = tl.full((R,), -1.0e30, tl.float32)
    for j0 in range(0, hi, BN):
        cols = j0 + tl.arange(0, BN)
        k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=(cols < hi)[:, None],
                    other=0).to(tl.float16)
        s = tl.dot(q, tl.trans(k))
        m = tl.maximum(m, tl.max(tl.where((cols < hi)[None, :], s, -1.0e30), axis=1))
    D = tl.zeros((R,), tl.int32)
    for kb in range(0, NB):
        ah = tl.zeros((R, DH), tl.float32)
        al = tl.zeros((R, DH), tl.float32)
        for j0 in range(kb * KB, tl.minimum((kb + 1) * KB, hi), BN):
            cols = j0 + tl.arange(0, BN)
            cm = cols < hi
            k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=cm[:, None], other=0).to(tl.float16)
            v = tl.load(V + b * svb + cols[:, None] * svs + g * DH + d[None, :], mask=cm[:, None], other=0).to(tl.float16)
            s = tl.dot(q, tl.trans(k))
            x = tl.minimum(m[:, None] - s, 8388607.0).to(tl.int32)
            x = tl.where(cm[None, :], x, 0)
            p = tl.where(cm[None, :], tl.load(TEXP + x), 0)
            D += tl.sum(p, axis=1)
            ah += tl.dot((p >> 5).to(tl.float16), v)
            al += tl.dot((p & 31).to(tl.float16), v)
        pvb = ah.to(tl.int32) * 32 + al.to(tl.int32)
        tl.store(PVO + (((b * nh + g * grp + r[:, None]) * NB + kb) * DH + d[None, :]), pvb, mask=rmask[:, None])
    tl.store(DO + b * nh + g * grp + r, D, mask=rmask)


@triton.jit
def _dmax(Q, K, POS, MXP, Sk, NS, sqb, skb, sks, nh: tl.constexpr, G: tl.constexpr, DH: tl.constexpr,
          R: tl.constexpr, BN: tl.constexpr, CH: tl.constexpr):
    """Split decode, pass 1: partial row maxima of the group's heads over keys [s CH, (s+1) CH)."""
    bg, sp = tl.program_id(0), tl.program_id(1)
    b, g = bg // G, bg % G
    grp: tl.constexpr = nh // G
    r = tl.arange(0, R)
    d = tl.arange(0, DH)
    q = tl.load(Q + b * sqb + (g * grp + r[:, None]) * DH + d[None, :], mask=(r < grp)[:, None], other=0).to(tl.float16)
    hi = tl.minimum(tl.load(POS + b) + 1, Sk)
    m = tl.full((R,), -1.0e30, tl.float32)
    for j0 in range(sp * CH, tl.minimum((sp + 1) * CH, hi), BN):
        cols = j0 + tl.arange(0, BN)
        k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=(cols < hi)[:, None],
                    other=0).to(tl.float16)
        s = tl.dot(q, tl.trans(k))
        m = tl.maximum(m, tl.max(tl.where((cols < hi)[None, :], s, -1.0e30), axis=1))
    tl.store(MXP + (bg * NS + sp) * R + r, m)


@triton.jit
def _dpv(Q, K, V, POS, MX, TEXP, PVP, DP, Sk, NS, sqb, skb, sks, svb, svs, nh: tl.constexpr, G: tl.constexpr,
         DH: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, CH: tl.constexpr):
    """Split decode, pass 2: partial D and P V of keys [s CH, (s+1) CH) (inside one key block), with the
    row maxima of all splits.  Splits past the position write zeros."""
    bg, sp = tl.program_id(0), tl.program_id(1)
    b, g = bg // G, bg % G
    grp: tl.constexpr = nh // G
    r = tl.arange(0, R)
    d = tl.arange(0, DH)
    q = tl.load(Q + b * sqb + (g * grp + r[:, None]) * DH + d[None, :], mask=(r < grp)[:, None], other=0).to(tl.float16)
    m = tl.load(MX + bg * R + r)
    hi = tl.minimum(tl.load(POS + b) + 1, Sk)
    D = tl.zeros((R,), tl.int32)
    ah = tl.zeros((R, DH), tl.float32)
    al = tl.zeros((R, DH), tl.float32)
    for j0 in range(sp * CH, tl.minimum((sp + 1) * CH, hi), BN):
        cols = j0 + tl.arange(0, BN)
        cm = cols < hi
        k = tl.load(K + b * skb + cols[:, None] * sks + g * DH + d[None, :], mask=cm[:, None], other=0).to(tl.float16)
        v = tl.load(V + b * svb + cols[:, None] * svs + g * DH + d[None, :], mask=cm[:, None], other=0).to(tl.float16)
        s = tl.dot(q, tl.trans(k))
        x = tl.minimum(m[:, None] - s, 8388607.0).to(tl.int32)
        x = tl.where(cm[None, :], x, 0)
        p = tl.where(cm[None, :], tl.load(TEXP + x), 0)
        D += tl.sum(p, axis=1)
        ah += tl.dot((p >> 5).to(tl.float16), v)
        al += tl.dot((p & 31).to(tl.float16), v)
    tl.store(PVP + ((bg * NS + sp) * R + r[:, None]) * DH + d[None, :], ah.to(tl.int32) * 32 + al.to(tl.int32))
    tl.store(DP + (bg * NS + sp) * R + r, D)


def _bitlen(x):
    e = torch.frexp(x.to(torch.float64)).exponent.to(torch.int64)
    return torch.where(x > 0, e, torch.zeros_like(e))


def attend(rQ, rK, V, qpos, texp32, vshift, G):
    """rQ (B, Sq, nh, dh) int; rK (B, Sk, G, dh) int16 and V (B, Sk, G, dh) int8 (cache views: any batch
    and position strides); qpos (B, Sq) int64; texp32 the exp table as int32; vshift (nh,)
    -> o (B, Sq, nh, dh) int64.  Values are converted to fp16 in registers."""
    B, Sq, nh, dh = rQ.shape
    Sk = rK.shape[1]
    q = rQ if rQ.stride(-1) == 1 and rQ.stride(-2) == dh else rQ.contiguous()
    assert rK.stride(-1) == 1 and rK.stride(-2) == dh and V.stride(-1) == 1 and V.stride(-2) == dh
    pos = qpos.to(torch.int32).contiguous()
    NB = (Sk + (1 << KEYBLK) - 1) >> KEYBLK
    mx = torch.empty((B * nh, Sq), dtype=torch.float32, device=rQ.device)
    pvo = torch.empty((B * nh, Sq, NB, dh), dtype=torch.int32, device=rQ.device)
    Do = torch.empty((B * nh, Sq), dtype=torch.int32, device=rQ.device)
    grid = (triton.cdiv(Sq, BM), B * nh)
    _rowmax[grid](q, rK, pos, mx, Sq, Sk, q.stride(0), q.stride(1), rK.stride(0), rK.stride(1), nh, G, dh, BM, BN)
    _pv[grid](q, rK, V, pos, mx, texp32, pvo, Do, Sq, Sk, NB, q.stride(0), q.stride(1), rK.stride(0), rK.stride(1),
              V.stride(0), V.stride(1), nh, G, dh, BM, BN, 1 << KEYBLK)
    return _finish(pvo, Do, vshift, B, nh, Sq, dh)


def _finish(pvo, Do, vshift, B, nh, Sq, dh):
    D = Do.to(torch.int64)
    E = _bitlen(D) - 16
    delta = (D >> E.clamp(min=0)) << (-E).clamp(min=0)
    km = E.clamp(min=KM_MIN).unsqueeze(-1).unsqueeze(-1)
    pv = pvo.to(torch.int64)
    Ns = ((pv + ((1 << km) >> 1)) >> km).sum(dim=2)                              # (B nh, Sq, dh)
    vs = vshift.to(torch.int64).view(1, nh, 1).expand(B, nh, 1).reshape(B * nh, 1)
    recip = (RECIP_NUM // delta).unsqueeze(-1)
    kk = (O_BASE + E - E.clamp(min=KM_MIN)).unsqueeze(-1) + vs.unsqueeze(-1)
    xh = torch.div(Ns, 1 << 12, rounding_mode="floor")
    xl = Ns - (xh << 12)
    t = torch.div(xl * recip, 1 << 12, rounding_mode="floor")
    H = xh * recip + t
    kp = kk - 12
    o = (H + ((1 << kp) >> 1)) >> kp
    return o.view(B, nh, Sq, dh).permute(0, 2, 1, 3)


CH = 512          # keys per decode split (divides the key block, so a split lies in one block)


def attend_dec(rQ, rK, V, pos, texp32, vshift, G):
    """One decode query per sequence: rQ (B, nh, dh) at positions pos (B,); cache views rK (B, L, G, dh)
    int16, V int8 (keys j <= pos are valid) -> o (B, nh, dh) int64.  Keys are split into chunks of CH
    across programs; partial P V sums are exact integers (one key block each) added in int64."""
    B, nh, dh = rQ.shape
    Sk = rK.shape[1]
    grp = nh // G
    q = rQ.contiguous()
    R = max(16, triton.next_power_of_2(grp))
    NS = triton.cdiv(Sk, CH)
    NB = (Sk + (1 << KEYBLK) - 1) >> KEYBLK
    p32 = pos.to(torch.int32).contiguous()
    mxp = torch.empty((B * G, NS, R), dtype=torch.float32, device=rQ.device)
    _dmax[(B * G, NS)](q, rK, p32, mxp, Sk, NS, q.stride(0), rK.stride(0), rK.stride(1), nh, G, dh, R, BN, CH)
    mx = mxp.amax(dim=1).contiguous()
    pvp = torch.empty((B * G, NS, R, dh), dtype=torch.int32, device=rQ.device)
    dp = torch.empty((B * G, NS, R), dtype=torch.int32, device=rQ.device)
    _dpv[(B * G, NS)](q, rK, V, p32, mx, texp32, pvp, dp, Sk, NS, q.stride(0), rK.stride(0), rK.stride(1),
                      V.stride(0), V.stride(1), nh, G, dh, R, BN, CH)
    per = (1 << KEYBLK) // CH                                       # splits per key block
    pad = NB * per - NS
    pv = pvp.to(torch.int64)
    if pad:
        pv = torch.cat([pv, pv.new_zeros((B * G, pad, R, dh))], dim=1)
    pv = pv.view(B * G, NB, per, R, dh).sum(dim=2)[:, :, :grp]          # (B G, NB, grp, dh)
    pvo = pv.permute(0, 2, 1, 3).reshape(B * nh, 1, NB, dh)
    Do = dp.to(torch.int64).sum(dim=1)[:, :grp].reshape(B * nh, 1)
    return _finish(pvo, Do, vshift, B, nh, 1, dh).reshape(B, nh, dh)
