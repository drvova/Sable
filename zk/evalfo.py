"""A batched evaluation engine for F_O (accuracy and throughput experiments, not the prover).

It computes the same integer function as model.forward / model.generate, bit for bit, with
attention vectorized over heads and sequences instead of one call per (sequence, head):
  - prefill of B right-padded prompts (rows past a prompt's end do not affect earlier rows:
    attention is causal and every other operation is row-wise);
  - batched greedy decoding, each sequence at its own position, with integer KV caches
    (rK as int16, V as int8; key blocks of 2^12 stay at absolute positions).
check() compares it with model.forward / model.generate."""
import torch
from . import model as mdl
torch.backends.cuda.matmul.allow_tf32 = False      # float32 matmuls must be exact (see _attend)
from .model import (requant8, normscale, imm, rope8, gshiftround, satadd, shiftround,
                    norm_den, norm_div, final_logits, KEYBLK)


class Engine:
    def __init__(self, M, Smax):
        self.M = M
        cfg = M.cfg
        for Ly in M.layers:                     # weights once in the layout cuBLAS int8 prefers
            for n in ("Wqkv", "Wo", "Wg", "Wu", "Wd"):
                if n not in Ly.__dict__:                # MoE layers keep their expert stacks as they are
                    continue
                W = getattr(Ly, n)
                if W.device.type != "cuda":
                    W = W.cuda()
                setattr(Ly, n, W.T.contiguous().T)
        M.Wlm = M.Wlm.T.contiguous().T
        self.cos, self.sin = mdl.rope_tables(cfg, Smax)
        self.Smax = Smax

    # ------------------------------------------------------------------ attention
    def _attend(self, Ly, rQ, rK, V, qpos, texp, kmax):
        """rQ (B, Sq, nh, dh) at positions qpos (B, Sq); keys rK (B, Sk, nkv, dh) int16, V int8;
        key j of sequence b is valid for a query at position p iff j <= p; kmax = max position + 1.
        Exact in float32: |q k| <= 128 * 180^2 < 2^24, and P V is split as P = 32 P_hi + P_lo with
        every partial sum of a 2^12-key block below 32 * 127 * 2^12 < 2^24.  -> o (B, Sq, nh, dh)."""
        cfg = self.M.cfg
        nh, G, dh = cfg.nh, cfg.nkv, cfg.dh
        grp = nh // G
        B, Sq = rQ.shape[:2]
        Sk = rK.shape[1]
        out = torch.empty((B, Sq, nh, dh), dtype=torch.int64, device="cuda")
        jj = torch.arange(Sk, device="cuda")
        vsh = Ly.vshift.view(G, grp)
        step = max(1, int(2e8) // max(1, G * grp * Sq * Sk))          # sequences per chunk
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            nb = b1 - b0
            q = rQ[b0:b1].view(nb, Sq, G, grp, dh).permute(0, 2, 3, 1, 4).reshape(nb * G, grp * Sq, dh)
            k = rK[b0:b1].permute(0, 2, 1, 3).reshape(nb * G, Sk, dh)
            acc = torch.bmm(q.float(), k.float().transpose(1, 2)).to(torch.int64)    # (nb G, grp Sq, Sk)
            pos = qpos[b0:b1].view(nb, 1, 1, Sq).expand(nb, G, grp, Sq).reshape(nb * G, grp * Sq)
            mask = jj.view(1, 1, -1) <= pos.unsqueeze(-1)
            Mx = torch.where(mask, acc, torch.full_like(acc, -(1 << 40))).max(dim=2).values
            x = (Mx.unsqueeze(-1) - acc).clamp(0, (1 << 23) - 1)
            del acc
            Pm = torch.where(mask, texp[x], torch.zeros_like(x))
            del x, mask
            nd = norm_den(Pm.sum(dim=2).reshape(-1))
            km = nd["km"].view(nb * G, grp * Sq, 1)
            Vf = V[b0:b1].permute(0, 2, 1, 3).reshape(nb * G, Sk, dh).float()
            Ph, Pl = (Pm >> 5).float(), (Pm & 31).float()
            del Pm
            Ns = torch.zeros((nb * G, grp * Sq, dh), dtype=torch.int64, device="cuda")
            for kb in range((kmax + (1 << KEYBLK) - 1) >> KEYBLK):
                j0, j1 = kb << KEYBLK, min(Sk, (kb + 1) << KEYBLK)
                pv = (32 * torch.bmm(Ph[:, :, j0:j1], Vf[:, j0:j1]).to(torch.int64)
                      + torch.bmm(Pl[:, :, j0:j1], Vf[:, j0:j1]).to(torch.int64))
                Ns += shiftround(pv, km)
            vs = vsh.view(1, G, grp, 1).expand(nb, G, grp, Sq).reshape(-1)
            o, _ = norm_div(Ns.view(-1, dh), nd, vs)
            out[b0:b1] = o.view(nb, G, grp, Sq, dh).permute(0, 3, 1, 2, 4).reshape(nb, Sq, nh, dh)
        return out

    def _layer(self, Ly, h, B, S, pos, cache, wpos, kmax):
        """h (B*S, d) rows at positions pos (B, S); cache (rK, V) of (B, Lmax, nkv, dh): the new
        keys are written at wpos (B, S) before attending."""
        M, cfg = self.M, self.M.cfg
        nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        half = dh // 2
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        qkv = mdl.qkv_out(cfg, Ly, imm(qp, Ly.Wqkv), Eh)
        Qh = qkv[:, :nq].reshape(B, S, nh, dh)
        Kh = qkv[:, nq:nq + nk].reshape(B, S, nkv, dh)
        Vh = qkv[:, nq + nk:].reshape(B, S, nkv, dh)
        c_ = self.cos[pos].view(B, S, 1, half)
        s_ = self.sin[pos].view(B, S, 1, half)
        rq1, rq2 = rope8(Qh[..., :half], Qh[..., half:], c_, s_)
        rk1, rk2 = rope8(Kh[..., :half], Kh[..., half:], c_, s_)
        rQ = torch.cat([rq1, rq2], dim=-1)
        rK = torch.cat([rk1, rk2], dim=-1)
        bi = torch.arange(B, device="cuda").view(B, 1).expand(B, S)
        cache["rK"][bi, wpos] = rK.to(torch.int16)
        cache["V"][bi, wpos] = Vh.to(torch.int8)
        o = self._attend(Ly, rQ, cache["rK"][:, :kmax], cache["V"][:, :kmax], pos, mdl.exp_table(Ly.E, dh), kmax)
        o = o.reshape(B * S, nq)
        qo, eo, _ = requant8(o)
        acc = imm(qo, Ly.Wo)
        h, _ = satadd(h, gshiftround(acc, (Ly.Kco - eo).unsqueeze(-1)))
        q, e, _ = requant8(h)
        qp, Eh, _ = normscale(q)
        y, _, _ = mdl.mlp_out(M, Ly, qp, Eh)
        h, _ = satadd(h, y)
        return h

    # ------------------------------------------------------------------ API
    def new_caches(self, B, L):
        cfg = self.M.cfg
        return [dict(rK=torch.zeros((B, L, cfg.nkv, cfg.dh), dtype=torch.int16, device="cuda"),
                     V=torch.zeros((B, L, cfg.nkv, cfg.dh), dtype=torch.int8, device="cuda"))
                for _ in self.M.layers]

    def prefill(self, toks, caches):
        """toks (B, S) right-padded prompts; fills positions 0..S-1 of the caches.
        Returns the final hidden rows (B, S, d) (rows past a prompt's end are meaningless)."""
        B, S = toks.shape
        pos = torch.arange(S, device="cuda").view(1, S).expand(B, S).contiguous()
        h = self.M.emb[toks.reshape(-1).cuda()].to(torch.int64)
        with mdl.deferred_checks():
            for Ly, cache in zip(self.M.layers, caches):
                h = self._layer(Ly, h, B, S, pos, cache, pos, S)
        return h.view(B, S, -1)

    def logits(self, hrows):
        return final_logits(self.M, hrows)

    def generate(self, prompts, ngen, stop=None):
        """Greedy decoding of a batch of prompts (lists of token ids) for at most ngen tokens.
        stop(b, ids) -> bool ends sequence b early.  Returns lists of generated ids."""
        B = len(prompts)
        lens = torch.tensor([len(p) for p in prompts], device="cuda")
        S = int(lens.max())
        L = S + ngen
        assert L <= self.Smax
        toks = torch.zeros((B, S), dtype=torch.int64)
        for b, p in enumerate(prompts):
            toks[b, :len(p)] = torch.tensor(p)
        caches = self.new_caches(B, L)
        h = self.prefill(toks.cuda(), caches)
        last = h[torch.arange(B, device="cuda"), lens - 1]
        z = self.logits(last)
        out = [[] for _ in range(B)]
        done = [False] * B
        pos = lens.clone()
        kmax = S
        for t in range(ngen):
            y = torch.argmax(z, dim=1)                       # first index attaining the max
            yl = y.tolist()
            for b in range(B):
                if not done[b]:
                    out[b].append(yl[b])
                    if stop is not None and stop(b, out[b]):
                        done[b] = True
            if all(done) or t == ngen - 1:
                break
            h = self.M.emb[y].to(torch.int64)
            p = pos.view(B, 1)
            kmax = S + t + 1
            with mdl.deferred_checks():
                for Ly, cache in zip(self.M.layers, caches):
                    h = self._layer(Ly, h, B, 1, p, cache, p, kmax)
                z = self.logits(h)
            pos = pos + 1
        return out


def check(M, n=3, S=300, ngen=8, seed=0):
    """The engine equals model.forward (logits of every row) and model.generate (greedy tokens)."""
    g = torch.Generator().manual_seed(seed)
    eng = Engine(M, 4096)
    prompts = [torch.randint(0, M.cfg.V, (S - 37 * i,), generator=g) for i in range(n)]
    ref = [mdl.generate(M, p, ngen, 4096).tolist() for p in prompts]
    got = eng.generate([p.tolist() for p in prompts], ngen)
    hz = mdl.forward(M, prompts[0].view(1, -1))
    caches = eng.new_caches(1, S)
    he = eng.prefill(prompts[0].view(1, -1).cuda(), caches)[0]
    same_rows = torch.equal(mdl.final_logits(M, hz), mdl.final_logits(M, he))
    return ref == got and same_rows, ref, got


def first_token_envelope(M, chunk=8192):
    """Every vocabulary token as the first token of a prompt (position 0, which attends only to itself):
    the largest real gate input and SiLU output and the largest |GShiftRound output| / 2^A_BITS over the
    layers, with c and the shifts clamped as in calibration.  -> dict (model.ENV)."""
    import copy
    M2 = copy.copy(M)                             # the engine relayouts weights: keep M's tensors as they are
    M2.layers = [copy.copy(Ly) for Ly in M.layers]
    eng = Engine(M2, 2)
    mdl.ENV.clear()
    mdl._CALIB[0] = True
    try:
        for t0 in range(0, M.cfg.V, chunk):
            toks = torch.arange(t0, min(M.cfg.V, t0 + chunk), device="cuda").view(-1, 1)
            eng.prefill(toks, eng.new_caches(toks.shape[0], 1))
    finally:
        mdl._CALIB[0] = False
    return dict(mdl.ENV)
