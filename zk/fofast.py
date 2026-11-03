"""A serving engine for F_O: the integer function of model.forward / model.generate, bit for bit.
Each layer is a compiled pre-attention part (torch.compile fuses the integer element-wise work), the
exact integer attention kernels of foattn (Triton; reading the int16/int8 caches in place), and a
compiled post-attention part.  The decode step has fixed shapes and no host synchronization and is
replayed as one CUDA graph per step.

Covers Qwen2 (dense), Qwen3 (QK-norm) and Qwen3-MoE layers; the experts go through the exact grouped
GEMM of the prover's MoE witness (moe.expert_layout / grouped_imm), which needs no host reads.

Differences from evalfo.Engine, none of which changes the function where it is defined:
  - every primitive is a closed form without host reads (evalfo's helpers check ranges on the host);
    instead of stopping, the engine flags every row that leaves a range on which F_O is defined
    (the conditions model.forward checks), per sequence for the prefill and per decode step, and
    clamps so that the computation continues; check() compares the engine with evalfo;
  - SiLU is the cached table of model.silu_table (|c| <= SILU_CMAX is a flagged condition);
  - dense gate and up projections are one GEMM over the concatenated weights, with int32
    accumulation (exact: |sum| <= 27648 * 127^2 < 2^31)."""
import torch
from . import model as mdl
from .model import A_BITS
from .evalfo import Engine


def _bitlen(x):
    """bit length of x >= 0 (x < 2^53), as int64."""
    e = torch.frexp(x.to(torch.float64)).exponent.to(torch.int64)
    return torch.where(x > 0, e, torch.zeros_like(e))


def rowq(a, emin, emax):
    """requant8 / shiftq8: least e in [emin, emax] with 2a in [-255 2^e, 255 2^e) on the row,
    q = floor((2a + 2^e) / 2^(e+1)).  -> (q, e)"""
    t = torch.maximum(2 * a.amax(dim=-1) + 1, -2 * a.amin(dim=-1))
    u = (t + 254) // 255
    e = _bitlen(u - 1).clamp(emin, emax)
    q = (2 * a + (1 << e).unsqueeze(-1)) >> (e + 1).unsqueeze(-1)
    return q, e


def normscale(q):
    sig = (q * q).sum(dim=-1) + 1
    Eh = torch.div(_bitlen(sig) - 15, 2, rounding_mode="floor")
    E = 2 * Eh
    delta = (sig >> E.clamp(min=0)) << (-E).clamp(min=0)
    mu = mdl.rsqrt_table()[delta]
    qp = (q * mu.unsqueeze(-1) + (1 << 14)) >> 15
    return qp, Eh


def shiftround(acc, k):
    return (acc + ((1 << k) >> 1)) >> k


def gshiftround(acc, k):
    kk = k.clamp(min=0)
    return torch.where(k > 0, (acc + ((1 << kk) >> 1)) >> kk, acc << (-k).clamp(min=0))


def satadd(h, c):
    return (h + c).clamp(-(1 << A_BITS), (1 << A_BITS) - 1)


def silu_tab(c, tab):
    """SiLU by the table of model.silu_table: F_O requires |c| <= SILU_CMAX (flagged by the caller)."""
    return tab[c.clamp(-mdl.SILU_CMAX, mdl.SILU_CMAX) + mdl.SILU_CMAX]


def out_of(x, lo, hi):
    """Per row: some element of x (rows, ...) lies outside [lo, hi]."""
    x = x.reshape(x.shape[0], -1)
    return (x < lo).any(dim=1) | (x > hi).any(dim=1)


def gsr(acc, k):
    """GShiftRound with its F_O conditions: k in [KMIN_SR, 24], output in A. -> (c, bad rows)."""
    bad = out_of(k, mdl.KMIN_SR, 24)
    k = k.clamp(mdl.KMIN_SR, 24)
    c = gshiftround(acc, k)
    return c, bad | out_of(c, -(1 << A_BITS) + 1, (1 << A_BITS) - 1)


def rowq_b(a, B, emax):
    """model._rowq_b: least e in [0, emax] with every a in [-(B + 1/2) 2^e, (B + 1/2) 2^e), q = round(a / 2^e)."""
    m = torch.maximum(2 * a.amax(dim=-1) + 1, -2 * a.amin(dim=-1))
    e = _bitlen((m + 2 * B) // (2 * B + 1) - 1).clamp(0, emax)
    q = (2 * a + (1 << e).unsqueeze(-1)) >> (e + 1).unsqueeze(-1)
    return q, e


def mm(a, Wt):
    """a (m, k) int8-valued, Wt column-major int8 (k, n) -> int32 (m, n); rows padded to 32 (cuBLASLt)."""
    m = a.shape[0]
    if m < 32:
        a = torch.cat([a, a.new_zeros((32 - m, a.shape[1]))])
        return torch._int_mm(a.to(torch.int8), Wt)[:m]
    return torch._int_mm(a.to(torch.int8), Wt)


class FastEngine:
    def __init__(self, M, Smax, B, compile=True):
        from . import moe
        self.moe = moe
        self.ref = Engine(M, Smax)             # weights in cuBLAS layout, rope tables, final logits
        mdl.rsqrt_table()                       # built here: compiled code reads the cached table
        mdl.ensure_caps(M)
        self.silu_t = mdl.silu_table(M.eG, M.eA, "cuda")
        self.M, self.cfg, self.Smax, self.B = M, M.cfg, Smax, B
        cfg = M.cfg
        texp = {}
        dev = lambda v: torch.tensor(int(v), dtype=torch.int64, device="cuda")
        i64 = lambda t: t.to(device="cuda", dtype=torch.int64)
        self.lp = []                            # per layer: every parameter as a tensor (one compiled graph)
        for Ly in M.layers:
            if Ly.E not in texp:
                texp[Ly.E] = mdl.exp_table(Ly.E, cfg.dh)
            p = dict(Wqkv=mdl.tn(Ly.Wqkv), kc=i64(Ly.kc), bias=i64(Ly.bias), kqs=dev(Ly.kqs), Wo=mdl.tn(Ly.Wo),
                     Kco=dev(Ly.Kco), texp=texp[Ly.E], vshift=i64(Ly.vshift))
            if cfg.qk_norm:
                p.update(Gqk=i64(Ly.Gqk), kn=i64(Ly.kn))
            if cfg.moe:
                p.update(Wr=mdl.tn(Ly.Wr), kr0=dev(-(int(Ly.eWr) + mdl.R_FRAC)), eWge=i64(Ly.eWge),
                         eWue=i64(Ly.eWue), eWde=i64(Ly.eWde), Wge=Ly.Wge, Wue=Ly.Wue, Wde=Ly.Wde)
            else:
                Wgu = torch.cat([Ly.Wg, Ly.Wu], dim=1).T.contiguous().T          # (d, 2 dff), column-major
                dff = Ly.Wg.shape[1]
                Ly.Wg, Ly.Wu = Wgu[:, :dff], Wgu[:, dff:]                       # prefill reads the same storage
                p.update(Wgu=Wgu, kgc=i64(M.eG - Ly.eWg), kgl=dev(Ly.kgl),
                         Kcd=dev(-(Ly.eWu + M.eA + Ly.eWd + M.f_h)), Wd=mdl.tn(Ly.Wd))
            self.lp.append(p)
        self.Kz = dev(M.Kz)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)   # prefill lengths
        cf = (lambda f: torch.compile(f, dynamic=False)) if compile else (lambda f: f)
        self.pre_fn, self.ao_fn, self.mlp_fn = cf(self._pre), cf(self._ao), cf(self._mlp)
        self.route_fn, self.mid_fn, self.fin_fn = cf(self._route), cf(self._mid), cf(self._fin)
        self.t32 = {}
        for Ly, p in zip(M.layers, self.lp):
            if Ly.E not in self.t32:
                self.t32[Ly.E] = p["texp"].to(torch.int32).contiguous()
        self.graph = None

    # ------------------------------------------------------------------ layer parts (rows -> rows, flags)
    def _pre(self, h, cos, sin, p):
        """rows h (N, d) -> rQ (N, nh, dh), rK, V (N, nkv, dh) at positions cos/sin rows, bad (N,)."""
        cfg = self.cfg
        nh, nkv, dh = cfg.nh, cfg.nkv, cfg.dh
        nq, nk = nh * dh, nkv * dh
        half = dh // 2
        N = h.shape[0]
        q, _ = rowq(h, 0, 17)
        qp, Eh = normscale(q)
        bad = Eh > p["kqs"]
        acc = mm(qp, p["Wqkv"]).to(torch.int64)
        if cfg.qk_norm:
            q8, _ = rowq(acc[:, :nq + nk].reshape(-1, dh), 2, 22)
            qn, E2 = normscale(q8)
            k = p["kn"].view(1, -1) + E2.view(N, -1).repeat_interleave(dh, dim=1)
            bad = bad | (k > mdl.QKV_KMAX).any(dim=-1) | (k < -31).any(dim=-1)   # in [-31, 0]: shifted by 1
            qk = shiftround(qn.view(N, nq + nk) * p["Gqk"].unsqueeze(0), k.clamp(1, mdl.QKV_KMAX)).clamp(-127, 127)
            v = (shiftround(acc[:, nq + nk:], p["kc"][nq + nk:].unsqueeze(0) + Eh.unsqueeze(-1))
                 + p["bias"][nq + nk:].unsqueeze(0)).clamp(-127, 127)
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = (shiftround(acc, p["kc"].unsqueeze(0) + Eh.unsqueeze(-1)) + p["bias"].unsqueeze(0)).clamp(-127, 127)
        c_, s_ = cos.view(N, 1, half), sin.view(N, 1, half)

        def rope(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([(x1 * c_ - x2 * s_ + (1 << 14)) >> 15, (x2 * c_ + x1 * s_ + (1 << 14)) >> 15], dim=-1)
        return rope(qkv[:, :nq].view(N, nh, dh)), rope(qkv[:, nq:nq + nk].view(N, nkv, dh)), \
            qkv[:, nq + nk:].view(N, nkv, dh), bad

    def _ao(self, h, o, Wo, Kco):
        """attention output projection and residual; the MLP input rows -> (h, qp, Eh, bad)."""
        qo, eo = rowq(o, 0, 17)
        c, bad = gsr(mm(qo, Wo).to(torch.int64), (Kco - eo).unsqueeze(-1))
        h = satadd(h, c)
        q, _ = rowq(h, 0, 17)
        qp, Eh = normscale(q)
        return h, qp, Eh, bad

    def _mlp(self, h, qp, Eh, Wgu, kgc, kgl, Kcd, Wd):
        gu = mm(qp, Wgu).to(torch.int64)
        dff = gu.shape[1] // 2
        kg = kgc.unsqueeze(0) + torch.maximum(Eh, kgl).unsqueeze(-1)
        c = shiftround(gu[:, :dff], kg.clamp(1, mdl.SILU_KMAX))
        bad = out_of(kg, 1, mdl.SILU_KMAX) | out_of(c, -mdl.SILU_CMAX, mdl.SILU_CMAX)
        a = silu_tab(c, self.silu_t)
        qu, eup = rowq(gu[:, dff:], 2, 22)
        if self.cfg.down_b == 127:
            qm, em = rowq(a * qu, 0, 17)
            accd = mm(qm, Wd).to(torch.int64)
        else:                                   # 9-bit input: int8 GEMMs on qm = 16 hi + lo, lo in [0, 16)
            qm, em = rowq_b(a * qu, self.cfg.down_b, 17)
            accd = 16 * mm(qm >> 4, Wd).to(torch.int64) + mm(qm & 15, Wd).to(torch.int64)
        y, b2 = gsr(accd, (Kcd - (em + eup - Eh)).unsqueeze(-1))
        return satadd(h, y), bad | b2

    def _route(self, qp, Eh, Wr, kr0, eWge, eWue):
        """router (model.moe_route) -> expert ids (T, k), weights (T, k), gate shifts (T, k), eWue[ids], bad."""
        E, k = Wr.shape[1], self.cfg.moe["k"]
        kr = Eh + kr0
        bad = kr > mdl.R_KMAX
        z = shiftround(mm(qp, Wr).to(torch.int64), kr.clamp(1, mdl.R_KMAX).unsqueeze(-1)).clamp(-mdl.R_SAT, mdl.R_SAT)
        key = z * E + (E - 1 - torch.arange(E, device=z.device)).unsqueeze(0)
        idx = torch.topk(key, k, dim=-1).indices
        zs = z.gather(1, idx)
        pr = mdl.router_table()[zs.max(dim=-1, keepdim=True).values - zs]
        S = pr.sum(dim=-1, keepdim=True)
        w = (2 * (1 << mdl.W_FRAC) * pr + S) // (2 * S)
        kg = self.M.eG - eWge[idx] + Eh.unsqueeze(1)
        return idx, w, kg.clamp(min=1), eWue[idx], bad | (kg > mdl.SILU_KMAX).any(dim=-1)

    def _mid(self, accg, accu, kg, w, xu, xd):
        """per (token, slot): SiLU, up requant, product requant, slot weights -> q2 (T, k, de), e2 + Et, bad."""
        T, k, de = accg.shape
        c = shiftround(accg, kg.clamp(1, mdl.SILU_KMAX).unsqueeze(-1))
        bad = out_of(c, -mdl.SILU_CMAX, mdl.SILU_CMAX)
        a = silu_tab(c, self.silu_t)
        qu, eu = rowq(accu, 2, 22)
        qm, em = rowq(a * qu, 0, 17)
        es = em + eu + xu + xd
        Et = es.amax(dim=1)
        so = Et.unsqueeze(1) - es
        bad = bad | (so > 24).any(dim=1)
        om = shiftround(w, so.clamp(max=24))
        q2, e2 = rowq_b((qm * om.unsqueeze(-1)).reshape(T, k * de), mdl.MOE_B, mdl.MOE_EMAX)
        return q2.view(T, k, de), e2 + Et, bad

    def _fin(self, h, accd, e2t, Eh):
        y, bad = gsr(accd, (-(e2t + self.M.eA - Eh - mdl.W_FRAC + self.M.f_h)).clamp(max=24).unsqueeze(-1))
        return satadd(h, y), bad

    def _moe(self, h, qp, Eh, p):
        moe, cfg = self.moe, self.cfg
        T, k, de = qp.shape[0], cfg.moe["k"], cfg.moe["de"]
        idx, w, kg, xu, b1 = self.route_fn(qp, Eh, p["Wr"], p["kr0"], p["eWge"], p["eWue"])
        xd = p["eWde"][idx]
        lay = moe.expert_layout(idx.reshape(-1), p["Wr"].shape[1], self.tile(T * k))
        xs = qp.repeat_interleave(k, dim=0)
        accg = moe.grouped_imm(xs, lay, p["Wge"], tile=self.tile(T * k)).view(T, k, de)
        accu = moe.grouped_imm(xs, lay, p["Wue"], tile=self.tile(T * k)).view(T, k, de)
        q2, e2t, b2 = self.mid_fn(accg, accu, kg, w, xu, xd)
        accd = moe.grouped_imm(q2.view(T * k, de), lay, p["Wde"], wide=True, tile=self.tile(T * k))
        h, b3 = self.fin_fn(h, accd.view(T, k, -1).sum(dim=1), e2t, Eh)
        return h, b1 | b2 | b3

    @staticmethod
    def tile(n):
        """rows per expert tile: small batches (decode) waste less of each tile on padding."""
        return 16 if n <= 1024 else 128

    def _post(self, h, o, p):
        h, qp, Eh, bad = self.ao_fn(h, o, p["Wo"], p["Kco"])
        if self.cfg.moe:
            h, b2 = self._moe(h, qp, Eh, p)
        else:
            h, b2 = self.mlp_fn(h, qp, Eh, p["Wgu"], p["kgc"], p["kgl"], p["Kcd"], p["Wd"])
        return h, bad | b2

    def _head(self, h):
        """greedy ids of rows h and their bad flags."""
        q, _ = rowq(h, 0, 17)
        qp, Eh = normscale(q)
        z, bad = gsr(mm(qp, mdl.tn(self.M.Wlm)).to(torch.int64), (self.Kz + Eh).unsqueeze(-1))
        return torch.argmax(z, dim=1), bad

    # ------------------------------------------------------------------ prefill
    def prefill(self, toks, caches, lens=None, start=0):
        """toks (B, S) right-padded prompts at positions start..start+S-1 (positions below start already in
        the caches); fills those cache positions -> rows (B, S, d) and, per prompt, whether a row before its
        end (lens, absolute) left F_O's domain."""
        from . import foattn
        M, cfg = self.M, self.cfg
        B, S = toks.shape
        T = start + S
        lens = torch.full((B,), T, device="cuda") if lens is None else lens
        valid = (torch.arange(start, T, device="cuda").view(1, S) < lens.view(B, 1)).reshape(-1)
        pos = torch.arange(start, T, device="cuda").view(1, S).expand(B, S).contiguous()
        cos = self.ref.cos[start:T].repeat(B, 1)
        sin = self.ref.sin[start:T].repeat(B, 1)
        h = M.emb[toks.reshape(-1).cuda()].to(torch.int64)
        bad = torch.zeros(B * S, dtype=torch.bool, device="cuda")
        for l, Ly in enumerate(M.layers):
            p = self.lp[l]
            rQ, rK, Vh, b1 = self.pre_fn(h, cos, sin, p)
            caches[l]["rK"][:, start:T] = rK.view(B, S, cfg.nkv, cfg.dh).to(torch.int16)
            caches[l]["V"][:, start:T] = Vh.view(B, S, cfg.nkv, cfg.dh).to(torch.int8)
            o = foattn.attend(rQ.view(B, S, cfg.nh, cfg.dh), caches[l]["rK"][:, :T], caches[l]["V"][:, :T], pos,
                              self.t32[Ly.E], p["vshift"], cfg.nkv)
            h, b2 = self._post(h, o.reshape(B * S, -1), p)
            bad |= b1 | b2
        return h.view(B, S, -1), (bad & valid).view(B, S).any(dim=1)

    # ------------------------------------------------------------------ decode
    def _step(self, y, pos, caches):
        """One decode step: token ids y (B,) at positions pos (B,) -> next greedy ids (B,), bad (B,)."""
        from . import foattn
        M, cfg = self.M, self.cfg
        h = M.emb[y].to(torch.int64)
        cos, sin = self.ref.cos[pos], self.ref.sin[pos]
        bi = torch.arange(self.B, device=h.device)
        bad = torch.zeros(self.B, dtype=torch.bool, device=h.device)
        for l, Ly in enumerate(M.layers):
            p = self.lp[l]
            rQ, rK, Vh, b1 = self.pre_fn(h, cos, sin, p)
            caches[l]["rK"][bi, pos] = rK.to(torch.int16)
            caches[l]["V"][bi, pos] = Vh.to(torch.int8)
            o = foattn.attend_dec(rQ, caches[l]["rK"], caches[l]["V"], pos, self.t32[Ly.E], p["vshift"], cfg.nkv)
            h, b2 = self._post(h, o.reshape(self.B, -1), p)
            bad = bad | b1 | b2
        y, b3 = self._head(h)
        return y, bad | b3

    # ------------------------------------------------------------------ API
    def new_caches(self):
        return self.ref.new_caches(self.B, self.Smax)

    def capture(self, caches):
        """Record one decode step as a CUDA graph over static input/output buffers."""
        self.y = torch.zeros(self.B, dtype=torch.int64, device="cuda")
        # warm-up steps run for real: they write the scratch slot Smax - 1, which decoding never reaches
        self.pos = torch.full((self.B,), self.Smax - 1, dtype=torch.int64, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                  # compile and warm up outside the capture
                self.out, self.bad = self._step(self.y, self.pos, caches)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out, self.bad = self._step(self.y, self.pos, caches)
        self.caches = caches

    def generate(self, prompts, ngen, caches=None, pchunk=2, flags=False, spad=1):
        """Greedy decoding of B prompts (lists of ids, each len + ngen < Smax).  Returns
        (generated ids per prompt, time to first token in s, decode seconds); flags: also, per
        prompt, (prefill left F_O's domain, [step t left it] for the step producing token t + 1).
        spad: the prefill length is rounded up to a multiple of spad (fewer compiled shapes; rows
        past a prompt's end do not change the function)."""
        import time
        B = self.B
        assert len(prompts) == B
        caches = caches if caches is not None else self.new_caches()
        torch.cuda.synchronize()
        t0 = time.time()
        lens = torch.tensor([len(p) for p in prompts], device="cuda")
        S = -(-int(lens.max()) // spad) * spad
        assert S + ngen < self.Smax
        toks = torch.zeros((B, S), dtype=torch.int64)
        for b, p in enumerate(prompts):
            toks[b, :len(p)] = torch.tensor(p)
        ys, pb = [], []
        for b0 in range(0, B, pchunk):          # prefill pchunk sequences at a time, into their cache rows
            b1 = min(B, b0 + pchunk)
            cv = [{k: v[b0:b1] for k, v in c.items()} for c in caches]
            h, bad = self.prefill(toks[b0:b1].cuda(), cv, lens[b0:b1])
            y, b2 = self._head(h[torch.arange(b1 - b0, device="cuda"), lens[b0:b1] - 1])
            ys.append(y)
            pb.append(bad | b2)
            del h
        y = torch.cat(ys)
        torch.cuda.synchronize()
        ttft = time.time() - t0
        if self.graph is None or self.caches is not caches:
            self.capture(caches)
        out, sb = [y], []
        pos = lens.clone()
        t1 = time.time()
        for _ in range(ngen - 1):
            self.y.copy_(out[-1])
            self.pos.copy_(pos)
            self.graph.replay()
            out.append(self.out.clone())
            sb.append(self.bad.clone())
            pos += 1
        torch.cuda.synchronize()
        dec = time.time() - t1
        ids = torch.stack(out, dim=1).tolist()
        if not flags:
            return ids, ttft, dec
        steps = torch.stack(sb, dim=1).tolist() if sb else [[] for _ in range(B)]
        return ids, ttft, dec, list(zip(torch.cat(pb).tolist(), steps))


    def serve(self, prompts, ngen, stop, every=16, spad=128, done=None):
        """Greedy decoding of any number of prompts with continuous batching: the B slots decode
        independently, and a slot whose sequence has ended takes the next prompt.  A sequence ends at the
        first m <= ngen generated tokens with stop(i, ids[:m]) (prompt i), else at ngen.  The prompts'
        longest common prefix is prefilled once and copied into each slot.  Per sequence this is the
        function of generate (rows are computed independently; attention reads only the sequence's own
        keys), so the tokens are the same.  Stops are checked on the host every `every` steps.
        -> per prompt (generated ids, F_O undefined on it); done(i, ids, undefined) as each ends."""
        B, cfg = self.B, self.cfg
        P = 0
        while P < min(len(q) for q in prompts) - 1 and all(q[P] == prompts[0][P] for q in prompts):
            P += 1
        S = max(-(-(len(q) - P) // spad) * spad for q in prompts)
        assert P + S <= self.Smax - 1 and max(len(q) for q in prompts) + ngen + every < self.Smax - 1
        caches = self.new_caches()
        pre = [dict(rK=None, V=None) for _ in caches]
        pf0 = False
        if P:
            tmp = self.ref.new_caches(1, P)
            _, b = self.prefill(torch.tensor([prompts[0][:P]]).cuda(), tmp)
            pf0 = bool(b.item())
            pre = tmp
        if self.graph is None or self.caches is not caches:
            self.capture(caches)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))[::-1]   # longest first
        slot = [None] * B                       # per slot: [prompt, ids, step flags, prefill flag, ids checked]
        res = [None] * len(prompts)

        def fill(s):
            i = order.pop()
            q = prompts[i]
            n = -(-(len(q) - P) // spad) * spad
            toks = torch.zeros((1, n), dtype=torch.int64)
            toks[0, :len(q) - P] = torch.tensor(q[P:])
            cv = [{k: v[s:s + 1] for k, v in c.items()} for c in caches]
            for c, t in zip(cv, pre):
                if P:
                    c["rK"][0, :P] = t["rK"][0]
                    c["V"][0, :P] = t["V"][0]
            ln = torch.tensor([len(q)], device="cuda")
            h, bad = self.prefill(toks.cuda(), cv, ln, start=P)
            y, b2 = self._head(h[0, len(q) - P - 1:len(q) - P])
            self.y[s] = y[0]
            self.pos[s] = len(q)
            slot[s] = [i, [int(y[0])], [], pf0 or bool((bad | b2).item()), 0]

        def end(s):
            i, g, st, pf, c = slot[s]
            n = next((m for m in range(c + 1, min(len(g), ngen) + 1) if stop(i, g[:m])), None)
            if n is None and len(g) < ngen:
                slot[s][4] = len(g)
                return False
            n = n or ngen
            res[i] = (g[:n], bool(pf or any(st[:n - 1])))
            if done:
                done(i, *res[i])
            slot[s] = None
            return True

        for s in range(B):
            if order:
                fill(s)
            else:
                self.pos[s] = 0                 # idle slot: decodes position 0 of its own (unused) rows
        while any(x is not None for x in slot):
            hist, bh = [], []
            for _ in range(every):
                self.graph.replay()
                hist.append(self.out.clone())
                bh.append(self.bad.clone())
                self.y.copy_(self.out)
                self.pos += 1
            H = torch.stack(hist, dim=1).tolist()
            F = torch.stack(bh, dim=1).tolist()
            for s in range(B):
                if slot[s] is None:
                    continue
                slot[s][1] += H[s]
                slot[s][2] += F[s]
                if end(s) and order:
                    fill(s)
            for s in range(B):
                if slot[s] is None:
                    self.pos[s] = 0
        return res


def check(M, B=2, S=300, ngen=8, seed=0, compile=True):
    """FastEngine equals evalfo.Engine (greedy tokens) on random prompts, with no flagged row."""
    g = torch.Generator().manual_seed(seed)
    prompts = [torch.randint(0, M.cfg.V, (S - 37 * i,), generator=g).tolist() for i in range(B)]
    Smax = S + ngen + 1
    ref = Engine(M, Smax).generate(prompts, ngen)
    fe = FastEngine(M, Smax, B, compile=compile)
    got, _, _, fl = fe.generate(prompts, ngen, flags=True)
    clean = not any(pf or any(st) for pf, st in fl)
    return ref == got and clean, ref, got


def first_token_flags(M, chunk=512):
    """The vocabulary tokens whose position-0 row leaves F_O's domain (prefill or the final logits).
    chunk * nh must stay below 2^16 (the attention kernel's grid)."""
    fe = FastEngine(M, 2, chunk)
    caches = fe.new_caches()
    bad = []
    for t0 in range(0, M.cfg.V, chunk):
        n = min(M.cfg.V, t0 + chunk) - t0
        toks = torch.zeros((chunk, 1), dtype=torch.int64, device="cuda")
        toks[:n, 0] = torch.arange(t0, t0 + n, device="cuda")
        h, b1 = fe.prefill(toks, caches)
        _, b2 = fe._head(h[:, 0])
        bad += (torch.nonzero((b1 | b2)[:n]).view(-1) + t0).tolist()
    return bad
