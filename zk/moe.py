"""The MoE MLP of a Qwen3-MoE layer in the proof (F_O: model.moe_out), adapted from the GLM-5.3 branch.

Per layer l (prefix '{l}.e.'), from the normalized rows qp2 (exponents Eh) of the mid-layer NormScale:
  router  acc_r = qp2 Wr; z = SatShift(acc_r, kr) to |z| <= 2^15 - 1 (units 2^-8), kr = max(Eh - eWr - 8, 1)
  top-k   sel(t, j) = 1 for the k largest z (ties to the smaller index): threshold th = z(jt),
          [z > th] or [z = th and j <= jt], sum_j sel = k; the slots (t, s) carry (e, z, exponents) of
          the selected candidates by a dynamic multiset (candidates with multiplicity -sel)
  weights zmax = max_s z_s (a one-hot slot attains it, the table domain bounds the others),
          p_s = tau_r(zmax - z_s), S = sum_s p_s, w_s = Div(p_s, S, 2^16)
  experts per (t, s), expert e = e(t, s): c = ShiftRound(qp2 Wg_e, kg) into tau_SiLU -> a, kg = max(eG - eWg_e + Eh, 1);
          qu = ShiftQ8(qp2 Wu_e) (exponent eu); qm = Requant8(a qu) (exponent em)
  merge   es = em + eu + eWu_e + eWd_e, Et = max_s es, omega_s = ShiftRound(w_s, Et - es),
          q2 = Requant_1023(qm omega) over the token row (k de values), acc_d = sum_s q2_s Wd_e
  out     y = GShiftRound(acc_d, kd), kd = min(-(e2 + Et + eA - Eh - 14 + f_h), 24): the 'mo' step of llm.py
The three clamps (flags fr, fg, fd) only define rows of very small norm (E_h near -7), which F_O left undefined.
The expert products are gathered sumchecks: over the input dimension against the gathered weight rows,
then over the expert index bits against the one-hot of each slot's expert (claims on the stacked expert
weights and on the slots' index bits)."""
import numpy as np
import torch
from .field import P
from . import tables as tb
from . import model as mdl
from .poly import Poly, lparam
from .sumcheck import Tab, FULL, ROW, COL, Prog
from .proto import const_pt, cat, nbits
from .ext import dev as _dev

INV2 = (P + 1) // 2
RTEXP = tb.RTEXP                 # (x, tau_r(x)), x in [0, 2^16)
B2 = mdl.MOE_B                   # bound of the merged token row (1023)


def _llm():
    from . import llm
    return llm


def log2(x):
    lg = int(x).bit_length() - 1
    assert 1 << lg == x, x
    return lg


def bits_of(v, nb):
    return [(v >> i) & 1 for i in range(nb)]


def pub_c(D, v):
    out = np.zeros(D, dtype=np.int64)
    out[0] = int(v) % P
    return out


class Geo:
    """Block geometry of the MoE of a model."""

    def __init__(self, cfg, sh):
        mo = cfg.moe
        self.E, self.k, self.de = mo["E"], mo["k"], mo["de"]
        self.lk = log2(self.k)
        self.cbr = min(sh.cb, (self.E & -self.E).bit_length() - 1)
        self.KE = self.E >> self.cbr
        self.cbe = min(sh.cb, (self.de & -self.de).bit_length() - 1)
        self.Kde = self.de >> self.cbe
        self.Ke = self.k * self.Kde
        self.nbE = nbits(self.E)
        self.nbu = self.cbe + nbits(self.Kde)          # index bits of a (t, s) row of de values
        self.nb2 = self.cbe + nbits(self.Ke)           # index bits of a token row of k de values


# =================================================================== witness
def rq_pack_b(a, cb, e0, emax, B):
    """Requant to |q| <= B (B = 127: Requant8 / ShiftQ8) with its minimality witness, rows of a."""
    llm = _llm()
    if B == 127:
        return llm.rq_pack(a, cb, e0, emax)
    q, e, _ = mdl._rowq_b(a, B, emax)
    s = mdl.pow2(e)
    f = e > e0
    sc = s.unsqueeze(-1)
    W = 2 * B + 1
    cand = (4 * a >= W * sc) | (4 * a < -W * sc)
    js = torch.argmax(cand.to(torch.int8), dim=-1)
    js = torch.where(f, js, torch.zeros_like(js))
    asel = torch.gather(a, -1, js.unsqueeze(-1)).squeeze(-1)
    sig = torch.where(f, (4 * asel >= W * s).to(torch.int64), torch.zeros_like(e))
    fi = f.to(torch.int64)
    w = fi * (sig * (4 * asel - W * s) + (1 - sig) * (-4 * asel - W * s - 1))
    mdl.check((w.min() >= 0) & (w.max() < (1 << 30)), "requant: minimality witness range")
    K = a.shape[-1] >> cb
    return dict(q=q, e=e, s=s, f=fi, sig=sig, wl=w & 32767, wh=w >> 15, asel=asel, jb=llm._jbits(js, cb, K))


def satshift_pack(acc, k, B):
    """SatShift (bias 0): q = clamp(round(acc / 2^k), -B, B), saturation sign o and witness wh."""
    s = mdl.pow2(k)
    r = mdl.fdiv(acc + (s >> 1), s)
    q = r.clamp(-B, B)
    o = torch.sign(r - q)
    w = torch.where(o != 0, o * acc - (2 * B + 1) * (s >> 1) - (o == -1).to(torch.int64), torch.zeros_like(acc))
    mdl.check((w.min() >= 0) & (w.max() < (1 << 30)), "satshift: saturation witness range")
    return dict(q=q, o=o, wh=w >> 15)


TILE = 128                       # rows of one expert per tile of the grouped expert GEMM
TCHUNK = 128                     # tiles per batched GEMM (bounds the gathered weights' memory)
KC = 1024                        # inner-dimension chunk: |x| <= 128, |w| <= 127 sums stay below 2^24


def expert_layout(ef, E, tile=TILE):
    """Fixed-shape grouping of rows by expert (no host synchronization, so it can be captured in
    a CUDA graph): rows sorted by expert into tiles of `tile` rows of one expert each.  Returns
    (src, dest, te): row src[i] goes to padded row dest[i]; tile j belongs to expert te[j]."""
    n, dev = ef.numel(), ef.device
    cnt = torch.zeros(E, dtype=torch.int64, device=dev).scatter_add_(0, ef, torch.ones_like(ef))
    pc = (cnt + tile - 1) // tile * tile
    end = torch.cumsum(pc, 0)
    src = torch.argsort(ef, stable=True)
    es = ef[src]
    dest = (end - pc)[es] + torch.arange(n, device=dev) - (torch.cumsum(cnt, 0) - cnt)[es]
    nt = -(-(n + E * (tile - 1)) // tile)
    te = torch.searchsorted(end, torch.arange(nt, device=dev) * tile, right=True).clamp(max=E - 1)
    return src, dest, te


def grouped_imm(X, lay, Ws, wide=False, tile=TILE):
    """Exact X[i] Ws[e_i] (int64) for rows grouped by expert_layout: one batched bf16 GEMM with fp32
    accumulation per chunk of tiles and of KC inner indices, every partial sum an integer below
    2^24.  |X| <= 128 (wide: |X| < 2^11, as 16 hi + lo); tile: that of expert_layout."""
    if wide:
        hi = mdl.fdiv(X, 16)
        return 16 * grouped_imm(hi, lay, Ws, tile=tile) + grouped_imm(X - 16 * hi, lay, Ws, tile=tile)
    src, dest, te = lay
    n, din = X.shape
    dout = Ws.shape[2]
    nt = te.numel()
    Xp = torch.zeros((nt * tile, din), dtype=torch.bfloat16, device=X.device)
    Xp[dest] = X[src].to(torch.bfloat16)
    Xp = Xp.view(nt, tile, din)
    Yp = torch.empty((nt, tile, dout), dtype=torch.int64, device=X.device)
    for t0 in range(0, nt, TCHUNK):
        t1 = min(nt, t0 + TCHUNK)
        Wt = Ws[te[t0:t1]]
        acc = None
        for k0 in range(0, din, KC):
            k1 = min(din, k0 + KC)
            y = torch.bmm(Xp[t0:t1, :, k0:k1], Wt[:, k0:k1].to(torch.bfloat16),
                          out_dtype=torch.float32).to(torch.int64)
            acc = y if acc is None else acc + y
        Yp[t0:t1] = acc
    Y = torch.empty((n, dout), dtype=torch.int64, device=X.device)
    Y[src] = Yp.view(-1, dout)[dest]
    return Y


def moe_wit(M, Ly, qp, Eh, geo):
    """Every committed and virtual value of the MoE of layer Ly (equal to model.moe_out)."""
    llm = _llm()
    T, dev = qp.shape[0], qp.device
    E, k, de = geo.E, geo.k, geo.de
    qp, Eh = qp.to(torch.int64), Eh.to(torch.int64)
    acc_r = mdl.imm(qp, Ly.Wr)
    krv = Eh + Ly.krc_t
    mdl.check(krv.max() <= mdl.R_KMAX, "F_O: router shift out of range")
    fr = (krv < 1).to(torch.int64)                                 # clamped rows: kr = max(krv, 1)
    kr = krv.clamp(min=1)
    sz = satshift_pack(acc_r, kr.unsqueeze(-1), mdl.R_SAT)
    z = sz["q"]
    j = torch.arange(E, device=dev)
    key = z * E + (E - 1 - j).unsqueeze(0)
    idx = torch.topk(key, k, dim=-1).indices                       # (T, k): descending, ties to smaller j
    jt = idx[:, -1]
    th = z.gather(1, jt.unsqueeze(1)).squeeze(1)
    sel = (z > th.unsqueeze(1)) | ((z == th.unsqueeze(1)) & (j.unsqueeze(0) <= jt.unsqueeze(1)))
    dfl = (z == th.unsqueeze(1)).to(torch.int64)
    zs = z.gather(1, idx)
    zmax = zs[:, 0]
    flz = torch.zeros_like(zs)
    flz[:, 0] = 1
    pr = mdl.router_table()[zmax.unsqueeze(1) - zs]
    S = pr.sum(dim=1)
    w = mdl.div_round(pr, S.unsqueeze(1), 1 << mdl.W_FRAC)
    xg, xu, xd = Ly.eWge[idx], Ly.eWue[idx], Ly.eWde[idx]           # (T, k)
    kgv = M.eG - xg + Eh.unsqueeze(1)
    mdl.check(kgv.max() <= mdl.SILU_KMAX, "F_O: expert gate shift out of range")
    fg = (kgv < 1).to(torch.int64)                                 # clamped slots: kg = max(kgv, 1)
    kg = kgv.clamp(min=1)
    lay = expert_layout(idx.reshape(-1), E)                          # rows (t, s)
    xs = qp.repeat_interleave(k, dim=0)
    accg = grouped_imm(xs, lay, Ly.Wge).view(T, k, de)
    accu = grouped_imm(xs, lay, Ly.Wue).view(T, k, de)
    del xs
    c = mdl.shiftround(accg, kg.unsqueeze(-1))
    mdl.check(c.abs().max() <= mdl.SILU_CMAX, "F_O: SiLU input out of range")
    # table lookup without a host read (|c| <= SILU_CMAX is checked above; graph capture)
    a = mdl.silu_table(M.eG, M.eA, dev)[(c.clamp(-mdl.SILU_CMAX, mdl.SILU_CMAX) + mdl.SILU_CMAX)]
    ru = rq_pack_b(accu.reshape(T * k, de), geo.cbe, 2, 22, 127)    # rows (t, s)
    qu = ru["q"].reshape(T, k, de)
    m = a * qu
    rmq = rq_pack_b(m.reshape(T * k, de), geo.cbe, 0, 17, 127)
    qm = rmq["q"].reshape(T, k, de)
    eu, em = ru["e"].reshape(T, k), rmq["e"].reshape(T, k)
    es = em + eu + xu + xd
    Et = es.max(dim=1).values
    fl = torch.zeros_like(es)
    fl.scatter_(1, es.argmax(dim=1, keepdim=True), 1)
    so = Et.unsqueeze(1) - es
    mdl.check(so.max() <= 24, "F_O: MoE slot scale shift above 24")
    om = mdl.shiftround(w, so)
    r = qm * om.unsqueeze(-1)
    r2 = rq_pack_b(r.reshape(T, k * de), geo.cbe, 0, mdl.MOE_EMAX, B2)
    q2 = r2["q"].reshape(T, k, de)
    accd = grouped_imm(q2.reshape(T * k, de), lay, Ly.Wde, wide=True).view(T, k, M.cfg.d).sum(dim=1)
    kdv = -(r2["e"] + Et + M.eA - Eh - mdl.W_FRAC + M.f_h)
    fd = (kdv > 24).to(torch.int64)                                # clamped rows: kd = min(kdv, 24)
    kd = kdv.clamp(max=24)

    def slotmajor(r):                     # per-(t, s) row values -> (s, t) order (k blocks over t)
        out = {f: v.reshape(T, k).T.reshape(-1) for f, v in r.items() if f not in ("jb", "q")}
        out["jb"] = [b.reshape(T, k).T.reshape(-1) for b in r["jb"]]
        return out
    ru_s, rm_s = slotmajor(ru), slotmajor(rmq)
    return dict(acc_r=acc_r, kr=kr, fr=fr, pkr=llm.tb.p2t(kr).to(torch.int64), fg=fg, fd=fd, z=z, zo=sz["o"], zwh=sz["wh"], sel=sel.to(torch.int64), dfl=dfl, jt=jt, th=th,
                idx=idx, zs=zs, zmax=zmax, flz=flz, p=pr, S=S, w=w, xg=xg, xu=xu, xd=xd, kg=kg,
                pkg=llm.tb.p2t(kg).to(torch.int64), accg=accg, accu=accu, c=c, a=a, qu=qu, qm=qm, ru=ru_s, rm=rm_s,
                eu=eu, em=em, es=es, Et=Et, fl=fl, so=so, om=om, r2=r2, q2=q2, accd=accd, kd=kd)


# =================================================================== the proof
class MoEProof:
    """Steps of one layer's MoE inside llm.Proto (prove pass and verifier)."""

    def __init__(self, pr, l):
        self.pr, self.l = pr, l
        self.ctx, self.sh, self.M = pr.ctx, pr.sh, pr.M
        self.p = f"{l}."
        self.q = f"{l}.e."
        self.g = Geo(self.M.cfg, self.sh)
        self.dyn = lparam("Dme")         # the tag of the layer's slot multiset (a layer parameter)

    @property
    def Ly(self):
        return self.M.layers[self.l]

    def W(self):
        return self.pr.wit_of(self.l).moe

    def HW(self):
        return self.pr.wit_of(self.l)

    # ------------------------------------------------------------ declarations
    EF = ("c", "a", "qu", "qm", "q2")
    TF = ("th", "zmax", "S", "Et", "kr", "pkr", "fr", "fd")      # per token
    SF = ("flz", "fl", "om", "kg", "pkg", "fg")                 # per (token, slot)

    def items(self):
        g, t, q = self.g, self.sh.t, self.q
        k, lk = g.k, g.lk
        rows = lambda pre, fields, kb=1: [(pre + f, kb, t) for f in fields]
        return ([(q + f, g.KE, g.cbr + t) for f in ("z", "zo", "zwh", "sel", "dfl")]
                + [(q + "jt", g.nbE, t)] + rows(q, self.TF)
                + [(q + "eb", g.nbE, lk + t)] + [(q + f, 1, lk + t) for f in ("zs", "xg", "xu", "xd", "p", "w")]
                + rows(q, self.SF, k) + rows(q + "so.", _llm().SR_F, k)
                + [(q + f, g.Ke, g.cbe + t) for f in self.EF]
                + rows(q + "ru.", _llm().RQ_F, k) + [(q + "ru.jb", g.nbu * k, t)]
                + rows(q + "rm.", _llm().RQ_F, k) + [(q + "rm.jb", g.nbu * k, t)]
                + rows(q + "r2.", _llm().RQ_F) + [(q + "r2.jb", g.nb2, t)])

    def weight_items(self):
        """Router in the layer's weight group; the expert stacks are groups of their own."""
        sh = self.sh
        return [(self.q + "Wr", sh.Kd * self.g.KE, sh.cb + self.g.cbr, (sh.Kd, self.g.KE))]

    def expert_groups(self):
        """One column per weight block (xblock, nblock), holding that block of every expert
        (expert index in the high bits): few long columns, so a query opens few leaves."""
        sh, g = self.sh, self.g
        assert g.E == 1 << nbits(g.E)
        return {f"{self.l}.{nm}": [(self.q + nm, sh.Kd * g.Kde, sh.cb + g.cbe + nbits(g.E), False)]
                for nm in ("Wge", "Wue", "Wde")}

    def tensors(self, w=None):
        """The MoE group's tensors from the layer witness's MoE values w (default: the prover's)."""
        llm = _llm()
        g, q = self.g, self.q
        w = self.W() if w is None else w
        T, _k = w["z"].shape[0], g.k
        f32, rm = llm.f32, llm.rm
        out = {}
        for f in ("z", "zo", "zwh", "sel", "dfl"):
            out[q + f] = rm(w[f], g.cbr)
        out[q + "jt"] = torch.cat([f32(b) for b in bits_of(w["jt"], g.nbE)])
        for f in self.TF:
            out[q + f] = f32(w[f])
        out[q + "eb"] = torch.cat([rm(b, g.lk) for b in bits_of(w["idx"], g.nbE)])
        for f in ("zs", "xg", "xu", "xd", "p", "w"):
            out[q + f] = rm(w[f], g.lk)
        tr = lambda v: f32(v.T.contiguous())
        for f in self.SF:
            out[q + f] = tr(w[f])
        so = llm.sr_pack(w["so"].T.contiguous())
        for f in llm.SR_F:
            out[q + "so." + f] = f32(so[f])
        ex = lambda X: rm(X.reshape(T, -1), g.cbe)
        for f in self.EF:
            out[q + f] = ex(w[f])
        for pre, r in (("ru.", w["ru"]), ("rm.", w["rm"])):
            for f in llm.RQ_F:
                out[q + pre + f] = f32(r[f])
            out[q + pre + "jb"] = torch.cat([f32(b) for b in r["jb"]])
        for f in llm.RQ_F:
            out[q + "r2." + f] = f32(w["r2"][f])
        out[q + "r2.jb"] = torch.cat([f32(b) for b in w["r2"]["jb"]])
        return out

    def weight_tensors(self):
        return {self.q + "Wr": _wl(self.Ly.Wr, self.sh.cb, self.g.cbr)}

    def expert_tensor(self, nm):
        """Stack nm in field layout: block j of expert e at column j, offset e * 2^(cb + cbe)."""
        cb, cbe = self.sh.cb, self.g.cbe
        Ws = getattr(self.Ly, nm)
        ci, co = (cbe, cb) if nm == "Wde" else (cb, cbe)
        E, n, nb = Ws.shape[0], Ws[0].numel(), 1 << (cb + cbe)
        out = torch.empty((n // nb, E, nb), dtype=torch.int32, device=_dev())
        for e in range(E):
            out[:, e] = _wl(Ws[e].to(_dev()).contiguous(), ci, co).reshape(n // nb, nb)
        return {self.q + nm: out.reshape(-1)}

    # ------------------------------------------------------------ helpers
    def _pubcol(self, db, name, vec, c, kind=COL):
        llm = _llm()
        ctx = self.ctx
        v = vec.to(torch.int64)
        if kind == COL:
            K = v.numel() >> c

            def fn(pt):
                Xr = v.reshape(K, -1)
                return np.stack([llm.mle_host(ctx, Xr[b], pt) for b in range(K)], axis=1)
            return db.col(name, lambda: llm.f32(v), llm.R_pub(ctx, fn), kind=COL, c=c)

        def fnr(pt):
            return llm.mle_host(ctx, v, pt).reshape(-1, 1)
        return db.col(name, lambda: llm.f32(v), llm.R_pub(ctx, fnr), kind=ROW, c=c, shared=True)

    def _r_slot(self, name, route):
        """Claims of a per-(t, s) ROW (non-shared) column of the expert domain (block s Kde + fb): the
        blocks of one slot hold the same values (checked), routed once per slot."""
        ctx, Kde = self.ctx, self.g.Kde

        def f(pt, vals):
            d = {}
            for b in range(vals.shape[1]):
                if b % Kde == 0:
                    d[b // Kde] = vals[:, b]
                elif not ctx.P and not np.array_equal(np.asarray(vals[:, b]) % P, np.asarray(vals[:, b - b % Kde]) % P):
                    raise ValueError(f"{name}: per-slot column differs across blocks")
            route(pt, d)
        return f

    def rq_rows(self, db, pre, getr, Ktot, cb, asel_key, e0, emax, B, kb):
        """Per-row Requant (|q| <= B) minimality over a row domain with kb blocks (slots); B = 127 with
        e0 = 2 is ShiftQ8 (|a| < 2^29): its minimality relation is divided by 4 (llm.rq_rows)."""
        llm = _llm()
        ctx = self.ctx
        c = {f: db.col(pre + f, (lambda f=f: llm.f32(getr()[f])), llm.R_blk(ctx, pre + f)) for f in llm.RQ_F}
        nb = cb + nbits(Ktot)
        bits = [db.col(pre + f"jb{i}", (lambda i=i: llm.f32(getr()["jb"][i])),
                       llm.R_blk(ctx, pre + "jb", bmap=lambda b, i=i: i * kb + b)) for i in range(nb)]
        asel = db.col(pre + "asel", lambda: llm.f32(getr()["asel"]), llm.R_virt(ctx, asel_key))
        d = db.dom
        e, s, f, sg, wl, wh = (c[x] for x in llm.RQ_F)
        d.lookup(tb.POW, e + tb.POW_OFF, s)
        d.lookup(5, e - e0)
        d.lookup(5, emax - e)
        d.zero(f * (1 - f))
        d.zero((e - e0) * (1 - f))
        d.zero(sg * (1 - sg))
        W = 2 * B + 1
        if e0 == 2 and B == 127:
            q4 = pow(4, P - 2, P)
            d.zero(f * (sg * (asel - 255 * q4 * s) + (1 - sg) * (-asel - 255 * q4 * s - 1)) - (1 << 14) * wh - wl)
            d.lookup(14, wl)
            d.lookup(14, wh)
        else:
            assert W << emax < P - (1 << 30), f"{pre}: minimality relation may wrap"
            d.zero(f * (sg * (4 * asel - W * s) + (1 - sg) * (-4 * asel - W * s - 1)) - (1 << 15) * wh - wl)
            d.lookup(15, wl)
            d.lookup(15, wh)
        for b in bits:
            d.zero(b * (1 - b))
        if Ktot & (Ktot - 1):
            js = bits[0]
            for i, b in enumerate(bits[1:], 1):
                js = js + (1 << i) * b
            d.lookup(24, (Ktot << cb) - 1 - js)
        return c

    # ------------------------------------------------------------ domains (commit and prove passes)
    def steps(self):
        """Every domain of the MoE; the sumchecks of gathers and products follow when active."""
        self.trows()
        self.srows()
        self.omega()
        self.expert_elem()
        self.slots()
        self.router()
        if not self.ctx.active:
            return
        self.gathers()
        self.products()

    def trows(self):
        """Per token: Requant_1023 minimality of the merged row q2."""
        llm, g, q = _llm(), self.g, self.q
        db = llm.DB(self.ctx, q + "trows", self.sh.t)
        self.rq_rows(db, q + "r2.", lambda: self.W()["r2"], g.Ke, g.cbe, (self.l, "e.asel2"), 0, mdl.MOE_EMAX, B2, 1)
        db.run()

    def srows(self):
        """Per (t, s): ShiftQ8 minimality of qu, Requant8 minimality of qm (k blocks)."""
        llm, g, q = _llm(), self.g, self.q
        db = llm.DB(self.ctx, q + "srows", self.sh.t, g.k)
        self.rq_rows(db, q + "ru.", lambda: self.W()["ru"], g.Kde, g.cbe, (self.l, "e.aselu"), 2, 22, 127, g.k)
        self.rq_rows(db, q + "rm.", lambda: self.W()["rm"], g.Kde, g.cbe, (self.l, "e.aselm"), 0, 17, 127, g.k)
        db.run()

    OM_SLOTS = 2                         # slots per domain of the weight / scale lookups

    def omega(self):
        """Per token: router weights (zmax, tau_r, Div), slot scales (Et, fl, omega = ShiftRound(w,
        Et - es)) and gate shift powers, in domains of OM_SLOTS slots each (a domain's lookups are
        bounded by the device transcript's columns per GKR layer); then one domain without lookups
        for the sums over all slots (one slot attains zmax and one attains Et, S = sum_s p_s)."""
        llm, g, q, ctx, M = _llm(), self.g, self.q, self.ctx, self.M
        W = self.W if ctx.P else None
        HW = self.HW if ctx.P else None
        D, k, lk = ctx.D, g.k, g.lk

        def blk(name, s):
            return lambda pt, vals: ctx.claim(name, pt, {s: vals[:, 0]})

        def slotv(name, s):                       # a (k x T) slot tensor at slot s: (bits(s), t)
            return lambda pt, vals: ctx.claim(name, cat(const_pt(D, bits_of(s, lk)), pt), {0: vals[:, 0]})

        def token_cols(db, names):
            src = {self.p + "n2.Eh": lambda: HW().n2["Eh"]}
            src.update({q + f: (lambda f=f: W()[f]) for f in self.TF})
            return [db.col(n, (lambda n=n: llm.f32(src[n]())), llm.R_blk(ctx, n)) for n in names]
        for s0 in range(0, k, self.OM_SLOTS):
            db = llm.DB(ctx, q + f"om{s0}", self.sh.t, 1)
            Et, zmax, S, Eh = token_cols(db, (q + "Et", q + "zmax", q + "S", self.p + "n2.Eh"))
            d = db.dom
            for s in range(s0, min(k, s0 + self.OM_SLOTS)):
                col = lambda nm, fn, route, s=s: db.col(f"{nm}{s}", fn, route)
                em = col("em", lambda s=s: llm.f32(W()["em"][:, s]), blk(q + "rm.e", s))
                eu = col("eu", lambda s=s: llm.f32(W()["eu"][:, s]), blk(q + "ru.e", s))
                xd = col("xd", lambda s=s: llm.f32(W()["xd"][:, s]), slotv(q + "xd", s))
                xg = col("xg", lambda s=s: llm.f32(W()["xg"][:, s]), slotv(q + "xg", s))
                xu = col("xu", lambda s=s: llm.f32(W()["xu"][:, s]), slotv(q + "xu", s))
                zs = col("zs", lambda s=s: llm.f32(W()["zs"][:, s]), slotv(q + "zs", s))
                pp = col("p", lambda s=s: llm.f32(W()["p"][:, s]), slotv(q + "p", s))
                ww = col("w", lambda s=s: llm.f32(W()["w"][:, s]), slotv(q + "w", s))
                fl = col("fl", lambda s=s: llm.f32(W()["fl"][:, s]), blk(q + "fl", s))
                flz = col("flz", lambda s=s: llm.f32(W()["flz"][:, s]), blk(q + "flz", s))
                om = col("om", lambda s=s: llm.f32(W()["om"][:, s]), blk(q + "om", s))
                pkg = col("pkg", lambda s=s: llm.f32(W()["pkg"][:, s]), blk(q + "pkg", s))
                kg = col("kg", lambda s=s: llm.f32(W()["kg"][:, s]), blk(q + "kg", s))
                fg = col("fg", lambda s=s: llm.f32(W()["fg"][:, s]), blk(q + "fg", s))
                so = {f: col("so" + f, (lambda f=f, s=s: llm.f32(llm.sr_pack(W()["so"][:, s])[f])),
                             blk(q + "so." + f, s)) for f in llm.SR_F}
                # router weights: p = tau_r(zmax - zs) (the table domain gives zmax >= zs)
                d.lookup(RTEXP, zmax - zs, pp)
                d.zero(flz * (1 - flz))
                d.zero(flz * (zmax - zs))
                rho = 2 * (1 << mdl.W_FRAC) * pp + S - 2 * S * ww         # w = Div(p, S, 2^W_FRAC)
                d.lookup(24, rho)
                d.lookup(24, 2 * S - 1 - rho)
                d.lookup(mdl.W_FRAC + 1, ww)
                d.lookup(mdl.W_FRAC + 1, (1 << mdl.W_FRAC) - ww)          # w <= 2^14 (p_s <= S)
                # slot scales: omega = ShiftRound(w, Et - es), Et >= es
                es = em + eu + xu + xd
                d.lookup(tb.SR, *[so[f] for f in llm.SR_F])
                d.zero(so["k"] - (Et - es - tb.SR_KMIN))
                d.lookup(so["tR"], ww + so["R"] - so["s"] * om)
                d.lookup(so["tB"], om + so["oB"])
                d.lookup(24, Et - es)
                d.zero(fl * (1 - fl))
                d.zero(fl * (Et - es))
                # gate shift kg = max(kgv, 1) in [1, 15], kgv = eG - xg + Eh, and 2^kg: fg = [kgv < 1]
                # (|kgv| far below 2^24, so the lookups cannot wrap)
                kgv = M.eG - xg + Eh
                d.zero(fg * (1 - fg))
                d.zero(kg - fg - (1 - fg) * kgv)
                d.lookup(24, fg * (0 - kgv))
                d.lookup(5, kg - 1)
                d.lookup(5, mdl.SILU_KMAX - kg)
                d.lookup(tb.POW, kg + tb.POW_OFF, pkg)
            if s0 == 0:
                # the router shift kr = max(krv, 1) in [1, R_KMAX], krv = Eh - eWr - 8, and 2^kr: fr = [krv < 1];
                # the down shift kd = min(kdv, 24) is the 'mo' step's (llm.py), with fd = [kdv > 24]
                krc = db.col("krc", lambda: llm.f32(torch.zeros_like(W()["Et"]).to(torch.int64) + self.Ly.krc_t),
                             llm.R_pub(ctx, lambda pt: pub_c(ctx.D, self.Ly.krc).reshape(-1, 1)))
                kr, pkr, fr = token_cols(db, (q + "kr", q + "pkr", q + "fr"))
                krv = Eh + krc
                d.zero(fr * (1 - fr))
                d.zero(kr - fr - (1 - fr) * krv)
                d.lookup(24, fr * (0 - krv))
                d.lookup(5, kr - 1)
                d.lookup(5, mdl.R_KMAX - kr)
                d.lookup(tb.POW, kr + tb.POW_OFF, pkr)
            db.run()
        # sums over the slots
        db = llm.DB(ctx, q + "omsum", self.sh.t, 1)
        S, = token_cols(db, (q + "S",))
        fsum, zsum, psum = Poly.const(0), Poly.const(0), Poly.const(0)
        for s in range(k):
            fsum = fsum + db.col(f"fl{s}", (lambda s=s: llm.f32(W()["fl"][:, s])), blk(q + "fl", s))
            zsum = zsum + db.col(f"flz{s}", (lambda s=s: llm.f32(W()["flz"][:, s])), blk(q + "flz", s))
            psum = psum + db.col(f"p{s}", (lambda s=s: llm.f32(W()["p"][:, s])), slotv(q + "p", s))
        db.dom.zero(fsum - 1)
        db.dom.zero(zsum - 1)
        db.dom.zero(psum - S)
        db.run()

    def expert_elem(self):
        """Expert values over (cbe + t) x k Kde blocks (block s Kde + fb)."""
        llm, g, q, ctx, M = _llm(), self.g, self.q, self.ctx, self.M
        W = self.W if ctx.P else None
        D, _k, lk, Kde, cbe = ctx.D, g.k, g.lk, g.Kde, g.cbe
        T = self.sh.T
        ex = lambda X: llm.rm(X.reshape(T, -1), cbe)
        db = llm.DB(ctx, q + "e", cbe + self.sh.t, g.Ke)
        accg = db.col("accg", lambda: ex(W()["accg"]), llm.R_virt(ctx, (self.l, "e.accg")))
        accu = db.col("accu", lambda: ex(W()["accu"]), llm.R_virt(ctx, (self.l, "e.accu")))
        cc = {f: db.col(q + f, (lambda f=f: ex(W()[f])), llm.R_blk(ctx, q + f)) for f in self.EF}
        rep = lambda v: llm.f32(v.T.contiguous().repeat_interleave(Kde, dim=0))      # (T, k) -> blocks
        e2 = db.col(q + "r2.e", lambda: llm.f32(W()["r2"]["e"]), llm.R_blk(ctx, q + "r2.e"), kind=ROW, c=cbe, shared=True)
        s2 = db.col(q + "r2.s", lambda: llm.f32(W()["r2"]["s"]), llm.R_blk(ctx, q + "r2.s"), kind=ROW, c=cbe, shared=True)

        def per_slot(name, data, tens, slotten=False):
            def route(pt, dd):
                for s, v in dd.items():
                    if slotten:
                        ctx.claim(q + tens, cat(const_pt(D, bits_of(s, lk)), pt), {0: v})
                    else:
                        ctx.claim(q + tens, pt, {s: v})
            return db.col(name, data, self._r_slot(name, route), kind=ROW, c=cbe)
        kg = per_slot("kg", lambda: rep(W()["kg"]), "kg")
        pkg = per_slot("pkg", lambda: rep(W()["pkg"]), "pkg")
        eu = per_slot("ru.e", lambda: rep(W()["eu"]), "ru.e")
        su = per_slot("ru.s", lambda: rep(mdl.pow2(W()["eu"])), "ru.s")
        em = per_slot("rm.e", lambda: rep(W()["em"]), "rm.e")
        sm = per_slot("rm.s", lambda: rep(mdl.pow2(W()["em"])), "rm.s")
        om = per_slot("om", lambda: rep(W()["om"]), "om")
        d = db.dom
        c, a, qu, qm, q2 = (cc[f] for f in self.EF)
        d.lookup(kg, accg + pkg * INV2 - pkg * c)                        # c = ShiftRound(accg, kg)
        d.lookup(tb.SILU, c + (1 << 14), a)
        d.lookup(eu, accu + su * INV2 - su * qu)                         # qu = ShiftQ8(accu)
        d.lookup(tb.STAR8, qu + 127)
        d.lookup(em + 1, 2 * a * qu - 2 * sm * qm + sm)                  # qm = Requant8(a qu)
        d.lookup(tb.STAR8, qm + 127)
        d.lookup(e2 + 1, 2 * qm * om - 2 * s2 * q2 + s2)                 # q2 = Requant_1023(qm omega)
        d.lookup(11, q2 + B2)
        d.lookup(11, B2 - q2)
        db.run()

    def slots(self):
        """Slot side of the multiset: (t, e, z, eWg, eWu, eWd) per (s, t), multiplicity +1."""
        llm, g, q, ctx = _llm(), self.g, self.q, self.ctx
        W = self.W if ctx.P else None
        lk = g.lk
        db = llm.DB(ctx, q + "s", lk + self.sh.t, 1)
        sl = lambda f: (lambda: llm.rm(W()[f], lk))
        bits = [db.col(f"eb{i}", (lambda i=i: llm.rm((W()["idx"] >> i) & 1, lk)),
                       llm.R_blk(ctx, q + "eb", bmap=lambda b, i=i: i)) for i in range(g.nbE)]
        xs = [db.col(q + f, sl(f), llm.R_blk(ctx, q + f)) for f in ("zs", "xg", "xu", "xd")]
        ti = self._pubcol(db, "ti", torch.arange(self.sh.T, device=_dev()), lk, kind=ROW)
        d = db.dom
        e = Poly.const(0)
        for i, b in enumerate(bits):
            d.zero(b * (1 - b))
            e = e + (1 << i) * b
        if g.E & (g.E - 1):
            d.lookup(24, g.E - 1 - e)
        d.lookup(self.dyn, ti, e, *xs, dynamic=True)
        db.run()

    def router(self):
        """Router SatShift, top-k and the candidate side of the multiset (multiplicity -sel)."""
        llm, g, q, ctx = _llm(), self.g, self.q, self.ctx
        W = self.W if ctx.P else None
        Ly, E, cbr = self.Ly, g.E, g.cbr
        db = llm.DB(ctx, q + "r", cbr + self.sh.t, g.KE)
        accr = db.col("accr", lambda: llm.rm(W()["acc_r"], cbr), llm.R_virt(ctx, (self.l, "e.accr")))
        cc = {f: db.col(q + f, (lambda f=f: llm.rm(W()[f], cbr)), llm.R_blk(ctx, q + f))
              for f in ("z", "zo", "zwh", "sel", "dfl")}
        rowc = lambda name, fn: db.col(name, fn, llm.R_blk(ctx, name), kind=ROW, c=cbr, shared=True)
        kr = rowc(q + "kr", lambda: llm.f32(W()["kr"]))           # the clamped router shift (om domain)
        pkr = rowc(q + "pkr", lambda: llm.f32(W()["pkr"]))
        th = rowc(q + "th", lambda: llm.f32(W()["th"]))
        jb = [db.col(f"jt{i}", (lambda i=i: llm.f32((W()["jt"] >> i) & 1)),
                     llm.R_blk(ctx, q + "jt", bmap=lambda b, i=i: i), kind=ROW, c=cbr, shared=True)
              for i in range(g.nbE)]
        jt = sum(((1 << i) * b for i, b in enumerate(jb)), Poly.const(0))
        jidx = self._pubcol(db, "jidx", torch.arange(E, device=_dev()), cbr)
        xs = [self._pubcol(db, nm, v.to(torch.int64), cbr) for nm, v in (("xgp", Ly.eWge), ("xup", Ly.eWue), ("xdp", Ly.eWde))]
        ti = self._pubcol(db, "ti", torch.arange(self.sh.T, device=_dev()), cbr, kind=ROW)
        d = db.dom
        z, zo, zwh = cc["z"], cc["zo"], cc["zwh"]
        s = pkr
        o2 = zo * zo
        u = accr + s * INV2 - s * z
        wv = zo * accr - (2 * mdl.R_SAT + 1) * s * INV2 - (o2 - zo) * INV2 - (1 << 15) * zwh
        d.lookup((1 - o2) * kr + o2 * 15, (1 - o2) * u + o2 * wv)        # z = SatShift(accr, kr)
        d.lookup(15, o2 * zwh)
        d.zero(o2 * zo - zo)
        d.zero(o2 * (z - mdl.R_SAT * zo))
        d.lookup(16, z + mdl.R_SAT)
        d.lookup(16, mdl.R_SAT - z)
        sel, df = cc["sel"], cc["dfl"]
        d.zero(sel * (1 - sel))
        d.zero(df * (1 - df))
        d.zero(df * (z - th))
        d.lookup(24, (1 - df) * (sel * (z - th - 1) + (1 - sel) * (th - z - 1))
                 + df * (sel * (jt - jidx) + (1 - sel) * (jidx - jt - 1)))
        d.lookup(self.dyn, ti, jidx, z, *xs, num=-sel)
        db.run()
        # per token: threshold index bits and the count of selected candidates
        db = llm.DB(ctx, q + "rrows", self.sh.t, 1)
        jb = [db.col(f"jt{i}", (lambda i=i: llm.f32((W()["jt"] >> i) & 1)), llm.R_blk(ctx, q + "jt", bmap=lambda b, i=i: i))
              for i in range(g.nbE)]

        def r_th(pt, vals):
            ctx.claim(q + "th", pt, {0: vals[:, 0]})
            ctx.put((self.l, "e.th"), (pt, vals))
        db.col(q + "th", lambda: llm.f32(W()["th"]), r_th)
        cnt = db.col("selcnt", lambda: llm.f32(W()["sel"].sum(1)), llm.R_virt(ctx, (self.l, "e.selcnt")))
        jt = sum(((1 << i) * b for i, b in enumerate(jb)), Poly.const(0))
        for b in jb:
            db.dom.zero(b * (1 - b))
        if E & (E - 1):
            db.dom.lookup(24, E - 1 - jt)
        db.dom.zero(cnt - g.k)
        db.run()

    # ------------------------------------------------------------ sumchecks
    def gathers(self):
        llm, g, q, ctx = _llm(), self.g, self.q, self.ctx
        W = self.W if ctx.P else None
        t, cbr, KE, cbe, Kde, k, T = self.sh.t, g.cbr, g.KE, g.cbe, g.Kde, g.k, self.sh.T
        ex = lambda X: llm.rm(X.reshape(T, -1), cbe)
        # threshold th = z(jt) and the count sum sel^2 = k
        (pt, vth), = ctx.take((self.l, "e.th"))
        (pc, vc), = ctx.take((self.l, "e.selcnt"))
        if not ctx.P:
            assert np.array_equal(np.asarray(pt), np.asarray(pc))
        gather_rowsum(ctx, b"moe-th", t, pt, KE, cbr,
                      [(q + "z", lambda: llm.rm(W()["z"], cbr), llm.R_blk(ctx, q + "z")),
                       (q + "sel", lambda: llm.rm(W()["sel"], cbr), llm.R_blk(ctx, q + "sel"))],
                      lambda c: c[0], 1, vth[:, 0], vc[:, 0], q + "jt", lambda: bits_of(W()["jt"], g.nbE))
        # Requant minimality of qu and qm per (t, s), and of the merged token row
        for key, pre, h, cols in (
                ("e.aselu", "ru.", lambda c: c[0], lambda s: [("accu", (lambda s=s: llm.rm(W()["accu"][:, s], cbe)), self._accu_route(s))]),
                ("e.aselm", "rm.", lambda c: c[0] * c[1], lambda s: [
                    (q + "a", (lambda s=s: llm.rm(W()["a"][:, s], cbe)), llm.R_blk(ctx, q + "a", bmap=lambda b, s=s: s * Kde + b)),
                    (q + "qu", (lambda s=s: llm.rm(W()["qu"][:, s], cbe)), llm.R_blk(ctx, q + "qu", bmap=lambda b, s=s: s * Kde + b))])):
            (pa, va), = ctx.take((self.l, key))
            for s in range(k):
                gather_rowsum(ctx, b"moe-g" + pre.encode() + b"%d" % s, t, pa, Kde, cbe, cols(s), h, None, va[:, s], None,
                              _JbSlot(q + pre + "jb", s, k, g.nbu),
                              (lambda s=s, pre=pre: [b.reshape(k, T)[s] for b in W()[pre[:-1]]["jb"]]))
        (pa, va), = ctx.take((self.l, "e.asel2"))
        gather_rowsum(ctx, b"moe-g2", t, pa, g.Ke, cbe,
                      [(q + "qm", lambda: ex(W()["qm"]), llm.R_blk(ctx, q + "qm")),
                       ("omx", lambda: llm.f32(W()["om"].unsqueeze(-1).expand(T, k, g.de).reshape(T, -1)
                                               .reshape(T, g.Ke, 1 << cbe).permute(1, 0, 2)), self._om_route())],
                      lambda c: c[0] * c[1], None, va[:, 0], None, q + "r2.jb", lambda: W()["r2"]["jb"])

    def _accu_route(self, s):
        """Claims on accu at slot s (blocks fb of the expert domain s Kde + fb) -> the up product."""
        def f(rho, vals):
            self.ctx.put((self.l, "e.accu.g"), (s, rho, vals))
        return f

    def _om_route(self):
        """omega as a FULL column over (cbe + t) x k Kde: per block, constant over the in-block f bits."""
        ctx, q, Kde, cbe = self.ctx, self.q, self.g.Kde, self.g.cbe

        def f(rho, vals):
            for b in range(vals.shape[1]):
                if b % Kde == 0:
                    ctx.claim(q + "om", rho[:, cbe:], {b // Kde: vals[:, b]})
                elif not ctx.P and not np.array_equal(np.asarray(vals[:, b]) % P, np.asarray(vals[:, b - b % Kde]) % P):
                    raise ValueError("omega blocks")
        return f

    def products(self):
        """Router matmul; gate / up and down as gathered products."""
        llm, g, q, ctx = _llm(), self.g, self.q, self.ctx
        sh, Ly = self.sh, self.Ly
        P_ = ctx.P
        HW = self.HW if P_ else None
        cb, T, Kd = sh.cb, sh.T, sh.Kd
        cl = []
        for pt, vals in ctx.take((self.l, "e.accr")):
            po, v = llm.combine_blocks(ctx, pt[:, :g.cbr], vals, nbits(g.KE))
            cl.append((po, pt[:, g.cbr:], v))
        llm.matmul(ctx, b"moe-router", self.p + "qp2", (lambda: llm.rm(HW().n2["qp"], cb)) if P_ else None, Kd, T,
                   q + "Wr", Ly.Wr if P_ else None, g.KE, cb, cl, cbo=g.cbr)
        # gate and up at the expert domain's point (one claim pair), then the up gathers' claims
        (pg, vg), = ctx.take((self.l, "e.accg"))
        (pu, vu), = ctx.take((self.l, "e.accu"))
        if not P_:
            assert np.array_equal(np.asarray(pg), np.asarray(pu))
        self.gathered(b"moe-gu", pg, [vg, vu], [q + "Wge", q + "Wue"], [Ly.Wge, Ly.Wue])
        for s, rho, vals in ctx.take((self.l, "e.accu.g")):
            self.gathered_slot(b"moe-ug%d" % s, s, rho, vals, q + "Wue", Ly.Wue)
        for pdn, vdn in ctx.take((self.l, "accd")):
            self.down(pdn, vdn)

    def gathered(self, label, pt, vals_list, wnames, Ws):
        """Claims acc_i(s, t, n) at the expert-domain point pt (vals (D, k Kde) each), with
        acc_i = sum_c qp2(t, c) W_i[e(t, s)](c, n) -> a qp2 claim and stacked weight claims."""
        g, ctx, sh = self.g, self.ctx, self.sh
        cbe, Kde, k, _t = g.cbe, g.Kde, g.k, sh.t
        F, _Gx, D = ctx.F, ctx.G, ctx.D
        P_ = ctx.P
        log2(self.M.cfg.d)
        tau = ctx.chal(nbits(Kde)) if Kde > 1 else np.zeros((D, 0), dtype=np.int64)
        gam = ctx.chal(len(vals_list))
        rn = cat(pt[:, :cbe], tau)                      # point over de (bits cbe + nbits(Kde))
        rt = pt[:, cbe:]
        bw = ctx.chal(k)
        vs = []
        for s in range(k):
            v = F.zero
            for i, vals in enumerate(vals_list):
                vv = ctx.blockcomb(tau, np.asarray(vals)[:, s * Kde:(s + 1) * Kde]) if not P_ else None
                if not P_:
                    v = F.add(v, F.mul(F.c(gam[:, i]), F.c(vv)))
            vs.append(np.array(v, dtype=np.int64))
        claim = F.zero
        if not P_:
            for s in range(k):
                claim = F.add(claim, F.mul(F.c(bw[:, s]), F.c(vs[s])))
        self._gath1(label, rn, rt, gam, bw, np.array(claim, dtype=np.int64) if not P_ else np.zeros(D, dtype=np.int64),
                    wnames, Ws, list(range(k)))

    def gathered_slot(self, label, s, rho, vals, wname, Wst):
        """A claim on accu at slot s only (from a minimality gather: vals (D, Kde) over the de blocks)."""
        g, ctx = self.g, self.ctx
        F, D = ctx.F, ctx.D
        cbe, Kde = g.cbe, g.Kde
        tau = ctx.chal(nbits(Kde)) if Kde > 1 else np.zeros((D, 0), dtype=np.int64)
        v = ctx.blockcomb(tau, np.asarray(vals)) if not ctx.P else None
        rn = cat(rho[:, :cbe], tau)
        rt = rho[:, cbe:]
        gam = np.zeros((D, 1), dtype=np.int64)
        gam[0, 0] = 1
        bw = np.zeros((D, g.k), dtype=np.int64)
        bw[0, s] = 1
        claim = np.zeros(D, dtype=np.int64) if ctx.P else np.array(F.c(v), dtype=np.int64)
        self._gath1(label, rn, rt, gam, bw, claim, [wname], [Wst], [s])

    def _gath1(self, label, rn, rt, gam, bw, claim, wnames, Ws, slots):
        llm, g, ctx, sh = _llm(), self.g, self.ctx, self.sh
        k, t = g.k, sh.t
        F, Gx, D = ctx.F, ctx.G, ctx.D
        P_ = ctx.P
        HW = self.HW if P_ else None
        d = self.M.cfg.d
        ld = log2(d)

        def wfold(rn_, Wst):
            """(E, d): W_e(c, rn) for every expert (stacked (E, d, de) int8)."""
            e_n = Gx.eq_table(rn_)[:, :g.de].contiguous()
            return Gx.rowdot(_rowmajor(Wst).reshape(-1), e_n, g.E * d, g.de)

        def tabs1():
            Wc = None
            for i, Wst in enumerate(Ws):
                f = Gx.scale_p(wfold(rn, Wst), gam[:, i])
                Wc = f if Wc is None else Gx.add(Wc, f)
            Wc = Wc.reshape(D, g.E, d)
            ids = self.W()["idx"]
            Wg = Wc[:, ids.T.reshape(-1)]                                     # (D, k T, d): [slot][t][c]
            A = llm.f32(HW().n2["qp"]).reshape(-1)                            # [t][c]
            return [Tab(Gx.eq_table(rt), ROW, ld, shared=True), Tab(A, ROW, 0, shared=True),
                    Tab(Wg.reshape(D, -1).contiguous(), FULL)]
        shapes = [(ROW, ld), (ROW, 0), (FULL, 0)]
        rho, fin = ctx.sumcheck(label + b"-1", ld + t, k, shapes, claim, tabs=tabs1 if ctx.proving else None,
                                terms=[(np.array(F.one), [0, 1, 2])], bw=bw)
        rc, rt2 = rho[:, :ld], rho[:, ld:]
        llm.check_eq(ctx, label, fin[0], rt, rt2)
        cb = sh.cb
        ctx.point_claim(self.p + "qp2", cat(rc[:, :cb], rt2, rc[:, cb:]), fin[1][:, 0],
                        data=llm.rm(HW().n2["qp"], cb) if P_ else None)
        self.phase2(label, rt2, {s: fin[2][:, s] for s in slots}, rc, rn, wnames, Ws, gam, inner="c")

    def phase2(self, label, rt, wvals, rx, rn, wnames, Ws, gam, inner):
        """Gathered rows W_i[e(t, s)](x, n) at (rx, rt) per slot s in wvals (combined by gam) ->
        one-hot claims per slot (bit planes of the slot's expert index) and stacked weight claims."""
        llm, g, ctx, sh = _llm(), self.g, self.ctx, self.sh
        k, t, T, nbE, E = g.k, sh.t, sh.T, g.nbE, g.E
        F, Gx, D = ctx.F, ctx.G, ctx.D
        P_ = ctx.P
        if len(wvals) == k:
            bw = ctx.chal(k)
        else:                                     # one slot (a minimality gather's claim)
            (s1,) = wvals
            bw = np.zeros((D, k), dtype=np.int64)
            bw[0, s1] = 1
        claim = F.zero
        if not P_:
            for s, v in wvals.items():
                claim = F.add(claim, F.mul(F.c(bw[:, s]), F.c(v)))
        d = self.M.cfg.d
        xin, xout = (d, g.de) if inner == "c" else (g.de, d)

        def wv(Wst):
            """(D, 2^nbE): W_e(rx, rn) per expert, zero padded."""
            ex = Gx.eq_table(rx)[:, :xin].contiguous()
            eo = Gx.eq_table(rn)[:, :xout].contiguous()
            f1 = Gx.rowdot(_rowmajor(Wst).reshape(-1), eo, E * xin, xout).reshape(D, E, xin)
            out = torch.zeros((D, 1 << nbE), dtype=torch.int32, device=_dev())
            for e in range(E):
                out[:, e] = (Gx.mul(f1[:, e].contiguous(), ex).to(torch.int64).sum(dim=1) % P).to(torch.int32)
            return out
        wvs = []

        def tabs2():
            Wc = None
            for i, Wst in enumerate(Ws):
                f = wv(Wst)
                wvs.append(f)
                gg = Gx.scale_p(f, gam[:, i])
                Wc = gg if Wc is None else Gx.add(Wc, gg)
            ids = self.W()["idx"]
            OH = torch.zeros((k, T, 1 << nbE), dtype=torch.int32, device=_dev())
            OH.scatter_(2, ids.T.unsqueeze(-1).to(torch.int64), 1)
            return [Tab(OH.reshape(-1), FULL), Tab(Wc, COL, nbE, shared=True), Tab(Gx.eq_table(rt), ROW, nbE, shared=True)]
        shapes = [(FULL, 0), (COL, nbE), (ROW, nbE)]
        rho, fin = ctx.sumcheck(label + b"-2", nbE + t, k, shapes, np.array(claim, dtype=np.int64),
                                tabs=tabs2 if ctx.proving else None, terms=[(np.array(F.one), [0, 1, 2])], bw=bw)
        re_, rt2 = rho[:, :nbE], rho[:, nbE:]
        llm.check_eq(ctx, label + b"-2", fin[2], rt, rt2)
        sent = None
        if P_:
            e = Gx.eq_table(re_)
            sent = Gx.dots_dev([(w_, e) for w_ in wvs]).T.contiguous()          # (D, len(Ws)), no sync
        sent = ctx.send_vals(sent, (D, len(Ws)))
        if not P_:
            tot = F.zero
            for i in range(len(Ws)):
                tot = F.add(tot, F.mul(F.c(gam[:, i]), F.c(sent[:, i])))
            if tot != F.c(fin[1][:, 0]):
                raise ValueError("expert weight values")
        for i, name in enumerate(wnames):
            self.weight_claim(name, re_, rx, rn, inner, sent[:, i])
        for s in wvals:
            r2, f2 = oh_slot(ctx, label + b"%d" % s, nbE, t, cat(re_, rt2), fin[0][:, s],
                             (lambda s=s: bits_of(self.W()["idx"][:, s], nbE)))
            for i in range(nbE):
                ctx.claim(self.q + "eb", cat(const_pt(D, bits_of(s, g.lk)), r2), {i: f2[i][:, 0]})

    def weight_claim(self, name, re_, rx, rn, inner, val):
        """W(re, rx, rn) on a stacked expert tensor: columns (xblock, nblock), in-column (n' low, x', e high)."""
        g, ctx, sh = self.g, self.ctx, self.sh
        D = ctx.D
        cb, cbe = sh.cb, g.cbe
        d = self.M.cfg.d
        cin, cout = (cb, cbe) if inner == "c" else (cbe, cb)
        Kin = (d if inner == "c" else g.de) >> cin
        Kout = (g.de if inner == "c" else d) >> cout
        wb = {}
        if ctx.P:                                 # coefficients on the device (the points stay lazy)
            Gx = ctx.G
            ei, eo = _eq_dev(Gx, D, rx[:, cin:], Kin), _eq_dev(Gx, D, rn[:, cout:], Kout)
            w = Gx.mul(ei.repeat_interleave(Kout, dim=1).contiguous(), eo.repeat(1, Kin).contiguous())
            for j in range(Kin * Kout):
                wb[j] = w[:, j]
        else:
            ei = _eq_host(D, rx[:, cin:], Kin)
            eo = _eq_host(D, rn[:, cout:], Kout)
            F = ctx.F
            for a in range(Kin):
                for b in range(Kout):
                    wb[a * Kout + b] = np.array(F.mul(F.c(ei[:, a]), F.c(eo[:, b])), dtype=np.int64)
        ctx.wclaim(name, cat(rn[:, :cout], rx[:, :cin], re_), wb, val)

    def down(self, pt, vals):
        """acc_d(t, n) = sum_s sum_f q2(t, s, f) Wd[e(t, s)](f, n) at the 'mo' domain's point."""
        llm, g, ctx, sh = _llm(), self.g, self.ctx, self.sh
        cb, cbe, Kde, k, t, T, Kd = sh.cb, g.cbe, g.Kde, g.k, sh.t, sh.T, sh.Kd
        F, Gx, D = ctx.F, ctx.G, ctx.D
        P_ = ctx.P
        W = self.W if P_ else None
        lde = nbits(g.de)
        d = self.M.cfg.d
        po, v = llm.combine_blocks(ctx, pt[:, :cb], vals, nbits(Kd))
        rn, rt = po, pt[:, cb:]
        bw1 = np.zeros((D, k), dtype=np.int64)
        bw1[0, :] = 1

        def tabs1():
            e_n = Gx.eq_table(rn)[:, :d].contiguous()
            f = Gx.rowdot(_rowmajor(self.Ly.Wde).reshape(-1), e_n, g.E * g.de, d).reshape(D, g.E, g.de)
            ids = W()["idx"]
            Wg = f[:, ids.T.reshape(-1)]                                       # (D, k T, de)
            Wp = torch.zeros((D, k * T, 1 << lde), dtype=torch.int32, device=_dev())
            Wp[:, :, :g.de] = Wg
            A = torch.zeros((k, T, 1 << lde), dtype=torch.int32, device=_dev())
            A[:, :, :g.de] = llm.f32(W()["q2"].permute(1, 0, 2).contiguous()).reshape(k, T, g.de)
            return [Tab(Gx.eq_table(rt), ROW, lde, shared=True), Tab(A.reshape(-1), FULL), Tab(Wp.reshape(D, -1).contiguous(), FULL)]
        shapes = [(ROW, lde), (FULL, 0), (FULL, 0)]
        rho, fin = ctx.sumcheck(b"moe-d-1", lde + t, k, shapes, np.array(v) if not P_ else np.zeros(D, dtype=np.int64),
                                tabs=tabs1 if ctx.proving else None, terms=[(np.array(F.one), [0, 1, 2])], bw=bw1)
        rf, rt2 = rho[:, :lde], rho[:, lde:]
        llm.check_eq(ctx, b"moe-d", fin[0], rt, rt2)
        # q2 at (rf, rt2) per slot: the de dimension is padded to 2^lde with zeros in the sumcheck, so the
        # claim is on q2's MLE over (cbe, fb) with fb = rf's high bits restricted to Kde blocks
        # q2 at (rf, rt2) per slot: the sumcheck pads de to 2^lde; the prover sends the slot's Kde block
        # values at (rf_lo, rt2), which the verifier combines with eq(rf_hi, fb)
        pin = cat(rf[:, :cbe], rt2)
        fhi = rf[:, cbe:]
        ehi = _eq_host(D, fhi, Kde) if not P_ else None
        for s in range(k):
            vals = None
            if P_:
                X = llm.rm(W()["q2"][:, s], cbe)                                # Kde blocks of (cbe + t)
                vals = Gx.rowdot(X, Gx.eq_table(pin), Kde, 1 << (cbe + t))
            vals = ctx.send_vals(vals, (D, Kde))
            if not P_:
                tot = F.zero
                for b in range(Kde):
                    tot = F.add(tot, F.mul(F.c(ehi[:, b]), F.c(vals[:, b])))
                if tot != F.c(fin[1][:, s]):
                    raise ValueError("moe-d: q2 block values")
            ctx.claim(self.q + "q2", pin, {s * Kde + b: vals[:, b] for b in range(Kde)})
        gam = np.zeros((D, 1), dtype=np.int64)
        gam[0, 0] = 1
        self.phase2(b"moe-d", rt2, {s: fin[2][:, s] for s in range(k)}, rf, rn, [self.q + "Wde"], [self.Ly.Wde],
                    gam, inner="f")


# =================================================================== helpers
def _wl(Wn, cbi, cbo):
    """Weight (Nin x Nout) -> blocks (tb, ob), in-block (o' low, t' high) (llm.weight_tensors)."""
    f32 = _llm().f32
    Nin, Nout = Wn.shape
    Ki, Ko = Nin >> cbi, Nout >> cbo
    return f32(Wn.contiguous().reshape(Ki, 1 << cbi, Ko, 1 << cbo).permute(0, 2, 1, 3))


def _rowmajor(Wst):
    """A stacked (E, x, y) expert tensor (stored column-major per expert) in row-major order on the device."""
    W = Wst.to(_dev())
    return W.contiguous()


def _eq_dev(Gx, D, r, n):
    """eq(r, b) for b < n on the device (D, n) int32, r possibly lazy."""
    if r.shape[1] == 0:
        out = torch.zeros((D, max(n, 1)), dtype=torch.int32, device=_dev())
        out[0] = 1
        return out[:, :n]
    return Gx.eq_table(r)[:, :n].contiguous()


def _eq_host(D, r, n):
    if r.shape[1] == 0:
        out = np.zeros((D, max(n, 1)), dtype=np.int64)
        out[0, :] = 1
        return out[:, :n]
    return _llm().eq_host(D, r, n)


def _oh_coefs(ctx, x):
    """Coefficients (2 x_i - 1, 1 - x_i) per column of a point x (D, n): on the device for the
    prover (x may be lazy; no host read), host values for the verifier."""
    if ctx.P:
        from .lazy import Lz, to_dev
        X = to_dev(x).to(torch.int64)
        one = torch.zeros((ctx.D, 1), dtype=torch.int64, device=X.device)
        one[0] = 1
        c1, c0 = ((2 * X - one) % P).to(torch.int32), ((one - X) % P).to(torch.int32)
        return [c for i in range(x.shape[1]) for c in (Lz(c1[:, i].contiguous()), Lz(c0[:, i].contiguous()))]
    F = ctx.F
    out = []
    for i in range(x.shape[1]):
        xi = x[:, i]
        out += [np.array(F.sub(F.add(F.c(xi), F.c(xi)), F.one), dtype=np.int64),
                np.array(F.sub(F.one, F.c(xi)), dtype=np.int64)]
    return out


def oh_slot(ctx, label, nbE, t, pt, val, getbits):
    """A claim OH(pt) on the one-hot of a slot's expert index -> claims on the bit planes at a point."""
    pe, pm = pt[:, :nbE], pt[:, nbE:]
    coefs, polys, expr = _oh_coefs(ctx, pe), [], None
    for i in range(nbE):
        polys.append(Poly.var(i))
        e = ("+", ("*", ("c", 2 * i), ("p", i)), ("c", 2 * i + 1))
        expr = e if expr is None else ("*", expr, e)
    prog = Prog(polys, expr, coefs)
    f32 = _llm().f32
    claim = np.zeros(ctx.D, dtype=np.int64) if ctx.P else np.asarray(val)    # the prover's is unused
    return ctx.sumcheck(label + b"-oh", t, 1, [(FULL, 0)] * nbE, claim,
                        tabs=(lambda: [Tab(f32(b), FULL) for b in getbits()]) if ctx.proving else None,
                        prog=prog, eq_r=pm)


class _JbSlot(str):
    """The name of a per-slot bit tensor (k slots): bit i of slot s is block i k + s."""

    def __new__(cls, name, s, K1, nb):
        o = str.__new__(cls, name)
        o.s, o.K1, o.nb = s, K1, nb
        return o


def gather_rowsum(ctx, label, n_rows, rowpt, K, cb, cols, h, qi, asel, sq, jb_name, getbits):
    """sum_{m,x} eq(r, m) [OH(m, x) h(m, x) + gamma q(m, x)^2] = asel(r) + gamma sq(r) over a
    row-major (rows x K 2^cb) domain; cols [(name, data_fn, route)], h(polys) -> Poly, qi the index of
    q in cols (or None: no row sum)."""
    llm = _llm()
    F = ctx.F
    n = cb + n_rows
    gam = ctx.chal()
    if ctx.P:
        claim = np.zeros(ctx.D, dtype=np.int64)
    else:
        claim = F.c(asel)
        if qi is not None:
            claim = F.add(claim, F.mul(F.c(gam), F.c(sq)))
    nc = len(cols)
    polys = [Poly.var(1) * h([Poly.var(2 + i) for i in range(nc)]), Poly.var(0)]
    expr = ("*", ("p", 1), ("p", 0))
    coefs = []
    if qi is not None:
        qv = Poly.var(2 + qi)
        polys.append(qv * qv)
        coefs.append(gam if ctx.P else np.asarray(gam, dtype=np.int64))
        expr = ("*", ("p", 1), ("+", ("p", 0), ("*", ("c", 0), ("p", 2))))
    prog = Prog(polys, expr, coefs)
    shapes = [(ROW, cb), (FULL, 0)] + [(FULL, 0)] * nc

    def tabs():
        Gx = ctx.G
        eqr = Gx.eq_table(rowpt)
        bits = getbits()
        j = torch.zeros_like(bits[0])
        for i, b in enumerate(bits):
            j = j | (b << i)
        OH = torch.zeros((K, 1 << n_rows, 1 << cb), dtype=torch.int32, device=_dev())
        m = torch.arange(1 << n_rows, device=_dev())
        OH.index_put_((j >> cb, m, j & ((1 << cb) - 1)), torch.ones_like(m, dtype=torch.int32))
        return [Tab(eqr, ROW, cb, shared=True), Tab(OH.reshape(-1), FULL)] + [Tab(f(), FULL) for _, f, _ in cols]
    rho, fin = ctx.sumcheck(label, n, K, shapes, np.array(claim), tabs=tabs if ctx.proving else None, prog=prog)
    llm.check_eq(ctx, label, fin[0], rowpt, rho[:, cb:])
    for i, (name, _, route) in enumerate(cols):
        route(rho, fin[2 + i])
    oh_check_m(ctx, label, jb_name, (cb, nbits(K)), rho, fin[1], n_rows, getbits)


def oh_check_m(ctx, label, jb_name, nbit, pt, vals, t, data_bits):
    """llm.oh_check for one claim; jb_name may be a _JbSlot (bit i of slot s is block i K1 + s)."""
    cb, nbk = nbit
    tv = ctx.chal(nbk) if nbk else np.zeros((ctx.D, 0), dtype=np.int64)
    claim = ctx.blockcomb(tv, vals)
    pj, pm = pt[:, :cb], pt[:, cb:]
    nb = cb + nbk
    coefs, polys, expr = _oh_coefs(ctx, cat(pj, tv)), [], None
    for i in range(nb):
        polys.append(Poly.var(i))
        e = ("+", ("*", ("c", 2 * i), ("p", i)), ("c", 2 * i + 1))
        expr = e if expr is None else ("*", expr, e)
    prog = Prog(polys, expr, coefs)
    f32 = _llm().f32
    rho, fin = ctx.sumcheck(label + b"-oh", t, 1, [(FULL, 0)] * nb, claim,
                            tabs=(lambda: [Tab(f32(b), FULL) for b in data_bits()]) if ctx.proving else None,
                            prog=prog, eq_r=pm)
    if isinstance(jb_name, _JbSlot):
        ctx.claim(str(jb_name), rho, {i * jb_name.K1 + jb_name.s: fin[i][:, 0] for i in range(nb)})
    else:
        ctx.claim(jb_name, rho, {i: fin[i][:, 0] for i in range(nb)})
