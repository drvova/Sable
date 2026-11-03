"""Global multilinear PCS (profile O): rate-1/2 RS over KoalaBear, cosets of 4,
BLAKE3 Merkle subtrees with caps, one mixed-length BaseFold for the whole proof.

A unit (commitment group) holds base-field tensors of K blocks x 2^c0.  Each
block is split into columns of at most 2^CMAX entries; columns are grouped by
size class c = clamp(c0, CMIN, CMAX).  Per (unit, class), the codewords of all
columns form one tree whose leaf l holds symbols 4l..4l+3 of every column:
BLAKE3 of the bytes for <= 64 columns, else a hash chain over 64-column chunk
digests.  Each tree sends its cap (CAPLOG levels below the root).

Passes (the prover recomputes the tensors in P3, P4 and P5):
  P1 commit: caps.
  P2 open (per unit, once its claims are known): per class, one sumcheck in Fq
     reduces the unit's claims to a point rho; f_k(rho) is sent for every column;
     alpha'_k in Fo; g = sum alpha'_k f_k, y = g(rho).
  finish:
     lambda in Fo per (unit, class); a sumcheck in Fo over NV global variables
       sum_x sum_t lambda_t eq(rho_t, x_c) g_t(x_c),  x_c = x[NV - c:],
     whose claim is sum_t 2^(NV - c) lambda_t y_t.  The prover runs its first NLOW
     rounds on v_t = g_t(., rho_hi) (kept from P2), then (P3) recomputes
     w_t = g_t(r_lo, .) for the rest.  At the end the claim is
       sum_c H_c(r_c),  H_c = sum_{t in c} lambda_t eq(rho_t, r_c) g_t,
     a fixed combination of committed columns.
     P4: H_c.  BaseFold proves sum_c H_c(r_c) = V: sumcheck of eq(r, x) Phi(x),
       Phi(x) = sum_c H_c(x[NV - c:]); class c joins the folded codeword (and
       the folded table) at level NV - c; oracles at even levels below the
       final level NV - FLOG, whose message is sent in the clear.
     Grinding, QUERIES positions q in [0, 2^(NV-1)); class c opens leaf
       q >> (NV - c), oracle level L opens leaf q >> L.
     P5: leaves and paths of every tree."""
import os
import time
import numpy as np
import torch
import blake3
from .field import P
from . import kern, codec, rs
from .transcript import label_name
from .kern import ptr, i64, i32
from .ext import gops, _grid, NT
from .fe import fe
from . import sumcheck as sc
from .sumcheck import Tab, FULL
from .ext import dev as _dev

CMIN = 5
CMAX = 23
NLOW = 12
CAPLOG = 7
QUERIES = 153
GRIND = 30
FLOG = 5
# host memory for the combined columns g_t of P3, so that P4 combines them instead of rebuilding
# the witness (ZK_GPCS_G bytes; at most half of the available memory)
G_BUDGET = int(os.environ.get("ZK_GPCS_G", str(48 << 30)))
# keep the Merkle tree levels of committed units in host memory, so the query pass encodes only the
# queried tiles of each column and takes paths from them instead of re-hashing every leaf
KEEP_TREES = bool(int(os.environ.get("ZK_KEEPTREES", "0")))
CHUNK = 64
DQ, DO = 4, 8
RED_BYTES = 8 << 30
_KM = {}
_ITW = {}


def _k(D):
    if D not in _KM:
        m = kern.file_module("gpcs.cu", D)
        _KM[D] = {n: m.get_function(n) for n in ("lincomb", "leaf_chunk", "cols_fold")}
    return _KM[D]


def lincomb(acc, cols, coefs):
    """acc (D, m) += sum_k coefs[:, k] cols[k]; cols: contiguous int32 field tensors of length <= m."""
    D, m = acc.shape
    f = _k(D)["lincomb"]
    dev = torch.is_tensor(coefs)     # device (D, n) coefficients
    co = coefs if dev else np.asarray(coefs, dtype=np.int64) % P
    for i in range(0, len(cols), 128):
        cs = cols[i:i + 128]
        for c in cs:
            assert c.dtype == torch.int32 and c.is_contiguous() and c.numel() <= m
        pt = kern.h2d(np.array([c.data_ptr() for c in cs], dtype=np.int64))
        ln = kern.h2d(np.array([c.numel() for c in cs], dtype=np.int64))
        cf = co[:, i:i + len(cs)].T.contiguous() if dev else \
            kern.h2d(np.ascontiguousarray(co[:, i:i + len(cs)].T).astype(np.uint32).view(np.int32))
        kern.launch(f, _grid(m), NT, [ptr(pt), ptr(ln), ptr(cf), i32(len(cs)),
                                      ptr(acc), i64(acc.stride(0)), i64(m)])
    return acc


class _HostPool:
    """Pinned host buffers for g_t, reused across proofs (pinning costs about 0.5 s per GiB, so it
    is paid once per process).  get() returns None when the budget is exhausted."""

    def __init__(self):
        self.free, self.used, self.total = {}, [], 0
        self.limit = None
        self.stream = None

    def begin(self):
        if self.limit is None:
            try:
                avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (ValueError, OSError):
                avail = 0
            self.limit = min(G_BUDGET, avail // 2) if avail else G_BUDGET
            self.stream = torch.cuda.Stream()

    def get(self, shape, force=False):
        lst = self.free.get(shape)
        if lst:
            t = lst.pop()
        else:
            need = int(np.prod(shape)) * 4
            if self.total + need > self.limit and not force:
                return None
            t = torch.empty(shape, dtype=torch.int32, pin_memory=True)
            self.total += need
        self.used.append(t)
        return t

    def release(self):
        for t in self.used:
            self.free.setdefault(tuple(t.shape), []).append(t)
        self.used = []


_POOL = _HostPool()


def _eq_rows(G, rows):
    """eq table (D, 2^k) of a point given as device rows (k, D); k = 0: the constant 1."""
    k = rows.shape[0]
    if k == 0:
        one = torch.zeros((G.D, 1), dtype=torch.int32, device=_dev())
        one[0] = 1
        return one
    from .sumcheck import eq_segments
    if k <= 14:
        return eq_segments(G, rows, [(0, k)], [0], 1 << k)
    return G.eq_table(rows.T.cpu().numpy().view(np.uint32).astype(np.int64))


def _eq_split(G, z, c):
    """eq(z, x) over c variables as two tables: COL over the low half, ROW over the high half
    (their product in the sumcheck program; no table of 2^c entries)."""
    from .lazy import Lz
    from .sumcheck import ROW, COL
    if isinstance(z, Lz):
        rows = z.dev.T.contiguous()
    else:
        zz = np.asarray(z, dtype=np.int64) % P
        rows = torch.from_numpy(zz.T.copy().astype(np.uint32).view(np.int32)).to(_dev())
    h = c // 2
    return [Tab(_eq_rows(G, rows[:h]), COL, h, shared=True), Tab(_eq_rows(G, rows[h:]), ROW, h, shared=True)]


def fold_cols(cols, e, Lo):
    """Columns (int32 base field, length <= H Lo, zero beyond) read as H rows of Lo, folded at their
    rows with weights e (DQ, H): -> (DQ, len(cols) Lo), [:, k Lo + c] = sum_h e[:, h] col_k[h Lo + c]."""
    D, H = e.shape
    n = len(cols)
    out = torch.empty((D, n * Lo), dtype=torch.int32, device=_dev())
    HCH = max(64, min(H, (1 << 22) // max(1, n * Lo)))
    nh = (H + HCH - 1) // HCH
    nct = (Lo + NT - 1) // NT
    for i in range(0, n, 1 << 14):
        cs = cols[i:i + (1 << 14)]
        for c in cs:
            assert c.dtype == torch.int32 and c.is_contiguous() and c.numel() <= H * Lo
        pt = kern.h2d(np.array([c.data_ptr() for c in cs], dtype=np.int64))
        ln = kern.h2d(np.array([c.numel() for c in cs], dtype=np.int64))
        pn = len(cs) * Lo
        part = torch.empty((nh, D, pn), dtype=torch.int32, device=_dev())
        kern.launch(_k(D)["cols_fold"], len(cs) * nh * nct, NT,
                    [ptr(pt), ptr(ln), i64(Lo), i64(H), ptr(e), i64(e.stride(0)), i64(HCH), ptr(part), i64(pn)])
        out[:, i * Lo:i * Lo + pn] = ((part.to(torch.int64) & 0xFFFFFFFF).sum(dim=0) % P).to(torch.int32)
    return out


def leaf_state(rows, state, chain):
    """rows (n <= 64, N) int32 codeword rows -> state (N/4, 8) (see module doc)."""
    n, N = rows.shape
    assert n <= CHUNK and rows.stride(1) == 1
    nl = N // 4
    kern.launch(_k(DQ)["leaf_chunk"], _grid(nl), NT,
                [ptr(rows), i64(rows.stride(0)), i32(n), i64(nl), ptr(state), i32(chain)])


def leaf_digest_host(vals, h=None):
    """vals (ncols, 4) field values -> 32-byte leaf digest; h: digest of the leaf's earlier part
    (merged trees: the units before this one), chained like the chunks."""
    b = np.ascontiguousarray(np.asarray(vals) % P).astype("<u4")
    for i in range(0, b.shape[0], CHUNK):
        d = blake3.blake3(b[i:i + CHUNK].tobytes()).digest()
        h = d if h is None else blake3.blake3(h + d).digest()
    return h


def cap_level(nleaves):
    return max(0, nleaves.bit_length() - 1 - CAPLOG)


def _paths(levels, idx, depth):
    """Siblings of leaves idx (torch int64) up to `depth` levels -> (Q, depth, 8)."""
    out = []
    i = idx.clone()
    for lv in levels[:depth]:
        out.append(lv[i ^ 1])
        i = i >> 1
    if not out:
        return torch.zeros((idx.numel(), 0, 8), dtype=torch.int32, device=_dev())
    return torch.stack(out, dim=1)


def _check_mixed(dig, cm, mixed, path, i, cap):
    """Path of a mixed-height tree (GPCS._tree_levels): dig[c] the leaf digest of class c at this
    query; the classes in mixed below cm are hashed into the levels with 2^(c-1) nodes."""
    h = dig[cm]
    for L, sib in enumerate(path, 1):
        sb = sib.astype("<u4").tobytes()
        h = blake3.blake3(sb + h if i & 1 else h + sb).digest()
        i >>= 1
        if cm - L in mixed and cm - L != cm:
            h = blake3.blake3(h + dig[cm - L]).digest()
    return h == cap[i].astype("<u4").tobytes()


def _check_path(h, path, i, cap):
    for sib in path:
        sb = sib.astype("<u4").tobytes()
        h = blake3.blake3(sb + h if i & 1 else h + sb).digest()
        i >>= 1
    return h == cap[i].astype("<u4").tobytes()


def _itw2(G, n, level):
    """1 / (2 tw[level]) on the GPU for the 2^n domain."""
    key = (n, level)
    if key not in _ITW:
        tw = codec.get_twiddles(n)
        a = tw.dev[tw.offs[level]:tw.offs[level + 1]]
        two = ((a.to(torch.int64) * 2) % P).to(torch.int32)
        out = torch.empty_like(two)
        kern.launch(G.f["base_inv"], _grid(two.numel()), NT, [ptr(two), ptr(out), i64(two.numel())])
        _ITW[key] = out
    return _ITW[key]


def encode_cols(cols, c, pos=None):
    """Codewords of the columns; pos (device int64): only those symbols are needed (selective
    last pass, codec.encode_sel)."""
    n = 1 << c
    if all(x.numel() == n and x.dtype == torch.int32 for x in cols):
        msg = torch.stack([x.reshape(n) for x in cols])
    else:
        msg = torch.zeros((len(cols), n), dtype=torch.int32, device=_dev())
        for k, x in enumerate(cols):
            msg[k, :x.numel()] = x
    if pos is not None:
        return codec.encode_sel(msg, c + 1, pos)
    return codec.encode(msg, c + 1, 0)


def leaf_pass(cols, c, state=None, qleaf=None):
    """Encode one (unit, class) and hash its columns into the leaf states (state: the states of
    the leaves' earlier part, chained; None starts new leaves).  Returns (state, leaf values
    (Q, ncols, 4) or None)."""
    nl = 1 << (c - 1)
    chain = state is not None
    if state is None:
        state = torch.empty((nl, 8), dtype=torch.int32, device=_dev())
    vals = []
    if qleaf is not None:
        sym = (qleaf.view(-1, 1) * 4 + torch.arange(4, device=_dev()).view(1, -1)).reshape(-1)
    for i in range(0, len(cols), CHUNK):
        cw = encode_cols(cols[i:i + CHUNK], c)
        leaf_state(cw, state, 1 if (i or chain) else 0)
        if qleaf is not None:
            vals.append(cw[:, sym].reshape(cw.shape[0], -1, 4).permute(1, 0, 2))
        del cw
    return state, (torch.cat(vals, dim=1) if vals else None)


def tree_pass(cols, c, qleaf=None):
    """Encode + hash one (unit, class).  Returns (levels, leaf values (Q, ncols, 4) or None)."""
    state, vals = leaf_pass(cols, c, None, qleaf)
    return codec.merkle_levels(state), vals


def _emb(a):
    a = np.asarray(a, dtype=np.int64)
    if a.shape[0] == DO:
        return a
    out = np.zeros((DO,) + a.shape[1:], dtype=np.int64)
    out[:a.shape[0]] = a
    return out


class Unit:
    def __init__(self, name, items, cmax):
        self.name, self.items, self.cmax = name, list(items), cmax
        self.info = {}
        cls = {}
        for tn, K, c0 in self.items:
            self.info[tn] = (K, c0)
            c = self.tclass(tn)
            for b in range(K):
                for j in range(1 << max(0, c0 - cmax)):
                    cls.setdefault(c, []).append((tn, b, j))
        self.cls = dict(sorted(cls.items()))
        self.colidx = {c: {key: k for k, key in enumerate(v)} for c, v in self.cls.items()}

    def tclass(self, tn):
        return min(max(self.info[tn][1], CMIN), self.cmax)

    def columns(self, tensors, c):
        out = []
        for tn, b, j in self.cls[c]:
            K, c0 = self.info[tn]
            ln = 1 << min(c0, self.cmax)
            t = tensors[tn].reshape(-1)
            assert t.numel() == K << c0, (self.name, tn, t.numel(), K, c0)
            off = (b << c0) + j * ln
            out.append(t[off:off + ln])
        return out

    def col_claims(self, claims):
        """claims [(tname, pt (DQ, c0), {b: coef}, value)] -> class -> [(z (DQ, c), {k: coef}, value)]."""
        F = fe(DQ)
        out = {}
        for tn, pt, wb, v in claims:
            K, c0 = self.info[tn]
            c = self.tclass(tn)
            idx = self.colidx[c]
            assert pt.shape[1] == c0, (self.name, tn, pt.shape, c0)
            cd = {}
            if c0 > self.cmax:
                z, hi = pt[:, :self.cmax], pt[:, self.cmax:]
                for b, w in wb.items():
                    for j in range(1 << (c0 - self.cmax)):
                        e = F.c(w)
                        for t in range(hi.shape[1]):
                            x = F.c(hi[:, t])
                            e = F.mul(e, x if (j >> t) & 1 else F.sub(F.one, x))
                        cd[idx[(tn, b, j)]] = e
            else:
                z = pt if c == c0 else np.concatenate([pt, np.zeros((DQ, c - c0), dtype=np.int64)], axis=1)
                for b, w in wb.items():
                    cd[idx[(tn, b, 0)]] = F.c(w)
            out.setdefault(c, []).append((np.asarray(z, dtype=np.int64) % P, cd, np.asarray(v, dtype=np.int64) % P))
        return out

    def col_claims_dev(self, claims, G):
        """col_claims for the prover: points stay lazy, coefficients on the device, no values."""
        from .lazy import lcat
        out = {}
        for tn, pt, wb, _ in claims:
            K, c0 = self.info[tn]
            c = self.tclass(tn)
            idx = self.colidx[c]
            assert pt.shape[1] == c0, (self.name, tn, pt.shape, c0)
            cd = {}
            if c0 > self.cmax:
                z, hi = pt[:, :self.cmax], pt[:, self.cmax:]
                eh = G.eq_table(hi)                                 # (DQ, 2^(c0 - cmax)) eq(hi, j)
                for b, w in wb.items():
                    wd = _dev_ef(G, w).view(DQ, 1).expand(DQ, eh.shape[1]).contiguous()
                    e = G.mul(eh, wd)
                    for j in range(1 << (c0 - self.cmax)):
                        cd[idx[(tn, b, j)]] = e[:, j]
            else:
                z = pt if c == c0 else lcat([pt, np.zeros((DQ, c - c0), dtype=np.int64)])
                for b, w in wb.items():
                    cd[idx[(tn, b, 0)]] = w
            out.setdefault(c, []).append((z, cd, None))
        return out


def _by_point_dev(cl):
    """_by_point for the prover's device claims: points grouped by their draw keys (lazy.col_keys),
    so equal points group without being read; if a point has no keys every point is read once."""
    from .lazy import col_keys, materialize
    ks = [col_keys(z) for z, _, _ in cl]
    if any(k is None for k in ks):
        materialize([z for z, _, _ in cl])
        ks = [np.asarray(z, dtype=np.int64).tobytes() for z, _, _ in cl]
    out, pos = [], {}
    for (z, cd, v), key in zip(cl, ks):
        if key not in pos:
            pos[key] = len(out)
            out.append((z, []))
        out[pos[key]][1].append((cd, v))
    return out


def _dev_ef(G, x):
    """An ext value (numpy/F tuple/Lz/tensor (D,)) as a device int32 (D,)."""
    from .lazy import Lz
    if isinstance(x, Lz):
        return x.dev.reshape(-1)
    if torch.is_tensor(x):
        return x.reshape(-1)
    return G.to_dev(np.asarray(x, dtype=np.int64))


def _by_point(cl):
    out, pos = [], {}
    for z, cd, v in cl:
        key = z.tobytes()
        if key not in pos:
            pos[key] = len(out)
            out.append((z, []))
        out[pos[key]][1].append((cd, v))
    return out


def _combine(tr, F, cl):
    """Per-claim coefficients beta -> ({k: sum beta_i coef_ik}, sum beta_i v_i)."""
    beta = tr.challenge(len(cl))
    comb, claim = {}, F.zero
    for i, (cd, v) in enumerate(cl):
        bi = F.c(beta[:, i])
        claim = F.add(claim, F.mul(bi, F.c(v)))
        for k, cf in cd.items():
            comb[k] = F.add(comb.get(k, F.zero), F.mul(bi, F.c(cf)))
    return comb, claim


# a reduction kernel takes three tables per point: more points than this overflow the CUDA kernel
# parameter space (32 KiB), so larger sets are reduced in rounds
MAX_RED_POINTS = 64


def _max_points(c):
    return max(2, min(MAX_RED_POINTS, RED_BYTES // (2 * (1 << c) * 4 * DQ)))


def _fv(G, cols, ks, e):
    return G.dots([(cols[k], e[:, :cols[k].numel()]) for k in ks]).T


def _eqv(F, a, b):
    from . import vcore
    if vcore.ENABLED:
        return vcore.eqv(len(F.zero), a, b)
    e = F.one
    for j in range(a.shape[1]):
        e = F.mul(e, F.eq1(F.c(a[:, j]), F.c(b[:, j])))
    return e


class UC:
    """State of one (unit, class) after P2."""

    def __init__(self, unit, c, rho, alpha, y, v=None):
        self.unit, self.c, self.rho, self.alpha, self.y, self.v = unit, c, rho, alpha, y, v
        self.ncols = len(unit.cls[c])
        self.w = None
        self.g = None
        self.gfull = None          # prover, segment units: g on the device until the segment closes
        self.cap = None


class SEG:
    """A reduced segment: its units' claims (rho_t, y_t on g_t = sum_k alpha_tk f_tk) are brought
    to one point r by a sumcheck over NV variables, as in finish().  Per class c, H_c = sum_t lam_t
    eq(rho_t, r[NV-c:]) g_t, and the prover sends y_c = H_c(r[NV-c:]).  finish() treats every
    (segment, class) as one unit PU whose combined column is H_c (kept in host memory)."""

    def __init__(self, names, keys):
        self.names, self.keys = tuple(names), list(keys)
        self.pus = []              # one PU per class
        self.e1 = {}               # verifier: unit key -> lam_t eq(rho_t, r[NV-c:])


class PU:
    """(segment, class) as a unit of the global opening: point rho (DO, c), value y, folded v,
    combined column H in host memory (prover)."""

    def __init__(self, seg, c, rho, y, v=None):
        self.seg, self.c, self.rho, self.y, self.v = seg, c, rho, y, v
        self.unit = None
        self.alpha = None
        self.gh = None             # prover: pinned host copy of H_c
        self.hd = None             # prover: H_c on the device until it is copied
        self.w = None


class GPCS:
    def __init__(self, tr, prover, cmax=CMAX, queries=QUERIES, grind=GRIND):
        self.tr, self.prover = tr, prover
        self.cmax, self.queries, self.grind = cmax, queries, grind
        self.units = {}
        self.order = []
        self.caps = {}             # (unit, c) -> numpy (ncap, 8)
        self.ucs = {}              # (unit, c) -> UC
        self.stats = {}
        self._gbytes = 0
        # merged trees: units committed one after another (one layer's groups) share one Merkle
        # tree per class, whose leaf is the concatenation of their leaves (fewer paths per query)
        self.msets = []            # member unit names of each merge set, in commit order
        self.mset_of = {}          # unit name -> merge set id
        self._mstate = {}          # tree id -> leaf states while its members are committed
        # segments (one layer's units): reduced to one point per class when the last is opened
        self.seg_of = {}           # unit name -> segment index
        self.seg_decl = []         # member unit names of each segment
        self.segs = []             # SEG records of the reduced segments, in proof order
        self.seg_pending = []      # prover: (SEG, class) whose H still has to go to host memory
        self.kept = {}             # prover: tree id -> its levels in pinned host memory

    def segment_decl(self, names):
        """Declare that units names (one layer's groups and weights) form a segment: once all are
        opened, segment() reduces their claims to one point per class (see SEG)."""
        i = len(self.seg_decl)
        self.seg_decl.append(tuple(names))
        for n in names:
            assert n in self.units and n not in self.seg_of
            self.seg_of[n] = i

    def merge(self, names):
        """Declare that units names (committed consecutively, in this order) share their trees."""
        sid = len(self.msets)
        self.msets.append(tuple(names))
        for n in names:
            assert n in self.units and n not in self.mset_of
            self.mset_of[n] = sid

    # Trees: the units of a set (a merge set, or a single unit) share one mixed-height tree: it is
    # built over the leaves of the set's largest class cm, and the leaf digests of a smaller class
    # c are hashed into the node level with 2^(c-1) nodes (node = H(H(left, right), leaf_c)), so a
    # query opens one path for every class.  Classes whose level would be above the cap (at most
    # 2^CAPLOG leaves: no path below their cap anyway) keep their own trees.
    def _sid(self, name):
        sid = self.mset_of.get(name)
        return ("u", name) if sid is None else sid

    def _members(self, sid):
        return (sid[1],) if isinstance(sid, tuple) else self.msets[sid]

    def _set_classes(self, sid):
        return sorted(set(c for n in self._members(sid) for c in self.units[n].cls))

    def _mixed(self, sid):
        """(cm, classes in the mixed tree) of a set."""
        cls = self._set_classes(sid)
        cm = cls[-1]
        lc = cap_level(1 << (cm - 1))
        return cm, [c for c in cls if cm - c <= lc]

    def _set_trees(self, sid):
        """Tree ids of a set, in the order their caps are sent: the mixed tree, then small ones."""
        cm, mixed = self._mixed(sid)
        return [("T", sid)] + [("S", sid, c) for c in self._set_classes(sid) if c not in mixed]

    def tree(self, name, c):
        sid = self._sid(name)
        return ("T", sid) if c in self._mixed(sid)[1] else ("S", sid, c)

    def set_last(self, keys):
        """Per (unit, class) key: True if it is the last key of its set in keys order."""
        last = {}
        for k in keys:
            last[self._sid(k[0])] = k
        return {k: last[self._sid(k[0])] == k for k in keys}

    def _tree_levels(self, tid, states):
        """Levels of tree tid from the leaf states {(sid, c): (2^(c-1), 8)} of its classes."""
        if tid[0] == "S":
            return codec.merkle_levels(states[(tid[1], tid[2])])
        sid = tid[1]
        cm, mixed = self._mixed(sid)
        m = codec.module(8)
        cur = states[(sid, cm)]
        levels = [cur]
        L = 0
        while cur.shape[0] > 1:
            nxt = torch.empty((cur.shape[0] // 2, 8), dtype=torch.int32, device=cur.device)
            m.node_hash(cur, nxt)
            L += 1
            if cm - L in mixed:
                inj = torch.stack([nxt, states[(sid, cm - L)]], dim=1).reshape(-1, 8).contiguous()
                nxt = torch.empty_like(nxt)
                m.node_hash(inj, nxt)
            levels.append(nxt)
            cur = nxt
        return levels

    def _tree_c(self, tid):
        """The class whose leaf index addresses tree tid (its largest class)."""
        return self._mixed(tid[1])[0] if tid[0] == "T" else tid[2]

    # ------------------------------------------------------------ declare / P1
    def declare(self, name, items):
        self.units[name] = Unit(name, items, self.cmax)
        self.order.append(name)

    @property
    def NV(self):
        return max(c for u in self.units.values() for c in u.cls)

    def commit(self, name, tensors=None):
        u = self.units[name]
        tr = self.tr
        sid = self._sid(name)
        if self.prover:
            for c in u.cls:
                self._mstate[(sid, c)], _ = leaf_pass(u.columns(tensors, c), c, self._mstate.get((sid, c)))
        if name != self._members(sid)[-1]:
            return
        for tid in self._set_trees(sid):
            if self.prover:
                levels = self._tree_levels(tid, self._mstate)
                self._send_cap(levels, self._tree_c(tid), tid)
                if KEEP_TREES:
                    self._keep_levels(tid, levels)
                del levels
            else:
                b = tr.send_bytes(None)
                self.caps[tid] = np.frombuffer(b, dtype="<u4").reshape(-1, 8).astype(np.uint32)
        if self.prover:
            for c in self._set_classes(sid):
                self._mstate.pop((sid, c), None)

    def _keep_levels(self, tid, levels):
        """Copy a tree's levels to the pinned pool (side stream); skipped if the budget is out."""
        _POOL.begin()
        hs = []
        for lv in levels:
            h = _POOL.get(tuple(lv.shape))
            if h is None:
                return
            hs.append(h)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(_POOL.stream):
            _POOL.stream.wait_event(ev)
            for h, lv in zip(hs, levels):
                h.copy_(lv, non_blocking=True)
                lv.record_stream(_POOL.stream)
        self.kept[tid] = hs

    def _send_cap(self, levels, c, key):
        """Prover: send a tree's cap (asynchronous download, absorbed in order later)."""
        cap = levels[cap_level(1 << (c - 1))]
        hb = torch.empty(cap.shape, dtype=cap.dtype, pin_memory=True)
        hb.copy_(cap, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        del levels

        def sent(hb=hb, ev=ev, key=key):
            # asynchronous cap download: absorbed (in order) when the transcript is next used
            ev.synchronize()
            b = hb.numpy().astype("<i4").tobytes()
            self.caps[key] = np.frombuffer(b, dtype="<u4").reshape(-1, 8).astype(np.uint32)
            return b
        self.tr.send_bytes_deferred(sent)

    def cap_bytes(self, name):
        """The caps of (unmerged) unit name, in commit order (what commit sent)."""
        self.tr.flush()
        sid = self._sid(name)
        assert self._members(sid) == (name,)
        return [self.caps[tid].astype("<u4").tobytes() for tid in self._set_trees(sid)]

    def commit_caps(self, name, caps):
        """Prover: commit an (unmerged) unit whose caps are known (precomputed weight commitment
        C_W).  The trees are rebuilt from the tensors when the unit is queried, as for every unit."""
        assert self.prover
        sid = self._sid(name)
        tids = self._set_trees(sid)
        assert self._members(sid) == (name,) and len(caps) == len(tids)
        for tid, b in zip(tids, caps):
            self.tr.send_bytes(b)
            self.caps[tid] = np.frombuffer(b, dtype="<u4").reshape(-1, 8).astype(np.uint32)

    # ------------------------------------------------------------ P2
    def open(self, name, claims, tensors=None):
        """claims [(tname, pt (DQ, c0), {b: coef}, value)]: sum_b coef_b f~_{tname,b}(pt) = value."""
        if self.prover:                 # the prover opens on the device transcript; this path is the verifier's
            return self._open_dev(name, claims, tensors)
        u = self.units[name]
        tr = self.tr
        F4, F8 = fe(DQ), fe(DO)
        G4, G8 = gops(DQ), gops(DO)
        E4 = G4.E
        bycls = u.col_claims(claims)
        NV = self.NV
        tr.absorb(b"pcs-open|" + label_name(name))
        for c, cnames in u.cls.items():
            ncols = len(cnames)
            cols = u.columns(tensors, c) if self.prover else None
            pc = _by_point(bycls.get(c, []))
            mp = _max_points(c)
            while len(pc) > mp:
                new = []
                for i in range(0, len(pc), mp):
                    batch = pc[i:i + mp]
                    r, chk = self._reduce(tr, G4, F4, E4, cols, c, batch)
                    ks = sorted(set(k for _, cl in batch for cd, _ in cl for k in cd))
                    if self.prover:
                        e = G4.eq_table(r)
                        fk = tr.send(_fv(G4, cols, ks, e).T).T
                        del e
                    else:
                        fk = tr.send(None).T
                        assert fk.shape == (DQ, len(ks))
                    if not self.prover and not chk({k: fk[:, j] for j, k in enumerate(ks)}):
                        raise ValueError(f"pcs {name}: point reduction failed")
                    new.append((r, [({k: F4.one}, fk[:, j]) for j, k in enumerate(ks)]))
                pc = new
            if pc:
                rho, chk = self._reduce(tr, G4, F4, E4, cols, c, pc)
            else:
                rho, chk = tr.challenge(c), None
            if self.prover:
                e = G4.eq_table(rho)
                fv = tr.send(_fv(G4, cols, range(ncols), e).T).T
                del e
            else:
                fv = tr.send(None).T
                assert fv.shape == (DQ, ncols)
            if not self.prover and chk is not None and not chk({k: fv[:, k] for k in range(ncols)}):
                raise ValueError(f"pcs {name}: point reduction failed")
            alpha = tr.challenge(ncols, D=DO)
            y = F8.zero
            for k in range(ncols):
                y = F8.add(y, F8.mul(F8.c(alpha[:, k]), F8.c(_emb(fv[:, k]))))
            uc = UC(u, c, rho, alpha, y)
            if self.prover:
                m = max(0, c - (NV - min(NLOW, NV)))
                g = G8.zeros(1 << c)
                lincomb(g, cols, alpha)
                re = _emb(rho)
                for j in range(c - 1, m - 1, -1):
                    g = G8.fold_hi(g, re[:, j])
                uc.v = g
                del cols
            self.ucs[(name, c)] = uc

    def _open_dev(self, name, claims, tensors):
        """The prover's open with every challenge, point and value on the device: one device
        transcript segment per open (none in a device session); the transcript bytes and
        messages are those of open().  rho, alpha and y stay lazy until finish."""
        from . import devtr as _dv
        from .lazy import Lz
        u = self.units[name]
        tr = self.tr
        F4 = fe(DQ)
        G4, G8 = gops(DQ), gops(DO)
        bycls = u.col_claims_dev(claims, G4)
        NV = self.NV
        tr.absorb(b"pcs-open|" + label_name(name))
        dt = _dv.get(tr)
        for c, cnames in u.cls.items():
            ncols = len(cnames)
            cols = u.columns(tensors, c)
            pc = [(z, [cd for cd, _ in cl]) for z, cl in _by_point_dev(bycls.get(c, []))]
            mp = _max_points(c)
            while len(pc) > mp:
                new = []
                for i in range(0, len(pc), mp):
                    batch = pc[i:i + mp]
                    r = self._reduce_dev(dt, G4, F4, cols, c, batch)
                    ks = sorted(set(k for _, cds in batch for cd in cds for k in cd))
                    e = G4.eq_table(r)
                    dt.send(G4.dots_dev([(cols[k], e[:, :cols[k].numel()]) for k in ks]))
                    del e
                    new.append((r, [{k: F4.one} for k in ks]))
                pc = new
            if pc:
                rho = self._reduce_dev(dt, G4, F4, cols, c, pc)
            else:
                rho = Lz(dt.challenge(c).T)
            # one pass over the columns: fold the high variables at rho (small eq table), then the
            # column values f_k(rho) and the folded combination v = sum_k alpha_k f_k(., rho_hi)
            m = max(0, c - (NV - min(NLOW, NV)))
            rr = rho.dev.T.contiguous() if c else torch.zeros((0, DQ), dtype=torch.int32, device=_dev())
            fo = fold_cols(cols, _eq_rows(G4, rr[m:]), 1 << m)                             # (DQ, ncols 2^m)
            e_lo = _eq_rows(G4, rr[:m])
            fv = G4.mul(fo, e_lo.repeat(1, ncols).contiguous()).to(torch.int64).view(DQ, ncols, 1 << m)
            fv = (fv.sum(dim=2) % P).to(torch.int32).T.contiguous()                         # (ncols, DQ)
            dt.send(fv)
            alpha = dt.challenge(ncols, D=DO)                                                # (ncols, DO)
            fv8 = torch.zeros((DO, ncols), dtype=torch.int32, device=_dev())
            fv8[:DQ] = fv.T
            yv = (G8.mul(alpha.T.contiguous(), fv8).to(torch.int64).sum(dim=1) % P).to(torch.int32)
            uc = UC(u, c, rho, Lz(alpha.T), Lz(yv))
            fo8 = torch.zeros((DO, ncols << m), dtype=torch.int32, device=_dev())
            fo8[:DQ] = fo
            del fo
            g = G8.mul(alpha.T.contiguous().repeat_interleave(1 << m, dim=1), fo8).to(torch.int64)
            g = (g.view(DO, ncols, 1 << m).sum(dim=1) % P).to(torch.int32)
            del fo8
            re = rr
            uc.v = g
            uc._keep = re
            if name in self.seg_of:               # the segment's reduction needs g in full
                uc.gfull = G8.zeros(1 << c)
                lincomb(uc.gfull, cols, alpha.T.contiguous())
            del cols
            self.ucs[(name, c)] = uc
        dt.finish()

    def _reduce_dev(self, dt, G4, F4, cols, c, groups):
        """_reduce on the device transcript dt (prover): groups [(z, [coef dict])] -> rho Lz (DQ, c)."""
        from .lazy import Lz
        tabs, terms = [], []
        for z, cds in groups:
            beta = dt.challenge(len(cds))                                                   # (n, DQ)
            ks = sorted(set(k for cd in cds for k in cd))
            kp = {k: j for j, k in enumerate(ks)}
            C = torch.zeros((len(cds) * len(ks), DQ), dtype=torch.int32, device=_dev())
            pos, rows = [], []
            for i, cd in enumerate(cds):
                for k, cf in cd.items():
                    pos.append(i * len(ks) + kp[k])
                    rows.append(_dev_ef(G4, cf))
            C[kern.h2d(np.array(pos, dtype=np.int64))] = torch.stack(rows)
            comb = G4.comb_coefs(beta, C)
            g = G4.zeros(1 << c)
            lincomb(g, [cols[k] for k in ks], comb)
            tabs += _eq_split(G4, z, c) + [Tab(g, FULL)]
            terms.append((np.array(F4.one), [len(tabs) - 3, len(tabs) - 2, len(tabs) - 1]))
        rho, _, _ = sc.prove(dt, DQ, c, tabs, terms, np.zeros(DQ, dtype=np.int64), label=b"pcs-red", dev=True)
        del tabs
        return Lz(rho.T.contiguous())

    # ------------------------------------------------------------ segments
    def segment(self, i):
        """Close segment i (all its units opened): the sumcheck of finish() over its units alone
        brings their claims to one point r; the prover keeps H_c on the device (seg_pending: the
        caller copies it to host memory, see flush_segments) and sends y_c = H_c(r[NV-c:])."""
        names = self.seg_decl[i]
        keys = [(n, c) for n in names for c in self.units[n].cls]
        assert all(k in self.ucs for k in keys), f"segment {i}: units not opened"
        seg = SEG(names, keys)
        self.tr.absorb(b"pcs-seg")
        if self.prover:
            self._segment_dev(seg)
        else:
            self._segment_ver(seg)
        self.segs.append(seg)
        return seg

    def _segment_dev(self, seg):
        from . import devtr as _dv
        from .lazy import Lz
        G8 = gops(DO)
        NV = self.NV
        NL = min(NLOW, NV)
        dt = _dv.get(self.tr)
        ucs = [self.ucs[k] for k in seg.keys]
        lam = dt.challenge(len(ucs), D=DO)                                    # (n, DO)
        classes = sorted(set(uc.c for uc in ucs))
        idx = {c: [t for t, uc in enumerate(ucs) if uc.c == c] for c in classes}
        mc = {c: max(0, c - (NV - NL)) for c in classes}

        def rows8(pt, a, b):                     # coordinates a..b-1 of a point (D, k) as rows (b-a, DO)
            d = pt.dev if isinstance(pt, Lz) else pt
            out = torch.zeros((b - a, DO), dtype=torch.int32, device=_dev())
            out[:, :d.shape[0]] = d[:, a:b].T
            return out
        st = {}
        for c in classes:
            Y = torch.stack([ucs[t].y.dev for t in idx[c]], dim=1)            # (DO, n_c)
            L = torch.stack([lam[t] for t in idx[c]], dim=1)                       # (DO, n_c)
            C = (G8.mul(L, Y).to(torch.int64).sum(dim=1, keepdim=True) * pow(2, NV - c, P) % P).to(torch.int32)
            E = V = None
            if mc[c] > 0:
                E = torch.cat([G8.scale_p(_eq_rows(G8, rows8(ucs[t].rho, 0, mc[c])), lam[t]) for t in idx[c]], dim=1)
                V = torch.cat([ucs[t].v for t in idx[c]], dim=1)
            st[c] = {"C": C, "E": E, "V": V}
        r = []
        self._rounds_dev(dt, G8, st, classes, NV, 0, NL, r)
        for c in classes:                        # P3 on the kept g_t: w_t = g_t(r_lo, .)
            m = mc[c]
            ws = []
            for t in idx[c]:
                g = ucs[t].gfull
                for j in range(m):
                    g = G8.fold_p(g, r[NV - c + j].data_ptr())
                ws.append(g)
            coefs = [st[c]["E"][:, k:k + 1] for k in range(len(idx[c]))] if m > 0 else [lam[t] for t in idx[c]]
            if c - m > 0:
                st[c]["E"] = torch.cat([G8.scale_p(_eq_rows(G8, rows8(ucs[t].rho, m, c)), cf.reshape(-1))
                                        for t, cf in zip(idx[c], coefs)], dim=1)
                st[c]["V"] = torch.cat(ws, dim=1)
            del ws
        self._rounds_dev(dt, G8, st, classes, NV, NL, NV, r)
        ys = []
        for c in classes:
            E, V = st[c]["E"], st[c]["V"]                                    # (DO, n_c): coefficients, g_t(r)
            ys.append((G8.mul(E, V).to(torch.int64).sum(dim=1) % P).to(torch.int32))
            H = G8.zeros(1 << c)
            for k, t in enumerate(idx[c]):
                H = G8.add(H, G8.scale_p(ucs[t].gfull, E[:, k].contiguous()))
                ucs[t].gfull = None
            m = mc[c]
            v = H
            for j in range(c - 1, m - 1, -1):
                v = G8.fold_hi_p(v, r[NV - c + j].data_ptr())
            pt = torch.stack(r[NV - c:], dim=0).T.contiguous()                # (DO, c)
            pu = PU(seg, c, Lz(pt), Lz(ys[-1]), v)
            pu.hd = H
            seg.pus.append(pu)
        Ysend = torch.stack(ys, dim=0)                                       # (n classes, DO)
        dt.send(Ysend)
        seg._keep = (lam, r, Ysend)
        dt.finish()
        self.seg_pending.extend(seg.pus)
        if not torch.cuda.is_current_stream_capturing():
            self.flush_segments()

    def flush_segments(self):
        """Prover: copy the pending segments' H_c to pinned host memory (side stream; the device
        tensors are released once the copies have run)."""
        if not self.seg_pending:
            return
        _POOL.begin()
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(_POOL.stream):
            _POOL.stream.wait_event(ev)
            for pu in self.seg_pending:
                gh = _POOL.get(tuple(pu.hd.shape), force=True)
                gh.copy_(pu.hd, non_blocking=True)
                pu.hd.record_stream(_POOL.stream)
                pu.gh, pu.hd = gh, None
        self.seg_pending = []

    def _rounds_dev(self, dt, G8, st, classes, NV, j0, j1, r):
        """_rounds for the prover on a device transcript: no host values."""
        inv2 = (P + 1) // 2
        for j in range(j0, j1):
            pairs, consts = [], []
            for c in classes:
                s = st[c]
                if j < NV - c:
                    consts.append(s["C"])
                else:
                    E, V = s["E"], s["V"]
                    E0, E1 = E[:, 0::2].contiguous(), E[:, 1::2].contiguous()
                    V0, V1 = V[:, 0::2].contiguous(), V[:, 1::2].contiguous()
                    E2 = ((2 * E1.to(torch.int64) - E0.to(torch.int64)) % P).to(torch.int32)
                    V2 = ((2 * V1.to(torch.int64) - V0.to(torch.int64)) % P).to(torch.int32)
                    pairs += [(E0, V0), (E1, V1), (E2, V2)]
            ev = torch.zeros((3, DO), dtype=torch.int64, device=_dev())
            if pairs:
                ds = G8.dots_dev(pairs).to(torch.int64)                       # (3 n, DO)
                ev += ds.view(-1, 3, DO).sum(dim=0)
            if consts:
                h = torch.cat(consts, dim=1).to(torch.int64).sum(dim=1) % P * inv2 % P
                ev += h.view(1, DO)
            dt.send((ev % P).to(torch.int32))
            x = dt.challenge(D=DO)
            r.append(x)
            for c in classes:
                s = st[c]
                if j < NV - c:
                    s["C"] = (s["C"].to(torch.int64) * inv2 % P).to(torch.int32)
                else:
                    s["E"] = G8.fold_p(s["E"], x.data_ptr())
                    s["V"] = G8.fold_p(s["V"], x.data_ptr())

    def _segment_ver(self, seg):
        F8 = fe(DO)
        NV = self.NV
        tr = self.tr
        ucs = [self.ucs[k] for k in seg.keys]
        lam = tr.challenge(len(ucs), D=DO)
        classes = sorted(set(uc.c for uc in ucs))
        claim = F8.zero
        for t, uc in enumerate(ucs):
            claim = F8.add(claim, F8.smul(F8.mul(F8.c(lam[:, t]), F8.c(uc.y)), pow(2, NV - uc.c, P)))
        r = []
        claim = self._rounds(tr, F8, None, {}, classes, NV, 0, NV, claim, r)
        Y = tr.send(None)
        assert Y.shape == (len(classes), DO)
        tot = F8.zero
        for i in range(len(classes)):
            tot = F8.add(tot, F8.c(Y[i]))
        if tot != claim:
            raise ValueError("pcs: segment reduction failed")
        rr = np.stack([np.array(x) for x in r], axis=1)
        for t, (k, uc) in enumerate(zip(seg.keys, ucs)):
            seg.e1[k] = F8.mul(F8.c(lam[:, t]), _eqv(F8, _emb(uc.rho), rr[:, NV - uc.c:]))
        for i, c in enumerate(classes):
            seg.pus.append(PU(seg, c, rr[:, NV - c:], F8.c(Y[i])))

    def _prov(self, provider, skip=None):
        if provider is None:
            return
        try:
            it = provider(skip) if skip else provider()
        except TypeError:
            it = provider()
        yield from it

    def _reduce(self, tr, G4, F4, E4, cols, c, groups):
        """One sumcheck sum_z eq(z, x) g_z(x).  Returns (rho, check(fv: k -> value) -> bool)."""
        combs, claim = [], F4.zero
        tabs, terms = [], []
        for z, cl in groups:
            comb, cz = _combine(tr, F4, cl)
            claim = F4.add(claim, cz)
            combs.append((z, comb))
            if self.prover:
                ks = sorted(comb)
                g = G4.zeros(1 << c)
                lincomb(g, [cols[k] for k in ks], np.array([comb[k] for k in ks], dtype=np.int64).T)
                tabs += _eq_split(G4, z, c) + [Tab(g, FULL)]
                terms.append((np.array(F4.one), [len(tabs) - 3, len(tabs) - 2, len(tabs) - 1]))
        if self.prover:
            rho, _, red = sc.prove(tr, DQ, c, tabs, terms, np.array(claim), label=b"pcs-red")
            del tabs
        else:
            rho, red = sc.verify(tr, E4, c, 2, np.array(claim), label=b"pcs-red")

        def check(fv):
            exp = F4.zero
            for z, comb in combs:
                s_ = F4.zero
                for k, cf in comb.items():
                    s_ = F4.add(s_, F4.mul(cf, F4.c(fv[k])))
                exp = F4.add(exp, F4.mul(_eqv(F4, z, rho), s_))
            return exp == F4.c(red)
        return rho, check

    # ------------------------------------------------------------ finish
    def finish(self, provider=None):
        """provider(skip): iterator of (unit name, tensors) over every unit not in skip (prover).
        The global opening runs over the units outside segments and one unit PU per (segment,
        class); only the units outside segments are read in P3 and P4, the PUs' combined columns
        come from host memory, and P5 reads every unit."""
        tr = self.tr
        F8 = fe(DO)
        G8 = gops(DO)
        NV = self.NV
        NL = min(NLOW, NV)
        keys = [(n, c) for n in self.order for c in self.units[n].cls]
        assert all(k in self.ucs for k in keys), "every unit must be opened"
        assert len(self.segs) == len(self.seg_decl), "every segment must be closed"
        gkeys = [k for k in keys if k[0] not in self.seg_of]
        pus = [pu for seg in self.segs for pu in seg.pus]
        if self.prover:                  # lazy values of _open_dev and the segments, one transfer
            from .lazy import materialize, is_lazy
            self.flush_segments()
            ucs_ = [self.ucs[k] for k in gkeys]
            materialize([[uc.rho, uc.alpha, uc.y] for uc in ucs_] + [[pu.rho, pu.y] for pu in pus])
            for uc in ucs_ + pus:
                for a in ("rho", "alpha", "y"):
                    v = getattr(uc, a)
                    if is_lazy(v):
                        setattr(uc, a, F8.c(v.np()) if a == "y" else v.np())
                uc._keep = None
            for k in keys:
                if k[0] in self.seg_of:
                    self.ucs[k]._keep = None
                    self.ucs[k].v = None
        tr.absorb(b"pcs-finish")
        ucs = [self.ucs[k] for k in gkeys] + pus
        lam = tr.challenge(len(ucs), D=DO)
        for t, uc in enumerate(ucs):
            uc.lam = F8.c(lam[:, t])
        classes = sorted(set(uc.c for uc in ucs))
        byc = {c: [uc for uc in ucs if uc.c == c] for c in classes}
        mc = {c: max(0, c - (NV - NL)) for c in classes}
        claim = F8.zero
        for uc in ucs:
            claim = F8.add(claim, F8.smul(F8.mul(uc.lam, uc.y), pow(2, NV - uc.c, P)))
        # class state: constant part C_c (inactive) or tables (E, V)
        st = {}
        for c in classes:
            C = F8.zero
            for uc in byc[c]:
                C = F8.add(C, F8.smul(F8.mul(uc.lam, uc.y), pow(2, NV - c, P)))
            st[c] = {"C": C, "E": None, "V": None}
        if self.prover:
            for c in classes:
                if mc[c] > 0:
                    st[c]["E"] = torch.cat([G8.scale(G8.eq_table(_emb(uc.rho[:, :mc[c]])), np.array(uc.lam))
                                            for uc in byc[c]], dim=1)
                    st[c]["V"] = torch.cat([uc.v for uc in byc[c]], dim=1)
                    for uc in byc[c]:
                        uc.v = None
        r = []
        tlog = os.environ.get("ZK_PCSLOG") and self.prover
        t0 = time.time()

        def tick(what):
            nonlocal t0
            if tlog:
                torch.cuda.synchronize()
                print(f"[pcs] {what}: {time.time() - t0:.2f}s", flush=True)
                t0 = time.time()
        claim = self._rounds(tr, F8, G8, st, classes, NV, 0, NL, claim, r)
        tick("rounds lo")
        if self.prover:
            # P3: w_t = g_t(r_lo, .): the units outside segments from the provider, the PUs from
            # their combined columns in host memory
            segn = set(self.seg_of)
            if _POOL.stream is not None:
                _POOL.stream.synchronize()
            for name, tensors in self._prov(provider, segn or None):
                if name in segn:                  # providers without skip yield every unit
                    continue
                u = self.units[name]
                for c in u.cls:
                    uc = self.ucs[(name, c)]
                    g = G8.zeros(1 << c)
                    lincomb(g, u.columns(tensors, c), uc.alpha)
                    for j in range(mc[c]):
                        g = G8.fold(g, r[NV - c + j])
                    uc.w = g
            for pu in pus:
                g = pu.gh.cuda(non_blocking=True)
                for j in range(mc[pu.c]):
                    g = G8.fold(g, r[NV - pu.c + j])
                pu.w = g
            for c in classes:
                m = mc[c]
                if m > 0:
                    Ef = G8.to_host(st[c]["E"])          # (8, n_t): lambda eq(rho_lo, r_lo)
                    coefs = [Ef[:, i] for i in range(len(byc[c]))]
                else:
                    coefs = [np.array(uc.lam) for uc in byc[c]]
                if c - m > 0:
                    st[c]["E"] = torch.cat([G8.scale(G8.eq_table(_emb(uc.rho[:, m:])), cf)
                                            for uc, cf in zip(byc[c], coefs)], dim=1)
                    st[c]["V"] = torch.cat([uc.w for uc in byc[c]], dim=1)
                for uc in byc[c]:
                    uc.w = None
        tick("P3 pass")
        claim = self._rounds(tr, F8, G8, st, classes, NV, NL, NV, claim, r)
        tick("rounds hi")
        V = claim
        rr = np.stack([np.array(x) for x in r], axis=1) if r else np.zeros((DO, 0), dtype=np.int64)
        for uc in ucs:
            uc.sc = F8.mul(uc.lam, _eqv(F8, _emb(uc.rho), rr[:, NV - uc.c:]))
        if not self.prover:              # column coefficients for the query checks
            for k in gkeys:
                uc = self.ucs[k]
                uc.coef = np.stack([np.array(F8.mul(uc.sc, F8.c(uc.alpha[:, j]))) for j in range(uc.ncols)], axis=1)
            for seg in self.segs:
                scc = {pu.c: pu.sc for pu in seg.pus}
                for k in seg.keys:
                    uc = self.ucs[k]
                    s2 = F8.mul(scc[uc.c], seg.e1[k])
                    uc.coef = np.stack([np.array(F8.mul(s2, F8.c(uc.alpha[:, j]))) for j in range(uc.ncols)], axis=1)
        H = None
        if self.prover:
            H = {c: G8.zeros(1 << c) for c in classes}
            for pu in pus:                            # H_c += lam eq(rho, r_c) H_seg,c
                G8.axpy(H[pu.c], pu.gh.cuda(non_blocking=True), pu.sc)
            for name, tensors in self._prov(provider, segn or None):
                if name in segn:
                    continue
                u = self.units[name]
                for c in u.cls:
                    uc = self.ucs[(name, c)]
                    coef = np.stack([np.array(F8.mul(uc.sc, F8.c(uc.alpha[:, j]))) for j in range(uc.ncols)], axis=1)
                    lincomb(H[c], u.columns(tensors, c), coef)
        tick("P4 pass")
        qs, oracles = self._basefold(tr, F8, G8, classes, NV, rr, V, H)
        del H
        if self.prover:
            for pu in pus:
                pu.gh = None
        tick("basefold")
        # P5
        if self.prover:
            out, tout, state = {}, {}, {}
            last = self.set_last(keys)
            if _POOL.stream is not None:
                _POOL.stream.synchronize()
            for name, tensors in provider():
                u = self.units[name]
                sid = self._sid(name)
                if all(tid in self.kept for tid in self._set_trees(sid)):
                    # stored trees: only the queried symbols, from the queried tiles of each column
                    for c in u.cls:
                        ql = torch.from_numpy(qs >> (NV - c)).to(_dev())
                        sym = (ql.view(-1, 1) * 4 + torch.arange(4, device=_dev()).view(1, -1)).reshape(-1)
                        cols = u.columns(tensors, c)
                        vals = []
                        for i in range(0, len(cols), CHUNK):
                            cw = encode_cols(cols[i:i + CHUNK], c, sym)
                            vals.append(cw[:, sym].reshape(cw.shape[0], -1, 4).permute(1, 0, 2))
                            del cw
                        out[(name, c)] = torch.cat(vals, dim=1)
                        if not last[(name, c)]:
                            continue
                        for tid in self._set_trees(sid):
                            tc = self._tree_c(tid)
                            levels = self.kept[tid]
                            lc = cap_level(1 << (tc - 1))
                            ti = torch.from_numpy(qs >> (NV - tc))
                            tout[tid] = (levels[lc].clone(), _paths(levels, ti, lc))
                    continue
                for c in u.cls:
                    ql = torch.from_numpy(qs >> (NV - c)).pin_memory().cuda(non_blocking=True)
                    # leaves of (set, class): continued across the set's units
                    state[(sid, c)], out[(name, c)] = leaf_pass(u.columns(tensors, c), c, state.get((sid, c)), ql)
                    if not last[(name, c)]:
                        continue
                    for tid in self._set_trees(sid):
                        tc = self._tree_c(tid)
                        levels = self._tree_levels(tid, state)
                        lc = cap_level(1 << (tc - 1))
                        ti = torch.from_numpy(qs >> (NV - tc)).pin_memory().cuda(non_blocking=True)
                        # device results; one host transfer after the pass keeps the GPU queue fed
                        tout[tid] = (levels[lc].clone(), _paths(levels, ti, lc))
                        del levels
                    for cc in self._set_classes(sid):
                        state.pop((sid, cc), None)
            assert not state
            for tid, (cap, pth) in tout.items():
                assert np.array_equal(cap.cpu().numpy().view(np.uint32), self.caps[tid]), \
                    f"{tid}: recomputed tree differs"
                tout[tid] = pth.cpu().numpy()
            for k, vals in out.items():
                out[k] = vals.cpu().numpy().view(np.uint32).astype(np.int64)
            self.kept = {}
            _POOL.release()
            tick("P5 pass")
            log = {}
            for k in keys:
                vals = out.pop(k)
                tr.record(vals)               # after the last challenge: recorded, not absorbed
                pth = np.zeros((0,), dtype=np.int32)
                if last[k]:                   # the set's paths: one message per tree
                    for tid in self._set_trees(self._sid(k[0])):
                        tp = tout.pop(tid)
                        tr.record_bytes(tp.astype("<i4").tobytes())
                        pth = np.concatenate([pth, tp.reshape(-1).astype(np.int32)])
                kk = k[0].split(".")[-1].rstrip("0123456789") + f"/{k[1]}"
                a = log.setdefault(kk, [0, 0, 0])
                a[0] += 4 * vals.size
                a[1] += pth.nbytes
                a[2] += self.units[k[0]].cls[k[1]].__len__()
            if os.environ.get("ZK_PCSLOG"):
                for kk, (vb, pb, nc) in sorted(log.items(), key=lambda x: -x[1][0] - x[1][1]):
                    print(f"[pcs] {kk:14s} cols {nc:6d} vals {vb / 2**20:7.2f} MiB paths {pb / 2**20:6.2f} MiB")
        else:
            self._verify_queries(tr, F8, keys, classes, NV, qs, oracles)
        return True

    def _rounds(self, tr, F8, G8, st, classes, NV, j0, j1, claim, r):
        inv2 = (P + 1) // 2
        for j in range(j0, j1):
            if self.prover:
                ev = [F8.zero, F8.zero, F8.zero]
                pairs = []                   # every active class's three dots, one host transfer
                for c in classes:
                    s = st[c]
                    if j < NV - c:
                        h = F8.smul(s["C"], inv2)
                        ev = [F8.add(e, h) for e in ev]
                    else:
                        E, V = s["E"], s["V"]
                        E0, E1 = E[:, 0::2].contiguous(), E[:, 1::2].contiguous()
                        V0, V1 = V[:, 0::2].contiguous(), V[:, 1::2].contiguous()
                        E2 = ((2 * E1.to(torch.int64) - E0.to(torch.int64)) % P).to(torch.int32)
                        V2 = ((2 * V1.to(torch.int64) - V0.to(torch.int64)) % P).to(torch.int32)
                        pairs += [(E0, V0), (E1, V1), (E2, V2)]
                if pairs:
                    ds = G8.dots(pairs)
                    for i in range(0, len(pairs), 3):
                        ev = [F8.add(ev[t], F8.c(ds[i + t])) for t in range(3)]
                evs = tr.send(np.array(ev, dtype=np.int64))
            else:
                evs = tr.send(None)
                assert evs.shape == (3, DO)
            ev = [F8.c(evs[i]) for i in range(3)]
            if F8.add(ev[0], ev[1]) != claim:
                raise ValueError("pcs: cross-unit sumcheck round failed")
            x = tr.challenge(D=DO)
            r.append(x)
            claim = F8.lagrange(ev, F8.c(x))
            if self.prover:
                for c in classes:
                    s = st[c]
                    if j < NV - c:
                        s["C"] = F8.smul(s["C"], inv2)
                    else:
                        s["E"] = G8.fold(s["E"], x)
                        s["V"] = G8.fold(s["V"], x)
        return claim

    def _basefold(self, tr, F8, G8, classes, NV, rr, V, H):
        """Returns (query leaf indices, oracle info for the verifier)."""
        FL = max(0, NV - FLOG)
        olv = [L for L in range(2, FL, 2)]
        join = {}
        for c in classes:
            join.setdefault(NV - c, []).append(c)
        tw_n = NV + 1
        prf = F8.one
        claim = V
        oracles = {}
        chal = []
        if self.prover:
            yc = {c: F8.c(G8.mle(H[c], rr[:, NV - c:])) for c in classes}
            Ysum = F8.zero
            for c in classes:
                if NV - c > 0:
                    Ysum = F8.add(Ysum, yc[c])
            T = G8.zeros(1 << NV)
            cw = G8.zeros(1 << (NV + 1))
            for c in join.get(0, []):
                T = G8.add(T, H[c])
                cw = G8.add(cw, codec.encode(H[c].contiguous(), c + 1, 0))
            Et = G8.eq_table(rr)
        for i in range(FL):
            ri = F8.c(rr[:, i])
            if self.prover:
                T0, T1 = T[:, 0::2].contiguous(), T[:, 1::2].contiguous()
                E0, E1 = Et[:, 0::2].contiguous(), Et[:, 1::2].contiguous()
                T2 = ((2 * T1.to(torch.int64) - T0.to(torch.int64)) % P).to(torch.int32)
                E2 = ((2 * E1.to(torch.int64) - E0.to(torch.int64)) % P).to(torch.int32)
                a = F8.mul(prf, Ysum)
                one_r = F8.sub(F8.one, ri)
                eqx = [one_r, ri, F8.sub(F8.smul(ri, 2), one_r)]
                ev = [F8.add(F8.c(G8.dot(A, B)), F8.mul(a, e)) for (A, B), e in zip(((T0, E0), (T1, E1), (T2, E2)), eqx)]
                del T0, T1, E0, E1, T2, E2
                evs = tr.send(np.array(ev, dtype=np.int64))
            else:
                evs = tr.send(None)
                assert evs.shape == (3, DO)
            ev = [F8.c(evs[k]) for k in range(3)]
            if F8.add(ev[0], ev[1]) != claim:
                raise ValueError("pcs: basefold sumcheck round failed")
            x = tr.challenge(D=DO)
            xf = F8.c(x)
            chal.append(xf)
            claim = F8.lagrange(ev, xf)
            prf = F8.mul(prf, F8.eq1(ri, xf))
            L = i + 1
            if self.prover:
                T = G8.fold(T, x)
                Et = G8.fold(Et, x)
                m = cw.shape[1] // 2
                o = torch.empty((DO, m), dtype=torch.int32, device=_dev())
                kern.launch(G8.f["cw_fold"], _grid(m), NT,
                            [ptr(cw), i64(cw.stride(0)), ptr(_itw2(G8, tw_n, i)), *G8.r8(x), ptr(o),
                             i64(o.stride(0)), i64(m)])
                cw = o
                if L in olv:
                    state = torch.empty((m // 4, 8), dtype=torch.int32, device=_dev())
                    leaf_state(cw, state, 0)
                    levels = codec.merkle_levels(state)
                    lc = cap_level(m // 4)
                    tr.send_bytes(levels[lc].cpu().numpy().astype("<i4").tobytes())
                    oracles[L] = (cw, levels, lc)
                for c in join.get(L, []):
                    T = G8.add(T, H[c])
                    cw = G8.add(cw, codec.encode(H[c].contiguous(), c + 1, 0))
                    Ysum = F8.sub(Ysum, yc[c])
            elif L in olv:
                b = tr.send_bytes(None)
                oracles[L] = np.frombuffer(b, dtype="<u4").reshape(-1, 8).astype(np.uint32)
        if self.prover:
            M = tr.send(G8.to_host(T).T).T
        else:
            M = tr.send(None).T
            assert M.shape == (DO, 1 << (NV - FL))
        if F8.mul(prf, F8.c(gops(DO).E.mle_eval(M, rr[:, FL:]) if FL < NV else M[:, 0])) != claim:
            raise ValueError("pcs: basefold final claim")
        if self.prover:
            tr.grind_bytes(self.grind)
        elif not tr.verify_grind_bytes(self.grind):
            raise ValueError("pcs: grinding")
        qs = tr.challenge_bits(NV - 1, self.queries)
        if self.prover:
            for L in olv:
                cw, levels, lc = oracles[L]
                li = torch.from_numpy(qs >> L).to(_dev())
                sym = (li.view(-1, 1) * 4 + torch.arange(4, device=_dev()).view(1, -1)).reshape(-1)
                tr.record(cw[:, sym].reshape(DO, -1, 4).permute(1, 0, 2).cpu().numpy().view(np.uint32).astype(np.int64))
                tr.record_bytes(_paths(levels, li, lc).cpu().numpy().astype("<i4").tobytes())
            return qs, None
        self._vb = dict(FL=FL, olv=olv, join=join, chal=chal, M=M, ocaps=oracles)
        return qs, oracles

    def _verify_queries(self, tr, F8, keys, classes, NV, qs, _):
        vb = self._vb
        FL, olv, join, chal, M = vb["FL"], vb["olv"], vb["join"], vb["chal"], vb["M"]
        Q = len(qs)
        odata = {}
        for L in olv:
            vals = tr.record(None)
            pth = _paths_msg(tr.record_bytes(None), Q, 1 << (NV - 1 - L))
            cap = vb["ocaps"][L]
            _expect(np.shape(vals) == (Q, DO, 4) and cap.shape == (_cap_rows(1 << (NV - 1 - L)), 8),
                    "pcs: oracle opening shape")
            for qi in range(Q):
                li = int(qs[qi]) >> L
                if not _check_path(leaf_digest_host(vals[qi]), pth[qi], li, cap):
                    raise ValueError("pcs: oracle merkle path")
            odata[L] = vals
        inj = {c: np.zeros((Q, DO, 4), dtype=np.int64) for c in classes}
        last = self.set_last(keys)
        hpart = {}                     # (set, class) -> leaf digests of the set's units so far
        for k in keys:
            uc = self.ucs[k]
            name, c = k
            vals = tr.record(None)
            _expect(np.shape(vals) == (Q, uc.ncols, 4), "pcs: leaf opening shape")
            sid = self._sid(name)
            prev = hpart.get((sid, c), [None] * Q)
            hpart[(sid, c)] = [leaf_digest_host(vals[qi], prev[qi]) for qi in range(Q)]
            if last[k]:
                for tid in self._set_trees(sid):
                    tc = self._tree_c(tid)
                    mixed = self._mixed(sid)[1] if tid[0] == "T" else [tc]
                    pth = _paths_msg(tr.record_bytes(None), Q, 1 << (tc - 1))
                    cap = self.caps[tid]
                    _expect(cap.shape == (_cap_rows(1 << (tc - 1)), 8), "pcs: cap size")
                    for qi in range(Q):
                        li = int(qs[qi]) >> (NV - tc)
                        if not _check_mixed([hpart[(sid, cc)][qi] if (sid, cc) in hpart else None
                                             for cc in range(tc + 1)], tc, mixed, pth[qi], li, cap):
                            raise ValueError(f"pcs: merkle path {name}/{tc}")
                for cc in self._set_classes(sid):
                    hpart.pop((sid, cc), None)
            inj[c] = (inj[c] + _combine_leaves(uc.coef, vals)) % P
        tw = rs.twiddles(NV + 1)
        Menc = rs.encode_ref(M, NV - FL + 1)            # (8, 2^(NV-FL+1))
        inv2 = (P + 1) // 2
        for qi in range(Q):
            q = int(qs[qi])
            base, held = 4 * q, [F8.zero] * 4
            for c in join.get(0, []):
                held = [F8.add(h, F8.c(inj[c][qi, :, s])) for s, h in enumerate(held)]
            for L in range(FL):
                nh = []
                for t in range(len(held) // 2):
                    a, b = held[2 * t], held[2 * t + 1]
                    s = base // 2 + t
                    f0 = F8.smul(F8.add(a, b), inv2)
                    f1 = F8.smul(F8.sub(a, b), pow(2 * int(tw[L][s]) % P, P - 2, P))
                    nh.append(F8.add(f0, F8.mul(chal[L], F8.sub(f1, f0))))
                held, base = nh, base // 2
                Ln = L + 1
                if Ln in olv:
                    ov = odata[Ln][qi]
                    ob = 4 * (q >> Ln)
                    if F8.c(ov[:, base - ob]) != held[0]:
                        raise ValueError("pcs: fold consistency")
                    held, base = [F8.c(ov[:, s]) for s in range(4)], ob
                for c in join.get(Ln, []):
                    cb = 4 * (q >> Ln)
                    held = [F8.add(h, F8.c(inj[c][qi, :, base - cb + s])) for s, h in enumerate(held)]
            for s, h in enumerate(held):
                if F8.c(Menc[:, base + s]) != h:
                    raise ValueError("pcs: final codeword consistency")


def _expect(cond, msg):
    if not cond:
        raise ValueError(msg)


def _cap_rows(nleaves):
    """Entries of the cap of a tree with nleaves leaves."""
    return nleaves >> cap_level(nleaves)


def _paths_msg(b, Q, nleaves):
    """Q Merkle paths of a tree with nleaves leaves from a message: (Q, depth, 8), the depth fixed
    by the tree (a path of another length is a rejection, not an index error)."""
    depth = cap_level(nleaves)
    _expect(len(b) == Q * depth * 32, "pcs: merkle path length")
    return np.frombuffer(b, dtype="<u4").reshape(Q, depth, 8)


def _combine_leaves(coef, vals):
    """coef (8, n) Fo, vals (Q, n, 4) base -> (Q, 8, 4) = sum_k coef[:, k] vals[:, k, :]."""
    co = np.asarray(coef, dtype=np.int64) % P
    v = np.asarray(vals, dtype=np.int64) % P
    lo, hi = v & 0xFFFF, v >> 16
    out = np.zeros((v.shape[0], DO, 4), dtype=np.int64)
    for i in range(0, co.shape[1], 1 << 15):
        cs = co[:, i:i + (1 << 15)]
        a = np.einsum("dk,qks->qds", cs, lo[:, i:i + (1 << 15)]) % P
        b = np.einsum("dk,qks->qds", cs, hi[:, i:i + (1 << 15)]) % P
        out = (out + a + b * 65536) % P
    return out
