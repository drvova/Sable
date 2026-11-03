"""GPU-resident transcript segments (prover only).

The BLAKE3 transcript state moves to the device; absorbs, sends and challenges are
queued as single-thread kernels, sent messages go to a device log, and finish()
brings state and messages back (one sync).  The absorbed bytes are exactly those of
the host Transcript, so the proof and the verifier are unchanged."""
import os
import numpy as np
import torch
from . import kern
from .kern import ptr, i32, i64, h2d

_F = {}
_IMM = np.dtype([("w", np.uint64, 8)])
_COLS = np.dtype([("c", np.int32, 256)])


def funcs():
    if not _F:
        d = os.path.join(os.path.dirname(__file__), "cuda")
        src = '#include "field.cuh"\n#include "b3w.cuh"\n' + open(os.path.join(d, "trk.cu")).read()
        m = kern.compile_module(src, 4)
        for n in ("tr_round", "tr_sends", "tr_bytes", "tr_chal", "tr_bytes_imm", "tr_send_cols"):
            _F[n] = m.get_function(n)
    return _F


def get(tr, cap=1 << 20):
    """The transcript's device session if one is active, else a new device segment."""
    return tr.dev if getattr(tr, "dev", None) is not None else DevTr(tr, cap=cap)


class DevTr:
    def __init__(self, tr, cap=1 << 20, session=False):
        self.tr, self.D = tr, tr.D
        self.verifier = False
        self.session = session          # a whole-pass session: finish() is a no-op, end() ends it
        self.st = tr.to_device()
        self.log = torch.empty(cap, dtype=torch.int32, device="cuda")
        self.cur = 0
        self.ents = []
        self.ndraw = 0                  # challenge draws so far: their keys name points (lazy.KEYS)
        self.f = funcs()

    def reserve(self, nw):
        """Room for nw more log words, so pointers from the next _alloc calls stay valid."""
        if self.cur + nw > self.log.numel():
            nl = torch.empty(2 * (self.cur + nw), dtype=torch.int32, device="cuda")
            nl[:self.cur].copy_(self.log[:self.cur])
            self.log = nl

    def alloc_many(self, k, nw, shape):
        """k consecutive messages of nw words: device pointer of the first (the rest follow)."""
        self.reserve(k * nw)
        off = self.cur
        self.cur += k * nw
        self.ents.extend((off + i * nw, shape) for i in range(k))
        return self.log.data_ptr() + 4 * off

    def _alloc(self, nw, shape):
        off = self.cur
        self.cur += nw
        if self.cur > self.log.numel():
            nl = torch.empty(2 * self.cur, dtype=torch.int32, device="cuda")
            nl[:off].copy_(self.log[:off])
            self.log = nl
        self.ents.append((off, shape))
        return self.log.data_ptr() + 4 * off

    def absorb(self, data: bytes):
        if len(data) <= 64:
            imm = np.zeros((), dtype=_IMM)
            imm["w"].view(np.uint8)[:len(data)] = np.frombuffer(data, dtype=np.uint8)
            kern.launch(self.f["tr_bytes_imm"], 1, 32, [ptr(self.st), i32(len(data)), imm[()]])
            return
        b = h2d(np.frombuffer(data, dtype=np.uint8).copy())
        kern.launch(self.f["tr_bytes"], 1, 32, [ptr(self.st), ptr(b), i64(len(data))])

    def send_cols(self, T, cols):
        """Send columns cols of the (D, *) device table T, each as one (1, D) message.
        Returns the device rows (len(cols), D) (a view of the log)."""
        D = self.D
        ns = len(cols)
        assert ns <= 256 and T.shape[0] == D
        c = np.zeros((), dtype=_COLS)
        c["c"][:ns] = cols
        lp = self.cur
        for _ in range(ns):
            self._alloc(D, (1, D))
        kern.launch(self.f["tr_send_cols"], 1, 32, [ptr(self.st), ptr(T), i64(T.stride(0)), i32(ns), i32(D),
                                                    c[()], np.uint64(self.log.data_ptr() + 4 * lp)])
        return self.log[lp:lp + ns * D].view(ns, D)

    def round(self, sums_ptr, npts):
        """One sumcheck round: send (npts, D) sums, draw a D-challenge into a new device row."""
        D = self.D
        out = torch.empty(D, dtype=torch.int32, device="cuda")
        self.round_into(sums_ptr, npts, out.data_ptr())
        return out

    def round_into(self, sums_ptr, npts, r_ptr):
        D = self.D
        lp = self._alloc(npts * D, (npts, D))
        kern.launch(self.f["tr_round"], 1, 32, [ptr(self.st), np.uint64(sums_ptr), i32(npts * D),
                                                np.uint64(lp), np.uint64(r_ptr), i32(D)])

    def send(self, msg):
        """One message: device int32 field array (any shape, row-major), as Transcript.send."""
        nw = msg.numel()
        off = self.cur
        lp = self._alloc(nw, tuple(msg.shape))
        self.log[off:off + nw].copy_(msg.reshape(-1))
        kern.launch(self.f["tr_bytes"], 1, 32, [ptr(self.st), np.uint64(lp), i64(4 * nw)])

    def send_rows(self, rows, shape):
        """rows: device int32 (ns, nw), each row sent as one message of the given shape."""
        ns, nw = rows.shape
        lp = self.cur
        for _ in range(ns):
            self._alloc(nw, shape)
        kern.launch(self.f["tr_sends"], 1, 32, [ptr(self.st), ptr(rows), i32(ns), i32(nw), i64(rows.stride(0)),
                                                i64(rows.stride(1)), np.uint64(self.log.data_ptr() + 4 * lp)])

    def challenge(self, k=None, D=None):
        """Device int32: (D,) for k None, else (k, D) (row i = challenge i)."""
        D = self.D if D is None else D
        kk = D if k is None else D * k
        out = torch.empty(kk, dtype=torch.int32, device="cuda")
        kern.launch(self.f["tr_chal"], 1, 32, [ptr(self.st), i32(kk), ptr(out)])
        from .lazy import KEYS
        d = self.ndraw
        self.ndraw += 1
        if k is None:
            KEYS[out] = ((d, 0),)
            return out
        o = out.view(k, D)
        KEYS[o] = tuple((d, i) for i in range(k))
        return o

    def draw_keys(self, n):
        """Keys of n draws made by device kernels (sumcheck round challenges)."""
        d = self.ndraw
        self.ndraw += n
        return tuple((d + i, 0) for i in range(n))

    def send_bytes(self, b):
        """A byte message (as Transcript.send_bytes): absorbed on the device, recorded as is."""
        self.absorb(b)
        self.ents.append(("B", bytes(b)))

    def finish(self):
        if self.session:
            return
        self.end()

    def end(self):
        stc = self.st.cpu()
        logh = self.log[:self.cur].cpu().numpy().view(np.uint32).astype(np.int64)
        self.tr.from_device(stc, [e[1] if e[0] == "B" else logh[e[0]:e[0] + int(np.prod(e[1]))].reshape(e[1])
                                  for e in self.ents])
