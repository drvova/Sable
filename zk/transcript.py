"""Fiat-Shamir transcript (BLAKE3 compression function in sequential mode, zk/cuda/b3s.h), field
elements by rejection sampling."""
import re
import numpy as np
import blake3
import torch
from .field import P
from .b3s import Hasher


class Transcript:
    """Prover mode records every sent message; verifier mode replays them."""

    def __init__(self, label: bytes, D: int, msgs=None):
        self.D = D
        self._h = Hasher(b"LLM-SUMCHECK-v6-O|" + label)
        self._pending = []           # prover: byte messages whose data is still being copied
        self.dev = None              # prover: device session (begin_device), see DevTr
        self.n_absorbed = 0
        self.verifier = msgs is not None
        self.msgs = list(msgs) if msgs is not None else []
        self.pos = 0
        self.sent_bytes = 0

    @property
    def h(self):
        """The hash state; deferred sends are absorbed first, a device session is ended."""
        if self._pending:
            self.flush()
        if self.dev is not None:
            self.end_device()
        return self._h

    # ---------------------------------------------------------------- device session (prover)
    def begin_device(self):
        """Keep the state on the device: absorbs, sends and challenges are queued as kernels and
        challenges come back as lazy device arrays (lazy.Lz) until an operation needs the host
        state (grinding, query sampling) or end_device()."""
        from .devtr import DevTr
        if self.dev is None and not self.verifier:
            self.flush()
            self.dev = DevTr(self, session=True)

    def end_device(self):
        d, self.dev = self.dev, None
        if d is not None:
            d.end()

    def send_bytes_deferred(self, fn):
        """Prover: a byte message produced later by fn() (e.g. after an asynchronous device copy);
        it is sent, in order, before the transcript state is next observed or another message sent."""
        self._pending.append(fn)

    def flush(self):
        pend, self._pending = self._pending, []
        for fn in pend:
            self.send_bytes(fn())

    def send(self, arr):
        """Prover: record + absorb a field array.  Verifier: pop + absorb."""
        if self._pending:
            self.flush()
        if self.dev is not None:
            from .lazy import to_dev, Lz
            d = to_dev(arr)
            self.dev.send(d.contiguous())
            if torch.is_tensor(arr):
                return Lz(d)
            return arr if isinstance(arr, Lz) else np.asarray(arr, dtype=np.int64) % P
        if self.verifier:
            arr = self.msgs[self.pos]
            self.pos += 1
        else:
            arr = np.asarray(arr, dtype=np.int64) % P
            self.msgs.append(arr)
        self.sent_bytes += 4 * arr.size
        self.absorb_ext(arr)
        return arr

    def record(self, arr=None):
        """A field-array message that is part of the proof but not absorbed: for the final query
        openings, after the last challenge (absorbing them would bind nothing, and a recursive
        verifier would hash the whole opening data a second time)."""
        assert self._pending == [] and self.dev is None
        if self.verifier:
            arr = self.msgs[self.pos]
            self.pos += 1
            return arr
        arr = np.asarray(arr, dtype=np.int64) % P
        self.msgs.append(arr)
        self.sent_bytes += 4 * arr.size
        return arr

    def record_bytes(self, b=None):
        """A byte message recorded without absorbing (see record)."""
        assert self._pending == [] and self.dev is None
        if self.verifier:
            b = self.msgs[self.pos]
            self.pos += 1
            return b
        self.msgs.append(b)
        self.sent_bytes += len(b)
        return b

    def send_bytes(self, b=None):
        if self._pending:
            self.flush()
        if self.dev is not None:
            self.dev.send_bytes(b)
            return b
        if self.verifier:
            b = self.msgs[self.pos]
            self.pos += 1
        else:
            self.msgs.append(b)
        self.sent_bytes += len(b)
        self.absorb(b)
        return b

    # GPU-resident transcript segments (prover): the state moves to the device,
    # device kernels absorb/squeeze, and the sent messages are recorded afterwards.
    def to_device(self):
        from .kern import h2d
        return h2d(self.h.st)

    def from_device(self, st, msgs):
        """st: device state (after .cpu() sync); msgs: list of int64 arrays sent on the device."""
        self.h.st[:] = st.cpu().numpy()
        for m in msgs:
            self.msgs.append(m)
            self.sent_bytes += len(m) if isinstance(m, (bytes, bytearray)) else 4 * m.size

    def absorb(self, data: bytes):
        if self.dev is not None:
            if self._pending:
                self.flush()
            self.dev.absorb(data)
            self.n_absorbed += len(data)
            return
        self.h.update(len(data).to_bytes(8, "little"))
        self.h.update(data)
        self.n_absorbed += len(data)

    def absorb_ext(self, a):
        """a: numpy/torch array of field values (any shape)."""
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        a = np.asarray(a, dtype=np.int64) % P
        self.absorb(a.astype("<u4").tobytes())

    def _squeeze(self, nbytes):
        out = self.h.copy().digest(length=nbytes)
        self.h.update(b"squeeze")
        self.h.update(out[:32])
        return out

    def challenge_base(self, k=1):
        if self.dev is not None:
            from .lazy import Lz
            if self._pending:
                self.flush()
            return Lz(self.dev.challenge(None, D=k))
        vals = []
        while len(vals) < k:
            buf = self._squeeze(4 * (k - len(vals) + 4))
            w = np.frombuffer(buf, dtype="<u4").astype(np.int64) & 0x7FFFFFFF
            vals.extend(int(v) for v in w if v < P)
        return np.array(vals[:k], dtype=np.int64)

    def challenge(self, k=None, D=None):
        """One ext element (D,) or k of them (D, k); D defaults to the transcript's."""
        D = self.D if D is None else D
        if self.dev is not None:
            from .lazy import Lz
            if self._pending:
                self.flush()
            from .lazy import KEYS, rows_lz
            if k is None:
                x = self.dev.challenge(None, D=D)
                return Lz(x, keys=KEYS.get(x))
            return rows_lz(self.dev.challenge(k, D=D))
        if k is None:
            return self.challenge_base(D)
        return self.challenge_base(D * k).reshape(k, D).T.copy()

    def challenge_bits(self, nbits, k):
        buf = self._squeeze(8 * k)
        w = np.frombuffer(buf, dtype="<u8")
        return (w & np.uint64((1 << nbits) - 1)).astype(np.int64)

    def grind_bytes(self, bits):
        """Proof of work on the GPU: the smallest nonce with H(state || nonce) < 2^(64-bits), sent
        (and absorbed) as 8 bytes."""
        from .gpu import module
        base = self.h.copy().digest()
        bt = torch.from_numpy(np.frombuffer(base, dtype="<i4").copy()).cuda()
        nonce = int(module(self.D).grind(bt, bits)) if bits else 0
        assert self.check_grind(base, nonce, bits)
        self.send_bytes(int(nonce).to_bytes(8, "little"))
        return nonce

    def verify_grind_bytes(self, bits):
        base = self.h.copy().digest()
        nonce = int.from_bytes(self.send_bytes(None), "little")
        return self.check_grind(base, nonce, bits)

    @staticmethod
    def check_grind(base_digest, nonce, bits):
        d = blake3.blake3(base_digest + int(nonce).to_bytes(8, "little")).digest(length=8)
        return int.from_bytes(d, "little") < (1 << (64 - bits))


_LAYER_PREFIX = re.compile(rb"(^|\|)\d+\.")


def label_name(name):
    """A domain / group name as absorbed in the transcript: without its layer prefixes ("12.mo" ->
    "mo", "3.oproj|3.rqo" -> "oproj|rqo"); the layer index is absorbed once per layer instead
    (llm.Proto.prove_layer), so a layer's transcript labels do not depend on the layer."""
    b = name.encode() if isinstance(name, str) else bytes(name)
    return _LAYER_PREFIX.sub(rb"\1", b)

