"""A plain byte format for proofs (the transcript's message list), so that a verifier never unpickles
data from a prover.

  proof   = MAGIC | u32 count | message*
  message = u8 0 | u64 length | bytes                                  (a byte message)
          | u8 1 | u8 ndim | u64 dim * ndim | u32 value * prod(dims)  (a field array, values < p)

All integers little-endian.  loads() checks every length against the remaining input and every value
against p, and raises ValueError on any malformed proof."""
import struct
import numpy as np
from .field import P

MAGIC = b"SABLEPF1"
MAX_NDIM = 8


def dumps(msgs):
    out = [MAGIC, struct.pack("<I", len(msgs))]
    for m in msgs:
        if isinstance(m, (bytes, bytearray)):
            out += [b"\x00", struct.pack("<Q", len(m)), bytes(m)]
        else:
            a = np.asarray(m)
            if a.ndim > MAX_NDIM:
                raise ValueError("proof: array rank")
            a = (a.astype(np.int64) % P).astype("<u4")
            out += [b"\x01", struct.pack("<B", a.ndim), struct.pack(f"<{a.ndim}Q", *a.shape), a.tobytes()]
    return b"".join(out)


def loads(b):
    b = memoryview(bytes(b))
    pos = 0

    def take(n):
        nonlocal pos
        if n < 0 or pos + n > len(b):
            raise ValueError("proof: truncated")
        v = b[pos:pos + n]
        pos += n
        return v
    if bytes(take(len(MAGIC))) != MAGIC:
        raise ValueError("proof: not a Sable proof")
    (count,) = struct.unpack("<I", take(4))
    msgs = []
    for _ in range(count):
        tag = take(1)[0]
        if tag == 0:
            (n,) = struct.unpack("<Q", take(8))
            msgs.append(bytes(take(n)))
        elif tag == 1:
            ndim = take(1)[0]
            if ndim > MAX_NDIM:
                raise ValueError("proof: array rank")
            shape = struct.unpack(f"<{ndim}Q", take(8 * ndim))
            n = int(np.prod(shape, dtype=np.uint64)) if ndim else 1
            if n > (len(b) - pos) // 4:
                raise ValueError("proof: truncated")
            a = np.frombuffer(take(4 * n), dtype="<u4")
            if a.size and int(a.max()) >= P:
                raise ValueError("proof: field value not below p")
            msgs.append(a.astype(np.int64).reshape(shape))
        else:
            raise ValueError("proof: unknown message type")
    if pos != len(b):
        raise ValueError("proof: trailing bytes")
    return msgs
