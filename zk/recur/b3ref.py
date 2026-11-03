"""Reference BLAKE3 compression, vectorized over many compressions, recording the G-function
trace that the circuit (b3dom) proves.  Words are uint32 held in int64 numpy arrays."""
import numpy as np

IV = np.array([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=np.int64)
MSG_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
CHUNK_START, CHUNK_END, PARENT, ROOT = 1, 2, 4, 8
M32 = 0xFFFFFFFF
# G calls of a round: (a, b, c, d) state indices, message words 2i, 2i + 1
G_IDX = [(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
         (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)]
# message word order of round r: SCHED[r][i] is the block word used as the round's i-th word
SCHED = [list(range(16))]
for _ in range(6):
    SCHED.append([SCHED[-1][MSG_PERM[i]] for i in range(16)])


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def g_trace(a, b, c, d, mx, my):
    """One G call on word arrays: (outputs (a, b, c, d), intermediates dict)."""
    s1 = (a + b + mx) & M32
    x1 = d ^ s1
    d1 = rotr(x1, 16)
    s2 = (c + d1) & M32
    x2 = b ^ s2
    b1 = rotr(x2, 12)
    s3 = (s1 + b1 + my) & M32
    x3 = d1 ^ s3
    d2 = rotr(x3, 8)
    s4 = (s2 + d2) & M32
    x4 = b1 ^ s4
    b2 = rotr(x4, 7)
    return (s3, b2, s4, d2), dict(a=a, b=b, c=c, d=d, mx=mx, my=my, s1=s1, x1=x1, s2=s2, x2=x2,
                                  s3=s3, x3=x3, s4=s4, x4=x4)


def compress(cv, block, counter, blen, flags, trace=None):
    """cv (N, 8), block (N, 16) words; counter, blen, flags: (N,) or scalars.  Returns (N, 16)
    output words.  trace: list appended with each G call's intermediates (in call order:
    round r, call i = 8 r + i), each a dict of (N,) arrays."""
    N = cv.shape[0]
    counter = np.broadcast_to(np.asarray(counter, dtype=np.int64), (N,))
    st = [cv[:, i] for i in range(8)] + [np.full(N, IV[i]) for i in range(4)] + \
        [counter & M32, (counter >> 32) & M32, np.broadcast_to(np.asarray(blen, dtype=np.int64), (N,)),
         np.broadcast_to(np.asarray(flags, dtype=np.int64), (N,))]
    st = [np.array(x, dtype=np.int64) for x in st]
    for r in range(7):
        m = [block[:, SCHED[r][i]] for i in range(16)]
        for i, (ia, ib, ic, id_) in enumerate(G_IDX):
            (st[ia], st[ib], st[ic], st[id_]), w = g_trace(st[ia], st[ib], st[ic], st[id_], m[2 * i], m[2 * i + 1])
            if trace is not None:
                trace.append(w)
    out = [st[i] ^ st[i + 8] for i in range(8)] + [st[i + 8] ^ cv[:, i] for i in range(8)]
    return np.stack(out, axis=1)


def words_le(b):
    """bytes (multiple of 4) -> little-endian uint32 words."""
    return np.frombuffer(b, dtype="<u4").astype(np.int64)
