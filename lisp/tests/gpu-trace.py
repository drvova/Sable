"""Proves a tiny model on a GPU machine and records what the host verifier does with the proof, so the
step bodies of llm.verify (Proto.run) can be ported to Lisp and checked message by message.

  python lisp/tests/gpu-trace.py prove OUT [--model tiny|qwen3|moe]   # needs CUDA
  python lisp/tests/gpu-trace.py trace OUT                            # CPU only (no GPU used)
  python lisp/tests/gpu-trace.py all OUT [--model ...]                # both; trace in a child process

Run from the repository root.  `prove` writes into the directory OUT:

  proof.bin        the proof in the SABLEPF1 byte format (zk/proofio.py)
  verifier.pkl     the verifier's setup: the model moved to the host, tokens, targets, C_W, policy
  statement.json   cfg (and its repr), shape, tokens, targets, C_W, policy bytes, nonce, salt,
                   const digest, the statement bytes, sampling seed: the inputs of the Lisp outer
                   layer (lisp/src/llm.lisp), which `trace` checks again

`trace` runs llm.verify(..., cpu=True) with CUDA hidden and writes trace.jsonl, one JSON object per
line, in call order:

  {"ev": "call", "fn": "sumcheck.verify", "depth": d, "pos": transcript message index,
   "absorbed": bytes absorbed, "state": hex digest of the transcript state, "args": {...}}
  {"ev": "ret", "fn": ..., "pos": ..., "absorbed": ..., "state": ..., "out": ...}
  {"ev": "raise", "fn": ..., "error": ...}

Sub-protocols (domain.verify, gkr.verify_upper, sumcheck.verify / verify_blocks, the GPCS methods)
record their full inputs and outputs; the Proto / Ctx / llm helper steps around them record spans
(no arguments) when they move the transcript, so the trace shows which step sends which messages.
Arrays larger than --max-elems are recorded as {"shape": ..., "sha256": ...} of their little-endian
u32 values mod p.  Field values are integers in [0, p); extension elements are lists of D coordinates
as the Python verifier holds them.

This script has not been run (the session that wrote it had no GPU); expect small fixes."""
import argparse
import functools
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODELS = {
    # name: (Cfg kwargs, (B, S, ngen)); column block bits come from llm.block_bits
    "tiny": (dict(L=2, d=256, nh=4, nkv=2, dh=128, dff=384, V=1024), (2, 256, 16)),
    "qwen3": (dict(L=2, d=256, nh=4, nkv=2, dh=128, dff=512, V=1024, qk_norm=True, qkv_bias=False),
              (2, 256, 16)),
    "moe": (dict(L=2, d=512, nh=4, nkv=2, dh=64, dff=128, V=1024, qk_norm=True, qkv_bias=False,
                 moe=dict(E=8, k=2, de=64)), (2, 256, 16)),
}


# ---------------------------------------------------------------------------- prove (GPU)
def _host(o):
    import torch
    for k, v in list(vars(o).items()):
        if torch.is_tensor(v):
            setattr(o, k, v.cpu())


def _shape(llm, cfg, B, S, ngen):
    return llm.Shp(cfg, B, S, *llm.block_bits(cfg), ngen)


def prove(out, model):
    import torch
    from zk import model as mdl, llm, proofio
    from tests.tiny import tiny_get
    from tests.test_llm import statement as greedy_statement
    os.makedirs(out, exist_ok=True)
    kw, (B, S, ngen) = MODELS[model]
    cfg = mdl.Cfg(**kw)
    M = mdl.quantize_weights(cfg, tiny_get(cfg, 3))
    mdl.calibrate(M, torch.randint(0, cfg.V, (1, 256), generator=torch.Generator().manual_seed(1)))
    sh = _shape(llm, M.cfg, B, S, ngen)
    tokens, targets = greedy_statement(M, B, S, ngen)
    policy = llm.Policy()
    nonce, salt = bytes(range(32)), bytes(range(32, 64))
    cw = llm.weights_digest(M, sh)
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw, policy=policy, nc=nonce, salt=salt)
    # the GPU verifier first: it also fills the fixed-unit digests the host verifier needs (setup)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw, policy=policy, nc=nonce, salt=salt)
    stmt = llm.statement(M, sh, tokens, targets, cw, policy, nonce, salt)
    const = mdl.const_digest(M, sh.S, policy.theta)
    seed = llm.sampling_seed(tokens, sh.ng, cw, policy, nonce)
    _host(M)
    for Ly in M.layers:
        _host(Ly)
    tok, tgt = tokens.cpu(), targets.cpu() if torch.is_tensor(targets) else targets
    open(os.path.join(out, "proof.bin"), "wb").write(proofio.dumps(msgs))
    with open(os.path.join(out, "verifier.pkl"), "wb") as f:
        pickle.dump(dict(model=model, M=M, tokens=tok, targets=tgt, cw=cw, policy=policy, nonce=nonce,
                         salt=salt), f)
    json.dump(dict(model=model, cfg=vars(M.cfg), cfg_repr=repr(vars(M.cfg)),
                   shape=dict(B=sh.B, S=sh.S, cb=sh.cb, cbv=sh.cbv, ngen=sh.ng, T=int(sh.T)),
                   tokens=np.asarray(tok).tolist(), targets=np.asarray(tgt).tolist(), C_W=cw.hex(),
                   policy=policy.encode().hex(), theta=policy.theta, nonce=nonce.hex(), salt=salt.hex(),
                   const_digest=const.hex(), statement=stmt.hex(), sampling_seed=bytes(seed).hex(),
                   messages=len(msgs), proof_bytes=os.path.getsize(os.path.join(out, "proof.bin"))),
              open(os.path.join(out, "statement.json"), "w"), indent=1, default=str)
    print(f"{model}: {len(msgs)} messages, proof in {out}")


# ---------------------------------------------------------------------------- trace (CPU)
class Trace:
    def __init__(self, path, max_elems):
        self.f = open(path, "w")
        self.max_elems = max_elems
        self.depth = 0
        self.tr = None               # the verifier's transcript, once seen
        self.n = 0

    # -- values
    def enc(self, o, d=0):
        import torch
        from zk.field import P
        if d > 12:
            return "..."
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (bytes, bytearray, memoryview)):
            return {"bytes": bytes(o).hex()}
        if torch.is_tensor(o):
            o = o.detach().cpu().numpy()
        if isinstance(o, np.ndarray):
            if o.dtype.kind in "iu":
                if o.size > self.max_elems:
                    a = (o.astype(np.int64) % P).astype("<u4")
                    return {"shape": list(o.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                return o.tolist()
            return {"dtype": str(o.dtype), "shape": list(o.shape)}
        if isinstance(o, dict):
            return {str(self.enc(k, d + 1)): self.enc(v, d + 1) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            if len(o) > self.max_elems:
                return {"len": len(o), "head": [self.enc(x, d + 1) for x in o[:8]]}
            return [self.enc(x, d + 1) for x in o]
        if isinstance(o, (set, frozenset)):
            return self.enc(sorted(o, key=repr), d)
        name = type(o).__name__
        if name == "Transcript":
            return {"transcript": self.where(o)}
        if hasattr(o, "__dict__") and name in ("UC", "SEG", "PU", "Unit"):
            return {"type": name, **{k: self.enc(v, d + 1) for k, v in vars(o).items()}}
        if hasattr(o, "name"):
            return {"type": name, "name": self.enc(getattr(o, "name"), d + 1)}
        return {"type": name, "repr": repr(o)[:200]}

    def where(self, tr):
        return dict(pos=tr.pos, absorbed=tr.n_absorbed, state=tr.h.copy().digest().hex())

    def emit(self, ev):
        self.f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        self.n += 1

    # -- wrappers
    def leaf(self, fn, name, tr_of):
        """A sub-protocol: full arguments and result."""
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def w(*a, **k):
            tr = tr_of(a, k)
            if tr is not None:
                self.tr = tr
            b = sig.bind(*a, **k)
            args = {n: self.enc(v) for n, v in b.arguments.items() if n not in ("self", "tr")}
            ev = dict(ev="call", fn=name, depth=self.depth, args=args)
            if self.tr is not None:
                ev.update(self.where(self.tr))
            self.emit(ev)
            self.depth += 1
            try:
                r = fn(*a, **k)
            except Exception as e:
                self.depth -= 1
                self.emit(dict(ev="raise", fn=name, error=f"{type(e).__name__}: {e}"))
                raise
            self.depth -= 1
            ev = dict(ev="ret", fn=name, out=self.enc(r))
            if self.tr is not None:
                ev.update(self.where(self.tr))
            self.emit(ev)
            return r
        return w

    def span(self, fn, name):
        """A protocol step: recorded only when the transcript moves inside it."""
        @functools.wraps(fn)
        def w(*a, **k):
            tr = self.tr
            if tr is None:
                return fn(*a, **k)
            p0, n0, i0 = tr.pos, tr.n_absorbed, self.n
            d = self.depth
            self.depth += 1
            try:
                r = fn(*a, **k)
            finally:
                self.depth -= 1
            if (tr.pos, tr.n_absorbed) != (p0, n0):
                self.emit(dict(ev="span", fn=name, depth=d, pos0=p0, absorbed0=n0, first_event=i0,
                               **self.where(tr)))
            return r
        return w


def _tr_arg(a, k):
    t = k.get("tr", a[0] if a else None)
    return t if type(t).__name__ == "Transcript" else None


def install(T):
    from zk import sumcheck, domain, gkr, gpcs, proto, llm, transcript
    init = transcript.Transcript.__init__

    def tr_init(self, *a, **k):              # the verifier's transcript: where every event is located
        init(self, *a, **k)
        if self.verifier:
            T.tr = self
    transcript.Transcript.__init__ = tr_init
    for mod, names, label in ((sumcheck, ("verify", "verify_blocks"), "sumcheck"),
                              (domain, ("verify",), "domain"), (gkr, ("verify_upper",), "gkr")):
        for n in names:
            setattr(mod, n, T.leaf(getattr(mod, n), f"{label}.{n}", _tr_arg))
    for n in ("declare", "merge", "segment_decl", "commit", "commit_caps", "open", "segment",
              "flush_segments", "finish"):
        if hasattr(gpcs.GPCS, n):
            setattr(gpcs.GPCS, n, T.leaf(getattr(gpcs.GPCS, n), f"gpcs.{n}", lambda a, k: a[0].tr))
    for cls, label in ((proto.Ctx, "ctx"), (llm.Proto, "proto")):
        for n, f in list(vars(cls).items()):
            if inspect.isfunction(f) and not n.startswith("__"):
                setattr(cls, n, T.span(f, f"{label}.{n}"))
    for n in ("matmul", "oh_check", "check_eq", "combine_blocks", "mle_pub", "pub_const_rows"):
        if hasattr(llm, n):
            setattr(llm, n, T.span(getattr(llm, n), f"llm.{n}"))


def trace(out, max_elems):
    from zk import llm, proofio
    d = pickle.load(open(os.path.join(out, "verifier.pkl"), "rb"))
    msgs = proofio.loads(open(os.path.join(out, "proof.bin"), "rb").read())
    M = d["M"]
    kw, (B, S, ngen) = MODELS[d["model"]]
    sh = _shape(llm, M.cfg, B, S, ngen)
    st = json.load(open(os.path.join(out, "statement.json")))
    stmt = llm.statement(M, sh, d["tokens"], d["targets"], d["cw"], d["policy"], d["nonce"], d["salt"])
    assert stmt.hex() == st["statement"], "the host statement differs from the prover's"
    T = Trace(os.path.join(out, "trace.jsonl"), max_elems)
    install(T)
    T.emit(dict(ev="begin", model=d["model"], messages=len(msgs), statement=stmt.hex()))
    try:
        r = llm.verify(M, sh, d["tokens"], d["targets"], msgs, C_W=d["cw"], policy=d["policy"], nc=d["nonce"],
                       salt=d["salt"], cpu=True)
    except Exception as e:
        T.emit(dict(ev="end", accepted=False, error=f"{type(e).__name__}: {e}"))
        T.f.close()
        raise
    T.emit(dict(ev="end", accepted=True, result=T.enc(r), **(T.where(T.tr) if T.tr else {})))
    T.f.close()
    print(f"host verifier accepted; {T.n} trace events in {os.path.join(out, 'trace.jsonl')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("prove", "trace", "all"))
    ap.add_argument("out")
    ap.add_argument("--model", choices=sorted(MODELS), default="tiny")
    ap.add_argument("--max-elems", type=int, default=4096)
    a = ap.parse_args()
    if a.what in ("prove", "all"):
        prove(a.out, a.model)
    if a.what == "trace":
        trace(a.out, a.max_elems)
    elif a.what == "all":                    # the host verifier in a process that sees no GPU
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
        sys.exit(subprocess.run([sys.executable, os.path.abspath(__file__), "trace", a.out,
                                 "--max-elems", str(a.max_elems)], env=env, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
