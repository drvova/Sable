"""Check llm.lisp, the outer layer of the transformer protocol, against zk/llm.py on the host:
the commitment layout (Proto.declare: groups, items as packed, merge sets, segments, weight groups)
for tiny models of the three families and for the paper's three models at its shapes; the policy's
bytes, the statement and the sampling seed; the verifier's statement checks; and C_W from caps.

model.const_digest (the F_O constants, from the model's registry entry) is replaced by the digest
the Lisp side used.  Needs numpy and PyTorch (the CPU build is enough); CuPy is replaced by a stand-in.

    make crosscheck PYTHON=/path/to/python-with-torch
"""
import hashlib
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.modules.setdefault("cupy", types.ModuleType("cupy"))
import torch  # noqa: E402
from zk import ext, llm, model as mdl  # noqa: E402
from zk.proto import Ctx  # noqa: E402
from zk.transcript import Transcript  # noqa: E402

ext.CPU[0] = True


def cfg_of(c):
    return mdl.Cfg(c["L"], c["d"], c["nh"], c["nkv"], c["dh"], c["dff"], c["V"], qk_norm=c["qk_norm"],
                   qkv_bias=c["qkv_bias"], moe=c["moe"] and dict(E=c["moe"]["E"], k=c["moe"]["k"], de=c["moe"]["de"]),
                   down_b=c["down_b"])


def shape_of(cfg, B, S, ng):
    cb, cbv = llm.block_bits(cfg)
    return llm.Shp(cfg, B, S, cb, cbv, ng)


def policy_of(p):
    return llm.Policy(gmax=p["gmax"], stop=p["stop"], theta=p["theta"], ctx_bound=p["ctx_bound"],
                      tokenizer=bytes.fromhex(p["tokenizer"]), template=bytes.fromhex(p["template"]),
                      hpre=bytes.fromhex(p["hpre"]), autoregressive=p["autoregressive"])


def main(path):
    cases = json.load(open(path))
    for c in cases["layouts"]:
        cfg = cfg_of(c["cfg"])
        sh = shape_of(cfg, c["B"], c["S"], c["ng"])
        ctx = Ctx("verify", Transcript(b"layout", 4, msgs=[]), 4)
        pr = llm.Proto(ctx, types.SimpleNamespace(cfg=cfg), sh, None, None)
        pr.theta = c["theta"]
        pr.declare()
        groups = [[g, [list(it) for it in ctx.pcs.units[g].items]] for g in ctx.pcs.order]
        assert groups == c["groups"], (c["name"], "groups")
        assert [list(m) for m in ctx.pcs.msets] == c["merges"], (c["name"], "merges")
        assert [list(s) for s in ctx.pcs.seg_decl] == c["segments"], (c["name"], "segments")
        assert pr.wgroups() == c["weight_groups"], (c["name"], "weight groups")
        items = sum(len(it) for _, it in groups)
        print(f"layout {c['name']} B={c['B']} S={c['S']} ng={c['ng']} theta={c['theta']}: {len(groups)} groups, "
              f"{items} items, {len(c['merges'])} merge sets, {len(c['segments'])} segments: equal")
    for c in cases["statements"]:
        cfg = cfg_of(c["cfg"])
        sh = shape_of(cfg, c["B"], c["S"], c["ng"])
        pol = policy_of(c["policy"])
        tokens, targets = torch.tensor(c["tokens"]), torch.tensor(c["targets"])
        const = bytes.fromhex(c["const"])
        mdl.const_digest = lambda M, S, theta=0: const       # the registry's F_O constants digest
        C_W, nonce, salt = (bytes.fromhex(c[k]) for k in ("C_W", "nonce", "salt"))
        assert pol.encode().hex() == c["policy_bytes"], (c["name"], "policy")
        st = llm.statement(types.SimpleNamespace(cfg=cfg), sh, tokens, targets, C_W, pol, nonce, salt)
        assert st.hex() == c["statement"], (c["name"], "statement")
        assert llm.sampling_seed(tokens, sh.ng, C_W, pol, nonce).hex() == c["seed"], (c["name"], "seed")
        print(f"statement {c['name']} theta={c['policy']['theta']}: policy, statement ({len(st)} bytes) "
              f"and sampling seed equal")
    cfg = cfg_of(cases["layouts"][0]["cfg"])
    sh = shape_of(cfg, 2, 256, 4)
    for i, c in enumerate(cases["checks"]):
        try:
            policy_of(c["policy"]).check(sh, torch.tensor(c["tokens"]), torch.tensor(c["targets"]))
            ok = True
        except ValueError:
            ok = False
        assert ok == c["ok"], (i, "statement check", ok)
    print(f"{len(cases['checks'])} statement checks: Policy.check accepts and rejects the same "
          f"({sum(c['ok'] for c in cases['checks'])} accepted)")
    h = hashlib.sha256(b"C_W|")
    for cap in cases["wdigest"]["caps"]:
        h.update(bytes.fromhex(cap))
    assert h.hexdigest() == cases["wdigest"]["digest"], "C_W"
    print("C_W from caps: equal")


if __name__ == "__main__":
    main(sys.argv[1])
