"""Check domain proofs made by the Lisp port with zk/domain.py's own verifier (domain.verify, on the
host: llm.verify(..., cpu=True)'s mode, with the C++ verifier core).  For each proof, verify must
accept it, consume every message and compute the same claims and lookup roots as the Lisp
verifier; with any one value changed, it must reject the proof.

Needs numpy, blake3, g++ and PyTorch (the CPU build is enough).  CuPy is not needed: zk/kern.py
imports it at module level but uses it only to compile GPU kernels, so a stand-in module is used.

    make crosscheck PYTHON=/path/to/python-with-torch
"""
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.modules.setdefault("cupy", types.ModuleType("cupy"))
from zk import ext, domain, proofio  # noqa: E402
from zk.poly import Poly  # noqa: E402
from zk.sumcheck import FULL, ROW, COL  # noqa: E402
from zk.transcript import Transcript  # noqa: E402

ext.CPU[0] = True
KINDS = {"full": FULL, "row": ROW, "col": COL}


def poly(form, var):
    if isinstance(form, int):
        return Poly.const(form)
    if isinstance(form, str):
        return var[form]
    op, *args = form
    ps = [poly(a, var) for a in args]
    if op == "+":
        out = Poly.const(0)
        for p in ps:
            out = out + p
        return out
    if op == "*":
        out = Poly.const(1)
        for p in ps:
            out = out * p
        return out
    if op == "-":
        return -ps[0] if len(ps) == 1 else ps[0] - sum(ps[1:], Poly.const(0))
    if op == "expt":
        return ps[0] ** args[1]
    raise ValueError(op)


def names(form):
    """The column names in a JSON form."""
    if isinstance(form, str):
        return [form]
    if isinstance(form, list):
        return [n for f in form for n in names(f)]
    return []


def build(case):
    dom = domain.Domain(case["name"], case["n"], case["K"], merge=case["merge"])
    var = {c["name"]: dom.col(c["name"], kind=KINDS[c["kind"]], c=c["c"], shared=c["shared"])
           for c in case["columns"]}
    for c in case["constraints"]:
        dom.zero(poly(c, var))
    for lk in case["lookups"]:
        dom.lookup(poly(lk["tag"], var), *[poly(v, var) for v in lk["values"]],
                   num=None if lk["num"] is None else poly(lk["num"], var))
    return dom


def verify(case, msgs):
    D = case["D"]
    tr = Transcript(case["label"].encode(), D, msgs)
    Z = np.array(case["Z"], dtype=np.int64)
    beta = np.array(case["beta"], dtype=np.int64).T            # (D, NBETA)
    claims, roots = domain.verify(tr, build(case), Z, beta)
    return tr, claims, roots


def main(path):
    cases = json.load(open(path))
    for case in cases:
        msgs = proofio.loads(bytes.fromhex(case["proof"]))
        tr, claims, roots = verify(case, msgs)
        shape = (case["name"], case["n"], case["K"], case["D"])
        assert tr.pos == len(msgs), (shape, "messages left over")
        assert [[list(map(int, p)), list(map(int, q))] for p, q in roots] == case["roots"], (shape, "roots")
        assert set(claims) == set(case["claims"]), (shape, "claimed columns")
        for name, (pt, val) in claims.items():
            want = case["claims"][name]
            assert np.asarray(pt).T.tolist() == want["point"], (shape, name, "point")
            assert np.asarray(val).T.tolist() == want["value"], (shape, name, "value")
        # A column that no constraint or lookup uses is not constrained by the domain proof: its final
        # only becomes its claim, which the polynomial commitment's opening checks.  Any other change
        # must be rejected.
        used = set(names(case["constraints"]) + names([[lk["tag"], lk["values"], lk["num"]] for lk in case["lookups"]]))
        rejected = passed_on = 0
        for i, m in enumerate(msgs):
            forged = list(msgs)
            forged[i] = m.copy()
            forged[i].reshape(-1)[0] = (forged[i].reshape(-1)[0] + 1) % proofio.P
            try:
                verify(case, forged)
            except (ValueError, AssertionError):
                rejected += 1
                continue
            # the finals are the last messages, one per table, in zk/domain.py's table order
            _, tabs = build(case)._tabmap()
            j = i - (len(msgs) - len(tabs))
            column = case["columns"][tabs[j][0]]["name"] if j >= 0 else None
            assert column is not None and column not in used, (shape, "forgery accepted at message", i)
            passed_on += 1
        print(f"{case['name']} n={case['n']} K={case['K']} D={case['D']}: accepted, {len(msgs)} messages, "
              f"same claims and roots as Lisp; {rejected} forgeries rejected"
              + (f", {passed_on} accepted that only change the final of a column the statement does not use"
                 if passed_on else ""))
    print(f"{len(cases)} domain proofs: zk/domain.py's verify agrees with the Lisp port")


if __name__ == "__main__":
    main(sys.argv[1])
