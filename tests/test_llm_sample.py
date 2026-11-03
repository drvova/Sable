"""Temperature sampling (theta > 0): prove/verify, and the sampled tokens are bound to the seed."""
import torch
from zk import llm
from zk import model as mdl
from tests.test_llm import qwen_tiny


def main():
    M = qwen_tiny()
    B, S, ngen = 2, 256, 16
    sh = llm.Shp(M.cfg, B, S, 7, 7, ngen)
    cw = llm.weights_digest(M, sh)
    pol = llm.Policy(theta=mdl.THETA.index(1.0))
    nc = bytes(range(32))
    tokens = torch.randint(0, M.cfg.V, (B, S), generator=torch.Generator().manual_seed(2)).cuda()
    u = mdl.prf_u(llm.sampling_seed(tokens, ngen, cw, pol, nc), B * ngen)
    tokens, targets = llm.decode(M, tokens, ngen, pol, u)
    _, greedy = llm.decode(M, tokens, ngen)
    print("sampled", targets[0].tolist())
    print("greedy ", greedy[0].tolist())
    assert not torch.equal(targets, greedy), "temperature 1 decoded the greedy sequence (test is vacuous)"
    msgs = llm.prove(M, sh, tokens, targets, C_W=cw, policy=pol, nc=nc)
    llm.verify(M, sh, tokens, targets, msgs, C_W=cw, policy=pol, nc=nc)
    print("sampled proof verifies")
    bad = targets.clone()
    bad[1, -1] = (bad[1, -1] + 1) % M.cfg.V
    for what, kw in (("last token", dict(targets=bad)), ("client nonce", dict(nc=bytes(32))),
                     ("temperature", dict(policy=llm.Policy(theta=mdl.THETA.index(0.7))))):
        args = dict(targets=targets, policy=pol, nc=nc)
        args.update(kw)
        try:
            llm.verify(M, sh, tokens, args["targets"], msgs, C_W=cw, policy=args["policy"], nc=args["nc"])
        except Exception as e:                     # noqa: BLE001
            print(f"other {what} rejected:", type(e).__name__, str(e)[:80])
        else:
            raise SystemExit(f"other {what} ACCEPTED")
    # a prover that claims a different last token cannot prove it
    try:
        llm.prove(M, sh, tokens, bad, C_W=cw, policy=pol, nc=nc)
    except AssertionError as e:
        print("prover refuses a non-sample:", str(e)[:60])
    else:
        raise SystemExit("prover produced a proof for a non-sampled token")


def test_llm_sample():
    main()


if __name__ == "__main__":
    main()
