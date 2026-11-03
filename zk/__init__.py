"""Sable: a GPU prover and verifier for bit-exact LLM inference (see README.md)."""
if not __debug__:
    # parts of the prover and verifier state their checks as assert statements, which python -O removes
    raise ImportError("zk must not run with python -O (assertions are part of its checks)")
