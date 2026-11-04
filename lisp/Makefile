# Build and test the Common Lisp port with SBCL.
#   make            the standalone executable ./sable
#   make core       sable.core, for `sbcl --core sable.core'
#   make test       compile and run the test suite
#   make vectors    regenerate tests/vectors.sexp from the Python code (numpy, blake3, g++)
#   make crosscheck check Lisp proofs with the original verifiers: GKR with zk/cuda/vcore.cpp
#                   (numpy, g++); domains and the commitment with zk/domain.py and zk/gpcs.py
#                   (also PyTorch, CPU build: PYTHON=...); the statement, policy, sampling seed and
#                   commitment layout of llm.lisp with zk/llm.py (PyTorch)
SBCL ?= sbcl
HEAP ?= 8GB
PYTHON ?= python3
SRC := sable.asd build.lisp $(wildcard src/*.lisp) $(wildcard tests/*.lisp) tests/vectors.sexp

all: sable

sable: $(SRC)
	$(SBCL) --dynamic-space-size $(HEAP) --non-interactive --load build.lisp

sable.core: $(SRC)
	$(SBCL) --non-interactive --load build.lisp core

core: sable.core

test:
	$(SBCL) --non-interactive --eval '(require "asdf")' --eval '(asdf:load-asd (truename "sable.asd"))' \
	  --eval '(asdf:test-system "sable")'

vectors:
	cd .. && python3 lisp/tests/gen-vectors.py

crosscheck:
	$(SBCL) --non-interactive --eval '(require "asdf")' --eval '(asdf:load-asd (truename "sable.asd"))' \
	  --eval '(asdf:load-system "sable/tests")' --eval '(sable.tests::export-gkr-cases "gkr-cases.json")' \
	  --eval '(sable.tests::export-domain-cases "domain-cases.json")' \
	  --eval '(sable.tests::export-pcs-cases "pcs-cases.json")' \
	  --eval '(sable.tests::export-llm-cases "llm-cases.json")'
	cd .. && python3 lisp/tests/check-gkr-vcore.py lisp/gkr-cases.json
	@if $(PYTHON) -c "import torch" 2>/dev/null; then \
	  cd .. && $(PYTHON) lisp/tests/check-domain-python.py lisp/domain-cases.json && \
	  $(PYTHON) lisp/tests/check-pcs-python.py lisp/pcs-cases.json && \
	  $(PYTHON) lisp/tests/check-llm-python.py lisp/llm-cases.json; \
	else echo "skipping the domain, commitment and statement checks: $(PYTHON) has no PyTorch (make crosscheck PYTHON=...)"; fi
	rm -f gkr-cases.json domain-cases.json pcs-cases.json llm-cases.json

clean:
	rm -f sable sable.core
	find . -name '*.fasl' -delete

.PHONY: all core test vectors crosscheck clean
