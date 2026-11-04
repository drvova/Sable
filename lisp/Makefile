# Build and test the Common Lisp port with SBCL.
#   make            the standalone executable ./sable
#   make core       sable.core, for `sbcl --core sable.core'
#   make test       compile and run the test suite
#   make vectors    regenerate tests/vectors.sexp from the Python code (numpy, blake3, g++)
#   make crosscheck check the Lisp GKR proofs with the original C++ verifier (numpy, g++)
SBCL ?= sbcl
HEAP ?= 8GB
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
	  --eval '(asdf:load-system "sable/tests")' --eval '(sable.tests::export-gkr-cases "gkr-cases.json")'
	cd .. && python3 lisp/tests/check-gkr-vcore.py lisp/gkr-cases.json
	rm -f gkr-cases.json

clean:
	rm -f sable sable.core
	find . -name '*.fasl' -delete

.PHONY: all core test vectors crosscheck clean
