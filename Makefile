# Build and test Exacta with SBCL.
#   make            the standalone executable ./exacta
#   make core       exacta.core, for `sbcl --core exacta.core'
#   make test       compile and run the test suite
SBCL ?= sbcl
HEAP ?= 8GB
SRC := exacta.asd build.lisp $(wildcard src/*.lisp) $(wildcard tests/*.lisp) tests/vectors.sexp $(wildcard tests/models/*.exm)

all: exacta

exacta: $(SRC)
	$(SBCL) --dynamic-space-size $(HEAP) --non-interactive --load build.lisp

exacta.core: $(SRC)
	$(SBCL) --non-interactive --load build.lisp core

core: exacta.core

test:
	$(SBCL) --dynamic-space-size $(HEAP) --non-interactive --eval '(require "asdf")' --eval '(asdf:load-asd (truename "exacta.asd"))' \
	  --eval '(asdf:test-system "exacta")'

clean:
	rm -f exacta exacta.core
	find . -name '*.fasl' -delete

.PHONY: all core test clean
