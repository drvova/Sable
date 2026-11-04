;;;; exacta.asd -- Exacta: sumcheck proofs of bit-exact LLM inference, in Common Lisp (SBCL).

(defsystem "exacta"
  :description "Exacta: the verifier of Sable's sumcheck proofs of bit-exact LLM inference, with CPU
reference provers: the KoalaBear field tower, arithmetic as Lisp forms, the BLAKE3 Fiat-Shamir
transcript, the proof format, Reed-Solomon codes, sumcheck, LogUp-GKR, domains, the BaseFold
commitment and the transformer statement.  Ported from Sable's Python implementation."
  :version "1.0.0"
  :author "Tomer Solberg <research@moonmath.ai>"
  :license "MIT"
  :depends-on ((:require "sb-introspect"))
  :serial t
  :pathname "src/"
  :components ((:file "util")
               (:file "sha256")
               (:file "conditions")
               (:file "field")
               (:file "forms")
               (:file "blake3")
               (:file "transcript")
               (:file "proofio")
               (:file "rs")
               (:file "sumcheck")
               (:file "gkr")
               (:file "domain")
               (:file "pcs")
               (:file "proto")
               (:file "exact")
               (:file "tables")
               (:file "public")
               (:file "llm")
               (:file "model")
               (:file "toplevel"))
  :in-order-to ((test-op (test-op "exacta/tests"))))

(defsystem "exacta/tests"
  :description "Tests, against vectors computed by the original Python implementation."
  :depends-on ("exacta")
  :serial t
  :pathname "tests/"
  :components ((:file "framework")
               (:static-file "vectors.sexp")
               (:static-file "models/tiny.exm")
               (:static-file "models/qwen3.exm")
               (:file "tests"))
  :perform (test-op (o c)
             (unless (uiop:symbol-call '#:exacta.tests '#:run-tests)
               (error "exacta tests failed"))))
