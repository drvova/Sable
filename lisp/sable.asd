;;;; sable.asd -- the Common Lisp (SBCL) port of Sable's host-side verifier core.

(defsystem "sable"
  :description "Sable: KoalaBear field tower, BLAKE3 Fiat-Shamir transcript, proof byte format,
Reed-Solomon code and sumcheck verifier, ported from the Python implementation (zk/)."
  :version "1.0.0"
  :author "Tomer Solberg <research@moonmath.ai>"
  :license "MIT"
  :depends-on ()
  :serial t
  :pathname "src/"
  :components ((:file "conditions")
               (:file "field")
               (:file "blake3")
               (:file "transcript")
               (:file "proofio")
               (:file "poly")
               (:file "rs")
               (:file "sumcheck")
               (:file "cli"))
  :in-order-to ((test-op (test-op "sable/tests"))))

(defsystem "sable/tests"
  :description "Tests of the port against vectors computed by the Python implementation."
  :depends-on ("sable")
  :serial t
  :pathname "tests/"
  :components ((:file "framework")
               (:static-file "vectors.sexp")
               (:file "tests"))
  :perform (test-op (o c)
             (unless (uiop:symbol-call '#:sable.tests '#:run-tests)
               (error "sable tests failed"))))
