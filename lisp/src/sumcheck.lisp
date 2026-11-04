;;;; sumcheck.lisp -- the sumcheck protocol (zk/sumcheck.py, zk/cuda/vcore.cpp sc_verify), with a
;;;; reference CPU prover.
;;;;
;;;; The statement is a form (sable.forms) over multilinear tables, and optionally over fixed
;;;; values (CONSTANTS, e.g. challenges):
;;;;
;;;;   sum_{x in {0,1}^n} w(x) F(t0(x), t1(x), ...) = claim        F = '(* t0 t1), '(+ (* a t0) t1) ...
;;;;
;;;; with w(x) = eq(r, x) in eq mode, else 1.  Its degree in the tables fixes the round messages, and
;;;; the same form gives the verifier's final check.  Variables are bound LSB first.  The transcript
;;;; first absorbs  label | n u16 | d u16 | K u32.  Round j sends g_j at X = 0, 2, 3, ..., d, a field
;;;; array of shape (max(d,1), D); g_j(1) follows from the claim.  In eq mode g_j excludes the eq
;;;; factor of the round variable and the reduced claim the whole eq factor, so
;;;; claim_j = (1 - r_j) g_j(0) + r_j g_j(1).

(defpackage #:sable.sumcheck
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.forms #:sable.transcript)
  (:export #:header #:statement-degree #:hypercube-sum #:prove #:verify #:next-claim
           #:check-final #:demo))

(in-package #:sable.sumcheck)

(defun header (label n degree &optional (blocks 1))
  "label | n (u16 LE) | degree (u16 LE) | blocks (u32 LE)."
  (concatenate 'octets (to-octets label) (to-octets "|")
               (le-bytes n 2) (le-bytes degree 2) (le-bytes blocks 4)))

(defun table-variables (statement constants)
  "The variables of STATEMENT that are tables: all but the CONSTANTS."
  (set-difference (form-variables statement) (mapcar #'car constants)))

(defun statement-degree (statement &optional constants)
  "The degree of STATEMENT in its tables: the degree of each round polynomial."
  (form-degree statement (table-variables statement constants)))

(defun statement-function (statement tables constants d)
  "STATEMENT compiled as a function of the tables' values, in the order of TABLES (an alist)."
  (let ((f (compile-form statement (append (mapcar #'car tables) (mapcar #'car constants)) :degree d))
        (constant-values (mapcar #'cdr constants)))
    (lambda (&rest table-values)
      (apply f (append table-values constant-values)))))

(defun weights (eq from n d)
  "w over the variables FROM..N-1: eq(r_from.., x), or all ones without EQ."
  (if eq
      (eq-table (subseq (coerce eq 'vector) from n) d)
      (make-array (expt 2 (- n from)) :initial-element (ext-one d))))

(defun hypercube-sum (statement n tables d &key eq constants)
  "sum_x w(x) F(tables(x)): the claim of an honest prover.  TABLES: an alist (variable . vector of
2^N EXTs)."
  (let ((f (statement-function statement tables constants d)))
    (reduce #'ext+
            (loop for w across (weights eq 0 n d)
                  for x from 0
                  collect (ext* w (apply f (mapcar (lambda (table) (aref (cdr table) x)) tables))))
            :initial-value (make-ext d))))

;;; ------------------------------------------------------------------ verifier

(defun message-rows (message)
  "The rows of a round message, a field array (points, D), as EXTs."
  (destructuring-bind (points d) (array-dimensions message)
    (loop for i below points
          collect (coerce-ext (loop for c below d collect (aref message i c))))))

(defun next-claim (claim message eq-rj x)
  "g(X), from the round MESSAGE (g at 0, 2, 3, ..., d) and the CLAIM, which gives g(1).  EQ-RJ is
the eq-mode coordinate of the round, or NIL."
  (destructuring-bind (g0 &rest g2-and-up) (message-rows message)
    (let ((g1 (if eq-rj
                  ;; claim = (1 - r) g(0) + r g(1)
                  (ext* (ext- claim (ext* (ext- (ext-one (length g0)) eq-rj) g0)) (ext-inv eq-rj))
                  (ext- claim g0))))
      (lagrange (list* g0 g1 g2-and-up) x))))

(defun verify (tr n statement claim &key eq constants (blocks 1) (label "sc"))
  "Replay the N-round sumcheck of STATEMENT on the verifier transcript TR.  CLAIM is an EXT (or
its coordinates); EQ, in eq mode, a sequence of at least N EXTs; CONSTANTS an alist of the
statement's non-table variables.  Returns the round challenges and the reduced claim, for
CHECK-FINAL.  A round message of the wrong shape signals MESSAGE-SHAPE-ERROR."
  (let* ((degree (statement-degree statement constants))
         (shape (list (max degree 1) (transcript-degree tr)))
         (claim (coerce-ext claim)))
    (absorb tr (header label n degree blocks))
    (values (loop for j below n
                  collect (let ((message (send tr nil shape))
                                (r (challenge tr)))
                            (setf claim (next-claim claim message (and eq (elt eq j)) r))
                            r))
            claim)))

(defun check-final (reduced statement finals &key constants)
  "Reject the proof unless REDUCED is STATEMENT at FINALS, an alist (table . EXT) of the tables'
values at the round challenges."
  (let ((expected (evaluate statement (append finals constants) :degree (length reduced))))
    (unless (ext= reduced expected)
      (reject :sumcheck-final "reduced claim (~{~D~^ ~}), statement at the final values (~{~D~^ ~})"
              (coerce reduced 'list) (coerce expected 'list))))
  t)

;;; ------------------------------------------------------------------ prover

(defun line-value (table i x)
  "TABLE along its lowest variable at pair I, at X: lo + X (hi - lo)."
  (let ((lo (aref table (* 2 i)))
        (hi (aref table (1+ (* 2 i)))))
    (ext+ lo (ext-scale (ext- hi lo) x))))

(defun round-message (f tables weights points d)
  "The round polynomial g(X) = sum_i w_i F(line values at X) at X = 0, 2, 3, ..., POINTS, as the
rows of a field array."
  (flet ((g (x)
           (reduce #'ext+
                   (loop for w across weights
                         for i from 0
                         collect (ext* w (apply f (mapcar (lambda (table) (line-value (cdr table) i x))
                                                          tables))))
                   :initial-value (make-ext d))))
    (make-array (list points d)
                :element-type '(unsigned-byte 32)
                :initial-contents (loop for x in (cons 0 (loop for x from 2 to points collect x))
                                        collect (coerce (g x) 'list)))))

(defun prove (tr n statement tables &key eq constants (label "sc"))
  "Prove the sumcheck of STATEMENT on the prover transcript TR (a reference prover, on the CPU).
TABLES: an alist (variable . vector of 2^N EXTs).  Returns the round challenges and the finals, an
alist (variable . value at the challenges) for CHECK-FINAL."
  (let* ((d (transcript-degree tr))
         (degree (statement-degree statement constants))
         (tables (loop for (variable . table) in tables collect (cons variable (coerce table 'vector))))
         (f (statement-function statement tables constants d)))
    (absorb tr (header label n degree))
    (values (loop for j below n
                  collect (progn
                            (send tr (round-message f tables (weights eq (1+ j) n d) (max degree 1) d))
                            (let ((r (challenge tr)))
                              (setf tables (loop for (variable . table) in tables
                                                 collect (cons variable (fold-table table r))))
                              r)))
            (loop for (variable . table) in tables collect (cons variable (aref table 0))))))

;;; ------------------------------------------------------------------ demonstration

(defun demo (&key (statement '(* t0 t1 t2)) (variables 10) (degree 8) (seed 1))
  "Prove and verify the sumcheck of STATEMENT (in eq mode) on random tables of VARIABLES variables
over F_p^DEGREE, then check that a tampered proof is rejected.  Returns the proof's size in bytes."
  (let* ((state (sb-ext:seed-random-state seed))
         (random-ext (lambda () (coerce-ext (loop repeat degree collect (random +p+ state)))))
         (tables (loop for variable in (form-variables statement)
                       collect (cons variable (coerce (loop repeat (expt 2 variables)
                                                            collect (funcall random-ext))
                                                      'vector))))
         (eq (loop repeat variables collect (funcall random-ext)))
         (claim (hypercube-sum statement variables tables degree :eq eq))
         (start (get-internal-real-time))
         (prover (make-transcript "demo" degree))
         (finals (nth-value 1 (prove prover variables statement tables :eq eq)))
         (proof (sable.proofio:encode-proof (transcript-messages prover)))
         (proved (get-internal-real-time)))
    (flet ((verify-messages (messages)
             (let ((verifier (make-transcript "demo" degree :messages messages)))
               (check-final (nth-value 1 (verify verifier variables statement claim :eq eq))
                            statement finals)
               (unless (transcript-finished-p verifier)
                 (reject :unread-messages)))))
      (verify-messages (sable.proofio:decode-proof proof))
      (format t "~&sumcheck of ~S over ~D variables in F_p^~D, eq mode~%" statement variables degree)
      (format t "proved in ~,3F s (~:D-byte proof), verified in ~,3F s~%"
              (/ (- proved start) internal-time-units-per-second) (length proof)
              (/ (- (get-internal-real-time) proved) internal-time-units-per-second))
      ;; one value of the first round message changed: the proof must be rejected
      (let ((tampered (sable.proofio:decode-proof proof)))
        (setf (aref (aref tampered 0) 0 0) (fadd (aref (aref tampered 0) 0 0) 1))
        (handler-case (progn (verify-messages tampered)
                             (error "a tampered proof was accepted"))
          (verification-failure (c)
            (format t "tampered proof rejected: ~A~%" c))))
      (length proof))))
