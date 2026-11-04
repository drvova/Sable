;;;; sumcheck.lisp -- the sumcheck verifier (zk/sumcheck.py verify, zk/cuda/vcore.cpp sc_verify)
;;;; and a reference CPU prover for products of multilinear tables.
;;;;
;;;; Proves  sum_{x in {0,1}^n} w(x) F(x) = claim  with w(x) = eq(r, x) (eq mode) or 1.  Variables
;;;; are bound LSB first.  The transcript first absorbs the header  label | n u16 | d u16 | K u32.
;;;; Round j sends g_j at X = 0, 2, 3, ..., d, as a field array of shape (max(d,1), D); g_j(1)
;;;; follows from the claim.  In eq mode g_j excludes the eq factor of the round variable and the
;;;; reduced claim excludes the whole eq factor, so  claim_j = (1 - r_j) g_j(0) + r_j g_j(1).

(defpackage #:sable.sumcheck
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.transcript)
  (:export #:header #:verify #:next-claim #:check-reduced
           #:product-claim #:prove-product #:eval-terms))

(in-package #:sable.sumcheck)

(defun header (label n degree blocks)
  "label | n (u16 LE) | degree (u16 LE) | blocks (u32 LE)."
  (concatenate 'octets (to-octets label) (to-octets "|")
               (le-bytes n 2) (le-bytes degree 2) (le-bytes blocks 4)))

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

(defun verify (tr n degree claim &key eq-r (blocks 1) (label "sc"))
  "Replay an N-round sumcheck of DEGREE on the verifier transcript TR.  CLAIM is an EXT (or a
sequence of coordinates); EQ-R, in eq mode, a sequence of at least N EXTs.  Returns the round
challenges and the reduced claim, which the caller checks against the final evaluations
(CHECK-REDUCED).  A round message of the wrong shape signals MESSAGE-SHAPE-ERROR."
  (let ((claim (coerce-ext claim))
        (shape (list (max degree 1) (transcript-degree tr))))
    (absorb tr (header label n degree blocks))
    (values (loop for j below n
                  collect (let ((message (send tr nil shape))
                                (r (challenge tr)))
                            (setf claim (next-claim claim message (and eq-r (elt eq-r j)) r))
                            r))
            claim)))

(defun check-reduced (reduced expected &optional (check :sumcheck-final))
  "Reject the proof unless REDUCED = EXPECTED."
  (unless (ext= reduced expected)
    (reject check "reduced claim (~{~D~^ ~}), final evaluations give (~{~D~^ ~})"
            (coerce reduced 'list) (coerce expected 'list)))
  t)

;;; ------------------------------------------------------------------ reference prover

(defun remaining-weights (eq-r from n d)
  "w over the variables FROM..N-1: eq(r_from.., x), or all ones without EQ-R."
  (if eq-r
      (eq-table (subseq (coerce eq-r 'vector) from n) d)
      (make-array (expt 2 (- n from)) :initial-element (ext-one d))))

(defun product-claim (tables n d &key eq-r)
  "sum_x w(x) prod_k T_k(x) for TABLES, a sequence of vectors of 2^N EXTs."
  (reduce #'ext+
          (loop for w across (remaining-weights eq-r 0 n d)
                for x from 0
                collect (reduce #'ext* tables :key (lambda (table) (aref table x)) :initial-value w))
          :initial-value (make-ext d)))

(defun line-value (table i x)
  "TABLE along its lowest variable at pair I, evaluated at X: lo + X (hi - lo)."
  (let ((lo (aref table (* 2 i)))
        (hi (aref table (1+ (* 2 i)))))
    (ext+ lo (ext-scale (ext- hi lo) x))))

(defun round-message (tables weights points d)
  "The round polynomial g(X) = sum_i w_i prod_k line_k,i(X) at X = 0, 2, 3, ..., POINTS, as the
rows of a field array."
  (flet ((g (x)
           (reduce #'ext+ (loop for w across weights
                                for i from 0
                                collect (reduce #'ext* tables :key (lambda (table) (line-value table i x))
                                                              :initial-value w))
                   :initial-value (make-ext d))))
    (make-array (list points d)
                :element-type '(unsigned-byte 32)
                :initial-contents (loop for x in (cons 0 (loop for x from 2 to points collect x))
                                        collect (coerce (g x) 'list)))))

(defun prove-product (tr n tables &key eq-r (label "sc"))
  "Prove sum_x w(x) prod_k T_k(x) on the prover transcript TR, for TABLES a sequence of vectors
of 2^N EXTs (a reference prover, on the CPU).  Returns the round challenges and the tables'
values at them, whose product is the verifier's reduced claim."
  (let* ((d (transcript-degree tr))
         (tables (map 'list (lambda (table) (coerce table 'vector)) tables))
         (degree (length tables)))
    (absorb tr (header label n degree 1))
    (values (loop for j below n
                  collect (let ((weights (remaining-weights eq-r (1+ j) n d)))
                            (send tr (round-message tables weights (max degree 1) d))
                            (let ((r (challenge tr)))
                              (setf tables (mapcar (lambda (table) (fold-table table r)) tables))
                              r)))
            (mapcar (lambda (table) (aref table 0)) tables))))

;;; ------------------------------------------------------------------ final evaluation

(defun eval-terms (terms finals d &optional block-weights)
  "sum_b bw_b sum_t c_t prod_{k in t} finals[k][b] (zk/sumcheck.py eval_terms).  TERMS: a list of
(c . table-indices), c an EXT; FINALS: per table a vector of K EXTs, or of one EXT shared by every
block; BLOCK-WEIGHTS: K EXTs, or NIL for weights 1."
  (let* ((finals (coerce finals 'vector))
         (blocks (reduce #'max finals :key #'length :initial-value 1)))
    (flet ((final (k b)
             (let ((f (aref finals k)))
               (aref f (if (> (length f) 1) b 0))))
           (weight (b)
             (if block-weights (elt block-weights b) (ext-one d))))
      (reduce #'ext+
              (loop for b below blocks
                    collect (ext* (weight b)
                                  (reduce #'ext+
                                          (loop for (c . indices) in terms
                                                collect (reduce #'ext* indices
                                                                :key (lambda (k) (final k b))
                                                                :initial-value (coerce-ext c)))
                                          :initial-value (make-ext d))))
              :initial-value (make-ext d)))))
