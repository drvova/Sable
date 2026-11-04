;;;; sumcheck.lisp -- the sumcheck verifier (zk/sumcheck.py verify, zk/cuda/vcore.cpp sc_verify)
;;;; and a reference CPU prover for products of multilinear tables.
;;;;
;;;; Proves  sum_{x in {0,1}^n} w(x) F(x) = claim  with w(x) = eq(r, x) (eq mode) or 1.  Variables
;;;; are bound LSB first.  The transcript first absorbs the header  label | n u16 | d u16 | K u32.
;;;; Round j sends g_j at X in {0, 2, ..., d} (a field array of shape (max(d,1), D)); g_j(1) follows
;;;; from the claim.  In eq mode g_j excludes the eq factor of the round variable and the reduced
;;;; claim excludes the whole eq factor, so  claim_j = (1 - r_j) g_j(0) + r_j g_j(1).

(defpackage #:sable.sumcheck
  (:use #:cl #:sable.conditions #:sable.field #:sable.blake3 #:sable.transcript)
  (:export #:header #:verify #:next-claim #:prove-product #:check-reduced #:eval-terms
           #:product-claim))

(in-package #:sable.sumcheck)

(defun header (label n d k)
  "label | n (u16 LE) | d (u16 LE) | K (u32 LE)."
  (octets-concat (if (stringp label) (sb-ext:string-to-octets label :external-format :utf-8) label)
                 (list (char-code #\|)) (u64-le n 2) (u64-le d 2) (u64-le k 4)))

(defun row (msg i d)
  "Row I of the round message MSG (a field array (npts, D)) as an EXT."
  (let ((e (make-ext d)))
    (dotimes (c d e) (setf (aref e c) (aref msg i c)))))

(defun next-claim (claim msg rj x d deg)
  "From the round message MSG (rows g(0), g(2..npts)) and the claim: g(X).  RJ is the eq-mode
coordinate of this round (or NIL)."
  (let* ((npts (max deg 1))
         (g0 (row msg 0 d))
         (g1 (if rj
                 (ext* (ext- claim (ext* (ext- (ext-one d) rj) g0)) (ext-inv rj))
                 (ext- claim g0)))
         (vals (list* g0 g1 (loop for i from 1 below npts collect (row msg i d)))))
    (lagrange (subseq vals 0 (1+ (max deg 1))) x)))

(defun verify (tr n deg claim &key eq-r (k 1) (label "sc"))
  "Replay an N-round sumcheck of degree DEG on the verifier transcript TR.  CLAIM is an EXT; EQ-R
(eq mode) a sequence of at least N EXTs.  Returns (values rho reduced-claim): RHO the N round
challenges; the caller checks REDUCED-CLAIM against the final evaluations (CHECK-REDUCED).
A message of the wrong shape signals MESSAGE-SHAPE-ERROR."
  (let* ((d (transcript-d tr))
         (npts (max deg 1))
         (claim (ext-coerce claim d))
         (rho (make-array n)))
    (absorb tr (header label n deg k))
    (dotimes (j n (values rho claim))
      (let* ((msg (send tr nil (list npts d)))
             (rj (challenge tr d)))
        (setf (aref rho j) rj
              claim (next-claim claim msg (and eq-r (elt eq-r j)) rj d deg))))))

(defun check-reduced (reduced expected &optional (what :sumcheck-final))
  "Reject the proof unless REDUCED = EXPECTED (both EXTs)."
  (unless (ext= reduced expected)
    (reject what "reduced claim (~{~D~^ ~}), final evaluations give (~{~D~^ ~})"
            (coerce reduced 'list) (coerce expected 'list)))
  t)

;;; ------------------------------------------------------------------ reference prover

(defun eq-weights (eq-r from n d)
  "eq(r_{from..n-1}, x) for x over the remaining variables, or all ones."
  (if eq-r
      (eq-table (subseq (coerce eq-r 'simple-vector) from n) d)
      (make-array (ash 1 (- n from)) :initial-element (ext-one d))))

(defun product-claim (tables n d &key eq-r)
  "sum_x w(x) prod_k T_k(x) for TABLES, a sequence of vectors of 2^N EXTs."
  (let ((w (eq-weights eq-r 0 n d))
        (acc (ext-zero d)))
    (dotimes (x (ash 1 n) acc)
      (let ((m (aref w x)))
        (map nil (lambda (tb) (setf m (ext* m (aref tb x)))) tables)
        (setf acc (ext+ acc m))))))

(defun prove-product (tr n tables &key eq-r (label "sc"))
  "Prover (CPU reference): the sumcheck of sum_x w(x) prod_k T_k(x), for TABLES a sequence of
vectors of 2^N EXTs, on the prover transcript TR.  Returns (values rho finals): FINALS are the
tables' values at RHO, whose product is the verifier's reduced claim."
  (let* ((d (transcript-d tr))
         (tables (map 'list (lambda (tb) (coerce tb 'simple-vector)) tables))
         (deg (length tables))
         (npts (max deg 1))
         (rho (make-array n)))
    (absorb tr (header label n deg 1))
    (dotimes (j n)
      (let* ((half (floor (length (first tables)) 2))
             (w (if eq-r (eq-weights eq-r (1+ j) n d) (make-array half :initial-element (ext-one d))))
             (msg (make-array (list npts d) :element-type '(unsigned-byte 32))))
        (loop for xv in (cons 0 (loop for v from 2 to npts collect v))
              for row from 0
              do (let ((acc (ext-zero d)))
                   (dotimes (i half)
                     (let ((m (aref w i)))
                       (dolist (tb tables)
                         (let ((lo (aref tb (* 2 i))) (hi (aref tb (1+ (* 2 i)))))
                           (setf m (ext* m (ext+ lo (ext-scale (ext- hi lo) xv))))))
                       (setf acc (ext+ acc m))))
                   (dotimes (c d) (setf (aref msg row c) (aref acc c)))))
        (send tr msg)
        (let ((r (challenge tr d)))
          (setf (aref rho j) r
                tables (mapcar (lambda (tb)
                                 (let ((nx (make-array half)))
                                   (dotimes (i half nx)
                                     (let ((lo (aref tb (* 2 i))) (hi (aref tb (1+ (* 2 i)))))
                                       (setf (aref nx i) (ext+ lo (ext* r (ext- hi lo))))))))
                               tables)))))
    (values rho (mapcar (lambda (tb) (aref tb 0)) tables))))

;;; ------------------------------------------------------------------ final evaluation

(defun eval-terms (terms finals d &optional bw)
  "sum_b bw_b sum_t c_t prod_{k in t} finals[k][b] (zk/sumcheck.py eval_terms).  TERMS: a list of
(c . refs), c an EXT and refs table indices; FINALS: per table a vector of K EXTs (or of 1, shared
by every block); BW: a vector of K EXTs or NIL."
  (let* ((finals (coerce finals 'simple-vector))
         (kk (if (plusp (length finals)) (reduce #'max finals :key #'length) 1))
         (acc (ext-zero d)))
    (dotimes (b kk acc)
      (let ((s (ext-zero d)))
        (loop for (c . refs) in terms
              do (let ((v (ext-coerce c d)))
                   (dolist (k refs)
                     (let ((f (aref finals k)))
                       (setf v (ext* v (aref f (if (> (length f) 1) b 0))))))
                   (setf s (ext+ s v))))
        (when bw (setf s (ext* s (elt bw b))))
        (setf acc (ext+ acc s))))))
