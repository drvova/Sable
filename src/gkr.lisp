;;;; gkr.lisp -- batched LogUp-GKR fractional sums (zk/gkr.py, zk/cuda/vcore.cpp vc_gkr_upper).
;;;;
;;;; A domain has K blocks of 2^n leaves and L lookups.  Leaf (l, b, i) is a fraction p/q: a
;;;; numerator and a denominator Z - Q_l(i).  Each lookup has one tree of fractions, summed pairwise:
;;;; p/q = p0/q0 + p1/q1, so p = p0 q1 + p1 q0 and q = q0 q1.  In-block layers: the children of
;;;; (b, x) in the layer of 2^k entries per block are (b, x) and (b, x + 2^k).  Above the per-block
;;;; roots, a top tree over the block index, K padded to a power of two with fractions 0/1: the
;;;; children of b are b and b + 2^j.
;;;;
;;;; The proof sends the root fraction of each lookup, then proves each layer from the one above
;;;; with a sumcheck batched over the lookups, of the statement LAYER-STATEMENT.  In-block layers sum
;;;; over the blocks with weights eq(rb, b) and then bind the block index in a second sumcheck
;;;; (label "-blk").  The leaf layer is proved by the caller (zk/domain.py).

(defpackage #:exacta.gkr
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.forms #:exacta.transcript
        #:exacta.sumcheck)
  (:export #:layer-statement #:fraction-trees #:prove-upper #:verify-upper #:block-bits
           #:layer-claim #:block-weights))

(in-package #:exacta.gkr)

(defun block-bits (k)
  "The number of variables of the block index: K blocks padded to a power of two."
  (integer-length (1- k)))

(defun lookup-variable (name l)
  (intern (format nil "~A~D" name l) '#:exacta.gkr))

(defun layer-statement (lookups)
  "One layer of the batched trees, as a statement form:
  sum_l a_l (p0_l q1_l + p1_l q0_l) + (a_l lambda) q0_l q1_l      (zk/gkr.py _layer_terms).
Returns the form, its tables in the proof's order (p0, p1, q0, q1 for each lookup) and its
constants (a_l and a_l lambda for each lookup)."
  (loop for l below lookups
        for (p0 p1 q0 q1 a al) = (mapcar (lambda (name) (lookup-variable name l))
                                         '("P0-" "P1-" "Q0-" "Q1-" "A" "AL"))
        append (list p0 p1 q0 q1) into tables
        append (list a al) into constants
        append `((* ,a ,p0 ,q1) (* ,a ,p1 ,q0) (* ,al ,q0 ,q1)) into terms
        finally (return (values `(+ ,@terms) tables constants))))

(defun layer-challenges (tr lookups)
  "a_l for each lookup, then lambda; as the statement's constants (a_l and a_l lambda)."
  (let ((as (challenges tr lookups))
        (lam (challenge tr)))
    (values (loop for a in as
                  for l from 0
                  collect (cons (lookup-variable "A" l) a)
                  collect (cons (lookup-variable "AL" l) (ext* a lam)))
            as lam)))

(defun layer-claim (as lam cp cq)
  "sum_l a_l (cp_l + lambda cq_l): the layer's claim from the claims on the layer above."
  (reduce #'ext+ (mapcar (lambda (a p q) (ext* a (ext+ p (ext* lam q)))) as cp cq)))

(defun next-claims (finals lookups tt)
  "The claims on the layer's tables at the fresh point TT: p0 + tt (p1 - p0), q0 + tt (q1 - q0)."
  (flet ((final (name l) (cdr (assoc (lookup-variable name l) finals))))
    (loop for l below lookups
          collect (fold-pair (final "P0-" l) (final "P1-" l) tt) into cp
          collect (fold-pair (final "Q0-" l) (final "Q1-" l) tt) into cq
          finally (return (values cp cq)))))

(defun fold-pair (lo hi r)
  (ext+ lo (ext* r (ext- hi lo))))

(defun padded-mass (rb k d)
  "sum_{b >= K} eq(rb, b): the denominators of the padded per-block roots (zk/gkr.py pad_q)."
  (let ((e (eq-table rb d)))
    (reduce #'ext+ (subseq e k) :initial-value (make-ext d))))

(defun block-weights (rb k d)
  "eq(rb, b) for b < K; (1) when there is one block."
  (if (null rb)
      (vector (ext-one d))
      (subseq (eq-table rb d) 0 k)))

(defun label+ (label suffix)
  (concatenate 'octets (to-octets label) (to-octets suffix)))

;;; ------------------------------------------------------------------ verifier

(defun receive-finals (tr tables d)
  "The layer's finals, one message (1, D) per table, then the fresh point."
  (values (loop for name in tables
                collect (cons name (coerce-ext (let ((m (send tr nil (list 1 d))))
                                                 (loop for c below d collect (aref m 0 c))))))
          (challenge tr)))

(defun verify-upper (tr n k lookups &key (label "gkr"))
  "Replay the upper layers of the batched trees of LOOKUPS lookups over K blocks of 2^N leaves
(zk/gkr.py verify_upper).  Returns the root fractions (a list of (p . q)), the in-block point
(N - 1 EXTs), the block point and the claims p and q on the leaf layer for each lookup.  A failed
layer check signals VERIFICATION-FAILURE (:GKR-TOP or :GKR-LAYER)."
  (let* ((d (transcript-degree tr))
         (lk (block-bits k))
         (roots (loop repeat lookups
                      collect (let ((m (send tr nil (list 2 d))))
                                (cons (coerce-ext (loop for c below d collect (aref m 0 c)))
                                      (coerce-ext (loop for c below d collect (aref m 1 c)))))))
         (cp (mapcar #'car roots))
         (cq (mapcar #'cdr roots))
         (r '()))
    (multiple-value-bind (statement tables) (layer-statement lookups)
      ;; the top tree, over the block index
      (dotimes (j lk)
        (multiple-value-bind (constants as lam) (layer-challenges tr lookups)
          (multiple-value-bind (rho reduced)
              (verify tr j statement (layer-claim as lam cp cq) :eq (and (plusp j) r)
                      :constants constants :label (label+ label "-top") :degree (if (plusp j) 2 0))
            (multiple-value-bind (finals tt) (receive-finals tr tables d)
              (unless (ext= reduced (evaluate statement (append finals constants) :degree d))
                (reject :gkr-top "layer ~D of the top tree" j))
              (setf (values cp cq) (next-claims finals lookups tt)
                    r (append rho (list tt)))))))
      ;; the padded blocks' roots have denominators 1
      (let ((rb r)
            (ri '()))
        (unless (= (expt 2 lk) k)
          (let ((mass (padded-mass rb k d)))
            (setf cq (mapcar (lambda (q) (ext- q mass)) cq))))
        ;; the in-block layers
        (dotimes (layer (1- n))
          (multiple-value-bind (constants as lam) (layer-challenges tr lookups)
            (let ((bw (block-weights rb k d)))
              (multiple-value-bind (rho reduced)
                  (verify tr layer statement (layer-claim as lam cp cq) :eq (and (plusp layer) ri)
                          :constants constants :blocks k :label label :degree (if (plusp layer) 2 0))
                (let ((rbn '()))
                  (when (plusp lk)
                    (setf (values rbn reduced)
                          (verify tr lk `(* bw ,statement) reduced :constants constants
                                  :label (label+ label "-blk"))))
                  (multiple-value-bind (finals tt) (receive-finals tr tables d)
                    (let ((weight (if (> k 1)
                                      (reduce #'ext+ (map 'list #'ext* bw (eq-table rbn d)))
                                      (svref bw 0))))
                      (unless (ext= reduced (ext* weight (evaluate statement (append finals constants)
                                                                   :degree d)))
                        (reject :gkr-layer "in-block layer ~D" layer)))
                    (setf (values cp cq) (next-claims finals lookups tt)
                          ri (append rho (list tt))
                          rb rbn)))))))
        (values roots ri rb cp cq)))))

;;; ------------------------------------------------------------------ prover (CPU reference)

(defun fraction-trees (leaves n k)
  "The fraction trees of LEAVES, a list per lookup of (p . q), each a vector of K blocks of 2^N
EXTs (block-major).  Returns the in-block layers (a vector indexed by the layer m, of a list per
lookup of (p . q) with 2^m entries per block) and the top tree (indexed by j, 2^j entries)."
  (let* ((d (length (aref (car (first leaves)) 0)))
         (lk (block-bits k))
         (layers (make-array (1+ n))))
    (setf (aref layers n) leaves)
    (loop for m from (1- n) downto 0
          do (setf (aref layers m)
                   (loop for (p . q) in (aref layers (1+ m))
                         collect (let ((pp (make-array (* k (expt 2 m))))
                                       (qq (make-array (* k (expt 2 m)))))
                                   (dotimes (b k)
                                     (dotimes (x (expt 2 m))
                                       (let* ((c0 (+ (* b (expt 2 (1+ m))) x))
                                              (c1 (+ c0 (expt 2 m)))
                                              (o (+ (* b (expt 2 m)) x)))
                                         (setf (aref pp o) (ext+ (ext* (aref p c0) (aref q c1))
                                                                 (ext* (aref p c1) (aref q c0)))
                                               (aref qq o) (ext* (aref q c0) (aref q c1))))))
                                   (cons pp qq)))))
    (let ((top (make-array (1+ lk))))
      ;; the per-block roots, padded with 0/1
      (setf (aref top lk)
            (loop for (p . q) in (aref layers 0)
                  collect (cons (concatenate 'vector p (loop repeat (- (expt 2 lk) k) collect (make-ext d)))
                                (concatenate 'vector q (loop repeat (- (expt 2 lk) k) collect (ext-one d))))))
      (loop for j from (1- lk) downto 0
            do (setf (aref top j)
                     (loop for (p . q) in (aref top (1+ j))
                           collect (let ((pp (make-array (expt 2 j)))
                                         (qq (make-array (expt 2 j))))
                                     (dotimes (b (expt 2 j))
                                       (let ((c0 b) (c1 (+ b (expt 2 j))))
                                         (setf (aref pp b) (ext+ (ext* (aref p c0) (aref q c1))
                                                                 (ext* (aref p c1) (aref q c0)))
                                               (aref qq b) (ext* (aref q c0) (aref q c1)))))
                                     (cons pp qq)))))
      (values layers top))))

(defun halves-per-block (vector k size)
  "VECTOR's K blocks of 2 SIZE entries split into their lower and upper halves (each K blocks of
SIZE entries): the children (b, x) and (b, x + SIZE)."
  (let ((lower (make-array (* k size)))
        (upper (make-array (* k size))))
    (dotimes (b k (values lower upper))
      (replace lower vector :start1 (* b size) :start2 (* b 2 size) :end2 (+ (* b 2 size) size))
      (replace upper vector :start1 (* b size) :start2 (+ (* b 2 size) size) :end2 (* (1+ b) 2 size)))))

(defun layer-tables (layer tables k size)
  "The statement's tables (p0, p1, q0, q1 per lookup) for proving the layer above LAYER."
  (loop for (p . q) in layer
        for (p0 p1 q0 q1) on tables by (lambda (l) (nthcdr 4 l))
        append (multiple-value-bind (lo hi) (halves-per-block p k size)
                 (list (cons p0 lo) (cons p1 hi)))
        append (multiple-value-bind (lo hi) (halves-per-block q k size)
                 (list (cons q0 lo) (cons q1 hi)))))

(defun send-finals (tr finals tables d)
  "Send each table's final as a message (1, D), in the proof's order; return the fresh point."
  (dolist (name tables)
    (send tr (make-array (list 1 d) :element-type '(unsigned-byte 32)
                                    :initial-contents (list (coerce (cdr (assoc name finals)) 'list)))))
  (challenge tr))

(defun prove-upper (tr n k leaves &key (label "gkr"))
  "Prove the upper layers of the fraction trees of LEAVES (see FRACTION-TREES): send the roots and
prove every layer down to the leaves' parents (zk/gkr.py prove_upper_host).  Returns the in-block
point, the block point and the claims p and q on the leaf layer for each lookup."
  (let* ((d (transcript-degree tr))
         (lookups (length leaves))
         (lk (block-bits k))
         (r '()))
    (multiple-value-bind (layers top) (fraction-trees leaves n k)
      (multiple-value-bind (statement tables) (layer-statement lookups)
        (dolist (root (aref top 0))
          (send tr (make-array (list 2 d) :element-type '(unsigned-byte 32)
                                          :initial-contents (list (coerce (aref (car root) 0) 'list)
                                                                  (coerce (aref (cdr root) 0) 'list)))))
        (let ((cp (mapcar (lambda (root) (aref (car root) 0)) (aref top 0)))
              (cq (mapcar (lambda (root) (aref (cdr root) 0)) (aref top 0))))
          ;; the top tree
          (dotimes (j lk)
            (multiple-value-bind (constants as lam) (layer-challenges tr lookups)
              (declare (ignore as lam))
              (multiple-value-bind (rho finals)
                  (prove tr j statement (layer-tables (aref top (1+ j)) tables 1 (expt 2 j))
                         :eq (and (plusp j) r) :constants constants :label (label+ label "-top")
                         :degree (if (plusp j) 2 0))
                (let ((tt (send-finals tr finals tables d)))
                  (setf (values cp cq) (next-claims finals lookups tt)
                        r (append rho (list tt)))))))
          (let ((rb r)
                (ri '()))
            (unless (= (expt 2 lk) k)
              (let ((mass (padded-mass rb k d)))
                (setf cq (mapcar (lambda (q) (ext- q mass)) cq))))
            ;; the in-block layers
            (dotimes (layer (1- n))
              (multiple-value-bind (constants as lam) (layer-challenges tr lookups)
                (declare (ignore as lam))
                (let ((bw (block-weights rb k d)))
                  (multiple-value-bind (rho finals folded)
                      (prove tr layer statement (layer-tables (aref layers (1+ layer)) tables k (expt 2 layer))
                             :eq (and (plusp layer) ri) :constants constants :label label
                             :block-weights bw :degree (if (plusp layer) 2 0))
                    (let ((rbn '()))
                      (when (plusp lk)
                        ;; bind the block index: the blocks' values and weights, padded with zeros
                        (flet ((padded (values)
                                 (concatenate 'vector values
                                              (loop repeat (- (expt 2 lk) k) collect (make-ext d)))))
                          (setf (values rbn finals)
                                (prove tr lk `(* bw ,statement)
                                       (cons (cons 'bw (padded bw))
                                             (loop for (name . values) in folded
                                                   collect (cons name (padded values))))
                                       :constants constants :label (label+ label "-blk")))))
                      (let ((tt (send-finals tr finals tables d)))
                        (setf (values cp cq) (next-claims finals lookups tt)
                              ri (append rho (list tt))
                              rb rbn)))))))
            (values ri rb cp cq)))))))
