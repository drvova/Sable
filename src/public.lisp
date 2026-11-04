;;;; public.lisp -- closed forms of the multilinear extensions of structured public vectors
;;;; (zk/mlepub.py), so the verifier evaluates them in O(bits) instead of folding the vector:
;;;;
;;;;   iota(i) = i, or (i >> s) << s           the sum of 2^j x_j over the kept bits
;;;;   [c <= r] on (c low bits, r high bits)    a comparator from the top bit
;;;;   [v < y], [v == y] for constants y        a comparator, or eq against y's bits
;;;;   vocabulary indicators of the targets     the above per row, weighted by eq over the rows
;;;;
;;;; A point is a list of EXTs, coordinate j the bit j of the index.

(defpackage #:exacta.public
  (:use #:cl #:exacta.field)
  (:export #:iota #:le-low-high #:lt-const #:eq-const #:vocab-indicator))

(in-package #:exacta.public)

(defun d-of (point) (if point (length (first point)) 4))

(defun iota (point &optional (shift 0))
  "MLE of i -> (i >> SHIFT) << SHIFT."
  (let ((acc (make-ext (d-of point))))
    (loop for x in (nthcdr shift point)
          for j from shift
          do (setf acc (ext+ acc (ext-scale x (fpow 2 j)))))
    acc))

(defun compare (xs ys)
  "The MLEs of x < y and x == y, bit by bit from the most significant (bit j at index j)."
  (let* ((d (length (first xs)))
         (one (ext-one d))
         (eq one)
         (lt (make-ext d)))
    (loop for a in (reverse xs)
          for b in (reverse ys)
          do (let ((ab (ext* a b)))
               (setf lt (ext+ lt (ext* eq (ext- b ab)))
                     eq (ext* eq (ext+ (ext- (ext- (ext-scale ab 2) a) b) one)))))
    (values lt eq)))

(defun le-low-high (point k)
  "MLE of index -> [c <= r], c the low K bits and r the next K."
  (multiple-value-bind (lt eq) (compare (subseq point 0 k) (subseq point k (* 2 k)))
    (ext+ lt eq)))

(defun const-bits (y n d) (loop for j below n collect (base->ext (ldb (byte 1 j) y) d)))

(defun lt-const (xs y)
  "MLE of v -> [v < Y] over the bits XS (Y may lie outside [0, 2^n])."
  (let ((n (length xs)) (d (d-of xs)))
    (cond ((<= y 0) (make-ext d))
          ((>= y (expt 2 n)) (ext-one d))
          (t (values (compare xs (const-bits y n d)))))))

(defun eq-const (xs y)
  "MLE of v -> [v == Y] over the bits XS."
  (let ((n (length xs)) (d (d-of xs)))
    (if (or (minusp y) (>= y (expt 2 n)))
        (make-ext d)
        (reduce #'ext* (loop for x in xs for j from 0
                             collect (if (logbitp j y) x (ext- (ext-one d) x)))
                :initial-value (ext-one d)))))

(defun vocab-indicator (point ys cb kv kind)
  "The MLEs of the KV blocks of rm(X, CB), X (R, V) the row indicators of the targets YS:
KIND :LT, X[r, v] = [v < y_r]; :EQ, X[r, v] = [v == y_r].  Block k is indexed by (v's low CB bits,
r): v = k 2^cb + v_lo.  Returns a vector of KV EXTs."
  (let* ((d (d-of point))
         (lo (subseq point 0 cb))
         (er (eq-table (nthcdr cb point) d))
         (out (make-array kv :initial-element (make-ext d))))
    (assert (= (length er) (length ys)))
    (ecase kind
      (:eq (loop for y in ys for r from 0
                 do (let ((k (ash y (- cb))))
                      (when (< k kv)
                        (setf (aref out k) (ext+ (aref out k) (ext* (aref er r) (eq-const lo (ldb (byte cb 0) y)))))))))
      (:lt (let ((above (make-array (1+ kv) :initial-element (make-ext d))))
             ;; rows whose target block lies above k count fully in block k
             (loop for y in ys for r from 0
                   do (let ((k (min (ash y (- cb)) kv)))
                        (setf (aref above k) (ext+ (aref above k) (aref er r)))
                        (when (< k kv)
                          (setf (aref out k) (ext+ (aref out k) (ext* (aref er r) (lt-const lo (ldb (byte cb 0) y))))))))
             (let ((run (make-ext d)))
               (loop for k from (1- kv) downto 0
                     do (setf run (ext+ run (aref above (1+ k)))
                              (aref out k) (ext+ (aref out k) run)))))))
    out))
