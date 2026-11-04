;;;; rs.lisp -- the Reed-Solomon code over KoalaBear, rate 1/2 (zk/rs.py, reference versions).
;;;;
;;;; A message m of length 2^(n-1) is the coefficient vector of f(X) = sum_j m_j X^j.  Its codeword
;;;; is f on the subgroup H_n of order 2^n in bit-reversed order: position t holds
;;;; f(w_n^bitrev_n(t)).  Positions (2s, 2s+1) hold f(x), f(-x) with x = w_n^bitrev_{n-1}(s), so the
;;;; 2^k symbols that fold to one symbol after k folds are contiguous.
;;;;
;;;; Folding with challenge r (MLE-compatible, LSB variable first: the folded message is
;;;; (1 - r) m_even + r m_odd):  f0 = (a + b)/2, f1 = (a - b)/(2x), folded = f0 + r (f1 - f0).
;;;;
;;;; TWIDDLES returns levels l = 0..n-2: level l has 2^(n-l-1) entries, the x of each position
;;;; pair, and tw[l+1][s] = tw[l][2s]^2.

(defpackage #:sable.rs
  (:use #:cl #:sable.conditions #:sable.field)
  (:export #:+generator+ #:+two-adicity+ #:root-of-unity #:bitrev #:bitrev-perm #:powers
           #:twiddles #:domain-points #:encode #:encode-line #:brute-eval #:fold #:+inv2+))

(in-package #:sable.rs)

(defconstant +generator+ 3 "A generator of F_p^*; p - 1 = 2^24 * 127.")
(defconstant +two-adicity+ 24)
(defparameter +inv2+ (finv 2))

(defun root-of-unity (k)
  "A primitive 2^K-th root of unity (K <= 24; else INVALID-PARAMETER with USE-VALUE)."
  (let ((k (require-parameter k (lambda (k) (typep k `(integer 0 ,+two-adicity+)))
                              'two-adic-order (format nil "an integer in [0, ~D]" +two-adicity+))))
    (fpow +generator+ (ash (1- +p+) (- k)))))

(defun bitrev (i bits)
  (let ((r 0))
    (dotimes (b bits r)
      (setf r (logior r (ash (ldb (byte 1 b) i) (- bits 1 b)))))))

(defun bitrev-perm (bits)
  (let ((v (make-array (ash 1 bits))))
    (dotimes (i (length v) v) (setf (aref v i) (bitrev i bits)))))

(defun powers (w n)
  "w^0 .. w^(n-1) as a simple-vector."
  (let ((v (make-array n)) (x 1))
    (dotimes (i n v) (setf (aref v i) x x (fmul x w)))))

(defun permute (v perm)
  (map 'simple-vector (lambda (i) (aref v i)) perm))

(defun twiddles (n)
  (let* ((t0 (permute (powers (root-of-unity n) (ash 1 (1- n))) (bitrev-perm (1- n))))
         (tw (list t0)))
    (loop repeat (- n 2)
          do (let ((prev (first tw)))
               (push (coerce (loop for i from 0 below (length prev) by 2
                                   collect (fmul (aref prev i) (aref prev i)))
                             'simple-vector)
                     tw)))
    (coerce (nreverse tw) 'simple-vector)))

(defun domain-points (n)
  (permute (powers (root-of-unity n) (ash 1 n)) (bitrev-perm n)))

(defun fft (x tw level l)
  "In-place butterflies of zk/rs.py _fft on X (length 2^L), twiddle levels from LEVEL."
  (let ((nn (ash 1 l)))
    (loop for b from (- l 2) downto 0
          do (let* ((half (ash 1 b))
                    (tl (aref tw (+ level b))))
               (loop for g from 0 below (ash nn (- (1+ b)))
                     for base = (* g 2 half)
                     for tg = (aref tl g)
                     do (dotimes (i half)
                          (let* ((a (aref x (+ base i)))
                                 (tc (fmul tg (aref x (+ base half i)))))
                            (setf (aref x (+ base i)) (fadd a tc)
                                  (aref x (+ base half i)) (fsub a tc)))))))
    x))

(defun encode-line (msg n level tw)
  "Encode MSG (length 2^(n-level-1)) on level LEVEL of the 2^n tables."
  (let* ((l (- n level))
         (m (length msg)))
    (unless (= m (ash 1 (1- l)))
      (error 'invalid-parameter :name 'message-length :value m :expected (ash 1 (1- l))))
    (let ((x (make-array (* 2 m))))
      (dotimes (i m)
        (setf (aref x i) (fp-canon (elt msg i))
              (aref x (+ i m)) (fp-canon (elt msg i))))
      (fft x tw level l))))

(defun encode (msg n &optional (tw (twiddles n)))
  "The codeword of MSG (2^(n-1) base elements) on H_n, bit-reversed order."
  (encode-line msg n 0 tw))

(defun brute-eval (msg n)
  "f on the points of H_n by Horner's rule (a slow reference for ENCODE)."
  (map 'simple-vector
       (lambda (x) (let ((acc 0))
                     (loop for i from (1- (length msg)) downto 0
                           do (setf acc (fadd (fmul acc x) (fp-canon (elt msg i)))))
                     acc))
       (domain-points n)))

(defun fold (cw tw-level r)
  "Fold the codeword CW (2m EXTs) with the challenge R (an EXT) on twiddles TW-LEVEL (m entries)."
  (let* ((m (floor (length cw) 2))
         (out (make-array m)))
    (dotimes (s m out)
      (let* ((a (aref cw (* 2 s)))
             (b (aref cw (1+ (* 2 s))))
             (f0 (ext-scale (ext+ a b) +inv2+))
             (f1 (ext-scale (ext- a b) (finv (fmul 2 (aref tw-level s))))))
        (setf (aref out s) (ext+ f0 (ext* (ext- f1 f0) r)))))))
