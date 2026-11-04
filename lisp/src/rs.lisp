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
;;;; TWIDDLES returns the levels 0..n-2: level l holds the x of each position pair of the level-l
;;;; codeword (2^(n-l-1) values), and tw[l+1][s] = tw[l][2s]^2.

(defpackage #:sable.rs
  (:use #:cl #:sable.conditions #:sable.field)
  (:export #:+generator+ #:+two-adicity+ #:root-of-unity #:bitrev #:powers #:twiddles
           #:domain-points #:encode #:encode-line #:brute-eval #:fold))

(in-package #:sable.rs)

(defconstant +generator+ 3 "A generator of F_p^*; p - 1 = 2^24 * 127.")
(defconstant +two-adicity+ 24)

(defun root-of-unity (k)
  "A primitive 2^K-th root of unity; K <= 24, else INVALID-PARAMETER (USE-VALUE)."
  (check-parameter k (lambda (k) (typep k `(integer 0 ,+two-adicity+)))
                   (format nil "an integer in [0, ~D]" +two-adicity+))
  (fpow +generator+ (ash (1- +p+) (- k))))

(defun bitrev (i bits)
  "I with its low BITS bits reversed."
  (loop for b below bits
        sum (ash (ldb (byte 1 b) i) (- bits 1 b))))

(defun powers (w n)
  "w^0 .. w^(n-1)."
  (coerce (loop repeat n for x = 1 then (fmul x w) collect x) 'vector))

(defun bitrev-order (v)
  "V permuted into bit-reversed order (its length is a power of two)."
  (let ((bits (1- (integer-length (length v)))))
    (coerce (loop for i below (length v) collect (aref v (bitrev i bits))) 'vector)))

(defun square-evens (level)
  "The squares of the twiddles at even positions: the next level."
  (coerce (loop for i from 0 below (length level) by 2
                collect (fmul (aref level i) (aref level i)))
          'vector))

(defun twiddles (n)
  (coerce (loop repeat (max 1 (1- n))
                for level = (bitrev-order (powers (root-of-unity n) (ash 1 (1- n))))
                  then (square-evens level)
                collect level)
          'vector))

(defun domain-points (n)
  (bitrev-order (powers (root-of-unity n) (ash 1 n))))

(defun fft (x tw level size-bits)
  "The butterflies of zk/rs.py _fft, in place on X (2^SIZE-BITS values), on twiddle levels from
LEVEL: for b = size-bits - 2 .. 0, each block of 2^(b+1) values gets (a, c) -> (a + t c, a - t c)
with t the block's twiddle on level LEVEL + b."
  (loop for b from (- size-bits 2) downto 0
        for half = (ash 1 b)
        do (loop for tg across (aref tw (+ level b))
                 for base from 0 by (* 2 half)
                 do (loop for i from base below (+ base half)
                          for a = (aref x i)
                          for tc = (fmul tg (aref x (+ i half)))
                          do (setf (aref x i) (fadd a tc)
                                   (aref x (+ i half)) (fsub a tc)))))
  x)

(defun encode-line (message n level tw)
  "Encode MESSAGE (2^(n-level-1) base elements) on level LEVEL of the 2^n tables."
  (let ((size-bits (- n level)))
    (check-parameter message (lambda (m) (= (length m) (ash 1 (1- size-bits))))
                     (format nil "~D values" (ash 1 (1- size-bits))))
    (let ((m (map 'vector #'modp message)))
      (fft (concatenate 'vector m m) tw level size-bits))))

(defun encode (message n &optional (tw (twiddles n)))
  "The codeword of MESSAGE (2^(n-1) base elements) on H_n, in bit-reversed order."
  (encode-line message n 0 tw))

(defun brute-eval (message n)
  "f on the points of H_n by Horner's rule: a slow reference for ENCODE."
  (map 'vector (lambda (x) (reduce (lambda (c acc) (fadd (fmul acc x) (modp c))) message
                                   :from-end t :initial-value 0))
       (domain-points n)))

(defun fold (codeword twiddle-level r)
  "Fold CODEWORD (2m EXTs) with the challenge R on TWIDDLE-LEVEL (m twiddles)."
  (let ((inverse-2 (finv 2)))
    (coerce (loop for (a b) on (coerce codeword 'list) by #'cddr
                  for x across twiddle-level
                  collect (let ((f0 (ext-scale (ext+ a b) inverse-2))
                                (f1 (ext-scale (ext- a b) (finv (fmul 2 x)))))
                            (ext+ f0 (ext* (ext- f1 f0) r))))
            'vector)))
