;;;; exact.lisp -- correctly rounded values of exp, sin and cos (zk/model.py round_exact).
;;;;
;;;; The model's tables hold integers nearest to real values such as 2^10 exp(-x alpha).  They are
;;;; computed in double floats, and an entry whose fractional part lies within TOL of 1/2 is
;;;; recomputed exactly: here in fixed point with +PRECISION+ bits (Sable used mpmath at 60 digits).
;;;; Outside that band every approximation within TOL/2 rounds the same way, so the tables do not
;;;; depend on the platform's floating point.

(defpackage #:exacta.exact
  (:use #:cl #:exacta.conditions)
  (:export #:+precision+ #:fixed #:fixed-exp #:fixed-ln #:fixed-sin #:fixed-cos #:fixed-sqrt
           #:fixed-pi #:fixed-ln2 #:round-fixed #:round-exact))

(in-package #:exacta.exact)

(defconstant +precision+ 320 "Fraction bits of the fixed-point values.")

(defun one () (expt 2 +precision+))

(defun fixed (q)
  "The rational Q in fixed point (rounded down)."
  (floor (* q (one))))

(defun mul (a b) (ash (* a b) (- +precision+)))
(defun div (a b) (floor (ash a +precision+) b))

(defun atanh-series (z)
  "atanh(z) = z + z^3/3 + z^5/5 + ..., for fixed |z| <= 1/3."
  (let ((z2 (mul z z)))
    (loop for k from 1 by 2
          for power = z then (mul power z2)
          for term = (truncate power k)
          until (zerop term)
          sum term)))

(defun atan-inverse (n)
  "atan(1/n) for an integer n > 1."
  (loop for k from 1 by 2
        for power = (floor (one) n) then (floor power (* n n))
        for sign = 1 then (- sign)
        for term = (truncate power k)
        until (zerop term)
        sum (* sign term)))

(let ((ln2 nil) (pi* nil))
  (defun fixed-ln2 () (or ln2 (setf ln2 (* 2 (atanh-series (fixed 1/3))))))
  (defun fixed-pi () (or pi* (setf pi* (- (* 16 (atan-inverse 5)) (* 4 (atan-inverse 239)))))))

(defun exp-series (r)
  "exp(r) for fixed |r| <= ln 2."
  (loop for k from 1
        for term = (one) then (truncate (mul term r) (1- k))
        until (zerop term)
        sum term))

(defun fixed-exp (x)
  "exp(x) for fixed X."
  (let ((k (round x (fixed-ln2))))
    (ash (exp-series (- x (* k (fixed-ln2)))) k)))

(defun fixed-ln (x)
  "ln(x) for fixed X > 0."
  (check-parameter x #'plusp "positive")
  (let* ((e (- (integer-length x) 1 +precision+))
         (m (ash x (- e))))                   ; m in [1, 2)
    (+ (* e (fixed-ln2)) (* 2 (atanh-series (div (- m (one)) (+ m (one))))))))

(defun fixed-sqrt (x)
  "sqrt(x) for fixed X >= 0."
  (isqrt (ash x +precision+)))

(defun reduce-angle (x)
  "X modulo 2 pi, in [-pi, pi]."
  (let ((two-pi (* 2 (fixed-pi))))
    (- x (* two-pi (round x two-pi)))))

(defun fixed-sin (x)
  (let* ((x (reduce-angle x)) (x2 (mul x x)))
    (loop for k from 1 by 2
          for term = x then (- (truncate (mul term x2) (* k (1- k))))
          until (zerop term)
          sum term)))

(defun fixed-cos (x)
  (let* ((x (reduce-angle x)) (x2 (mul x x)))
    (loop for k from 0 by 2
          for term = (one) then (- (truncate (mul term x2) (* k (1- k))))
          until (zerop term)
          sum term)))

(defun round-fixed (v)
  "The integer nearest to the fixed value V (ties to even); a value too close to a tie for the
precision signals an error."
  (multiple-value-bind (n frac) (floor v (one))
    (let ((distance (abs (- (* 2 frac) (one)))))   ; |2 frac - 1|, zero at a tie
      (when (and (plusp distance) (< distance (expt 2 (floor +precision+ 2))))
        (error "round-fixed: ~A is too close to a tie" (/ v (one))))
      (cond ((> (* 2 frac) (one)) (1+ n))
            ((< (* 2 frac) (one)) n)
            ((oddp n) (1+ n))
            (t n)))))

(defun round-exact (approximation exact &optional (tol 1d-4))
  "Sable's round_exact for one entry: APPROXIMATION (a double) rounded to even, unless its
fractional part lies within TOL of 1/2; then (funcall EXACT), a fixed value, rounded."
  (declare (type double-float approximation))
  (if (< (abs (- approximation (ffloor approximation) 0.5d0)) tol)
      (round-fixed (funcall exact))
      (values (round approximation))))
