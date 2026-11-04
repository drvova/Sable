;;;; field.lisp -- the KoalaBear field and its tower extensions (zk/field.py, zk/fe.py).
;;;;
;;;;   F_p    p = 2^31 - 2^24 + 1
;;;;   F_p2 = F_p[x]/(x^2 - 3)
;;;;   F_p4 = F_p2[y]/(y^2 - x)       (= F_p[Y]/(Y^4 - 3))
;;;;   F_p8 = F_p4[z]/(z^2 - y)       (= F_p[X]/(X^8 - 3))
;;;;
;;;; A base element is an integer in [0, p).  An extension element (an EXT) is a
;;;; (SIMPLE-ARRAY (UNSIGNED-BYTE 32) (D)) for D = 4 or 8, in the tower order of zk/field.py:
;;;; F_p8 = (A, B) with A, B in F_p4, F_p4 = (a, b) with a, b in F_p2, F_p2 = (a0, a1).  F_p4 embeds
;;;; in F_p8 as (A, 0).  Every function returns a fresh EXT and never modifies its arguments.
;;;;
;;;; The product of two values below p is below 2^62, so it is a fixnum on 64-bit SBCL and the
;;;; arithmetic below compiles to machine integer operations.

(defpackage #:sable.field
  (:use #:cl #:sable.conditions)
  (:export #:+p+ #:+nonres+ #:fp #:fp-canon #:fadd #:fsub #:fneg #:fmul #:fpow #:finv #:to-signed
           #:ext #:ext-degree #:make-ext #:ext-zero #:ext-one #:ext-from-base #:ext-copy #:ext-coerce
           #:ext= #:ext+ #:ext- #:ext-neg #:ext* #:ext-scale #:ext-inv #:ext-expt #:ext-mul-y
           #:ext-eq1 #:ext-eq #:ext-dot #:embed #:lagrange #:eq-table #:mle-eval #:check-tower
           #:check-degree))

(in-package #:sable.field)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defconstant +p+ (+ (- (expt 2 31) (expt 2 24)) 1) "The KoalaBear prime 2^31 - 2^24 + 1.")
  (defconstant +nonres+ 3 "The non-residue of the tower: F_p2 = F_p[x]/(x^2 - 3)."))

(deftype fp () `(integer 0 (,+p+)))
(deftype ext () '(simple-array (unsigned-byte 32) (*)))

(declaim (optimize (speed 3) (safety 1) (debug 1)))

;;; ------------------------------------------------------------------ F_p

(declaim (inline fp-canon fadd fsub fneg fmul))

(defun fp-canon (x)
  "Any integer -> its residue in [0, p)."
  (declare (type integer x))
  (mod x +p+))

(defun fadd (a b)
  (declare (type fp a b))
  (let ((s (+ a b)))
    (if (>= s +p+) (- s +p+) s)))

(defun fsub (a b)
  (declare (type fp a b))
  (if (>= a b) (- a b) (+ (- a b) +p+)))

(defun fneg (a)
  (declare (type fp a))
  (if (zerop a) 0 (- +p+ a)))

(defun fmul (a b)
  (declare (type fp a b))
  (mod (* a b) +p+))

(defun fpow (a e)
  "A^E mod p for an integer A and E >= 0."
  (declare (type (integer 0) e))
  (let ((r 1) (b (fp-canon a)))
    (declare (type fp r b))
    (loop while (plusp e)
          do (when (oddp e) (setf r (fmul r b)))
             (setf b (fmul b b) e (ash e -1)))
    r))

(defun finv (a)
  "The inverse of A in F_p.  On zero, signals FIELD-DIVISION-BY-ZERO with the restarts USE-VALUE
and RETURN-ZERO (the convention of zk/fe.py and zk/cuda/vcore.cpp, where 0^(p-2) = 0)."
  (let ((a (fp-canon a)))
    (if (zerop a)
        (restart-case (error 'field-division-by-zero :operation 'finv :operands (list a))
          (use-value (v)
            :report "Supply the value to use as the inverse."
            (fp-canon v))
          (return-zero ()
            :report "Return 0, as the Python and C++ implementations do."
            0))
        (fpow a (- +p+ 2)))))

(defun to-signed (x)
  "The representative of X in (-p/2, p/2]."
  (let ((x (fp-canon x)))
    (if (> x (floor +p+ 2)) (- x +p+) x)))

;;; ------------------------------------------------------------------ construction

(declaim (inline ext-degree))
(defun ext-degree (a)
  (declare (type ext a))
  (length a))

(defun check-degree (d)
  "D if it is an extension degree of the tower (4 or 8); else INVALID-PARAMETER (USE-VALUE)."
  (require-parameter d (lambda (d) (member d '(4 8))) 'extension-degree "4 or 8"))

(defun make-ext (d &optional (initial 0))
  (make-array d :element-type '(unsigned-byte 32) :initial-element initial))

(defun ext-zero (d) (make-ext d))

(defun ext-one (d)
  (let ((r (make-ext d)))
    (setf (aref r 0) 1)
    r))

(defun ext-from-base (d x)
  (let ((r (make-ext d)))
    (setf (aref r 0) (fp-canon x))
    r))

(defun ext-copy (a)
  (declare (type ext a))
  (copy-seq a))

(defun ext-coerce (seq &optional d)
  "A sequence of integers (any representatives) -> an EXT; D, if given, must be its length."
  (let* ((n (length seq))
         (r (make-ext n)))
    (when (and d (/= d n))
      (error 'invalid-parameter :name 'ext-length :value n :expected d))
    (map-into r #'fp-canon seq)))

(defun embed (a &optional (d 8))
  "An F_p4 element -> F_p8 (zero upper half); an element already of degree D is returned as is."
  (declare (type ext a))
  (if (= (length a) d)
      a
      (let ((r (make-ext d)))
        (replace r a)
        r)))

;;; ------------------------------------------------------------------ elementwise

(defun ext= (a b)
  (declare (type ext a b))
  (and (= (length a) (length b)) (every #'= a b)))

(macrolet ((elementwise (name op doc)
             `(defun ,name (a b)
                ,doc
                (declare (type ext a b))
                (let* ((d (length a))
                       (r (make-ext d)))
                  (dotimes (i d r)
                    (setf (aref r i) (,op (aref a i) (aref b i))))))))
  (elementwise ext+ fadd "A + B.")
  (elementwise ext- fsub "A - B."))

(defun ext-neg (a)
  (declare (type ext a))
  (map 'ext #'fneg a))

(defun ext-scale (a s)
  "A times the base-field scalar S."
  (declare (type ext a))
  (let ((s (fp-canon s))
        (r (make-ext (length a))))
    (dotimes (i (length a) r)
      (setf (aref r i) (fmul (aref a i) s)))))

;;; ------------------------------------------------------------------ the tower

(declaim (inline f2m))
(defun f2m (a0 a1 b0 b1)
  "(a0 + a1 x)(b0 + b1 x) in F_p2."
  (declare (type fp a0 a1 b0 b1))
  (values (fadd (fmul a0 b0) (fmul +nonres+ (fmul a1 b1)))
          (fadd (fmul a0 b1) (fmul a1 b0))))

(defun f4m (a ao b bo r ro)
  "R[ro..ro+3] = A[ao..ao+3] * B[bo..bo+3] in F_p4 (Karatsuba over F_p2, as zk/fe.py)."
  (declare (type ext a b r) (type (integer 0 4) ao bo ro))
  (let ((a0 (aref a ao)) (a1 (aref a (+ ao 1))) (a2 (aref a (+ ao 2))) (a3 (aref a (+ ao 3)))
        (b0 (aref b bo)) (b1 (aref b (+ bo 1))) (b2 (aref b (+ bo 2))) (b3 (aref b (+ bo 3))))
    (multiple-value-bind (x0 x1) (f2m a0 a1 b0 b1)
      (multiple-value-bind (y0 y1) (f2m a2 a3 b2 b3)
        (multiple-value-bind (m0 m1) (f2m (fadd a0 a2) (fadd a1 a3) (fadd b0 b2) (fadd b1 b3))
          ;; y * x = 3 y1 + y0 x
          (setf (aref r ro) (fadd x0 (fmul +nonres+ y1))
                (aref r (+ ro 1)) (fadd x1 y0)
                (aref r (+ ro 2)) (fsub (fsub m0 x0) y0)
                (aref r (+ ro 3)) (fsub (fsub m1 x1) y1))))))
  r)

(defun f4-mul-y (a ao r ro)
  "R[ro..] = A[ao..] * y in F_p4: (a0, a1, a2, a3) -> (3 a3, a2, a0, a1)."
  (declare (type ext a r) (type (integer 0 4) ao ro))
  (let ((a0 (aref a ao)) (a1 (aref a (+ ao 1))) (a2 (aref a (+ ao 2))) (a3 (aref a (+ ao 3))))
    (setf (aref r ro) (fmul +nonres+ a3)
          (aref r (+ ro 1)) a2
          (aref r (+ ro 2)) a0
          (aref r (+ ro 3)) a1))
  r)

(defun ext-mul-y (a)
  "A * y, for A in F_p4."
  (f4-mul-y a 0 (make-ext 4) 0))

(defun ext* (a b)
  "A * B in F_p4 or F_p8."
  (declare (type ext a b))
  (let ((d (length a)))
    (unless (= d (length b))
      (error 'invalid-parameter :name 'operand-degree :value (length b) :expected d))
    (case d
      (4 (f4m a 0 b 0 (make-ext 4) 0))
      (8 (let ((x (make-ext 4)) (y (make-ext 4)) (m (make-ext 4)) (yu (make-ext 4))
               (s0 (make-ext 4)) (s1 (make-ext 4)) (r (make-ext 8)))
           (declare (dynamic-extent x y m yu s0 s1))
           (dotimes (i 4)
             (setf (aref s0 i) (fadd (aref a i) (aref a (+ i 4)))
                   (aref s1 i) (fadd (aref b i) (aref b (+ i 4)))))
           (f4m a 0 b 0 x 0)
           (f4m a 4 b 4 y 0)
           (f4m s0 0 s1 0 m 0)
           (f4-mul-y y 0 yu 0)
           (dotimes (i 4 r)
             (setf (aref r i) (fadd (aref x i) (aref yu i))
                   (aref r (+ i 4)) (fsub (fsub (aref m i) (aref x i)) (aref y i))))))
      (t (check-degree d)))))

(defun f2inv (a0 a1)
  (declare (type fp a0 a1))
  (let ((n (finv (fsub (fmul a0 a0) (fmul +nonres+ (fmul a1 a1))))))
    (values (fmul a0 n) (fmul (fneg a1) n))))

(defun f4inv (a ao r ro)
  "R[ro..] = 1 / A[ao..] in F_p4: (A0 + A1 y)^-1 = (A0 - A1 y) / (A0^2 - x A1^2)."
  (declare (type ext a r) (type (integer 0 4) ao ro))
  (let ((a0 (aref a ao)) (a1 (aref a (+ ao 1))) (a2 (aref a (+ ao 2))) (a3 (aref a (+ ao 3))))
    (multiple-value-bind (s0 s1) (f2m a0 a1 a0 a1)
      (multiple-value-bind (t0 t1) (f2m a2 a3 a2 a3)
        (multiple-value-bind (i0 i1) (f2inv (fsub s0 (fmul +nonres+ t1)) (fsub s1 t0))
          (multiple-value-bind (c0 c1) (f2m a0 a1 i0 i1)
            (multiple-value-bind (d0 d1) (f2m a2 a3 i0 i1)
              (setf (aref r ro) c0
                    (aref r (+ ro 1)) c1
                    (aref r (+ ro 2)) (fneg d0)
                    (aref r (+ ro 3)) (fneg d1))))))))
  r)

(defun %ext-inv (a)
  (declare (type ext a))
  (case (length a)
    (4 (f4inv a 0 (make-ext 4) 0))
    (8 (let ((aa (make-ext 4)) (bb (make-ext 4)) (bu (make-ext 4)) (n (make-ext 4)) (ni (make-ext 4))
             (r (make-ext 8)))
         (f4m a 0 a 0 aa 0)
         (f4m a 4 a 4 bb 0)
         (f4-mul-y bb 0 bu 0)
         (dotimes (i 4) (setf (aref n i) (fsub (aref aa i) (aref bu i))))
         (f4inv n 0 ni 0)
         (f4m a 0 ni 0 r 0)
         (f4m a 4 ni 0 r 4)
         (loop for i from 4 below 8 do (setf (aref r i) (fneg (aref r i))))
         r))
    (t (check-degree (length a)))))

(defun ext-inv (a)
  "1 / A.  On zero, signals FIELD-DIVISION-BY-ZERO with the restarts USE-VALUE and RETURN-ZERO."
  (declare (type ext a))
  (if (every #'zerop a)
      (restart-case (error 'field-division-by-zero :operation 'ext-inv :operands (list a))
        (use-value (v)
          :report "Supply the value to use as the inverse."
          (ext-coerce v (length a)))
        (return-zero ()
          :report "Return 0, as the Python and C++ implementations do."
          (make-ext (length a))))
      (%ext-inv a)))

(defun ext-expt (a e)
  (declare (type ext a) (type (integer 0) e))
  (let ((r (ext-one (length a))) (b a))
    (loop while (plusp e)
          do (when (oddp e) (setf r (ext* r b)))
             (setf b (ext* b b) e (ash e -1)))
    r))

;;; ------------------------------------------------------------------ multilinear helpers

(defun ext-eq1 (a b)
  "eq(a, b) = a b + (1 - a)(1 - b) = 2ab - a - b + 1."
  (let* ((ab (ext* a b))
         (r (make-ext (length a))))
    (dotimes (i (length a))
      (setf (aref r i) (fsub (fsub (fadd (aref ab i) (aref ab i)) (aref a i)) (aref b i))))
    (setf (aref r 0) (fadd (aref r 0) 1))
    r))

(defun ext-eq (as bs d)
  "prod_k eq(a_k, b_k) over two sequences of EXTs of degree D."
  (let ((r (ext-one d)))
    (map nil (lambda (a b) (setf r (ext* r (ext-eq1 a b)))) as bs)
    r))

(defun ext-dot (as bs d)
  (let ((r (ext-zero d)))
    (map nil (lambda (a b) (setf r (ext+ r (ext* a b)))) as bs)
    r))

(defvar *lagrange-weights* (make-hash-table)
  "d -> the barycentric weights 1 / prod_{j /= i} (i - j) of the nodes 0..d.")

(defun lagrange-weights (d)
  (or (gethash d *lagrange-weights*)
      (setf (gethash d *lagrange-weights*)
            (coerce (loop for i from 0 to d
                          collect (finv (let ((den 1))
                                          (loop for j from 0 to d
                                                unless (= i j) do (setf den (fmul den (fp-canon (- i j)))))
                                          den)))
                    'simple-vector))))

(defun lagrange (vals x)
  "The polynomial of degree <= d through VALS at 0..d (a sequence of EXTs), evaluated at X."
  (let* ((vals (coerce vals 'simple-vector))
         (d (1- (length vals)))
         (deg (length x))
         (w (lagrange-weights d))
         (xm (make-array (1+ d)))
         (pre (make-array (1+ d)))
         (res (ext-zero deg))
         (suf (ext-one deg)))
    (dotimes (j (1+ d))
      (let ((v (ext-copy x)))
        (setf (aref v 0) (fsub (aref x 0) (fp-canon j))
              (aref xm j) v)))
    (setf (aref pre 0) (ext-one deg))
    (dotimes (j d)
      (setf (aref pre (1+ j)) (ext* (aref pre j) (aref xm j))))
    (loop for i from d downto 0
          do (setf res (ext+ res (ext-scale (ext* (ext* (aref pre i) suf) (aref vals i)) (aref w i)))
                   suf (ext* suf (aref xm i))))
    res))

(defun eq-table (r d)
  "eq(r, x) for x in {0,1}^k as a vector of 2^k EXTs; x's bit 0 (LSB) pairs with r_0."
  (let* ((k (length r))
         (v (make-array (ash 1 k))))
    (setf (aref v 0) (ext-one d))
    (dotimes (b k v)
      (let ((rb (elt r b))
            (h (ash 1 b)))
        (dotimes (x h)
          (let ((tt (ext* (aref v x) rb)))
            (setf (aref v (+ x h)) tt
                  (aref v x) (ext- (aref v x) tt))))))))

(defun mle-eval (vals r d)
  "The multilinear extension of VALS (2^k base integers or EXTs) at R (k EXTs), LSB first."
  (let* ((k (length r))
         (n (length vals))
         (cur (map 'simple-vector (lambda (v) (if (integerp v) (ext-from-base d v) v)) vals)))
    (unless (= n (ash 1 k))
      (error 'invalid-parameter :name 'mle-length :value n :expected (format nil "2^~D" k)))
    (dotimes (j k (aref cur 0))
      (let* ((rj (elt r j))
             (m (ash (length cur) -1))
             (nxt (make-array m)))
        (dotimes (i m)
          (let ((lo (aref cur (* 2 i))) (hi (aref cur (1+ (* 2 i)))))
            (setf (aref nxt i) (ext+ lo (ext* (ext- hi lo) rj)))))
        (setf cur nxt)))))

(defun check-tower ()
  "The irreducibility facts behind the tower (zk/field.py check_tower); signals on failure."
  (assert (and (= 1 (mod +p+ 4)) (= (1- +p+) (* (expt 2 24) 127))))
  ;; 3 is a non-square, so X^(2^k) - 3 is irreducible for every k (p = 1 mod 4)
  (assert (= (fpow +nonres+ (floor (1- +p+) 2)) (1- +p+)))
  (let ((z (make-ext 8)))
    (setf (aref z 4) 1)
    (assert (ext= (ext-expt z 8) (ext-from-base 8 3))))
  t)
