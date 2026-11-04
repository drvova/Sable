;;;; field.lisp -- the KoalaBear field and its tower extensions (zk/field.py, zk/fe.py).
;;;;
;;;;   F_p    p = 2^31 - 2^24 + 1
;;;;   F_p2 = F_p[x]/(x^2 - 3)
;;;;   F_p4 = F_p2[y]/(y^2 - x)       (= F_p[Y]/(Y^4 - 3))
;;;;   F_p8 = F_p4[z]/(z^2 - y)       (= F_p[X]/(X^8 - 3))
;;;;
;;;; A base element is an integer in [0, p).  An extension element is an EXT, a vector of D
;;;; coordinates in the tower order of zk/field.py: its first half is the constant term and its
;;;; second half the coefficient of the top generator (z for F_p8, y for F_p4, x for F_p2), so
;;;; each level is a pair of elements of the level below.  Multiplication and inversion are
;;;; written once, recursively over the levels, and expanded by a macro into straight-line code
;;;; for each degree (DEFINE-TOWER-FUNCTION).  The functions return fresh EXTs and never modify
;;;; their arguments.

(defpackage #:sable.field
  (:use #:cl #:sable.conditions)
  (:export #:+p+ #:+nonres+ #:fp #:modp #:fadd #:fsub #:fneg #:fmul #:fpow #:finv #:to-signed
           #:ext #:degree #:check-degree #:make-ext #:coerce-ext #:ext-one #:base->ext #:embed
           #:ext= #:ext+ #:ext- #:ext-neg #:ext* #:ext-scale #:ext-inv #:ext-expt
           #:ext-eq1 #:ext-eq #:ext-dot #:lagrange #:eq-table #:fold-table #:mle-eval #:check-tower))

(in-package #:sable.field)

(defconstant +p+ (1+ (- (expt 2 31) (expt 2 24))) "The KoalaBear prime 2^31 - 2^24 + 1.")
(defconstant +nonres+ 3 "The non-residue of the tower: F_p2 = F_p[x]/(x^2 - 3).")

(deftype fp () `(mod ,+p+))
(deftype ext () '(simple-array (unsigned-byte 32) (*)))
(deftype degree () '(member 4 8))

;;; ------------------------------------------------------------------ F_p

;; Two values below p multiply to less than 2^62, a fixnum on 64-bit SBCL, so with these
;; declarations the arithmetic compiles to machine operations.
(declaim (inline modp fadd fsub fneg fmul))

(defun modp (x)
  "The residue of the integer X in [0, p)."
  (mod x +p+))

(defun fadd (a b)
  (declare (type fp a b) (optimize speed))
  (let ((s (+ a b))) (if (< s +p+) s (- s +p+))))

(defun fsub (a b)
  (declare (type fp a b) (optimize speed))
  (if (>= a b) (- a b) (+ (- a b) +p+)))

(defun fneg (a)
  (declare (type fp a))
  (if (zerop a) 0 (- +p+ a)))

(defun fmul (a b)
  (declare (type fp a b) (optimize speed))
  (mod (* a b) +p+))

(defun fpow (a e)
  "A^E mod p, by square and multiply."
  (declare (type (integer 0) e))
  (loop with result of-type fp = 1
        for base of-type fp = (modp a) then (fmul base base)
        for k = e then (ash k -1)
        while (plusp k)
        when (oddp k) do (setf result (fmul result base))
        finally (return result)))

(defun finv (a)
  "1/A in F_p.  The inverse of zero signals FIELD-DIVISION-BY-ZERO, with the restarts USE-VALUE
and RETURN-ZERO (the convention of zk/fe.py and zk/cuda/vcore.cpp: 0^(p-2) = 0)."
  (let ((a (modp a)))
    (if (plusp a)
        (fpow a (- +p+ 2))
        (restart-case (error 'field-division-by-zero :operation 'finv :operands (list a))
          (use-value (inverse)
            :report "Supply the inverse to use."
            (modp inverse))
          (return-zero ()
            :report "Return 0, as the Python and C++ code does."
            0)))))

(defun to-signed (x)
  "The representative of X in (-p/2, p/2]."
  (let ((x (modp x)))
    (if (> x (floor +p+ 2)) (- x +p+) x)))

;;; ------------------------------------------------------------------ construction

(defun check-degree (d)
  "D, if it is an extension degree of the tower (4 or 8); else INVALID-PARAMETER (USE-VALUE)."
  (check-parameter d (lambda (d) (typep d 'degree)) "4 or 8")
  d)

(defun make-ext (d)
  "The zero of F_p^D."
  (make-array d :element-type '(unsigned-byte 32) :initial-element 0))

(defun coerce-ext (coordinates)
  "An EXT from a sequence of integers (any representatives)."
  (map 'ext #'modp coordinates))

(defun base->ext (x d)
  "The base element X in F_p^D."
  (let ((e (make-ext d)))
    (setf (aref e 0) (modp x))
    e))

(defun ext-one (d) (base->ext 1 d))

(defun embed (a &optional (d 8))
  "A as an element of degree D (zero upper coordinates): F_p4 -> F_p8."
  (if (= (length a) d)
      a
      (replace (make-ext d) a)))

;;; ------------------------------------------------------------------ arithmetic

(defun ext= (a b) (equalp a b))
;; Elementwise operations.
(macrolet ((define-elementwise (name lambda-list function documentation)
             `(defun ,name ,lambda-list
                ,documentation
                (declare (type ext ,@lambda-list) (optimize speed))
                (map 'ext ,function ,@lambda-list))))
  (define-elementwise ext+ (a b) #'fadd "A + B.")
  (define-elementwise ext- (a b) #'fsub "A - B.")
  (define-elementwise ext-neg (a) #'fneg "-A."))

(defun ext-scale (a s)
  "A times the base element S."
  (declare (type ext a))
  (let ((s (modp s)))
    (map 'ext (lambda (x) (fmul x s)) a)))

;;; The tower operations are defined once, recursively, on symbolic coordinates: lists of forms,
;;; one per coordinate.  Each level is a pair (A0, A1) = A0 + A1 w of elements of the level below,
;;; where w is the level's generator (x in F_p2, y in F_p4, z in F_p8) and w^2 is the generator of
;;; the level below (3 in F_p).  DEFINE-TOWER-FUNCTION expands the recursion at compile time into
;;; straight-line fixnum code for one degree, with every intermediate value bound once.

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defvar *bindings* '()
    "The bindings (variable form) of the symbolic computation being expanded, most recent first.")

  (defun bind (coordinates)
    "Bind each coordinate form to a fresh variable; return the variables."
    (loop for form in coordinates
          collect (let ((variable (gensym "C")))
                    (push (list variable form) *bindings*)
                    variable)))

  (defun split (a)
    "The constant term and the top coefficient of A, two elements of the level below."
    (let ((h (floor (length a) 2)))
      (values (subseq a 0 h) (subseq a h))))

  (defun add* (a b) (mapcar (lambda (x y) `(fadd ,x ,y)) a b))
  (defun sub* (a b) (mapcar (lambda (x y) `(fsub ,x ,y)) a b))
  (defun neg* (a) (mapcar (lambda (x) `(fneg ,x)) a))

  (defun mul-by-generator* (a)
    "A w, for w the generator of A's level: (A0 + A1 w) w = A1 w^2 + A0 w."
    (if (rest a)
        (multiple-value-bind (a0 a1) (split a)
          (append (mul-by-generator* a1) a0))
        (list `(fmul +nonres+ ,(first a)))))

  (defun mul* (a b)
    "A B, by Karatsuba:
  (A0 + A1 w)(B0 + B1 w) = (A0 B0 + A1 B1 w^2) + ((A0 + A1)(B0 + B1) - A0 B0 - A1 B1) w."
    (if (rest a)
        (multiple-value-bind (a0 a1) (split a)
          (multiple-value-bind (b0 b1) (split b)
            (let ((lo (bind (mul* a0 b0)))
                  (hi (bind (mul* a1 b1)))
                  (mid (bind (mul* (bind (add* a0 a1)) (bind (add* b0 b1))))))
              (append (add* lo (mul-by-generator* hi))
                      (sub* (sub* mid lo) hi)))))
        (list `(fmul ,(first a) ,(first b)))))

  (defun inv* (a)
    "1/A for nonzero A: (A0 + A1 w)^-1 = (A0 - A1 w) / (A0^2 - A1^2 w^2), the norm one level down."
    (if (rest a)
        (multiple-value-bind (a0 a1) (split a)
          (let ((norm-inverse (bind (inv* (bind (sub* (mul* a0 a0) (mul-by-generator* (mul* a1 a1))))))))
            (append (mul* a0 norm-inverse) (neg* (mul* a1 norm-inverse)))))
        (list `(finv ,(first a))))))

(defmacro define-tower-function (name degree lambda-list symbolic-function)
  "Define NAME, a function of the EXTs of degree DEGREE in LAMBDA-LIST, whose coordinates are
those computed by SYMBOLIC-FUNCTION (MUL* or INV*) on the arguments' coordinates."
  (let* ((*bindings* '())
         (result (apply symbolic-function
                        (mapcar (lambda (argument)
                                  (bind (loop for i below degree collect `(aref ,argument ,i))))
                                lambda-list)))
         (bindings (reverse *bindings*)))
    `(defun ,name ,lambda-list
       (declare (type ext ,@lambda-list) (optimize speed))
       (let* ,bindings
         (declare (type fp ,@(mapcar #'first bindings)))
         (let ((result (make-ext ,degree)))
           (setf ,@(loop for form in result
                         for i from 0
                         append `((aref result ,i) ,form)))
           result)))))

(define-tower-function mul-4 4 (a b) mul*)
(define-tower-function mul-8 8 (a b) mul*)
(define-tower-function inv-4 4 (a) inv*)
(define-tower-function inv-8 8 (a) inv*)

(defun ext* (a b)
  "A B in F_p4 or F_p8."
  (ecase (length a)
    (4 (mul-4 a b))
    (8 (mul-8 a b))))

(defun ext-inv (a)
  "1/A.  The inverse of zero signals FIELD-DIVISION-BY-ZERO, with the restarts USE-VALUE and
RETURN-ZERO."
  (if (some #'plusp a)
      (ecase (length a)
        (4 (inv-4 a))
        (8 (inv-8 a)))
      (restart-case (error 'field-division-by-zero :operation 'ext-inv :operands (list a))
        (use-value (inverse)
          :report "Supply the inverse to use."
          (coerce-ext inverse))
        (return-zero ()
          :report "Return 0, as the Python and C++ code does."
          (make-ext (length a))))))

(defun ext-expt (a e)
  (loop with result = (ext-one (length a))
        for base = a then (ext* base base)
        for k = e then (ash k -1)
        while (plusp k)
        when (oddp k) do (setf result (ext* result base))
        finally (return result)))

;;; ------------------------------------------------------------------ multilinear helpers

(defun ext-eq1 (a b)
  "eq(a, b) = a b + (1 - a)(1 - b)."
  (let ((one (ext-one (length a))))
    (ext+ (ext* a b) (ext* (ext- one a) (ext- one b)))))

(defun ext-eq (as bs d)
  "prod_k eq(a_k, b_k)."
  (reduce #'ext* (map 'list #'ext-eq1 as bs) :initial-value (ext-one d)))

(defun ext-dot (as bs d)
  "sum_k a_k b_k."
  (reduce #'ext+ (map 'list #'ext* as bs) :initial-value (make-ext d)))

(defvar *lagrange-weights* (make-hash-table)
  "d -> the weights 1 / prod_{j /= i} (i - j) of the nodes 0..d.")

(defun lagrange-weight (i d)
  (aref (or (gethash d *lagrange-weights*)
            (setf (gethash d *lagrange-weights*)
                  (coerce (loop for i from 0 to d
                                collect (finv (reduce #'fmul (loop for j from 0 to d
                                                                    unless (= i j) collect (modp (- i j)))
                                                      :initial-value 1)))
                          'vector)))
        i))

(defun lagrange (values x)
  "The polynomial of degree d through VALUES (d + 1 EXTs) at the nodes 0..d, evaluated at X:
  sum_i w_i v_i prod_{j /= i} (x - j)."
  (let* ((d (1- (length values)))
         (x-minus (loop for j from 0 to d collect (ext- x (base->ext j (length x))))))
    (reduce #'ext+
            (loop for v in (coerce values 'list)
                  for i from 0
                  collect (ext-scale (reduce #'ext* (loop for factor in x-minus
                                                          for j from 0
                                                          unless (= i j) collect factor)
                                             :initial-value v)
                                     (lagrange-weight i d))))))

(defun eq-table (r d)
  "eq(r, x) for every x in {0,1}^k, a vector of 2^k EXTs indexed by x (x's bit 0 pairs with r_0).
Each variable doubles the table: entry e becomes e (1 - r_b) in the lower half and e r_b in the
upper half."
  (let ((table (vector (ext-one d))))
    (map nil (lambda (rb)
               (let ((upper (map 'vector (lambda (e) (ext* e rb)) table)))
                 (setf table (concatenate 'vector (map 'vector #'ext- table upper) upper))))
         r)
    table))

(defun fold-table (table r)
  "Bind the lowest variable of the multilinear TABLE (a vector of EXTs) to R: each pair (lo, hi)
becomes lo + r (hi - lo)."
  (coerce (loop for (lo hi) on (coerce table 'list) by #'cddr
                collect (ext+ lo (ext* r (ext- hi lo))))
          'vector))

(defun mle-eval (values r d)
  "The multilinear extension of VALUES (2^k base integers or EXTs) at R (k EXTs, LSB first)."
  (check-parameter values (lambda (v) (= (length v) (expt 2 (length r))))
                   (format nil "~D values" (expt 2 (length r))))
  (let ((table (map 'vector (lambda (v) (if (integerp v) (base->ext v d) v)) values)))
    (aref (reduce #'fold-table r :initial-value table) 0)))

(defun check-tower ()
  "The facts behind the tower (zk/field.py check_tower)."
  (assert (= (1- +p+) (* (expt 2 24) 127)))
  ;; 3 is not a square, so X^(2^k) - 3 is irreducible for every k (as p = 1 mod 4)
  (assert (= (fpow +nonres+ (floor (1- +p+) 2)) (1- +p+)))
  (let ((z (make-ext 8)))
    (setf (aref z 4) 1)
    (assert (ext= (ext-expt z 8) (base->ext 3 8))))
  t)
