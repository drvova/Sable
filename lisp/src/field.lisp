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
;;;; each level is a pair of elements of the level below.
;;;;
;;;; Multiplication and inversion are written once, as functions on symbolic coordinates: MUL*,
;;;; the lazily reduced schoolbook product over F_p[w]/(w^D - 3), and INV*, recursive over the
;;;; levels.  TOWER-CODE runs them on the coordinates of the arguments and returns the resulting
;;;; LAMBDA form, straight-line fixnum code for one degree; TOWER-FUNCTION compiles it on first
;;;; use.  So any power-of-two degree works, and the code behind EXT* can be read:
;;;; (tower-code 'mul* 4).  The operations write into an optional OUT vector, fresh by default,
;;;; and never modify their other arguments.

(defpackage #:sable.field
  (:use #:cl #:sable.conditions)
  (:export #:+p+ #:+nonres+ #:fp #:modp #:fadd #:fsub #:fneg #:fmul #:fpow #:finv #:to-signed
           #:ext #:degree #:check-degree #:make-ext #:coerce-ext #:ext-one #:base->ext #:embed
           #:ext= #:ext+ #:ext- #:ext-neg #:ext* #:ext-scale #:ext-inv #:ext-expt
           #:mul* #:inv* #:tower-code #:tower-function
           #:ext-eq1 #:ext-eq #:ext-dot #:lagrange #:eq-table #:fold-table #:mle-eval #:check-tower))

(in-package #:sable.field)

(defconstant +p+ (1+ (- (expt 2 31) (expt 2 24))) "The KoalaBear prime 2^31 - 2^24 + 1.")
(defconstant +nonres+ 3 "The non-residue of the tower: F_p2 = F_p[x]/(x^2 - 3).")

(deftype fp () `(mod ,+p+))
(deftype ext () '(simple-array (unsigned-byte 32) (*)))
(deftype degree () '(member 4 8))

;;; ------------------------------------------------------------------ F_p

;; Two values below p multiply to less than 2^62, a fixnum on 64-bit SBCL, and four such products
;; sum to less than 2^64, an unsigned machine word; with these declarations the arithmetic
;; compiles to machine operations.  Addition and subtraction have no branches: on random data a
;; branch on the carry is mispredicted half the time.
(declaim (inline modp fadd fsub fneg fmul times-3))

(defun modp (x)
  "The residue of the integer X in [0, p)."
  (mod x +p+))

(defun fadd (a b)
  "A + B: subtract p, then add it back if that went below zero (the sign as a mask)."
  (declare (type fp a b) (optimize speed))
  (let ((s (- (+ a b) +p+)))
    (sb-ext:truly-the fp (+ s (logand +p+ (ash s -63))))))

(defun fsub (a b)
  "A - B: add p back if the difference is negative (the sign as a mask)."
  (declare (type fp a b) (optimize speed))
  (let ((d (- a b)))
    (sb-ext:truly-the fp (+ d (logand +p+ (ash d -63))))))

(defun fneg (a)
  (fsub 0 a))

(defun times-3 (a)
  "3A, the non-residue times A, by additions."
  (let ((a2 (fadd a a)))
    (fadd a2 a)))

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

(declaim (inline make-ext))
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
;;;
;;; Each operation writes its result into OUT, a fresh EXT by default.  OUT may be one of the
;;; arguments: every coordinate is read before any is written.

(defun ext= (a b) (equalp a b))

(defun ext+ (a b &optional (out (make-ext (length a))))
  "A + B."
  (declare (type ext a b out) (optimize speed))
  (dotimes (i (length a) out)
    (setf (aref out i) (fadd (aref a i) (aref b i)))))

(defun ext- (a b &optional (out (make-ext (length a))))
  "A - B."
  (declare (type ext a b out) (optimize speed))
  (dotimes (i (length a) out)
    (setf (aref out i) (fsub (aref a i) (aref b i)))))

(defun ext-neg (a &optional (out (make-ext (length a))))
  "-A."
  (declare (type ext a out) (optimize speed))
  (dotimes (i (length a) out)
    (setf (aref out i) (fneg (aref a i)))))

(defun ext-scale (a s &optional (out (make-ext (length a))))
  "A times the base element S."
  (declare (type ext a out) (optimize speed))
  (let ((s (if (typep s 'fp) s (modp s))))
    (declare (type fp s))
    (dotimes (i (length a) out)
      (setf (aref out i) (fmul (aref a i) s)))))

;;; The tower, on symbolic coordinates: an element is a list of forms, one per coordinate, in the
;;; tower order.  Each level is a pair (A0, A1) = A0 + A1 w of elements of the level below, where
;;; w is the level's generator (x in F_p2, y in F_p4, z in F_p8) and w^2 the generator of the
;;; level below (3 in F_p).  So the tower of degree D is also F_p[w]/(w^D - 3) for its top
;;; generator w, whose powers are the coordinates in bit-reversed order (POWER-ORDER).
;;; Intermediate values are bound once, as they are produced (BIND).

(defvar *bindings* '()
  "The bindings (variable form) of the symbolic computation in progress, most recent first.")

(defun bind (coordinates)
  "Bind each coordinate form to a fresh variable; return the variables."
  (loop for form in coordinates
        collect (let ((variable (gensym "C")))
                  (push (list variable form) *bindings*)
                  variable)))

(defun bound (coordinates)
  "COORDINATES, bound to variables unless they are variables already."
  (if (every #'symbolp coordinates) coordinates (bind coordinates)))

(defun split (a)
  "The constant term and the top coefficient of A, two elements of the level below."
  (let ((h (floor (length a) 2)))
    (values (subseq a 0 h) (subseq a h))))

(defun sub* (a b) (mapcar (lambda (x y) `(fsub ,x ,y)) a b))
(defun neg* (a) (mapcar (lambda (x) `(fneg ,x)) a))

(defun bit-reverse (i bits)
  (loop for b below bits
        sum (ash (ldb (byte 1 b) i) (- bits 1 b))))

(defun power-order (a)
  "A's coordinates by power of the top generator w (its own inverse).  In F_p8 the coordinate of
x^i y^j z^k is at index i + 2j + 4k, and x^i y^j z^k = z^(4i + 2j + k): the index bit-reversed."
  (let ((bits (1- (integer-length (length a)))))
    (loop for e below (length a)
          collect (nth (bit-reverse e bits) a))))

(defun mul-by-generator* (a)
  "A w, for w the generator of A's level: (A0 + A1 w) w = A1 w^2 + A0 w."
  (if (rest a)
      (multiple-value-bind (a0 a1) (split a)
        (append (mul-by-generator* a1) a0))
      (list `(times-3 ,(first a)))))

(defun reduced-sum (products)
  "The sum of PRODUCTS (forms, each below p^2) mod p, reduced once per four products: four fit
in an unsigned 64-bit word."
  (reduce (lambda (x y) `(fadd ,x ,y))
          (loop for tail on products by (lambda (l) (nthcdr 4 l))
                collect `(mod (+ ,@(subseq tail 0 (min 4 (length tail)))) +p+))))

(defun mul* (a b)
  "A B on symbolic coordinates, as the schoolbook product in F_p[w]/(w^D - 3):
  c_e = sum_{i+j=e} a_i b_j + 3 sum_{i+j=e+D} a_i b_j,
each coefficient one sum of D products with B's wrapping coefficients tripled first, and reduced
lazily (REDUCED-SUM)."
  (let* ((d (length a))
         (a (power-order (bound a)))
         (b (power-order (bound b)))
         ;; 3 b_j, for the j that wrap (j >= 1)
         (b3 (cons nil (bind (mapcar (lambda (x) `(times-3 ,x)) (rest b))))))
    (power-order
     (loop for e below d
           collect (reduced-sum (loop for i below d
                                      for j = (mod (- e i) d)
                                      collect `(* ,(nth i a) ,(nth j (if (> i e) b3 b)))))))))

(defun inv* (a)
  "1/A on symbolic coordinates, for nonzero A:
  (A0 + A1 w)^-1 = (A0 - A1 w) / (A0^2 - A1^2 w^2), with the norm one level down."
  (if (rest a)
      (multiple-value-bind (a0 a1) (split a)
        (let ((norm-inverse (bind (inv* (bind (sub* (mul* a0 a0) (mul-by-generator* (mul* a1 a1))))))))
          (append (mul* a0 norm-inverse) (neg* (mul* a1 norm-inverse)))))
      (list `(finv ,(first a)))))

(defun tower-code (operation degree)
  "The LAMBDA form computing OPERATION (MUL* or INV*, of two or one EXTs) in degree DEGREE into
its last argument, OUT: OPERATION run on the arguments' coordinates, as straight-line code."
  (let* ((*bindings* '())
         (arguments (subseq '(a b) 0 (if (eq operation 'inv*) 1 2)))
         (result (apply operation
                        (mapcar (lambda (argument)
                                  (bind (loop for i below degree collect `(aref ,argument ,i))))
                                arguments)))
         (bindings (reverse *bindings*)))
    `(lambda (,@arguments out)
       (declare (type ext ,@arguments out) (optimize speed)
                (sb-ext:muffle-conditions sb-ext:compiler-note))
       (let* ,bindings
         (declare (type fp ,@(mapcar #'first bindings)))
         (setf ,@(loop for form in result
                       for i from 0
                       append `((aref out ,i) ,form)))
         out))))

(defvar *tower-functions* '()
  "An alist: operation -> a hash table, degree -> the compiled TOWER-CODE.")

(defun tower-function (operation degree)
  "TOWER-CODE of OPERATION in DEGREE, compiled on first use."
  (let ((table (or (cdr (assoc operation *tower-functions*))
                   (cdar (push (cons operation (make-hash-table)) *tower-functions*)))))
    (or (gethash degree table)
        (setf (gethash degree table)
              (compile nil (tower-code operation degree))))))

(defun by-degree (operation)
  "A function of a degree giving OPERATION's compiled tower code, through a vector indexed by the
degree in front of TOWER-FUNCTION (EXT* calls it for every product)."
  (let ((cache (make-array 17 :initial-element nil)))
    (lambda (degree)
      (or (svref cache degree)
          (setf (svref cache degree) (tower-function operation degree))))))

(defparameter *multiplier* (by-degree 'mul*))
(defparameter *inverter* (by-degree 'inv*))

(defun ext* (a b &optional (out (make-ext (length a))))
  "A B."
  (funcall (the function (funcall *multiplier* (length a))) a b out))

(defun ext-inv (a &optional (out (make-ext (length a))))
  "1/A.  The inverse of zero signals FIELD-DIVISION-BY-ZERO, with the restarts USE-VALUE and
RETURN-ZERO."
  (if (some #'plusp a)
      (funcall (the function (funcall *inverter* (length a))) a out)
      (restart-case (error 'field-division-by-zero :operation 'ext-inv :operands (list a))
        (use-value (inverse)
          :report "Supply the inverse to use."
          (replace out (coerce-ext inverse)))
        (return-zero ()
          :report "Return 0, as the Python and C++ code does."
          (fill out 0)))))

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
