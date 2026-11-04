;;;; sumcheck.lisp -- the sumcheck protocol (zk/sumcheck.py, zk/cuda/vcore.cpp sc_verify), with a
;;;; reference CPU prover.
;;;;
;;;; The statement is a form (sable.forms) over multilinear tables, and optionally over fixed
;;;; values (CONSTANTS, e.g. challenges):
;;;;
;;;;   sum_{x in {0,1}^n} w(x) F(t0(x), t1(x), ...) = claim        F = '(* t0 t1), '(+ (* a t0) t1) ...
;;;;
;;;; with w(x) = eq(r, x) in eq mode, else 1.  Its degree in the tables fixes the round messages, and
;;;; the same form gives the verifier's final check.  Variables are bound LSB first.  The transcript
;;;; first absorbs  label | n u16 | d u16 | K u32.  Round j sends g_j at X = 0, 2, 3, ..., d, a field
;;;; array of shape (max(d,1), D); g_j(1) follows from the claim.  In eq mode g_j excludes the eq
;;;; factor of the round variable and the reduced claim the whole eq factor, so
;;;; claim_j = (1 - r_j) g_j(0) + r_j g_j(1).

(defpackage #:sable.sumcheck
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.forms #:sable.transcript)
  (:export #:header #:statement-degree #:hypercube-sum #:prove #:verify #:next-claim
           #:check-final #:round-kernel-code #:*kernel-size-limit* #:small-value-code
           #:*small-value-rounds* #:*small-value-size-limit* #:demo))

(in-package #:sable.sumcheck)

(defun header (label n degree &optional (blocks 1))
  "label | n (u16 LE) | degree (u16 LE) | blocks (u32 LE)."
  (concatenate 'octets (to-octets label) (to-octets "|")
               (le-bytes n 2) (le-bytes degree 2) (le-bytes blocks 4)))

(defun table-variables (statement constants)
  "The variables of STATEMENT that are tables: all but the CONSTANTS."
  (set-difference (form-variables statement) (mapcar #'car constants)))

(defun statement-degree (statement &optional constants)
  "The degree of STATEMENT in its tables: the degree of each round polynomial."
  (form-degree statement (table-variables statement constants)))

(defun weights (eq from n d)
  "w over the variables FROM..N-1: eq(r_from.., x), or all ones without EQ."
  (if eq
      (eq-table (subseq (coerce eq 'vector) from n) d)
      (make-array (expt 2 (- n from)) :initial-element (ext-one d))))

(defun split-weights (eq from n d)
  "WEIGHTS over the variables FROM..N-1 as two tables, LOW over the lower half of them and HIGH over
the upper half, so that the weight of x = h 2^|low| + l is LOW[l] HIGH[h] (Dao and Thaler, ePrint
2024/1210): two tables of about the square root of the size of one."
  (let ((middle (+ from (floor (- n from) 2))))
    (values (weights eq from middle d)
            (weights eq middle n d))))

;;; ------------------------------------------------------------------ tables
;;;
;;; A table is stored flat: entry x is ENTRIES[stride x, stride (x + 1)), a base element (stride 1)
;;; or the D coordinates of an extension element (stride D).  A table given as base elements stays
;;; in the base field until its first fold, which then needs only base-times-extension products.

(defstruct (table (:constructor %make-table (stride entries)))
  (stride 1 :type (integer 1 16) :read-only t)
  (entries nil :type ext :read-only t))

(defun make-table (values d)
  "A table of VALUES: a sequence of base elements (integers) or of extension elements of degree D."
  (let ((values (coerce values 'vector)))
    (if (every #'integerp values)
        (%make-table 1 (coerce-ext values))
        (let ((entries (make-ext (* d (length values)))))
          (loop for v across values
                for x from 0
                do (replace entries (if (integerp v) (base->ext v d) v) :start1 (* d x)))
          (%make-table d entries)))))

(defun table-size (table)
  (floor (length (table-entries table)) (table-stride table)))

(defun table-entry (table x out)
  "Entry X of TABLE as an extension element, written into OUT."
  (declare (type ext out) (type (integer 0 #.(floor most-positive-fixnum 16)) x) (optimize speed))
  (let* ((stride (table-stride table))
         (entries (table-entries table))
         (start (* stride x)))
    (dotimes (c (length out) out)
      (setf (aref out c) (if (< c stride) (aref entries (+ start c)) 0)))))

(defun fold-flat-table (table r)
  "TABLE with its lowest variable bound to R: each pair (lo, hi) becomes lo + r (hi - lo).  For a
base table the slope hi - lo is a base element, and r (hi - lo) costs D base products."
  (declare (type ext r) (optimize speed))
  (let* ((d (length r))
         (half (floor (table-size table) 2))
         (entries (table-entries table))
         (base (= (table-stride table) 1))
         (folded (make-ext (* d half)))
         (lo (make-ext d)) (hi (make-ext d)) (term (make-ext d)))
    (dotimes (i half (%make-table d folded))
      (if base
          (let ((lo (aref entries (* 2 i))))
            (ext-scale r (fsub (aref entries (1+ (* 2 i))) lo) term)
            (setf (aref term 0) (fadd (aref term 0) lo)))
          (progn
            (table-entry table (* 2 i) lo)
            (table-entry table (1+ (* 2 i)) hi)
            (ext+ lo (ext* r (ext- hi lo term) term) term)))
      (dotimes (c d)
        (setf (aref folded (+ (* d i) c)) (aref term c))))))

;;; ------------------------------------------------------------------ the statement

(defun statement-function (statement tables constants d)
  "STATEMENT compiled over F_p^D as a function of the tables' values and then the constants'."
  (compile-form statement (append (mapcar #'car tables) (mapcar #'car constants)) :degree d))

(defun hypercube-sum (statement n tables d &key eq constants)
  "sum_x w(x) F(tables(x)): the claim of an honest prover.  TABLES: an alist (variable . 2^N base
elements or EXTs)."
  (let* ((f (statement-function statement tables constants d))
         (flat (loop for (nil . values) in tables collect (make-table values d)))
         (buffers (loop repeat (length flat) collect (make-ext d)))
         (arguments (append buffers (mapcar #'cdr constants)))
         (sum (make-ext d))
         (term (make-ext d)))
    (loop for w across (weights eq 0 n d)
          for x from 0
          do (loop for table in flat
                   for buffer in buffers
                   do (table-entry table x buffer))
             (ext+ sum (ext* w (apply f arguments) term) sum))
    sum))

;;; ------------------------------------------------------------------ verifier

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

(defun verify (tr n statement claim &key eq constants (blocks 1) (label "sc")
                                         (degree (statement-degree statement constants)))
  "Replay the N-round sumcheck of STATEMENT on the verifier transcript TR.  CLAIM is an EXT (or
its coordinates); EQ, in eq mode, a sequence of at least N EXTs; CONSTANTS an alist of the
statement's non-table variables; BLOCKS the number of blocks summed in each round (the header's
K); DEGREE the round polynomials' degree as the header states it (zk/ uses 0 when N is 0).
Returns the round challenges and the reduced claim, for CHECK-FINAL.  A round message of the wrong
shape signals MESSAGE-SHAPE-ERROR."
  (let* ((shape (list (max degree 1) (transcript-degree tr)))
         (claim (coerce-ext claim)))
    (absorb tr (header label n degree blocks))
    (values (loop for j below n
                  collect (let ((message (send tr nil shape))
                                (r (challenge tr)))
                            (setf claim (next-claim claim message (and eq (elt eq j)) r))
                            r))
            claim)))

(defun check-final (reduced statement finals &key constants)
  "Reject the proof unless REDUCED is STATEMENT at FINALS, an alist (table . EXT) of the tables'
values at the round challenges."
  (let ((expected (evaluate statement (append finals constants) :degree (length reduced))))
    (unless (ext= reduced expected)
      (reject :sumcheck-final "reduced claim (~{~D~^ ~}), statement at the final values (~{~D~^ ~})"
              (coerce reduced 'list) (coerce expected 'list))))
  t)

;;; ------------------------------------------------------------------ prover

;;; One round of the prover computes g(X) = sum_x w(x) F(line values at X) at X = 0, 2, 3, ...,
;;; points, over the pairs x of the tables.  For each pair (lo, hi) of a table, the line values at
;;; X = 0, 1, 2, ... are lo, hi, hi + s, hi + 2s, ... for the slope s = hi - lo: additions only.
;;; The weight w(x) = LOW[l] HIGH[h] for x = h |LOW| + l (SPLIT-WEIGHTS): the inner loop sums over l
;;; with the LOW weights, and each block is multiplied by its HIGH weight once per point.
;;; ROUND-KERNEL-CODE writes the whole round for one statement as a LAMBDA form, with the statement
;;; inlined and the tables and points unrolled, every buffer made once per round;
;;; GENERIC-ROUND-SUMS is the same loop calling the compiled statement, for statements whose
;;; kernel would be too large.

(defun evaluation-points (points)
  "X = 0, 2, 3, ..., POINTS: g(1) follows from the claim."
  (cons 0 (loop for x from 2 to points collect x)))

(defun round-kernel-code (statement table-names constant-names d points base)
  "The LAMBDA form of one prover round for STATEMENT: a function of the tables (TABLEs, in the
order of TABLE-NAMES), the LOW and HIGH eq weights (vectors of EXTs, SPLIT-WEIGHTS) and the
constants' values (in the order of CONSTANT-NAMES), returning g's sums at the EVALUATION-POINTS.
With BASE (all tables in the base field, no constants), the line values and the statement are
base-field integers, and each term costs one base-times-extension product by its weight."
  (let* ((xs (evaluation-points points))
         (n-tables (length table-names))
         (tables (gensym "TABLES")) (low (gensym "LOW")) (high (gensym "HIGH"))
         (h (gensym "H")) (l (gensym "L")) (i (gensym "I")) (w (gensym "W"))
         (slope (gensym "SLOPE")) (term (gensym "TERM"))
         (table-vars (loop repeat n-tables collect (gensym "TABLE")))
         (entries (loop repeat n-tables collect (gensym "ENTRIES")))
         (slopes (loop repeat n-tables collect (gensym "SLOPE")))
         ;; the line values of each table at X = 0 .. points
         (lines (loop repeat n-tables collect (loop repeat (1+ points) collect (gensym "LINE"))))
         (inners (loop repeat (length xs) collect (gensym "INNER")))
         (sums (loop repeat (length xs) collect (gensym "SUM"))))
    (multiple-value-bind (buffers code)
        (if base (values '() (arithmetic-form statement)) (in-place-code statement d))
      (flet ((at (x)
               "STATEMENT's code with each table variable bound to its line value at X."
               `(let ,(loop for name in table-names for line in lines collect `(,name ,(nth x line)))
                  ,code)))
        `(lambda (,tables ,low ,high ,@constant-names)
           (declare (type list ,tables) (type simple-vector ,low ,high) (optimize speed)
                    (sb-ext:muffle-conditions sb-ext:compiler-note))
           (destructuring-bind ,table-vars ,tables
             (let* (,@(loop for table in table-vars for e in entries
                            collect `(,e (table-entries ,table)))
                    ,@(unless base
                        (loop for line in lines append (loop for v in line collect `(,v (make-ext ,d)))))
                    ,@(loop for inner in inners collect `(,inner (make-ext ,d)))
                    ,@(loop for sum in sums collect `(,sum (make-ext ,d)))
                    (,slope (make-ext ,d))
                    (,term (make-ext ,d))
                    ,@buffers)
               (declare (type ext ,@entries) (ignorable ,@entries ,slope))
               (dotimes (,h (length ,high))
                 ,@(loop for inner in inners collect `(fill ,inner 0))
                 (dotimes (,l (length ,low))
                   (let ((,i (+ (* ,h (length ,low)) ,l))
                         (,w (svref ,low ,l)))
                     (declare (type fixnum ,i))
                     ,(if base
                          `(let* ,(loop for line in lines for e in entries for s in slopes
                                        append `((,(first line) (aref ,e (* 2 ,i)))
                                                 (,(second line) (aref ,e (1+ (* 2 ,i))))
                                                 (,s (fsub ,(second line) ,(first line)))
                                                 ,@(loop for (previous next) on (rest line)
                                                         while next collect `(,next (fadd ,previous ,s)))))
                             (declare (type fp ,@(loop for line in lines append line) ,@slopes)
                                      (ignorable ,@slopes))
                             ,@(loop for x in xs for inner in inners
                                     collect `(ext+ ,inner (ext-scale ,w ,(at x) ,term) ,inner)))
                          `(progn
                             ,@(loop for line in lines for table in table-vars
                                     append `((table-entry ,table (* 2 ,i) ,(first line))
                                              (table-entry ,table (1+ (* 2 ,i)) ,(second line))
                                              (ext- ,(second line) ,(first line) ,slope)
                                              ,@(loop for (previous next) on (rest line)
                                                      while next collect `(ext+ ,previous ,slope ,next))))
                             ,@(loop for x in xs for inner in inners
                                     collect `(ext+ ,inner (ext* ,w ,(at x) ,term) ,inner))))))
                 ,@(loop for inner in inners for sum in sums
                         collect `(ext+ ,sum (ext* (svref ,high ,h) ,inner ,term) ,sum)))
               (list ,@sums))))))))

(defparameter *kernel-size-limit* 20000
  "The largest round kernel, in conses of its code, that ROUND-KERNEL compiles; a larger statement
uses GENERIC-ROUND-SUMS (compiling a huge straight-line function takes longer than it saves).")

(defun code-size (form)
  (if (consp form) (+ 1 (code-size (car form)) (code-size (cdr form))) 0))

(defvar *round-kernels* (make-hash-table :test 'equal)
  "(statement table-names constant-names d points base) -> the compiled kernel, or NIL.")

(defun round-kernel (statement table-names constant-names d points base)
  "ROUND-KERNEL-CODE compiled, once per statement and shape; NIL if its code would be larger than
*KERNEL-SIZE-LIMIT*."
  (let ((key (list statement table-names constant-names d points base *kernel-size-limit*)))
    (multiple-value-bind (kernel found) (gethash key *round-kernels*)
      (if found
          kernel
          (setf (gethash key *round-kernels*)
                (let ((code (round-kernel-code statement table-names constant-names d points base)))
                  (and (<= (code-size code) *kernel-size-limit*)
                       (compile nil code))))))))

(defun generic-round-sums (f tables constant-values low high points d)
  "g's sums at the EVALUATION-POINTS, calling the compiled statement F for each pair and point."
  (let* ((xs (evaluation-points points))
         (lines (loop repeat (length tables)
                      collect (coerce (loop repeat (1+ points) collect (make-ext d)) 'vector)))
         (arguments (loop for x in xs
                          collect (append (mapcar (lambda (line) (aref line x)) lines) constant-values)))
         (inners (loop repeat (length xs) collect (make-ext d)))
         (sums (loop repeat (length xs) collect (make-ext d)))
         (slope (make-ext d))
         (term (make-ext d)))
    (loop for wh across high
          for h from 0
          do (dolist (inner inners) (fill inner 0))
             (loop for w across low
                   for i from (* h (length low))
                   do (loop for table in tables
                            for line in lines
                            do (table-entry table (* 2 i) (aref line 0))
                               (table-entry table (1+ (* 2 i)) (aref line 1))
                               (ext- (aref line 1) (aref line 0) slope)
                               (loop for x from 2 to points
                                     do (ext+ (aref line (1- x)) slope (aref line x))))
                      (loop for args in arguments
                            for inner in inners
                            do (ext+ inner (ext* w (apply f args) term) inner)))
             (loop for inner in inners
                   for sum in sums
                   do (ext+ sum (ext* wh inner term) sum)))
    sums))

(defun round-message (sums points d)
  "The round message: the sums as the rows of a field array (POINTS, D)."
  (make-array (list points d)
              :element-type '(unsigned-byte 32)
              :initial-contents (mapcar (lambda (sum) (coerce sum 'list)) sums)))

;;; ------------------------------------------------------------------ small values
;;;
;;; The small-value prover (Bagad, Dao, Domb and Thaler, "Speeding Up Sum-Check Proving", ePrint
;;; 2025/1117).  While every table is in the base field, the first L0 rounds need not fold the
;;; tables into the extension.  Write the variables as (z, y), z the first L0 of them and y the
;;; rest.  Each table is multilinear in z and the statement has degree d in the tables, so F's
;;; value at bound challenges rho is a polynomial of degree at most d in each rho_t, and is the
;;; Lagrange interpolation of its values on the grid {0..d}^L0.  One pass over the base tables
;;; therefore computes the accumulators
;;;
;;;   A[z] = sum_y eq(r_tail, y) F(tables(z, y))          for every z in {0..d}^L0,
;;;
;;; with F evaluated in base-field arithmetic and only the weighting by eq in the extension.  Round
;;; j < L0 then reads its message off A:
;;;
;;;   g_j(X) = sum_{v in {0..d}^j} L_v(rho_0..rho_{j-1}) sum_{u in {0,1}^(L0-1-j)} eq(r_u, u) A[v, X, u]
;;;
;;; and after round L0 - 1 the tables are bound to all L0 challenges at once (BIND-PREFIX).  The
;;; messages are the same field elements as the folding rounds', so the proof is the same.

(defparameter *small-value-rounds* 3
  "L0: the number of first rounds the small-value prover answers from its accumulators, at most
(a smaller L0 is used when the statement's kernel would exceed *SMALL-VALUE-SIZE-LIMIT*).  Below 2 it
is the same as folding (the first round already runs in the base field).")

(defun grid-code (grid entries offset l0 g)
  "Straight-line code that fills the array GRID (index sum z_t G^t) with the values on
{0..G-1}^L0 of the multilinear table whose 2^L0 values on {0,1}^L0 are ENTRIES[OFFSET + b] (bit t
of b: variable t).  It places them at the grid's 0/1 points, then extends one variable at a time,
in place: each pair (v0, v1) becomes v0, v1, v1 + s, v1 + 2s, ... for the slope s = v1 - v0.
Every step is a SETF with one short-lived local, so the code stays cheap to compile."
  (flet ((position-of (digits)
           (loop for digit in digits for weight = 1 then (* weight g) sum (* digit weight)))
         (digit-lists (radices)
           (let ((lists '(())))
             (dolist (radix (reverse radices) lists)
               (setf lists (loop for digit below radix
                                 append (mapcar (lambda (rest) (cons digit rest)) lists)))))))
    `(progn
       ;; the 2^L0 entries at the grid's 0/1 points
       ,@(loop for b below (expt 2 l0)
               collect `(setf (aref ,grid ,(position-of (loop for v below l0 collect (ldb (byte 1 v) b))))
                              (aref ,entries (+ ,offset ,b))))
       ;; extend variable v; variables before it already range over the grid, those after over {0,1}
       ,@(when (> g 2)
           (loop for v below l0
                 append (loop for digits in (digit-lists (loop for u below l0
                                                               collect (cond ((< u v) g) ((= u v) 1) (t 2))))
                              collect (let ((at (lambda (x) (position-of (append (subseq digits 0 v)
                                                                                 (list x)
                                                                                 (subseq digits (1+ v)))))))
                                        `(let ((slope (fsub (aref ,grid ,(funcall at 1))
                                                            (aref ,grid ,(funcall at 0)))))
                                           ,@(loop for x from 2 below g
                                                   collect `(setf (aref ,grid ,(funcall at x))
                                                                  (fadd (aref ,grid ,(funcall at (1- x)))
                                                                        slope)))))))))))

(defun small-value-code (statement table-names d l0 g)
  "The LAMBDA form of the small-value prover's pass for STATEMENT: a function of the base tables
(in the order of TABLE-NAMES) and the LOW and HIGH eq weights of the variables after the first L0,
returning the accumulators A[z] for z in {0..G-1}^L0, a vector of EXTs.  For each y, the tables'
grid values come from unrolled straight-line code (GRID-CODE); a loop over the grid evaluates the
statement in base-field arithmetic and adds it, weighted, to the accumulators."
  (let* ((size (expt g l0))
         (block (expt 2 l0))
         (n-tables (length table-names))
         (tables (gensym "TABLES")) (low (gensym "LOW")) (high (gensym "HIGH"))
         (h (gensym "H")) (l (gensym "L")) (y (gensym "Y")) (z (gensym "Z")) (w (gensym "W"))
         (value (gensym "VALUE")) (inner (gensym "INNER")) (outer (gensym "OUTER"))
         (term (gensym "TERM"))
         (table-vars (loop repeat n-tables collect (gensym "TABLE")))
         (entries (loop repeat n-tables collect (gensym "ENTRIES")))
         (grids (loop repeat n-tables collect (gensym "GRID"))))
    `(lambda (,tables ,low ,high)
       (declare (type list ,tables) (type simple-vector ,low ,high) (optimize speed)
                (sb-ext:muffle-conditions sb-ext:compiler-note))
       (destructuring-bind ,table-vars ,tables
         (let* (,@(loop for table in table-vars for e in entries
                        collect `(,e (table-entries ,table)))
                ,@(loop for grid in grids collect `(,grid (make-ext ,size)))
                (,inner (make-array ,size))
                (,outer (make-array ,size))
                (,term (make-ext ,d)))
           (declare (type ext ,@entries ,@grids) (type simple-vector ,inner ,outer))
           (dotimes (,z ,size)
             (setf (svref ,inner ,z) (make-ext ,d)
                   (svref ,outer ,z) (make-ext ,d)))
           (dotimes (,h (length ,high))
             (dotimes (,z ,size) (fill (the ext (svref ,inner ,z)) 0))
             (dotimes (,l (length ,low))
               (let ((,y (+ (* ,h (length ,low)) ,l))
                     (,w (svref ,low ,l)))
                 (declare (type fixnum ,y) (type ext ,w))
                 ,@(loop for grid in grids for e in entries
                         collect (grid-code grid e `(* ,y ,block) l0 g))
                 (dotimes (,z ,size)
                   (let ((,value (let ,(loop for name in table-names for grid in grids
                                             collect `(,name (aref ,grid ,z)))
                                   ,(arithmetic-form statement)))
                         (acc (svref ,inner ,z)))
                     (declare (type fp ,value) (type ext acc))
                     ;; acc += w value: D base products
                     (dotimes (c ,d)
                       (setf (aref acc c) (fadd (aref acc c) (fmul (aref ,w c) ,value))))))))
             (dotimes (,z ,size)
               (let ((acc (svref ,outer ,z)))
                 (ext+ acc (ext* (svref ,high ,h) (svref ,inner ,z) ,term) acc))))
           ,outer)))))

(defparameter *small-value-size-limit* 5000
  "The largest small-value kernel, in conses of its code, that SMALL-VALUE-KERNEL compiles.  The
grid has G^L0 points per table, and compiling grows faster than the code: about 0.5 s at 4,000
conses (G = 4, L0 = 3) but 10 s at 16,000 (L0 = 4).")

(defvar *small-value-kernels* (make-hash-table :test 'equal)
  "(statement table-names d l0 g) -> the compiled SMALL-VALUE-CODE.")

(defun small-value-kernel (statement table-names d l0 g)
  "SMALL-VALUE-CODE compiled, once per statement and shape; NIL if its code would be larger than
*SMALL-VALUE-SIZE-LIMIT*."
  (let ((key (list statement table-names d l0 g *small-value-size-limit*)))
    (multiple-value-bind (kernel found) (gethash key *small-value-kernels*)
      (if found
          kernel
          (setf (gethash key *small-value-kernels*)
                (and (<= (* (expt g l0) (length table-names)) *small-value-size-limit*) ; cheap bound first
                     (let ((code (small-value-code statement table-names d l0 g)))
                       (and (<= (code-size code) *small-value-size-limit*)
                            (compile nil code)))))))))

(defun choose-small-value-kernel (statement table-names d n g)
  "The largest L0 from *SMALL-VALUE-ROUNDS* down to 2 (and at most N) whose kernel fits, with the
kernel; or NIL, to fold from the first round."
  (loop for l0 from (min *small-value-rounds* n) downto 2
        for kernel = (small-value-kernel statement table-names d l0 g)
        when kernel return (values l0 kernel)))

(defun lagrange-basis (x g)
  "L_a(X) for the nodes a = 0..G-1, as a list of EXTs."
  (let ((d (length x)))
    (loop for a below g
          collect (lagrange (loop for b below g collect (if (= a b) (ext-one d) (make-ext d))) x))))

(defun lagrange-tensor (rhos g d)
  "L_v(rho) = prod_t L_{v_t}(rho_t) for every v in {0..G-1}^|RHOS|, indexed by sum v_t G^t."
  (let ((tensor (vector (ext-one d))))
    (dolist (rho rhos tensor)
      (let ((basis (lagrange-basis rho g))
            (next (make-array (* g (length tensor)))))
        (loop for la in basis
              for a from 0
              do (loop for lv across tensor
                       for v from 0
                       do (setf (svref next (+ v (* (length tensor) a))) (ext* lv la))))
        (setf tensor next)))))

(defun radix (bits g)
  "The integer whose base-G digits are the binary digits of BITS."
  (loop for b below (integer-length bits)
        sum (* (ldb (byte 1 b) bits) (expt g b))))

(defun small-value-sums (accumulators j l0 g points rhos eq d)
  "Round J's sums at the EVALUATION-POINTS, from the accumulators and the challenges RHOS so far."
  (let ((lagrange (lagrange-tensor rhos g d))
        (eqs (weights eq (1+ j) l0 d))
        (term (make-ext d)))
    (loop for x in (evaluation-points points)
          collect (let ((sum (make-ext d)))
                    (loop for lv across lagrange
                          for v from 0
                          do (loop for e across eqs
                                   for u from 0
                                   for z = (+ v (* (expt g j) x) (* (expt g (1+ j)) (radix u g)))
                                   do (ext+ sum (ext* (ext* lv e term) (svref accumulators z) term) sum)))
                    sum))))

(defun bind-prefix (table rhos d)
  "The base TABLE with its first |RHOS| variables bound to RHOS: entry y becomes
sum_b eq(rhos, b) T(b, y), D base products per term."
  (let* ((block (expt 2 (length rhos)))
         (eqs (eq-table rhos d))
         (size (floor (table-size table) block))
         (entries (table-entries table))
         (bound (make-ext (* d size)))
         (acc (make-ext d))
         (term (make-ext d)))
    (dotimes (y size (%make-table d bound))
      (fill acc 0)
      (dotimes (b block)
        (ext+ acc (ext-scale (svref eqs b) (aref entries (+ (* y block) b)) term) acc))
      (replace bound acc :start1 (* d y)))))

(defun prove (tr n statement tables &key eq constants (label "sc") block-weights
                                        (degree (statement-degree statement constants)))
  "Prove the sumcheck of STATEMENT on the prover transcript TR (a reference prover, on the CPU).
TABLES: an alist (variable . 2^N base elements or EXTs).  With base tables and no constants, the
first *SMALL-VALUE-ROUNDS* rounds come from one base-field pass (small values); the others run the
statement's compiled ROUND-KERNEL, with the eq weights split in two (SPLIT-WEIGHTS).

With BLOCK-WEIGHTS (K EXTs), each table holds K blocks of 2^N entries, block-major, and the
statement summed is sum_b bw_b sum_x eq(r, x) F(tables(x, b)): the N rounds bind the in-block
variables (zk/sumcheck.py prove with K and bw).  DEGREE is the round polynomials' degree as the
header states it (zk/ uses 0 when N is 0).

Returns the round challenges, the finals (an alist (variable . EXT at the challenges, block 0) for
CHECK-FINAL) and the folded tables (an alist (variable . vector of the K blocks' EXTs))."
  (let* ((d (transcript-degree tr))
         (points (max degree 1))
         (names (mapcar #'car tables))
         (constant-names (mapcar #'car constants))
         (constant-values (mapcar #'cdr constants))
         (flat (loop for (nil . values) in tables collect (make-table values d)))
         (g (1+ points))
         (rhos '()))
    (absorb tr (header label n degree (if block-weights (length block-weights) 1)))
    (multiple-value-bind (l0 kernel)
        (and (null constants)
             (null block-weights)
             (every (lambda (table) (= (table-stride table) 1)) flat)
             (choose-small-value-kernel statement names d n g))
      (when kernel
        (let ((accumulators (multiple-value-call kernel flat (split-weights eq l0 n d))))
          (dotimes (j l0)
            (send tr (round-message (small-value-sums accumulators j l0 g points rhos eq d) points d))
            (setf rhos (append rhos (list (challenge tr)))))
          (setf flat (mapcar (lambda (table) (bind-prefix table rhos d)) flat)))))
    (values (append rhos
                    (loop for j from (length rhos) below n
                          collect (multiple-value-bind (low high)
                                      (if block-weights
                                          (values (weights eq (1+ j) n d) (coerce block-weights 'simple-vector))
                                          (split-weights eq (1+ j) n d))
                                    (let* ((base (and (null constants)
                                                      (every (lambda (table) (= (table-stride table) 1)) flat)))
                                           (kernel (round-kernel statement names constant-names d points base))
                                           (sums (if kernel
                                                     (apply kernel flat low high constant-values)
                                                     (generic-round-sums
                                                      (statement-function statement tables constants d)
                                                      flat constant-values low high points d))))
                                      (send tr (round-message sums points d))
                                      (let ((r (challenge tr)))
                                        (setf flat (mapcar (lambda (table) (fold-flat-table table r)) flat))
                                        r)))))
            (loop for name in names
                  for table in flat
                  collect (cons name (table-entry table 0 (make-ext d))))
            (loop for name in names
                  for table in flat
                  collect (cons name (coerce (loop for x below (table-size table)
                                                   collect (table-entry table x (make-ext d)))
                                             'vector))))))

;;; ------------------------------------------------------------------ demonstration

(defun demo (&key (statement '(* t0 t1 t2)) (variables 10) (degree 8) base (seed 1))
  "Prove and verify the sumcheck of STATEMENT (in eq mode) on random tables of VARIABLES variables
over F_p^DEGREE, then check that a tampered proof is rejected.  With BASE, the tables hold base
elements, as Sable's do.  Returns the proof's size in bytes."
  (let* ((state (sb-ext:seed-random-state seed))
         (random-ext (lambda () (coerce-ext (loop repeat degree collect (random +p+ state)))))
         (tables (loop for variable in (form-variables statement)
                       collect (cons variable (coerce (loop repeat (expt 2 variables)
                                                            collect (if base
                                                                        (random +p+ state)
                                                                        (funcall random-ext)))
                                                      'vector))))
         (eq (loop repeat variables collect (funcall random-ext)))
         (claim (hypercube-sum statement variables tables degree :eq eq))
         (start (get-internal-real-time))
         (prover (make-transcript "demo" degree))
         (finals (nth-value 1 (prove prover variables statement tables :eq eq)))
         (proof (sable.proofio:encode-proof (transcript-messages prover)))
         (proved (get-internal-real-time)))
    (flet ((verify-messages (messages)
             (let ((verifier (make-transcript "demo" degree :messages messages)))
               (check-final (nth-value 1 (verify verifier variables statement claim :eq eq))
                            statement finals)
               (unless (transcript-finished-p verifier)
                 (reject :unread-messages)))))
      (verify-messages (sable.proofio:decode-proof proof))
      (format t "~&sumcheck of ~S over ~D variables in F_p^~D, eq mode, ~:[extension~;base~] tables~%"
              statement variables degree base)
      (format t "proved in ~,3F s (~:D-byte proof), verified in ~,3F s~%"
              (/ (- proved start) internal-time-units-per-second) (length proof)
              (/ (- (get-internal-real-time) proved) internal-time-units-per-second))
      ;; one value of the first round message changed: the proof must be rejected
      (let ((tampered (sable.proofio:decode-proof proof)))
        (setf (aref (aref tampered 0) 0 0) (fadd (aref (aref tampered 0) 0 0) 1))
        (handler-case (progn (verify-messages tampered)
                             (error "a tampered proof was accepted"))
          (verification-failure (c)
            (format t "tampered proof rejected: ~A~%" c))))
      (length proof))))
