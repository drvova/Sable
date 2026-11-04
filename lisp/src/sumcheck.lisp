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
           #:check-final #:round-kernel-code #:*kernel-size-limit* #:demo))

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

(defun verify (tr n statement claim &key eq constants (blocks 1) (label "sc"))
  "Replay the N-round sumcheck of STATEMENT on the verifier transcript TR.  CLAIM is an EXT (or
its coordinates); EQ, in eq mode, a sequence of at least N EXTs; CONSTANTS an alist of the
statement's non-table variables.  Returns the round challenges and the reduced claim, for
CHECK-FINAL.  A round message of the wrong shape signals MESSAGE-SHAPE-ERROR."
  (let* ((degree (statement-degree statement constants))
         (shape (list (max degree 1) (transcript-degree tr)))
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

;;; One round of the prover computes g(X) = sum_i w_i F(line values at X) at X = 0, 2, 3, ...,
;;; points.  For each pair (lo, hi) of a table, the line values at X = 0, 1, 2, ... are lo, hi,
;;; hi + s, hi + 2s, ... for the slope s = hi - lo: additions only.  ROUND-KERNEL-CODE writes the
;;; whole round for one statement as a LAMBDA form, with the statement inlined and the tables and
;;; points unrolled, every buffer made once per round; GENERIC-ROUND-SUMS is the same loop
;;; calling the compiled statement, for statements whose kernel would be too large.

(defun evaluation-points (points)
  "X = 0, 2, 3, ..., POINTS: g(1) follows from the claim."
  (cons 0 (loop for x from 2 to points collect x)))

(defun round-kernel-code (statement table-names constant-names d points base)
  "The LAMBDA form of one prover round for STATEMENT: a function of the tables (TABLEs, in the
order of TABLE-NAMES), the eq weights (a vector of EXTs) and the constants' values (in the order
of CONSTANT-NAMES), returning g's sums at the EVALUATION-POINTS.  With BASE (all tables in the
base field, no constants), the line values and the statement are base-field integers, and each
term costs one base-times-extension product by its weight."
  (let* ((xs (evaluation-points points))
         (n-tables (length table-names))
         (tables (gensym "TABLES")) (weights (gensym "WEIGHTS")) (i (gensym "I")) (w (gensym "W"))
         (slope (gensym "SLOPE")) (term (gensym "TERM"))
         (table-vars (loop repeat n-tables collect (gensym "TABLE")))
         (entries (loop repeat n-tables collect (gensym "ENTRIES")))
         (slopes (loop repeat n-tables collect (gensym "SLOPE")))
         ;; the line values of each table at X = 0 .. points
         (lines (loop repeat n-tables collect (loop repeat (1+ points) collect (gensym "LINE"))))
         (sums (loop repeat (length xs) collect (gensym "SUM"))))
    (multiple-value-bind (buffers code)
        (if base (values '() (arithmetic-form statement)) (in-place-code statement d))
      (flet ((at (x)
               "STATEMENT's code with each table variable bound to its line value at X."
               `(let ,(loop for name in table-names for line in lines collect `(,name ,(nth x line)))
                  ,code)))
        `(lambda (,tables ,weights ,@constant-names)
           (declare (type list ,tables) (type simple-vector ,weights) (optimize speed)
                    (sb-ext:muffle-conditions sb-ext:compiler-note))
           (destructuring-bind ,table-vars ,tables
             (let* (,@(loop for table in table-vars for e in entries
                            collect `(,e (table-entries ,table)))
                    ,@(unless base
                        (loop for line in lines append (loop for v in line collect `(,v (make-ext ,d)))))
                    ,@(loop for sum in sums collect `(,sum (make-ext ,d)))
                    (,slope (make-ext ,d))
                    (,term (make-ext ,d))
                    ,@buffers)
               (declare (type ext ,@entries) (ignorable ,@entries ,slope))
               (dotimes (,i (length ,weights))
                 (let ((,w (svref ,weights ,i)))
                   ,(if base
                        `(let* ,(loop for line in lines for e in entries for s in slopes
                                      append `((,(first line) (aref ,e (* 2 ,i)))
                                               (,(second line) (aref ,e (1+ (* 2 ,i))))
                                               (,s (fsub ,(second line) ,(first line)))
                                               ,@(loop for (previous next) on (rest line)
                                                       while next collect `(,next (fadd ,previous ,s)))))
                           (declare (type fp ,@(loop for line in lines append line) ,@slopes) (ignorable ,@slopes))
                           ,@(loop for x in xs for sum in sums
                                   collect `(ext+ ,sum (ext-scale ,w ,(at x) ,term) ,sum)))
                        `(progn
                           ,@(loop for line in lines for table in table-vars
                                   append `((table-entry ,table (* 2 ,i) ,(first line))
                                            (table-entry ,table (1+ (* 2 ,i)) ,(second line))
                                            (ext- ,(second line) ,(first line) ,slope)
                                            ,@(loop for (previous next) on (rest line)
                                                    while next collect `(ext+ ,previous ,slope ,next))))
                           ,@(loop for x in xs for sum in sums
                                   collect `(ext+ ,sum (ext* ,w ,(at x) ,term) ,sum))))))
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

(defun generic-round-sums (f tables constant-values weights points d)
  "g's sums at the EVALUATION-POINTS, calling the compiled statement F for each pair and point."
  (let* ((xs (evaluation-points points))
         (lines (loop repeat (length tables)
                      collect (coerce (loop repeat (1+ points) collect (make-ext d)) 'vector)))
         (arguments (loop for x in xs
                          collect (append (mapcar (lambda (line) (aref line x)) lines) constant-values)))
         (sums (loop repeat (length xs) collect (make-ext d)))
         (slope (make-ext d))
         (term (make-ext d)))
    (loop for w across weights
          for i from 0
          do (loop for table in tables
                   for line in lines
                   do (table-entry table (* 2 i) (aref line 0))
                      (table-entry table (1+ (* 2 i)) (aref line 1))
                      (ext- (aref line 1) (aref line 0) slope)
                      (loop for x from 2 to points
                            do (ext+ (aref line (1- x)) slope (aref line x))))
             (loop for args in arguments
                   for sum in sums
                   do (ext+ sum (ext* w (apply f args) term) sum)))
    sums))

(defun round-message (sums points d)
  "The round message: the sums as the rows of a field array (POINTS, D)."
  (make-array (list points d)
              :element-type '(unsigned-byte 32)
              :initial-contents (mapcar (lambda (sum) (coerce sum 'list)) sums)))

(defun prove (tr n statement tables &key eq constants (label "sc"))
  "Prove the sumcheck of STATEMENT on the prover transcript TR (a reference prover, on the CPU).
TABLES: an alist (variable . 2^N base elements or EXTs); base tables stay in the base field until
the first fold.  Each round runs the statement's compiled ROUND-KERNEL.  Returns the round
challenges and the finals, an alist (variable . EXT at the challenges) for CHECK-FINAL."
  (let* ((d (transcript-degree tr))
         (points (max (statement-degree statement constants) 1))
         (names (mapcar #'car tables))
         (constant-names (mapcar #'car constants))
         (constant-values (mapcar #'cdr constants))
         (flat (loop for (nil . values) in tables collect (make-table values d))))
    (absorb tr (header label n (statement-degree statement constants)))
    (values (loop for j below n
                  collect (let* ((weights (weights eq (1+ j) n d))
                                 (base (and (null constants)
                                            (every (lambda (table) (= (table-stride table) 1)) flat)))
                                 (kernel (round-kernel statement names constant-names d points base))
                                 (sums (if kernel
                                           (apply kernel flat weights constant-values)
                                           (generic-round-sums
                                            (statement-function statement tables constants d)
                                            flat constant-values weights points d))))
                            (send tr (round-message sums points d))
                            (let ((r (challenge tr)))
                              (setf flat (mapcar (lambda (table) (fold-flat-table table r)) flat))
                              r)))
            (loop for name in names
                  for table in flat
                  collect (cons name (table-entry table 0 (make-ext d)))))))

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
