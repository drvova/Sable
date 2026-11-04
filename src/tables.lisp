;;;; tables.lisp -- the global tagged lookup table and its side of LogUp (zk/tables.py, zk/logup.py,
;;;; the table functions of zk/model.py).
;;;;
;;;; A row is a tuple (tag, v0, ..., v5); a lookup of a domain compresses its tuple as
;;;; tag + beta_0 v0 + beta_1 v1 + ...  The static tags, each a segment of rows:
;;;;
;;;;   k in [0, 24]  (k, v)                       0 <= v < 2^k          range
;;;;   STAR8         (v)                          0 <= v < 255          q + 127
;;;;   STAR9         (v)                          0 <= v < 361          RoPE output + 180
;;;;   POW           (e + 16, 2^e)                -16 <= e <= 30
;;;;   SR            (k + 16, s, R, tR, tB, oB)   -16 <= k <= 24        shift regimes
;;;;   SRA           (k + 16, tA, oA, f)          -16 <= k <= 24        left-shift input bound
;;;;   NPOW          (E + 15, E+, 2^E+, 2^E-, [E even], [E < 0])        -15 <= E <= 15
;;;;   RSQ           (delta - 2^14, floor(127 2^15 / sqrt(delta)))      2^14 <= delta < 2^16
;;;;   RECIP         (delta - 2^15, floor(2^32 / delta))                2^15 <= delta < 2^16
;;;;   SILU          (c + 2^14, tau_SiLU(c))      |c| <= 2^14
;;;;   EXP           (x, tau_exp(x))              0 <= x < 2^23
;;;;   EXPT          (d, tau_exp,theta(d))        0 <= d < 2^24         sampling (theta > 0)
;;;;   RTEXP         (x, tau_r(x))                0 <= x < 2^16         MoE router weights
;;;;
;;;; Tags from DYN0 on are dynamic tables (defined by the witness).  Lookups into the static table
;;;; are assigned, in commit order, to chunks; chunk i has its multiplicity column mult<i>.  The
;;;; table side (TABLE-SIDE) proves sum_v M(v) / (Z - T(v)) = sum_i gamma_i S_i for the chunks'
;;;; sums S_i, M = sum_i gamma_i m_i, with one LogUp-GKR over the table's rows.

(defpackage #:exacta.tables
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.forms #:exacta.transcript
        #:exacta.sumcheck #:exacta.gkr #:exacta.domain #:exacta.proto #:exacta.exact)
  (:export #:+star8+ #:+star9+ #:+pow+ #:+sr+ #:+rsq+ #:+recip+ #:+silu+ #:+exp+ #:+npow+ #:+expt+
           #:+sra+ #:+rtexp+ #:+dyn0+ #:+table-values+ #:*log-rows* #:+theta+
           #:sr-row #:sra-row #:npow-row #:table-layout #:table-row
           #:exp-values #:silu-values #:samp-values #:router-values #:rope-tables
           #:build-table #:chunks #:make-chunks #:chunks-count #:chunks-assignments #:chunk-multiplicities
           #:count-lookups #:table-side))

(in-package #:exacta.tables)

(defconstant +star8+ 25) (defconstant +star9+ 26) (defconstant +pow+ 27) (defconstant +sr+ 28)
(defconstant +rsq+ 29) (defconstant +recip+ 30) (defconstant +silu+ 31) (defconstant +exp+ 32)
(defconstant +npow+ 33) (defconstant +expt+ 34) (defconstant +sra+ 35) (defconstant +rtexp+ 36)
(defconstant +dyn0+ 64 "The first dynamic tag.")
(defconstant +table-values+ 6 "Values per row beside the tag (tables.NV).")
(defparameter *log-rows* 26 "The table has 2^*LOG-ROWS* rows (tables.LOG_ROWS).")

(defparameter +theta+ '(0d0 0.2d0 0.4d0 0.6d0 0.7d0 0.8d0 0.9d0 1d0 1.1d0 1.25d0 1.5d0)
  "The temperatures (model.THETA); index 0 is greedy decoding.")

(defun p2 (k) "2^k in F_p (k may be negative)." (fpow 2 (mod k (1- +p+))))

(defun sr-row (k)
  (list (+ k 16) (p2 k) (if (>= k 1) (expt 2 (1- k)) 0) (max k 0)
        (if (>= k 7) (- 31 k) 24) (if (>= k 7) (expt 2 (- 30 k)) (expt 2 23))))

(defun sra-row (k)
  (if (<= k 0) (list (+ k 16) (+ 24 k) (expt 2 (+ 23 k)) 1) (list (+ k 16) 0 0 0)))

(defun npow-row (e)
  (list (+ e 15) (max e 0) (expt 2 (max e 0)) (expt 2 (max (- e) 0)) (- 1 (logand e 1)) (if (minusp e) 1 0)))

(defun table-layout ()
  "Each tag's segment: an alist tag -> (base lo size), in row order (tables.Layout)."
  (let ((offset 0))
    (flet ((segment (tag size) (prog1 (list tag offset 0 size) (incf offset size))))
      (append (loop for k from 0 to 24 collect (segment k (expt 2 k)))
              (loop for (tag size) in `((,+star8+ 255) (,+star9+ 361) (,+pow+ 47) (,+sr+ 41)
                                        (,+rsq+ ,(- (expt 2 16) (expt 2 14))) (,+recip+ ,(expt 2 15))
                                        (,+silu+ ,(1+ (expt 2 15))) (,+exp+ ,(expt 2 23)) (,+npow+ 31)
                                        (,+expt+ ,(expt 2 24)) (,+sra+ 41) (,+rtexp+ ,(expt 2 16)))
                    collect (segment tag size))))))

(defun table-row (layout tag v)
  "The row of the lookup (TAG, V, ...), or NIL outside the table."
  (let ((segment (and (< tag +dyn0+) (cdr (assoc tag layout)))))
    (when segment
      (destructuring-bind (base lo size) segment
        (let ((r (- v lo))) (and (<= 0 r) (< r size) (+ base r)))))))

;;; ------------------------------------------------------------------ the model's value tables

(defun exp-values (e-attn dh)
  "tau_exp(x) = round(2^10 exp(-x 2^E / sqrt(dh))) for x < 2^23 (model.exp_table)."
  (let* ((alpha (/ (expt 2d0 e-attn) (sqrt (float dh 1d0))))
         (alpha* (div-sqrt (fixed (expt 2 e-attn)) dh))
         (out (make-array (expt 2 23) :element-type '(unsigned-byte 32))))
    (declare (type double-float alpha))
    (dotimes (x (expt 2 23) out)
      (setf (aref out x)
            (round-exact (* 1024d0 (exp (- (* x alpha))))
                         (lambda () (* 1024 (fixed-exp (- (* x alpha*))))))))))

(defun div-sqrt (a n) "A / sqrt(N) in fixed point." (floor (* a (expt 2 +precision+)) (fixed-sqrt (fixed n))))

(defun silu-values (eg ea)
  "tau_SiLU(c) = round(SiLU(c 2^eG) 2^-eA), clipped to 16 bits, for c = -2^14 .. 2^14."
  (let ((out (make-array (1+ (expt 2 15)))))
    (loop for c from (- (expt 2 14)) to (expt 2 14)
          for i from 0
          do (let* ((x (* c (expt 2d0 eg)))
                    (y (* (/ x (+ 1 (exp (- x)))) (expt 2d0 (- ea))))
                    (xe (* c (expt 2 eg))))
               (setf (aref out i)
                     (max (- (expt 2 15)) (min (1- (expt 2 15))
                                               (round-exact y (lambda ()
                                                                (floor (* (floor (* (fixed xe) (expt 2 +precision+))
                                                                                 (+ (expt 2 +precision+) (fixed-exp (fixed (- xe)))))
                                                                          (expt 2 (- ea)))))))))))
    out))

(defun samp-values (theta)
  "tau_exp,theta(d) = round(2^12 exp(-d / (2^8 T))) for d < 2^24, T = THETA[theta] (model.samp_table)."
  (check-parameter theta (lambda (th) (and (integerp th) (< 0 th (length +theta+)))) "a temperature index above 0")
  (let* ((temperature (nth theta +theta+))
         (scale (/ 1 (* 256 (rational temperature))))
         (out (make-array (expt 2 24) :element-type '(unsigned-byte 32))))
    (dotimes (d (expt 2 24) out)
      (setf (aref out d)
            (round-exact (* 4096d0 (exp (- (/ d (* 256d0 temperature)))))
                         (lambda () (* 4096 (fixed-exp (fixed (- (* d scale)))))))))))

(defun router-values ()
  "tau_r(x) = round(2^12 exp(-x / 2^8)) for x < 2^16 (model.router_table)."
  (let ((out (make-array (expt 2 16) :element-type '(unsigned-byte 32))))
    (dotimes (x (expt 2 16) out)
      (setf (aref out x)
            (round-exact (* 4096d0 (exp (- (/ x 256d0))))
                         (lambda () (* 4096 (fixed-exp (fixed (- (/ x 256)))))) 1d-6)))))

(defun rope-tables (theta dh s)
  "round(2^15 cos(t theta^(-2j/dh))) and the same for sin, for t < S, j < dh/2: two arrays (S, dh/2)
(model.rope_tables)."
  (let* ((half (floor dh 2))
         (ln-theta (fixed-ln (fixed (rational theta))))
         (cos (make-array (list s half)))
         (sin (make-array (list s half))))
    (dotimes (j half (values cos sin))
      (let ((inv (/ 1d0 (expt (float theta 1d0) (/ (float (* 2 j) 1d0) dh))))
            (inv* (fixed-exp (- (floor (* ln-theta 2 j) dh)))))
        (dotimes (tt s)
          (let ((angle (* tt inv)))
            (setf (aref cos tt j) (round-exact (* 32768d0 (cos angle)) (lambda () (* 32768 (fixed-cos (* tt inv*)))))
                  (aref sin tt j) (round-exact (* 32768d0 (sin angle)) (lambda () (* 32768 (fixed-sin (* tt inv*))))))))))))

;;; ------------------------------------------------------------------ the table

(defun build-table (exp silu &key samp router (log-rows *log-rows*))
  "The table's columns (tag, v0 .. v5), each 2^LOG-ROWS field values, unused rows (0, 0): EXP and
SILU from EXP-VALUES and SILU-VALUES, SAMP (sampling at a temperature above 0) and ROUTER (MoE)
when given."
  (let ((columns (loop repeat (1+ +table-values+)
                       collect (make-array (expt 2 log-rows) :element-type '(unsigned-byte 32) :initial-element 0)))
        (layout (table-layout)))
    (flet ((put (tag size row)
             "Rows of TAG: (funcall ROW i) gives the values of its I-th row."
             (let ((base (second (assoc tag layout))))
               (dotimes (i size)
                 (setf (aref (first columns) (+ base i)) tag)
                 (loop for v in (funcall row i)
                       for column in (rest columns)
                       do (setf (aref column (+ base i)) (mod v +p+)))))))
      (dotimes (k 25) (put k (expt 2 k) #'list))
      (put +star8+ 255 #'list)
      (put +star9+ 361 #'list)
      (put +pow+ 47 (lambda (e) (list e (p2 (- e 16)))))
      (put +sr+ 41 (lambda (i) (sr-row (- i 16))))
      (put +sra+ 41 (lambda (i) (sra-row (- i 16))))
      (put +npow+ 31 (lambda (i) (npow-row (- i 15))))
      (let ((c2 (expt (* 127 (expt 2 15)) 2)))
        (put +rsq+ (- (expt 2 16) (expt 2 14)) (lambda (i) (list i (isqrt (floor c2 (+ i (expt 2 14))))))))
      (put +recip+ (expt 2 15) (lambda (i) (list i (floor (expt 2 32) (+ i (expt 2 15))))))
      (put +silu+ (1+ (expt 2 15)) (lambda (i) (list i (aref silu i))))
      (put +exp+ (expt 2 23) (lambda (x) (list x (aref exp x))))
      (when samp (put +expt+ (expt 2 24) (lambda (d) (list d (aref samp d)))))
      (when router (put +rtexp+ (expt 2 16) (lambda (x) (list x (aref router x))))))
    columns))

;;; ------------------------------------------------------------------ chunks and multiplicities

(defstruct (chunks (:constructor make-chunks (&key (limit +lookup-limit+) (log-rows *log-rows*))))
  "Static lookups assigned, in commit order, to chunks of at most LIMIT lookups (logup.Chunks),
each with its multiplicities over the table's rows."
  limit log-rows (assignments '()) (multiplicities '()) (current-size 0))

(defun chunks-count (chunks) (length (chunks-multiplicities chunks)))
(defun chunk-multiplicities (chunks i) (nth i (chunks-multiplicities chunks)))

(defun take-chunk (chunks key count)
  "The chunk that takes COUNT lookups of KEY (domain lookup): the current one, or a new one."
  (when (or (null (chunks-multiplicities chunks))
            (> (+ (chunks-current-size chunks) count) (chunks-limit chunks)))
    (setf (chunks-multiplicities chunks)
          (append (chunks-multiplicities chunks)
                  (list (make-array (expt 2 (chunks-log-rows chunks)) :element-type '(unsigned-byte 32)
                                                                       :initial-element 0)))
          (chunks-current-size chunks) 0))
  (incf (chunks-current-size chunks) count)
  (let ((i (1- (chunks-count chunks))))
    (setf (chunks-assignments chunks) (append (chunks-assignments chunks) (list (cons key i))))
    (chunk-multiplicities chunks i)))

(defun count-lookups (chunks domain &key (layout (table-layout)))
  "Count DOMAIN's lookups into the static table (its columns' data) in their chunks: the :COMMIT
pass's counter (tables.count).  A lookup outside the table signals an error."
  (let* ((n (domain-n domain))
         (k (domain-blocks domain))
         (columns (domain-columns domain))
         (names (mapcar #'column-name columns)))
    (loop for lookup in (domain-lookups domain)
          for l from 0
          unless (or (getf lookup :dynamic) (getf lookup :num))
            do (let ((mult (take-chunk chunks (list (domain-name domain) l) (* k (expt 2 n))))
                     (tag (compile-form (getf lookup :tag) names))
                     (value (compile-form (first (getf lookup :values)) names)))
                 (dotimes (b k)
                   (dotimes (x (expt 2 n))
                     (let* ((args (mapcar (lambda (column) (exacta.domain::column-value column b x n)) columns))
                            (tg (apply tag args))
                            (v (apply value args))
                            (row (table-row layout tg v)))
                       (unless row
                         (error 'invalid-parameter :name 'lookup :value (list (domain-name domain) l tg v)
                                                   :expected "a tuple in the table"))
                       (incf (aref mult row)))))))))

;;; ------------------------------------------------------------------ the table side of LogUp

(defun tuple-coefficients (beta count)
  "The coefficients of a tuple's entries: 1 for the tag, then beta_0 ... (gkr.tuple_coefs)."
  (cons (ext-one 4) (subseq beta 0 count)))

(defun compress (row-values coefficients z)
  "Z - sum_j coefficient_j value_j."
  (reduce #'ext- (mapcar (lambda (c v) (if (integerp v) (ext-scale c v) (ext* c v))) coefficients row-values)
          :initial-value z))

(defun table-side (context &key columns chunks sums)
  "The table side of every chunk (logup.table_side): the prover gives the table's COLUMNS and its
CHUNKS, the verifier the chunks' root SUMS.  Leaves claims on the tab<j> and mult<i> tensors."
  (let* ((tr (context-tr context))
         (n (if columns (integer-length (1- (length (first columns)))) *log-rows*))
         (nchunks (if (prover-p context) (chunks-count chunks) (length sums)))
         (half (expt 2 (1- n))))
    (absorb tr "table-side")
    (let ((gammas (context-challenge context nchunks))
          (coefficients (tuple-coefficients (context-beta context) +table-values+))
          (z (context-z context))
          m q ri cp cq)
      (if (prover-p context)
          (progn
            ;; M = sum_i gamma_i m_i; leaves -M / Q, Q = Z - T(v)
            (setf m (make-array (expt 2 n))
                  q (make-array (expt 2 n)))
            (dotimes (v (expt 2 n))
              (setf (aref m v) (reduce #'ext+ (loop for g in gammas for i from 0
                                                    collect (ext-scale g (aref (chunk-multiplicities chunks i) v)))
                                       :initial-value (make-ext 4))
                    (aref q v) (compress (mapcar (lambda (column) (aref column v)) columns) coefficients z)))
            (multiple-value-bind (r rb p* q*) (prove-upper tr n 1 (list (cons (map 'vector #'ext-neg m) q)) :label "tgkr")
              (declare (ignore rb))
              (setf ri r cp p* cq q*)))
          (multiple-value-bind (roots r rb p* q*) (verify-upper tr n 1 1 :label "tgkr")
            (declare (ignore rb))
            (destructuring-bind ((p . qq)) roots
              (let ((total (reduce #'ext+ (mapcar #'ext* gammas sums)
                                   :initial-value (ext* p (handler-bind ((field-division-by-zero
                                                                            (lambda (c) (declare (ignore c)) (invoke-restart 'return-zero))))
                                                            (ext-inv qq))))))
                (unless (ext= total (make-ext 4))
                  (reject :logup "the chunk sums do not match the table side"))))
            (setf ri r cp p* cq q*)))
      ;; the leaf layer: sum_x eq(ri, x) [-a (M0 Q1 + M1 Q0) + a lam Q0 Q1]
      (let* ((a (context-challenge context))
             (lam (context-challenge context))
             (claim (layer-claim (list a) lam cp cq))
             (statement '(+ (* na m0 q1) (* na m1 q0) (* al q0 q1)))
             (constants `((na . ,(ext-neg a)) (al . ,(ext* a lam)))))
        (multiple-value-bind (rho fm)
            (if (prover-p context)
                (multiple-value-bind (rho finals)
                    (prove tr (1- n) statement
                           `((m0 . ,(subseq m 0 half)) (m1 . ,(subseq m half)) (q0 . ,(subseq q 0 half)) (q1 . ,(subseq q half)))
                           :eq ri :constants constants :label "tleaf" :degree 2)
                  (values rho (list (aref (send-values context (vector (cdr (assoc 'm0 finals)))) 0)
                                    (aref (send-values context (vector (cdr (assoc 'm1 finals)))) 0))))
                (multiple-value-bind (rho reduced) (verify tr (1- n) statement claim :eq ri :constants constants
                                                                                     :label "tleaf" :degree 2)
                  (setf claim reduced)
                  (values rho (list (aref (send-values context) 0) (aref (send-values context) 0)))))
          ;; the table's values at (rho, 0) and (rho, 1): claims on the committed table
          (let ((tvs (loop for h below 2
                           collect (let* ((point (append rho (list (base->ext h 4))))
                                          (values (if (prover-p context)
                                                      (send-values context (map 'vector (lambda (column) (mle-eval column point 4)) columns))
                                                      (send-values context nil (list (1+ +table-values+) 4)))))
                                     (loop for v across values for j from 0
                                           do (claim context (format nil "tab~D" j) point (list (cons 0 v))))
                                     values))))
            (unless (prover-p context)
              (destructuring-bind (q0 q1) (mapcar (lambda (tv) (compress (coerce tv 'list) coefficients z)) tvs)
                (unless (ext= claim (evaluate statement `((m0 . ,(first fm)) (m1 . ,(second fm)) (q0 . ,q0) (q1 . ,q1)
                                                          ,@constants)
                                              :degree 4))
                  (reject :logup "the table's leaf sumcheck failed")))))
          ;; the multiplicities at (rho, t)
          (let* ((tt (context-challenge context))
                 (point (append rho (list tt)))
                 (mt (ext+ (first fm) (ext* tt (ext- (second fm) (first fm)))))
                 (mv (if (prover-p context)
                         (send-values context (coerce (loop for i below nchunks
                                                            collect (mle-eval (chunk-multiplicities chunks i) point 4))
                                                      'vector))
                         (send-values context nil (list nchunks 4)))))
            (unless (prover-p context)
              (unless (ext= mt (reduce #'ext+ (loop for g in gammas for v across mv collect (ext* g v))
                                       :initial-value (make-ext 4)))
                (reject :logup "the multiplicity values do not match")))
            (loop for v across mv for i from 0
                  do (claim context (format nil "mult~D" i) point (list (cons 0 v))))))))))
