;;;; steps.lisp -- the per-layer steps of the transformer proof (zk/llm.py Proto): each builds
;;;; domains over a layer's witness, routes their column claims to committed tensors, and opens
;;;; its group.  The same code proves (from the witness) and verifies (from the transcript).
;;;;
;;;; A domain is built with a DB: columns with their data (prover) and a route for their claim --
;;;; ROUTE-BLOCK (a claim on a committed tensor), ROUTE-VIRTUAL (left for the step that defines the
;;;; column) or ROUTE-PUBLIC (evaluated by the verifier) -- then constraints and lookups as forms over
;;;; the columns.  Column symbols live in the package EXACTA.COLUMNS, named as Sable names them.

(defpackage #:exacta.columns (:use))

(defpackage #:exacta.steps
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.transcript
        #:exacta.domain #:exacta.proto #:exacta.llm #:exacta.model)
  (:import-from #:exacta.tables #:sr-row #:sra-row #:+pow+ #:+npow+ #:+rsq+ #:+sr+ #:+sra+ #:+dyn0+)
  (:shadowing-import-from #:exacta.model #:rows #:columns)
  (:export #:layer-witness #:rq-pack #:ns-pack #:sr-pack #:rm-layout
           #:db #:make-db #:db-column #:db-zero #:db-lookup #:db-run
           #:route-block #:route-virtual #:route-public
           #:rq-rows #:ns-rows #:sr-rows #:cap-col #:combine-blocks
           #:layer-parameters #:layer-head #:step-mo #:step-mlp #:step-mi #:step-ao #:with-domain-batch #:step-state #:matmul #:oh-check #:gather-rowsum #:rq-norm-dom))

(in-package #:exacta.steps)

(defconstant +inv2+ (/ (1+ +p+) 2))
(defconstant +rsq-lo+ (expt 2 14))

(defun field (x) (mod x +p+))
(defun p2 (k) (fpow 2 (mod k (1- +p+))))

;;; ------------------------------------------------------------------ witness layouts

(defun rm-layout (m cb)
  "A (T x N) matrix as field values block-major [N / 2^cb][T][2^cb] (llm.rm)."
  (let* ((tt (rows m)) (n (columns m)) (w (expt 2 cb))
         (out (make-array (* tt n))))
    (dotimes (blk (floor n w) out)
      (dotimes (i tt)
        (dotimes (lo w)
          (setf (aref out (+ (* (+ (* blk tt) i) w) lo)) (field (aref m i (+ (* blk w) lo)))))))))

(defun row-field (v) (map 'vector #'field v))

(defun rq-pack (a cb &optional (e0 0) (emax 17))
  "Requant8 / ShiftQ8 of the rows of A with the per-row witnesses (llm.rq_pack): q, e, s = 2^e, f,
sig, the minimality witness w in two limbs (14-bit, quartered, for ShiftQ8), asel and the bits of
jstar (cb low bits, then the block's)."
  (multiple-value-bind (q e w) (requant8 a e0 emax)
    (let* ((n (rows a)) (kblocks (ash (columns a) (- cb)))
           (s (map 'vector #'pow2* e))
           (quarter (= e0 2))
           (wl (make-array n)) (wh (make-array n)))
      (dotimes (i n)
        (let* ((f (aref (getf w :f) i)) (sig (aref (getf w :sigma) i)) (asel (aref (getf w :asel) i)) (si (aref s i))
               (ww (* f (if quarter
                            (if (= sig 1) (- asel (* 255 (ash si -2))) (- (- asel) (* 255 (ash si -2)) 1))
                            (if (= sig 1) (- (* 4 asel) (* 255 si)) (- (* -4 asel) (* 255 si) 1))))))
          (witness (and (<= 0 ww) (< ww (if quarter (expt 2 28) (expt 2 30)))) "requant: minimality witness range")
          (setf (aref wl i) (ldb (byte (if quarter 14 15) 0) ww)
                (aref wh i) (ash ww (if quarter -14 -15)))))
      (list :q q :e e :s s :f (getf w :f) :sig (getf w :sigma) :wl wl :wh wh :asel (getf w :asel)
            :jb (let ((js (getf w :jstar)))
                  (append (loop for i below cb collect (map 'vector (lambda (j) (ldb (byte 1 i) j)) js))
                          (loop for i below (nbits kblocks)
                                collect (map 'vector (lambda (j) (ldb (byte 1 i) (ash j (- cb)))) js))))))))

(defun pow2* (e) (ash 1 e))

(defun ns-pack (q)
  "NormScale of the rows of Q with its witnesses (llm.ns_pack)."
  (multiple-value-bind (qp eh w) (normscale q)
    (list :qp qp :eh eh :ep (getf w :ep) :pep (map 'vector #'pow2* (getf w :ep)) :pen (map 'vector #'pow2* (getf w :en))
          :peh (map 'vector #'p2 eh) :delta (getf w :delta) :rho (getf w :rho) :mu (getf w :mu) :sig (getf w :sig)
          :ng (map 'vector (lambda (e) (if (minusp e) 1 0)) eh))))

(defun sr-pack (k)
  "The SR and SRA rows of the per-row shifts K (llm.sr_pack): k + 16, s, R, tR, tB, oB, tA, oA, f."
  (let ((rows (map 'list (lambda (x)
                          (witness (<= -16 x 24) "SR: shift outside [-16, 24]")
                          (append (sr-row x) (rest (sra-row x))))
                   k)))
    (loop for name in '(:k :s :r :tr :tb :ob :ta :oa :f)
          for i from 0
          collect name
          collect (coerce (loop for row in rows collect (nth i row)) 'vector))))

(defun sr-field (pack name) (row-field (getf pack (intern (string-upcase name) :keyword))))

;;; ------------------------------------------------------------------ the layer witness

(defun sign-of (x) (signum x))

(defun layer-witness (model l h shape tables &optional prev)
  "Layer L's witness on its input rows H (llm.LayerWit, the values the steps read so far); PREV, the
previous layer's witness (its output's SatAdd values are this layer's input columns)."
  (let* ((cfg (model-cfg model))
         (layer (aref (model-layers model) l))
         (cb (shape-cb shape)))
    (multiple-value-bind (hout w) (layer-forward model layer h (shape-b shape) (shape-s shape) tables)
      (let* ((mlp (getf w :mlp))
             (hmid (getf w :h-mid))
             (s1 (map-matrix2 #'+ h (getf w :c-o)))
             (s2 (map-matrix2 #'+ hmid (getf mlp :y)))
             (r2 (rq-pack hmid cb))
             (n2 (ns-pack (getf r2 :q)))
             (kgl (layer-value layer "kgl"))
             (eg (map 'vector (lambda (e) (max e kgl)) (getf n2 :eh))))
        (when (cfg-moe cfg) (error "layer-witness: MoE layers are not ported yet"))
        (list :hin h :hout hout :forward w
              :s1 s1 :os1 (map-matrix2 (lambda (s hm) (signum (- s hm))) s1 hmid)
              :s2 s2 :os2 (map-matrix2 (lambda (s ho) (signum (- s ho))) s2 hout)
              :accd (getf mlp :accd) :kd (getf mlp :kd)
              :r2 r2 :n2 n2
              :eg (list :e eg :pe (map 'vector #'p2 eg) :f (map 'vector (lambda (e) (if (< e kgl) 1 0)) (getf n2 :eh)))
              :ru (rq-pack (getf mlp :accu) cb 2 22)
              :rm (rq-pack (getf mlp :m) cb)
              :acc-o (getf w :acc-o) :o (getf w :o) :ns (getf w :ns) :ndxh (getf w :ndxh) :ndt (getf w :ndt)
              :nd (getf w :nd)
              :ro (rq-pack (getf w :o) cb)
              :ko (map 'vector (lambda (e) (- (layer-value layer "Kco") e)) (getf w :eo))
              :prev prev)))))

(defun map-matrix2 (f a b)
  (let ((out (make-matrix (rows a) (columns a))))
    (dotimes (i (rows a) out)
      (dotimes (j (columns a)) (setf (aref out i j) (funcall f (aref a i j) (aref b i j)))))))

;;; ------------------------------------------------------------------ domain builder

(defstruct (db (:constructor %make-db (context name n k merge)))
  context name n k merge (columns '()) (constraints '()) (lookups '()) (routes '()))

(defun make-db (context name n &key (k 1) merge) (%make-db context name n k merge))

(defun column-symbol (name) (intern name '#:exacta.columns))

(defun db-column (db name data route &key (kind :full) (c 0) shared)
  "Add column NAME: DATA (a vector or a function giving one; read by the prover only), its claim's
ROUTE.  Returns the column's symbol, for forms."
  (let ((symbol (column-symbol name)))
    (setf (db-columns db)
          (append (db-columns db)
                  (list (list symbol :kind kind :c (if (eq kind :full) 0 c) :shared shared
                              :data (and (prover-p (db-context db))
                                         (row-field (if (functionp data) (funcall data) data))))))
          (db-routes db) (append (db-routes db) (list (cons symbol route))))
    symbol))

(defun db-zero (db form) (setf (db-constraints db) (append (db-constraints db) (list form))))

(defun db-lookup (db tag values &key num dynamic)
  "The tuple (TAG, VALUES...) is in the table (or, with NUM or DYNAMIC, in a dynamic set)."
  (setf (db-lookups db) (append (db-lookups db) (list (list :tag tag :values values :num num :dynamic dynamic)))))

(defvar *layer-parameters* '() "The layer parameters of the layer being proved (dm.set_lparams).")

(defvar *domain-batch* nil
  "Inside WITH-DOMAIN-BATCH: the domains run so far (they are proved when the batch closes).")

(defun db-run (db)
  (if *domain-batch*
      (vector-push-extend db *domain-batch*)
      (run-db db)))

(defun run-db (db)
  (run-domain (db-context db)
              (make-domain :name (db-name db) :n (db-n db) :blocks (db-k db) :merge (db-merge db)
                           :columns (db-columns db) :constraints (db-constraints db) :lookups (db-lookups db)
                           :parameters *layer-parameters*)
              (db-routes db)))

;;; routes: (lambda (point value)); VALUE a vector of the blocks' values, or one EXT (merged)

(defun block-values (value) (if (typep value 'ext) (vector value) value))

(defun route-block (context name &key bmap pre post)
  "Claims on the blocks of tensor NAME at the column's point (PRE and POST extend it); BMAP maps a
domain block to a tensor block, or to NIL to drop it."
  (lambda (point value)
    (let ((point (append pre point post))
          (values (block-values value)))
      (claim context name point
             (if bmap
                 (loop for v across values for b from 0
                       for target = (funcall bmap b)
                       when target collect (cons target v))
                 values)))))

(defun route-virtual (context key)
  (lambda (point value) (put-virtual context key (list point (block-values value)))))

(defun route-public (context fn)
  "The verifier evaluates the column itself: (FN point) gives its blocks' values."
  (lambda (point value)
    (unless (prover-p context)
      (unless (every #'ext= (funcall fn point) (block-values value))
        (reject :public-column "public column mismatch")))))

;;; ------------------------------------------------------------------ per-row gadgets

(defparameter +rq-f+ '("e" "s" "f" "sig" "wl" "wh"))
(defparameter +ns-f+ '("Eh" "Ep" "pEp" "pEn" "pEh" "delta" "rho" "mu" "ng"))
(defparameter +sr-f+ '("k" "s" "R" "tR" "tB" "oB"))
(defparameter +sra-f+ '("tA" "oA" "f"))

(defun pack-field (pack name)
  "Field NAME (Sable's spelling) of a witness plist."
  (getf pack (intern (string-upcase name) :keyword)))

(defun s+ (&rest parts) (apply #'concatenate 'string parts))

(defun rq-rows (context db pre r k &key (e0 0) (emax 17) asel-key (cb nil cb-p) (b 127) (shape-cb 0))
  "Per-row Requant8 / ShiftQ8 minimality over rows of K 2^cb elements (Proto.rq_rows).  Returns
the columns, an alist field -> symbol."
  (let* ((cb (if cb-p cb shape-cb))
         (c (loop for f in +rq-f+
                  collect (cons f (db-column db (s+ pre f) (lambda () (pack-field r f)) (route-block context (s+ pre f))))))
         (nb (+ cb (nbits k)))
         (bits (loop for i below nb
                     collect (let ((i i))
                               (db-column db (format nil "~Ajb~D" pre i) (lambda () (nth i (getf r :jb)))
                                          (route-block context (s+ pre "jb") :bmap (lambda (blk) (declare (ignore blk)) i))))))
         (asel (db-column db (s+ pre "asel") (lambda () (getf r :asel)) (route-virtual context asel-key))))
    (destructuring-bind (e s f sg wl wh) (mapcar (lambda (x) (cdr (assoc x c :test #'string=))) +rq-f+)
      (db-lookup db +pow+ `((+ ,e 16) ,s))
      (db-lookup db 5 `((- ,e ,e0)))
      (db-lookup db 5 `((- ,emax ,e)))
      (db-zero db `(* ,f (- 1 ,f)))
      (db-zero db `(* (- ,e ,e0) (- 1 ,f)))
      (db-zero db `(* ,sg (- 1 ,sg)))
      (if (= e0 2)
          (let ((q4 (fpow 4 (- +p+ 2))))
            (db-zero db `(- (* ,f (+ (* ,sg (- ,asel (* ,(* 255 q4) ,s))) (* (- 1 ,sg) (- (- ,asel) (* ,(* 255 q4) ,s) 1))))
                            (* ,(expt 2 14) ,wh) ,wl))
            (db-lookup db 14 (list wl))
            (db-lookup db 14 (list wh)))
          (let ((w (1+ (* 2 b))))
            (db-zero db `(- (* ,f (+ (* ,sg (- (* 4 ,asel) (* ,w ,s))) (* (- 1 ,sg) (- (* -4 ,asel) (* ,w ,s) 1))))
                            (* ,(expt 2 15) ,wh) ,wl))
            (db-lookup db 15 (list wl))
            (db-lookup db 15 (list wh))))
      (dolist (bit bits) (db-zero db `(* ,bit (- 1 ,bit))))
      (when (logtest k (1- k))
        (db-lookup db 24 (list `(- ,(1- (* k (expt 2 cb))) (+ ,@(loop for bit in bits for i from 0 collect `(* ,(expt 2 i) ,bit))))))))
    c))

(defun ns-rows (context db pre n sig-key)
  "Per-row NormScale (Proto.ns_rows)."
  (let ((c (loop for f in +ns-f+
                 collect (cons f (db-column db (s+ pre f) (lambda () (pack-field n f)) (route-block context (s+ pre f))))))
        (sig (db-column db (s+ pre "sig") (lambda () (getf n :sig)) (route-virtual context sig-key))))
    (destructuring-bind (eh ep pep pen peh delta rho mu ng) (mapcar (lambda (x) (cdr (assoc x c :test #'string=))) +ns-f+)
      (db-lookup db +npow+ `((+ (* 2 ,eh) 15) ,ep ,pep ,pen 1 ,ng))
      (db-lookup db +pow+ `((+ ,eh 16) ,peh))
      (db-lookup db ep (list rho))
      (db-lookup db 16 (list `(* ,ng ,sig)))
      (db-lookup db +rsq+ `((- ,delta ,+rsq-lo+) ,mu))
      (db-zero db `(- (* ,pen ,sig) (* ,pep ,delta) ,rho)))
    c))

(defun sr-rows (context db pre k kexpr &key left)
  "The SR regime tuple of a per-row shift K, whose value is the form KEXPR (Proto.sr_rows); LEFT: K
may be <= 0, and the row also carries the SRA input bound."
  (let* ((pack (sr-pack k))
         (fields (if left (append +sr-f+ +sra-f+) +sr-f+))
         (c (loop for f in fields
                  collect (cons f (db-column db (s+ pre f) (lambda () (sr-field pack f)) (route-block context (s+ pre f)))))))
    (flet ((col (f) (cdr (assoc f c :test #'string=))))
      (db-lookup db +sr+ (mapcar #'col +sr-f+))
      (when left (db-lookup db +sra+ (cons (col "k") (mapcar #'col +sra-f+))))
      (db-zero db `(- ,(col "k") (+ ,kexpr 16))))
    c))

(defun cap-col (context db name rows value)
  "A layer constant VALUE as a constant public column of a per-row domain of ROWS rows."
  (db-column db name (make-array rows :initial-element value)
             (route-public context (lambda (point) (declare (ignore point)) (vector (base->ext (field value) 4))))))

(defun combine-blocks (context point values nb)
  "Per-block claims at POINT -> one claim at (POINT, tau), tau fresh over NB bits."
  (let ((tau (if (plusp nb) (context-challenge context nb) '())))
    (values (append point tau) (block-combination context tau values))))

;;; ------------------------------------------------------------------ the layer's parameters

(defun layer-parameters (model l)
  "Layer L's constants Kco, Kcd, Dyn (its first dynamic tag) and Dme (llm.layer_lparams), as the
alist DB-RUN substitutes."
  (let* ((cfg (model-cfg model)) (layer (aref (model-layers model) l)))
    (list (cons 'kco (layer-value layer "Kco"))
          (cons 'kcd (if (cfg-moe cfg) 0 (- (+ (layer-value layer "eWu") (model-value model "eA")
                                                (layer-value layer "eWd") (model-value model "f_h")))))
          (cons 'dyn (+ +dyn0+ (* l (cfg-nkv cfg))))
          (cons 'dme (+ +dyn0+ (* (cfg-l cfg) (cfg-nkv cfg)) l)))))

(defun layer-head (context model l)
  "Absorb the layer's index; its parameters hold until the next layer (Proto.layer_head)."
  (absorb (context-tr context) "layer|")
  (absorb (context-tr context) (le-bytes l 4))
  (setf *layer-parameters* (layer-parameters model l)))

;;; ------------------------------------------------------------------ the steps

(defstruct step-state
  "What the steps pass on within a layer (Proto attributes): the down projection's per-slice
claims, the gather targets and their per-slice partial sums, and the one-hot claims waiting for
their row check."
  down-claims gather gather-parts oh-pending)

(defun hexpr (s o)
  "The SatAdd output from the pre-saturation sum S and the flag O."
  `(+ (* (- 1 (* ,o ,o)) ,s)
      (* (+ (* ,o ,o) ,o) ,(field (* +inv2+ (1- (expt 2 23)))))
      (- (* (- (* ,o ,o) ,o) ,(field (* +inv2+ (expt 2 23)))))))

(defun satlookup (s o)
  `(+ (* (- 1 (* ,o ,o)) (+ ,s ,(expt 2 23)))
      (* (* ,o ,o) (- (* ,o ,s) ,(expt 2 23) (* (- (* ,o ,o) ,o) ,+inv2+)))))

(defun step-mo (context model shape l state &optional witness)
  "The MLP output (Proto.step_mo, dense layers): the down projection's shift and SatAdd over the
layer's elements, the per-row domain of G_mi, the opening of G_mo, and the down product's claims
split into per-slice partial sums."
  (let* ((cb (shape-cb shape)) (tt (shape-t* shape)) (kd (shape-kd shape)) (kf (shape-kf shape))
         (layer (aref (model-layers model) l))
         (p (format nil "~D." l))
         (w witness)
         (db (make-db context (s+ p "mo") (+ cb tt) :k kd)))
    (flet ((wm (key) (lambda () (rm-layout (getf w key) cb))))
      (let* ((acc (db-column db "accd" (wm :accd) (route-virtual context (list l "accd"))))
             (s2 (db-column db (s+ p "s2") (wm :s2) (route-block context (s+ p "s2"))))
             (o2 (db-column db (s+ p "os2") (wm :os2) (route-block context (s+ p "os2"))))
             (s1 (db-column db (s+ p "s1") (wm :s1) (route-block context (s+ p "s1"))))
             (o1 (db-column db (s+ p "os1") (wm :os1) (route-block context (s+ p "os1"))))
             (pack (and w (sr-pack (getf w :kd))))
             (sr (loop for f in (append (remove "k" +sr-f+ :test #'string=) +sra-f+)
                       collect (let ((f f))
                                 (cons f (db-column db (s+ p "sd." f) (lambda () (sr-field pack f))
                                                    (route-block context (s+ p "sd." f)) :kind :row :c cb :shared t)))))
             (c2 `(- ,s2 ,(hexpr s1 o1))))
        (flet ((sr (f) (cdr (assoc f sr :test #'string=))))
          (db-lookup db (sr "tR") (list `(- (+ ,acc ,(sr "R")) (* ,(sr "s") ,c2))))
          (db-lookup db (sr "tB") (list `(+ ,c2 ,(sr "oB"))))
          (db-lookup db (sr "tA") (list `(+ (* ,(sr "f") ,acc) ,(sr "oA"))))
          (db-lookup db 24 (list (satlookup s2 o2)))
          (db-zero db `(- (* ,o2 ,o2 ,o2) ,o2)))
        (db-run db)))
    ;; the per-row domain of G_mi
    (let ((db (make-db context (s+ p "mi.rows") tt)))
      (rq-rows context db (s+ p "r2.") (getf w :r2) kd :asel-key (list l "asel2") :shape-cb cb)
      (let* ((nsc (ns-rows context db (s+ p "n2.") (getf w :n2) (list l "sig2")))
             (eh (cdr (assoc "Eh" nsc :test #'string=)))
             (rows (expt 2 tt))
             (kgm (cap-col context db (s+ p "kgm") rows (layer-value layer "kgm")))
             (_ (db-lookup db 5 (list `(- ,kgm ,eh))))
             (kgl (cap-col context db (s+ p "kgl") rows (layer-value layer "kgl")))
             (eg (loop for f in '("E" "pE" "f")
                       collect (let ((f f))
                                 (cons f (db-column db (s+ p "g." f) (lambda () (row-field (pack-field (getf w :eg) f)))
                                                    (route-block context (s+ p "g." f))))))))
        (declare (ignore _))
        (flet ((eg (f) (cdr (assoc f eg :test #'string=))))
          (db-zero db `(* ,(eg "f") (- 1 ,(eg "f"))))
          (db-zero db `(- ,(eg "E") (* ,(eg "f") ,kgl) (* (- 1 ,(eg "f")) ,eh)))
          (db-lookup db 24 (list `(* (- 1 ,(eg "f")) (- ,eh ,kgl))))
          (db-lookup db 24 (list `(* ,(eg "f") (- ,kgl 1 ,eh))))
          (db-lookup db +pow+ `((+ ,(eg "E") 16) ,(eg "pE"))))
        (let ((ru (rq-rows context db (s+ p "ru.") (getf w :ru) kf :e0 2 :emax 22 :asel-key (list l "aselu") :shape-cb cb))
              (rmm (rq-rows context db (s+ p "rm.") (getf w :rm) kf :asel-key (list l "aselm") :shape-cb cb
                            :b (cfg-down-b (model-cfg model)))))
          (sr-rows context db (s+ p "sd.") (and w (getf w :kd))
                   `(- kcd (- (+ ,(cdr (assoc "e" rmm :test #'string=)) ,(cdr (assoc "e" ru :test #'string=))) ,eh))
                   :left t))
        (db-run db)))
    (when (active-p context)
      (open-group context (s+ p "mo"))
      ;; the down product's claims -> per-slice partial values
      (let* ((slices (mlp-slices shape))
             (d (cfg-d (model-cfg model)))
             (wd (layer-value layer "Wd")))
        (setf (step-state-down-claims state)
              ;; every claim's combination first, then every claim's partial sums
              (loop for (po pm v) in (loop for (point values) in (take-virtual context (list l "accd"))
                                           collect (multiple-value-bind (po v)
                                                       (combine-blocks context (subseq point 0 cb) values (nbits kd))
                                                     (list po (nthcdr cb point) v)))
                    collect (let ((parts (if (prover-p context)
                                             (send-values context (down-partials (getf (getf w :rm) :q) wd po pm slices cb d))
                                             (send-values context nil (list (length slices) 4)))))
                              (unless (prover-p context)
                                (unless (ext= v (reduce #'ext+ parts :initial-value (make-ext 4)))
                                  (reject :down-projection "down projection partial sums")))
                              (list po pm parts)))))
      ;; the gather targets
      (setf (step-state-gather state)
            (loop for key in '("aselu" "aselm")
                  collect (destructuring-bind ((point values)) (take-virtual context (list l key))
                            (list key point (aref values 0))))))
    state))

(defun down-partials (qm wd po pm slices cb d)
  "Per MLP slice s: sum over f in s of (sum_t eq(pm, t) qm[t, f]) (sum_c eq(po, c) Wd[f, c])."
  (let* ((eo (eq-table po 4))
         (et (eq-table pm 4))
         (dff (rows wd)))
    (coerce (loop for (b0 b1) in slices
                  collect (let ((acc (make-ext 4)))
                            (loop for f from (* b0 (expt 2 cb)) below (min dff (* b1 (expt 2 cb)))
                                  do (let ((wf (make-ext 4)) (af (make-ext 4)))
                                       (dotimes (c d) (setf wf (ext+ wf (ext-scale (aref eo c) (field (aref wd f c))))))
                                       (dotimes (tk (rows qm)) (setf af (ext+ af (ext-scale (aref et tk) (field (aref qm tk f))))))
                                       (setf acc (ext+ acc (ext* af wf)))))
                            acc))
            'vector)))

;;; ------------------------------------------------------------------ sub-protocols

(defun check-eq (context label final a b)
  "The verifier: every block's final of a shared eq(A, .) table equals eq(A, B)."
  (unless (prover-p context)
    (let ((e (reduce #'ext* (mapcar #'ext-eq1 a b) :initial-value (ext-one 4))))
      (loop for v across final
            unless (ext= v e) do (reject :sumcheck "~A: eq table final" label)))))

(defun col-dot (a e kin mrows width)
  "Per column x of A (rm layout: KIN blocks of MROWS rows of WIDTH): sum_m e[m] A[block][m][x]."
  (let ((out (make-array (* kin width))))
    (dotimes (blk kin out)
      (dotimes (x width)
        (let ((acc (make-ext 4)))
          (dotimes (m mrows)
            (let ((v (aref a (+ (* (+ (* blk mrows) m) width) x))))
              (unless (zerop v) (setf acc (ext+ acc (ext-scale (aref e m) v))))))
          (setf (aref out (+ (* blk width) x)) acc))))))

(defun row-dot (w e nin nout)
  "Per row r of the natural (NIN x NOUT) matrix W: sum_c e[c] W[r, c]."
  (let ((out (make-array nin)))
    (dotimes (r nin out)
      (let ((acc (make-ext 4)))
        (dotimes (c nout)
          (let ((v (aref w r c))) (unless (zerop v) (setf acc (ext+ acc (ext-scale (aref e c) (field v)))))))
        (setf (aref out r) acc)))))

(defun matmul (context label a-name a-data kin mrows w-nat kout cb claims weight-claim &key (cbo cb) defer-weights)
  "Claims (po pm value) on acc = A W, A row-major (MROWS x KIN 2^cb, A-DATA in rm layout) and W
natural (KIN 2^cb x KOUT 2^cbo), reduced by one sumcheck over the inner index to claims on A's
blocks and on W's blocks (tb, ob), in-block (o' low, t' high); WEIGHT-CLAIM receives those (point
values) (llm.matmul): within the loop, or after it with DEFER-WEIGHTS (Proto.matmul_rows and
matmul_cols claim a temporary tensor and move the claims to the weight tensor afterwards)."
  (when claims
    (let* ((nin (* kin (expt 2 cb))) (nout (* kout (expt 2 cbo)))
           (gam (context-challenge context (length claims)))
           (claim (claim-sum context gam (mapcar #'third claims)))
           (names (loop for i below (length claims)
                        collect (list (intern (format nil "A~D" i) '#:exacta.steps) (intern (format nil "W~D" i) '#:exacta.steps)
                                      (intern (format nil "G~D" i) '#:exacta.steps))))
           (statement (piece `(+ ,@(loop for (an wn gn) in names collect `(* ,gn ,an ,wn)))))
           (constants (loop for (nil nil gn) in names for g in gam collect (cons gn g)))
           (tables (loop for (po pm) in claims for (an wn) in names
                         append (if (prover-p context)
                                    (list (list an :full 0 nil (col-dot (funcall a-data) (eq-table pm 4) kin mrows (expt 2 cb)))
                                          (list wn :full 0 nil (row-dot w-nat (eq-table po 4) nin nout)))
                                    (list (list an :full 0 nil nil) (list wn :full 0 nil nil))))))
      (multiple-value-bind (rho finals) (run-sumcheck context label cb kin statement claim tables :constants constants)
        (loop with weight-claims = '()
              for (po pm) in claims
              for i from 0
              do (claim context a-name (append rho pm) (nth (* 2 i) finals))
                 (let* ((po-low (subseq po 0 cbo))
                        (vals (if (prover-p context)
                                  (send-values context (weight-block-values w-nat po-low rho kin kout cb cbo))
                                  (send-values context nil (list (* kin kout) 4))))
                        (tau (nthcdr cbo po)))
                   (unless (prover-p context)
                     (dotimes (tb kin)
                       (unless (ext= (block-combination context tau (subseq vals (* tb kout) (* (1+ tb) kout)))
                                     (aref (nth (1+ (* 2 i)) finals) tb))
                         (reject :matmul "matmul ~A: weight block values inconsistent" label))))
                   (if defer-weights
                       (push (list (append po-low rho) vals) weight-claims)
                       (funcall weight-claim (append po-low rho) vals)))
              finally (loop for (point vals) in (reverse weight-claims) do (funcall weight-claim point vals)))))))

(defun weight-block-values (w po-low rho kin kout cb cbo)
  "W's block (tb, ob) at (PO-LOW over o', RHO over t'), for every tb < KIN, ob < KOUT."
  (let ((eo (eq-table po-low 4)) (et (eq-table rho 4))
        (out (make-array (* kin kout))))
    (dotimes (tb kin out)
      (dotimes (ob kout)
        (let ((acc (make-ext 4)))
          (dotimes (tp (expt 2 cb))
            (let ((row (make-ext 4)))
              (dotimes (op (expt 2 cbo))
                (let ((v (aref w (+ (* tb (expt 2 cb)) tp) (+ (* ob (expt 2 cbo)) op))))
                  (unless (zerop v) (setf row (ext+ row (ext-scale (aref eo op) (field v)))))))
              (setf acc (ext+ acc (ext* row (aref et tp))))))
          (setf (aref out (+ (* tb kout) ob)) acc))))))

(defun oh-check (context label jb-name cb nbk claims t* data-bits)
  "Claims (point values) on OH(m, i) = eq(bits(j*_m), i) over a row-major (T x K 2^cb) domain,
reduced to claims on the bit planes JB-NAME (llm.oh_check)."
  (let ((taus (loop repeat (length claims) collect (context-challenge context nbk)))
        (nb (+ cb nbk)))
    (loop for (point values) in claims
          for tau in taus
          do (let* ((pj (subseq point 0 cb)) (pm (nthcdr cb point))
                    (claim (block-combination context tau values))
                    (xs (append pj tau))
                    (bits (loop for i below nb collect (intern (format nil "B~D" i) '#:exacta.steps)))
                    (cs (loop for i below nb
                              collect (list (intern (format nil "X~DA" i) '#:exacta.steps) (intern (format nil "X~DB" i) '#:exacta.steps))))
                    (statement `(* ,@(loop for b in bits for (ca cb*) in cs collect `(+ (* ,ca ,(piece b)) ,cb*))))
                    (constants (loop for x in xs for (ca cb*) in cs
                                     collect (cons ca (ext- (ext+ x x) (ext-one 4)))
                                     collect (cons cb* (ext- (ext-one 4) x))))
                    (tables (loop for b in bits for i from 0
                                  collect (list b :full 0 nil (and (prover-p context) (row-field (nth i (funcall data-bits))))))))
               (multiple-value-bind (rho finals)
                   (run-sumcheck context (s+ label "-oh") t* 1 statement claim tables :eq pm :constants constants)
                 (claim context jb-name rho (loop for f in finals for i from 0 collect (cons i (aref f 0)))))))))

(defun jstar (bits)
  "j* per row from its bit planes."
  (let ((out (make-array (length (first bits)) :initial-element 0)))
    (loop for plane in bits for i from 0
          do (dotimes (m (length plane)) (incf (aref out m) (ash (aref plane m) i))))
    out))

(defun oh-slice (context state jb-name rho values b0 ks kf cb bits)
  "One-hot claims on the slice's blocks: the prover sends OH on every block of the full row (KF
blocks), the verifier checks the slice's against VALUES; the rest waits for the row check."
  (let ((full (if (prover-p context)
                  (let* ((j (jstar bits))
                         (er (eq-table (nthcdr cb rho) 4)) (ec (eq-table (subseq rho 0 cb) 4))
                         (acc (make-array kf :initial-element (make-ext 4))))
                    (dotimes (m (length j))
                      (let ((k (ash (aref j m) (- cb))))
                        (setf (aref acc k) (ext+ (aref acc k) (ext* (aref er m) (aref ec (ldb (byte cb 0) (aref j m))))))))
                    (send-values context acc))
                  (send-values context nil (list kf 4)))))
    (unless (prover-p context)
      (dotimes (k ks)
        (unless (ext= (aref full (+ b0 k)) (aref values k))
          (reject :one-hot "one-hot slice values"))))
    (setf (step-state-oh-pending state)
          (append (step-state-oh-pending state) (list (list jb-name rho full))))))

(defun slice-columns (m b0 b1 cb)
  "Columns [b0 2^cb, b1 2^cb) of the matrix M."
  (let ((out (make-matrix (rows m) (* (- b1 b0) (expt 2 cb)))))
    (dotimes (i (rows m) out)
      (dotimes (j (columns out)) (setf (aref out i j) (aref m i (+ j (* b0 (expt 2 cb)))))))))

(defun slice-rows (m b0 b1 cb)
  (let ((out (make-matrix (* (- b1 b0) (expt 2 cb)) (columns m))))
    (dotimes (i (rows out) out)
      (dotimes (j (columns m)) (setf (aref out i j) (aref m (+ i (* b0 (expt 2 cb))) j))))))

(defun one-hot-table (j b0 ks tt cb)
  "1 at each row's j* within the slice's blocks, in rm layout (Ks blocks of T x 2^cb)."
  (let ((out (make-array (* ks tt (expt 2 cb)) :initial-element 0)))
    (dotimes (m tt out)
      (let ((blk (- (ash (aref j m) (- cb)) b0)))
        (when (and (>= blk 0) (< blk ks))
          (setf (aref out (+ (ash (+ (* blk tt) m) cb) (ldb (byte cb 0) (aref j m)))) 1))))))

(defun step-mlp (context model shape l state witness s)
  "MLP slice S (Proto.step_mlp): its domain (gate shift and SiLU, the up row's quantization, the
product and its Requant), the minimality gather sumcheck, the one-hot slice claims, the down, gate
and up products, and the opening of G_mlp<s>."
  (let* ((cb (shape-cb shape)) (tt (shape-t* shape)) (kd (shape-kd shape)) (kf (shape-kf shape))
         (layer (aref (model-layers model) l))
         (p (format nil "~D." l)) (q (format nil "~D.m~D." l s))
         (slices (mlp-slices shape))
         (w witness) (mlp (getf (getf w :forward) :mlp)))
    (destructuring-bind (b0 b1) (nth s slices)
      (let* ((ks (- b1 b0))
             (db (make-db context (s+ q "dom") (+ cb tt) :k ks))
             (eg-minus (model-value model "eG"))
             (kgc (map 'vector (lambda (e) (- eg-minus e)) (subseq (layer-value layer "eWg") (* b0 (expt 2 cb)) (* b1 (expt 2 cb))))))
        (witness (every (lambda (k) (and (>= k 0) (< k 31))) kgc) "gate column shift outside [0, 31)")
        (flet ((slice (m) (lambda () (rm-layout (slice-columns m b0 b1 cb) cb)))
               (rowc (name data) (db-column db name data (route-block context name) :kind :row :c cb :shared t)))
          (let* ((accg (db-column db "accg" (slice (getf mlp :accg)) (route-virtual context (list l s "accg"))))
                 (accu (db-column db "accu" (slice (getf mlp :accu)) (route-virtual context (list l s "accu"))))
                 (c (db-column db (s+ q "c") (slice (getf mlp :c)) (route-block context (s+ q "c"))))
                 (a (db-column db (s+ q "a") (slice (getf mlp :a)) (route-block context (s+ q "a"))))
                 (qu (db-column db (s+ q "qu") (slice (getf (getf w :ru) :q)) (route-block context (s+ q "qu"))))
                 (qm (db-column db (s+ q "qm") (slice (getf (getf w :rm) :q)) (route-block context (s+ q "qm"))))
                 (eg (rowc (s+ p "g.E") (lambda () (getf (getf w :eg) :e))))
                 (peg (rowc (s+ p "g.pE") (lambda () (getf (getf w :eg) :pe))))
                 (eu (rowc (s+ p "ru.e") (lambda () (getf (getf w :ru) :e))))
                 (su (rowc (s+ p "ru.s") (lambda () (getf (getf w :ru) :s))))
                 (em (rowc (s+ p "rm.e") (lambda () (getf (getf w :rm) :e))))
                 (sm (rowc (s+ p "rm.s") (lambda () (getf (getf w :rm) :s))))
                 (kg (db-column db "kgc" kgc (route-public context (lambda (pt) (column-mles kgc ks cb pt))) :kind :col :c cb))
                 (pkv (map 'vector #'pow2* kgc))
                 (pk (db-column db "pkg" pkv (route-public context (lambda (pt) (column-mles pkv ks cb pt))) :kind :col :c cb))
                 (sg `(* ,pk ,peg)))
            (db-lookup db `(+ ,kg ,eg) (list `(- (+ ,accg (* ,sg ,+inv2+)) (* ,sg ,c))))
            (db-lookup db exacta.tables:+silu+ (list `(+ ,c ,(expt 2 14)) a))
            (db-lookup db eu (list `(- (+ ,accu (* ,su ,+inv2+)) (* ,su ,qu))))
            (db-lookup db exacta.tables:+star8+ (list `(+ ,qu 127)))
            (db-lookup db `(+ ,em 1) (list `(+ (- (* 2 ,a ,qu) (* 2 ,sm ,qm)) ,sm)))
            (let ((bd (cfg-down-b (model-cfg model))))
              (if (= bd 127)
                  (db-lookup db exacta.tables:+star8+ (list `(+ ,qm 127)))
                  (let ((nb (1- (integer-length (1+ bd)))))
                    (db-lookup db (1+ nb) (list `(+ ,qm ,bd)))
                    (db-lookup db (1+ nb) (list `(- ,bd ,qm))))))
            (db-run db)))
        (when (active-p context)
          (let* ((gather (step-state-gather state))
                 (pu (second (assoc "aselu" gather :test #'string=)))
                 (nsl (length slices)))
            ;; the gathers' per-slice partial sums, once per layer
            (when (= s 0)
              (setf (step-state-gather-parts state)
                    (loop for (key pt v) in gather
                          collect (cons key
                                        (let ((parts (if (prover-p context)
                                                         (send-values context (gather-partials w key pt slices cb))
                                                         (send-values context nil (list nsl 4)))))
                                          (unless (prover-p context)
                                            (unless (ext= v (reduce #'ext+ parts :initial-value (make-ext 4)))
                                              (reject :gather "gather partial sums")))
                                          parts)))))
            (let* ((gam (context-challenge context))
                   (parts (step-state-gather-parts state))
                   (claim (if (prover-p context)
                              (make-ext 4)
                              (ext+ (aref (cdr (assoc "aselu" parts :test #'string=)) s)
                                    (ext* gam (aref (cdr (assoc "aselm" parts :test #'string=)) s)))))
                   (statement `(* ,(piece 't0) (+ ,(piece '(* t1 t3)) (* c0 ,(piece '(* t2 t4 t5))))))
                   (prover (prover-p context))
                   (tables (list (list 't0 :row cb t (and prover (eq-table pu 4)))
                                 (list 't1 :full 0 nil (and prover (one-hot-table (jstar (getf (getf w :ru) :jb)) b0 ks (expt 2 tt) cb)))
                                 (list 't2 :full 0 nil (and prover (one-hot-table (jstar (getf (getf w :rm) :jb)) b0 ks (expt 2 tt) cb)))
                                 (list 't3 :full 0 nil (and prover (rm-layout (slice-columns (getf mlp :accu) b0 b1 cb) cb)))
                                 (list 't4 :full 0 nil (and prover (rm-layout (slice-columns (getf mlp :a) b0 b1 cb) cb)))
                                 (list 't5 :full 0 nil (and prover (rm-layout (slice-columns (getf (getf w :ru) :q) b0 b1 cb) cb))))))
              (multiple-value-bind (rho finals)
                  (run-sumcheck context "mlp-gather" (+ cb tt) ks statement claim tables :constants `((c0 . ,gam)))
                (check-eq context "mlp-gather" (first finals) pu (nthcdr cb rho))
                (put-virtual context (list l s "accu") (list rho (nth 3 finals)))
                (claim context (s+ q "a") rho (nth 4 finals))
                (claim context (s+ q "qu") rho (nth 5 finals))
                (loop for (jbn idx key) in (list (list (s+ p "ru.jb") 1 :ru) (list (s+ p "rm.jb") 2 :rm))
                      do (oh-slice context state jbn rho (nth idx finals) b0 ks kf cb (getf (getf w key) :jb)))))
            ;; the down projection's partial over this slice
            (let ((wd (and (prover-p context) (layer-value layer "Wd")))
                  (wg (and (prover-p context) (layer-value layer "Wg")))
                  (wu (and (prover-p context) (layer-value layer "Wu"))))
              (matmul context "down" (s+ q "qm") (lambda () (rm-layout (slice-columns (getf (getf w :rm) :q) b0 b1 cb) cb))
                      ks (expt 2 tt) (and wd (slice-rows wd b0 b1 cb)) kd cb
                      (loop for (po pm parts) in (step-state-down-claims state) collect (list po pm (aref parts s)))
                      (lambda (point vals)
                        (claim context (s+ p "Wd") point
                               (loop for v across vals for b from 0
                                     collect (cons (+ (* (+ b0 (floor b kd)) kd) (mod b kd)) v))))
                      :defer-weights t)
              ;; the gate and up products
              (flet ((combined (key)
                       (loop for (point values) in (take-virtual context (list l s key))
                             collect (multiple-value-bind (po v) (combine-blocks context (subseq point 0 cb) values (nbits ks))
                                       (list po (nthcdr cb point) v)))))
                (let ((clg (combined "accg")) (clu (combined "accu")))
                  (loop for (label wname wnat cl) in (list (list "gate" "Wg" wg clg) (list "up" "Wu" wu clu))
                        do (let ((wname (s+ p wname)))
                             (matmul context label (s+ p "qp2") (lambda () (rm-layout (getf (getf w :n2) :qp) cb))
                                     kd (expt 2 tt) (and wnat (slice-columns wnat b0 b1 cb)) ks cb cl
                                     (lambda (point vals)
                                       (claim context wname point
                                              (loop for v across vals for b from 0
                                                    collect (cons (+ (* (floor b ks) kf) b0 (mod b ks)) v))))
                                     :defer-weights t)))))))
          (open-group context (format nil "~D.mlp~D" l s)))))
    state))

(defun column-mles (values ks cb point)
  "Per block k < KS: the MLE of VALUES[k 2^cb ..] at POINT."
  (coerce (loop for k below ks
                collect (mle-eval (row-field (subseq values (* k (expt 2 cb)) (* (1+ k) (expt 2 cb)))) point 4))
          'vector))

(defun gather-partials (w key pt slices cb)
  "Per slice: sum_m eq(PT, m) X[m, j*_m] [j*_m's block in the slice], X = accu (aselu) or m (aselm)."
  (let* ((pack (getf w (if (string= key "aselu") :ru :rm)))
         (j (jstar (getf pack :jb)))
         (x (getf (getf (getf w :forward) :mlp) (if (string= key "aselu") :accu :m)))
         (e (eq-table pt 4)))
    (coerce (loop for (c0 c1) in slices
                  collect (let ((acc (make-ext 4)))
                            (dotimes (m (length j) acc)
                              (let ((blk (ash (aref j m) (- cb))))
                                (when (and (>= blk c0) (< blk c1))
                                  (setf acc (ext+ acc (ext-scale (aref e m) (field (aref x m (aref j m)))))))))))
            'vector)))

;;; ------------------------------------------------------------------ the MLP input

(defun flush-oh (context state shape l witness)
  "The row checks of the slices' one-hot claims of layer L (Proto.flush_oh), one per kind."
  (let ((pending (step-state-oh-pending state))
        (prefix (format nil "~D." l)))
    (dolist (jbn (remove-duplicates (mapcar #'first pending) :test #'string= :from-end t))
      (let ((mine (loop for (name rho full) in pending
                        when (and (string= name jbn) (eql 0 (search prefix name))) collect (list rho full))))
        (when mine
          (let ((key (if (search ".ru." jbn) :ru :rm)))
            (oh-check context "mlp-oh" jbn (shape-cb shape) (nbits (shape-kf shape)) mine (shape-t* shape)
                      (lambda () (getf (getf witness key) :jb))))
          (setf pending (remove-if (lambda (entry) (and (string= (first entry) jbn) (eql 0 (search prefix jbn))))
                                   pending)))))
    (setf (step-state-oh-pending state) pending)))

(defun rq-norm-dom (context db h qn qpn rpre npre r n cb)
  "Requant8 + NormScale lookups over a row-major domain of the rows' elements (Proto.rq_norm_dom)."
  (flet ((rowc (name data) (db-column db name data (route-block context name) :kind :row :c cb :shared t)))
    (let* ((q (db-column db qn (lambda () (rm-layout (getf r :q) cb)) (route-block context qn)))
           (qp (db-column db qpn (lambda () (rm-layout (getf n :qp) cb)) (route-block context qpn)))
           (e (rowc (s+ rpre "e") (lambda () (getf r :e))))
           (s (rowc (s+ rpre "s") (lambda () (getf r :s))))
           (mu (rowc (s+ npre "mu") (lambda () (getf n :mu)))))
      (db-lookup db `(+ ,e 1) (list `(+ (- (* 2 ,h) (* 2 ,s ,q)) ,s)))
      (db-lookup db exacta.tables:+star8+ (list `(+ ,q 127)))
      (db-lookup db 15 (list `(- (+ (* ,q ,mu) ,(expt 2 14)) (* ,(expt 2 15) ,qp))))
      (db-lookup db exacta.tables:+star8+ (list `(+ ,qp 127)))
      (values q qp))))

(defun gather-rowsum (context label n-rows rowpt k cols h q asel sig jb-name bits cb)
  "The gather and row-sum sumcheck over a row-major (rows x K 2^cb) domain (Proto.gather_rowsum):
sum_{m,x} eq(r, m) [OH(m, x) h(m, x) + gamma q(m, x)^2] = asel(r) + gamma (sig(r) - 1).  COLS:
(name data route) per column, H a function of their symbols giving the gathered form, Q the index
of the squared column or NIL; BITS a function giving the j* bit planes."
  (let* ((n (+ cb n-rows))
         (gam (context-challenge context))
         (prover (prover-p context))
         (claim (if prover
                    (make-ext 4)
                    (if sig (ext+ asel (ext* gam (ext- sig (ext-one 4)))) asel)))
         (symbols (loop for i below (length cols) collect (intern (format nil "G~D" i) '#:exacta.steps)))
         (gathered (piece `(* oh ,(funcall h symbols))))
         (statement (if q
                        `(* ,(piece 'eqr) (+ ,gathered (* c0 ,(piece `(* ,(nth q symbols) ,(nth q symbols))))))
                        `(* ,(piece 'eqr) ,gathered)))
         (tables (append (list (list 'eqr :row cb t (and prover (eq-table rowpt 4)))
                               (list 'oh :full 0 nil (and prover (gather-one-hot (jstar (funcall bits)) k n-rows cb))))
                         (loop for (nil data) in cols for g in symbols
                               collect (list g :full 0 nil (and prover (row-field (funcall data))))))))
    (multiple-value-bind (rho finals)
        (run-sumcheck context label n k statement claim tables :constants (and q `((c0 . ,gam))))
      (check-eq context label (first finals) rowpt (nthcdr cb rho))
      (loop for (nil nil route) in cols for f in (nthcdr 2 finals)
            do (funcall route rho f))
      (oh-check context label jb-name cb (nbits k) (list (list rho (second finals))) n-rows bits))))

(defun gather-one-hot (j k n-rows cb)
  "1 at each row's j*, in rm layout (K blocks of 2^n-rows rows of 2^cb)."
  (let ((out (make-array (* k (expt 2 (+ n-rows cb))) :initial-element 0)))
    (dotimes (m (expt 2 n-rows) out)
      (setf (aref out (+ (ash (+ (ash (ash (aref j m) (- cb)) n-rows) m) cb) (ldb (byte cb 0) (aref j m)))) 1))))

(defun step-mi (context model shape l state witness)
  "The MLP input (Proto.step_mi): the slices' one-hot row checks; the Requant8 + NormScale of the
residual h_mid (the attention's SatAdd output) over the layer's elements; the gather and row-sum
sumcheck of its minimality and squared sum; the opening of G_mi."
  (declare (ignore model))
  (let* ((cb (shape-cb shape)) (tt (shape-t* shape)) (kd (shape-kd shape))
         (p (format nil "~D." l))
         (w witness))
    (flush-oh context state shape l witness)
    (let ((db (make-db context (s+ p "mi") (+ cb tt) :k kd)))
      (let* ((s1 (db-column db (s+ p "s1") (lambda () (rm-layout (getf w :s1) cb)) (route-block context (s+ p "s1"))))
             (o1 (db-column db (s+ p "os1") (lambda () (rm-layout (getf w :os1) cb)) (route-block context (s+ p "os1")))))
        (rq-norm-dom context db (hexpr s1 o1) (s+ p "q2") (s+ p "qp2") (s+ p "r2.") (s+ p "n2.") (getf w :r2) (getf w :n2) cb))
      (db-run db))
    (when (active-p context)
      (destructuring-bind ((pa va)) (take-virtual context (list l "asel2"))
        (destructuring-bind ((ps vs)) (take-virtual context (list l "sig2"))
          (declare (ignore ps))
          (gather-rowsum context "mi-g" tt pa kd
                         (list (list (s+ p "s1") (lambda () (rm-layout (getf w :s1) cb)) (route-block context (s+ p "s1")))
                               (list (s+ p "os1") (lambda () (rm-layout (getf w :os1) cb)) (route-block context (s+ p "os1")))
                               (list (s+ p "q2") (lambda () (rm-layout (getf (getf w :r2) :q) cb)) (route-block context (s+ p "q2"))))
                         (lambda (c) (hexpr (first c) (second c))) 2 (aref va 0) (aref vs 0)
                         (s+ p "r2.jb") (lambda () (getf (getf w :r2) :jb)) cb)))
      (open-group context (s+ p "mi")))
    state))

;;; ------------------------------------------------------------------ domain batches

(defmacro with-domain-batch (() &body body)
  "Domains run in BODY are proved together when it ends (llm.domain_batch): those of one shape (n,
K, merge) as one merged domain, in order of first appearance.  BODY must not read their claims."
  `(let ((*domain-batch* (make-array 0 :adjustable t :fill-pointer t)))
     (multiple-value-prog1 (progn ,@body)
       (run-domain-batch *domain-batch*))))

(defun run-domain-batch (batch)
  (let ((*domain-batch* nil)
        (groups '()))
    (loop for db across batch
          do (let* ((key (list (db-n db) (db-k db) (db-merge db)))
                    (group (assoc key groups :test #'equal)))
               (if group
                   (setf (cdr group) (append (cdr group) (list db)))
                   (setf groups (append groups (list (list key db)))))))
    (loop for (nil . dbs) in groups
          do (run-db (if (rest dbs) (merge-dbs dbs) (first dbs))))))

(defun merge-dbs (dbs)
  "One domain of the columns, constraints and lookups of DBS (llm.merge_domains): column names
prefixed by the domain's position, routes following them."
  (let* ((first (first dbs))
         (merged (%make-db (db-context first) (format nil "~{~A~^|~}" (mapcar #'db-name dbs))
                           (db-n first) (db-k first) (db-merge first))))
    (loop for db in dbs
          for j from 0
          do (let ((renames (loop for (symbol) in (db-columns db)
                                  collect (cons symbol (column-symbol (format nil "~D:~A" j (symbol-name symbol)))))))
               (setf (db-columns merged)
                     (append (db-columns merged)
                             (loop for (symbol . options) in (db-columns db)
                                   collect (cons (cdr (assoc symbol renames)) options)))
                     (db-routes merged)
                     (append (db-routes merged)
                             (loop for (symbol . route) in (db-routes db)
                                   collect (cons (cdr (assoc symbol renames)) route)))
                     (db-lookups merged)
                     (append (db-lookups merged)
                             (loop for lookup in (db-lookups db)
                                   collect (destructuring-bind (&key tag values num dynamic) lookup
                                             (list :tag (sublis renames tag) :values (sublis renames values)
                                                   :num (and num (sublis renames num)) :dynamic dynamic))))
                     (db-constraints merged)
                     (append (db-constraints merged) (sublis renames (db-constraints db))))))
    merged))

;;; ------------------------------------------------------------------ the attention output

(defun hin-cols (context db l w cb &key pre post)
  "The layer's residual input as a form over columns of DB (Proto.hin_cols): the embedding rows
(a virtual column) in layer 0, else the previous layer's SatAdd values."
  (if (zerop l)
      (db-column db "h0" (lambda () (rm-layout (getf w :hin) cb)) (route-virtual context "h0"))
      (let ((q (format nil "~D." (1- l))))
        (hexpr (db-column db (s+ q "s2") (lambda () (rm-layout (getf (getf w :prev) :s2) cb))
                          (route-block context (s+ q "s2") :pre pre :post post))
               (db-column db (s+ q "os2") (lambda () (rm-layout (getf (getf w :prev) :os2) cb))
                          (route-block context (s+ q "os2") :pre pre :post post))))))

(defun head-rows (m)
  "An (nh x T) matrix's rows, head-major, as one vector."
  (let ((out (make-array (array-total-size m))))
    (dotimes (i (array-total-size m) out) (setf (aref out i) (row-major-aref m i)))))

(defun step-ao (context model shape l state witness)
  "The attention output (Proto.step_ao): the O-projection's shift and SatAdd, the per-row Requant8
of o and SR(k_o), Requant8(o) over its elements, o = MulRound(N, recip, k) elementwise, the per
(row, head) denominator Norm and shifts -- one domain batch -- then o's point claim, the numerator
and denominator claims for the attention groups, the gather of Requant8(o), the O-projection
product and the opening of G_ao."
  (let* ((layer (aref (model-layers model) l))
         (cb (shape-cb shape)) (tt (shape-t* shape)) (kd (shape-kd shape)) (kq (shape-kq shape))
         (ldh (shape-ldh shape)) (nh (shape-nh shape)) (nkv (shape-nkv shape)) (grp (shape-grp shape))
         (p (format nil "~D." l))
         (w witness)
         (nd (getf w :nd)))
    (with-domain-batch ()
      ;; the O-projection's shift + SatAdd over T x d
      (let* ((db (make-db context (s+ p "oproj") (+ cb tt) :k kd))
             (acc (db-column db "acco" (lambda () (rm-layout (getf w :acc-o) cb)) (route-virtual context (list l "acco"))))
             (s1 (db-column db (s+ p "s1") (lambda () (rm-layout (getf w :s1) cb)) (route-block context (s+ p "s1"))))
             (o1 (db-column db (s+ p "os1") (lambda () (rm-layout (getf w :os1) cb)) (route-block context (s+ p "os1"))))
             (h (hin-cols context db l w cb))
             (pack (and w (sr-pack (getf w :ko))))
             (sr (loop for f in (append (remove "k" +sr-f+ :test #'string=) +sra-f+)
                       collect (let ((f f))
                                 (cons f (db-column db (s+ p "so." f) (lambda () (sr-field pack f))
                                                    (route-block context (s+ p "so." f)) :kind :row :c cb :shared t)))))
             (c1 `(- ,s1 ,h)))
        (flet ((sr (f) (cdr (assoc f sr :test #'string=))))
          (db-lookup db (sr "tR") (list `(- (+ ,acc ,(sr "R")) (* ,(sr "s") ,c1))))
          (db-lookup db (sr "tB") (list `(+ ,c1 ,(sr "oB"))))
          (db-lookup db (sr "tA") (list `(+ (* ,(sr "f") ,acc) ,(sr "oA"))))
          (db-lookup db 24 (list (satlookup s1 o1)))
          (db-zero db `(- (* ,o1 ,o1 ,o1) ,o1)))
        (db-run db))
      ;; per row: Requant8(o) minimality + SR(k_o)
      (let* ((db (make-db context (s+ p "ao.rows") tt))
             (ro (rq-rows context db (s+ p "ro.") (getf w :ro) kq :asel-key (list l "aselo") :shape-cb cb)))
        (sr-rows context db (s+ p "so.") (getf w :ko) `(- kco ,(cdr (assoc "e" ro :test #'string=))) :left t)
        (db-run db))
      ;; Requant8(o) over T x nq
      (let* ((db (make-db context (s+ p "rqo") (+ cb tt) :k kq))
             (o (db-column db (s+ p "o") (lambda () (rm-layout (getf w :o) cb)) (route-block context (s+ p "o"))))
             (qo (db-column db (s+ p "qo") (lambda () (rm-layout (getf (getf w :ro) :q) cb)) (route-block context (s+ p "qo"))))
             (e (db-column db (s+ p "ro.e") (lambda () (getf (getf w :ro) :e)) (route-block context (s+ p "ro.e")) :kind :row :c cb :shared t))
             (s (db-column db (s+ p "ro.s") (lambda () (getf (getf w :ro) :s)) (route-block context (s+ p "ro.s")) :kind :row :c cb :shared t)))
        (db-lookup db `(+ ,e 1) (list `(+ (- (* 2 ,o) (* 2 ,s ,qo)) ,s)))
        (db-lookup db exacta.tables:+star8+ (list `(+ ,qo 127)))
        (db-run db))
      ;; o = MulRound(N, recip, k) elementwise over (c, m) x nh
      (let* ((db (make-db context (s+ p "nd") (+ ldh tt) :k nh))
             (n (db-column db "Nsum" (lambda () (rm-layout (getf w :ns) ldh)) (route-virtual context (list l "Nsum"))))
             (xh (db-column db (s+ p "nd.xh") (lambda () (rm-layout (getf w :ndxh) ldh)) (route-block context (s+ p "nd.xh"))))
             (tt* (db-column db (s+ p "nd.t") (lambda () (rm-layout (getf w :ndt) ldh)) (route-block context (s+ p "nd.t"))))
             (ov (db-column db "o_view" (lambda () (rm-layout (getf w :o) ldh)) (route-virtual context (list l "o_view"))))
             (kp-pack (and w (sr-pack (head-rows (getf nd :kp)))))
             (sm (loop for f in (remove "k" +sr-f+ :test #'string=)
                       collect (let ((f f))
                                 (cons f (db-column db (s+ p "nM." f) (lambda () (sr-field kp-pack f))
                                                    (route-block context (s+ p "nM." f)) :kind :row :c ldh)))))
             (rc (db-column db (s+ p "nd.recip") (lambda () (head-rows (getf nd :recip)))
                            (route-block context (s+ p "nd.recip")) :kind :row :c ldh))
             (xl `(- ,n (* ,(expt 2 12) ,xh))))
        (flet ((sm (f) (cdr (assoc f sm :test #'string=))))
          (db-lookup db 12 (list xl))
          (db-lookup db 12 (list `(+ ,xh ,(expt 2 11))))
          (db-lookup db 12 (list `(- (* ,xl ,rc) (* ,(expt 2 12) ,tt*))))
          (db-lookup db 18 (list `(+ ,tt* ,(expt 2 17))))
          (db-lookup db (sm "tR") (list `(- (+ (* ,xh ,rc) ,tt* ,(sm "R")) (* ,(sm "s") ,ov))))
          (db-lookup db (sm "tB") (list `(+ ,ov ,(sm "oB")))))
        (db-run db))
      ;; per (row, head): D = 2^E delta + rhoD, recip, k_m = max(7, E), the MulRound shift
      (let* ((db (make-db context (s+ p "nd.rows") tt :k nh))
             (dc (db-column db "D" (lambda () (head-rows (getf nd :d))) (route-virtual context (list l "D"))))
             (e-pack (and w (sr-pack (head-rows (getf nd :e)))))
             (kp-pack (and w (sr-pack (head-rows (getf nd :kp)))))
             (km-pack (and w (sr-pack (head-rows (getf nd :km)))))
             (sre (loop for f in +sr-f+
                        collect (let ((f f)) (cons f (db-column db (s+ p "nE." f) (lambda () (sr-field e-pack f)) (route-block context (s+ p "nE." f)))))))
             (srm (loop for f in +sr-f+
                        collect (let ((f f)) (cons f (db-column db (s+ p "nM." f) (lambda () (sr-field kp-pack f)) (route-block context (s+ p "nM." f)))))))
             (nc (loop for (f key) in '(("delta" :delta) ("rhoD" :rhod) ("recip" :recip) ("fk" :fk))
                       collect (let ((key key))
                                 (cons f (db-column db (s+ p "nd." f) (lambda () (head-rows (getf nd key))) (route-block context (s+ p "nd." f)))))))
             (kmc (db-column db "nKk" (lambda () (sr-field km-pack "k"))
                             (lambda (point value)
                               (dotimes (g nkv)
                                 (claim context (format nil "~D.g~D.nK.k" l g) point
                                        (loop for i below grp collect (cons i (aref value (+ (* g grp) i)))))))))
             (vsh (layer-value layer "vshift"))
             (vs (db-column db "vshift" (subseq vsh 0 nh)
                            (route-public context (lambda (point) (declare (ignore point))
                                                    (map 'vector (lambda (v) (base->ext (field v) 4)) (subseq vsh 0 nh))))
                            :kind :row :c tt)))
        (flet ((sre (f) (cdr (assoc f sre :test #'string=)))
               (srm (f) (cdr (assoc f srm :test #'string=)))
               (nc (f) (cdr (assoc f nc :test #'string=))))
          (let ((e `(- ,(sre "k") 16))
                (km `(- ,kmc 16))
                (fk (nc "fk")))
            (db-lookup db exacta.tables:+sr+ (mapcar #'sre +sr-f+))
            (db-lookup db exacta.tables:+sr+ (mapcar #'srm +sr-f+))
            (db-lookup db (sre "tR") (list (nc "rhoD")))
            (db-lookup db exacta.tables:+recip+ (list `(- ,(nc "delta") ,(expt 2 15)) (nc "recip")))
            (db-zero db `(- ,dc (* ,(sre "s") ,(nc "delta")) ,(nc "rhoD")))
            (db-zero db `(* ,fk (- 1 ,fk)))
            (db-lookup db 4 (list `(+ (* ,fk (- ,e 7)) (* (- 1 ,fk) (- 6 ,e)))))
            (db-lookup db 5 (list `(+ ,e 5)))
            (db-lookup db 5 (list `(- 14 ,e)))
            (db-lookup db 22 (list `(* (- 1 ,fk) ,dc)))
            (db-zero db `(- ,km 7 (* ,fk (- ,e 7))))
            (db-zero db `(- (- ,(srm "k") 16) (+ 13 (- ,e ,km) ,vs)))))
        (db-run db)))
    (when (active-p context)
      ;; o's view claims -> a point claim on o (head block hh -> o's columns hh dh + c)
      (loop for (point values) in (take-virtual context (list l "o_view"))
            do (let* ((ptc (subseq point 0 ldh)) (ptm (nthcdr ldh point))
                      (tau (context-challenge context (nbits nh)))
                      (v (block-combination context tau values))
                      (lh (log2* (shape-hb shape))))
                 (point-claim context (s+ p "o") (append ptc (subseq tau 0 lh) ptm (nthcdr lh tau))
                              :value v :data (and (prover-p context) (rm-layout (getf w :o) cb)))))
      ;; Nsum -> Nr, summed over the key blocks
      (loop for (point values) in (take-virtual context (list l "Nsum"))
            do (let* ((nbb (shape-nbb shape))
                      (half (loop repeat nbb collect (base->ext +inv2+ 4)))
                      (scale (fpow +inv2+ nbb)))
                 (dotimes (g nkv)
                   (claim context (format nil "~D.g~D.Nr" l g) (append (subseq point 0 ldh) half (nthcdr ldh point))
                          (loop for i below grp collect (cons i (ext-scale (aref values (+ (* g grp) i)) scale)))))))
      ;; D -> per group, for the attention steps
      (loop for (point values) in (take-virtual context (list l "D"))
            do (dotimes (g nkv)
                 (put-virtual context (list l g "D") (list point (subseq values (* g grp) (* (1+ g) grp))))))
      ;; the gather of Requant8(o)
      (destructuring-bind ((pa va)) (take-virtual context (list l "aselo"))
        (gather-rowsum context "ao-g" tt pa kq
                       (list (list (s+ p "o") (lambda () (rm-layout (getf w :o) cb)) (route-block context (s+ p "o"))))
                       #'first nil (aref va 0) nil (s+ p "ro.jb") (lambda () (getf (getf w :ro) :jb)) cb))
      ;; the O-projection product
      (let ((cl (loop for (point values) in (take-virtual context (list l "acco"))
                      collect (multiple-value-bind (po v) (combine-blocks context (subseq point 0 cb) values (nbits kd))
                                (list po (nthcdr cb point) v)))))
        (matmul context "oproj" (s+ p "qo") (lambda () (rm-layout (getf (getf w :ro) :q) cb)) kq (expt 2 tt)
                (and (prover-p context) (layer-value layer "Wo")) kd cb cl
                (lambda (point vals) (claim context (s+ p "Wo") point vals))))
      (open-group context (s+ p "ao")))
    state))

(defun log2* (x) (1- (integer-length x)))
