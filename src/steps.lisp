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
           #:layer-parameters #:layer-head #:step-mo #:step-state))

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
  (let ((rows (map 'list (lambda (x) (append (sr-row x) (rest (sra-row x)))) k)))
    (loop for name in '(:k :s :r :tr :tb :ob :ta :oa :f)
          for i from 0
          collect name
          collect (coerce (loop for row in rows collect (nth i row)) 'vector))))

(defun sr-field (pack name) (row-field (getf pack (intern (string-upcase name) :keyword))))

;;; ------------------------------------------------------------------ the layer witness

(defun sign-of (x) (signum x))

(defun layer-witness (model l h shape tables)
  "Layer L's witness on its input rows H (llm.LayerWit, the values the steps read so far)."
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
              :rm (rq-pack (getf mlp :m) cb))))))

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

(defun db-run (db)
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
claims and the gather targets."
  down-claims gather)

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
