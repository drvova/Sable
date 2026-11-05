;;;; model.lisp -- the integer function F_O of a Qwen2-style transformer (zk/model.py): the model
;;;; file, the gadgets, one layer's forward pass and the final logits.
;;;;
;;;; Every gadget is the exact integer definition the constraint system proves, and returns its
;;;; output and the witness values the prover commits (a plist).  A range check that fails signals
;;;; WITNESS-ERROR: the input is outside F_O's domain.  Conventions (fixed by the weight commitment):
;;;;
;;;;   residual h: f_h fractional bits, h in A = [-2^23, 2^23)
;;;;   matmul inputs: int8 rows with a per-row exponent (Requant8 + NormScale);
;;;;   weights int8 with a per-output-column exponent (per tensor for W_up)
;;;;   Q, K, V: static per-head exponents, bias in output units; RoPE in the 8-bit domain
;;;;   P = tau_exp(acc - M) in [0, 2^10]; D = 2^E delta + rho (signed Norm); N in aligned key
;;;;   blocks of 2^12, each shifted by k_m = max(7, E); o = MulRound(N, floor(2^32 / delta),
;;;;   25 + E - k_m + vshift_h)
;;;;   MLP gate: SiLU by table on c = round(acc / 2^k); up: dynamic row exponent; m = a q_up
;;;;   logits: z = round(acc / 2^k) in units 2^-8
;;;;
;;;; Tensors are matrices (rows x columns) of fixnums; per-row values are vectors.
;;;;
;;;; The model file (EXACTFO1): magic | u32 count | entry*, each entry u16 name length | name
;;;; (UTF-8) | u8 width (1, 2, 4 or 8 bytes) | u8 rank | u64 dimension * rank | the values, signed
;;;; little-endian integers of that width, row-major.  Configuration fields are "cfg.<field>"
;;;; (booleans 0/1, MoE as cfg.moe.E, .k, .de), the model's values are named as in Sable (f_h, Kz,
;;;; E_attn, eG, eA, eWlm, emb, Wlm), layer l's as "layers.<l>.<name>".

(defpackage #:exacta.model
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.llm)
  (:import-from #:exacta.tables #:exp-values #:silu-values #:rope-tables)
  (:export #:model #:read-model #:model-cfg #:model-value #:model-layers #:layer-value
           #:matrix #:make-matrix #:rows #:columns
           #:requant8 #:shiftq8 #:requant-b #:normscale #:imm #:shiftq8s #:qkv-out #:rope8
           #:shiftround #:gshiftround #:satadd #:norm-den #:mulround #:norm-div #:attention-rows
           #:mlp-out #:embed-tokens #:layer-forward #:final-logits #:forward #:model-tables))

(in-package #:exacta.model)

(defconstant +a-bits+ 23)
(defconstant +keyblk+ 12)
(defconstant +recip-num+ (expt 2 32))
(defconstant +o-base+ 25)
(defconstant +km-min+ 7)
(defconstant +rsq-c+ (* 127 (expt 2 15)))
(defconstant +silu-cmax+ (expt 2 14))
(defconstant +kmin-sr+ -16)
(defconstant +silu-kmax+ 15)
(defconstant +qkv-kmax+ 22)

;;; ------------------------------------------------------------------ matrices

(deftype matrix () '(simple-array fixnum (* *)))

(defun make-matrix (rows columns &optional (initial 0))
  (make-array (list rows columns) :element-type 'fixnum :initial-element initial))
(defun rows (m) (array-dimension m 0))
(defun columns (m) (array-dimension m 1))
(defun make-row-vector (n) (make-array n :element-type 'fixnum :initial-element 0))

(defmacro do-matrix ((i j m) &body body)
  `(dotimes (,i (rows ,m)) (dotimes (,j (columns ,m)) ,@body)))

(defun map-matrix (f m)
  (let ((out (make-matrix (rows m) (columns m))))
    (do-matrix (i j m) (setf (aref out i j) (funcall f (aref m i j) i j)))
    out))

(defun pow2 (e) (ash 1 e))

;;; ------------------------------------------------------------------ the model file

(defstruct (model (:constructor %make-model (cfg values layers)))
  "A quantized model: its configuration, its values (an EQUAL hash table name -> integer, vector
or matrix) and per layer the same."
  cfg values layers)

(defun model-value (model name)
  (multiple-value-bind (v found) (gethash name (model-values model))
    (unless found (error 'invalid-parameter :name 'model-value :value name :expected "a value of the model"))
    v))

(defun layer-value (layer name &optional (default nil default-p))
  (multiple-value-bind (v found) (gethash name layer)
    (cond (found v)
          (default-p default)
          (t (error 'invalid-parameter :name 'layer-value :value name :expected "a value of the layer")))))

(defun read-entries (octets)
  "The entries of an EXACTFO1 file, an alist name -> integer (rank 0), vector or matrix."
  (let ((position 0))
    (flet ((take (n)
             (when (> (+ position n) (length octets)) (error 'malformed-proof :reason :truncated :offset position))
             (prog1 (subseq octets position (+ position n)) (incf position n)))
           (int (n) (prog1 (le-integer octets position n) (incf position n))))
      (unless (equalp (take 8) (to-octets "EXACTFO1"))
        (error 'malformed-proof :reason :magic :offset 0))
      (loop repeat (int 4)
            collect (let* ((name (map 'string #'code-char (take (int 2))))
                           (width (int 1))
                           (rank (int 1))
                           (dims (loop repeat rank collect (int 8)))
                           (count (reduce #'* dims))
                           (values (make-array count :element-type 'fixnum)))
                      (dotimes (i count)
                        (let ((u (int width)))
                          (setf (aref values i) (if (logbitp (1- (* 8 width)) u) (- u (ash 1 (* 8 width))) u))))
                      (cons name (case rank
                                   (0 (aref values 0))
                                   (1 values)
                                   (2 (let ((m (make-matrix (first dims) (second dims))))
                                        (dotimes (i count m) (setf (row-major-aref m i) (aref values i)))))
                                   (t (error 'invalid-parameter :name 'rank :value rank :expected "0, 1 or 2")))))
            finally (unless (= position (length octets))
                      (error 'malformed-proof :reason :trailing-bytes :offset position))))))

(defun read-model (path)
  "Read a model file (EXACTFO1)."
  (let* ((entries (read-entries (with-open-file (in path :element-type '(unsigned-byte 8))
                                  (let ((v (make-array (file-length in) :element-type '(unsigned-byte 8))))
                                    (read-sequence v in) v))))
         (values (make-hash-table :test 'equal)))
    (flet ((cfg (name &optional default)
             (let ((e (assoc (concatenate 'string "cfg." name) entries :test #'string=)))
               (if e (cdr e) default))))
      (let* ((l (cfg "L"))
             (cfg (make-cfg :l l :d (cfg "d") :nh (cfg "nh") :nkv (cfg "nkv") :dh (cfg "dh") :dff (cfg "dff")
                            :v (cfg "V") :theta (cfg "theta") :qk-norm (= 1 (cfg "qk_norm"))
                            :qkv-bias (= 1 (cfg "qkv_bias")) :down-b (cfg "down_b")
                            :moe (and (cfg "moe.E") (list :e (cfg "moe.E") :k (cfg "moe.k") :de (cfg "moe.de")))))
             (layers (coerce (loop repeat l collect (make-hash-table :test 'equal)) 'vector)))
        (loop for (name . v) in entries
              do (cond ((and (> (length name) 4) (string= name "cfg." :end1 4)))
                       ((and (> (length name) 7) (string= name "layers." :end1 7))
                        (let* ((dot (position #\. name :start 7)))
                          (setf (gethash (subseq name (1+ dot)) (aref layers (parse-integer name :start 7 :end dot)))
                                v)))
                       (t (setf (gethash name values) v))))
        (%make-model cfg values layers)))))

;;; ------------------------------------------------------------------ gadgets

(defun requant8 (a &optional (emin 0) (emax 17))
  "Row quantization (Requant8; ShiftQ8 with EMIN 2, EMAX 22): per row the least e in [EMIN, EMAX]
with 2a in [-255 2^e, 255 2^e), and q = round(a / 2^e) (ties up), |q| <= 127.  Witness: jstar, an
entry that does not fit at e - 1 (when e > EMIN), its value asel and side sigma, and f = [e > EMIN]."
  (declare (type matrix a))
  (let* ((n (rows a)) (m (columns a))
         (q (make-matrix n m))
         (e (make-row-vector n)) (jstar (make-row-vector n)) (sigma (make-row-vector n))
         (f (make-row-vector n)) (asel (make-row-vector n)))
    (dotimes (i n)
      (let ((hi (aref a i 0)) (lo (aref a i 0)))
        (dotimes (j m) (setf hi (max hi (aref a i j)) lo (min lo (aref a i j))))
        (let ((ei emax))
          (loop for c from emax downto emin
                when (and (< (* 2 hi) (* 255 (pow2 c))) (>= (* 2 lo) (* -255 (pow2 c))))
                  do (setf ei c))
          (let ((s (pow2 ei)))
            (dotimes (j m)
              (let ((v (floor (+ (* 2 (aref a i j)) s) (* 2 s))))
                (witness (<= (abs v) 127) "requant: |q| > 127")
                (setf (aref q i j) v)))
            (setf (aref e i) ei
                  (aref f i) (if (> ei emin) 1 0))
            (when (> ei emin)
              (let ((js (or (loop for j below m
                                  when (or (>= (* 4 (aref a i j)) (* 255 s)) (< (* 4 (aref a i j)) (* -255 s)))
                                    return j)
                            (error 'witness-error :check "requant: no witness column"))))
                (setf (aref jstar i) js
                      (aref asel i) (aref a i js)
                      (aref sigma i) (if (>= (* 4 (aref a i js)) (* 255 s)) 1 0))))
            (when (= ei emin) (setf (aref asel i) (aref a i 0)))))))
    (values q e (list :jstar jstar :sigma sigma :f f :asel asel))))

(defun shiftq8 (acc &optional (emin 2) (emax 22)) (requant8 acc emin emax))

(defun requant-b (a b &optional (emax 17))
  "The least e in [0, EMAX] with every entry in [-(B + 1/2) 2^e, (B + 1/2) 2^e); q = round(a / 2^e)."
  (declare (type matrix a))
  (let* ((n (rows a)) (m (columns a)) (q (make-matrix n m)) (e (make-row-vector n)))
    (dotimes (i n)
      (let ((mx (loop for j below m maximize (max (1+ (* 2 (aref a i j))) (- (* 2 (aref a i j)))))))
        (setf (aref e i) (loop for c below emax count (< (* (1+ (* 2 b)) (pow2 c)) mx)))
        (let ((s (pow2 (aref e i))))
          (dotimes (j m)
            (let ((v (floor (+ (* 2 (aref a i j)) s) (* 2 s))))
              (witness (<= (abs v) b) "requant: |q| > B")
              (setf (aref q i j) v))))))
    (values q e nil)))

(defun signed-norm (x e)
  "Signed Norm: 2^(E-) X = 2^(E+) delta + rho with rho in [0, 2^(E+))."
  (let ((ep (max e 0)) (en (max (- e) 0)))
    (values (ash (ash x (- ep)) en) (- x (ash (ash x (- ep)) ep)) ep en)))

(defun normscale (q)
  "Sigma = sum q^2 + 1 = 2^E delta + rho per row, E = 2 Eh even, delta in [2^14, 2^16); q' =
round(q mu / 2^15) with mu = floor(127 2^15 / sqrt(delta)), |q'| <= 127."
  (declare (type matrix q))
  (let* ((n (rows q)) (m (columns q))
         (qp (make-matrix n m))
         (eh (make-row-vector n))
         (witness (list :sig (make-row-vector n) :delta (make-row-vector n) :rho (make-row-vector n)
                        :mu (make-row-vector n) :ep (make-row-vector n) :en (make-row-vector n))))
    (dotimes (i n)
      (let* ((sig (1+ (loop for j below m sum (* (aref q i j) (aref q i j)))))
             (e (floor (- (integer-length sig) 15) 2)))
        (witness (< sig (expt 2 40)) "bitlen: input >= 2^40")
        (multiple-value-bind (delta rho ep en) (signed-norm sig (* 2 e))
          (witness (and (>= delta (expt 2 14)) (< delta (expt 2 16))) "normscale: mantissa range")
          (let ((mu (isqrt (floor (* +rsq-c+ +rsq-c+) delta))))
            (dotimes (j m)
              (let ((v (floor (+ (* (aref q i j) mu) (expt 2 14)) (expt 2 15))))
                (witness (<= (abs v) 127) "normscale: |q'| > 127")
                (setf (aref qp i j) v)))
            (setf (aref eh i) e)
            (loop for (key value) on (list :sig sig :delta delta :rho rho :mu mu :ep ep :en en) by #'cddr
                  do (setf (aref (getf witness key) i) value))))))
    (values qp eh witness)))

(defun imm (a w)
  "The exact product A W."
  (declare (type matrix a w) (optimize speed))
  (let* ((n (rows a)) (k (columns a)) (m (columns w))
         (out (make-matrix n m)))
    (declare (type matrix out))
    (dotimes (i n out)
      (dotimes (x k)
        (let ((v (aref a i x)))
          (declare (type (signed-byte 40) v))
          (unless (zerop v)
            (dotimes (j m)
              (setf (aref out i j) (the fixnum (+ (aref out i j) (the fixnum (* v (the (signed-byte 20) (aref w x j))))))))))))))

(defun shiftq8s (acc k beta)
  "ShiftQ8S: r = round(acc / 2^k) + beta (ties up), q = clamp(r, -127, 127), o = sign(r - q), and
w = o (acc + 2^k beta) - 255 2^(k-1) - [o = -1] in [0, 2^30) when saturated.  K and BETA are
functions of (row, column)."
  (let* ((n (rows acc)) (m (columns acc))
         (q (make-matrix n m)) (o (make-matrix n m)) (w (make-matrix n m)))
    (do-matrix (i j acc)
      (let* ((kk (funcall k i j))
             (b (funcall beta i j))
             (s (pow2 kk))
             (r (+ (floor (+ (aref acc i j) (floor s 2)) s) b))
             (qq (max -127 (min 127 r)))
             (oo (signum (- r qq)))
             (ww (if (zerop oo) 0 (- (* oo (+ (aref acc i j) (* s b))) (* 255 (floor s 2)) (if (= oo -1) 1 0)))))
        (witness (<= 0 kk) "shiftq8s: shift out of range")
        (witness (and (<= 0 ww) (< ww (expt 2 30))) "shiftq8s: w out of range")
        (setf (aref q i j) qq (aref o i j) oo (aref w i j) ww)))
    (values q (list :o o :w w))))

(defun qkv-out (cfg layer acc eh)
  "The Q, K and V rows (8-bit) from the fused accumulator ACC of rows with NormScale exponents EH.
Without QK-norm: ShiftQ8S by kc + Eh with the bias.  With QK-norm, each Q and K head row goes
through ShiftQ8, NormScale (exponent E2) and ShiftQ8S of q' G by kn + E2 (no bias)."
  (let* ((dh (cfg-dh cfg)) (nq (* (cfg-nh cfg) dh)) (nk (* (cfg-nkv cfg) dh))
         (kc (layer-value layer "kc")) (bias (layer-value layer "bias"))
         (r (rows acc)) (out (make-matrix r (columns acc))))
    (if (not (cfg-qk-norm cfg))
        (values (shiftq8s acc (lambda (i j) (+ (aref kc j) (aref eh i))) (lambda (i j) (declare (ignore i)) (aref bias j))))
        (let* ((heads (floor (+ nq nk) dh))
               (a (make-matrix (* r heads) dh)))
          (do-matrix (i j a)
            (setf (aref a i j) (aref acc (floor i heads) (+ (* (mod i heads) dh) j))))
          (let* ((q8 (shiftq8 a))
                 (gqk (layer-value layer "Gqk"))
                 (kn (layer-value layer "kn")))
            (multiple-value-bind (qp e2) (normscale q8)
              (let ((prod (make-matrix r (+ nq nk))))
                (do-matrix (i j prod)
                  (setf (aref prod i j) (* (aref qp (+ (* i heads) (floor j dh)) (mod j dh)) (aref gqk j))))
                (let ((qk (shiftq8s prod
                                    (lambda (i j)
                                      (let ((k (+ (aref kn j) (aref e2 (+ (* i heads) (floor j dh))))))
                                        (witness (<= k +qkv-kmax+) "F_O: QK-norm shift above QKV_KMAX")
                                        (witness (>= k -31) "F_O: QK-norm shift below -31")
                                        (max k 1)))
                                    (constantly 0)))
                      (vacc (make-matrix r (- (columns acc) nq nk))))
                  (do-matrix (i j vacc) (setf (aref vacc i j) (aref acc i (+ nq nk j))))
                  (let ((v (shiftq8s vacc (lambda (i j) (+ (aref kc (+ nq nk j)) (aref eh i)))
                                     (lambda (i j) (declare (ignore i)) (aref bias (+ nq nk j))))))
                    (do-matrix (i j out)
                      (setf (aref out i j) (if (< j (+ nq nk)) (aref qk i j) (aref v i (- j nq nk))))))
                  out))))))))

(defun rope8 (q1 q2 cos sin)
  "RoPE in the 8-bit domain: (round((q1 cos - q2 sin) / 2^15), round((q2 cos + q1 sin) / 2^15))."
  (values (floor (+ (- (* q1 cos) (* q2 sin)) (expt 2 14)) (expt 2 15))
          (floor (+ (* q2 cos) (* q1 sin) (expt 2 14)) (expt 2 15))))

(defun shiftround (acc k)
  "round(acc / 2^k), ties up."
  (let ((s (pow2 k))) (floor (+ acc (floor s 2)) s)))

(defun gshiftround (acc k)
  "GShiftRound per row: shift K[row] in [-16, 24] (K <= 0 an exact left shift); the output must lie
in A."
  (let ((out (make-matrix (rows acc) (columns acc))))
    (do-matrix (i j acc)
      (let* ((kk (aref k i))
             (c (if (> kk 0) (floor (+ (aref acc i j) (ash (pow2 kk) -1)) (pow2 kk)) (* (aref acc i j) (pow2 (- kk))))))
        (witness (and (>= kk +kmin-sr+) (<= kk 24)) "gshiftround: shift out of range")
        (witness (< (abs c) (expt 2 +a-bits+)) "gshiftround: output outside A")
        (setf (aref out i j) c)))
    out))

(defun satadd (h c)
  "h + c saturated to A; witness o = sign(h + c - result)."
  (let ((out (make-matrix (rows h) (columns h))) (o (make-matrix (rows h) (columns h))))
    (do-matrix (i j h)
      (let* ((s (+ (aref h i j) (aref c i j)))
             (hp (max (- (expt 2 +a-bits+)) (min (1- (expt 2 +a-bits+)) s))))
        (setf (aref out i j) hp (aref o i j) (signum (- s hp)))))
    (values out (list :o o))))

(defun norm-den (d)
  "The softmax denominator D in [2^10, 2^30]: D = 2^E delta + rhoD, delta in [2^15, 2^16), E in
[-5, 14]; block shift k_m = max(7, E).  A plist of the row's values."
  (let ((e (- (integer-length d) 16)))
    (witness (< d (expt 2 40)) "bitlen: input >= 2^40")
    (multiple-value-bind (delta rho) (signed-norm d e)
      (witness (and (>= delta (expt 2 15)) (< delta (expt 2 16))) "norm_den: mantissa range")
      (witness (and (>= e -5) (<= e 14)) "norm_den: exponent range")
      (list :e e :delta delta :rhod rho :km (max e +km-min+) :fk (if (>= e +km-min+) 1 0)))))

(defun mulround (x y k)
  "round(x y / 2^k) for x in A, |y| <= 2^17, k in [13, 30], without forming x y."
  (let* ((xh (floor x (expt 2 12)))
         (xl (- x (* xh (expt 2 12))))
         (tt (floor (* xl y) (expt 2 12)))
         (hh (+ (* xh y) tt))
         (kp (- k 12)))
    (values (floor (+ hh (ash (pow2 kp) -1)) (pow2 kp)) (list :xh xh :t tt :kp kp))))

(defun norm-div (n nd vshift)
  "o = MulRound(N, floor(2^32 / delta), 25 + E - k_m + VSHIFT) per entry of the row N; also each
entry's MulRound witness."
  (let ((recip (floor +recip-num+ (getf nd :delta)))
        (k (+ +o-base+ (getf nd :e) (- (getf nd :km)) vshift))
        (splits (make-array (length n))))
    (witness (and (>= k 13) (<= k 30)) "norm_div: shift range")
    (values (let ((out (make-array (length n))))
              (dotimes (i (length n) out)
                (let ((x (aref n i)))
                  (witness (< (abs x) (expt 2 +a-bits+)) "norm_div: N outside A")
                  (multiple-value-bind (o split) (mulround x recip k)
                    (setf (aref out i) o (aref splits i) split)))))
            splits)))

(defun attention-rows (rq rk v texp vshift)
  "Causal attention of the query rows RQ (S x dh) over the keys RK and values V of the same
positions: scores acc = rq rk^T, P = tau_exp(max - acc) on j <= i, the denominator's Norm, the
numerator N = sum over key blocks of 2^12 of round(P V / 2^km), o = NormDiv(N).  Returns o (S x dh)
and the witness: per row D and its Norm (:d :e :delta :rhod :km :fk), the reciprocal and MulRound's
shift (:recip :kp); per entry N and MulRound's split (:n :xh :t, S x dh)."
  (declare (type matrix rq rk v) (type (simple-array (unsigned-byte 32) (*)) texp))
  (let* ((s (rows rq)) (dh (columns rq))
         (out (make-matrix s dh))
         (acc (make-array s :element-type 'fixnum))
         (p (make-array s :element-type 'fixnum))
         (w (list :d (make-row-vector s) :e (make-row-vector s) :delta (make-row-vector s) :rhod (make-row-vector s)
                  :km (make-row-vector s) :fk (make-row-vector s) :recip (make-row-vector s) :kp (make-row-vector s)
                  :n (make-matrix s dh) :xh (make-matrix s dh) :t (make-matrix s dh))))
    (dotimes (i s (values out w))
      (let ((mx most-negative-fixnum))
        (dotimes (j (1+ i))
          (let ((a 0))
            (declare (type fixnum a))
            (dotimes (x dh) (incf a (* (aref rq i x) (aref rk j x))))
            (setf (aref acc j) a mx (max mx a))))
        (let ((d 0))
          (dotimes (j (1+ i))
            (let ((pj (aref texp (min (- mx (aref acc j)) (1- (expt 2 23))))))
              (setf (aref p j) pj)
              (incf d pj)))
          (let* ((nd (norm-den d))
                 (km (getf nd :km))
                 (n (make-array dh :initial-element 0)))
            (loop for j0 from 0 to i by (expt 2 +keyblk+)
                  do (dotimes (x dh)
                       (incf (aref n x)
                             (shiftround (loop for j from j0 to (min i (+ j0 (expt 2 +keyblk+) -1))
                                               sum (* (aref p j) (aref v j x)))
                                         km))))
            (multiple-value-bind (o splits) (norm-div n nd vshift)
              (dotimes (x dh)
                (setf (aref out i x) (aref o x)
                      (aref (getf w :n) i x) (aref n x)
                      (aref (getf w :xh) i x) (getf (aref splits x) :xh)
                      (aref (getf w :t) i x) (getf (aref splits x) :t)))
              (setf (aref (getf w :d) i) d
                    (aref (getf w :recip) i) (floor +recip-num+ (getf nd :delta))
                    (aref (getf w :kp) i) (getf (aref splits 0) :kp))
              (loop for key in '(:e :delta :rhod :km :fk)
                    do (setf (aref (getf w key) i) (getf nd key))))))))))

;;; ------------------------------------------------------------------ the layer

(defun model-tables (model s)
  "The tables F_O reads (computed once per model and S): tau_exp per layer exponent, tau_SiLU and
the RoPE tables; a plist."
  (let ((cfg (model-cfg model)))
    (multiple-value-bind (cos sin) (rope-tables (cfg-theta cfg) (cfg-dh cfg) s)
      (list :exp (let ((tables (make-hash-table)))
                   (loop for layer across (model-layers model)
                         for e = (layer-value layer "E")
                         unless (gethash e tables) do (setf (gethash e tables) (exp-values e (cfg-dh cfg))))
                   tables)
            :silu (silu-values (model-value model "eG") (model-value model "eA"))
            :cos cos :sin sin))))

(defun embed-tokens (model tokens)
  "The embedding rows of TOKENS (a vector)."
  (let* ((emb (model-value model "emb"))
         (out (make-matrix (length tokens) (columns emb))))
    (do-matrix (i j out) (setf (aref out i j) (aref emb (aref tokens i) j)))
    out))

(defun mlp-out (model layer qp eh tables)
  "The MLP's contribution to the residual from the normalized rows QP with exponents EH (dense
layers): c = ShiftRound(qp W_g, eG - eW_g + max(Eh, K_gl)), a = tau_SiLU(c), q_up = ShiftQ8(qp
W_u), m = a q_up, requantized, and GShiftRound(qm W_d, Kcd - (e_m + e_up - Eh)).  A plist of the
values, the output under :Y."
  (let* ((cfg (model-cfg model))
         (eg (model-value model "eG")) (ewg (layer-value layer "eWg")) (kgl (layer-value layer "kgl"))
         (accg (imm qp (layer-value layer "Wg")))
         (silu (getf tables :silu))
         (c (make-matrix (rows accg) (columns accg)))
         (a (make-matrix (rows accg) (columns accg))))
    (when (cfg-moe cfg) (error "mlp-out: MoE layers are not ported yet"))
    (do-matrix (i j accg)
      (let ((kg (+ (- eg (aref ewg j)) (max (aref eh i) kgl))))
        (witness (and (>= kg 1) (<= kg +silu-kmax+)) "F_O: gate shift outside [1, SILU_KMAX]")
        (let ((cc (shiftround (aref accg i j) kg)))
          (witness (<= (abs cc) +silu-cmax+) "F_O: SiLU input beyond SILU_CMAX")
          (setf (aref c i j) cc (aref a i j) (aref silu (+ cc +silu-cmax+))))))
    (let ((accu (imm qp (layer-value layer "Wu"))))
      (multiple-value-bind (qu eup) (shiftq8 accu)
        (let ((m (make-matrix (rows a) (columns a))))
          (do-matrix (i j m) (setf (aref m i j) (* (aref a i j) (aref qu i j))))
          (multiple-value-bind (qm em) (if (= (cfg-down-b cfg) 127) (requant8 m) (requant-b m (cfg-down-b cfg)))
            (let* ((accd (imm qm (layer-value layer "Wd")))
                   (kcd (- (+ (layer-value layer "eWu") (model-value model "eA") (layer-value layer "eWd")
                              (model-value model "f_h"))))
                   (kd (map 'vector (lambda (em eup eh) (- kcd (- (+ em eup) eh))) em eup eh)))
              (list :accg accg :c c :a a :accu accu :qu qu :eup eup :m m :qm qm :em em :accd accd :kd kd
                    :y (gshiftround accd kd)))))))))

(defun layer-forward (model layer h b s tables)
  "One layer on the rows H (B S x d) of B sequences of S positions.  Returns the new rows and a
plist of every intermediate value (the layer's witness)."
  (let* ((cfg (model-cfg model))
         (nh (cfg-nh cfg)) (nkv (cfg-nkv cfg)) (dh (cfg-dh cfg))
         (nq (* nh dh)) (nk (* nkv dh)) (half (floor dh 2))
         (cos (getf tables :cos)) (sin (getf tables :sin))
         (w '()))
    (flet ((note (&rest kv) (setf w (append w kv))))
      (multiple-value-bind (q e wq) (requant8 h)
        (note :q q :e e :wq wq)
        (multiple-value-bind (qp eh wn) (normscale q)
          (note :qp qp :eh eh :wn wn)
          (witness (every (lambda (x) (<= x (layer-value layer "kqs"))) eh) "F_O: row exponent beyond the ShiftQ8S bound")
          (let* ((acc (imm qp (layer-value layer "Wqkv")))
                 (qkv (qkv-out cfg layer acc eh))
                 (rq (make-array (list b nh s dh)))
                 (rk (make-array (list b nkv s dh))))
            (note :acc-qkv acc :qkv qkv)
            ;; RoPE on the Q and K heads
            (dotimes (bb b)
              (dotimes (pos s)
                (let ((row (+ (* bb s) pos)))
                  (flet ((rotate (target head base)
                           (dotimes (x half)
                             (multiple-value-bind (r1 r2)
                                 (rope8 (aref qkv row (+ base (* head dh) x)) (aref qkv row (+ base (* head dh) half x))
                                        (aref cos pos x) (aref sin pos x))
                               (setf (aref target bb head pos x) r1 (aref target bb head pos (+ half x)) r2)))))
                    (dotimes (hh nh) (rotate rq hh 0))
                    (dotimes (g nkv) (rotate rk g nq))))))
            (note :rq rq :rk rk)
            (let ((o (make-matrix (* b s) nq))
                  (ns (make-matrix (* b s) nq)) (ndxh (make-matrix (* b s) nq)) (ndt (make-matrix (* b s) nq))
                  (nd (loop for key in '(:d :e :delta :rhod :km :fk :recip :kp)
                            collect key collect (make-matrix nh (* b s))))
                  (texp (gethash (layer-value layer "E") (getf tables :exp)))
                  (grp (floor nh nkv))
                  (vshift (layer-value layer "vshift")))
              (dotimes (bb b)
                (dotimes (hh nh)
                  (let ((g (floor hh grp))
                        (mq (make-matrix s dh)) (mk (make-matrix s dh)) (mv (make-matrix s dh)))
                    (dotimes (pos s)
                      (dotimes (x dh)
                        (setf (aref mq pos x) (aref rq bb hh pos x)
                              (aref mk pos x) (aref rk bb g pos x)
                              (aref mv pos x) (aref qkv (+ (* bb s) pos) (+ nq nk (* g dh) x)))))
                    (multiple-value-bind (oh aw) (attention-rows mq mk mv texp (aref vshift hh))
                      (dotimes (pos s)
                        (let ((row (+ (* bb s) pos)))
                          (dotimes (x dh)
                            (setf (aref o row (+ (* hh dh) x)) (aref oh pos x)
                                  (aref ns row (+ (* hh dh) x)) (aref (getf aw :n) pos x)
                                  (aref ndxh row (+ (* hh dh) x)) (aref (getf aw :xh) pos x)
                                  (aref ndt row (+ (* hh dh) x)) (aref (getf aw :t) pos x)))
                          (loop for key in '(:d :e :delta :rhod :km :fk :recip :kp)
                                do (setf (aref (getf nd key) hh row) (aref (getf aw key) pos)))))))))
              (note :o o :ns ns :ndxh ndxh :ndt ndt :nd nd)
              (multiple-value-bind (qo eo) (requant8 o)
                (let* ((acco (imm qo (layer-value layer "Wo")))
                       (co (gshiftround acco (map 'vector (lambda (e) (- (layer-value layer "Kco") e)) eo)))
                       (hmid (satadd h co)))
                  (note :qo qo :eo eo :acc-o acco :c-o co :h-mid hmid)
                  (multiple-value-bind (q2 e2) (requant8 hmid)
                    (declare (ignore e2))
                    (multiple-value-bind (qp2 eh2) (normscale q2)
                      (let ((mlp (mlp-out model layer qp2 eh2 tables)))
                        (note :q2 q2 :qp2 qp2 :eh2 eh2 :mlp mlp)
                        (let ((hout (satadd hmid (getf mlp :y))))
                          (note :h hout)
                          (values hout w))))))))))))))

(defun final-logits (model h)
  "z = GShiftRound(qp W_lm, Kz + Eh) in units of 2^-8."
  (let ((q (requant8 h)))
    (multiple-value-bind (qp eh) (normscale q)
      (gshiftround (imm qp (model-value model "Wlm"))
                   (map 'vector (lambda (e) (+ (model-value model "Kz") e)) eh)))))

(defun forward (model tokens b s &key (tables (model-tables model s)))
  "Teacher-forced F_O over TOKENS (B S of them): the final rows and the logits."
  (let ((h (embed-tokens model tokens)))
    (loop for layer across (model-layers model)
          do (setf h (layer-forward model layer h b s tables)))
    (values h (final-logits model h))))
