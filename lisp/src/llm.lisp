;;;; llm.lisp -- the outer layer of the transformer protocol (zk/llm.py): the model configuration,
;;;; the proof shape, the commitment layout (Proto.declare) as data, the public policy, the
;;;; statement that seeds the transcript, the sampling seed and the weight commitment C_W.
;;;;
;;;; The protocol's steps (Proto.run: embedding, the layers' domains, the final rows, sampling,
;;;; tables) are not ported yet: they need a real proof to be checked against, which only the GPU
;;;; prover makes (tests/gpu-trace.py records one, with every sub-protocol's inputs and outputs).

(defpackage #:sable.llm
  (:use #:cl #:sable.util #:sable.conditions #:sable.sha256)
  (:export #:cfg #:make-cfg #:cfg-repr #:shape #:make-shape #:column-block-bits
           #:commitment-layout #:declare-units
           #:policy #:make-policy #:encode-policy #:check-statement
           #:statement #:sampling-seed #:weights-digest #:+theta-count+))

(in-package #:sable.llm)

;;; ------------------------------------------------------------------ constants (zk/llm.py, model.py)

(defconstant +tile+ 7 "Attention tiles are 2^7 x 2^7; the head dimension is 2^7.")
(defconstant +keyblk+ 12)
(defconstant +theta-count+ 11 "The temperatures of model.THETA (index 0: greedy).")
(defconstant +table-rows+ 26 "tables.LOG_ROWS: the LogUp table has 2^26 rows.")
(defconstant +table-values+ 6 "tables.NV: values per table row (beside the tag).")
(defconstant +mlp-slice+ 9)
(defconstant +pack-abs+ (expt 2 20))
(defparameter *pack-pad* 0.1d0 "proto.PACK_PAD: packing may pad by this fraction of the data.")
(defparameter *weight-pack-pad* 0.15d0 "llm.W_PACK_PAD, for the weight groups.")
(defparameter *p-pad* 0.1d0 "llm.P_PAD: the attention probabilities' column padding.")

(defparameter +sr-f+ '("k" "s" "R" "tR" "tB" "oB"))
(defparameter +srx-f+ (append +sr-f+ '("tA" "oA" "f")))
(defparameter +rq-f+ '("e" "s" "f" "sig" "wl" "wh"))
(defparameter +qk-f+ '("kq" "pkq" "fq"))
(defparameter +eg-f+ '("E" "pE" "f"))
(defparameter +ns-f+ '("Eh" "Ep" "pEp" "pEn" "pEh" "delta" "rho" "mu" "ng"))
(defparameter +nd-f+ '("delta" "rhoD" "recip" "fk"))
(defparameter +samp-f+ '("Z" "A" "P" "z0" "z1" "z2" "q0" "q1" "q2" "w0" "w1" "w2"
                         "k0" "k1" "k2" "k3" "k4" "d1l" "d1h" "d2l" "d2h"))

(defun nbits (k) (max 0 (integer-length (1- k))))

(defun log2 (x)
  (check-parameter x (lambda (x) (and (integerp x) (plusp x) (= x (expt 2 (1- (integer-length x))))))
                   "a power of two")
  (1- (integer-length x)))

;;; ------------------------------------------------------------------ the model configuration

(defstruct (cfg (:constructor make-cfg (&key l d nh nkv dh dff v (theta 1000000) qk-norm (qkv-bias t)
                                             moe (down-b 127))))
  "A model configuration (model.Cfg); MOE a plist (:e experts :k active :de width) or NIL."
  l d nh nkv dh dff v theta qk-norm qkv-bias moe down-b)

(defun python-bool (x) (if x "True" "False"))

(defun python-float (x)
  "Python's repr of an integer-valued float."
  (check-parameter x (lambda (x) (and (realp x) (= x (round x)))) "an integer-valued theta")
  (format nil "~D.0" (round x)))

(defun cfg-repr (cfg)
  "Python's repr of vars(cfg), which the circuit digest hashes."
  (format nil "{'L': ~D, 'd': ~D, 'nh': ~D, 'nkv': ~D, 'dh': ~D, 'dff': ~D, 'V': ~D, 'theta': ~A, ~
               'qk_norm': ~A, 'qkv_bias': ~A, 'moe': ~A, 'down_b': ~D}"
          (cfg-l cfg) (cfg-d cfg) (cfg-nh cfg) (cfg-nkv cfg) (cfg-dh cfg) (cfg-dff cfg) (cfg-v cfg)
          (python-float (cfg-theta cfg)) (python-bool (cfg-qk-norm cfg)) (python-bool (cfg-qkv-bias cfg))
          (let ((moe (cfg-moe cfg)))
            (if moe
                (format nil "{'E': ~D, 'k': ~D, 'de': ~D}" (getf moe :e) (getf moe :k) (getf moe :de))
                "None"))
          (cfg-down-b cfg)))

(defun column-block-bits (cfg &key (cb-max 10) (cbv-max 9))
  "The column-block sizes: the largest cb <= CB-MAX dividing every split width, the largest
cbv <= CBV-MAX dividing V (llm.block_bits)."
  (let ((widths (list (cfg-d cfg) (* (cfg-nh cfg) (cfg-dh cfg)) (* (cfg-nkv cfg) (cfg-dh cfg)) (cfg-dff cfg))))
    (values (loop for c from +tile+ to cb-max
                  when (every (lambda (w) (zerop (mod w (expt 2 c)))) widths) maximize c)
            (loop for c from 0 to cbv-max
                  when (zerop (mod (cfg-v cfg) (expt 2 c))) maximize c))))

;;; ------------------------------------------------------------------ the proof shape

(defstruct (shape (:constructor %make-shape))
  "The proof shape (llm.Shp): B prompts of S tokens, ngen generated, column blocks of 2^cb."
  cfg b s tt t* ls lb cb cbv dh ldh nh nkv grp hb kd kq kk kf kv nt1 ltile ntile nbb ng lng r)

(defun make-shape (cfg b s cb cbv ngen)
  (let ((dh (cfg-dh cfg)))
    (check-parameter dh (lambda (dh) (= dh (expt 2 +tile+))) "2^7")
    (let ((nt1 (ash s (- +tile+))))
      (%make-shape :cfg cfg :b b :s s :tt (* b s) :t* (log2 (* b s)) :ls (log2 s) :lb (log2 b)
                   :cb cb :cbv cbv :dh dh :ldh (log2 dh) :nh (cfg-nh cfg) :nkv (cfg-nkv cfg)
                   :grp (floor (cfg-nh cfg) (cfg-nkv cfg)) :hb (floor (expt 2 cb) dh)
                   :kd (ash (cfg-d cfg) (- cb)) :kq (ash (* (cfg-nh cfg) dh) (- cb))
                   :kk (ash (* (cfg-nkv cfg) dh) (- cb)) :kf (ash (cfg-dff cfg) (- cb))
                   :kv (ash (cfg-v cfg) (- cbv)) :nt1 nt1 :ltile (log2 nt1)
                   :ntile (floor (* nt1 (1+ nt1)) 2) :nbb (max 0 (- (log2 s) +keyblk+))
                   :ng ngen :lng (log2 ngen) :r (* b ngen)))))

;;; ------------------------------------------------------------------ packing (proto.Pack)

(defun pack (k c shape cmax pad)
  "Pack K = Kin Kout blocks of 2^c into Kin G columns of 2^c2: the widest columns whose zero padding
stays within PAD of the data or within 2^20 entries.  Returns K2, c2 and the blocks per column."
  (destructuring-bind (kin kout) (or shape (list 1 k))
    (let ((lb (- (min cmax (+ c (nbits kout))) c)))
      (loop while (plusp lb)
            do (let ((npad (ash (* (- (ash (ceiling kout (expt 2 lb)) lb) kout) kin) c)))
                 ;; the same double-float arithmetic as Python's, compared exactly
                 (when (or (<= npad (* (* (* kout kin) pad) (expt 2 c))) (<= npad +pack-abs+))
                   (return))
                 (decf lb)))
      (values (* kin (ceiling kout (expt 2 lb))) (+ c lb) (expt 2 lb)))))

(defun packed-items (items cmax pad)
  "A group's items as the commitment declares them (proto.Ctx.group): (name K c), packed where a
tensor has many small blocks.  An item's optional fourth element is its block shape (Kin Kout),
or :NO-PACK."
  (loop for (name k c shape) in items
        collect (if (and (> k 1) (< c cmax) (not (eq shape :no-pack)))
                    (multiple-value-bind (k2 c2 bpc) (pack k c shape cmax pad)
                      (if (> bpc 1) (list name k2 c2) (list name k c)))
                    (list name k c))))

;;; ------------------------------------------------------------------ the layout (Proto.declare)

(defun moe-geometry (cfg shape)
  "moe.Geo: the router's and the experts' block sizes."
  (let ((e (getf (cfg-moe cfg) :e)) (k (getf (cfg-moe cfg) :k)) (de (getf (cfg-moe cfg) :de)))
    (let* ((cb (shape-cb shape))
           (cbr (min cb (1- (integer-length (logand e (- e))))))
           (cbe (min cb (1- (integer-length (logand de (- de))))))
           (kde (ash de (- cbe)))
           (ke (* k kde)))
      (list :e e :k k :lk (log2 k) :cbr cbr :ke-router (ash e (- cbr)) :cbe cbe :kde kde :ke ke
            :nbe (nbits e) :nbu (+ cbe (nbits kde)) :nb2 (+ cbe (nbits ke))))))

(defun moe-items (cfg shape l)
  "The items of layer L's MoE group (moe.MoEProof.items)."
  (let* ((geometry (moe-geometry cfg shape))
         (k (getf geometry :k)) (lk (getf geometry :lk)) (cbr (getf geometry :cbr))
         (ke-router (getf geometry :ke-router)) (cbe (getf geometry :cbe)) (ke (getf geometry :ke))
         (nbe (getf geometry :nbe)) (nbu (getf geometry :nbu)) (nb2 (getf geometry :nb2)))
    (let ((q (format nil "~D.e." l))
          (tt (shape-t* shape)))
      (flet ((rows (pre fields &optional (kb 1))
               (loop for f in fields collect (list (concatenate 'string pre f) kb tt))))
        (append (loop for f in '("z" "zo" "zwh" "sel" "dfl") collect (list (concatenate 'string q f) ke-router (+ cbr tt)))
                (list (list (concatenate 'string q "jt") nbe tt))
                (rows q '("th" "zmax" "S" "Et" "kr" "pkr" "fr" "fd"))
                (list (list (concatenate 'string q "eb") nbe (+ lk tt)))
                (loop for f in '("zs" "xg" "xu" "xd" "p" "w") collect (list (concatenate 'string q f) 1 (+ lk tt)))
                (rows q '("flz" "fl" "om" "kg" "pkg" "fg") k)
                (rows (concatenate 'string q "so.") +sr-f+ k)
                (loop for f in '("c" "a" "qu" "qm" "q2") collect (list (concatenate 'string q f) ke (+ cbe tt)))
                (rows (concatenate 'string q "ru.") +rq-f+ k) (list (list (concatenate 'string q "ru.jb") (* nbu k) tt))
                (rows (concatenate 'string q "rm.") +rq-f+ k) (list (list (concatenate 'string q "rm.jb") (* nbu k) tt))
                (rows (concatenate 'string q "r2.") +rq-f+) (list (list (concatenate 'string q "r2.jb") nb2 tt)))))))

(defun p-cols (shape)
  "The attention probabilities' columns per head: (count log-size), the widest whose padding stays
within P_PAD of the tiles."
  (let ((tot (* (shape-b shape) (shape-ntile shape))))
    (loop for c from (min 23 (+ 14 (nbits tot))) above 13
          for npc = (ash (+ (ash tot 14) (expt 2 c) -1) (- c))
          when (<= (ash npc c) (* (ash tot 14) (+ 1 *p-pad*)))
            return (list npc c)
          finally (return (list tot 14)))))

(defun mlp-slices (shape)
  "The MLP's slices of the d_ff blocks: (b0 b1) pairs."
  (let* ((kf (shape-kf shape))
         (per (max 1 (min kf (floor (ash +mlp-slice+ 24) (ash (shape-tt shape) (shape-cb shape)))))))
    (loop for b from 0 below kf by per collect (list b (min kf (+ b per))))))

(defun layer-groups (cfg shape l)
  (append (list (format nil "~D.in" l))
          (loop for g below (shape-nkv shape) collect (format nil "~D.att~D" l g))
          (list (format nil "~D.ao" l) (format nil "~D.mi" l))
          (if (cfg-moe cfg)
              (list (format nil "~D.moe" l))
              (loop for s below (length (mlp-slices shape)) collect (format nil "~D.mlp~D" l s)))
          (list (format nil "~D.mo" l))))

(defun layer-weight-groups (cfg l)
  (cons (format nil "~D.W" l)
        (and (cfg-moe cfg) (loop for nm in '("Wge" "Wue" "Wde") collect (format nil "~D.~A" l nm)))))

(defun commitment-layout (cfg shape &key (theta 0) (cmax 23))
  "The commitment layout of a proof (llm.Proto.declare), as data: the groups in declaration order,
each (name items) with its items (name K c) as the commitment sees them (packed); the merge sets
(one per layer); and the segments (one per layer)."
  (let* ((tt (shape-t* shape)) (cb (shape-cb shape)) (ldh (shape-ldh shape))
         (kd (shape-kd shape)) (kq (shape-kq shape)) (kk (shape-kk shape)) (kf (shape-kf shape))
         (kv (shape-kv shape)) (cbv (shape-cbv shape)) (nh (shape-nh shape)) (grp (shape-grp shape))
         (groups '()))
    (labels ((cat (&rest parts) (apply #'concatenate 'string parts))
             (rows (pre fields) (loop for f in fields collect (list (cat pre f) 1 tt)))
             (jb (pre k) (list (list (cat pre "jb") (+ cb (nbits k)) tt)))
             (group (name items &optional (pad *pack-pad*))
               (push (list name (packed-items items cmax pad)) groups)))
      (dotimes (l (cfg-l cfg))
        (let* ((p (format nil "~D." l))
               (ko (if (cfg-qk-norm cfg) kk (+ kq (* 2 kk))))
               (qkn (and (cfg-qk-norm cfg)
                         (loop for (x kx) in (list (list "qn" kq) (list "kn" kk))
                               append (let ((q (cat p x "."))
                                            (nr (+ (- cb ldh) tt (nbits kx))))
                                        (append (loop for f in '("q8" "qp" "o" "wh") collect (list (cat q f) kx (+ cb tt)))
                                                (loop for f in +rq-f+ collect (list (cat q "r." f) 1 nr))
                                                (list (list (cat q "r.jb") ldh nr))
                                                (loop for f in +ns-f+ collect (list (cat q "n." f) 1 nr))
                                                (loop for f in +qk-f+ collect (list (cat q f) 1 nr))))))))
          (group (cat p "in")
                 (append (list (list (cat p "q1") kd (+ cb tt)) (list (cat p "qp1") kd (+ cb tt)))
                         (rows (cat p "r1.") +rq-f+) (jb (cat p "r1.") kd) (rows (cat p "n1.") +ns-f+)
                         (list (list (cat p "Q") kq (+ cb tt)) (list (cat p "K") kk (+ cb tt)) (list (cat p "V") kk (+ cb tt))
                               (list (cat p "oq") ko (+ cb tt)) (list (cat p "whq") ko (+ cb tt))
                               (list (cat p "rQ") nh (+ ldh tt)) (list (cat p "rK") (shape-nkv shape) (+ ldh tt)))
                         qkn))
          (dotimes (g (shape-nkv shape))
            (let ((q (format nil "~D.g~D." l g))
                  (npc (p-cols shape)))
              (group (format nil "~D.att~D" l g)
                     (append (list (list (cat q "P") (* grp (first npc)) (second npc)) (list (cat q "js") grp tt)
                                   (list (cat q "Kg") grp (+ ldh tt)) (list (cat q "mk") 1 tt)
                                   (list (cat q "Nr") grp (+ ldh tt (shape-nbb shape))))
                             (loop for f in +sr-f+ collect (list (cat q "nK." f) grp tt))))))
          (group (cat p "ao")
                 (append (loop for f in +sr-f+ collect (list (cat p "nE." f) nh tt))
                         (loop for f in +sr-f+ collect (list (cat p "nM." f) nh tt))
                         (loop for f in +nd-f+ collect (list (cat p "nd." f) nh tt))
                         (list (list (cat p "nd.xh") nh (+ ldh tt)) (list (cat p "nd.t") nh (+ ldh tt))
                               (list (cat p "o") kq (+ cb tt)) (list (cat p "qo") kq (+ cb tt)))
                         (rows (cat p "ro.") +rq-f+) (jb (cat p "ro.") kq) (rows (cat p "so.") +srx-f+)
                         (list (list (cat p "s1") kd (+ cb tt)) (list (cat p "os1") kd (+ cb tt)))))
          (let ((mi-head (append (list (list (cat p "q2") kd (+ cb tt)) (list (cat p "qp2") kd (+ cb tt)))
                                 (rows (cat p "r2.") +rq-f+) (jb (cat p "r2.") kd) (rows (cat p "n2.") +ns-f+))))
            (if (cfg-moe cfg)
                (progn
                  (group (cat p "mi") (append mi-head (rows (cat p "sd.") +srx-f+)))
                  (group (cat p "moe") (moe-items cfg shape l))
                  (let* ((geometry (moe-geometry cfg shape))
                         (kde (getf geometry :kde)) (cbe (getf geometry :cbe)))
                    (dolist (nm '("Wge" "Wue" "Wde"))
                      (group (format nil "~D.~A" l nm)
                             (list (list (format nil "~D.e.~A" l nm) (* kd kde)
                                         (+ cb cbe (nbits (getf (cfg-moe cfg) :e))) :no-pack))))))
                (progn
                  (group (cat p "mi") (append mi-head (rows (cat p "ru.") +rq-f+) (jb (cat p "ru.") kf)
                                              (rows (cat p "rm.") +rq-f+) (jb (cat p "rm.") kf)
                                              (rows (cat p "sd.") +srx-f+) (rows (cat p "g.") +eg-f+)))
                  (loop for (b0 b1) in (mlp-slices shape)
                        for s from 0
                        do (let ((q (format nil "~D.m~D." l s)))
                             (group (format nil "~D.mlp~D" l s)
                                    (loop for x in '("c" "a" "qu" "qm") collect (list (cat q x) (- b1 b0) (+ cb tt)))))))))
          (group (cat p "mo") (list (list (cat p "s2") kd (+ cb tt)) (list (cat p "os2") kd (+ cb tt))))))
      (let ((r (log2 (shape-r shape))))
        (group "fin" (append (list (list "f.q" kd (+ cb r)) (list "f.qp" kd (+ cb r)))
                             (loop for f in +rq-f+ collect (list (cat "f.r." f) 1 r))
                             (list (list "f.r.jb" (+ cb (nbits kd)) r))
                             (loop for f in +ns-f+ collect (list (cat "f.n." f) 1 r))
                             (list (list "f.z" kv (+ cbv r)))
                             (loop for f in +srx-f+ collect (list (cat "f.sz." f) 1 r))
                             (and (plusp theta)
                                  (append (list (list "f.p" kv (+ cbv r)) (list "f.zj" (+ cbv (nbits kv)) r))
                                          (loop for f in +samp-f+ collect (list (cat "f.s." f) 1 r)))))))
      (dotimes (l (cfg-l cfg))
        (let ((w (lambda (name k c &optional shape) (list (format nil "~D.~A" l name) k c shape))))
          (group (format nil "~D.W" l)
                 (append (if (cfg-qk-norm cfg)
                             (list (funcall w "Wq" (* kd kq) (* 2 cb)) (funcall w "Wk" (* kd kk) (* 2 cb))
                                   (funcall w "Wv" (* kd kk) (* 2 cb)))
                             (list (funcall w "Wqkv" (* kd (+ kq (* 2 kk))) (* 2 cb))))
                         (list (funcall w "Wo" (* kq kd) (* 2 cb)))
                         (if (cfg-moe cfg)
                             (let* ((geometry (moe-geometry cfg shape))
                                    (ke-router (getf geometry :ke-router)) (cbr (getf geometry :cbr)))
                               (list (list (format nil "~D.e.Wr" l) (* kd ke-router) (+ cb cbr) (list kd ke-router))))
                             (list (funcall w "Wg" (* kd kf) (* 2 cb) (list kd kf))
                                   (funcall w "Wu" (* kd kf) (* 2 cb) (list kd kf))
                                   (funcall w "Wd" (* kf kd) (* 2 cb)))))
                 *weight-pack-pad*)))
      (group "Wlm" (list (list "Wlm" (* kd kv) (+ cb cbv) (list kd kv))))
      (group "emb" (list (list "emb" (* kv kd) (+ cbv cb) (list kv kd))))
      (group "tab" (loop for j to +table-values+ collect (list (format nil "tab~D" j) 1 +table-rows+)))
      (group "rope" (list (list "rope.cos" 1 (+ ldh -1 tt)) (list "rope.sin" 1 (+ ldh -1 tt)))))
    (let ((layers (loop for l below (cfg-l cfg) collect l)))
      (list :groups (nreverse groups)
            :merges (mapcar (lambda (l) (layer-groups cfg shape l)) layers)
            :segments (mapcar (lambda (l) (append (layer-groups cfg shape l) (layer-weight-groups cfg l))) layers)
            :weight-groups (append (loop for l in layers append (layer-weight-groups cfg l)) (list "Wlm" "emb"))))))

(defun declare-units (pcs layout)
  "Declare LAYOUT's groups, merge sets and segments on the commitment PCS, in zk/'s order."
  (loop for (name items) in (getf layout :groups)
        do (sable.pcs:pcs-declare pcs name items))
  (dolist (names (getf layout :merges)) (sable.pcs:pcs-merge pcs names))
  (dolist (names (getf layout :segments)) (sable.pcs:pcs-segment-decl pcs names))
  pcs)

;;; ------------------------------------------------------------------ the policy

(defstruct (policy (:constructor %make-policy))
  "The public policy (llm.Policy): context bound, temperature index (0: greedy), G_max, stop set,
tokenizer and template ids, server-prefix digest; AUTOREGRESSIVE: the generated tokens are fed back."
  gmax stop theta ctx-bound tokenizer template hpre autoregressive)

(defun make-policy (&key gmax stop (theta 0) ctx-bound (tokenizer #()) (template #()) (hpre (make-octets 32))
                         (autoregressive t))
  (check-parameter theta (lambda (x) (and (integerp x) (<= 0 x) (< x +theta-count+))) "a temperature index")
  (%make-policy :gmax gmax :stop (sort (remove-duplicates (copy-list stop)) #'<) :theta theta
                :ctx-bound ctx-bound :tokenizer (to-octets tokenizer) :template (to-octets template)
                :hpre (to-octets hpre) :autoregressive autoregressive))

(defun python-value (x) (if (null x) "None" (format nil "~D" x)))

(defun encode-policy (policy)
  "The policy's bytes in the statement: length-prefixed parts (llm.Policy.encode)."
  (let ((parts (list (to-octets (format nil "~A|~D|None|~A|~D|" (python-value (policy-ctx-bound policy))
                                        (policy-theta policy) (python-value (policy-gmax policy))
                                        (if (policy-autoregressive policy) 1 0)))
                     (join-octets (mapcar (lambda (s) (le-bytes (ldb (byte 64 0) s) 8)) (policy-stop policy)))
                     (sha256 (policy-tokenizer policy))
                     (sha256 (policy-template policy))
                     (policy-hpre policy))))
    (join-octets (loop for part in parts collect (le-bytes (length part) 4) collect part))))

(defun check-statement (policy shape tokens targets)
  "The verifier's checks on the public statement (llm.Policy.check): TOKENS (B prompts of S) and
TARGETS (B rows of ngen) in the vocabulary, the context bound, the stopping rule, G_max, and the
generated tokens fed back.  Signals VERIFICATION-FAILURE."
  (let ((v (cfg-v (shape-cfg shape)))
        (ng (shape-ng shape))
        (gmax (or (policy-gmax policy) (shape-ng shape))))
    (unless (every (lambda (row) (every (lambda (x) (< -1 x v)) row)) (append tokens targets))
      (reject :statement "token outside the vocabulary"))
    (when (and (policy-ctx-bound policy) (> (shape-s shape) (policy-ctx-bound policy)))
      (reject :statement "context bound"))
    (loop for x in tokens
          for y in targets
          do (let ((stops (mapcar (lambda (token) (member token (policy-stop policy))) y)))
               (when (some #'identity (butlast stops))
                 (reject :statement "output continues after a stop token"))
               (unless (or (car (last stops)) (= ng gmax))
                 (reject :statement "output truncated before a stop token or G_max"))
               (when (> ng gmax)
                 (reject :statement "output longer than G_max"))
               (when (and (policy-autoregressive policy)
                          (not (equal (nthcdr (1+ (- (shape-s shape) ng)) x) (butlast y))))
                 (reject :statement "generated tokens are not the fed-back sequence"))))
    t))

;;; ------------------------------------------------------------------ the statement

(defun int64-octets (rows)
  "Rows of integers as int64 LE bytes, row-major (numpy astype('<i8').tobytes())."
  (join-octets (loop for row in rows append (mapcar (lambda (x) (le-bytes (ldb (byte 64 0) x) 8)) row))))

(defun statement (cfg shape tokens targets c-w constants-digest &key (policy (make-policy))
                                                                    (nonce (make-octets 32)) (salt (make-octets 32)))
  "The statement that seeds the transcript (llm.statement):
  P| len(policy) | policy | C_W | h_circ | H(x) | H(y) | n_c | T | salt,
h_circ = SHA-256(configuration | B | S | cb | cbv | ngen || F_O constants digest).  C_W and
CONSTANTS-DIGEST (model.const_digest) come from the model's registry entry."
  (check-parameter c-w (lambda (x) (= (length x) 32)) "32 bytes")
  (let* ((circ (format nil "~A|~D|~D|~D|~D|~D" (cfg-repr cfg) (shape-b shape) (shape-s shape)
                       (shape-cb shape) (shape-cbv shape) (shape-ng shape)))
         (h-circ (sha256 (concatenate 'octets (to-octets circ) (to-octets constants-digest))))
         (pol (encode-policy policy)))
    (join-octets (list (to-octets "P|") (le-bytes (length pol) 4) pol (to-octets c-w) h-circ
                       (sha256 (int64-octets tokens)) (sha256 (int64-octets targets))
                       (to-octets nonce) (le-bytes (shape-tt shape) 8) (to-octets salt)))))

(defun sampling-seed (tokens ngen c-w policy &key (nonce (make-octets 32)))
  "r = H(seed | n_c | C_W | H(x) | policy), x the prompts tokens[:, :S - ngen + 1] (llm.sampling_seed)."
  (sha256 (join-octets (list (to-octets "seed") (to-octets nonce) (to-octets c-w)
                             (sha256 (int64-octets (mapcar (lambda (row) (subseq row 0 (1+ (- (length row) ngen))))
                                                           tokens)))
                             (encode-policy policy)))))

(defun weights-digest (caps)
  "C_W from the weight groups' caps, in order (llm._wdigest): SHA-256(C_W| caps...).  CAPS: a list
of the caps' bytes."
  (sha256 (join-octets (cons (to-octets "C_W|") (mapcar #'to-octets caps)))))
