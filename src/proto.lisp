;;;; proto.lisp -- the protocol context of the transformer proof (zk/proto.py Ctx): one record that
;;;; the steps of the proof share, in three modes:
;;;;
;;;;   :commit  the prover's first pass: witnesses -> commitments, lookups counted;
;;;;   :prove   the prover's second pass: domains and sumchecks, claims routed to tensors;
;;;;   :verify  the verifier: replays the transcript.
;;;;
;;;; A committed tensor is K blocks of 2^c base-field values in a commitment group (a unit of the
;;;; commitment, pcs.lisp).  Tensors of many small blocks are packed into wider columns (PACK).  A
;;;; claim is (point values): VALUES an alist (block . value) of f~_{tensor,block}(point); a weighted
;;;; claim (point weights value) says sum_b weight_b f~_{tensor,b}(point) = value.  A group is opened
;;;; once every claim on it is known; the commitment's global opening runs at the end.
;;;;
;;;; Points are lists of EXTs (LSB first), values EXTs, in F_p^4 throughout.

(defpackage #:exacta.proto
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.forms #:exacta.transcript
        #:exacta.sumcheck #:exacta.gkr #:exacta.domain #:exacta.pcs)
  (:export #:nbits #:+pack-abs+ #:*pack-pad*
           #:pack #:make-pack #:pack-kin #:pack-kout #:pack-c #:pack-c2 #:pack-lb #:pack-bpc #:pack-g
           #:pack-k2 #:pack-location #:pack-blocks #:pack-layout
           #:context #:make-context #:context-mode #:context-tr #:context-pcs #:context-z
           #:context-beta #:context-assign #:context-counter #:prover-p #:proving-p #:active-p
           #:+lookup-limit+ #:+beta-count+ #:draw-lookup-challenges
           #:declare-group #:group-items #:tensor-info #:commit-group #:open-group
           #:claim #:point-claim #:weighted-claim #:pending-claims #:claim-sum #:block-combination
           #:context-challenge #:send-values
           #:run-domain #:check-lookup-sets #:parse-assign #:assign-octets
           #:piece #:sumcheck-degree #:run-sumcheck
           #:put-virtual #:take-virtual))

(in-package #:exacta.proto)

(defconstant +lookup-limit+ (expt 2 30) "logup.LIMIT: the most lookups in one chunk.")
(defconstant +beta-count+ 8 "gkr.NBETA: the tuple coefficients drawn for every domain.")

;;; ------------------------------------------------------------------ packing (proto.Pack)

(defconstant +pack-abs+ (expt 2 20))
(defparameter *pack-pad* 0.1d0 "proto.PACK_PAD: packing may pad by this fraction of the data.")

(defun nbits (k) (max 0 (integer-length (1- k))))

(defstruct (pack (:constructor %make-pack (kin kout c c2 lb bpc g k2)))
  "K = Kin Kout blocks of 2^c stored as Kin G columns of 2^c2: column (tb, ob / bpc) holds the blocks
ob of row tb, as its low bits ob mod bpc, zero padded."
  kin kout c c2 lb bpc g k2)

(defun make-pack (k c shape cmax &optional (pad *pack-pad*))
  "The widest columns whose zero padding stays within PAD of the data or within 2^20 entries
(SHAPE: (Kin Kout), or NIL for (1 K))."
  (destructuring-bind (kin kout) (or shape (list 1 k))
    (assert (= (* kin kout) k))
    (let ((lb (- (min cmax (+ c (nbits kout))) c)))
      (loop while (plusp lb)
            do (let ((npad (ash (* (- (ash (ceiling kout (expt 2 lb)) lb) kout) kin) c)))
                 ;; the same double-float arithmetic as Python's, compared exactly
                 (when (or (<= npad (* (* (* kout kin) pad) (expt 2 c))) (<= npad +pack-abs+))
                   (return))
                 (decf lb)))
      (let ((g (ceiling kout (expt 2 lb))))
        (%make-pack kin kout c (+ c lb) lb (expt 2 lb) g (* kin g))))))

(defun pack-location (pack b)
  "Block B's column and its index within the column."
  (multiple-value-bind (tb ob) (floor b (pack-kout pack))
    (values (+ (* tb (pack-g pack)) (floor ob (pack-bpc pack))) (mod ob (pack-bpc pack)))))

(defun pack-blocks (pack j)
  "The real (unpadded) blocks of column J."
  (min (pack-bpc pack) (- (pack-kout pack) (* (mod j (pack-g pack)) (pack-bpc pack)))))

(defun pack-layout (pack data)
  "DATA (K blocks of 2^c) as the packed columns, K2 of 2^c2."
  (let* ((size (expt 2 (pack-c pack)))
         (width (* (pack-g pack) (pack-bpc pack)))
         (out (make-array (* (pack-kin pack) width size) :initial-element 0)))
    (dotimes (tb (pack-kin pack) out)
      (dotimes (ob (pack-kout pack))
        (replace out data :start1 (* (+ (* tb width) ob) size)
                          :start2 (* (+ (* tb (pack-kout pack)) ob) size)
                          :end2 (* (+ (* tb (pack-kout pack)) ob 1) size))))))

;;; ------------------------------------------------------------------ the context

(defstruct (context (:constructor %make-context (mode tr pcs)))
  mode tr pcs
  (tensors (make-hash-table :test 'equal))      ; name -> (K c group)
  (groups (make-hash-table :test 'equal))       ; group -> its items as the commitment has them
  (packs (make-hash-table :test 'equal))        ; name -> pack
  (claims (make-hash-table :test 'equal))       ; name -> ((point values) ...), in order
  (wclaims (make-hash-table :test 'equal))      ; name -> ((point weights value) ...)
  (virtual (make-hash-table :test 'equal))      ; key -> items (claims a later step consumes)
  (chunk-sums (make-hash-table))                ; chunk -> sum of its lookups' root fractions
  (chunk-counts (make-hash-table))              ; chunk -> lookups
  (dynamic-sums (make-hash-table))              ; tag -> sum of the root fractions
  (seen (make-hash-table :test 'equal))         ; (domain lookup) proved
  (assign (make-hash-table :test 'equal))       ; (domain lookup) -> chunk
  counter                                       ; :commit mode: (lambda (domain)) counts lookups
  z beta)

(defun make-context (mode tr &key (cmax 23) (queries 153) (grind 30) counter)
  "A context in MODE (:commit, :prove or :verify) on transcript TR (of degree 4), with a
commitment of these parameters.  COUNTER: in :commit mode, called with each domain run."
  (check-parameter mode (lambda (m) (member m '(:commit :prove :verify))) ":commit, :prove or :verify")
  (let ((context (%make-context mode tr (make-pcs tr :prover (not (eq mode :verify))
                                                     :cmax cmax :queries queries :grind grind))))
    (setf (context-counter context) counter)
    context))

(defun prover-p (context) (not (eq (context-mode context) :verify)))
(defun proving-p (context) (eq (context-mode context) :prove))
(defun active-p (context) (not (eq (context-mode context) :commit)))

(defun tensor-info (context name)
  "(K c group) of tensor NAME."
  (or (gethash name (context-tensors context))
      (error 'invalid-parameter :name 'tensor :value name :expected "a declared tensor")))

(defun group-items (context gname) (gethash gname (context-groups context)))

(defun append-to (table key item)
  (setf (gethash key table) (append (gethash key table) (list item))))

;;; ------------------------------------------------------------------ tensors and groups

(defun declare-group (context gname items &key (pad *pack-pad*))
  "Declare commitment group GNAME of ITEMS (name K c [shape]): SHAPE (Kin Kout) for packing, or
:NO-PACK."
  (let ((cmax (pcs-cmax (context-pcs context))))
    (setf (gethash gname (context-groups context))
          (loop for (name k c shape) in items
                do (setf (gethash name (context-tensors context)) (list k c gname))
                collect (let ((pack (and (> k 1) (< c cmax) (not (eq shape :no-pack))
                                         (make-pack k c shape cmax pad))))
                          (if (and pack (> (pack-bpc pack) 1))
                              (progn (setf (gethash name (context-packs context)) pack)
                                     (list name (pack-k2 pack) (pack-c2 pack)))
                              (list name k c)))))
    (pcs-declare (context-pcs context) gname (group-items context gname))))

(defun commit-group (context gname &optional tensors)
  "Commit group GNAME; the prover gives its TENSORS, an alist name -> K 2^c base values."
  (pcs-commit (context-pcs context) gname
              (loop for (name . data) in tensors
                    collect (let ((pack (gethash name (context-packs context))))
                              (cons name (if pack (pack-layout pack data) data))))))

(defun open-group (context gname)
  "Open group GNAME at every claim made on its tensors, which are then forgotten."
  (let ((one (ext-one 4)))
    (pcs-open (context-pcs context) gname
              (loop for (name) in (group-items context gname)
                    append (loop for (point values) in (gethash name (context-claims context))
                                 append (loop for (b . v) in values collect (list name point (list (cons b one)) v)))
                    append (loop for (point weights value) in (gethash name (context-wclaims context))
                                 collect (list name point weights value))
                    do (remhash name (context-claims context))
                       (remhash name (context-wclaims context))))))

(defun pending-claims (context name)
  "The claims on tensor NAME not yet opened: its block claims and its weighted claims."
  (values (gethash name (context-claims context)) (gethash name (context-wclaims context))))

;;; ------------------------------------------------------------------ claims

(defun bits-point (bits)
  "A point of base constants 0/1."
  (mapcar (lambda (bit) (base->ext bit 4)) bits))

(defun claim (context name point values)
  "Block claim on tensor NAME at its in-block POINT (c EXTs): VALUES an alist (block . EXT), or a
vector of every block's EXT.  On a packed tensor, the claims of a column whose every block is
claimed are combined with a challenge tau over its low bits; the others become claims at the
column point (POINT, the block's low bits)."
  (destructuring-bind (k c group) (tensor-info context name)
    (declare (ignore k group))
    (assert (= (length point) c) () "claim on ~A: a point of ~D variables, not ~D" name c (length point)))
  (let ((values (if (vectorp values) (loop for v across values for b from 0 collect (cons b v)) values))
        (pack (gethash name (context-packs context))))
    (if (null pack)
        (append-to (context-claims context) name (list point values))
        (let ((by-column '()))           ; column -> ((low . value) ...), both in first-seen order
          (loop for (b . v) in values
                do (multiple-value-bind (j low) (pack-location pack b)
                     (let ((entry (assoc j by-column)))
                       (if entry
                           (setf (cdr entry) (append (cdr entry) (list (cons low v))))
                           (setf by-column (append by-column (list (list j (cons low v)))))))))
          (let ((full (remove-if-not (lambda (entry) (= (length (cdr entry)) (pack-blocks pack (car entry))))
                                     by-column)))
            (when full
              (let* ((tau (context-challenge context (pack-lb pack)))
                     (e (eq-table tau 4)))
                (append-to (context-claims context) name
                           (list (append point tau)
                                 (loop for (j . lows) in full
                                       collect (cons j (reduce #'ext+ (loop for (low . v) in lows
                                                                            collect (ext* (aref e low) v))
                                                               :initial-value (make-ext 4))))))))
            (let ((by-low '()))          ; low -> ((column . value) ...)
              (loop for (j . lows) in by-column
                    unless (member j full :key #'car)
                      do (loop for (low . v) in lows
                               do (let ((entry (assoc low by-low)))
                                    (if entry
                                        (setf (cdr entry) (append (cdr entry) (list (cons j v))))
                                        (setf by-low (append by-low (list (list low (cons j v)))))))))
              (loop for (low . columns) in by-low
                    do (append-to (context-claims context) name
                                  (list (append point (bits-point (loop for i below (pack-lb pack)
                                                                        collect (ldb (byte 1 i) low))))
                                        columns)))))))))

(defun weighted-claim (context name point weights value)
  "sum_b weight_b f~_{NAME,b}(POINT) = VALUE (WEIGHTS: an alist block -> EXT); NAME is not packed."
  (destructuring-bind (k c group) (tensor-info context name)
    (declare (ignore k group))
    (assert (= (length point) c))
    (assert (null (gethash name (context-packs context))) () "weighted claim on packed ~A" name))
  (append-to (context-wclaims context) name (list point weights value)))

(defun row-dot (data e k size)
  "Per block b < K: sum_x data[b size + x] e[x]."
  (coerce (loop for b below k
                collect (let ((acc (make-ext 4)))
                          (dotimes (x size acc)
                            (let ((v (aref data (+ (* b size) x))))
                              (setf acc (ext+ acc (if (integerp v) (ext-scale (aref e x) v) (ext* (aref e x) v))))))))
          'vector))

(defun point-claim (context name point &key value data)
  "A claim on the whole tensor NAME at POINT (c + nbits K EXTs).  The prover sends every block's
value at the in-block point (from DATA, its K 2^c values) and both sides check their eq combination
against VALUE when given; the block values become block claims.  Returns the combination.  A tensor
packed in one row, with VALUE given, takes a weighted claim instead, and nothing is sent."
  (destructuring-bind (k c group) (tensor-info context name)
    (declare (ignore group))
    (assert (= (length point) (+ c (nbits k))))
    (let ((pack (gethash name (context-packs context))))
      (when (and pack (= (pack-kin pack) 1) value)
        (let* ((split (+ c (pack-lb pack)))
               (e (eq-table (nthcdr split point) 4)))
          (weighted-claim* context name (subseq point 0 split)
                           (loop for j below (pack-k2 pack) collect (cons j (aref e j)))
                           value)
          (return-from point-claim value)))
      (let* ((inner (subseq point 0 c))
             (values (if (prover-p context)
                         (send-values context (row-dot data (eq-table inner 4) k (expt 2 c)))
                         (send-values context nil (list k 4))))
             (total (block-combination context (nthcdr c point) values)))
        (when (and value (not (prover-p context)) (not (ext= total value)))
          (reject :point-claim "point claim on ~A: the block values do not match" name))
        (claim context name inner values)
        total))))

(defun weighted-claim* (context name point weights value)
  "A weighted claim on a packed tensor's columns (the commitment's own items)."
  (append-to (context-wclaims context) name (list point weights value)))

(defun claim-sum (context coefficients values)
  "sum_i coefficient_i value_i, the claim of a sumcheck: the verifier's value (the prover's claims
are not read; zero, as zk/ gives it)."
  (if (prover-p context)
      (make-ext 4)
      (reduce #'ext+ (mapcar #'ext* coefficients values) :initial-value (make-ext 4))))

(defun block-combination (context point values)
  "sum_b eq(POINT, b) values_b (VALUES: a vector of EXTs)."
  (declare (ignore context))
  (let ((e (eq-table point 4)))
    (reduce #'ext+ (loop for v across values for b from 0 collect (ext* (aref e b) v))
            :initial-value (make-ext 4))))

;;; ------------------------------------------------------------------ messages and challenges

(defun context-challenge (context &optional k)
  "One challenge (an EXT), or a list of K."
  (if k (challenges (context-tr context) k) (challenge (context-tr context))))

(defun send-values (context &optional values shape)
  "The prover sends VALUES (EXTs) as a (count, 4) array; the verifier receives them, of SHAPE when
given.  Returns them as a vector of EXTs."
  (let ((m (if (prover-p context)
               (send (context-tr context)
                     (make-array (list (length values) 4) :element-type '(unsigned-byte 32)
                                                         :initial-contents (map 'list (lambda (e) (coerce e 'list)) values)))
               (send (context-tr context) nil (or shape :any)))))
    (unless (= (array-rank m) 2) (reject :send-values "values of rank ~D" (array-rank m)))
    (coerce (loop for row below (array-dimension m 0)
                  collect (coerce-ext (loop for col below (array-dimension m 1) collect (aref m row col))))
            'vector)))

(defun draw-lookup-challenges (context)
  "Z and the tuple coefficients beta, after every commitment (Proto.run)."
  (setf (context-z context) (context-challenge context)
        (context-beta context) (context-challenge context +beta-count+)))

;;; ------------------------------------------------------------------ domains and LogUp sets

(defun constant-tag (tag)
  "A tag form's value when it is a constant (layer parameters are substituted already), else NIL."
  (let ((form (normalize tag)))
    (and (integerp form) (mod form +p+))))

(defun fraction (pq)
  (ext* (car pq) (handler-bind ((field-division-by-zero
                                  (lambda (c) (declare (ignore c)) (invoke-restart 'return-zero))))
                   (ext-inv (cdr pq)))))

(defun account-root (context domain l pq)
  "Add lookup L's root fraction to its chunk's sum (a lookup into the table: counted, assigned to
a chunk) or to its tag's dynamic sum (a lookup with a numerator, or :DYNAMIC)."
  (let ((lookup (nth l (domain-lookups domain))))
    (if (and (not (getf lookup :dynamic)) (null (getf lookup :num)))
        (let ((key (list (domain-name domain) l)))
          (multiple-value-bind (chunk found) (gethash key (context-assign context))
            (unless found (reject :logup "lookup ~S has no chunk" key))
            (when (gethash key (context-seen context)) (reject :logup "lookup ~S proved twice" key))
            (setf (gethash key (context-seen context)) t)
            (setf (gethash chunk (context-chunk-sums context))
                  (ext+ (gethash chunk (context-chunk-sums context) (make-ext 4)) (fraction pq)))
            (incf (gethash chunk (context-chunk-counts context) 0)
                  (* (domain-blocks domain) (expt 2 (domain-n domain))))))
        (let ((tag (constant-tag (getf lookup :tag))))
          (unless tag (reject :logup "domain ~A: dynamic lookup ~D without a constant tag" (domain-name domain) l))
          (setf (gethash tag (context-dynamic-sums context))
                (ext+ (gethash tag (context-dynamic-sums context) (make-ext 4)) (fraction pq)))))))

(defun run-domain (context domain routes)
  "Prove or verify DOMAIN, then give each column's claim to its route: ROUTES an alist column name
-> (lambda (point value)).  In :commit mode, only count its lookups (the context's counter)."
  (ecase (context-mode context)
    (:commit (when (context-counter context) (funcall (context-counter context) domain)))
    ((:prove :verify)
     (let ((tr (context-tr context)) (z (context-z context)) (beta (context-beta context)))
       (multiple-value-bind (claims roots)
           (if (proving-p context)
               (prove-domain tr domain z beta)
               (verify-domain tr domain z beta))
         (unless (proving-p context)
           (loop for pq in roots for l from 0 do (account-root context domain l pq)))
         (loop for (name point value) in claims
               do (let ((route (cdr (assoc name routes :test #'equal))))
                    (unless route (error 'invalid-parameter :name 'routes :value name :expected "a route for every column"))
                    (funcall route point value))))))))

(defun check-lookup-sets (context)
  "The verifier's last LogUp checks: every dynamic set sums to zero, no chunk is over its limit,
and every assigned lookup was proved."
  (maphash (lambda (tag sum)
             (unless (ext= sum (make-ext 4)) (reject :logup "dynamic set ~D does not sum to zero" tag)))
           (context-dynamic-sums context))
  (maphash (lambda (chunk n)
             (when (> n +lookup-limit+) (reject :logup "chunk ~D has ~D lookups" chunk n)))
           (context-chunk-counts context))
  (maphash (lambda (key chunk)
             (declare (ignore chunk))
             (unless (gethash key (context-seen context)) (reject :logup "lookup ~S assigned, not proved" key)))
           (context-assign context))
  t)

(defun parse-assign (octets)
  "The prover's chunk assignment, lines domain TAB lookup TAB chunk, as a hash table."
  (let ((table (make-hash-table :test 'equal)))
    (when (plusp (length octets))
      (dolist (line (split-string (map 'string #'code-char octets) #\Newline))
        (destructuring-bind (domain lookup chunk) (split-string line #\Tab)
          (setf (gethash (list domain (parse-integer lookup)) table) (parse-integer chunk)))))
    table))

(defun assign-octets (assignments)
  "ASSIGNMENTS, a list of ((domain lookup) . chunk) in commit order, as the prover sends them."
  (to-octets (format nil "~{~A~^~%~}"
                     (loop for ((domain lookup) . chunk) in assignments
                           collect (format nil "~A~C~D~C~D" domain #\Tab lookup #\Tab chunk)))))

;;; ------------------------------------------------------------------ sumchecks

;;; A statement is a form over table symbols and constants.  Its round polynomials' degree, as the
;;; header states it, follows zk/'s rules: a (PIECE form) -- a polynomial of zk/'s Prog, or a whole
;;; sum of terms -- counts per round the tables that vary in that round, maximised over the rounds;
;;; around the pieces, products add and sums take the maximum.

(defun piece (form) (list 'piece form))

(defun sumcheck-degree (statement tables n)
  "STATEMENT's header degree over N rounds; TABLES a list of (symbol kind c ...)."
  (labels ((varies-p (symbol j)
             (let ((table (assoc symbol tables)))
               (and table
                    (destructuring-bind (kind c) (subseq table 1 3)
                      (ecase kind (:full t) (:col (< j c)) (:row (>= j c)))))))
           (piece-degree (form)
             (if (zerop n)
                 0
                 (loop for j below n
                       maximize (loop for (monomial) in (expand form)
                                      maximize (count-if (lambda (v) (varies-p v j)) monomial)))))
           (walk (form)
             (cond ((and (consp form) (eq (first form) 'piece)) (piece-degree (second form)))
                   ((symbolp form) (if (assoc form tables) (piece-degree form) 0))
                   ((atom form) 0)
                   ((eq (first form) '*) (reduce #'+ (mapcar #'walk (rest form))))
                   ((eq (first form) 'expt) (* (walk (second form)) (third form)))
                   (t (reduce #'max (mapcar #'walk (rest form)) :initial-value 0)))))
    (walk statement)))

(defun strip-pieces (form)
  (cond ((atom form) form)
        ((eq (first form) 'piece) (strip-pieces (second form)))
        (t (cons (first form) (mapcar #'strip-pieces (rest form))))))

(defun full-table (kind c shared data k n)
  "A :ROW or :COL table (or a shared one) as K blocks of 2^N entries."
  (let ((out (make-array (* k (expt 2 n)))))
    (dotimes (b k out)
      (dotimes (x (expt 2 n))
        (setf (aref out (+ (* b (expt 2 n)) x))
              (aref data (ecase kind
                           (:full (+ (if shared 0 (* b (expt 2 n))) x))
                           (:row (+ (if shared 0 (* b (expt 2 (- n c)))) (ash x (- c))))
                           (:col (+ (if shared 0 (* b (expt 2 c))) (ldb (byte c 0) x))))))))))

(defun run-sumcheck (context label n k statement claim tables
                     &key eq block-weights constants (degree (sumcheck-degree statement tables n)))
  "The sumcheck of STATEMENT over K blocks of N variables (proto.Ctx.sumcheck):
sum_b bw_b sum_x w(x) F(tables(x, b)) = CLAIM, w = eq(EQ, x) or 1, bw the BLOCK-WEIGHTS or 1.
TABLES: (symbol kind c shared data) in the order their finals are sent; KIND :FULL, :ROW or :COL,
SHARED the same in every block, DATA the prover's values.  The verifier checks the reduced claim
against the finals the prover sends.  Returns the challenges and the finals (per table, a vector of
its blocks' values, of one if shared)."
  (let ((tr (context-tr context))
        (form (strip-pieces statement)))
    (if (prover-p context)
        (let ((weights (or block-weights (and (> k 1) (make-array k :initial-element (ext-one 4))))))
          (multiple-value-bind (rho finals folded)
              (prove tr n form (loop for (symbol kind c shared data) in tables
                                     collect (cons symbol (full-table kind c shared data k n)))
                     :eq eq :constants constants :label label :block-weights weights :degree degree)
            (declare (ignore finals))
            (values rho
                    (loop for (symbol kind nil shared) in tables
                          collect (let* ((blocks (cdr (assoc symbol folded)))
                                         (sent (if (and shared (not (eq kind :full))) (subseq blocks 0 1) blocks)))
                                    (send-values context sent))))))
        (multiple-value-bind (rho reduced)
            (verify tr n form claim :eq eq :constants constants :blocks k :label label :degree degree)
          (let* ((finals (loop repeat (length tables) collect (send-values context)))
                 (kk (reduce #'max (mapcar #'length finals) :initial-value 1))
                 (expected (make-ext 4)))
            (dolist (f finals)
              (unless (member (length f) (list 1 kk))
                (reject :sumcheck "~A: finals of ~D blocks" label (length f))))
            (dotimes (b kk)
              (let ((value (evaluate form (append (loop for (symbol) in tables for f in finals
                                                        collect (cons symbol (aref f (if (> (length f) 1) b 0))))
                                                  constants)
                                     :degree 4)))
                (setf expected (ext+ expected (if block-weights (ext* value (elt block-weights b)) value)))))
            (unless (ext= reduced expected)
              (reject :sumcheck "sumcheck ~A failed" label))
            (values rho finals))))))

;;; ------------------------------------------------------------------ virtual claims

(defun put-virtual (context key item)
  "Leave ITEM (a claim on a virtual column) for the step that defines it."
  (append-to (context-virtual context) key item))

(defun take-virtual (context key)
  "The items left under KEY, which are then gone."
  (prog1 (gethash key (context-virtual context))
    (remhash key (context-virtual context))))
