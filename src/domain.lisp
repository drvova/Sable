;;;; domain.lisp -- a domain: K blocks of 2^n elements, base-field columns, zerocheck constraints
;;;; and lookups, proved by one LogUp-GKR and one sumcheck (zk/domain.py).
;;;;
;;;; A domain is data: columns, constraints as forms over the column names, and lookups as
;;;; (:tag form :values (form ...) :num form), each leaf of a lookup's tree being the fraction
;;;; num / (Z - tag - beta_0 v_0 - beta_1 v_1 - ...).  The proof:
;;;;
;;;;   1. batched LogUp-GKR over all lookups down to layer n-1 (gkr.lisp), at a point r;
;;;;   2. one sumcheck over the halves of the columns (split by the top in-block variable) that proves
;;;;      the leaf layer of every tree and the zerocheck  sum_x eq(r, x) [C(x, 0) + eta C(x, 1)] = 0;
;;;;      its statement is one form (DOMAIN-STATEMENT);
;;;;   3. a challenge t turns the halves' values into column claims at (rho, t).
;;;;
;;;; Column kinds: :FULL (every variable), :ROW c (the variables from c on), :COL c (the variables
;;;; below c); a SHARED column is the same in every block.  With MERGE, the block index is bound
;;;; inside the protocol (a second sumcheck), and the claims are single values.

(defpackage #:exacta.domain
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.forms #:exacta.transcript
        #:exacta.sumcheck #:exacta.gkr)
  (:export #:domain #:make-domain #:domain-name #:domain-n #:domain-blocks #:domain-merge
           #:domain-columns #:domain-constraints #:domain-lookups
           #:column #:column-name #:column-kind #:column-c #:column-shared #:column-data
           #:domain-tables #:domain-statement #:domain-degree #:domain-leaves
           #:prove-domain #:verify-domain))

(in-package #:exacta.domain)

(defstruct (column (:constructor %make-column (name kind c shared data)))
  name kind c shared data)

(defstruct (domain (:constructor %make-domain (name n blocks merge columns constraints lookups)))
  name n blocks merge columns constraints lookups)

(defun make-domain (&key name n (blocks 1) merge columns constraints lookups)
  "A domain of BLOCKS blocks of 2^N elements.  COLUMNS: (name &key (kind :full) (c 0) shared data),
DATA a vector of base elements (K 2^n for :FULL, K 2^(n-c) for :ROW c, K 2^c for :COL c; without
the factor K if SHARED; needed by the prover only).  CONSTRAINTS: forms over the column names that
must vanish everywhere.  LOOKUPS: plists (:tag form :values (form ...) :num form)."
  (check-parameter n (lambda (n) (and (integerp n) (>= n 2))) "at least 2")
  (%make-domain name n blocks merge
                (loop for (column-name . options) in columns
                      collect (destructuring-bind (&key (kind :full) (c 0) shared data) options
                                ;; zk/domain.py's prover keeps a full column's data per block
                                (check-parameter shared (lambda (s) (not (and s (eq kind :full))))
                                                 "NIL for a :FULL column")
                                (%make-column column-name kind (if (eq kind :full) 0 c) shared data)))
                constraints lookups))

;;; ------------------------------------------------------------------ the tables of the sumcheck

(defun half-symbol (name half)
  (intern (format nil "~A@~D" (symbol-name name) half) (symbol-package name)))

(defun domain-tables (domain)
  "The sumcheck's tables, in zk/domain.py's order (_tabmap): (symbol column half kind c).  A column
that depends on the top in-block variable has two tables, its halves (HALF 0 and 1); one that does
not (:ROW n, :COL c < n) has one (HALF NIL).  KIND and C say in which rounds a table varies."
  (let ((n (domain-n domain)))
    (loop for column in (domain-columns domain)
          for name = (column-name column)
          for kind = (column-kind column)
          for c = (column-c column)
          append (cond ((or (eq kind :full) (and (eq kind :row) (< c n)))
                        (loop for h below 2 collect (list (half-symbol name h) column h kind c)))
                       ((and (eq kind :col) (= c n))
                        (loop for h below 2 collect (list (half-symbol name h) column h :col (1- n))))
                       ((eq kind :row)
                        (list (list name column nil :row (1- n))))
                       (t
                        (list (list name column nil :col c)))))))

(defun at-half (form domain half)
  "FORM with each two-half column replaced by its table for HALF."
  (sublis (loop for (symbol column h) in (domain-tables domain)
                when (eql h half) collect (cons (column-name column) symbol))
          form))

;;; ------------------------------------------------------------------ the statement

(defun constant-symbol (name &optional i)
  (intern (if i (format nil "~A~D" name i) name) '#:exacta.domain))

(defun arity (domain)
  (max 1 (reduce #'max (domain-lookups domain) :key (lambda (l) (length (getf l :values)))
                                                :initial-value 0)))

(defun denominator-form (lookup domain half)
  "q = Z - tag - sum_i beta_i v_i at HALF."
  `(- ,(constant-symbol "Z") ,(at-half (getf lookup :tag) domain half)
      ,@(loop for v in (getf lookup :values)
              for i from 0
              collect `(* ,(constant-symbol "BETA" i) ,(at-half v domain half)))))

(defun domain-statement (domain)
  "The leaf-layer and zerocheck statement, over the column halves:
  sum_l a_l (N_l(0) q_l(1) + N_l(1) q_l(0)) + a_l lambda q_l(0) q_l(1)
    + sum_i gamma_i C_i(0) + gamma_i eta C_i(1)
for lookup l's numerator N_l (1 if absent) and denominator q_l.  zk/domain.py writes the lookups
without a numerator in a factored form with the same value."
  `(+ ,@(loop for lookup in (domain-lookups domain)
              for l from 0
              append (let ((q0 (denominator-form lookup domain 0))
                           (q1 (denominator-form lookup domain 1))
                           (num (getf lookup :num)))
                       `((* ,(constant-symbol "A" l)
                            ,(if num
                                 `(+ (* ,(at-half num domain 0) ,q1) (* ,(at-half num domain 1) ,q0))
                                 `(+ ,q0 ,q1)))
                         (* ,(constant-symbol "AL" l) ,q0 ,q1))))
      ,@(loop for constraint in (domain-constraints domain)
              for i from 0
              collect `(* ,(constant-symbol "GAM" i) ,(at-half constraint domain 0))
              collect `(* ,(constant-symbol "GETA" i) ,(at-half constraint domain 1)))))

(defun domain-degree (domain &key block-phase)
  "The degree of the statement's round polynomials as zk/domain.py's header states it
(Prog.degree): each polynomial piece (tag, value, numerator, constraint) counts, in each round, only
the tables that vary in that round, maximised over the rounds; products add and sums take the
maximum.  With BLOCK-PHASE, the degree of the block sumcheck: shared tables do not vary with the
block, and the block weight adds 1."
  (let* ((n (domain-n domain))
         (tables (domain-tables domain))
         (rounds (if block-phase '(:block) (loop for j below (1- n) collect j))))
    (labels ((varies-p (symbol round)
               (destructuring-bind (column half kind c) (rest (assoc symbol tables))
                 (declare (ignore half))
                 (if (eq round :block)
                     (not (column-shared column))
                     (ecase kind
                       (:full t)
                       (:col (< round c))
                       (:row (>= round c))))))
             (piece (form)
               "zk/ Prog.degree's pdeg of one polynomial piece."
               (loop for round in rounds
                     maximize (loop for (monomial) in (expand form)
                                    maximize (count-if (lambda (v) (varies-p v round)) monomial))))
             (q (lookup half)
               (reduce #'max (cons (piece (at-half (getf lookup :tag) domain half))
                                   (mapcar (lambda (v) (piece (at-half v domain half)))
                                           (getf lookup :values))))))
      (let ((statement-degree
              (max 0
                   (loop for lookup in (domain-lookups domain)
                         maximize (let ((q0 (q lookup 0)) (q1 (q lookup 1)) (num (getf lookup :num)))
                                    (if num
                                        (max (+ (piece (at-half num domain 0)) q1)
                                             (+ (piece (at-half num domain 1)) q0)
                                             (+ q0 q1))
                                        (+ q0 q1))))
                   (loop for constraint in (domain-constraints domain)
                         maximize (max (piece (at-half constraint domain 0))
                                       (piece (at-half constraint domain 1)))))))
        (if block-phase (1+ statement-degree) statement-degree)))))

(defun domain-constants (domain z betas as lam gams eta)
  "The statement's constants: Z, the tuple coefficients beta_i, a_l and a_l lambda per lookup,
gamma_i and gamma_i eta per constraint."
  (append (list (cons (constant-symbol "Z") z))
          (loop for i below (arity domain) collect (cons (constant-symbol "BETA" i) (elt betas i)))
          (loop for a in as for l from 0
                collect (cons (constant-symbol "A" l) a)
                collect (cons (constant-symbol "AL" l) (ext* a lam)))
          (loop for g in gams for i from 0
                collect (cons (constant-symbol "GAM" i) g)
                collect (cons (constant-symbol "GETA" i) (ext* g eta)))))

(defun domain-challenges (tr domain)
  "a_l (one at least), lambda, gamma_i (one at least) and eta (zk/domain.py _challenges)."
  (let* ((as (challenges tr (max 1 (length (domain-lookups domain)))))
         (lam (challenge tr))
         (gams (challenges tr (max 1 (length (domain-constraints domain)))))
         (eta (challenge tr)))
    (values as lam gams eta)))

;;; ------------------------------------------------------------------ claims

(defun domain-claims (domain rho rb tt finals)
  "Each column's claim, (name point value), from the finals (an alist table -> value) and the
fresh point TT: a two-half column's value is f0 + tt (f1 - f0) at (rho, tt) (from c on for :ROW c);
a one-table column's is f0, at rho below c for :COL c and at no point for :ROW.  With merged blocks
a non-shared column's point ends with the block point RB."
  (let* ((n (domain-n domain))
         (tables (domain-tables domain))
         (full (append rho (list tt))))
    (loop for column in (domain-columns domain)
          for name = (column-name column)
          for halves = (remove name tables :key (lambda (table) (column-name (second table))) :test-not #'eq)
          collect (let ((f0 (cdr (assoc (first (first halves)) finals)))
                        (f1 (cdr (assoc (first (car (last halves))) finals))))
                    (multiple-value-bind (point value)
                        (if (rest halves)
                            (values (if (eq (column-kind column) :row) (nthcdr (column-c column) full) full)
                                    (flet ((interpolate (a b) (ext+ a (ext* tt (ext- b a)))))
                                      (if (typep f0 'ext)
                                          (interpolate f0 f1)                ; merged: one value
                                          (map 'vector #'interpolate f0 f1))))  ; per block
                            (values (if (eq (column-kind column) :col)
                                        (subseq rho 0 (min (column-c column) (1- n)))
                                        '())
                                    f0))
                      (list name
                            (if (and (domain-merge domain) (not (column-shared column)))
                                (append point rb)
                                point)
                            value))))))

;;; ------------------------------------------------------------------ verifier

(defun label+ (label suffix)
  (concatenate 'octets (to-octets label) (to-octets suffix)))

(defun receive-row (tr d)
  (let ((m (send tr nil (list 1 d))))
    (coerce-ext (loop for c below d collect (aref m 0 c)))))

(defun receive-rows (tr rows d)
  (let ((m (send tr nil (list rows d))))
    (coerce (loop for b below rows collect (coerce-ext (loop for c below d collect (aref m b c)))) 'vector)))

(defun verify-domain (tr domain z betas &key (label "dom"))
  "Replay a domain's proof (zk/domain.py verify).  Z (an EXT) and BETAS (at least the domain's
arity of EXTs) are the lookup challenges, drawn by the caller.  Returns the column claims, a list of
(name point value) -- with K blocks unmerged, VALUE is a vector of the blocks' values, or of one
value for a shared column -- and the lookups' root fractions."
  (let* ((d (transcript-degree tr))
         (n (domain-n domain))
         (k (domain-blocks domain))
         (lk (block-bits k))
         (lookups (length (domain-lookups domain)))
         (tables (domain-tables domain))
         (statement (domain-statement domain))
         (roots '()) r rb cp cq)
    (absorb tr (label+ label (label-name (domain-name domain))))
    (if (plusp lookups)
        (setf (values roots r rb cp cq) (verify-upper tr n k lookups))
        (setf r (challenges tr (1- n))
              rb (challenges tr lk)))
    (multiple-value-bind (as lam gams eta) (domain-challenges tr domain)
      (let ((constants (domain-constants domain z betas as lam gams eta))
            (claim (if (plusp lookups) (layer-claim as lam cp cq) (make-ext d)))
            (bw (block-weights rb k d)))
        (flet ((value (finals)
                 (evaluate statement (append finals constants) :degree d)))
          (multiple-value-bind (rho reduced)
              (verify tr (1- n) statement claim :eq r :constants constants :blocks k :label label
                                                 :degree (domain-degree domain))
            (if (domain-merge domain)
                (let ((rbn '()))
                  (when (plusp lk)
                    (setf (values rbn reduced)
                          (verify tr lk `(* bw ,statement) reduced :constants constants
                                  :label (label+ label "-blk")
                                  :degree (domain-degree domain :block-phase t))))
                  (let ((finals (loop for (symbol) in tables collect (cons symbol (receive-row tr d))))
                        (weight (if (> k 1)
                                    (reduce #'ext+ (map 'list #'ext* bw (eq-table rbn d)))
                                    (svref bw 0))))
                    (unless (ext= reduced (ext* weight (value finals)))
                      (reject :domain "~A: leaf/zerocheck sumcheck failed" (domain-name domain)))
                    (values (domain-claims domain rho rbn (challenge tr) finals) roots)))
                (let ((finals (loop for (symbol column) in tables
                                    collect (cons symbol (receive-rows tr (if (column-shared column) 1 k) d)))))
                  (flet ((block-finals (b)
                           "Each table's final in block B (a shared table has one)."
                           (loop for (symbol . values) in finals
                                 collect (cons symbol (aref values (if (> (length values) 1) b 0))))))
                    ;; sum_b bw_b F(finals of block b)
                    (unless (ext= reduced
                                  (reduce #'ext+ (loop for b below k
                                                       collect (ext* (svref bw b) (value (block-finals b))))))
                      (reject :domain "~A: leaf/zerocheck sumcheck failed" (domain-name domain))))
                  (values (domain-claims domain rho '() (challenge tr) finals) roots)))))))))

;;; ------------------------------------------------------------------ prover (CPU reference)

(defun column-value (column b x n)
  "COLUMN's value in block B at the in-block index X (N bits)."
  (let ((data (column-data column))
        (c (column-c column))
        (shared (column-shared column)))
    (ecase (column-kind column)
      (:full (aref data (+ (if shared 0 (* b (expt 2 n))) x)))
      (:row (aref data (+ (if shared 0 (* b (expt 2 (- n c)))) (ash x (- c)))))
      (:col (aref data (+ (if shared 0 (* b (expt 2 c))) (ldb (byte c 0) x)))))))

(defun domain-leaves (domain z betas)
  "The leaves of each lookup's tree: a list of (p . q), each a vector of K blocks of 2^n EXTs,
p the numerator (1 if absent) and q = Z - tag - sum_i beta_i v_i."
  (let* ((d (length z))
         (n (domain-n domain))
         (k (domain-blocks domain)))
    (loop for lookup in (domain-lookups domain)
          collect (let ((p (make-array (* k (expt 2 n))))
                        (q (make-array (* k (expt 2 n)))))
                    (dotimes (b k)
                      (dotimes (x (expt 2 n))
                        (let ((values (mapcar (lambda (column) (cons (column-name column) (column-value column b x n)))
                                              (domain-columns domain)))
                              (i (+ (* b (expt 2 n)) x)))
                          (flet ((at (form) (evaluate form values)))
                            (setf (aref p i) (if (getf lookup :num) (base->ext (at (getf lookup :num)) d) (ext-one d))
                                  (aref q i) (reduce #'ext-
                                                     (cons (ext- z (base->ext (at (getf lookup :tag)) d))
                                                           (loop for v in (getf lookup :values)
                                                                 for beta in (coerce betas 'list)
                                                                 collect (ext-scale beta (at v))))))))))
                    (cons p q)))))

(defun table-values (table k n)
  "A sumcheck table's entries, K blocks of 2^(n-1) base elements: its column at the half's points."
  (destructuring-bind (symbol column half &rest rest) table
    (declare (ignore symbol rest))
    (let ((values (make-array (* k (expt 2 (1- n))))))
      (dotimes (b k values)
        (dotimes (x (expt 2 (1- n)))
          (setf (aref values (+ (* b (expt 2 (1- n))) x))
                (column-value column b (if half (+ x (* half (expt 2 (1- n)))) x) n)))))))

(defun send-rows (tr rows d)
  (send tr (make-array (list (length rows) d) :element-type '(unsigned-byte 32)
                                              :initial-contents (map 'list (lambda (e) (coerce e 'list)) rows))))

(defun prove-domain (tr domain z betas &key (label "dom"))
  "Prove a domain on the prover transcript TR (zk/domain.py _prove_host), the columns' data in the
domain.  Returns the column claims, as VERIFY-DOMAIN does."
  (let* ((d (transcript-degree tr))
         (n (domain-n domain))
         (k (domain-blocks domain))
         (lk (block-bits k))
         (lookups (length (domain-lookups domain)))
         (tables (domain-tables domain))
         (statement (domain-statement domain))
         r rb cp cq)
    (absorb tr (label+ label (label-name (domain-name domain))))
    (if (plusp lookups)
        (setf (values r rb cp cq) (prove-upper tr n k (domain-leaves domain z betas)))
        (setf r (challenges tr (1- n))
              rb (challenges tr lk)))
    (multiple-value-bind (as lam gams eta) (domain-challenges tr domain)
      (declare (ignorable as lam))
      (let ((constants (domain-constants domain z betas as lam gams eta))
            (bw (block-weights rb k d)))
        (multiple-value-bind (rho finals folded)
            (prove tr (1- n) statement (loop for table in tables
                                             collect (cons (first table) (table-values table k n)))
                   :eq r :constants constants :label label :block-weights bw
                   :degree (domain-degree domain))
          (if (domain-merge domain)
              (let ((rbn '()))
                (when (plusp lk)
                  (flet ((padded (values shared)
                           (concatenate 'vector values
                                        (loop repeat (- (expt 2 lk) k)
                                              collect (if shared (aref values 0) (make-ext d))))))
                    (setf (values rbn finals)
                          (prove tr lk `(* bw ,statement)
                                 (cons (cons 'bw (padded bw nil))
                                       (loop for (symbol column) in tables
                                             collect (cons symbol (padded (cdr (assoc symbol folded))
                                                                          (column-shared column)))))
                                 :constants constants :label (label+ label "-blk")
                                 :degree (domain-degree domain :block-phase t)))))
                (let ((finals (loop for (symbol) in tables collect (cons symbol (cdr (assoc symbol finals))))))
                  (dolist (final finals) (send-rows tr (list (cdr final)) d))
                  (domain-claims domain rho rbn (challenge tr) finals)))
              (let ((finals (loop for (symbol column) in tables
                                  collect (cons symbol (let ((values (cdr (assoc symbol folded))))
                                                         (if (column-shared column) (subseq values 0 1) values))))))
                (dolist (final finals) (send-rows tr (cdr final) d))
                (domain-claims domain rho '() (challenge tr) finals))))))))
