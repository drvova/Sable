;;;; pcs.lisp -- the polynomial commitment: rate-1/2 Reed-Solomon codes over KoalaBear, BLAKE3
;;;; Merkle trees with caps, and one mixed-length BaseFold opening for the whole proof
;;;; (zk/gpcs.py).
;;;;
;;;; A unit (commitment group) holds base-field tensors of K blocks of 2^c0 entries.  Each block is
;;;; split into columns of at most 2^cmax entries, grouped by size class c = clamp(c0, 5, cmax).
;;;; Per (set, class) -- a set is one unit, or the units of a merge set -- the codewords of all its
;;;; columns form one tree whose leaf l holds symbols 4l..4l+3 of every column (LEAF-DIGEST); the
;;;; classes of a set share one mixed-height tree where their levels allow (MIXED-LEVELS).  Each
;;;; tree sends its cap.
;;;;
;;;;   commit   the caps.
;;;;   open     per class, a sumcheck reduces the unit's claims to one point rho (in F_p^4); every
;;;;            column's value there is sent; alpha in F_p^8 combines them: g = sum alpha_k f_k,
;;;;            y = g(rho).
;;;;   segment  a segment's units are brought to one point per class (the cross-unit sumcheck).
;;;;   finish   lambda per (unit, class); a sumcheck over the NV global variables brings every
;;;;            claim to one point r; BaseFold proves the combined columns H_c at r, class c joining
;;;;            the folded codeword at level NV - c; oracles at even levels; grinding; QUERIES
;;;;            positions, each opening a leaf of every tree, checked against the caps and folded.
;;;;
;;;; MAKE-PCS's :PROVER makes a CPU reference prover with the same messages as zk/gpcs.py's.

(defpackage #:exacta.pcs
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.forms #:exacta.transcript
        #:exacta.blake3 #:exacta.rs #:exacta.sumcheck)
  (:export #:make-pcs #:pcs-declare #:pcs-merge #:pcs-segment-decl #:pcs-commit #:pcs-open
           #:pcs-segment #:pcs-finish #:pcs-nv #:+cmin+ #:+caplog+ #:+flog+))

(in-package #:exacta.pcs)

(defconstant +cmin+ 5 "The smallest size class.")
(defconstant +caplog+ 7 "A tree's cap has at most 2^CAPLOG entries.")
(defconstant +flog+ 5 "BaseFold sends its last 2^FLOG entries in the clear.")
(defconstant +chunk+ 64 "Columns per BLAKE3 call in a leaf digest.")

;;; ------------------------------------------------------------------ Merkle trees

(defun words-octets (rows)
  "Rows of field values as u32 LE bytes, row-major."
  (join-octets (loop for row in rows
                     append (loop for v in row collect (le-bytes v 4)))))

(defun leaf-digest (rows &optional previous)
  "The digest of a leaf whose rows (a list of lists of 4 field values: one per column, or one per
coordinate) are hashed 64 at a time and chained: h = d_1, then h = BLAKE3(h || d_i).  PREVIOUS:
the digest of the leaf's earlier part (the units before this one in a merge set)."
  (loop with h = previous
        for chunk on rows by (lambda (l) (nthcdr +chunk+ l))
        for d = (blake3-hash (words-octets (subseq chunk 0 (min +chunk+ (length chunk)))))
        do (setf h (if h (blake3-hash (concatenate 'octets h d)) d))
        finally (return h)))

(defun node (left right)
  (blake3-hash (concatenate 'octets left right)))

(defun cap-level (leaves)
  "The level of a tree of LEAVES leaves whose nodes form its cap."
  (max 0 (- (integer-length leaves) 1 +caplog+)))

(defun cap-rows (leaves)
  (ash leaves (- (cap-level leaves))))

(defun merkle-levels (leaves &optional injections)
  "The levels of the tree over LEAVES (digests), leaves first.  INJECTIONS: level -> digests hashed
into that level's nodes (node = H(H(left, right), injected)), for mixed-height trees."
  (loop with level = leaves
        for l from 1
        collect level into levels
        while (> (length level) 1)
        do (setf level (coerce (loop for i below (length level) by 2
                                     collect (node (aref level i) (aref level (1+ i))))
                               'vector))
           (let ((injected (cdr (assoc l injections))))
             (when injected
               (setf level (map 'vector #'node level injected))))
        finally (return levels)))

(defun path (levels i depth)
  "The siblings of leaf I up to DEPTH levels."
  (loop for level in levels
        repeat depth
        collect (aref level (logxor i 1))
        do (setf i (ash i -1))))

(defun check-path (h path i cap &optional injected-at)
  "Hash leaf digest H up PATH from leaf I and compare with CAP.  INJECTED-AT: level -> digest
hashed in after that level (mixed-height trees)."
  (loop for sibling in path
        for l from 1
        do (setf h (if (oddp i) (node sibling h) (node h sibling))
                 i (ash i -1))
           (let ((injected (cdr (assoc l injected-at))))
             (when injected (setf h (node h injected)))))
  (equalp h (aref cap i)))

(defun cap-octets (cap)
  (join-octets (coerce cap 'list)))

(defun octets-cap (octets)
  (coerce (loop for i below (length octets) by 32 collect (subseq octets i (+ i 32))) 'vector))

;;; ------------------------------------------------------------------ units

(defstruct (unit (:constructor %make-unit))
  name items info classes (tensors '()))

(defun class-of-size (c0 cmax)
  (min (max c0 +cmin+) cmax))

(defun make-unit (name items cmax)
  "ITEMS: (tensor-name K c0).  CLASSES: an alist, class -> its columns (tensor-name block j), the
columns of a block of 2^c0 > 2^cmax entries being its chunks j of 2^cmax."
  (let ((classes '()))
    (loop for (tensor k c0) in items
          do (dotimes (b k)
               (dotimes (j (expt 2 (max 0 (- c0 cmax))))
                 (let ((entry (assoc (class-of-size c0 cmax) classes)))
                   (if entry
                       (setf (cdr entry) (append (cdr entry) (list (list tensor b j))))
                       (push (list (class-of-size c0 cmax) (list tensor b j)) classes))))))
    (%make-unit :name name :items items
                :info (loop for (tensor k c0) in items collect (list tensor k c0))
                :classes (sort classes #'< :key #'car))))

(defun unit-columns (unit c cmax)
  "Class C's columns of UNIT, each a vector of 2^c base elements (zero beyond the data)."
  (loop for (tensor b j) in (cdr (assoc c (unit-classes unit)))
        collect (destructuring-bind (k c0) (rest (assoc tensor (unit-info unit) :test #'equal))
                  (declare (ignore k))
                  (let* ((data (cdr (assoc tensor (unit-tensors unit) :test #'equal)))
                         (length (expt 2 (min c0 cmax)))
                         (offset (+ (* b (expt 2 c0)) (* j length)))
                         (column (make-array (expt 2 c) :initial-element 0)))
                    (replace column data :start2 offset :end2 (+ offset length))))))

(defun class-claims (unit claims cmax)
  "CLAIMS (tensor point coefficients value): sum_b coefficient_b f~_{tensor,b}(point) = value, a
point of c0 EXT4s and coefficients an alist (block . EXT4) -> an alist class -> (z columns value),
z the point in the class's c variables and columns an alist (column index . coefficient)."
  (let ((out '()))
    (loop for (tensor point coefficients value) in claims
          do (destructuring-bind (k c0) (rest (assoc tensor (unit-info unit) :test #'equal))
               (declare (ignore k))
               (let* ((c (class-of-size c0 cmax))
                      (columns (cdr (assoc c (unit-classes unit))))
                      (index (lambda (b j) (position (list tensor b j) columns :test #'equal)))
                      (claim
                        (if (> c0 cmax)
                            (let ((hi (nthcdr cmax point)))
                              (list (subseq point 0 cmax)
                                    (loop for (b . w) in coefficients
                                          append (loop for j below (expt 2 (- c0 cmax))
                                                       collect (cons (funcall index b j)
                                                                     (reduce #'ext* (loop for x in hi for bit from 0
                                                                                          collect (if (logbitp bit j) x (ext- (ext-one 4) x)))
                                                                             :initial-value w))))
                                    value))
                            (list (append point (loop repeat (- c c0) collect (make-ext 4)))
                                  (loop for (b . w) in coefficients collect (cons (funcall index b 0) w))
                                  value))))
                 (let ((entry (assoc c out)))
                   (if entry
                       (setf (cdr entry) (append (cdr entry) (list claim)))
                       (push (list c claim) out))))))
    out))

(defun by-point (claims)
  "CLAIMS grouped by point, in first-occurrence order: ((z (columns value) ...) ...)."
  (let ((groups '()))
    (loop for (z columns value) in claims
          do (let ((group (assoc z groups :test #'equalp)))
               (if group
                   (setf (cdr group) (append (cdr group) (list (list columns value))))
                   (setf groups (append groups (list (list z (list columns value))))))))
    groups))

;;; ------------------------------------------------------------------ the commitment

(defstruct (uc (:constructor make-uc (unit c rho alpha y ncols)))
  "A (unit, class) after its opening: point RHO (EXT4s), ALPHA, value Y; the prover's combined
column G; LAM, SC and the verifier's column coefficients COEF after finish."
  unit c rho alpha y ncols g lam sc coef)

(defstruct (pu (:constructor make-pu (c rho y)))
  "A (segment, class) as one unit of the final opening: point RHO (EXT8s), value Y, the prover's
combined column H."
  c rho y h lam sc)

(defstruct (pcs (:constructor %make-pcs))
  tr prover cmax queries grind
  (units '()) (order '())
  (caps (make-hash-table :test 'equal))
  (ucs (make-hash-table :test 'equal))
  (msets '()) (mset-of (make-hash-table :test 'equal))
  (seg-decl '()) (seg-of (make-hash-table :test 'equal)) (segs '())
  (states (make-hash-table :test 'equal))       ; prover: (set . class) -> leaf digests so far
  (codewords (make-hash-table :test 'equal))    ; prover: (unit . class) -> column codewords
  (trees (make-hash-table :test 'equal))        ; prover: tree id -> levels
  basefold)

(defun make-pcs (tr &key prover (cmax 23) (queries 153) (grind 30))
  "A commitment on transcript TR (of degree 4).  PROVER: commit and open data (a CPU reference
prover); else verify."
  (%make-pcs :tr tr :prover prover :cmax cmax :queries queries :grind grind))

(defun pcs-unit (pcs name) (cdr (assoc name (pcs-units pcs) :test #'equal)))

(defun pcs-declare (pcs name items)
  "Declare unit NAME of ITEMS (tensor-name K c0)."
  (setf (pcs-units pcs) (append (pcs-units pcs) (list (cons name (make-unit name items (pcs-cmax pcs)))))
        (pcs-order pcs) (append (pcs-order pcs) (list name))))

(defun pcs-nv (pcs)
  "NV: the number of global variables, the largest class."
  (loop for (nil . unit) in (pcs-units pcs) maximize (reduce #'max (mapcar #'car (unit-classes unit)))))

(defun pcs-merge (pcs names)
  "Declare that the units NAMES (committed one after another, in this order) share their trees."
  (let ((sid (length (pcs-msets pcs))))
    (setf (pcs-msets pcs) (append (pcs-msets pcs) (list names)))
    (dolist (name names) (setf (gethash name (pcs-mset-of pcs)) sid))))

(defun pcs-segment-decl (pcs names)
  "Declare that the units NAMES form a segment (see PCS-SEGMENT)."
  (let ((i (length (pcs-seg-decl pcs))))
    (setf (pcs-seg-decl pcs) (append (pcs-seg-decl pcs) (list names)))
    (dolist (name names) (setf (gethash name (pcs-seg-of pcs)) i))))

(defun set-id (pcs name)
  (multiple-value-bind (sid found) (gethash name (pcs-mset-of pcs))
    (if found sid (list :unit name))))

(defun set-members (pcs sid)
  (if (consp sid) (list (second sid)) (nth sid (pcs-msets pcs))))

(defun set-classes (pcs sid)
  (sort (remove-duplicates (loop for name in (set-members pcs sid)
                                 append (mapcar #'car (unit-classes (pcs-unit pcs name)))))
        #'<))

(defun set-mixed (pcs sid)
  "The set's largest class cm and the classes in its mixed tree (those with cm - c <= its cap level)."
  (let* ((classes (set-classes pcs sid))
         (cm (car (last classes)))
         (lc (cap-level (expt 2 (1- cm)))))
    (values cm (remove-if-not (lambda (c) (<= (- cm c) lc)) classes))))

(defun set-trees (pcs sid)
  "The set's trees, in the order their caps are sent: the mixed tree, then the classes outside it."
  (multiple-value-bind (cm mixed) (set-mixed pcs sid)
    (declare (ignore cm))
    (cons (list :mixed sid)
          (loop for c in (set-classes pcs sid)
                unless (member c mixed) collect (list :single sid c)))))

(defun tree-class (pcs tid)
  "The class whose leaf index addresses tree TID (its largest class)."
  (if (eq (first tid) :mixed) (nth-value 0 (set-mixed pcs (second tid))) (third tid)))

(defun tree-levels (pcs tid digests)
  "The levels of tree TID from DIGESTS: (set . class) -> leaf digests."
  (if (eq (first tid) :single)
      (merkle-levels (gethash (cons (second tid) (third tid)) digests))
      (let ((sid (second tid)))
        (multiple-value-bind (cm mixed) (set-mixed pcs sid)
          (merkle-levels (gethash (cons sid cm) digests)
                         (loop for c in mixed
                               unless (= c cm) collect (cons (- cm c) (gethash (cons sid c) digests))))))))

(defun encode-column (column c)
  "The rate-1/2 codeword of a column of 2^c base elements: 2^(c+1) symbols."
  (encode column (1+ c)))

(defun leaf-rows (codewords l)
  "Leaf L's rows: symbols 4l..4l+3 of each codeword."
  (loop for cw in codewords collect (loop for s below 4 collect (aref cw (+ (* 4 l) s)))))

(defun pcs-commit (pcs name &optional tensors)
  "Commit unit NAME (the prover gives its TENSORS, an alist tensor-name -> base values): once the
last unit of its set is committed, the set's caps are sent."
  (let* ((tr (pcs-tr pcs))
         (unit (pcs-unit pcs name))
         (sid (set-id pcs name)))
    (when (pcs-prover pcs)
      (setf (unit-tensors unit) tensors)
      (loop for (c) in (unit-classes unit)
            do (let* ((codewords (mapcar (lambda (column) (encode-column column c))
                                         (unit-columns unit c (pcs-cmax pcs))))
                      (previous (gethash (cons sid c) (pcs-states pcs))))
                 (setf (gethash (cons name c) (pcs-codewords pcs)) codewords
                       (gethash (cons sid c) (pcs-states pcs))
                       (coerce (loop for l below (expt 2 (1- c))
                                     collect (leaf-digest (leaf-rows codewords l) (and previous (aref previous l))))
                               'vector)))))
    (when (equal name (car (last (set-members pcs sid))))
      (dolist (tid (set-trees pcs sid))
        (if (pcs-prover pcs)
            (let ((levels (tree-levels pcs tid (pcs-states pcs))))
              (setf (gethash tid (pcs-trees pcs)) levels
                    (gethash tid (pcs-caps pcs)) (nth (cap-level (expt 2 (1- (tree-class pcs tid)))) levels))
              (send-bytes tr (cap-octets (gethash tid (pcs-caps pcs)))))
            (setf (gethash tid (pcs-caps pcs)) (octets-cap (send-bytes tr)))))))
  name)

;;; ------------------------------------------------------------------ opening

(defun field-rows (exts)
  "EXTs as the rows of a field array."
  (make-array (list (length exts) (if exts (length (elt exts 0)) 0))
              :element-type '(unsigned-byte 32)
              :initial-contents (map 'list (lambda (e) (coerce e 'list)) exts)))

(defun array-rows (array)
  "The rows of a field array (n, D) as EXTs."
  (destructuring-bind (n d) (array-dimensions array)
    (loop for i below n collect (coerce-ext (loop for c below d collect (aref array i c))))))

(defun combine (tr group)
  "Per-claim coefficients beta: the columns' combined coefficients and the combined value."
  (let* ((betas (challenges tr (length group)))
         (columns (make-hash-table))
         (claim (make-ext 4)))
    (loop for (cd value) in group
          for beta in betas
          do (setf claim (ext+ claim (ext* beta value)))
             (loop for (k . coefficient) in cd
                   do (setf (gethash k columns) (ext+ (gethash k columns (make-ext 4)) (ext* beta coefficient)))))
    (values (sort (loop for k being the hash-keys of columns using (hash-value v) collect (cons k v))
                  #'< :key #'car)
            claim)))

(defun reduce-points (pcs c groups columns)
  "One sumcheck sum_z eq(z, x) g_z(x) over the class's C variables (zk/gpcs.py _reduce).  Returns
the point rho and a check of the columns' values there (an alist k -> EXT4)."
  (let ((tr (pcs-tr pcs))
        (combined '())
        (claim (make-ext 4)))
    (dolist (group groups)
      (multiple-value-bind (comb cz) (combine tr (rest group))
        (setf claim (ext+ claim cz))
        (push (cons (first group) comb) combined)))
    (setf combined (nreverse combined))
    (let* ((statement `(+ ,@(loop for i below (length combined)
                                  collect `(* ,(intern (format nil "E~D" i) '#:exacta.pcs)
                                              ,(intern (format nil "G~D" i) '#:exacta.pcs)))))
           (rho (if (pcs-prover pcs)
                    (prove tr c statement
                           (loop for (z . comb) in combined
                                 for i from 0
                                 collect (cons (intern (format nil "E~D" i) '#:exacta.pcs) (eq-table z 4))
                                 collect (cons (intern (format nil "G~D" i) '#:exacta.pcs)
                                               (let ((g (make-array (expt 2 c))))
                                                 (dotimes (x (expt 2 c) g)
                                                   (setf (aref g x)
                                                         (reduce #'ext+ (loop for (k . cf) in comb
                                                                              collect (ext-scale cf (aref (nth k columns) x)))
                                                                 :initial-value (make-ext 4)))))))
                           :label "pcs-red" :degree 2)
                    nil))
           (reduced nil))
      (unless (pcs-prover pcs)
        (setf (values rho reduced) (verify tr c statement claim :label "pcs-red" :degree 2)))
      (values rho
              (lambda (values)
                (or (pcs-prover pcs)
                    (ext= reduced
                          (reduce #'ext+ (loop for (z . comb) in combined
                                               collect (ext* (ext-eq z rho 4)
                                                             (reduce #'ext+ (loop for (k . cf) in comb
                                                                                  collect (ext* cf (cdr (assoc k values))))
                                                                     :initial-value (make-ext 4))))
                                 :initial-value (make-ext 4)))))))))

(defparameter *max-reduction-points* 64
  "zk/gpcs.py reduces more points than this in batches (its kernels' parameter space).")

(defun max-points (c)
  (max 2 (min *max-reduction-points* (floor (* 8 (expt 2 30)) (* 2 (expt 2 c) 4 4)))))

(defun column-values (columns rho keys)
  "The columns KEYS at RHO, as an alist k -> EXT4."
  (loop for k in keys collect (cons k (mle-eval (nth k columns) rho 4))))

(defun pcs-open (pcs name claims)
  "Open unit NAME at CLAIMS, a list of (tensor point coefficients value): sum_b coefficient_b
f~_{tensor,b}(point) = value (POINT: c0 EXT4s; COEFFICIENTS: an alist block -> EXT4)."
  (let* ((tr (pcs-tr pcs))
         (unit (pcs-unit pcs name))
         (cmax (pcs-cmax pcs))
         (by-class (class-claims unit claims cmax)))
    (absorb tr (concatenate 'octets (to-octets "pcs-open|") (to-octets (label-name name))))
    (loop for (c . members) in (unit-classes unit)
          for ncols = (length members)
          do (let ((columns (and (pcs-prover pcs) (unit-columns unit c cmax)))
                   (groups (by-point (cdr (assoc c by-class)))))
               (flet ((receive (keys rho)
                        "The columns KEYS' values at RHO: sent by the prover, received by the verifier."
                        (if (pcs-prover pcs)
                            (let ((values (column-values columns rho keys)))
                              (send tr (field-rows (mapcar #'cdr values)))
                              values)
                            (mapcar #'cons keys (array-rows (send tr nil (list (length keys) 4)))))))
                 ;; more points than a reduction takes: reduce them in batches
                 (loop while (> (length groups) (max-points c))
                       do (setf groups
                                (loop for batch on groups by (lambda (l) (nthcdr (max-points c) l))
                                      collect (let ((batch (subseq batch 0 (min (max-points c) (length batch)))))
                                                (multiple-value-bind (r check) (reduce-points pcs c batch columns)
                                                  (let* ((keys (sort (remove-duplicates
                                                                      (loop for (nil . cl) in batch
                                                                            append (loop for (cd) in cl append (mapcar #'car cd))))
                                                                     #'<))
                                                         (values (receive keys r)))
                                                    (unless (funcall check values)
                                                      (reject :pcs "~A: point reduction failed" name))
                                                    (cons r (loop for (k . v) in values
                                                                  collect (list (list (cons k (ext-one 4))) v)))))))))
                 (multiple-value-bind (rho check)
                     (if groups (reduce-points pcs c groups columns) (values (challenges tr c) nil))
                   (let ((values (receive (loop for k below ncols collect k) rho)))
                     (when (and check (not (funcall check values)))
                       (reject :pcs "~A: point reduction failed" name))
                     (let* ((alpha (challenges tr ncols 8))
                            (y (reduce #'ext+ (loop for a in alpha for (nil . v) in values
                                                    collect (ext* a (embed v 8)))
                                       :initial-value (make-ext 8)))
                            (uc (make-uc unit c rho alpha y ncols)))
                       (when (pcs-prover pcs)
                         (setf (uc-g uc) (combine-columns columns alpha)))
                       (setf (gethash (list name c) (pcs-ucs pcs)) uc)))))))
    name))

(defun combine-columns (columns coefficients)
  "sum_k coefficient_k column_k: a vector of EXT8s."
  (let ((g (make-array (length (first columns)))))
    (dotimes (x (length g) g)
      (setf (aref g x)
            (reduce #'ext+ (loop for column in columns for cf in coefficients
                                 collect (ext-scale cf (aref column x)))
                    :initial-value (make-ext 8))))))

;;; ------------------------------------------------------------------ the cross-unit sumcheck

(defun lam-scaled (lam y c nv)
  "lambda y 2^(NV - c): a (unit, class)'s share of the claim, its sum over the variables it lacks."
  (ext-scale (ext* lam y) (fpow 2 (- nv c))))

(defun class-state (ucs nv)
  "The prover's state per class: the constant C while the class's variables are not reached,
then the tables E (lambda eq(rho, x) per unit, concatenated) and V (its combined columns)."
  (loop for c in (sort (remove-duplicates (mapcar #'uc-or-pu-c ucs)) #'<)
        collect (let ((members (remove c ucs :key #'uc-or-pu-c :test-not #'=)))
                  (list c
                        :c (reduce #'ext+ (mapcar (lambda (u) (lam-scaled (uc-or-pu-lam u) (uc-or-pu-y u) c nv)) members)
                                   :initial-value (make-ext 8))
                        :e (concatenate 'vector
                                        (mapcan (lambda (u)
                                                  (map 'list (lambda (e) (ext* e (uc-or-pu-lam u)))
                                                       (eq-table (mapcar (lambda (x) (embed x 8)) (uc-or-pu-rho u)) 8)))
                                                members))
                        :v (concatenate 'vector (mapcan (lambda (u) (coerce (uc-or-pu-column u) 'list)) members))))))

(defun uc-or-pu-c (u) (if (uc-p u) (uc-c u) (pu-c u)))
(defun uc-or-pu-y (u) (if (uc-p u) (uc-y u) (pu-y u)))
(defun uc-or-pu-rho (u) (if (uc-p u) (uc-rho u) (pu-rho u)))
(defun uc-or-pu-lam (u) (if (uc-p u) (uc-lam u) (pu-lam u)))
(defun uc-or-pu-column (u) (if (uc-p u) (uc-g u) (pu-h u)))
(defun (setf uc-or-pu-lam) (value u) (if (uc-p u) (setf (uc-lam u) value) (setf (pu-lam u) value)))
(defun (setf uc-or-pu-sc) (value u) (if (uc-p u) (setf (uc-sc u) value) (setf (pu-sc u) value)))
(defun uc-or-pu-sc (u) (if (uc-p u) (uc-sc u) (pu-sc u)))

(defun dot-pairs (e v parity)
  "sum_i e_i v_i over the entries of parity PARITY (0: even, 1: odd, 2: 2 odd - even)."
  (reduce #'ext+
          (loop for i from 0 below (length e) by 2
                collect (flet ((pick (table)
                                 (ecase parity
                                   (0 (aref table i))
                                   (1 (aref table (1+ i)))
                                   (2 (ext- (ext-scale (aref table (1+ i)) 2) (aref table i))))))
                          (ext* (pick e) (pick v))))
          :initial-value (make-ext 8)))

(defun cross-rounds (pcs states nv claim)
  "The cross-unit sumcheck over NV variables (zk/gpcs.py _rounds): each round sends g at 0, 1 and
2 (in F_p^8); a class contributes its constant until its variables are reached.  Returns the point
and the reduced claim."
  (let ((tr (pcs-tr pcs))
        (inverse-2 (finv 2))
        (r '()))
    (dotimes (j nv (values r claim))
      (let ((ev (if (pcs-prover pcs)
                    (let ((ev (list (make-ext 8) (make-ext 8) (make-ext 8))))
                      (loop for (c . s) in states
                            do (if (< j (- nv c))
                                   (setf ev (mapcar (lambda (e) (ext+ e (ext-scale (getf s :c) inverse-2))) ev))
                                   (setf ev (loop for e in ev for parity from 0
                                                  collect (ext+ e (dot-pairs (getf s :e) (getf s :v) parity))))))
                      (array-rows (send tr (field-rows ev))))
                    (array-rows (send tr nil (list 3 8))))))
        (unless (ext= (ext+ (first ev) (second ev)) claim)
          (reject :pcs "cross-unit sumcheck round ~D failed" j))
        (let ((x (challenge tr 8)))
          (setf r (append r (list x))
                claim (lagrange ev x))
          (when (pcs-prover pcs)
            (loop for (c . s) in states
                  do (if (< j (- nv c))
                         (setf (getf s :c) (ext-scale (getf s :c) inverse-2))
                         (setf (getf s :e) (fold-table (getf s :e) x)
                               (getf s :v) (fold-table (getf s :v) x))))))))))

(defun point-tail (r c nv) (nthcdr (- nv c) r))

;;; ------------------------------------------------------------------ segments

(defun pcs-segment (pcs i)
  "Close segment I (all its units opened): the cross-unit sumcheck over its units brings their
claims to one point r; per class the prover sends y_c = H_c(r_c) for H_c = sum_t lambda_t
eq(rho_t, r_c) g_t, and the segment becomes one unit (a PU) per class of the final opening."
  (let* ((tr (pcs-tr pcs))
         (nv (pcs-nv pcs))
         (names (nth i (pcs-seg-decl pcs)))
         (keys (loop for name in names
                     append (loop for (c) in (unit-classes (pcs-unit pcs name)) collect (list name c))))
         (ucs (mapcar (lambda (k) (gethash k (pcs-ucs pcs))) keys)))
    (absorb tr "pcs-seg")
    (loop for uc in ucs for lam in (challenges tr (length ucs) 8) do (setf (uc-lam uc) lam))
    (let* ((classes (sort (remove-duplicates (mapcar #'uc-c ucs)) #'<))
           (states (and (pcs-prover pcs) (class-state ucs nv)))
           (claim (reduce #'ext+ (mapcar (lambda (u) (lam-scaled (uc-lam u) (uc-y u) (uc-c u) nv)) ucs)
                          :initial-value (make-ext 8))))
      (multiple-value-bind (r claim) (cross-rounds pcs states nv claim)
        (let* ((coefficients (mapcar (lambda (u) (ext* (uc-lam u)
                                                       (ext-eq (mapcar (lambda (x) (embed x 8)) (uc-rho u))
                                                               (point-tail r (uc-c u) nv) 8)))
                                     ucs))
               (ys (if (pcs-prover pcs)
                       (array-rows (send tr (field-rows
                                             (loop for c in classes
                                                   collect (reduce #'ext+ (loop for u in ucs for e in coefficients
                                                                                when (= (uc-c u) c)
                                                                                  collect (ext* e (mle-eval (uc-g u) (point-tail r c nv) 8)))
                                                                   :initial-value (make-ext 8))))))
                       (array-rows (send tr nil (list (length classes) 8))))))
          (unless (ext= (reduce #'ext+ ys :initial-value (make-ext 8)) claim)
            (reject :pcs "segment reduction failed"))
          (let ((segment (list :keys keys
                               :coefficients (mapcar #'cons keys coefficients)
                               :pus (loop for c in classes for y in ys
                                          collect (let ((pu (make-pu c (point-tail r c nv) y)))
                                                    (when (pcs-prover pcs)
                                                      (setf (pu-h pu)
                                                            (let ((members (loop for u in ucs for e in coefficients
                                                                                 when (= (uc-c u) c) collect (cons u e))))
                                                              (let ((h (make-array (expt 2 c))))
                                                                (dotimes (x (expt 2 c) h)
                                                                  (setf (aref h x)
                                                                        (reduce #'ext+ (loop for (u . e) in members
                                                                                             collect (ext* e (aref (uc-g u) x)))
                                                                                :initial-value (make-ext 8))))))))
                                                    pu)))))
            (setf (pcs-segs pcs) (append (pcs-segs pcs) (list segment)))
            segment))))))

;;; ------------------------------------------------------------------ BaseFold

(defun encode-ext (table c)
  "The codeword (2^(c+1) EXT8s) of a table of 2^c EXT8s, coordinate by coordinate."
  (let ((coordinates (loop for d below 8
                           collect (encode (map 'vector (lambda (e) (aref e d)) table) (1+ c)))))
    (coerce (loop for i below (expt 2 (1+ c))
                  collect (coerce-ext (loop for cw in coordinates collect (aref cw i))))
            'vector)))

(defun oracle-rows (codeword l)
  "Leaf L of an oracle codeword: one row of 4 symbols per coordinate."
  (loop for d below 8 collect (loop for s below 4 collect (aref (aref codeword (+ (* 4 l) s)) d))))

(defun basefold (pcs classes nv rr v h)
  "BaseFold proves sum_c H_c(r_c) = V (zk/gpcs.py _basefold): a sumcheck of eq(r, x) Phi(x),
Phi(x) = sum_c H_c(x_c), class c joining at level NV - c; oracle trees at even levels below the
final level NV - FLOG, whose table is sent in the clear; grinding; the query positions."
  (let* ((tr (pcs-tr pcs))
         (fl (max 0 (- nv +flog+)))
         (olv (loop for l from 2 below fl by 2 collect l))
         (join (loop for c in classes collect (cons (- nv c) c)))
         (prf (ext-one 8))
         (claim v)
         (oracles '())
         (chal '())
         (prover (pcs-prover pcs))
         yc ysum tt cw et)
    (when prover
      (setf yc (loop for c in classes collect (cons c (mle-eval (cdr (assoc c h)) (point-tail rr c nv) 8)))
            ysum (reduce #'ext+ (loop for (c . y) in yc when (> (- nv c) 0) collect y) :initial-value (make-ext 8))
            tt (make-array (expt 2 nv) :initial-contents (loop repeat (expt 2 nv) collect (make-ext 8)))
            cw (make-array (expt 2 (1+ nv)) :initial-contents (loop repeat (expt 2 (1+ nv)) collect (make-ext 8)))
            et (eq-table rr 8))
      (loop for (level . c) in join
            when (zerop level)
              do (setf tt (map 'vector #'ext+ tt (cdr (assoc c h)))
                       cw (map 'vector #'ext+ cw (encode-ext (cdr (assoc c h)) c)))))
    (let ((tw (twiddles (1+ nv))))
      (dotimes (i fl)
        (let* ((ri (elt rr i))
               (ev (if prover
                       (let* ((a (ext* prf ysum))
                              (one-r (ext- (ext-one 8) ri))
                              (eqx (list one-r ri (ext- (ext-scale ri 2) one-r))))
                         (array-rows (send tr (field-rows (loop for parity below 3 for e in eqx
                                                                collect (ext+ (dot-pairs tt et parity) (ext* a e)))))))
                       (array-rows (send tr nil (list 3 8))))))
          (unless (ext= (ext+ (first ev) (second ev)) claim)
            (reject :pcs "basefold sumcheck round ~D failed" i))
          (let ((x (challenge tr 8))
                (l (1+ i)))
            (setf chal (append chal (list x))
                  claim (lagrange ev x)
                  prf (ext* prf (ext-eq1 ri x)))
            (if prover
                (progn
                  (setf tt (fold-table tt x)
                        et (fold-table et x)
                        cw (fold cw (aref tw i) x))
                  (when (member l olv)
                    (let* ((leaves (coerce (loop for li below (floor (length cw) 4)
                                                 collect (leaf-digest (oracle-rows cw li)))
                                           'vector))
                           (levels (merkle-levels leaves))
                           (lc (cap-level (length leaves))))
                      (send-bytes tr (cap-octets (nth lc levels)))
                      (push (list l cw levels lc) oracles)))
                  (loop for (level . c) in join
                        when (= level l)
                          do (setf tt (map 'vector #'ext+ tt (cdr (assoc c h)))
                                   cw (map 'vector #'ext+ cw (encode-ext (cdr (assoc c h)) c))
                                   ysum (ext- ysum (cdr (assoc c yc))))))
                (when (member l olv)
                  (push (cons l (octets-cap (send-bytes tr))) oracles)))))))
    (let ((m (if prover
                 (progn (send tr (field-rows tt)) tt)
                 (coerce (array-rows (send tr nil (list (expt 2 (- nv fl)) 8))) 'vector))))
      (unless (ext= (ext* prf (if (< fl nv) (mle-eval m (nthcdr fl rr) 8) (aref m 0))) claim)
        (reject :pcs "basefold final claim"))
      (if prover
          (grind tr (pcs-grind pcs))
          (verify-grind tr (pcs-grind pcs)))
      (let ((qs (challenge-bits tr (1- nv) (pcs-queries pcs))))
        (when prover
          (dolist (l olv)
            (destructuring-bind (cw levels lc) (rest (assoc l oracles))
              (record tr (field-array-of-leaves (loop for q in qs collect (oracle-rows cw (ash q (- l))))))
              (record-bytes tr (join-octets (loop for q in qs append (path levels (ash q (- l)) lc)))))))
        (setf (pcs-basefold pcs) (list :fl fl :olv olv :join join :chal chal :m m :oracles oracles))
        qs))))

(defun field-array-of-leaves (leaves)
  "Q leaves of rows of 4 symbols as a field array (Q, rows, 4)."
  (let ((array (make-array (list (length leaves) (length (first leaves)) 4) :element-type '(unsigned-byte 32))))
    (loop for leaf in leaves for qi from 0
          do (loop for row in leaf for k from 0
                   do (loop for v in row for s from 0 do (setf (aref array qi k s) v))))
    array))

;;; ------------------------------------------------------------------ finish

(defun set-last (pcs keys)
  "Per key, true if it is the last key of its set in KEYS order."
  (let ((last (make-hash-table :test 'equal)))
    (dolist (k keys) (setf (gethash (set-id pcs (first k)) last) k))
    (mapcar (lambda (k) (equal (gethash (set-id pcs (first k)) last) k)) keys)))

(defun pcs-finish (pcs)
  "The final opening (zk/gpcs.py finish): the cross-unit sumcheck over the units outside segments
and the segments' classes, BaseFold, grinding and the query openings.  Returns T; a failed check
signals VERIFICATION-FAILURE."
  (let* ((tr (pcs-tr pcs))
         (nv (pcs-nv pcs))
         (keys (loop for name in (pcs-order pcs)
                     append (loop for (c) in (unit-classes (pcs-unit pcs name)) collect (list name c))))
         (gkeys (remove-if (lambda (k) (nth-value 1 (gethash (first k) (pcs-seg-of pcs)))) keys))
         (pus (loop for segment in (pcs-segs pcs) append (getf segment :pus)))
         (members (append (mapcar (lambda (k) (gethash k (pcs-ucs pcs))) gkeys) pus)))
    (absorb tr "pcs-finish")
    (loop for u in members for lam in (challenges tr (length members) 8) do (setf (uc-or-pu-lam u) lam))
    (let ((classes (sort (remove-duplicates (mapcar #'uc-or-pu-c members)) #'<))
          (claim (reduce #'ext+ (mapcar (lambda (u) (lam-scaled (uc-or-pu-lam u) (uc-or-pu-y u) (uc-or-pu-c u) nv))
                                        members)
                         :initial-value (make-ext 8))))
      (multiple-value-bind (rr v) (cross-rounds pcs (and (pcs-prover pcs) (class-state members nv)) nv claim)
        (dolist (u members)
          (setf (uc-or-pu-sc u) (ext* (uc-or-pu-lam u)
                                      (ext-eq (mapcar (lambda (x) (embed x 8)) (uc-or-pu-rho u))
                                              (point-tail rr (uc-or-pu-c u) nv) 8))))
        ;; the columns' coefficients in the combined columns H_c
        (dolist (k gkeys)
          (let ((uc (gethash k (pcs-ucs pcs))))
            (setf (uc-coef uc) (mapcar (lambda (a) (ext* (uc-sc uc) a)) (uc-alpha uc)))))
        (dolist (segment (pcs-segs pcs))
          (dolist (k (getf segment :keys))
            (let* ((uc (gethash k (pcs-ucs pcs)))
                   (pu (find (uc-c uc) (getf segment :pus) :key #'pu-c))
                   (s2 (ext* (pu-sc pu) (cdr (assoc k (getf segment :coefficients) :test #'equal)))))
              (setf (uc-coef uc) (mapcar (lambda (a) (ext* s2 a)) (uc-alpha uc))))))
        (let* ((h (and (pcs-prover pcs)
                       (loop for c in classes
                             collect (cons c (let ((h (make-array (expt 2 c) :initial-contents
                                                                  (loop repeat (expt 2 c) collect (make-ext 8)))))
                                               (dolist (u members h)
                                                 (when (= (uc-or-pu-c u) c)
                                                   (setf h (map 'vector (lambda (a b) (ext+ a (ext* (uc-or-pu-sc u) b)))
                                                                h (uc-or-pu-column u))))))))))
               (qs (basefold pcs classes nv rr v h)))
          (if (pcs-prover pcs)
              (open-queries pcs keys qs nv)
              (verify-queries pcs keys classes nv qs))
          t)))))

(defun open-queries (pcs keys qs nv)
  "Prover: record each (unit, class)'s queried leaves and, after a set's last, its trees' paths."
  (let ((tr (pcs-tr pcs)))
    (loop for k in keys
          for last in (set-last pcs keys)
          do (destructuring-bind (name c) k
               (let ((codewords (gethash (cons name c) (pcs-codewords pcs))))
                 (record tr (field-array-of-leaves (loop for q in qs collect (leaf-rows codewords (ash q (- c nv))))))
                 (when last
                   (dolist (tid (set-trees pcs (set-id pcs name)))
                     (let* ((tc (tree-class pcs tid))
                            (levels (gethash tid (pcs-trees pcs)))
                            (lc (cap-level (expt 2 (1- tc)))))
                       (record-bytes tr (join-octets (loop for q in qs append (path levels (ash q (- tc nv)) lc))))))))))))

(defun receive-paths (tr q leaves)
  "Q Merkle paths of a tree of LEAVES leaves, as lists of digests: the depth is the tree's."
  (let* ((depth (cap-level leaves))
         (octets (record-bytes tr)))
    (unless (= (length octets) (* q depth 32))
      (reject :pcs "merkle path length"))
    (loop for qi below q
          collect (loop for l below depth
                        collect (subseq octets (* 32 (+ (* qi depth) l)) (* 32 (+ (* qi depth) l 1)))))))

(defun receive-leaves (tr q rows)
  "Q leaves of ROWS rows of 4 symbols, as lists of rows."
  (let ((m (record tr nil (list q rows 4))))
    (loop for qi below q
          collect (loop for k below rows collect (loop for s below 4 collect (aref m qi k s))))))

(defun verify-queries (pcs keys classes nv qs)
  "Check the query openings (zk/gpcs.py _verify_queries): each opened leaf against its tree's cap,
and the folding of the combined codeword through the oracles to the final table's codeword."
  (destructuring-bind (&key fl olv join chal m oracles) (pcs-basefold pcs)
    (let* ((tr (pcs-tr pcs))
           (q (length qs))
           (oracle-leaves '())
           (injected (loop for c in classes
                           collect (cons c (coerce (loop repeat q collect (loop repeat 4 collect (make-ext 8))) 'vector))))
           (partial (make-hash-table :test 'equal)))
      ;; the oracles' leaves and paths
      (dolist (l olv)
        (let ((leaves (receive-leaves tr q 8))
              (paths (receive-paths tr q (expt 2 (- nv 1 l))))
              (cap (cdr (assoc l oracles))))
          (unless (= (length cap) (cap-rows (expt 2 (- nv 1 l))))
            (reject :pcs "oracle opening shape"))
          (loop for leaf in leaves for p in paths for query in qs
                unless (check-path (leaf-digest leaf) p (ash query (- l)) cap)
                  do (reject :pcs "oracle merkle path"))
          (push (cons l leaves) oracle-leaves)))
      ;; every (unit, class)'s leaves; after a set's last, its trees' paths
      (loop for k in keys
            for last in (set-last pcs keys)
            do (destructuring-bind (name c) k
                 (let* ((uc (gethash k (pcs-ucs pcs)))
                        (sid (set-id pcs name))
                        (leaves (receive-leaves tr q (uc-ncols uc)))
                        (previous (gethash (cons sid c) partial)))
                   (setf (gethash (cons sid c) partial)
                         (loop for leaf in leaves for qi from 0
                               collect (leaf-digest leaf (and previous (nth qi previous)))))
                   (when last
                     (dolist (tid (set-trees pcs sid))
                       (let* ((tc (tree-class pcs tid))
                              (mixed (if (eq (first tid) :mixed) (nth-value 1 (set-mixed pcs sid)) (list tc)))
                              (paths (receive-paths tr q (expt 2 (1- tc))))
                              (cap (gethash tid (pcs-caps pcs))))
                         (unless (= (length cap) (cap-rows (expt 2 (1- tc))))
                           (reject :pcs "cap size"))
                         (loop for p in paths for query in qs for qi from 0
                               unless (check-path (nth qi (gethash (cons sid tc) partial)) p (ash query (- tc nv)) cap
                                                  (loop for cc in mixed
                                                        unless (= cc tc)
                                                          collect (cons (- tc cc) (nth qi (gethash (cons sid cc) partial)))))
                                 do (reject :pcs "merkle path ~A/~D" name tc))))
                     (dolist (cc (set-classes pcs sid)) (remhash (cons sid cc) partial)))
                   ;; inj_c += sum_k coef_k leaf_k
                   (let ((inj (cdr (assoc c injected))))
                     (loop for leaf in leaves for qi from 0
                           do (setf (aref inj qi)
                                    (loop for s below 4
                                          for held in (aref inj qi)
                                          collect (reduce #'ext+ (loop for row in leaf for cf in (uc-coef uc)
                                                                       collect (ext-scale cf (nth s row)))
                                                          :initial-value held))))))))
      ;; fold each query's symbols down to the final table's codeword
      (let ((tw (twiddles (1+ nv)))
            (menc (encode-ext m (- nv fl)))
            (inverse-2 (finv 2)))
        (loop for query in qs for qi from 0
              do (let ((base (* 4 query))
                       (held (loop repeat 4 collect (make-ext 8))))
                   (flet ((inject (level)
                            (loop for (lv . c) in join
                                  when (= lv level)
                                    do (let ((cb (* 4 (ash query (- level))))
                                             (symbols (aref (cdr (assoc c injected)) qi)))
                                         (setf held (loop for h in held for s from 0
                                                          collect (ext+ h (nth (+ (- base cb) s) symbols))))))))
                     (inject 0)
                     (dotimes (l fl)
                       (setf held (loop for (a b) on held by #'cddr
                                        for t2 from 0
                                        collect (let* ((s (+ (floor base 2) t2))
                                                       (f0 (ext-scale (ext+ a b) inverse-2))
                                                       (f1 (ext-scale (ext- a b) (finv (fmul 2 (aref (aref tw l) s))))))
                                                  (ext+ f0 (ext* (elt chal l) (ext- f1 f0)))))
                             base (floor base 2))
                       (let ((ln (1+ l)))
                         (when (member ln olv)
                           (let* ((leaf (nth qi (cdr (assoc ln oracle-leaves))))
                                  (ob (* 4 (ash query (- ln))))
                                  (symbol (lambda (s) (coerce-ext (loop for row in leaf collect (nth s row))))))
                             (unless (ext= (funcall symbol (- base ob)) (first held))
                               (reject :pcs "fold consistency"))
                             (setf held (loop for s below 4 collect (funcall symbol s))
                                   base ob)))
                         (inject ln)))
                     (loop for h in held for s from 0
                           unless (ext= h (aref menc (+ base s)))
                             do (reject :pcs "final codeword consistency")))))))))
