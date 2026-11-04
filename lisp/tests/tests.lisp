;;;; tests.lisp -- the port against the Python implementation (vectors.sexp, written by
;;;; gen-vectors.py), plus the behaviour of the conditions and restarts.

(in-package #:sable.tests)

(defun hex (s)
  "A hex string -> OCTETS."
  (let ((out (make-octets (floor (length s) 2))))
    (dotimes (i (length out) out)
      (setf (aref out i) (parse-integer s :start (* 2 i) :end (+ 2 (* 2 i)) :radix 16)))))

(defun read-vectors (path)
  (with-open-file (in path)
    (with-standard-io-syntax
      (let ((*read-eval* nil)
            (*package* (find-package '#:sable.tests)))
        (read in)))))

(defvar *vectors*
  (read-vectors (asdf:system-relative-pathname "sable" "tests/vectors.sexp"))
  "The test vectors, read when the tests are loaded (so a saved image carries them).")

(defun vectors (kind)
  (loop for v in *vectors* when (eq (first v) kind) collect (rest v)))

(defmacro do-vectors ((kind &rest keys) &body body)
  "Run BODY for each vector of KIND, binding each of KEYS to the value of the same-named keyword."
  (let ((v (gensym)))
    `(let ((n 0))
       (dolist (,v (vectors ,kind))
         (incf n)
         (let ,(mapcar (lambda (k) `(,k (getf ,v ,(intern (symbol-name k) :keyword)))) keys)
           ,@body))
       (check (plusp n) "no vectors of kind ~S" ,kind))))

(defun e (list) (ext-coerce list))
(defun exts (lists) (map 'simple-vector #'e lists))
(defun same (ext list) (equal (coerce ext 'list) list))

;;; ------------------------------------------------------------------ field

(deftest field-tower
  (check (check-tower))
  (do-vectors (:pow a e res)
    (check (= (fpow a e) res)))
  (do-vectors (:ext a b add sub mul inv eq1)
    (let ((a (e a)) (b (e b)))
      (check (same (ext+ a b) add))
      (check (same (ext- a b) sub))
      (check (same (ext* a b) mul))
      (check (same (ext-inv a) inv))
      (check (ext= (ext* a (ext-inv a)) (ext-one (length a))))
      (check (same (ext-eq1 a b) eq1)))))

(deftest field-multilinear
  (do-vectors (:lagrange vals x res)
    (check (same (lagrange (exts vals) (e x)) res)))
  (do-vectors (:eq d a b res)
    (check (same (ext-eq (exts a) (exts b) d) res)))
  (do-vectors (:mle d r eqtab base base-res ext ext-res)
    (let ((r (exts r)))
      (check (equalp (map 'list (lambda (x) (coerce x 'list)) (eq-table r d)) eqtab))
      (check (same (mle-eval (coerce base 'vector) r d) base-res))
      (check (same (mle-eval (exts ext) r d) ext-res)))))

(deftest field-inverse-of-zero
  ;; no handler: the error, a DIVISION-BY-ZERO as well as a SABLE-ERROR
  (let ((c (check-signals field-division-by-zero (finv 0))))
    (check (typep c 'division-by-zero))
    (check (typep c 'sable-error)))
  ;; the restarts: RETURN-ZERO (the Python convention) and USE-VALUE
  (check (= 0 (handler-bind ((field-division-by-zero #'return-zero)) (finv 0))))
  (check (= 7 (handler-bind ((field-division-by-zero (lambda (c) (use-value 7 c)))) (finv +p+))))
  (check (every #'zerop (handler-bind ((field-division-by-zero #'return-zero)) (ext-inv (ext-zero 8)))))
  (check (= 5 (finv (finv 5)))))

(deftest invalid-parameter-restart
  (check-signals invalid-parameter (sable.rs:root-of-unity 30))
  (check (= (sable.rs:root-of-unity 3)
            (handler-bind ((invalid-parameter (lambda (c) (use-value 3 c))))
              (sable.rs:root-of-unity 30))))
  (check-signals invalid-parameter (make-transcript "x" 5)))

;;; ------------------------------------------------------------------ BLAKE3 and the transcript

(deftest blake3-standard
  (do-vectors (:blake3 input digest xof)
    (let ((in (hex input)))
      (check (equalp (blake3-hash in) (hex digest)) "~D-byte input" (length in))
      (check (equalp (blake3-hash in 131) (hex xof)))
      ;; incremental updates in odd pieces give the same digest
      (let ((h (make-blake3)))
        (loop for s from 0 below (length in) by 333
              do (blake3-update h in :start s :end (min (length in) (+ s 333))))
        (check (equalp (blake3-digest h) (hex digest)))))))

(deftest blake3-sequential
  (do-vectors (:b3s chunks digest xof)
    (let ((s (b3s-init)))
      (dolist (c chunks) (b3s-update s (hex c)))
      (check (equalp (b3s-digest s 32) (hex digest)))
      (check (equalp (b3s-digest (b3s-copy s) 200) (hex xof))))))

(deftest transcript-replay
  (do-vectors (:transcript label ops)
    (let ((tr (make-transcript (hex label) 8)))
      (dolist (op ops)
        (destructuring-bind (kind &rest args) op
          (cond ((string= kind "absorb") (absorb tr (hex (first args))))
                ((string= kind "absorb-ext") (absorb-field tr (coerce (first args) 'vector)))
                ((string= kind "challenge-base")
                 (check (equal (challenge-base tr (first args)) (second args))))
                ((string= kind "challenge")
                 (check (same (challenge tr (first args)) (second args))))
                ((string= kind "challenge-bits")
                 (check (equal (challenge-bits tr (first args) (second args)) (third args))))
                ((string= kind "digest")
                 (check (equalp (b3s-digest (transcript-state tr) 32) (hex (first args)))))
                (t (check nil "unknown op ~S" kind)))))))
  (do-vectors (:grind base bits nonce ok)
    (check (eq ok (check-grind (hex base) nonce bits))))
  (do-vectors (:label-name name res)
    (check (string= (label-name name) res) "~S -> ~S" name (label-name name))))

(deftest transcript-grind-roundtrip
  (let* ((p (make-transcript "pow" 4))
         (nonce (grind p 6))
         (v (make-transcript "pow" 4 :messages (transcript-messages p))))
    (check (= nonce (verify-grind v 6)))
    (check (transcript-finished-p v))
    ;; a nonce that does not meet the bound is rejected
    (let* ((bad (copy-seq (aref (transcript-messages p) 0))))
      (setf (aref bad 0) (logxor (aref bad 0) 1))
      (unless (check-grind (b3s-digest (transcript-state (make-transcript "pow" 4)) 32)
                           (loop for i below 8 sum (ash (aref bad i) (* 8 i))) 6)
        (check-signals verification-failure
                       (verify-grind (make-transcript "pow" 4 :messages (list bad)) 6))))))

;;; ------------------------------------------------------------------ proof bytes

(deftest proofio-roundtrip
  (do-vectors (:proofio bytes msgs)
    (let* ((b (hex bytes))
           (parsed (sable.proofio:read-proof b)))
      (check (= (length parsed) (length msgs)))
      (loop for m across parsed
            for (kind . spec) in msgs
            do (if (string= kind "bytes")
                   (check (equalp m (hex (first spec))))
                   (progn (check (equal (array-dimensions m) (first spec)))
                          (check (equal (loop for i below (array-total-size m) collect (row-major-aref m i))
                                        (second spec))))))
      (check (equalp (sable.proofio:write-proof parsed) b)))))

(deftest proofio-rejects-malformed
  (let ((reasons '(("magic" . :magic) ("truncated" . :truncated) ("trailing" . :trailing-bytes)
                   ("tag" . :unknown-tag) ("range" . :value-range) ("rank" . :rank)
                   ;; zk/proofio.py rejects this one only through numpy's reshape (its np.prod of
                   ;; the shape wraps to 0); the exact product here exceeds the input
                   ("huge" . :truncated))))
    (do-vectors (:proofio-bad why bytes)
      (let ((c (check-signals malformed-proof (sable.proofio:read-proof (hex bytes)))))
        (check (eq (malformed-proof-reason c) (cdr (assoc why reasons :test #'string=)))
               "~A: got ~S (~A)" why (malformed-proof-reason c) c)))))

;;; ------------------------------------------------------------------ polynomials, Reed-Solomon

(deftest poly-arithmetic
  (do-vectors (:poly eval degree vars const terms)
    (let* ((x (sable.poly:poly-var 0)) (y (sable.poly:poly-var 1)) (z (sable.poly:poly-var 2))
           (p (sable.poly:poly- (sable.poly:poly+ (sable.poly:poly* (sable.poly:poly+ x 3)
                                                                    (sable.poly:poly- y x) z)
                                                  7)
                                (sable.poly:poly-expt x 2)))
           (vals (vector 11 (- +p+ 5) (ash 1 30))))
      (check (= (sable.poly:poly-eval p vals) eval))
      (check (= (sable.poly:poly-degree p) degree))
      (check (equal (sable.poly:poly-vars p) vars))
      (check (= (sable.poly:poly-const-term p) const))
      (check (sable.poly:poly= p (sable.poly:make-poly (mapcar (lambda (kv) (cons (first kv) (second kv)))
                                                               terms))))
      (check (sable.poly:poly= (sable.poly:poly-rename p (lambda (i) (- 2 i)))
                               (sable.poly:poly-rename (sable.poly:poly-rename p (lambda (i) i))
                                                       '((0 . 2) (1 . 1) (2 . 0)))))
      (check (equal (sable.poly:poly-vars (sable.poly:lparam "Dyn")) '(-3))))))

(deftest reed-solomon
  (do-vectors (:rs-encode n root msg cw points tw)
    (check (= (sable.rs:root-of-unity n) root))
    (check (equal (coerce (sable.rs:domain-points n) 'list) points))
    (check (equal (map 'list (lambda (v) (coerce v 'list)) (sable.rs:twiddles n)) tw))
    (check (equal (coerce (sable.rs:encode msg n) 'list) cw))
    (check (equal (coerce (sable.rs:brute-eval msg n) 'list) cw)))
  (do-vectors (:rs-fold n cw r res)
    (let ((folded (sable.rs:fold (exts cw) (aref (sable.rs:twiddles n) 0) (e r))))
      (check (equal (map 'list (lambda (v) (coerce v 'list)) folded) res)))))

;;; ------------------------------------------------------------------ sumcheck

(deftest sumcheck-against-python
  (do-vectors (:sumcheck d n tables eq-r claim proof rho reduced finals)
    (let* ((msgs (sable.proofio:read-proof (hex proof)))
           (eq-r (and eq-r (exts eq-r)))
           (tr (make-transcript "sc-vec" d :messages msgs)))
      ;; the verifier on the Python prover's proof
      (multiple-value-bind (vrho vred) (sable.sumcheck:verify tr n (length tables) (e claim) :eq-r eq-r)
        (check (equal (map 'list (lambda (v) (coerce v 'list)) vrho) rho))
        (check (same vred reduced))
        (check (sable.sumcheck:check-reduced vred (reduce #'ext* (exts finals))))
        (check (transcript-finished-p tr)))
      ;; the Lisp prover writes the same bytes as the Python prover
      (let ((ptr (make-transcript "sc-vec" d))
            (tabs (mapcar #'exts tables)))
        (check (same (sable.sumcheck:product-claim tabs n d :eq-r eq-r) claim))
        (multiple-value-bind (prho pfinals) (sable.sumcheck:prove-product ptr n tabs :eq-r eq-r)
          (check (equal (map 'list (lambda (v) (coerce v 'list)) prho) rho))
          (check (equal (mapcar (lambda (v) (coerce v 'list)) pfinals) finals))
          (check (equalp (sable.proofio:write-proof (transcript-messages ptr)) (hex proof))))))))

(deftest sumcheck-rejects-forgeries
  (let* ((v (first (vectors :sumcheck)))
         (d (getf v :d)) (n (getf v :n)) (ntab (length (getf v :tables)))
         (claim (e (getf v :claim)))
         (finals (exts (getf v :finals))))
    (flet ((run (msgs &optional (claim claim))
             (let ((tr (make-transcript "sc-vec" d :messages msgs)))
               (multiple-value-bind (rho red) (sable.sumcheck:verify tr n ntab claim)
                 (declare (ignore rho))
                 (sable.sumcheck:check-reduced red (reduce #'ext* finals))))))
      (let ((good (sable.proofio:read-proof (hex (getf v :proof)))))
        (check (run good))
        ;; a wrong claim
        (check-signals verification-failure (run good (ext+ claim (ext-one d))))
        ;; each round message changed in one coordinate
        (dotimes (j n)
          (let ((msgs (sable.proofio:read-proof (hex (getf v :proof)))))
            (setf (aref (aref msgs j) 0 1) (fadd (aref (aref msgs j) 0 1) 1))
            (check-signals verification-failure (run msgs))))
        ;; a message of the wrong shape, and a missing one
        (let ((msgs (copy-seq good)))
          (setf (aref msgs 0) (make-array (list 1 d) :element-type '(unsigned-byte 32) :initial-element 0))
          (check (typep (check-signals message-shape-error (run msgs)) 'verification-failure)))
        (check-signals transcript-exhausted (run (subseq good 0 (1- n))))
        (let ((msgs (copy-seq good)))
          (setf (aref msgs 0) (make-octets 4))
          (check-signals message-shape-error (run msgs)))))))

(deftest eval-terms-matches-product
  (let* ((d 8)
         (fs (list (vector (e '(1 2 3 4 5 6 7 8)) (e '(9 9 9 9 9 9 9 9)))
                   (vector (e '(3 0 0 0 0 0 0 0)))))
         (c (ext-one d)))
    ;; sum_b f0[b] * f1[0]
    (check (ext= (sable.sumcheck:eval-terms (list (list* c '(0 1))) fs d)
                 (ext* (ext+ (aref (first fs) 0) (aref (first fs) 1)) (aref (second fs) 0))))))
