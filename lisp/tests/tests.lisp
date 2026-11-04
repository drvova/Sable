;;;; tests.lisp -- the port against the Python implementation (vectors.sexp, written by
;;;; gen-vectors.py), forms as data and as code, and the conditions and restarts.

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
  "The vectors of KIND, as plists; there must be some."
  (or (loop for v in *vectors* when (eq (first v) kind) collect (rest v))
      (error "no test vectors of kind ~S" kind)))

(defun each-vector (kind function)
  "Apply FUNCTION to the keyword arguments of each vector of KIND."
  (dolist (v (vectors kind))
    (apply function v)))

(defun e (list) (coerce-ext list))
(defun exts (lists) (map 'vector #'e lists))
(defun same (ext list) (equal (coerce ext 'list) list))
(defun lists (exts) (map 'list (lambda (x) (coerce x 'list)) exts))

(defun symbols (prefix n)
  "PREFIX0 .. PREFIX(n-1), interned here: column or table variables."
  (loop for i below n collect (intern (format nil "~A~D" prefix i) '#:sable.tests)))

;;; ------------------------------------------------------------------ field

(defun test-field-tower ()
  (assert (check-tower))
  (each-vector :pow (lambda (&key a e res) (assert (= (fpow a e) res))))
  (each-vector :ext
               (lambda (&key a b add sub mul inv eq1 &allow-other-keys)
                 (let ((a (e a)) (b (e b)))
                   (assert (same (ext+ a b) add))
                   (assert (same (ext- a b) sub))
                   (assert (same (ext* a b) mul))
                   (assert (same (ext-inv a) inv))
                   (assert (same (ext-eq1 a b) eq1))))))

(defun test-field-tower-is-code ()
  ;; the code behind EXT* is a form that can be read ...
  (let ((code (tower-code 'mul* 4)))
    (assert (eq (first code) 'lambda))
    (assert (equal (second code) '(sable.field::a sable.field::b sable.field::out))))
  ;; ... and any power-of-two degree works, though only 4 and 8 are used (16 works too, but
  ;; its 4,600-cons inverse takes seconds to compile)
  (let ((state (sb-ext:seed-random-state 7)))
    (dolist (d '(2 4 8))
      (let ((a (coerce-ext (loop repeat d collect (random +p+ state)))))
        (assert (ext= (ext* a (ext-inv a)) (ext-one d)))
        (assert (ext= (ext-expt a 3) (ext* a (ext* a a))))))))

(defun test-field-multilinear ()
  (each-vector :lagrange
               (lambda (&key vals x res &allow-other-keys)
                 (assert (same (lagrange (exts vals) (e x)) res))))
  (each-vector :eq
               (lambda (&key d a b res)
                 (assert (same (ext-eq (exts a) (exts b) d) res))))
  (each-vector :mle
               (lambda (&key d r eqtab base base-res ext ext-res)
                 (let ((r (exts r)))
                   (assert (equal (lists (eq-table r d)) eqtab))
                   (assert (same (mle-eval base r d) base-res))
                   (assert (same (mle-eval (exts ext) r d) ext-res))))))

(defun test-field-inverse-of-zero ()
  ;; no handler: the error, a DIVISION-BY-ZERO as well as a SABLE-ERROR
  (let ((c (signalled 'field-division-by-zero (lambda () (finv 0)))))
    (assert (typep c 'division-by-zero))
    (assert (typep c 'sable-error)))
  ;; the restarts: RETURN-ZERO (the Python convention) and USE-VALUE
  (assert (= 0 (handler-bind ((field-division-by-zero #'return-zero)) (finv 0))))
  (assert (= 7 (handler-bind ((field-division-by-zero (lambda (c) (use-value 7 c)))) (finv +p+))))
  (assert (every #'zerop (handler-bind ((field-division-by-zero #'return-zero))
                           (ext-inv (make-ext 8))))))

(defun test-invalid-parameter-restart ()
  (assert (signalled 'invalid-parameter (lambda () (sable.rs:root-of-unity 30))))
  (assert (= (sable.rs:root-of-unity 3)
             (handler-bind ((invalid-parameter (lambda (c) (use-value 3 c))))
               (sable.rs:root-of-unity 30))))
  (assert (signalled 'invalid-parameter (lambda () (make-transcript "x" 5)))))

;;; ------------------------------------------------------------------ forms

(defun test-forms-against-python ()
  ;; zk/poly.py's  (x + 3) (y - x) z + 7 - x^2  over columns 0, 1, 2
  (destructuring-bind (c0 c1 c2) (symbols "C" 3)
    (let ((form `(- (+ (* (+ ,c0 3) (- ,c1 ,c0) ,c2) 7) (expt ,c0 2)))
          (values `((,c0 . 11) (,c1 . ,(- +p+ 5)) (,c2 . ,(expt 2 30)))))
      (each-vector :poly
                   (lambda (&key eval degree vars const terms)
                     ;; the same polynomial as Python's term dictionary
                     (assert (form= form `(+ ,@(loop for (ids c) in terms
                                                     collect `(* ,c ,@(mapcar (lambda (i) (nth i (list c0 c1 c2))) ids))))))
                     (assert (= (evaluate form values) eval))
                     (assert (= (form-degree form) degree))
                     (assert (equal (form-variables form) (mapcar (lambda (i) (nth i (list c0 c1 c2))) vars)))
                     (assert (= (constant-term form) const))))
      ;; normalising is idempotent and keeps the polynomial
      (let ((normal (normalize form)))
        (assert (form= normal form))
        (assert (equal (normalize normal) normal)))
      ;; renaming is SUBLIS
      (let ((swapped (sublis `((,c0 . ,c2) (,c2 . ,c0)) form)))
        (assert (= (evaluate swapped `((,c0 . ,(expt 2 30)) (,c1 . ,(- +p+ 5)) (,c2 . 11)))
                   (evaluate form values)))))))

(defun test-forms-as-code ()
  (assert (equal (normalize '(* (+ a 1) (- a 1))) '(+ -1 (* a a))))
  (assert (equal (normalize '(- (* 2 x) x x)) 0))
  (assert (= (form-degree '(* (+ alpha t0) t1 t1) '(t0 t1)) 3))
  ;; over F_p^8, a form computes what the field functions do
  (let ((a (coerce-ext '(1 2 3 4 5 6 7 8)))
        (b (coerce-ext '(9 10 11 12 13 14 15 16))))
    (assert (ext= (evaluate '(+ (* a b) 1 (- b)) `((a . ,a) (b . ,b)) :degree 8)
                  (ext- (ext+ (ext* a b) (ext-one 8)) b)))
    (assert (equal (arithmetic-form '(* a (expt b 2)) :degree 8) '(ext* a (ext-expt b 2)))))
  (assert (signalled 'invalid-parameter (lambda () (expand '(/ a b)))))
  (assert (signalled 'invalid-parameter (lambda () (compile-form '(* a b) '(a))))))

;;; ------------------------------------------------------------------ BLAKE3 and the transcript

(defun test-blake3-standard ()
  (each-vector :blake3
               (lambda (&key input digest xof)
                 (let ((in (hex input)))
                   (assert (equalp (blake3-hash in) (hex digest)))
                   (assert (equalp (blake3-hash in 131) (hex xof)))
                   ;; incremental updates in odd pieces give the same digest
                   (let ((h (make-blake3)))
                     (loop for s from 0 below (length in) by 333
                           do (blake3-update h in :start s :end (min (length in) (+ s 333))))
                     (assert (equalp (blake3-digest h) (hex digest))))))))

(defun test-blake3-sequential ()
  (each-vector :b3s
               (lambda (&key chunks digest xof)
                 (let ((s (make-b3s)))
                   (dolist (c chunks) (b3s-update s (hex c)))
                   (assert (equalp (b3s-digest s 32) (hex digest)))
                   (assert (equalp (b3s-digest (b3s-copy s) 200) (hex xof)))))))

(defun test-transcript-replay ()
  (each-vector :transcript
               (lambda (&key label ops)
                 (let ((tr (make-transcript (hex label) 8)))
                   (loop for (kind . args) in ops
                         do (cond ((string= kind "absorb") (absorb tr (hex (first args))))
                                  ((string= kind "absorb-ext")
                                   (absorb tr (message-octets (to-field-array (first args)))))
                                  ((string= kind "challenge-base")
                                   (assert (equal (challenge-base tr (first args)) (second args))))
                                  ((string= kind "challenge")
                                   (assert (same (challenge tr (first args)) (second args))))
                                  ((string= kind "challenge-bits")
                                   (assert (equal (challenge-bits tr (first args) (second args)) (third args))))
                                  ((string= kind "digest")
                                   (assert (equalp (b3s-digest (transcript-state tr) 32) (hex (first args)))))
                                  (t (error "unknown transcript op ~S" kind)))))))
  (each-vector :grind
               (lambda (&key base bits nonce ok)
                 (assert (eq ok (check-grind (hex base) nonce bits)))))
  (each-vector :label-name
               (lambda (&key name res)
                 (assert (string= (label-name name) res)))))

(defun test-transcript-grind-roundtrip ()
  (let* ((p (make-transcript "pow" 4))
         (nonce (grind p 6))
         (v (make-transcript "pow" 4 :messages (transcript-messages p))))
    (assert (= nonce (verify-grind v 6)))
    (assert (transcript-finished-p v))
    ;; a nonce that does not meet the bound is rejected
    (let ((bad (copy-seq (aref (transcript-messages p) 0))))
      (setf (aref bad 0) (logxor (aref bad 0) 1))
      (unless (check-grind (b3s-digest (transcript-state (make-transcript "pow" 4)) 32) (le-integer bad) 6)
        (assert (signalled 'verification-failure
                           (lambda () (verify-grind (make-transcript "pow" 4 :messages (list bad)) 6))))))))

;;; ------------------------------------------------------------------ proof bytes

(defun test-proofio-roundtrip ()
  (each-vector :proofio
               (lambda (&key bytes msgs)
                 (let ((parsed (sable.proofio:decode-proof (hex bytes))))
                   (assert (= (length parsed) (length msgs)))
                   (loop for m across parsed
                         for (kind . spec) in msgs
                         do (if (string= kind "bytes")
                                (assert (equalp m (hex (first spec))))
                                (progn
                                  (assert (equal (array-dimensions m) (first spec)))
                                  (assert (equal (loop for i below (array-total-size m)
                                                       collect (row-major-aref m i))
                                                 (second spec))))))
                   (assert (equalp (sable.proofio:encode-proof parsed) (hex bytes)))))))

(defun test-proofio-rejects-malformed ()
  (let ((reasons '(("magic" . :magic) ("truncated" . :truncated) ("trailing" . :trailing-bytes)
                   ("tag" . :unknown-tag) ("range" . :value-range) ("rank" . :rank)
                   ;; zk/proofio.py rejects this one only through numpy's reshape (its np.prod of
                   ;; the shape wraps to 0); the exact product here exceeds the input
                   ("huge" . :truncated))))
    (each-vector :proofio-bad
                 (lambda (&key why bytes &allow-other-keys)
                   (let ((c (signalled 'malformed-proof
                                       (lambda () (sable.proofio:decode-proof (hex bytes))))))
                     (assert (eq (malformed-proof-reason c) (cdr (assoc why reasons :test #'string=)))))))))

;;; ------------------------------------------------------------------ Reed-Solomon

(defun test-reed-solomon ()
  (each-vector :rs-encode
               (lambda (&key n root msg cw points tw)
                 (assert (= (sable.rs:root-of-unity n) root))
                 (assert (equal (coerce (sable.rs:domain-points n) 'list) points))
                 (assert (equal (lists (sable.rs:twiddles n)) tw))
                 (assert (equal (coerce (sable.rs:encode msg n) 'list) cw))
                 (assert (equal (coerce (sable.rs:brute-eval msg n) 'list) cw))))
  (each-vector :rs-fold
               (lambda (&key n cw r res)
                 (assert (equal (lists (sable.rs:fold (exts cw) (aref (sable.rs:twiddles n) 0) (e r)))
                                res)))))

;;; ------------------------------------------------------------------ sumcheck

(defun sumcheck-case (&key d n tables eq-r claim proof rho reduced finals)
  "A sumcheck vector as the statement (* t0 t1 ...), its tables and finals as alists."
  (let ((variables (symbols "T" (length tables))))
    (list :d d :n n :claim (e claim) :proof (hex proof) :rho rho :reduced reduced
          :statement `(* ,@variables)
          :eq (and eq-r (exts eq-r))
          :tables (mapcar (lambda (v table) (cons v (exts table))) variables tables)
          :finals (mapcar (lambda (v final) (cons v (e final))) variables finals))))

(defun test-sumcheck-against-python ()
  (each-vector :sumcheck
               (lambda (&rest vector)
                 (destructuring-bind (&key d n claim proof rho reduced statement eq tables finals)
                     (apply #'sumcheck-case vector)
                   ;; the verifier on the Python prover's proof
                   (let ((tr (make-transcript "sc-vec" d :messages (sable.proofio:decode-proof proof))))
                     (multiple-value-bind (vrho vreduced) (sable.sumcheck:verify tr n statement claim :eq eq)
                       (assert (equal (lists vrho) rho))
                       (assert (same vreduced reduced))
                       (assert (sable.sumcheck:check-final vreduced statement finals))
                       (assert (transcript-finished-p tr))))
                   ;; the Lisp prover writes the same bytes as the Python prover
                   (let ((tr (make-transcript "sc-vec" d)))
                     (assert (ext= (sable.sumcheck:hypercube-sum statement n tables d :eq eq) claim))
                     (multiple-value-bind (prho pfinals) (sable.sumcheck:prove tr n statement tables :eq eq)
                       (assert (equal (lists prho) rho))
                       (assert (equalp pfinals finals))
                       (assert (equalp (sable.proofio:encode-proof (transcript-messages tr)) proof))))))))

(defun test-sumcheck-rejects-forgeries ()
  (destructuring-bind (&key d n claim proof statement finals &allow-other-keys)
      (apply #'sumcheck-case (first (vectors :sumcheck)))
    (flet ((run (messages &optional (claim claim))
             (let ((tr (make-transcript "sc-vec" d :messages messages)))
               (sable.sumcheck:check-final (nth-value 1 (sable.sumcheck:verify tr n statement claim))
                                           statement finals)))
           (rejected (type thunk)
             (assert (signalled type thunk))))
      (assert (run (sable.proofio:decode-proof proof)))
      ;; a wrong claim
      (rejected 'verification-failure
                (lambda () (run (sable.proofio:decode-proof proof) (ext+ claim (ext-one d)))))
      ;; each round message changed in one coordinate
      (dotimes (j n)
        (let ((messages (sable.proofio:decode-proof proof)))
          (setf (aref (aref messages j) 0 1) (fadd (aref (aref messages j) 0 1) 1))
          (rejected 'verification-failure (lambda () (run messages)))))
      ;; a message of the wrong shape or kind, and a missing one
      (let ((messages (sable.proofio:decode-proof proof)))
        (setf (aref messages 0) (make-array (list 1 d) :element-type '(unsigned-byte 32) :initial-element 0))
        (rejected 'message-shape-error (lambda () (run messages)))
        (setf (aref messages 0) (make-octets 4))
        (rejected 'message-shape-error (lambda () (run messages))))
      (rejected 'transcript-exhausted
                (lambda () (run (subseq (sable.proofio:decode-proof proof) 0 (1- n))))))))

(defun test-sumcheck-of-any-statement ()
  ;; a statement that is not a plain product, with a constant (a challenge) beside the tables
  (let* ((d 4) (n 3)
         (statement '(+ (* alpha t0 t1) (* 3 t1) -2))
         (state (sb-ext:seed-random-state 3))
         (random-ext (lambda () (coerce-ext (loop repeat d collect (random +p+ state)))))
         (constants `((alpha . ,(funcall random-ext))))
         (tables (loop for v in '(t0 t1)
                       collect (cons v (coerce (loop repeat (expt 2 n) collect (funcall random-ext)) 'vector))))
         (claim (sable.sumcheck:hypercube-sum statement n tables d :constants constants))
         (prover (make-transcript "any" d))
         (finals (nth-value 1 (sable.sumcheck:prove prover n statement tables :constants constants)))
         (verifier (make-transcript "any" d :messages (transcript-messages prover))))
    (assert (= (sable.sumcheck:statement-degree statement constants) 2))
    (assert (sable.sumcheck:check-final
             (nth-value 1 (sable.sumcheck:verify verifier n statement claim :constants constants))
             statement finals :constants constants))))

(defun test-sumcheck-base-tables ()
  ;; base tables are the same polynomial as their lift to F_p^D: the same proof, byte for byte
  (dolist (d '(4 8))
    (let* ((n 4)
           (statement '(+ (* t0 t1 t2) (* 7 t1)))
           (state (sb-ext:seed-random-state d))
           (base (loop for v in '(t0 t1 t2)
                       collect (cons v (coerce (loop repeat (expt 2 n) collect (random +p+ state)) 'vector))))
           (lifted (loop for (v . values) in base
                         collect (cons v (map 'vector (lambda (x) (base->ext x d)) values))))
           (eq (loop repeat n collect (coerce-ext (loop repeat d collect (random +p+ state))))))
      (flet ((proof (tables)
               (let ((tr (make-transcript "base" d)))
                 (multiple-value-bind (rho finals) (sable.sumcheck:prove tr n statement tables :eq eq)
                   (list (lists rho) finals (sable.proofio:encode-proof (transcript-messages tr)))))))
        (assert (equalp (proof base) (proof lifted)))
        (assert (ext= (sable.sumcheck:hypercube-sum statement n base d :eq eq)
                      (sable.sumcheck:hypercube-sum statement n lifted d :eq eq)))))))

(defun test-sumcheck-kernels-match-generic ()
  ;; the compiled round kernels (base and extension) give the same proofs as the generic loop
  (let ((d 4) (n 3))
    (dolist (case '(((* t0 t1 t2) nil nil)
                    ((+ (* t0 t1 t2) (* 7 t1) -1) t nil)
                    ((- (* alpha (expt t0 2) t1) t1) nil ((alpha 5 6 7 8)))
                    (t0 t nil)))
      (destructuring-bind (statement base constants) case
        (let* ((state (sb-ext:seed-random-state 11))
               (constants (loop for (v . c) in constants collect (cons v (coerce-ext c))))
               (tables (loop for v in (set-difference (form-variables statement) (mapcar #'car constants))
                             collect (cons v (coerce (loop repeat (expt 2 n)
                                                           collect (if base
                                                                       (random +p+ state)
                                                                       (coerce-ext (loop repeat d collect (random +p+ state)))))
                                                     'vector))))
               (eq (loop repeat n collect (coerce-ext (loop repeat d collect (random +p+ state))))))
          (flet ((proof ()
                   (let ((tr (make-transcript "kernel" d)))
                     (multiple-value-bind (rho finals)
                         (sable.sumcheck:prove tr n statement tables :eq eq :constants constants)
                       (list (lists rho) finals (sable.proofio:encode-proof (transcript-messages tr)))))))
            (assert (equalp (proof) (let ((sable.sumcheck:*kernel-size-limit* 0)) (proof))))))))
    ;; and the kernel is code that can be read
    (assert (eq 'lambda (first (sable.sumcheck:round-kernel-code '(* a b) '(a b) '() 4 2 nil))))))

(defun test-sumcheck-small-values ()
  ;; the small-value rounds give the same proofs as folding, byte for byte
  (dolist (d '(4 8))
    (dolist (case '(((* t0 t1 t2) 5) ((+ (* t0 t1 t2) (* 7 t1) -1) 4) (t0 3) ((* t0 t1) 2)
                    ((- (expt t0 3) (* t0 t1)) 4) ((* t0 t1 t2 t3) 1)))
      (destructuring-bind (statement n) case
        (dolist (with-eq '(t nil))
          (let* ((state (sb-ext:seed-random-state (+ d n)))
                 (tables (loop for v in (form-variables statement)
                               collect (cons v (coerce (loop repeat (expt 2 n) collect (random +p+ state))
                                                       'vector))))
                 (eq (and with-eq
                          (loop repeat n collect (coerce-ext (loop repeat d collect (random +p+ state)))))))
            (flet ((proof (l0)
                     (let ((sable.sumcheck:*small-value-rounds* l0)
                           (tr (make-transcript "small" d)))
                       (multiple-value-bind (rho finals) (sable.sumcheck:prove tr n statement tables :eq eq)
                         (list (lists rho) finals (sable.proofio:encode-proof (transcript-messages tr)))))))
              (let ((folded (proof 0)))
                (dolist (l0 '(2 3 4))
                  (assert (equalp (proof l0) folded) () "~S, n = ~D, D = ~D, L0 = ~D, eq ~A"
                          statement n d l0 with-eq))))))))))

;;; ------------------------------------------------------------------ LogUp-GKR

(defun random-leaves (lookups n k d state)
  "LOOKUPS pairs of K blocks of 2^N random fractions (p . q), q nonzero."
  (flet ((random-ext (&optional nonzero)
           (loop for e = (coerce-ext (loop repeat d collect (random +p+ state)))
                 unless (and nonzero (every #'zerop e)) return e)))
    (loop repeat lookups
          collect (cons (coerce (loop repeat (* k (expt 2 n)) collect (random-ext)) 'vector)
                        (coerce (loop repeat (* k (expt 2 n)) collect (random-ext t)) 'vector)))))

(defun gkr-proof (n k leaves d)
  "Prove the upper layers; return the messages and the prover's outputs."
  (let ((tr (make-transcript "gkr-test" d)))
    (multiple-value-bind (ri rb cp cq) (sable.gkr:prove-upper tr n k leaves)
      (values (transcript-messages tr) (list ri rb cp cq)))))

(defun test-gkr-roundtrip ()
  (dolist (case '((3 1 1 4) (3 3 2 4) (2 4 1 8) (4 5 2 4) (1 2 1 4) (1 1 2 8) (2 2 3 4)))
    (destructuring-bind (n k lookups d) case
      (let ((leaves (random-leaves lookups n k d (sb-ext:seed-random-state (+ n k lookups d)))))
        (multiple-value-bind (messages outputs) (gkr-proof n k leaves d)
          (let ((tr (make-transcript "gkr-test" d :messages messages)))
            (multiple-value-bind (roots ri rb cp cq) (sable.gkr:verify-upper tr n k lookups)
              (assert (transcript-finished-p tr))
              (assert (equalp (list ri rb cp cq) outputs) () "~S: verifier and prover disagree" case)
              ;; LogUp: each root is the sum of its leaf fractions
              (loop for (p . q) in leaves
                    for (root-p . root-q) in roots
                    do (assert (ext= (ext* root-p (ext-inv root-q))
                                     (reduce #'ext+ (map 'list (lambda (a b) (ext* a (ext-inv b))) p q)))))
              ;; the claims are on the leaves' parents, sum_{b<K} eq(rb, b) parent_b(ri)
              (multiple-value-bind (layers) (sable.gkr:fraction-trees leaves n k)
                (loop for (p . q) in (aref layers (1- n))
                      for claim-p in cp
                      for claim-q in cq
                      do (flet ((claim (values)
                                  (let ((size (expt 2 (1- n)))
                                        (weights (if rb (sable.field:eq-table rb d) (vector (ext-one d)))))
                                    (reduce #'ext+
                                            (loop for b below k
                                                  collect (ext* (aref weights b)
                                                                (mle-eval (subseq values (* b size) (* (1+ b) size))
                                                                          ri d)))))))
                           (assert (ext= claim-p (claim p)))
                           (assert (ext= claim-q (claim q)))))))))))))

(defun test-gkr-rejects-forgeries ()
  (let* ((n 3) (k 3) (lookups 2) (d 4)
         (leaves (random-leaves lookups n k d (sb-ext:seed-random-state 9))))
    (multiple-value-bind (messages) (gkr-proof n k leaves d)
      (flet ((verify (messages)
               (sable.gkr:verify-upper (make-transcript "gkr-test" d :messages messages) n k lookups)))
        (assert (verify messages))
        ;; change one value in each field message in turn: every change must be rejected
        (loop for i below (length messages)
              for m = (aref messages i)
              when (typep m 'field-array)
                do (let ((forged (copy-seq messages))
                         (changed (make-array (array-dimensions m) :element-type '(unsigned-byte 32))))
                     (dotimes (j (array-total-size m))
                       (setf (row-major-aref changed j) (row-major-aref m j)))
                     (setf (row-major-aref changed 0) (fadd (row-major-aref changed 0) 1)
                           (aref forged i) changed)
                     (assert (signalled 'verification-failure (lambda () (verify forged))) ()
                             "message ~D: forgery accepted" i)))))))

(defun export-gkr-cases (path)
  "Write GKR proofs made by PROVE-UPPER, with the verifier's outputs, as JSON for
tests/check-gkr-vcore.py, which checks them with zk/cuda/vcore.cpp's vc_gkr_upper."
  (flet ((ints (exts) (format nil "[~{~A~^,~}]" (loop for e in exts append (coerce e 'list)))))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (format out "[~%")
      (loop for (case . more) on '((3 1 1 4) (3 3 2 4) (2 4 1 8) (4 5 2 4) (1 2 1 4) (1 1 2 8) (2 2 3 4) (5 6 2 8))
            do (destructuring-bind (n k lookups d) case
                 (let ((leaves (random-leaves lookups n k d (sb-ext:seed-random-state (* 7 (+ n k lookups d))))))
                   (multiple-value-bind (messages) (gkr-proof n k leaves d)
                     (multiple-value-bind (roots ri rb cp cq)
                         (sable.gkr:verify-upper (make-transcript "gkr-test" d :messages messages) n k lookups)
                       (format out "{\"n\": ~D, \"K\": ~D, \"L\": ~D, \"D\": ~D, \"label\": \"gkr-test\", ~
                                    \"proof\": \"~(~{~2,'0X~}~)\", \"roots\": ~A, \"ri\": ~A, \"rb\": ~A, ~
                                    \"cp\": ~A, \"cq\": ~A}~:[~;,~]~%"
                               n k lookups d (coerce (sable.proofio:encode-proof messages) 'list)
                               (ints (loop for (p . q) in roots collect p collect q))
                               (ints ri) (ints rb) (ints cp) (ints cq) more))))))
      (format out "]~%"))
    path))

;;; ------------------------------------------------------------------ domains

(defun toy-domain (&key (n 3) (k 3) merge (lookups t) (constraints t) (state (sb-ext:seed-random-state 5)) break)
  "A domain with every kind of column: bits A, products C = A B, a :ROW 1 shared column M, a :COL n
shared mask S, a per-block constant KB (:ROW n), a :COL 2 column W.  With BREAK, C is wrong at one
point, so a constraint fails."
  (let* ((size (expt 2 n))
         (a (coerce (loop repeat (* k size) collect (random 2 state)) 'vector))
         (b (coerce (loop repeat (* k size) collect (random +p+ state)) 'vector))
         (c (map 'vector (lambda (x y) (fmul x y)) a b))
         (m (coerce (loop repeat (expt 2 (1- n)) collect (random +p+ state)) 'vector))
         (s (coerce (loop repeat size collect (random 2 state)) 'vector))
         (kb (coerce (loop repeat k collect (random +p+ state)) 'vector))
         (w (coerce (loop repeat (* k 4) collect (random +p+ state)) 'vector)))
    (when break (setf (aref c 1) (fadd (aref c 1) 1)))
    (sable.domain:make-domain
     :name (format nil "12.toy~:[~;-merged~]" merge) :n n :blocks k :merge merge
     :columns `((a :data ,a) (b :data ,b) (c :data ,c) (m :kind :row :c 1 :shared t :data ,m)
                (s :kind :col :c ,n :shared t :data ,s) (kb :kind :row :c ,n :data ,kb)
                (w :kind :col :c 2 :data ,w))
     :constraints (and constraints '((* a (- a 1)) (- c (* a b)) (* s (- s 1) m)))
     :lookups (and lookups '((:tag 5 :values (a b))
                             (:tag 7 :values (c m w) :num (+ s 2))
                             (:tag 9 :values (kb)))))))

(defun lookup-challenges (d state)
  (values (coerce-ext (loop repeat d collect (random +p+ state)))
          (loop repeat 8 collect (coerce-ext (loop repeat d collect (random +p+ state))))))

(defun domain-proof (domain d z betas)
  (let ((tr (make-transcript "dom-test" d)))
    (values (transcript-messages tr) (sable.domain:prove-domain tr domain z betas))))

(defun domain-mle (column domain point b d)
  "The MLE of a :FULL column's block B at POINT (its N coordinates)."
  (let ((size (expt 2 (sable.domain:domain-n domain))))
    (mle-eval (subseq (sable.domain:column-data column) (* b size) (* (1+ b) size)) point d)))

(defun test-domain-roundtrip ()
  (dolist (case '((:n 3 :k 3) (:n 2 :k 1) (:n 3 :k 3 :merge t) (:n 4 :k 4 :merge t) (:n 3 :k 2 :lookups nil)
                  (:n 3 :k 2 :constraints nil) (:n 2 :k 1 :merge t)))
    (dolist (d '(4 8))
      (let* ((state (sb-ext:seed-random-state (+ d (getf case :n) (getf case :k))))
             (domain (apply #'toy-domain :state state case)))
        (multiple-value-bind (z betas) (lookup-challenges d state)
          (multiple-value-bind (messages proved) (domain-proof domain d z betas)
            (let ((tr (make-transcript "dom-test" d :messages messages)))
              (let ((claims (sable.domain:verify-domain tr domain z betas)))
                (assert (transcript-finished-p tr))
                (assert (equalp claims proved) () "~S: verifier and prover claims differ" case)
                ;; a full column's claim is its MLE at the claimed point
                (loop for (name point value) in claims
                      for column = (find name (sable.domain:domain-columns domain) :key #'sable.domain:column-name)
                      when (eq (sable.domain:column-kind column) :full)
                        do (let ((n (sable.domain:domain-n domain)) (k (sable.domain:domain-blocks domain)))
                             (if (sable.domain:domain-merge domain)
                                 (let ((weights (if (> k 1) (sable.field:eq-table (nthcdr n point) d) (vector (ext-one d)))))
                                   (assert (ext= value (reduce #'ext+ (loop for b below k
                                                                            collect (ext* (aref weights b)
                                                                                          (domain-mle column domain (subseq point 0 n) b d)))))))
                                 (dotimes (b k)
                                   (assert (ext= (aref value b) (domain-mle column domain point b d)))))))))))))))

(defun test-domain-rejects ()
  (let ((d 4))
    (dolist (merge '(nil t))
      ;; a constraint that does not hold
      (let ((state (sb-ext:seed-random-state 3)))
        (let ((domain (toy-domain :merge merge :state state :break t)))
          (multiple-value-bind (z betas) (lookup-challenges d state)
            (assert (signalled 'verification-failure
                               (lambda () (sable.domain:verify-domain
                                           (make-transcript "dom-test" d :messages (domain-proof domain d z betas))
                                           domain z betas)))))))
      ;; any changed value
      (let* ((state (sb-ext:seed-random-state 4))
             (domain (toy-domain :merge merge :state state)))
        (multiple-value-bind (z betas) (lookup-challenges d state)
          (let ((messages (domain-proof domain d z betas)))
            (loop for i below (length messages)
                  for m = (aref messages i)
                  do (let ((forged (copy-seq messages))
                           (changed (make-array (array-dimensions m) :element-type '(unsigned-byte 32))))
                       (dotimes (j (array-total-size m)) (setf (row-major-aref changed j) (row-major-aref m j)))
                       (setf (row-major-aref changed 0) (fadd (row-major-aref changed 0) 1)
                             (aref forged i) changed)
                       (assert (signalled 'verification-failure
                                          (lambda () (sable.domain:verify-domain
                                                      (make-transcript "dom-test" d :messages forged) domain z betas)))
                               () "merge ~A, message ~D: forgery accepted" merge i)))))))))

(defun json-form (form)
  "A form as JSON: a symbol as its name, a list as an array (operator first)."
  (etypecase form
    (null "[]")
    (integer (format nil "~D" form))
    (symbol (format nil "\"~(~A~)\"" (symbol-name form)))
    (list (format nil "[~{~A~^,~}]" (mapcar #'json-form form)))))

(defun json-ints (seq) (format nil "[~{~D~^,~}]" (coerce seq 'list)))
(defun json-exts (exts) (format nil "[~{~A~^,~}]" (map 'list #'json-ints exts)))

(defun export-domain-cases (path)
  "Write domain proofs made by PROVE-DOMAIN, with the claims and roots VERIFY-DOMAIN computes, as
JSON for tests/check-domain-python.py, which checks them with zk/domain.py's verify."
  (with-open-file (out path :direction :output :if-exists :supersede)
    (format out "[~%")
    (loop for (case . more) on '((:n 3 :k 3) (:n 2 :k 1) (:n 3 :k 3 :merge t) (:n 4 :k 4 :merge t)
                                 (:n 3 :k 2 :lookups nil) (:n 3 :k 2 :constraints nil) (:n 4 :k 5 :merge t)
                                 (:n 3 :k 5))
          do (dolist (d '(4 8))
               (let* ((state (sb-ext:seed-random-state (+ d (* 3 (getf case :n)) (getf case :k))))
                      (domain (apply #'toy-domain :state state case)))
                 (multiple-value-bind (z betas) (lookup-challenges d state)
                   (let ((messages (domain-proof domain d z betas)))
                     (multiple-value-bind (claims roots)
                         (sable.domain:verify-domain (make-transcript "dom-test" d :messages messages) domain z betas)
                       (format out "{\"name\": \"~A\", \"n\": ~D, \"K\": ~D, \"merge\": ~:[false~;true~], \"D\": ~D, ~
                                    \"label\": \"dom-test\", \"columns\": [~{~A~^,~}], \"constraints\": ~A, ~
                                    \"lookups\": [~{~A~^,~}], \"Z\": ~A, \"beta\": ~A, \"proof\": \"~(~{~2,'0X~}~)\", ~
                                    \"roots\": [~{~A~^,~}], \"claims\": {~{~A~^,~}}}~:[~;,~]~%"
                               (sable.domain:domain-name domain) (getf case :n) (getf case :k) (getf case :merge) d
                               (loop for column in (sable.domain:domain-columns domain)
                                     collect (format nil "{\"name\": \"~(~A~)\", \"kind\": \"~(~A~)\", \"c\": ~D, \"shared\": ~:[false~;true~]}"
                                                     (sable.domain:column-name column) (sable.domain:column-kind column)
                                                     (sable.domain:column-c column) (sable.domain:column-shared column)))
                               (json-form (sable.domain:domain-constraints domain))
                               (loop for lookup in (sable.domain:domain-lookups domain)
                                     collect (format nil "{\"tag\": ~A, \"values\": ~A, \"num\": ~A}"
                                                     (json-form (getf lookup :tag)) (json-form (getf lookup :values))
                                                     (if (getf lookup :num) (json-form (getf lookup :num)) "null")))
                               (json-ints z) (json-exts betas) (coerce (sable.proofio:encode-proof messages) 'list)
                               (loop for (p . q) in roots collect (format nil "[~A,~A]" (json-ints p) (json-ints q)))
                               (loop for (name point value) in claims
                                     collect (format nil "\"~(~A~)\": {\"point\": ~A, \"value\": ~A}"
                                                     name (json-exts point)
                                                     (if (typep value 'ext) (json-exts (list value)) (json-exts value))))
                               (or more (/= d 8)))))))))
    (format out "]~%"))
  path)

;;; ------------------------------------------------------------------ the polynomial commitment

;;; Scenarios: units (name (tensor K c0) ...), the merge set, the segment (or NIL) and cmax.
(defparameter *pcs-scenarios*
  '((:cmax 9 :seed 21 :merge ("12.g1" "12.g2") :segment ("12.g1" "12.g2")
     :units (("w" ("wa" 2 9) ("wb" 1 6) ("wc" 1 3))      ; classes 5 to 9; v split into columns
             ("12.g1" ("x" 3 8) ("y" 1 9))
             ("12.g2" ("z" 2 7) ("u" 1 9))
             ("big" ("v" 2 10))))
    (:cmax 10 :seed 22 :merge ("3.a" "3.b") :segment nil   ; two oracle levels, depth-2 paths
     :units (("w2" ("p" 1 10) ("q" 6 5))
             ("3.a" ("r" 2 10))
             ("3.b" ("s" 1 8) ("t" 4 6))))
    (:cmax 9 :seed 23 :merge nil :segment ("7.a" "7.b") :max-points 2   ; points reduced in batches
     :units (("7.a" ("e" 1 9) ("f" 2 9) ("g" 1 9))
             ("7.b" ("h" 3 7))))))

(defun pcs-data (units state)
  "Random base-field tensors for UNITS: an alist tensor -> K 2^c0 values."
  (loop for (nil . items) in units
        append (loop for (tensor k c0) in items
                     collect (cons tensor (coerce (loop repeat (* k (expt 2 c0)) collect (random +p+ state)) 'vector)))))

(defun pcs-claims (units data state)
  "Per unit, true claims: per tensor, two points, the first shared by two claims (one per block
weighting), as (tensor point coefficients value)."
  (loop for (name . items) in units
        collect (cons name
                      (loop for (tensor k c0) in items
                            for values = (cdr (assoc tensor data :test #'equal))
                            append (loop for p below 2
                                         for point = (loop repeat c0 collect (coerce-ext (loop repeat 4 collect (random +p+ state))))
                                         append (loop repeat (if (zerop p) 2 1)
                                                      collect (let ((coefficients (loop for b below k
                                                                                        collect (cons b (coerce-ext (loop repeat 4 collect (random +p+ state)))))))
                                                                (list tensor point coefficients
                                                                      (reduce #'ext+ (loop for (b . w) in coefficients
                                                                                           collect (ext* w (mle-eval (subseq values (* b (expt 2 c0)) (* (1+ b) (expt 2 c0)))
                                                                                                                     point 4)))
                                                                              :initial-value (make-ext 4))))))))))

(defun pcs-run (scenario tr prover data claims &key (queries 6) (grind 4))
  "The commitment protocol of SCENARIO, as the prover (with DATA) or the verifier."
  (destructuring-bind (&key cmax units merge segment (max-points 64) &allow-other-keys) scenario
    (let ((pcs (sable.pcs:make-pcs tr :prover prover :cmax cmax :queries queries :grind grind))
          (sable.pcs::*max-reduction-points* max-points))
      (loop for (name . items) in units do (sable.pcs:pcs-declare pcs name items))
      (when merge (sable.pcs:pcs-merge pcs merge))
      (when segment (sable.pcs:pcs-segment-decl pcs segment))
      (loop for (name . items) in units
            do (sable.pcs:pcs-commit pcs name (and prover (loop for (tensor) in items
                                                                collect (assoc tensor data :test #'equal)))))
      (loop for (name . unit-claims) in claims
            do (sable.pcs:pcs-open pcs name unit-claims)
               (when (and segment (equal name (car (last segment)))) (sable.pcs:pcs-segment pcs 0)))
      (sable.pcs:pcs-finish pcs))))

(defvar *pcs-proofs* (make-hash-table) "Cached (data claims messages) per scenario.")

(defun pcs-proof (scenario)
  (or (gethash (getf scenario :seed) *pcs-proofs*)
      (let* ((state (sb-ext:seed-random-state (getf scenario :seed)))
             (data (pcs-data (getf scenario :units) state))
             (claims (pcs-claims (getf scenario :units) data state))
             (tr (make-transcript "pcs-test" 4)))
        (pcs-run scenario tr t data claims)
        (setf (gethash (getf scenario :seed) *pcs-proofs*) (list data claims (transcript-messages tr))))))

(defun test-pcs-roundtrip ()
  (dolist (scenario *pcs-scenarios*)
    (destructuring-bind (data claims messages) (pcs-proof scenario)
      (let ((tr (make-transcript "pcs-test" 4 :messages messages)))
        (assert (pcs-run scenario tr nil data claims))
        (assert (transcript-finished-p tr))))))

(defun test-pcs-rejects-forgeries ()
  (dolist (scenario *pcs-scenarios*)
    (destructuring-bind (data claims messages) (pcs-proof scenario)
      (flet ((rejected-p (forged &optional (claims claims))
               (signalled 'verification-failure
                          (lambda () (pcs-run scenario (make-transcript "pcs-test" 4 :messages forged) nil data claims)))))
        ;; a false claim
        (let ((false (copy-tree claims)))
          (setf (fourth (second (first false))) (ext+ (fourth (second (first false))) (ext-one 4)))
          (assert (rejected-p messages false)))
        ;; every message changed: a field value, or a byte of a cap or path (the paths of a tree
        ;; whose cap is its leaves are empty)
        (loop for i below (length messages)
              for m = (aref messages i)
              unless (zerop (array-total-size m))
                do (let ((forged (copy-seq messages)))
                     (setf (aref forged i)
                           (if (typep m 'byte-message)
                               (let ((b (copy-seq m))) (setf (aref b 0) (logxor (aref b 0) 1)) b)
                               (let ((a (make-array (array-dimensions m) :element-type '(unsigned-byte 32))))
                                 (dotimes (j (array-total-size m)) (setf (row-major-aref a j) (row-major-aref m j)))
                                 (setf (row-major-aref a 0) (fadd (row-major-aref a 0) 1))
                                 a)))
                     ;; a grinding nonce may still pass the (small) bound; every other change is rejected
                     (unless (and (typep m 'byte-message) (= (length m) 8))
                       (assert (rejected-p forged) () "message ~D (~A): forgery accepted" i (message-shape m)))))))))

(defun export-pcs-cases (path)
  "Write the commitment proofs of *PCS-SCENARIOS*, with their units and claims, as JSON for
tests/check-pcs-python.py, which checks them with zk/gpcs.py's verifier."
  (with-open-file (out path :direction :output :if-exists :supersede)
    (format out "[~%")
    (loop for (scenario . more) on *pcs-scenarios*
          do (destructuring-bind (data claims messages) (pcs-proof scenario)
               (declare (ignore data))
               (destructuring-bind (&key cmax units merge segment (max-points 64) &allow-other-keys) scenario
                 (format out "{\"label\": \"pcs-test\", \"cmax\": ~D, \"max_points\": ~D, \"queries\": 6, \"grind\": 4, ~
                              \"units\": [~{~A~^,~}], \"merge\": [~{\"~A\"~^,~}], \"segment\": [~{\"~A\"~^,~}], ~
                              \"claims\": [~{~A~^,~}], \"proof\": \"~(~{~2,'0X~}~)\"}~:[~;,~]~%"
                         cmax max-points
                         (loop for (name . items) in units
                               collect (format nil "{\"name\": \"~A\", \"items\": [~{[\"~A\", ~D, ~D]~^,~}]}"
                                               name (loop for item in items append item)))
                         merge segment
                         (loop for (name . unit-claims) in claims
                               collect (format nil "{\"unit\": \"~A\", \"claims\": [~{~A~^,~}]}" name
                                               (loop for (tensor point coefficients value) in unit-claims
                                                     collect (format nil "{\"tensor\": \"~A\", \"point\": ~A, \"coefficients\": {~{~A~^,~}}, \"value\": ~A}"
                                                                     tensor (json-exts point)
                                                                     (loop for (b . w) in coefficients collect (format nil "\"~D\": ~A" b (json-ints w)))
                                                                     (json-ints value)))))
                         (coerce (sable.proofio:encode-proof messages) 'list)
                         more))))
    (format out "]~%"))
  path)
