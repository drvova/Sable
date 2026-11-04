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
