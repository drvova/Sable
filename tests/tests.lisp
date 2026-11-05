;;;; tests.lisp -- Exacta against the original Python implementation (vectors.sexp, and the cases
;;;; its verifiers accepted, pinned in *FROZEN-CASES*), forms as data and as code, and the
;;;; conditions and restarts.

(in-package #:exacta.tests)

(defun hex (s)
  "A hex string -> OCTETS."
  (let ((out (make-octets (floor (length s) 2))))
    (dotimes (i (length out) out)
      (setf (aref out i) (parse-integer s :start (* 2 i) :end (+ 2 (* 2 i)) :radix 16)))))

(defun read-vectors (path)
  (with-open-file (in path)
    (with-standard-io-syntax
      (let ((*read-eval* nil)
            (*package* (find-package '#:exacta.tests)))
        (read in)))))

(defvar *vectors*
  (read-vectors (asdf:system-relative-pathname "exacta" "tests/vectors.sexp"))
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
  (loop for i below n collect (intern (format nil "~A~D" prefix i) '#:exacta.tests)))

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
    (assert (equal (second code) '(exacta.field::a exacta.field::b exacta.field::out))))
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
  ;; no handler: the error, a DIVISION-BY-ZERO as well as a EXACTA-ERROR
  (let ((c (signalled 'field-division-by-zero (lambda () (finv 0)))))
    (assert (typep c 'division-by-zero))
    (assert (typep c 'exacta-error)))
  ;; the restarts: RETURN-ZERO (the Python convention) and USE-VALUE
  (assert (= 0 (handler-bind ((field-division-by-zero #'return-zero)) (finv 0))))
  (assert (= 7 (handler-bind ((field-division-by-zero (lambda (c) (use-value 7 c)))) (finv +p+))))
  (assert (every #'zerop (handler-bind ((field-division-by-zero #'return-zero))
                           (ext-inv (make-ext 8))))))

(defun test-invalid-parameter-restart ()
  (assert (signalled 'invalid-parameter (lambda () (exacta.rs:root-of-unity 30))))
  (assert (= (exacta.rs:root-of-unity 3)
             (handler-bind ((invalid-parameter (lambda (c) (use-value 3 c))))
               (exacta.rs:root-of-unity 30))))
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

(defun as-written (octets)
  "The proof OCTETS of the original prover (magic SABLEPF1) as Exacta writes them: the same bytes
after its own magic."
  (let ((m exacta.proofio:*magic*))
    (if (member (subseq octets 0 (min (length m) (length octets))) exacta.proofio:*legacy-magics*
                :test #'equalp)
        (concatenate '(vector (unsigned-byte 8)) m (subseq octets (length m)))
        octets)))

(defun test-proofio-roundtrip ()
  (each-vector :proofio
               (lambda (&key bytes msgs)
                 (let ((parsed (exacta.proofio:decode-proof (hex bytes))))
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
                   (assert (equalp (exacta.proofio:encode-proof parsed) (as-written (hex bytes))))))))

(defun test-proofio-rejects-malformed ()
  (let ((reasons '(("magic" . :magic) ("truncated" . :truncated) ("trailing" . :trailing-bytes)
                   ("tag" . :unknown-tag) ("range" . :value-range) ("rank" . :rank)
                   ;; zk/proofio.py rejects this one only through numpy's reshape (its np.prod of
                   ;; the shape wraps to 0); the exact product here exceeds the input
                   ("huge" . :truncated))))
    (each-vector :proofio-bad
                 (lambda (&key why bytes &allow-other-keys)
                   (let ((c (signalled 'malformed-proof
                                       (lambda () (exacta.proofio:decode-proof (hex bytes))))))
                     (assert (eq (malformed-proof-reason c) (cdr (assoc why reasons :test #'string=)))))))))

;;; ------------------------------------------------------------------ Reed-Solomon

(defun test-reed-solomon ()
  (each-vector :rs-encode
               (lambda (&key n root msg cw points tw)
                 (assert (= (exacta.rs:root-of-unity n) root))
                 (assert (equal (coerce (exacta.rs:domain-points n) 'list) points))
                 (assert (equal (lists (exacta.rs:twiddles n)) tw))
                 (assert (equal (coerce (exacta.rs:encode msg n) 'list) cw))
                 (assert (equal (coerce (exacta.rs:brute-eval msg n) 'list) cw))))
  (each-vector :rs-fold
               (lambda (&key n cw r res)
                 (assert (equal (lists (exacta.rs:fold (exts cw) (aref (exacta.rs:twiddles n) 0) (e r)))
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
                   (let ((tr (make-transcript "sc-vec" d :messages (exacta.proofio:decode-proof proof))))
                     (multiple-value-bind (vrho vreduced) (exacta.sumcheck:verify tr n statement claim :eq eq)
                       (assert (equal (lists vrho) rho))
                       (assert (same vreduced reduced))
                       (assert (exacta.sumcheck:check-final vreduced statement finals))
                       (assert (transcript-finished-p tr))))
                   ;; the Lisp prover writes the same bytes as the Python prover, after the magic
                   (let ((tr (make-transcript "sc-vec" d)))
                     (assert (ext= (exacta.sumcheck:hypercube-sum statement n tables d :eq eq) claim))
                     (multiple-value-bind (prho pfinals) (exacta.sumcheck:prove tr n statement tables :eq eq)
                       (assert (equal (lists prho) rho))
                       (assert (equalp pfinals finals))
                       (assert (equalp (exacta.proofio:encode-proof (transcript-messages tr)) (as-written proof)))))))))

(defun test-sumcheck-rejects-forgeries ()
  (destructuring-bind (&key d n claim proof statement finals &allow-other-keys)
      (apply #'sumcheck-case (first (vectors :sumcheck)))
    (flet ((run (messages &optional (claim claim))
             (let ((tr (make-transcript "sc-vec" d :messages messages)))
               (exacta.sumcheck:check-final (nth-value 1 (exacta.sumcheck:verify tr n statement claim))
                                           statement finals)))
           (rejected (type thunk)
             (assert (signalled type thunk))))
      (assert (run (exacta.proofio:decode-proof proof)))
      ;; a wrong claim
      (rejected 'verification-failure
                (lambda () (run (exacta.proofio:decode-proof proof) (ext+ claim (ext-one d)))))
      ;; each round message changed in one coordinate
      (dotimes (j n)
        (let ((messages (exacta.proofio:decode-proof proof)))
          (setf (aref (aref messages j) 0 1) (fadd (aref (aref messages j) 0 1) 1))
          (rejected 'verification-failure (lambda () (run messages)))))
      ;; a message of the wrong shape or kind, and a missing one
      (let ((messages (exacta.proofio:decode-proof proof)))
        (setf (aref messages 0) (make-array (list 1 d) :element-type '(unsigned-byte 32) :initial-element 0))
        (rejected 'message-shape-error (lambda () (run messages)))
        (setf (aref messages 0) (make-octets 4))
        (rejected 'message-shape-error (lambda () (run messages))))
      (rejected 'transcript-exhausted
                (lambda () (run (subseq (exacta.proofio:decode-proof proof) 0 (1- n))))))))

(defun test-sumcheck-of-any-statement ()
  ;; a statement that is not a plain product, with a constant (a challenge) beside the tables
  (let* ((d 4) (n 3)
         (statement '(+ (* alpha t0 t1) (* 3 t1) -2))
         (state (sb-ext:seed-random-state 3))
         (random-ext (lambda () (coerce-ext (loop repeat d collect (random +p+ state)))))
         (constants `((alpha . ,(funcall random-ext))))
         (tables (loop for v in '(t0 t1)
                       collect (cons v (coerce (loop repeat (expt 2 n) collect (funcall random-ext)) 'vector))))
         (claim (exacta.sumcheck:hypercube-sum statement n tables d :constants constants))
         (prover (make-transcript "any" d))
         (finals (nth-value 1 (exacta.sumcheck:prove prover n statement tables :constants constants)))
         (verifier (make-transcript "any" d :messages (transcript-messages prover))))
    (assert (= (exacta.sumcheck:statement-degree statement constants) 2))
    (assert (exacta.sumcheck:check-final
             (nth-value 1 (exacta.sumcheck:verify verifier n statement claim :constants constants))
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
                 (multiple-value-bind (rho finals) (exacta.sumcheck:prove tr n statement tables :eq eq)
                   (list (lists rho) finals (exacta.proofio:encode-proof (transcript-messages tr)))))))
        (assert (equalp (proof base) (proof lifted)))
        (assert (ext= (exacta.sumcheck:hypercube-sum statement n base d :eq eq)
                      (exacta.sumcheck:hypercube-sum statement n lifted d :eq eq)))))))

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
                         (exacta.sumcheck:prove tr n statement tables :eq eq :constants constants)
                       (list (lists rho) finals (exacta.proofio:encode-proof (transcript-messages tr)))))))
            (assert (equalp (proof) (let ((exacta.sumcheck:*kernel-size-limit* 0)) (proof))))))))
    ;; and the kernel is code that can be read
    (assert (eq 'lambda (first (exacta.sumcheck:round-kernel-code '(* a b) '(a b) '() 4 2 nil))))))

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
                     (let ((exacta.sumcheck:*small-value-rounds* l0)
                           (tr (make-transcript "small" d)))
                       (multiple-value-bind (rho finals) (exacta.sumcheck:prove tr n statement tables :eq eq)
                         (list (lists rho) finals (exacta.proofio:encode-proof (transcript-messages tr)))))))
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
    (multiple-value-bind (ri rb cp cq) (exacta.gkr:prove-upper tr n k leaves)
      (values (transcript-messages tr) (list ri rb cp cq)))))

(defun test-gkr-roundtrip ()
  (dolist (case '((3 1 1 4) (3 3 2 4) (2 4 1 8) (4 5 2 4) (1 2 1 4) (1 1 2 8) (2 2 3 4)))
    (destructuring-bind (n k lookups d) case
      (let ((leaves (random-leaves lookups n k d (sb-ext:seed-random-state (+ n k lookups d)))))
        (multiple-value-bind (messages outputs) (gkr-proof n k leaves d)
          (let ((tr (make-transcript "gkr-test" d :messages messages)))
            (multiple-value-bind (roots ri rb cp cq) (exacta.gkr:verify-upper tr n k lookups)
              (assert (transcript-finished-p tr))
              (assert (equalp (list ri rb cp cq) outputs) () "~S: verifier and prover disagree" case)
              ;; LogUp: each root is the sum of its leaf fractions
              (loop for (p . q) in leaves
                    for (root-p . root-q) in roots
                    do (assert (ext= (ext* root-p (ext-inv root-q))
                                     (reduce #'ext+ (map 'list (lambda (a b) (ext* a (ext-inv b))) p q)))))
              ;; the claims are on the leaves' parents, sum_{b<K} eq(rb, b) parent_b(ri)
              (multiple-value-bind (layers) (exacta.gkr:fraction-trees leaves n k)
                (loop for (p . q) in (aref layers (1- n))
                      for claim-p in cp
                      for claim-q in cq
                      do (flet ((claim (values)
                                  (let ((size (expt 2 (1- n)))
                                        (weights (if rb (exacta.field:eq-table rb d) (vector (ext-one d)))))
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
               (exacta.gkr:verify-upper (make-transcript "gkr-test" d :messages messages) n k lookups)))
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
  "Write GKR proofs made by PROVE-UPPER, with the verifier's outputs, as JSON.  The original C++
verifier (vc_gkr_upper) accepted exactly these; TEST-FROZEN-CASES pins them."
  (flet ((ints (exts) (format nil "[~{~A~^,~}]" (loop for e in exts append (coerce e 'list)))))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (format out "[~%")
      (loop for (case . more) on '((3 1 1 4) (3 3 2 4) (2 4 1 8) (4 5 2 4) (1 2 1 4) (1 1 2 8) (2 2 3 4) (5 6 2 8))
            do (destructuring-bind (n k lookups d) case
                 (let ((leaves (random-leaves lookups n k d (sb-ext:seed-random-state (* 7 (+ n k lookups d))))))
                   (multiple-value-bind (messages) (gkr-proof n k leaves d)
                     (multiple-value-bind (roots ri rb cp cq)
                         (exacta.gkr:verify-upper (make-transcript "gkr-test" d :messages messages) n k lookups)
                       (format out "{\"n\": ~D, \"K\": ~D, \"L\": ~D, \"D\": ~D, \"label\": \"gkr-test\", ~
                                    \"proof\": \"~(~{~2,'0X~}~)\", \"roots\": ~A, \"ri\": ~A, \"rb\": ~A, ~
                                    \"cp\": ~A, \"cq\": ~A}~:[~;,~]~%"
                               n k lookups d (coerce (exacta.proofio:encode-proof messages) 'list)
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
    (exacta.domain:make-domain
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
    (values (transcript-messages tr) (exacta.domain:prove-domain tr domain z betas))))

(defun domain-mle (column domain point b d)
  "The MLE of a :FULL column's block B at POINT (its N coordinates)."
  (let ((size (expt 2 (exacta.domain:domain-n domain))))
    (mle-eval (subseq (exacta.domain:column-data column) (* b size) (* (1+ b) size)) point d)))

(defun test-domain-roundtrip ()
  (dolist (case '((:n 3 :k 3) (:n 2 :k 1) (:n 3 :k 3 :merge t) (:n 4 :k 4 :merge t) (:n 3 :k 2 :lookups nil)
                  (:n 3 :k 2 :constraints nil) (:n 2 :k 1 :merge t)))
    (dolist (d '(4 8))
      (let* ((state (sb-ext:seed-random-state (+ d (getf case :n) (getf case :k))))
             (domain (apply #'toy-domain :state state case)))
        (multiple-value-bind (z betas) (lookup-challenges d state)
          (multiple-value-bind (messages proved) (domain-proof domain d z betas)
            (let ((tr (make-transcript "dom-test" d :messages messages)))
              (let ((claims (exacta.domain:verify-domain tr domain z betas)))
                (assert (transcript-finished-p tr))
                (assert (equalp claims proved) () "~S: verifier and prover claims differ" case)
                ;; a full column's claim is its MLE at the claimed point
                (loop for (name point value) in claims
                      for column = (find name (exacta.domain:domain-columns domain) :key #'exacta.domain:column-name)
                      when (eq (exacta.domain:column-kind column) :full)
                        do (let ((n (exacta.domain:domain-n domain)) (k (exacta.domain:domain-blocks domain)))
                             (if (exacta.domain:domain-merge domain)
                                 (let ((weights (if (> k 1) (exacta.field:eq-table (nthcdr n point) d) (vector (ext-one d)))))
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
                               (lambda () (exacta.domain:verify-domain
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
                                          (lambda () (exacta.domain:verify-domain
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
JSON.  The original Python verifier accepted exactly these; TEST-FROZEN-CASES pins them."
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
                         (exacta.domain:verify-domain (make-transcript "dom-test" d :messages messages) domain z betas)
                       (format out "{\"name\": \"~A\", \"n\": ~D, \"K\": ~D, \"merge\": ~:[false~;true~], \"D\": ~D, ~
                                    \"label\": \"dom-test\", \"columns\": [~{~A~^,~}], \"constraints\": ~A, ~
                                    \"lookups\": [~{~A~^,~}], \"Z\": ~A, \"beta\": ~A, \"proof\": \"~(~{~2,'0X~}~)\", ~
                                    \"roots\": [~{~A~^,~}], \"claims\": {~{~A~^,~}}}~:[~;,~]~%"
                               (exacta.domain:domain-name domain) (getf case :n) (getf case :k) (getf case :merge) d
                               (loop for column in (exacta.domain:domain-columns domain)
                                     collect (format nil "{\"name\": \"~(~A~)\", \"kind\": \"~(~A~)\", \"c\": ~D, \"shared\": ~:[false~;true~]}"
                                                     (exacta.domain:column-name column) (exacta.domain:column-kind column)
                                                     (exacta.domain:column-c column) (exacta.domain:column-shared column)))
                               (json-form (exacta.domain:domain-constraints domain))
                               (loop for lookup in (exacta.domain:domain-lookups domain)
                                     collect (format nil "{\"tag\": ~A, \"values\": ~A, \"num\": ~A}"
                                                     (json-form (getf lookup :tag)) (json-form (getf lookup :values))
                                                     (if (getf lookup :num) (json-form (getf lookup :num)) "null")))
                               (json-ints z) (json-exts betas) (coerce (exacta.proofio:encode-proof messages) 'list)
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
    (let ((pcs (exacta.pcs:make-pcs tr :prover prover :cmax cmax :queries queries :grind grind))
          (exacta.pcs::*max-reduction-points* max-points))
      (loop for (name . items) in units do (exacta.pcs:pcs-declare pcs name items))
      (when merge (exacta.pcs:pcs-merge pcs merge))
      (when segment (exacta.pcs:pcs-segment-decl pcs segment))
      (loop for (name . items) in units
            do (exacta.pcs:pcs-commit pcs name (and prover (loop for (tensor) in items
                                                                collect (assoc tensor data :test #'equal)))))
      (loop for (name . unit-claims) in claims
            do (exacta.pcs:pcs-open pcs name unit-claims)
               (when (and segment (equal name (car (last segment)))) (exacta.pcs:pcs-segment pcs 0)))
      (exacta.pcs:pcs-finish pcs))))

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
  "Write the commitment proofs of *PCS-SCENARIOS*, with their units and claims, as JSON.  The
original Python verifier accepted exactly these; TEST-FROZEN-CASES pins them."
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
                         (coerce (exacta.proofio:encode-proof messages) 'list)
                         more))))
    (format out "]~%"))
  path)

;;; ------------------------------------------------------------------ the outer layer of llm.verify

(defparameter *llm-configs*
  `(("tiny" . ,(exacta.llm:make-cfg :l 2 :d 256 :nh 4 :nkv 2 :dh 128 :dff 512 :v 1024))
    ("tiny3" . ,(exacta.llm:make-cfg :l 2 :d 256 :nh 4 :nkv 2 :dh 128 :dff 512 :v 1024 :qk-norm t :qkv-bias nil))
    ("tiny-moe" . ,(exacta.llm:make-cfg :l 2 :d 512 :nh 4 :nkv 2 :dh 128 :dff 128 :v 1024 :qk-norm t :qkv-bias nil
                                       :moe '(:e 8 :k 2 :de 64)))
    ("Qwen2.5-32B" . ,(exacta.llm:make-cfg :l 64 :d 5120 :nh 40 :nkv 8 :dh 128 :dff 27648 :v 152064))
    ("Qwen3-32B" . ,(exacta.llm:make-cfg :l 64 :d 5120 :nh 64 :nkv 8 :dh 128 :dff 25600 :v 151936
                                        :qk-norm t :qkv-bias nil))
    ("Qwen3-30B-A3B" . ,(exacta.llm:make-cfg :l 48 :d 2048 :nh 32 :nkv 4 :dh 128 :dff (* 8 768) :v 151936
                                            :qk-norm t :qkv-bias nil :moe '(:e 128 :k 8 :de 768))))
  "Configurations: tiny models of the three families, and the paper's three models.")

(defun llm-shape (cfg b s ng)
  (multiple-value-bind (cb cbv) (exacta.llm:column-block-bits cfg)
    (exacta.llm:make-shape cfg b s cb cbv ng)))

(defun json-cfg (cfg)
  (format nil "{\"L\": ~D, \"d\": ~D, \"nh\": ~D, \"nkv\": ~D, \"dh\": ~D, \"dff\": ~D, \"V\": ~D, \"qk_norm\": ~:[false~;true~], ~
               \"qkv_bias\": ~:[false~;true~], \"moe\": ~A, \"down_b\": ~D}"
          (exacta.llm::cfg-l cfg) (exacta.llm::cfg-d cfg) (exacta.llm::cfg-nh cfg) (exacta.llm::cfg-nkv cfg)
          (exacta.llm::cfg-dh cfg) (exacta.llm::cfg-dff cfg) (exacta.llm::cfg-v cfg) (exacta.llm::cfg-qk-norm cfg)
          (exacta.llm::cfg-qkv-bias cfg)
          (let ((moe (exacta.llm::cfg-moe cfg)))
            (if moe (format nil "{\"E\": ~D, \"k\": ~D, \"de\": ~D}" (getf moe :e) (getf moe :k) (getf moe :de)) "null"))
          (exacta.llm::cfg-down-b cfg)))

(defun json-hex (octets) (format nil "\"~(~{~2,'0X~}~)\"" (coerce octets 'list)))
(defun json-strings (strings) (format nil "[~{\"~A\"~^,~}]" strings))
(defun json-rows (rows) (format nil "[~{~A~^,~}]" (mapcar #'json-ints rows)))

(defun random-octets (n state)
  (coerce (loop repeat n collect (random 256 state)) 'octets))

(defun json-policy (p)
  (format nil "{\"gmax\": ~A, \"stop\": ~A, \"theta\": ~D, \"ctx_bound\": ~A, \"tokenizer\": ~A, \"template\": ~A, ~
               \"hpre\": ~A, \"autoregressive\": ~:[false~;true~]}"
          (or (exacta.llm::policy-gmax p) "null") (json-ints (exacta.llm::policy-stop p)) (exacta.llm::policy-theta p)
          (or (exacta.llm::policy-ctx-bound p) "null") (json-hex (exacta.llm::policy-tokenizer p))
          (json-hex (exacta.llm::policy-template p)) (json-hex (exacta.llm::policy-hpre p))
          (exacta.llm::policy-autoregressive p)))

(defun fed-back-tokens (b s ng v state &key stop-at-end)
  "B prompts of S tokens whose last ng - 1 are the generated ones, and the targets (ng per row)."
  (loop repeat b
        collect (let* ((tokens (loop repeat s collect (random v state)))
                       (targets (append (nthcdr (1+ (- s ng)) tokens) (list (or stop-at-end (random v state))))))
                  (list tokens targets))
          into rows
        finally (return (values (mapcar #'first rows) (mapcar #'second rows)))))

(defun test-llm-layout ()
  ;; every layout declares on the commitment, which finds its classes
  (loop for (name . cfg) in *llm-configs*
        do (let* ((shape (llm-shape cfg 1 (if (> (exacta.llm::cfg-l cfg) 2) 4096 256) 4))
                  (layout (exacta.llm:commitment-layout cfg shape))
                  (pcs (exacta.llm:declare-units (exacta.pcs:make-pcs nil) layout)))
             (assert (= (exacta.pcs:pcs-nv pcs) 23) () "~A" name)
             (assert (= (length (getf layout :merges)) (exacta.llm::cfg-l cfg))))))

(defun test-llm-statement-checks ()
  (let* ((state (sb-ext:seed-random-state 31))
         (cfg (cdr (assoc "tiny" *llm-configs* :test #'string=)))
         (shape (llm-shape cfg 2 256 4)))
    (multiple-value-bind (tokens targets) (fed-back-tokens 2 256 4 1024 state)
      (assert (exacta.llm:check-statement (exacta.llm:make-policy) shape tokens targets))
      (flet ((rejects (policy tokens targets)
               (assert (signalled 'verification-failure
                                  (lambda () (exacta.llm:check-statement policy shape tokens targets))))))
        (rejects (exacta.llm:make-policy :ctx-bound 128) tokens targets)
        (rejects (exacta.llm:make-policy) (cons (cons 5000 (rest (first tokens))) (rest tokens)) targets)
        (rejects (exacta.llm:make-policy :gmax 8 :stop '(3)) tokens targets)        ; truncated
        (rejects (exacta.llm:make-policy) tokens (list (reverse (first targets)) (second targets)))))))

(defun export-llm-cases (path)
  "Write layouts, statements, policies, sampling seeds, statement checks and C_W computed by
llm.lisp as JSON.  The original Python code computed the same; TEST-FROZEN-CASES pins them."
  (let ((state (sb-ext:seed-random-state 41)))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (format out "{\"layouts\": [~%")
      (loop for ((name . cfg) . more) on *llm-configs*
            do (loop for ((b s ng theta) . more-shapes) on (if (> (exacta.llm::cfg-l cfg) 2)
                                                                 '((1 4096 16 0) (4 4096 16 3) (1 16384 16 0))
                                                                 '((1 256 4 0) (2 512 8 3) (4 256 16 0)))
                     do (let ((layout (exacta.llm:commitment-layout cfg (llm-shape cfg b s ng) :theta theta)))
                          (format out "{\"name\": \"~A\", \"cfg\": ~A, \"B\": ~D, \"S\": ~D, \"ng\": ~D, \"theta\": ~D, ~
                                       \"groups\": [~{~A~^,~}], \"merges\": [~{~A~^,~}], \"segments\": [~{~A~^,~}], ~
                                       \"weight_groups\": ~A}~:[~;,~]~%"
                                  name (json-cfg cfg) b s ng theta
                                  (loop for (g items) in (getf layout :groups)
                                        collect (format nil "[\"~A\", [~{[\"~A\", ~D, ~D]~^,~}]]" g (apply #'append items)))
                                  (mapcar #'json-strings (getf layout :merges))
                                  (mapcar #'json-strings (getf layout :segments))
                                  (json-strings (getf layout :weight-groups))
                                  (or more more-shapes)))))
      (format out "],~%\"statements\": [~%")
      (loop for ((name . cfg) . more) on (subseq *llm-configs* 0 3)
            do (loop for (policy . more-policies)
                       on (list (exacta.llm:make-policy)
                                (exacta.llm:make-policy :theta 3 :gmax 8 :stop '(7 3 7) :ctx-bound 1024
                                                       :tokenizer "qwen-tokenizer" :template "chat"
                                                       :hpre (random-octets 32 state))
                                (exacta.llm:make-policy :autoregressive nil :theta 10))
                     do (multiple-value-bind (tokens targets) (fed-back-tokens 2 256 8 1024 state)
                          (let ((shape (llm-shape cfg 2 256 8))
                                (c-w (random-octets 32 state)) (constants (random-octets 32 state))
                                (nonce (random-octets 32 state)) (salt (random-octets 32 state)))
                            (format out "{\"name\": \"~A\", \"cfg\": ~A, \"B\": 2, \"S\": 256, \"ng\": 8, \"tokens\": ~A, \"targets\": ~A, ~
                                         \"C_W\": ~A, \"const\": ~A, \"nonce\": ~A, \"salt\": ~A, \"policy\": ~A, ~
                                         \"policy_bytes\": ~A, \"statement\": ~A, \"seed\": ~A}~:[~;,~]~%"
                                    name (json-cfg cfg) (json-rows tokens) (json-rows targets)
                                    (json-hex c-w) (json-hex constants) (json-hex nonce) (json-hex salt) (json-policy policy)
                                    (json-hex (exacta.llm:encode-policy policy))
                                    (json-hex (exacta.llm:statement cfg shape tokens targets c-w constants
                                                                   :policy policy :nonce nonce :salt salt))
                                    (json-hex (exacta.llm:sampling-seed tokens 8 c-w policy :nonce nonce))
                                    (or more more-policies))))))
      (format out "],~%\"checks\": [~%")
      (let* ((cfg (cdr (first *llm-configs*)))
             (shape (llm-shape cfg 2 256 4))
             (cases (multiple-value-bind (tokens targets) (fed-back-tokens 2 256 4 1024 state :stop-at-end 3)
                      (list (list (exacta.llm:make-policy) tokens targets)
                            (list (exacta.llm:make-policy :stop '(3)) tokens targets)
                            (list (exacta.llm:make-policy :stop '(3) :gmax 8) tokens targets)
                            (list (exacta.llm:make-policy :gmax 8) tokens targets)
                            (list (exacta.llm:make-policy :gmax 2) tokens targets)
                            (list (exacta.llm:make-policy :ctx-bound 100) tokens targets)
                            (list (exacta.llm:make-policy :stop (list (first (first targets)))) tokens targets)
                            (list (exacta.llm:make-policy) (cons (cons 1024 (rest (first tokens))) (rest tokens)) targets)
                            (list (exacta.llm:make-policy) tokens (list (reverse (first targets)) (second targets)))
                            (list (exacta.llm:make-policy :autoregressive nil) tokens (list (reverse (first targets)) (second targets)))))))
        (loop for ((policy tokens targets) . more) on cases
              do (format out "{\"policy\": ~A, \"tokens\": ~A, \"targets\": ~A, \"ok\": ~:[false~;true~]}~:[~;,~]~%"
                         (json-policy policy) (json-rows tokens) (json-rows targets)
                         (handler-case (exacta.llm:check-statement policy shape tokens targets)
                           (verification-failure () nil))
                         more)))
      (let ((caps (loop repeat 5 collect (random-octets (* 32 (1+ (random 4 state))) state))))
        (format out "],~%\"wdigest\": {\"caps\": [~{~A~^,~}], \"digest\": ~A}}~%"
                (mapcar #'json-hex caps) (json-hex (exacta.llm:weights-digest caps)))))
    path))

;;; ------------------------------------------------------------------ cases the original code accepted

(defparameter *frozen-cases*
  '((export-gkr-cases "100cba9c4c818c1d2bb79467fe807ed3f7f7054274b5331cb7c22f78bd538320")
    (export-domain-cases "7a53788916a303f668251d4e05b22085efcb4d57813bc58a0f93e7f214cbcf12")
    (export-pcs-cases "6ade2c3e6acad8ce06eea085b01b9534d625ff4476f7d970a2b3376949378d37")
    (export-llm-cases "ae877184d103fe2a5452288fe4e76236179eb960b78756e209546bbf6a0b86f7")
    (export-proto-cases "42336065bb45e4fa12f138c91bd4420f9dc48cd27c0f9cf99d415d9ad92f4dce")
    (export-table-cases "08854f911baa2c7006ff9a9d0372ce9ede9e0c50c2ed16fca5b4ad0776cdbd8c")
    (export-step-mo-cases "00348957a008539af802f9075518564012739cdf1a4e5376ae4880a549cd6710")
    (export-step-mlp-cases "f48dec4c0b3829d8d1d16a43ccccba42afd4d1ad91920fe7ee8ae6014061e338"))
  "The SHA-256 of each case file.  Before the Python code was removed, its verifiers checked these
exact cases: the C++ vc_gkr_upper accepted the 8 GKR proofs and rejected 231 forgeries,
zk/domain.py accepted the 16 domain proofs, zk/gpcs.py accepted the 3 commitment proofs and
rejected 287 forgeries, and zk/llm.py computed the same 18 layouts, 9 statements and seeds,
10 statement checks and C_W; and zk/proto.py's Ctx replayed the context scenario, computed the
same values and rejected its 95 forgeries; Sable's logup.table_side, at 2^9 rows, replayed the
table-side scenario with the same claims and chunk sums and rejected its 172 forgeries; Sable's
Proto.step_mo replayed the tiny model's layer-0 MLP output step, left the same claims on 44 tensors,
virtual claims and down-projection partials, and rejected all 1542 forgeries; with the MLP slice
after it (step_mlp), the same claims on 48 tensors and one-hot claims, all 2211 forgeries rejected.  The files differ from those only in the proof magic
(EXACTPF1 for SABLEPF1).")

(defun test-frozen-cases ()
  (dolist (case *frozen-cases*)
    (destructuring-bind (export digest) case
      (uiop:with-temporary-file (:pathname path :type "json")
        (funcall export path)
        (let ((octets (with-open-file (in path :element-type '(unsigned-byte 8))
                        (let ((v (make-array (file-length in) :element-type '(unsigned-byte 8))))
                          (read-sequence v in)
                          v))))
          (assert (string= (format nil "~(~{~2,'0X~}~)" (coerce (exacta.sha256:sha256 octets) 'list))
                           digest)
                  () "~A: the cases differ from those the original code checked" export))))))

;;; ------------------------------------------------------------------ the protocol context

(defparameter *proto-groups*
  '(("g0" ("a" 6 3) ("u" 3 4 :no-pack))
    ("g1" ("w" 4 4 (2 2)) ("b" 1 5)))
  "Commitment groups: a is packed into one row (claims become weighted claims), w into two columns
of two blocks, u is not packed, b has one block.")

(defparameter *proto-cmax* 7)
(defparameter *proto-kco* 17 "The layer parameter Kco of the scenario's domain.")

(defun proto-data (state)
  "Random tensor data for *PROTO-GROUPS*, and the sumcheck's and the domain's tables."
  (flet ((rand (n) (coerce (loop repeat n collect (random +p+ state)) 'vector))
         (small (n) (coerce (loop repeat n collect (random 50 state)) 'vector)))
    (let ((x (small 16)))
      (append (loop for (nil . items) in *proto-groups*
                    append (loop for (name k c) in items collect (cons name (rand (* k (expt 2 c))))))
              (list (cons "t0" (rand 24)) (cons "t1" (rand 6)) (cons "t2" (rand 2))
                    (cons "s0" (rand 2)) (cons "s1" (rand 2))
                    (cons "x" x)
                    (cons "y" (map 'vector (lambda (v) (mod (+ (* v v) *proto-kco*) +p+)) x)))))))

(defun data-of (name data) (cdr (assoc name data :test #'equal)))

(defun tensor-mle (values k c point)
  "The MLE of a tensor of K blocks of 2^c values at POINT (c + nbits K EXTs), absent blocks zero."
  (let ((padded (make-array (expt 2 (+ c (exacta.proto:nbits k))) :initial-element 0)))
    (replace padded values)
    (mle-eval padded point 4)))

(defun proto-domain (data)
  (exacta.domain:make-domain
   :name "toy.dom" :n 3 :blocks 2
   :columns `((x :data ,(data-of "x" data)) (y :data ,(data-of "y" data)))
   :constraints '((- y (* x x) kco))
   :lookups '((:tag 5 :values (x)) (:tag 9 :values (y) :dynamic t) (:tag 9 :values (y) :num -1))
   :parameters `((kco . ,*proto-kco*))))

(defun proto-sumcheck-claim (context statement tables k n eq weights constants)
  "The honest claim sum_b bw_b sum_x w(x) F(tables(x, b))."
  (let ((full (loop for (symbol kind c shared data) in tables
                    collect (cons symbol (exacta.proto::full-table kind c shared data k n)))))
    (if (exacta.proto:prover-p context)
        (make-ext 4)
        (reduce #'ext+ (loop for b below k
                             collect (let ((s (exacta.sumcheck:hypercube-sum
                                               (exacta.proto::strip-pieces statement) n
                                               (loop for (symbol . values) in full
                                                     collect (cons symbol (subseq values (* b (expt 2 n)) (* (1+ b) (expt 2 n)))))
                                               4 :eq eq :constants constants)))
                                       (if weights (ext* s (elt weights b)) s)))
                :initial-value (make-ext 4)))))

(defun proto-scenario (context data)
  "Every operation of the context, in order; returns what each produced (for the verifier, what
Sable's Ctx must produce too).  DATA stands in, for the verifier, for the earlier steps that would
have given it the claimed values."
  (let ((out '())
        (tr (exacta.proto:context-tr context))
        (prover (exacta.proto:prover-p context)))
    (flet ((note (key value) (push (cons key value) out)))
      (loop for (g . items) in *proto-groups* do (exacta.proto:declare-group context g items))
      (loop for (g . items) in *proto-groups*
            do (exacta.proto:commit-group context g (and prover (loop for (name) in items collect (assoc name data :test #'equal)))))
      (setf (exacta.proto:context-assign context)
            (exacta.proto:parse-assign (send-bytes tr (and prover (exacta.proto:assign-octets '((("toy.dom" 0) . 0)))))))
      (exacta.proto:draw-lookup-challenges context)
      ;; claims
      (let ((pa (exacta.proto:context-challenge context 6))
            (pu (exacta.proto:context-challenge context 6))
            (pw (exacta.proto:context-challenge context 4))
            (pb (exacta.proto:context-challenge context 5))
            (wb (exacta.proto:context-challenge context)))
        (let ((va (tensor-mle (data-of "a" data) 6 3 pa))
              (vu (tensor-mle (data-of "u" data) 3 4 pu))
              (vw (coerce (loop for b below 4 collect (mle-eval (subseq (data-of "w" data) (* b 16) (* (1+ b) 16)) pw 4)) 'vector))
              (vb (ext* wb (mle-eval (data-of "b" data) pb 4))))
          (note :inputs (list va vu vw vb))
          (note :a (exacta.proto:point-claim context "a" pa :value va :data (data-of "a" data)))
          (note :u (exacta.proto:point-claim context "u" pu :value vu :data (data-of "u" data)))
          (exacta.proto:claim context "w" pw vw)
          (exacta.proto:claim context "w" pw (list (cons 0 (aref vw 0)) (cons 3 (aref vw 3))))
          (exacta.proto:weighted-claim context "b" pb (list (cons 0 wb)) vb)))
      ;; sumchecks: Prog-style pieces, ROW/COL/shared tables, blocks with weights, eq mode; then the
      ;; same products as one piece (zk/'s terms) and as two (a Prog), whose header degrees differ
      (let* ((eq (exacta.proto:context-challenge context 3))
             (bw (exacta.proto:context-challenge context 3))
             (c0 (exacta.proto:context-challenge context))
             (tables `((t0 :full 0 nil ,(data-of "t0" data)) (t1 :row 2 nil ,(data-of "t1" data))
                       (t2 :col 1 t ,(data-of "t2" data))))
             (statement `(+ (* c0 ,(exacta.proto:piece '(* t0 t1))) ,(exacta.proto:piece '(* t2 t2))))
             (constants `((c0 . ,c0))))
        (let ((claim (proto-sumcheck-claim context statement tables 3 3 eq bw constants)))
          (multiple-value-bind (rho finals)
              (exacta.proto:run-sumcheck context "sc-a" 3 3 statement claim tables
                                         :eq eq :block-weights bw :constants constants)
            (note :sc-a (list claim rho finals)))))
      (dolist (style '(:terms :prog))
        (let* ((tables `((s0 :col 1 nil ,(data-of "s0" data)) (s1 :row 2 nil ,(data-of "s1" data))))
               (statement (if (eq style :terms)
                              (exacta.proto:piece '(* s0 s1))
                              `(* ,(exacta.proto:piece 's0) ,(exacta.proto:piece 's1)))))
          (let ((claim (proto-sumcheck-claim context statement tables 1 3 nil nil nil)))
            (multiple-value-bind (rho finals)
                (exacta.proto:run-sumcheck context (format nil "sc-~(~A~)" style) 3 1 statement claim tables)
              (note style (list (exacta.proto:sumcheck-degree statement tables 3) claim rho finals))))))
      ;; a domain: a constraint with a layer parameter, a lookup into the table, a dynamic pair
      (let ((routes '()))
        (exacta.proto:run-domain context (proto-domain data)
                                 (loop for name in '(x y)
                                       collect (let ((name name))
                                                 (cons name (lambda (point value) (push (list name point value) routes))))))
        (note :domain (reverse routes)))
      (dolist (g *proto-groups*) (exacta.proto:open-group context (car g)))
      (pcs-finish* context)
      (unless prover
        (exacta.proto:check-lookup-sets context)
        (note :chunk-sum (gethash 0 (exacta.proto::context-chunk-sums context))))
      (reverse out))))

(defun pcs-finish* (context) (exacta.pcs:pcs-finish (exacta.proto:context-pcs context)))

(defun proto-run (data &optional messages)
  "The scenario as the prover (no MESSAGES) or as the verifier of MESSAGES."
  (let* ((tr (if messages
                 (make-transcript "proto-test" 4 :messages messages)
                 (make-transcript "proto-test" 4)))
         (context (exacta.proto:make-context (if messages :verify :prove) tr
                                             :cmax *proto-cmax* :queries 6 :grind 4)))
    (values (proto-scenario context data) tr)))

(defun test-proto-roundtrip ()
  (let ((data (proto-data (sb-ext:seed-random-state 61))))
    (multiple-value-bind (proved tr) (proto-run data)
      (declare (ignore proved))
      (let ((messages (transcript-messages tr)))
        (multiple-value-bind (out vtr) (proto-run data messages)
          (assert (transcript-finished-p vtr))
          (assert (equal (mapcar #'car out) '(:inputs :a :u :sc-a :terms :prog :domain :chunk-sum)))
          ;; the header degrees of the two styles
          (assert (= (first (cdr (assoc :terms out))) 1))
          (assert (= (first (cdr (assoc :prog out))) 2)))
        ;; a changed field value or byte anywhere is rejected
        (let ((rejected 0) (accepted '()) (state (sb-ext:seed-random-state 7)))
          (loop for i below (length messages)
                for m = (aref messages i)
                when (and (typep m 'field-array) (plusp (array-total-size m)))
                  do (let ((bad (copy-seq messages))
                           (m2 (make-array (array-dimensions m) :element-type (array-element-type m)))
                           (j (random (array-total-size m) state)))
                       (dotimes (x (array-total-size m)) (setf (row-major-aref m2 x) (row-major-aref m x)))
                       (setf (row-major-aref m2 j) (mod (1+ (row-major-aref m j)) +p+))
                       (setf (aref bad i) m2)
                       (if (signalled 'exacta-error (lambda () (proto-run data bad)))
                           (incf rejected)
                           (push i accepted))))
          (assert (null accepted) () "forgeries accepted at messages ~A" accepted)
          (assert (plusp rejected)))))))

(defun json-ext (e) (json-ints e))
(defun json-point (point) (format nil "[~{~A~^,~}]" (mapcar #'json-ext point)))
(defun json-vector (values) (format nil "[~{~A~^,~}]" (map 'list #'json-ext values)))

(defun export-proto-cases (path)
  "Write the context scenario's proof, its inputs and what Exacta's verifier computed, as JSON.
Sable's Ctx (zk/proto.py) replayed exactly this; TEST-FROZEN-CASES pins it."
  (let ((data (proto-data (sb-ext:seed-random-state 61))))
    (multiple-value-bind (proved tr) (proto-run data)
      (declare (ignore proved))
      (let* ((messages (transcript-messages tr))
             (out (proto-run data messages)))
        (flet ((noted (key) (cdr (assoc key out))))
          (with-open-file (s path :direction :output :if-exists :supersede)
            (destructuring-bind (va vu vw vb) (noted :inputs)
              (format s "{\"proof\": \"~(~{~2,'0X~}~)\",~%\"cmax\": ~D, \"queries\": 6, \"grind\": 4, \"kco\": ~D,~%"
                      (coerce (exacta.proofio:encode-proof messages) 'list) *proto-cmax* *proto-kco*)
              (format s "\"groups\": [~{~A~^,~}],~%"
                      (loop for (g . items) in *proto-groups*
                            collect (format nil "[\"~A\", [~{~A~^,~}]]" g
                                            (loop for (name k c shape) in items
                                                  collect (format nil "[\"~A\", ~D, ~D, ~A]" name k c
                                                                  (cond ((eq shape :no-pack) "\"no-pack\"")
                                                                        (shape (json-ints shape))
                                                                        (t "null")))))))
              (format s "\"values\": {\"a\": ~A, \"u\": ~A, \"w\": ~A, \"b\": ~A},~%"
                      (json-ext va) (json-ext vu) (json-vector vw) (json-ext vb))
              (format s "\"a\": ~A, \"u\": ~A,~%" (json-ext (noted :a)) (json-ext (noted :u)))
              (destructuring-bind (claim rho finals) (noted :sc-a)
                (format s "\"sc-a\": {\"claim\": ~A, \"rho\": ~A, \"finals\": [~{~A~^,~}]},~%"
                        (json-ext claim) (json-point rho) (mapcar #'json-vector finals)))
              (dolist (style '(:terms :prog))
                (destructuring-bind (degree claim rho finals) (noted style)
                  (format s "\"sc-~(~A~)\": {\"degree\": ~D, \"claim\": ~A, \"rho\": ~A, \"finals\": [~{~A~^,~}]},~%"
                          style degree (json-ext claim) (json-point rho) (mapcar #'json-vector finals))))
              (format s "\"domain\": [~{~A~^,~}],~%"
                      (loop for (name point value) in (noted :domain)
                            collect (format nil "[\"~(~A~)\", ~A, ~A]" name (json-point point) (json-vector value))))
              (format s "\"chunk_sum\": ~A}~%" (json-ext (noted :chunk-sum)))))))))
  path)

;;; ------------------------------------------------------------------ the table side of LogUp

(defparameter *small-log-rows* 9)

(defun small-layout ()
  "Range tags 0 .. 6 and a 255-row segment for tag 25 (STAR8): 382 rows of a 2^9-row table."
  (let ((offset 0))
    (append (loop for k from 0 to 6 collect (prog1 (list k offset 0 (expt 2 k)) (incf offset (expt 2 k))))
            (list (list 25 offset 0 255)))))

(defun small-table ()
  (let ((columns (loop repeat 7 collect (make-array (expt 2 *small-log-rows*) :initial-element 0))))
    (loop for (tag base nil size) in (small-layout)
          do (dotimes (i size)
               (setf (aref (first columns) (+ base i)) tag
                     (aref (second columns) (+ base i)) i)))
    columns))

(defun range-domain (state)
  (let ((n 3) (k 2))
    (flet ((col (bound) (coerce (loop repeat (* k (expt 2 n)) collect (random bound state)) 'vector)))
      (exacta.domain:make-domain
       :name "toy.range" :n n :blocks k
       :columns `((x :data ,(col 64)) (y :data ,(col 255)) (z :data ,(col 8)))
       :lookups '((:tag 6 :values (x)) (:tag 25 :values (y)) (:tag 3 :values (z)))))))

(defun table-scenario (context domain)
  "The table's commitment, the commit pass's counting, the domain, the table side and the openings.
Returns the domain's claims, the chunk sums (verifier) and the claims left on tab<j> and mult<i>."
  (let* ((tr (exacta.proto:context-tr context))
         (prover (exacta.proto:prover-p context))
         (exacta.tables:*log-rows* *small-log-rows*)
         (columns (and prover (small-table)))
         (chunks (and prover (exacta.tables:make-chunks :limit 32 :log-rows *small-log-rows*)))
         (out '()))
    (exacta.proto:declare-group context "tab" (loop for j below 7 collect (list (format nil "tab~D" j) 1 *small-log-rows*)))
    (exacta.proto:commit-group context "tab" (and prover (loop for c in columns for j from 0 collect (cons (format nil "tab~D" j) c))))
    (when prover (exacta.tables:count-lookups chunks domain :layout (small-layout)))
    (let* ((assign (exacta.proto:parse-assign
                    (send-bytes tr (and prover (exacta.proto:assign-octets (exacta.tables:chunks-assignments chunks))))))
           (nchunks (1+ (loop for c being the hash-values of assign maximize c))))
      (setf (exacta.proto:context-assign context) assign)
      (dotimes (i nchunks)
        (let ((name (format nil "mult~D" i)))
          (exacta.proto:declare-group context name (list (list name 1 *small-log-rows*)))
          (exacta.proto:commit-group context name (and prover (list (cons name (exacta.tables:chunk-multiplicities chunks i)))))))
      (exacta.proto:draw-lookup-challenges context)
      (let ((claims '()))
        (exacta.proto:run-domain context domain
                                 (loop for name in '(x y z)
                                       collect (let ((name name)) (cons name (lambda (p v) (push (list name p v) claims))))))
        (push (cons :domain (reverse claims)) out))
      (let ((sums (and (not prover)
                       (loop for i below nchunks
                             collect (gethash i (exacta.proto::context-chunk-sums context) (make-ext 4))))))
        (push (cons :sums sums) out)
        (exacta.tables:table-side context :columns columns :chunks chunks :sums sums))
      (push (cons :pending (loop for name in (append (loop for j below 7 collect (format nil "tab~D" j))
                                                     (loop for i below nchunks collect (format nil "mult~D" i)))
                                 collect (cons name (exacta.proto:pending-claims context name))))
            out)
      (exacta.proto:open-group context "tab")
      (dotimes (i nchunks) (exacta.proto:open-group context (format nil "mult~D" i)))
      (exacta.pcs:pcs-finish (exacta.proto:context-pcs context))
      (unless prover (exacta.proto:check-lookup-sets context))
      (reverse out))))

(defun table-run (domain &optional messages)
  (let* ((tr (if messages (make-transcript "table-test" 4 :messages messages) (make-transcript "table-test" 4)))
         (context (exacta.proto:make-context (if messages :verify :prove) tr :cmax 9 :queries 6 :grind 4)))
    (values (table-scenario context domain) tr)))

(defun test-table-side ()
  (let ((domain (range-domain (sb-ext:seed-random-state 71))))
    (multiple-value-bind (proved tr) (table-run domain)
      (declare (ignore proved))
      (let ((messages (transcript-messages tr)))
        (multiple-value-bind (out vtr) (table-run domain messages)
          (assert (transcript-finished-p vtr))
          (assert (= (length (cdr (assoc :sums out))) 2)))
        (let ((accepted '()) (state (sb-ext:seed-random-state 9)))
          (loop for i below (length messages)
                for m = (aref messages i)
                when (and (typep m 'field-array) (plusp (array-total-size m)))
                  do (let ((bad (copy-seq messages))
                           (m2 (make-array (array-dimensions m) :element-type (array-element-type m)))
                           (j (random (array-total-size m) state)))
                       (dotimes (x (array-total-size m)) (setf (row-major-aref m2 x) (row-major-aref m x)))
                       (setf (row-major-aref m2 j) (mod (1+ (row-major-aref m j)) +p+))
                       (setf (aref bad i) m2)
                       (unless (signalled 'exacta-error (lambda () (table-run domain bad)))
                         (push i accepted))))
          (assert (null accepted) () "forgeries accepted at messages ~A" accepted))))))

(defun export-table-cases (path)
  "Write the table-side scenario's proof, its domain and what Exacta's verifier computed, as JSON.
Sable's Ctx and logup.table_side (at 2^9 rows) replayed exactly this; TEST-FROZEN-CASES pins it."
  (let ((domain (range-domain (sb-ext:seed-random-state 71))))
    (multiple-value-bind (proved tr) (table-run domain)
      (declare (ignore proved))
      (let* ((messages (transcript-messages tr))
             (out (table-run domain messages)))
        (with-open-file (s path :direction :output :if-exists :supersede)
          (format s "{\"proof\": \"~(~{~2,'0X~}~)\", \"log_rows\": ~D,~%"
                  (coerce (exacta.proofio:encode-proof messages) 'list) *small-log-rows*)
          (format s "\"domain\": {\"name\": \"~A\", \"n\": ~D, \"K\": ~D, \"lookups\": [~{~A~^,~}]},~%"
                  (exacta.domain:domain-name domain) (exacta.domain:domain-n domain) (exacta.domain:domain-blocks domain)
                  (loop for lookup in (exacta.domain:domain-lookups domain)
                        collect (format nil "{\"tag\": ~A, \"values\": [~{~A~^,~}]}"
                                        (json-form (getf lookup :tag)) (mapcar #'json-form (getf lookup :values)))))
          (format s "\"claims\": [~{~A~^,~}],~%"
                  (loop for (name point value) in (cdr (assoc :domain out))
                        collect (format nil "[\"~(~A~)\", ~A, ~A]" name (json-point point) (json-vector value))))
          (format s "\"sums\": ~A,~%" (json-vector (cdr (assoc :sums out))))
          (format s "\"pending\": {~{~A~^,~}}}~%"
                  (loop for (name . claims) in (cdr (assoc :pending out))
                        collect (format nil "\"~A\": [~{~A~^,~}]" name
                                        (loop for (point values) in claims
                                              collect (format nil "[~A, ~A]" (json-point point)
                                                              (json-vector (mapcar #'cdr values)))))))))))
  path)

;;; ------------------------------------------------------------------ public MLEs

(defun test-public-mles ()
  "Each closed form equals the MLE of its whole vector at random points."
  (let ((state (sb-ext:seed-random-state 81)))
    (flet ((point (n) (loop repeat n collect (coerce-ext (loop repeat 4 collect (random +p+ state)))))
           (mle (f n point) (mle-eval (coerce (loop for i below (expt 2 n) collect (funcall f i)) 'vector) point 4)))
      (dotimes (trial 3)
        (let ((p (point 6)))
          (dolist (s '(0 2 5))
            (assert (ext= (exacta.public:iota p s) (mle (lambda (i) (ash (ash i (- s)) s)) 6 p))))
          (assert (ext= (exacta.public:le-low-high p 3)
                        (mle (lambda (i) (if (<= (ldb (byte 3 0) i) (ldb (byte 3 3) i)) 1 0)) 6 p)))
          (dolist (y '(-3 0 1 17 63 64 100))
            (assert (ext= (exacta.public:lt-const p y) (mle (lambda (v) (if (< v y) 1 0)) 6 p)))
            (assert (ext= (exacta.public:eq-const p y) (mle (lambda (v) (if (= v y) 1 0)) 6 p)))))
        ;; vocabulary indicators: R = 4 rows, V = 40 (Kv = 5 blocks of 2^3), targets in and beyond V
        (let* ((cb 3) (kv 5) (ys (list 0 13 39 47)) (p (point (+ cb 2))))
          (dolist (kind '(:lt :eq))
            (let ((closed (exacta.public:vocab-indicator p ys cb kv kind)))
              (dotimes (k kv)
                (assert (ext= (aref closed k)
                              (mle (lambda (i)
                                     (let ((v (+ (* k (expt 2 cb)) (ldb (byte cb 0) i)))
                                           (y (nth (ash i (- cb)) ys)))
                                       (if (if (eq kind :lt) (< v y) (= v y)) 1 0)))
                                   (+ cb 2) p)))))))))))

;;; ------------------------------------------------------------------ the table's values

(defun fingerprint (values)
  "Two polynomial hashes mod p of VALUES (integers, taken mod p): a fast regression pin."
  (let ((a 0) (b 0) (r1 1234567) (r2 7654321))
    (declare (type (integer 0 #.+p+) a b))
    (loop for v across values
          do (let ((v (mod v +p+)))
               (setf a (mod (+ (* a r1) v) +p+)
                     b (mod (+ (* b r2) v) +p+))))
    (list a b)))

(defun table-value-fingerprints ()
  "Fingerprints of the tiny model's value tables (E = -8, dh = 128, eG = -11, eA = -12, sampling at
temperature index 3, the router's, RoPE for theta 10^6 and S = 256).  These values were checked
equal to Sable's (model.exp_table, silu_values, samp_table, router_table, rope_tables), and the
whole 2^26-row table built from them to tables.build_table's, column by column."
  (multiple-value-bind (cos sin) (exacta.tables:rope-tables 1000000 128 256)
    (list (fingerprint (exacta.tables:exp-values -8 128))
          (fingerprint (exacta.tables:silu-values -11 -12))
          (fingerprint (exacta.tables:samp-values 3))
          (fingerprint (exacta.tables:router-values))
          (fingerprint (make-array (array-total-size cos) :displaced-to cos))
          (fingerprint (make-array (array-total-size sin) :displaced-to sin)))))

(defun test-table-values ()
  (assert (equal (table-value-fingerprints)
                 '((1397366438 838731675) (1377669541 1092948269) (46828831 101124040)
                   (1818342257 1780444900) (741164168 1772295875) (799026709 234658675)))))

;;; ------------------------------------------------------------------ F_O

(defvar *test-models*
  (loop for name in '("tiny" "qwen3")
        collect (cons name (exacta.model:read-model
                            (asdf:system-relative-pathname "exacta" (format nil "tests/models/~A.exm" name)))))
  "The test models, read when the tests are loaded (so a saved image holds them): Sable's
quantized tiny models, a dense one and one with QK-norm (Qwen3-shaped), exported to EXACTFO1.")

(defun model-fingerprints (name)
  "Fingerprints of each layer's output and of the logits of F_O on tokens from seed 5 (B = 2,
S = 256), and the greedy argmax of the first 8 rows."
  (let* ((model (cdr (assoc name *test-models* :test #'string=)))
         (tokens (let ((state (sb-ext:seed-random-state 5))) (coerce (loop repeat 512 collect (random 1024 state)) 'vector)))
         (tables (exacta.model:model-tables model 256))
         (h (exacta.model:embed-tokens model tokens))
         (out '()))
    (flet ((flat (m) (make-array (array-total-size m) :element-type (array-element-type m) :displaced-to m)))
      (loop for layer across (exacta.model:model-layers model)
            do (setf h (exacta.model:layer-forward model layer h 2 256 tables))
               (push (fingerprint (flat h)) out))
      (let ((z (exacta.model:final-logits model h)))
        (push (fingerprint (flat z)) out)
        (push (loop for i below 8
                    collect (loop with best = 0 for j below (array-dimension z 1)
                                  when (> (aref z i j) (aref z i best)) do (setf best j)
                                  finally (return best)))
              out)))
    (reverse out)))

(defparameter *model-fingerprints*
  '(("tiny" (385209351 1921462728) (67992394 1468852158) (754246383 125402219) (772 119 774 141 116 979 729 218)) ("qwen3" (104177360 97824691) (329747358 808300888) (1050931097 150939515) (622 30 167 749 802 34 332 622)))
  "The fingerprints of MODEL-FINGERPRINTS.  Every intermediate value of these passes -- 54 per model:
the quantizations, NormScale, QKV, RoPE, attention, the output projection, the MLP and the residuals
of both layers, the logits -- was checked equal to Sable's (model.py, 81807ef, host) on the same
model files and tokens.")

(defun test-model-forward ()
  (loop for (name . pins) in *model-fingerprints*
        do (assert (equal (model-fingerprints name) pins) () "F_O of ~A differs" name)))

;;; ------------------------------------------------------------------ the per-layer steps

(defun step-shape (model)
  (let ((cfg (exacta.model:model-cfg model)))
    (multiple-value-bind (cb cbv) (exacta.llm:column-block-bits cfg)
      (exacta.llm:make-shape cfg 2 256 cb cbv 16))))

(defun step-witness (name l)
  "Layer L's witness of model NAME on the tokens from seed 5."
  (let* ((model (cdr (assoc name *test-models* :test #'string=)))
         (shape (step-shape model))
         (tables (exacta.model:model-tables model 256))
         (h (exacta.model:embed-tokens model (let ((state (sb-ext:seed-random-state 5)))
                                              (coerce (loop repeat 512 collect (random 1024 state)) 'vector)))))
    (dotimes (i l) (setf h (exacta.model:layer-forward model (aref (exacta.model:model-layers model) i) h 2 256 tables)))
    (values (exacta.steps:layer-witness model l h shape tables) model shape)))

(defun step-run (model shape l step groups tensors &optional witness messages)
  "One step of layer L in isolation (prover with WITNESS, or verifier of MESSAGES): the layout's
groups declared without merge sets or segments, GROUPS committed (TENSORS gives each group's), the step's lookups
counted and assigned, Z and beta drawn, the layer's header, the step.  Returns what it left: every
pending claim, the virtual claims and the step's state."
  (let* ((tr (if messages (make-transcript "step-test" 4 :messages messages) (make-transcript "step-test" 4)))
         (context (exacta.proto:make-context (if messages :verify :prove) tr :queries 6 :grind 4))
         (state (exacta.steps::make-step-state))
         (layout (exacta.llm:commitment-layout (exacta.model:model-cfg model) shape)))
    (loop for (name items pad) in (getf layout :raw-groups)
          do (exacta.proto:declare-group context name items :pad pad))
    (dolist (group groups)
      (exacta.proto:commit-group context group (and witness (funcall tensors witness group))))
    (let ((assignments
            (when witness
              (let ((chunks (exacta.tables:make-chunks)))
                (setf (exacta.proto::context-mode context) :commit
                      (exacta.proto:context-counter context) (lambda (domain) (exacta.tables:count-lookups chunks domain)))
                (setf exacta.steps::*layer-parameters* (exacta.steps:layer-parameters model l))
                (funcall step context model shape l (exacta.steps::make-step-state) witness)
                (setf (exacta.proto::context-mode context) :prove)
                (exacta.tables:chunks-assignments chunks)))))
      (setf (exacta.proto:context-assign context)
            (exacta.proto:parse-assign (send-bytes tr (and witness (exacta.proto:assign-octets assignments))))))
    (exacta.proto:draw-lookup-challenges context)
    (exacta.steps::layer-head context model l)
    (funcall step context model shape l state witness)
    (values (list :claims (sort (loop for name being the hash-keys of (exacta.proto::context-claims context)
                                      collect (cons name (gethash name (exacta.proto::context-claims context))))
                                #'string< :key #'car)
                  :wclaims (sort (loop for name being the hash-keys of (exacta.proto::context-wclaims context)
                                       collect (cons name (gethash name (exacta.proto::context-wclaims context))))
                                 #'string< :key #'car)
                  :virtual (sort (loop for key being the hash-keys of (exacta.proto::context-virtual context)
                                       collect (cons key (gethash key (exacta.proto::context-virtual context))))
                                 #'string< :key (lambda (k) (format nil "~S" (car k))))
                  :state state)
            tr)))

(defun json-claims (alist)
  "Pending block claims: {name: [[point, [[block, value], ...]], ...]}."
  (format nil "{~{~A~^,~}}"
          (loop for (name . claims) in alist
                collect (format nil "\"~A\": [~{~A~^,~}]" name
                                (loop for (point values) in claims
                                      collect (format nil "[~A, [~{~A~^,~}]]" (json-point point)
                                                      (loop for (b . v) in values collect (format nil "[~D, ~A]" b (json-ext v)))))))))

(defun json-wclaims (alist)
  (format nil "{~{~A~^,~}}"
          (loop for (name . claims) in alist
                collect (format nil "\"~A\": [~{~A~^,~}]" name
                                (loop for (point weights value) in claims
                                      collect (format nil "[~A, [~{~A~^,~}], ~A]" (json-point point)
                                                      (loop for (b . v) in weights collect (format nil "[~D, ~A]" b (json-ext v)))
                                                      (json-ext value)))))))

(defun export-step-case (path step groups)
  "Write STEP of the tiny model's layer 0 in isolation (its proof and everything it left), as JSON."
  (multiple-value-bind (w model shape) (step-witness "tiny" 0)
    (let ((tensors (lambda (w group) (step-tensors w group shape))))
      (multiple-value-bind (proved tr) (step-run model shape 0 step groups tensors w)
        (declare (ignore proved))
        (let* ((messages (transcript-messages tr))
               (out (step-run model shape 0 step groups tensors nil messages))
               (state (getf out :state)))
          (with-open-file (s path :direction :output :if-exists :supersede)
            (format s "{\"proof\": \"~(~{~2,'0X~}~)\", \"layer\": 0,~%" (coerce (exacta.proofio:encode-proof messages) 'list))
            (format s "\"claims\": ~A,~%\"wclaims\": ~A,~%" (json-claims (getf out :claims)) (json-wclaims (getf out :wclaims)))
            (format s "\"virtual\": {~{~A~^,~}},~%"
                    (loop for ((l key) . items) in (getf out :virtual)
                          collect (format nil "\"~D|~A\": [~{~A~^,~}]" l key
                                          (loop for (point values) in items
                                                collect (format nil "[~A, ~A]" (json-point point) (json-vector values))))))
            (format s "\"down\": [~{~A~^,~}],~%"
                    (loop for (po pm parts) in (exacta.steps::step-state-down-claims state)
                          collect (format nil "[~A, ~A, ~A]" (json-point po) (json-point pm) (json-vector parts))))
            (format s "\"gather\": {~{~A~^,~}}"
                    (loop for (key point value) in (exacta.steps::step-state-gather state)
                          collect (format nil "\"~A\": [~A, ~A]" key (json-point point) (json-ext value))))
            (let ((oh (exacta.steps::step-state-oh-pending state)))
              (when oh
                (format s ",~%\"oh\": [~{~A~^,~}]"
                        (loop for (jbn rho full) in oh
                              collect (format nil "[\"~A\", ~A, ~A]" jbn (json-point rho) (json-vector full))))))
            (format s "}~%"))))))
  path)

(defun export-step-mo-cases (path)
  "step_mo of the tiny model's layer 0.  Sable's Proto.step_mo replayed exactly this; TEST-FROZEN-CASES
pins it."
  (export-step-case path #'exacta.steps:step-mo '("0.mo")))

(defun export-step-mlp-cases (path)
  "step_mo, then the MLP slice, of the tiny model's layer 0.  Sable's Proto replayed exactly this;
TEST-FROZEN-CASES pins it."
  (export-step-case path #'step-mo-mlp '("0.mo" "0.mlp0")))

(defun step-tensors (w group shape)
  "The witness tensors of GROUP (Proto.tensors), for the groups the step tests commit."
  (let ((cb (exacta.llm:shape-cb shape)))
    (flet ((rm (m) (exacta.steps:rm-layout m cb)))
      (cond ((string= group "0.mo") (list (cons "0.s2" (rm (getf w :s2))) (cons "0.os2" (rm (getf w :os2)))))
            ((string= group "0.mlp0")
             (let ((mlp (getf (getf w :forward) :mlp)))
               (list (cons "0.m0.c" (rm (getf mlp :c))) (cons "0.m0.a" (rm (getf mlp :a)))
                     (cons "0.m0.qu" (rm (getf (getf w :ru) :q))) (cons "0.m0.qm" (rm (getf (getf w :rm) :q))))))
            (t (error "step-tensors: ~A" group))))))

(defun step-mo-mlp (context model shape l state witness)
  "step_mo, then every MLP slice (Proto.layer_steps' order)."
  (exacta.steps:step-mo context model shape l state witness)
  (dotimes (s (length (exacta.llm:mlp-slices shape)) state)
    (exacta.steps:step-mlp context model shape l state witness s)))
