;;;; cli.lisp -- the command line of the saved SBCL executable (build.lisp).
;;;;
;;;; MAIN is the toplevel function of the image.  It maps the condition hierarchy onto exit codes in
;;;; one HANDLER-CASE, so the rest of the code signals conditions and never calls EXIT:
;;;;
;;;;   0  success          1  proof rejected (MALFORMED-PROOF, VERIFICATION-FAILURE)
;;;;   2  usage, I/O       3  internal error           130  interrupted

(defpackage #:sable.cli
  (:use #:cl #:sable.conditions #:sable.field #:sable.blake3 #:sable.transcript)
  (:export #:main #:run #:usage-error #:*version*))

(in-package #:sable.cli)

(defparameter *version* "1.0.0-cl")

(define-condition usage-error (sable-error)
  ((message :initarg :message :reader usage-error-message))
  (:report (lambda (c s) (format s "~A" (usage-error-message c)))))

(defun usage (fmt &rest args)
  (error 'usage-error :message (apply #'format nil fmt args)))

(defparameter *help* "sable ~A -- Common Lisp (SBCL) port of the Sable host-side verifier core

Usage:
  sable inspect PROOF...           parse proof files (zk/proofio.py format) and summarise them
  sable hash FILE...               BLAKE3 digests, in b3sum's format
  sable sumcheck-demo [N] [D] [T]  prove and verify a random sumcheck of T tables over N variables
                                   in F_p^D on the CPU, then check that a tampered proof is rejected
  sable selftest                   run the test suite against the vectors of the Python code
  sable version | help

Exit status: 0 success, 1 proof rejected, 2 usage or I/O error, 3 internal error.
Set SABLE_DEBUG=1 to enter the debugger on an internal error instead of exiting.
")

(defun parse-int (s what &key (min 0) (max most-positive-fixnum))
  (let ((v (ignore-errors (parse-integer s))))
    (unless (and v (<= min v max))
      (usage "~A must be an integer in [~D, ~D], got ~S" what min max s))
    v))

(defun cmd-inspect (paths)
  (unless paths (usage "inspect: no proof file given"))
  (dolist (p paths)
    (format t "~&~A~%" p)
    (sable.proofio:describe-proof (sable.proofio:read-proof-file p))))

(defun cmd-hash (paths)
  (unless paths (usage "hash: no file given"))
  (dolist (p paths)
    (let ((h (make-blake3))
          (buf (make-octets (* 1024 1024))))
      (with-open-file (in p :element-type '(unsigned-byte 8))
        (loop for n = (read-sequence buf in)
              while (plusp n)
              do (blake3-update h buf :end n)))
      (format t "~(~{~2,'0X~}~)  ~A~%" (coerce (blake3-digest h) 'list) p))))

(defun random-ext (d state)
  (let ((e (make-ext d)))
    (dotimes (i d e) (setf (aref e i) (random +p+ state)))))

(defun cmd-sumcheck-demo (args)
  (let* ((n (if (first args) (parse-int (first args) "N" :min 1 :max 20) 10))
         (d (if (second args) (parse-int (second args) "D" :min 4 :max 8) 8))
         (ntab (if (third args) (parse-int (third args) "T" :min 1 :max 8) 3))
         (d (check-degree d))
         (state (sb-ext:seed-random-state 1))
         (tables (loop repeat ntab
                       collect (coerce (loop repeat (ash 1 n) collect (random-ext d state)) 'simple-vector)))
         (eq-r (coerce (loop repeat n collect (random-ext d state)) 'simple-vector))
         (claim (sable.sumcheck:product-claim tables n d :eq-r eq-r))
         (t0 (get-internal-real-time))
         (ptr (make-transcript "demo" d)))
    (multiple-value-bind (rho finals) (sable.sumcheck:prove-product ptr n tables :eq-r eq-r)
      (declare (ignore rho))
      (let* ((bytes (sable.proofio:write-proof (transcript-messages ptr)))
             (t1 (get-internal-real-time)))
        (flet ((verify-bytes (b)
                 (let ((vtr (make-transcript "demo" d :messages (sable.proofio:read-proof b))))
                   (multiple-value-bind (vrho reduced)
                       (sable.sumcheck:verify vtr n ntab claim :eq-r eq-r)
                     (declare (ignore vrho))
                     (sable.sumcheck:check-reduced reduced (reduce #'ext* finals))
                     (unless (transcript-finished-p vtr)
                       (reject :trailing-messages))))))
          (verify-bytes bytes)
          (let ((t2 (get-internal-real-time)))
            (format t "~&sumcheck: ~D tables over ~D variables in F_p^~D, eq mode~%" ntab n d)
            (format t "proved in ~,3F s (~:D-byte proof), verified in ~,3F s~%"
                    (/ (- t1 t0) internal-time-units-per-second) (length bytes)
                    (/ (- t2 t1) internal-time-units-per-second)))
          ;; change one value of the first round message: the proof must now be rejected
          (let* ((msgs (sable.proofio:read-proof bytes))
                 (m0 (aref msgs 0)))
            (setf (aref m0 0 0) (fadd (aref m0 0 0) 1))
            (handler-case (progn (verify-bytes (sable.proofio:write-proof msgs))
                                 (error "a tampered proof was accepted"))
              (verification-failure (c)
                (format t "tampered proof rejected: ~A~%" c)))))))))

(defun run (args)
  "Run the command line ARGS (a list of strings, without the program name).  Signals conditions."
  (let ((cmd (first args)))
    (cond ((or (null cmd) (member cmd '("help" "-h" "--help") :test #'string=))
           (format t *help* *version*))
          ((member cmd '("version" "--version") :test #'string=)
           (format t "sable ~A (~A ~A)~%" *version* (lisp-implementation-type) (lisp-implementation-version)))
          ((string= cmd "inspect") (cmd-inspect (rest args)))
          ((string= cmd "hash") (cmd-hash (rest args)))
          ((string= cmd "sumcheck-demo") (cmd-sumcheck-demo (rest args)))
          ((string= cmd "selftest")
           (let ((run-tests (find-symbol "RUN-TESTS" "SABLE.TESTS")))
             (unless (and run-tests (fboundp run-tests))
               (usage "selftest: this image was built without the test suite"))
             (unless (funcall run-tests)
               (error 'verification-failure :check :selftest :detail "some tests failed"))))
          (t (usage "unknown command ~S (try: sable help)" cmd)))))

(defun expected-condition-p (c)
  "Conditions that are answers (a rejected proof, a bad command line), not bugs."
  (typep c '(or usage-error malformed-proof verification-failure file-error)))

(defun main ()
  "The toplevel of the saved executable."
  (let ((debug (equal (sb-ext:posix-getenv "SABLE_DEBUG") "1")))
    (unless debug (sb-ext:disable-debugger))
    (sb-ext:exit
     :abort nil
     :code (handler-case
               ;; HANDLER-BIND runs before the stack unwinds, so with SABLE_DEBUG=1 the debugger
               ;; sees the frames of the failing code; HANDLER-CASE below then maps the condition
               ;; to an exit code.
               (handler-bind ((error (lambda (c)
                                       (when (and debug (not (expected-condition-p c)))
                                         (invoke-debugger c)))))
                 (run (rest sb-ext:*posix-argv*))
                 (finish-output)
                 0)
             (sb-sys:interactive-interrupt ()
               (format *error-output* "~&interrupted~%")
               130)
             (usage-error (c)
               (format *error-output* "~&sable: ~A~%" c)
               2)
             ((or malformed-proof verification-failure) (c)
               (format *error-output* "~&sable: ~A~%" c)
               1)
             (file-error (c)
               (format *error-output* "~&sable: ~A~%" c)
               2)
             (error (c)
               (format *error-output* "~&sable: internal error: ~A~%" c)
               3)))))
