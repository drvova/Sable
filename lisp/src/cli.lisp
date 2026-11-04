;;;; cli.lisp -- the command line of the saved SBCL executable (build.lisp).
;;;;
;;;; DEFINE-COMMAND defines a command with its lambda list and documentation, from which the help
;;;; is written.  MAIN maps the condition hierarchy onto exit codes in one place, so commands
;;;; signal conditions and never call EXIT:
;;;;
;;;;   0  success          1  proof rejected (MALFORMED-PROOF, VERIFICATION-FAILURE)
;;;;   2  usage, I/O       3  internal error           130  interrupted

(defpackage #:sable.cli
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.blake3 #:sable.transcript)
  (:export #:main #:run #:usage-error #:*version*))

(in-package #:sable.cli)

(defparameter *version* "1.0.0-cl")

(define-condition usage-error (sable-error)
  ((message :initarg :message :reader usage-error-message))
  (:report (lambda (c stream) (write-string (usage-error-message c) stream))))

(defun usage-error (control &rest arguments)
  (error 'usage-error :message (apply #'format nil control arguments)))

;;; ------------------------------------------------------------------ commands

(defvar *commands* '()
  "(name lambda-list documentation function), in definition order.")

(defun usage-words (lambda-list)
  "LAMBDA-LIST as usage text: (path &optional n &rest files) -> (\"PATH\" \"[N]\" \"[FILES...]\")."
  (loop with mode = '&required
        for item in lambda-list
        if (member item lambda-list-keywords)
          do (setf mode item)
        else
          collect (let ((name (string-upcase (if (consp item) (first item) item))))
                    (case mode
                      (&optional (format nil "[~A]" name))
                      (&rest (format nil "[~A...]" name))
                      (t name)))))

(defun check-arity (name lambda-list arguments)
  "ARGUMENTS, if their number fits LAMBDA-LIST; else a USAGE-ERROR."
  (let ((required (or (position-if (lambda (x) (member x lambda-list-keywords)) lambda-list)
                      (length lambda-list)))
        (maximum (unless (member '&rest lambda-list)
                   (length (remove '&optional lambda-list)))))
    (unless (and (<= required (length arguments))
                 (or (null maximum) (<= (length arguments) maximum)))
      (usage-error "usage: sable ~A~{ ~A~}" name (usage-words lambda-list)))
    arguments))

(defmacro define-command (name lambda-list documentation &body body)
  "Define the command NAME (a string), whose arguments (strings) are bound by LAMBDA-LIST (of
required, &optional and &rest parameters).  DOCUMENTATION is its line in the help."
  (let ((arguments (gensym "ARGUMENTS")))
    `(progn
       (setf *commands*
             (append (remove ,name *commands* :key #'first :test #'string=)
                     (list (list ,name ',lambda-list ,documentation
                                 (lambda (&rest ,arguments)
                                   (destructuring-bind ,lambda-list
                                       (check-arity ,name ',lambda-list ,arguments)
                                     ,@body))))))
       ,name)))

(defun find-command (name)
  (or (find name *commands* :key #'first :test #'string=)
      (usage-error "unknown command ~S (try: sable help)" name)))

(defun parse-count (string what min max)
  (let ((n (ignore-errors (parse-integer string))))
    (unless (and n (<= min n max))
      (usage-error "~A must be an integer in [~D, ~D], got ~S" what min max string))
    n))

(define-command "inspect" (path &rest paths)
  "parse proof files (zk/proofio.py format) and summarise their messages"
  (dolist (path (cons path paths))
    (format t "~&~A~%" path)
    (sable.proofio:describe-proof (sable.proofio:read-proof-file path))))

(defun file-digest (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (loop with hasher = (make-blake3)
          with buffer = (make-octets (expt 2 20))
          for n = (read-sequence buffer in)
          while (plusp n)
          do (blake3-update hasher buffer :end n)
          finally (return (blake3-digest hasher)))))

(define-command "hash" (path &rest paths)
  "BLAKE3 digests, in b3sum's format"
  (dolist (path (cons path paths))
    (format t "~(~{~2,'0X~}~)  ~A~%" (coerce (file-digest path) 'list) path)))

(defun random-ext (d state)
  (coerce-ext (loop repeat d collect (random +p+ state))))

(defun random-table (n d state)
  (coerce (loop repeat (expt 2 n) collect (random-ext d state)) 'vector))

(defun seconds-since (start)
  (/ (- (get-internal-real-time) start) internal-time-units-per-second))

(define-command "sumcheck-demo" (&optional (n "10") (d "8") (tables "3"))
  "prove and verify a random sumcheck of TABLES tables over N variables in F_p^D, on the CPU;
then check that a tampered proof is rejected"
  (let* ((n (parse-count n "N" 1 20))
         (d (parse-count d "D" 4 8))
         (ntables (parse-count tables "TABLES" 1 8))
         (state (sb-ext:seed-random-state 1))
         (tables (loop repeat ntables collect (random-table n (check-degree d) state)))
         (eq-r (loop repeat n collect (random-ext d state)))
         (claim (sable.sumcheck:product-claim tables n d :eq-r eq-r))
         (start (get-internal-real-time))
         (prover (make-transcript "demo" d))
         (finals (nth-value 1 (sable.sumcheck:prove-product prover n tables :eq-r eq-r)))
         (proof (sable.proofio:encode-proof (transcript-messages prover)))
         (proved (seconds-since start)))
    (flet ((verify (messages)
             (let ((verifier (make-transcript "demo" d :messages messages)))
               (sable.sumcheck:check-reduced
                (nth-value 1 (sable.sumcheck:verify verifier n ntables claim :eq-r eq-r))
                (reduce #'ext* finals))
               (unless (transcript-finished-p verifier)
                 (reject :unread-messages)))))
      (setf start (get-internal-real-time))
      (verify (sable.proofio:decode-proof proof))
      (format t "~&sumcheck: ~D tables over ~D variables in F_p^~D, eq mode~%" ntables n d)
      (format t "proved in ~,3F s (~:D-byte proof), verified in ~,3F s~%"
              proved (length proof) (seconds-since start))
      ;; change one value of the first round message: the proof must now be rejected
      (let ((tampered (sable.proofio:decode-proof proof)))
        (setf (aref (aref tampered 0) 0 0) (fadd (aref (aref tampered 0) 0 0) 1))
        (handler-case (progn (verify tampered)
                             (error "a tampered proof was accepted"))
          (verification-failure (c)
            (format t "tampered proof rejected: ~A~%" c)))))))

(define-command "selftest" ()
  "run the test suite against the vectors of the Python code"
  (let ((run-tests (find-symbol "RUN-TESTS" "SABLE.TESTS")))
    (unless (and run-tests (fboundp run-tests))
      (usage-error "selftest: this image was built without the test suite"))
    (unless (funcall run-tests)
      (reject :selftest "some tests failed"))))

(define-command "version" ()
  "print the version"
  (format t "sable ~A (~A ~A)~%" *version* (lisp-implementation-type) (lisp-implementation-version)))

(define-command "help" ()
  "print this help"
  (format t "sable ~A -- Common Lisp (SBCL) port of the Sable host-side verifier core~2%Usage:~%"
          *version*)
  (loop for (name lambda-list documentation) in *commands*
        do (format t "  sable ~A~{ ~A~}~%      ~A~%" name (usage-words lambda-list) documentation))
  (format t "~%Exit status: 0 success, 1 proof rejected, 2 usage or I/O error, 3 internal error.
SABLE_DEBUG=1 enters the debugger on an internal error instead of exiting.~%"))

;;; ------------------------------------------------------------------ toplevel

(defun run (arguments)
  "Run the command line ARGUMENTS (strings, without the program name).  Signals conditions."
  (destructuring-bind (&optional (name "help") &rest rest) arguments
    (when (member name '("-h" "--help") :test #'string=) (setf name "help"))
    (when (string= name "--version") (setf name "version"))
    (apply (fourth (find-command name)) rest)))

(defun standard-output-error-p (c)
  "True for an error writing standard output, as when the reader of a pipe, such as head(1), exits."
  (and (typep c 'stream-error)
       (member (stream-error-stream c) (list *standard-output* sb-sys:*stdout*))))

(defun expected-condition-p (c)
  "A condition that is an answer (a rejected proof, a bad command line), not a bug."
  (or (typep c '(or usage-error malformed-proof verification-failure file-error))
      (standard-output-error-p c)))

(defun main ()
  "The toplevel of the saved executable."
  (let ((debug (equal (sb-ext:posix-getenv "SABLE_DEBUG") "1")))
    (unless debug (sb-ext:disable-debugger))
    (flet ((fail (code c &optional (prefix ""))
             (format *error-output* "~&sable: ~A~A~%" prefix c)
             code))
      (let ((code (handler-case
                      ;; HANDLER-BIND runs before the stack unwinds, so with SABLE_DEBUG=1 the
                      ;; debugger sees the frames of the failing code; HANDLER-CASE then picks the
                      ;; exit code.
                      (handler-bind ((error (lambda (c)
                                              (unless (expected-condition-p c)
                                                (when debug (invoke-debugger c))))))
                        (run (rest sb-ext:*posix-argv*))
                        (finish-output)
                        0)
                    (sb-sys:interactive-interrupt ()
                      (format *error-output* "~&interrupted~%")
                      130)
                    ((satisfies standard-output-error-p) ()
                      141)
                    (usage-error (c) (fail 2 c))
                    ((or malformed-proof verification-failure) (c) (fail 1 c))
                    (file-error (c) (fail 2 c))
                    (error (c) (fail 3 c "internal error: ")))))
        ;; standard output may be a closed pipe: then leave without flushing it
        (sb-ext:exit :code code :abort (= code 141))))))
