;;;; toplevel.lisp -- the toplevel of the saved executable: it evaluates Lisp forms.
;;;;
;;;;   sable '(describe-proof (read-proof-file "proof_1x4096.proof"))'
;;;;   sable '(normalize (quote (* (+ c0 3) (- c1 c0))))'
;;;;   sable - < script.lisp          forms from standard input
;;;;   sable                          a REPL (on a terminal)
;;;;
;;;; Forms are read and evaluated in SABLE-USER, which uses every Sable package, so the exported
;;;; functions are the interface and (help) lists them from their own lambda lists and
;;;; documentation.  The exit status comes from the condition that ended the run:
;;;;
;;;;   0  success          1  proof rejected (MALFORMED-PROOF, VERIFICATION-FAILURE)
;;;;   2  bad input: an unreadable form, an unknown function or variable, a missing file
;;;;   3  internal error   130  interrupted       141  standard output closed

(defpackage #:sable.toplevel
  (:use #:cl #:sable.conditions)
  (:export #:main #:repl #:help #:*version* #:*api-packages*))

(defpackage #:sable-user
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.blake3 #:sable.transcript
        #:sable.proofio #:sable.forms #:sable.rs #:sable.sumcheck #:sable.gkr #:sable.domain #:sable.pcs #:sable.llm)
  (:import-from #:sable.toplevel #:help #:*version*)
  (:documentation "The package in which the executable reads and evaluates forms."))

(in-package #:sable.toplevel)

(defparameter *version* "1.0.0-cl")

(defparameter *api-packages*
  '(#:sable.field #:sable.forms #:sable.blake3 #:sable.transcript #:sable.proofio #:sable.rs
    #:sable.sumcheck #:sable.gkr #:sable.domain #:sable.pcs #:sable.llm #:sable.conditions #:sable.util)
  "The packages whose exported functions make up the API, as HELP lists them.")

;;; ------------------------------------------------------------------ help

(defun first-line (string)
  (and string (subseq string 0 (position #\Newline string))))

(defun help (&optional (stream *standard-output*))
  "List the exported functions of the API packages, with their lambda lists and the first line
of their documentation."
  (format stream "sable ~A: evaluate Lisp forms in SABLE-USER, which uses these packages.~%" *version*)
  (dolist (name *api-packages*)
    (let ((package (find-package name)))
      (when package
        (format stream "~%~(~A~)~%" (package-name package))
        (let ((*package* package))
          (dolist (symbol (sort (loop for s being the external-symbols of package
                                      when (and (fboundp s) (not (macro-function s))) collect s)
                                #'string<))
            (format stream "  ~(~S~)~%~@[      ~A~%~]"
                    (cons symbol (sb-introspect:function-lambda-list symbol))
                    (first-line (documentation symbol 'function))))))))
  (values))

;;; ------------------------------------------------------------------ reading and evaluating

(defun evaluate (form)
  "The values of FORM.  Style warnings from compiling it (an unknown function, say) are muffled:
the error that follows, if any, says the same."
  (handler-bind ((style-warning #'muffle-warning))
    (multiple-value-list (eval form))))

(defun print-values (values)
  "Print VALUES as a REPL does; a sole NIL (an operation done for effect) prints nothing."
  (unless (equal values '(nil))
    (format t "~{~S~%~}" values))
  (finish-output))

(defun evaluate-stream (stream)
  "Read the forms of STREAM in SABLE-USER, evaluate each and print its values."
  (let ((*package* (find-package '#:sable-user))
        (eof '#:eof))
    (loop for form = (read stream nil eof)
          until (eq form eof)
          do (print-values (evaluate form)))))

(defun repl ()
  "A read-eval-print loop in SABLE-USER.  An error enters the debugger, whose ABORT restart
returns here."
  (let ((*package* (find-package '#:sable-user))
        (eof '#:eof))
    (format t "Sable ~A on ~A ~A.  (help) lists the API; end of file (C-d) leaves.~%"
            *version* (lisp-implementation-type) (lisp-implementation-version))
    (loop
      (with-simple-restart (abort "Return to the Sable REPL.")
        (format t "~&~(~A~)> " (package-name *package*))
        (finish-output)
        (let ((form (read *standard-input* nil eof)))
          (when (eq form eof)
            (terpri)
            (return))
          (print-values (evaluate form)))))))

;;; ------------------------------------------------------------------ exit status

(defun standard-output-error-p (c)
  "True for an error writing standard output, as when the reader of a pipe (head(1)) exits."
  (and (typep c 'stream-error)
       (member (stream-error-stream c) (list *standard-output* sb-sys:*stdout*))))

(deftype bad-input ()
  "Conditions caused by what the user gave, not by a bug."
  '(or reader-error end-of-file unbound-variable undefined-function file-error))

(defun answer-p (c)
  "True for a condition that is an answer to the input, not a bug."
  (or (typep c '(or malformed-proof verification-failure bad-input))
      (standard-output-error-p c)))

(defun exit-status (thunk debug)
  "Call THUNK and return the exit status for how it ended.  With DEBUG, an internal error enters
the debugger first: HANDLER-BIND runs before the stack unwinds, HANDLER-CASE after."
  (flet ((fail (status c &optional (what ""))
           (format *error-output* "~&sable: ~A~A~%" what c)
           status))
    (handler-case
        (handler-bind ((error (lambda (c)
                                (when (and debug (not (answer-p c)))
                                  (invoke-debugger c)))))
          (funcall thunk)
          0)
      (sb-sys:interactive-interrupt () 130)
      ((satisfies standard-output-error-p) () 141)
      ((or malformed-proof verification-failure) (c) (fail 1 c))
      (bad-input (c) (fail 2 c))
      (error (c) (fail 3 c "internal error: ")))))

(defun main ()
  "The toplevel of the executable: evaluate the forms of each argument (\"-\" for standard
input), or with no arguments run a REPL on a terminal and read standard input otherwise."
  (let ((arguments (rest sb-ext:*posix-argv*))
        (debug (equal (sb-ext:posix-getenv "SABLE_DEBUG") "1")))
    (if (and (null arguments) (interactive-stream-p *standard-input*))
        (progn (sb-ext:enable-debugger)          ; the image was saved without one
               (repl)
               (sb-ext:exit :code 0))
        (progn
          (unless debug (sb-ext:disable-debugger))
          (let ((status (exit-status
                         (lambda ()
                           (dolist (argument (or arguments '("-")))
                             (cond ((member argument '("-h" "--help" "help") :test #'string=) (help))
                                   ((string= argument "--version")
                                    (format t "sable ~A (~A ~A)~%" *version*
                                            (lisp-implementation-type) (lisp-implementation-version)))
                                   ((string= argument "-") (evaluate-stream *standard-input*))
                                   (t (with-input-from-string (in argument)
                                        (evaluate-stream in))))))
                         debug)))
            ;; standard output may be a closed pipe: then leave without flushing it
            (sb-ext:exit :code status :abort (= status 141)))))))
