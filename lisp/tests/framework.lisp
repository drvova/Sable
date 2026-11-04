;;;; framework.lisp -- a small test harness built on the condition system (no dependencies).
;;;;
;;;; Each test runs inside RESTART-CASE with the restarts SKIP-TEST and RETRY-TEST.  RUN-TESTS
;;;; binds a handler (HANDLER-BIND, so the stack is still intact) that records the failure and
;;;; invokes SKIP-TEST; with :DEBUG T it declines instead, so the debugger comes up with SKIP-TEST,
;;;; RETRY-TEST and ABORT-TESTS on offer.  CHECK signals TEST-FAILURE, a condition of its own, so a
;;;; failed expectation and an unexpected error are reported differently.

(defpackage #:sable.tests
  (:use #:cl #:sable.conditions #:sable.field #:sable.blake3 #:sable.transcript)
  (:export #:run-tests #:deftest #:check #:check-signals #:*tests*))

(in-package #:sable.tests)

(defvar *tests* '() "Test names, in definition order (reversed).")

(define-condition test-failure (error)
  ((form :initarg :form :reader test-failure-form)
   (detail :initarg :detail :initform nil :reader test-failure-detail))
  (:report (lambda (c s)
             (format s "~S failed~@[: ~A~]" (test-failure-form c) (test-failure-detail c)))))

(defmacro check (form &optional detail &rest args)
  "Signal TEST-FAILURE unless FORM is true; DETAIL is a format control for ARGS."
  `(unless ,form
     (error 'test-failure :form ',form :detail ,(and detail `(format nil ,detail ,@args)))))

(defmacro check-signals (type form)
  "Check that FORM signals a condition of TYPE (via ERROR or SIGNAL); return the condition."
  (let ((c (gensym "C")))
    `(let ((,c (handler-case (progn ,form nil)
                 (,type (,c) ,c))))
       (unless ,c
         (error 'test-failure :form ',form :detail ,(format nil "no ~S signalled" type)))
       ,c)))

(defmacro deftest (name &body body)
  `(progn
     (defun ,name () ,@body)
     (pushnew ',name *tests*)
     ',name))

(defun run-test (name)
  "Run one test; returns T on success.  Restarts: SKIP-TEST, RETRY-TEST."
  (loop
    (restart-case (progn (funcall name) (return t))
      (skip-test ()
        :report (lambda (s) (format s "Skip the test ~S." name))
        (return nil))
      (retry-test ()
        :report (lambda (s) (format s "Run the test ~S again." name))))))

(defun run-tests (&key (tests (reverse *tests*)) debug (stream *standard-output*))
  "Run TESTS; report and return true when all pass.  With DEBUG, failures enter the debugger."
  (let ((failed '()) (passed 0)
        (t0 (get-internal-real-time)))
    (restart-case
        (dolist (name tests)
          (handler-bind ((error (lambda (c)
                                  (unless debug
                                    (push (cons name c) failed)
                                    (format stream "~&  FAIL ~(~A~): ~A~%" name c)
                                    (invoke-restart 'skip-test)))))
            (cond ((run-test name)
                   (incf passed)
                   (format stream "~&  ok   ~(~A~)~%" name))
                  ((not (assoc name failed))      ; skipped from the debugger
                   (push (cons name :skipped) failed)))))
      (abort-tests ()
        :report "Stop running tests."
        nil))
    (format stream "~&~D passed, ~D failed (~,2F s)~%" passed (length failed)
            (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    (null failed)))
