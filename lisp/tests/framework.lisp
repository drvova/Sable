;;;; framework.lisp -- running the tests, on the condition system (no dependencies).
;;;;
;;;; A test is any function whose name starts with TEST- in SABLE.TESTS; its checks are ASSERT
;;;; forms.  Each runs inside RESTART-CASE with the restarts SKIP-TEST and RETRY-TEST.  RUN-TESTS
;;;; binds a handler (HANDLER-BIND, so the stack is still intact) that records a failure and
;;;; invokes SKIP-TEST; with :DEBUG T it declines, so the debugger comes up with SKIP-TEST,
;;;; RETRY-TEST and ABORT-TESTS on offer.

(defpackage #:sable.tests
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.forms #:sable.blake3
        #:sable.transcript)
  (:export #:run-tests #:all-tests #:signalled))

(in-package #:sable.tests)

(defun all-tests ()
  "The tests: the functions of this package named TEST-..., sorted."
  (sort (loop for symbol being the present-symbols of '#:sable.tests
              when (and (fboundp symbol) (eql 0 (search "TEST-" (symbol-name symbol))))
                collect symbol)
        #'string<))

(defun signalled (type thunk)
  "The condition of TYPE that calling THUNK signals, or NIL if THUNK returns without one."
  (block nil
    (handler-bind ((condition (lambda (c) (when (typep c type) (return c)))))
      (funcall thunk)
      nil)))

(defun run-test (name)
  "Run one test; returns T on success.  Restarts: SKIP-TEST, RETRY-TEST."
  (loop
    (restart-case (progn (funcall name) (return t))
      (skip-test ()
        :report (lambda (s) (format s "Skip the test ~S." name))
        (return nil))
      (retry-test ()
        :report (lambda (s) (format s "Run the test ~S again." name))))))

(defun run-tests (&key (tests (all-tests)) debug (stream *standard-output*))
  "Run TESTS; report, and return true when all pass.  With DEBUG, failures enter the debugger."
  (let ((failed '()) (passed 0)
        (start (get-internal-real-time)))
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
            (/ (- (get-internal-real-time) start) internal-time-units-per-second))
    (null failed)))
