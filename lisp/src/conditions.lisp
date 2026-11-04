;;;; conditions.lisp -- the condition hierarchy of the Sable port.
;;;;
;;;; Every condition Sable signals inherits from SABLE-CONDITION, so a caller can catch the whole
;;;; family with one HANDLER-CASE clause, or pick out one kind:
;;;;
;;;;   sable-condition
;;;;     sable-error                      (also ERROR)
;;;;       malformed-proof                the proof bytes do not parse (zk/proofio.py's ValueError)
;;;;       verification-failure           a check of the verifier failed: the proof is rejected
;;;;         message-shape-error          a proof message has the wrong kind or shape
;;;;         transcript-exhausted         the verifier asked for more messages than the proof holds
;;;;       invalid-parameter              a caller passed a value outside a function's domain
;;;;       field-division-by-zero         an inverse of zero (also CL:DIVISION-BY-ZERO)
;;;;
;;;; Restarts.  A verifier must never be talked into accepting a proof, so MALFORMED-PROOF and
;;;; VERIFICATION-FAILURE are signalled with ERROR and no restart that continues verification: the
;;;; only way out is a non-local exit (HANDLER-CASE, ABORT).  Restarts appear only where continuing
;;;; is sound:
;;;;
;;;;   FINV, EXT-INV        USE-VALUE (supply the inverse yourself) and RETURN-ZERO (the
;;;;                        Python/C++ convention 0^(p-2) = 0)
;;;;   INVALID-PARAMETER    USE-VALUE (supply a value in the domain), where the signalling
;;;;                        function can carry on with a replacement
;;;;   the test runner      SKIP-TEST, RETRY-TEST (tests/framework.lisp)

(defpackage #:sable.conditions
  (:use #:cl)
  (:export #:sable-condition #:sable-error
           #:malformed-proof #:malformed-proof-reason #:malformed-proof-offset #:malformed-proof-detail
           #:verification-failure #:verification-failure-check #:verification-failure-detail
           #:message-shape-error #:message-shape-expected #:message-shape-actual #:message-shape-index
           #:transcript-exhausted
           #:invalid-parameter #:invalid-parameter-name #:invalid-parameter-value #:invalid-parameter-expected
           #:field-division-by-zero
           #:return-zero
           #:malformed #:reject #:require-parameter))

(in-package #:sable.conditions)

(define-condition sable-condition (condition) ()
  (:documentation "The root of every condition signalled by Sable."))

(define-condition sable-error (sable-condition error) ()
  (:documentation "An error signalled by Sable."))

;;; ------------------------------------------------------------------ proof bytes

(define-condition malformed-proof (sable-error)
  ((reason :initarg :reason :reader malformed-proof-reason
           :documentation "A keyword: :MAGIC :TRUNCATED :TRAILING-BYTES :UNKNOWN-TAG :VALUE-RANGE :RANK
:DIMENSION.")
   (offset :initarg :offset :initform nil :reader malformed-proof-offset
           :documentation "The byte offset at which the problem was found, if known.")
   (detail :initarg :detail :initform nil :reader malformed-proof-detail))
  (:report (lambda (c stream)
             (format stream "Malformed proof: ~A~@[ at byte ~D~]~@[ (~A)~]"
                     (case (malformed-proof-reason c)
                       (:magic "not a Sable proof")
                       (:truncated "truncated")
                       (:trailing-bytes "trailing bytes")
                       (:unknown-tag "unknown message type")
                       (:value-range "field value not below p")
                       (:rank "array rank")
                       (:dimension "array dimension too large")
                       (t (malformed-proof-reason c)))
                     (malformed-proof-offset c)
                     (malformed-proof-detail c))))
  (:documentation "The input is not a well-formed proof in the byte format of zk/proofio.py."))

(defun malformed (reason &key offset detail)
  "Signal a MALFORMED-PROOF.  Never returns."
  (error 'malformed-proof :reason reason :offset offset :detail detail))

;;; ------------------------------------------------------------------ verification

(define-condition verification-failure (sable-error)
  ((check :initarg :check :reader verification-failure-check
          :documentation "A keyword naming the check that failed, e.g. :SUMCHECK-FINAL.")
   (detail :initarg :detail :initform nil :reader verification-failure-detail))
  (:report (lambda (c stream)
             (format stream "Proof rejected: check ~(~A~) failed~@[: ~A~]"
                     (verification-failure-check c) (verification-failure-detail c))))
  (:documentation "A verifier check failed.  The proof must be rejected; there is no restart."))

(defun reject (check &optional detail &rest args)
  "Signal a VERIFICATION-FAILURE for CHECK.  DETAIL is a format control applied to ARGS."
  (error 'verification-failure :check check
                               :detail (and detail (apply #'format nil detail args))))

(define-condition message-shape-error (verification-failure)
  ((index :initarg :index :initform nil :reader message-shape-index)
   (expected :initarg :expected :reader message-shape-expected)
   (actual :initarg :actual :reader message-shape-actual))
  (:default-initargs :check :message-shape)
  (:report (lambda (c stream)
             (format stream "Proof rejected: message~@[ ~D~] has shape ~S, expected ~S"
                     (message-shape-index c) (message-shape-actual c) (message-shape-expected c)))))

(define-condition transcript-exhausted (verification-failure)
  ((index :initarg :index :initform nil :reader message-shape-index))
  (:default-initargs :check :transcript-exhausted)
  (:report (lambda (c stream)
             (format stream "Proof rejected: the verifier needs message ~D, but the proof has no more"
                     (message-shape-index c)))))

;;; ------------------------------------------------------------------ programming errors

(define-condition invalid-parameter (sable-error)
  ((name :initarg :name :reader invalid-parameter-name)
   (value :initarg :value :reader invalid-parameter-value)
   (expected :initarg :expected :reader invalid-parameter-expected))
  (:report (lambda (c stream)
             (format stream "Invalid ~(~A~): ~S (expected ~A)"
                     (invalid-parameter-name c) (invalid-parameter-value c)
                     (invalid-parameter-expected c)))))

(defun require-parameter (value test name expected)
  "Return VALUE if (FUNCALL TEST VALUE), else signal INVALID-PARAMETER.  Its USE-VALUE restart
supplies a replacement, which is tested again."
  (loop
    (when (funcall test value) (return value))
    (restart-case (error 'invalid-parameter :name name :value value :expected expected)
      (use-value (new)
        :report "Supply another value."
        :interactive (lambda () (format *query-io* "~&New ~(~A~): " name) (force-output *query-io*)
                       (list (read *query-io*)))
        (setf value new)))))

(define-condition field-division-by-zero (sable-error division-by-zero) ()
  (:report (lambda (c stream)
             (declare (ignore c))
             (format stream "Inverse of zero in the KoalaBear field")))
  (:documentation "FINV or EXT-INV of zero.  Restarts: USE-VALUE, RETURN-ZERO."))

(defun return-zero (&optional condition)
  "Invoke the RETURN-ZERO restart (the convention of the Python and C++ code: inv(0) = 0)."
  (let ((r (find-restart 'return-zero condition)))
    (when r (invoke-restart r))))
