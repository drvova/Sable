;;;; conditions.lisp -- the condition hierarchy of the Sable port.
;;;;
;;;;   sable-condition
;;;;     sable-error                      (also ERROR)
;;;;       malformed-proof                the proof bytes do not parse (zk/proofio.py's ValueError)
;;;;       verification-failure           a check of the verifier failed: the proof is rejected
;;;;         message-shape-error          a proof message has the wrong kind or shape
;;;;         transcript-exhausted         the verifier asked for more messages than the proof holds
;;;;       invalid-parameter              a value outside a function's domain
;;;;       field-division-by-zero         an inverse of zero (also CL:DIVISION-BY-ZERO)
;;;;
;;;; A verifier must never be talked into accepting a proof, so MALFORMED-PROOF and
;;;; VERIFICATION-FAILURE come with no restart: the only way out is a non-local exit.  Restarts
;;;; appear only where continuing is sound: USE-VALUE for INVALID-PARAMETER (CHECK-PARAMETER), and
;;;; USE-VALUE and RETURN-ZERO for an inverse of zero.

(defpackage #:sable.conditions
  (:use #:cl)
  (:export #:sable-condition #:sable-error
           #:malformed-proof #:malformed-proof-reason #:malformed-proof-offset #:malformed-proof-detail
           #:verification-failure #:verification-failure-check #:verification-failure-detail
           #:message-shape-error #:message-index #:message-shape-expected #:message-shape-actual
           #:transcript-exhausted
           #:invalid-parameter #:invalid-parameter-name #:invalid-parameter-value #:invalid-parameter-expected
           #:check-parameter
           #:field-division-by-zero #:return-zero
           #:malformed #:reject))

(in-package #:sable.conditions)

(define-condition sable-condition (condition) ()
  (:documentation "The root of every condition signalled by Sable."))

(define-condition sable-error (sable-condition error) ())

;;; ------------------------------------------------------------------ proof bytes

(defparameter *malformed-proof-reasons*
  '(:magic "not a Sable proof"
    :truncated "truncated"
    :trailing-bytes "trailing bytes"
    :unknown-tag "unknown message type"
    :value-range "field value not below p"
    :rank "array rank"
    :dimension "array dimension too large"))

(define-condition malformed-proof (sable-error)
  ((reason :initarg :reason :reader malformed-proof-reason
           :documentation "One of the keys of *MALFORMED-PROOF-REASONS*.")
   (offset :initarg :offset :initform nil :reader malformed-proof-offset)
   (detail :initarg :detail :initform nil :reader malformed-proof-detail))
  (:report (lambda (c stream)
             (format stream "Malformed proof: ~A~@[ at byte ~D~]~@[ (~A)~]"
                     (getf *malformed-proof-reasons* (malformed-proof-reason c))
                     (malformed-proof-offset c)
                     (malformed-proof-detail c)))))

(defun malformed (reason &key offset detail)
  "Signal a MALFORMED-PROOF."
  (error 'malformed-proof :reason reason :offset offset :detail detail))

;;; ------------------------------------------------------------------ verification

(define-condition verification-failure (sable-error)
  ((check :initarg :check :reader verification-failure-check
          :documentation "A keyword naming the failed check, e.g. :SUMCHECK-FINAL.")
   (detail :initarg :detail :initform nil :reader verification-failure-detail))
  (:report (lambda (c stream)
             (format stream "Proof rejected: check ~(~A~) failed~@[: ~A~]"
                     (verification-failure-check c) (verification-failure-detail c)))))

(defun reject (check &optional format-control &rest format-arguments)
  "Signal a VERIFICATION-FAILURE of CHECK."
  (error 'verification-failure
         :check check
         :detail (and format-control (apply #'format nil format-control format-arguments))))

(define-condition message-shape-error (verification-failure)
  ((index :initarg :index :initform nil :reader message-index)
   (expected :initarg :expected :reader message-shape-expected)
   (actual :initarg :actual :reader message-shape-actual))
  (:default-initargs :check :message-shape)
  (:report (lambda (c stream)
             (format stream "Proof rejected: message~@[ ~D~] has shape ~S, expected ~S"
                     (message-index c) (message-shape-actual c) (message-shape-expected c)))))

(define-condition transcript-exhausted (verification-failure)
  ((index :initarg :index :initform nil :reader message-index))
  (:default-initargs :check :transcript-exhausted)
  (:report (lambda (c stream)
             (format stream "Proof rejected: the verifier needs message ~D, but the proof has no more"
                     (message-index c)))))

;;; ------------------------------------------------------------------ parameters

(define-condition invalid-parameter (sable-error)
  ((name :initarg :name :reader invalid-parameter-name)
   (value :initarg :value :reader invalid-parameter-value)
   (expected :initarg :expected :reader invalid-parameter-expected))
  (:report (lambda (c stream)
             (format stream "Invalid ~(~A~): ~S (expected ~A)"
                     (invalid-parameter-name c) (invalid-parameter-value c)
                     (invalid-parameter-expected c)))))

(defun signal-invalid-parameter (name value expected)
  "Signal INVALID-PARAMETER; return the value given to its USE-VALUE restart."
  (restart-case (error 'invalid-parameter :name name :value value :expected expected)
    (use-value (new)
      :report "Supply another value."
      :interactive (lambda ()
                     (format *query-io* "~&New ~(~A~): " name)
                     (finish-output *query-io*)
                     (list (eval (read *query-io*))))
      new)))

(defmacro check-parameter (place test expected)
  "Like CHECK-TYPE: unless (FUNCALL TEST PLACE), signal INVALID-PARAMETER.  Its USE-VALUE restart
stores a new value in PLACE, which is tested again."
  `(loop until (funcall ,test ,place)
         do (setf ,place (signal-invalid-parameter ',place ,place ,expected))))

;;; ------------------------------------------------------------------ field arithmetic

(define-condition field-division-by-zero (sable-error division-by-zero) ()
  (:report "Inverse of zero in the KoalaBear field")
  (:documentation "Signalled by FINV and EXT-INV, with the restarts USE-VALUE and RETURN-ZERO."))

(defun return-zero (&optional condition)
  "Invoke the RETURN-ZERO restart, if there is one: the inverse of zero is taken to be zero, as in
the Python and C++ code (0^(p-2) = 0)."
  (let ((restart (find-restart 'return-zero condition)))
    (when restart (invoke-restart restart))))
