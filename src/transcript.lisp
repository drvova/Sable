;;;; transcript.lisp -- the Fiat-Shamir transcript (zk/transcript.py, host side).
;;;;
;;;; The hash is the sequential BLAKE3 mode of zk/cuda/b3s.h.  Absorbing DATA hashes its length as
;;;; a u64 LE, then DATA.  A squeeze of n bytes reads n output bytes, then absorbs "squeeze" and the
;;;; first min(32, n) of them.  Field challenges are 31-bit words below p (rejection sampling).
;;;;
;;;; Messages are byte vectors (BYTE-MESSAGE) or arrays of field values (FIELD-ARRAY), the latter
;;;; absorbed as their values, u32 LE, in row-major order.  A prover transcript records the
;;;; messages it sends; a verifier transcript is made from a proof's messages and replays them, so
;;;; both absorb the same bytes.  A message of the wrong kind or shape, or a missing one, rejects
;;;; the proof.

(defpackage #:exacta.transcript
  (:use #:cl #:exacta.util #:exacta.conditions #:exacta.field #:exacta.blake3)
  (:export #:byte-message #:field-array #:to-field-array #:message-octets #:message-size #:message-shape
           #:transcript #:make-transcript #:transcript-degree #:verifier-p #:transcript-messages
           #:transcript-position #:transcript-sent-bytes #:transcript-state #:transcript-finished-p
           #:absorb #:send #:send-bytes #:record #:record-bytes
           #:squeeze #:challenge-base #:challenge #:challenges #:challenge-bits
           #:check-grind #:grind #:verify-grind #:label-name #:split-string))

(in-package #:exacta.transcript)

(defparameter *domain* "LLM-SUMCHECK-v6-O|"
  "The prefix of every transcript label.")

;;; ------------------------------------------------------------------ messages

(deftype byte-message () '(array (unsigned-byte 8)))
(deftype field-array () '(array (unsigned-byte 32)))

(defun to-field-array (x)
  "A FIELD-ARRAY of the canonical values of X: an integer (rank 0), a sequence or an array."
  (etypecase x
    (integer (make-array '() :element-type '(unsigned-byte 32) :initial-element (modp x)))
    (list (to-field-array (coerce x 'vector)))
    (array (let ((a (make-array (array-dimensions x) :element-type '(unsigned-byte 32))))
             (dotimes (i (array-total-size x) a)
               (setf (row-major-aref a i) (modp (row-major-aref x i))))))))

(defun message-octets (message)
  "The bytes of MESSAGE as absorbed: a byte message itself, a field array as u32 LE values."
  (etypecase message
    (byte-message (to-octets message))
    (field-array (let ((out (make-octets (* 4 (array-total-size message)))))
                   (dotimes (i (array-total-size message) out)
                     (let ((value (row-major-aref message i)))
                       (dotimes (b 4)
                         (setf (aref out (+ (* 4 i) b)) (ldb (byte 8 (* 8 b)) value)))))))))

(defun message-size (message)
  "The number of bytes of MESSAGE in a proof's payload."
  (etypecase message
    (byte-message (length message))
    (field-array (* 4 (array-total-size message)))))

(defun message-shape (message)
  "The dimensions of a field array; (:BYTES length) for a byte message."
  (etypecase message
    (byte-message (list :bytes (length message)))
    (field-array (array-dimensions message))))

;;; ------------------------------------------------------------------ the transcript

(defclass transcript ()
  ((degree :initarg :degree :reader transcript-degree
           :documentation "Challenges are elements of F_p^DEGREE.")
   (state :initarg :state :reader transcript-state :documentation "The B3S hash state.")
   (messages :initarg :messages :reader transcript-messages
             :documentation "Prover: the messages sent so far.  Verifier: the proof's messages.")
   (position :initform 0 :accessor transcript-position)
   (verifier-p :initarg :verifier-p :reader verifier-p)
   (sent-bytes :initform 0 :accessor transcript-sent-bytes)))

(defun make-transcript (label degree &key (messages nil verifier-p))
  "A transcript for LABEL (a string or octets) with challenges in F_p^DEGREE.  With MESSAGES (a
sequence of proof messages), a verifier transcript that replays them."
  (make-instance 'transcript
                 :degree (check-degree degree)
                 :state (make-b3s (concatenate 'octets (to-octets *domain*) (to-octets label)))
                 :verifier-p verifier-p
                 :messages (if verifier-p
                               (coerce messages 'vector)
                               (make-array 16 :adjustable t :fill-pointer 0))))

(defmethod print-object ((tr transcript) stream)
  (print-unreadable-object (tr stream :type t :identity t)
    (format stream "~:[prover~;verifier~] F_p^~D, ~D/~D messages" (verifier-p tr)
            (transcript-degree tr) (transcript-position tr) (length (transcript-messages tr)))))

(defun transcript-finished-p (tr)
  "True when a verifier has taken every message of the proof."
  (= (transcript-position tr) (length (transcript-messages tr))))

(defun absorb (tr data)
  "Absorb DATA (octets or a string): its length as u64 LE, then the bytes."
  (let ((data (to-octets data)))
    (b3s-update (transcript-state tr) (le-bytes (length data)))
    (b3s-update (transcript-state tr) data)
    tr))

(defun take-message (tr type shape)
  "Verifier: the next message, which must be of TYPE and, unless SHAPE is :ANY, of SHAPE."
  (with-accessors ((position transcript-position) (messages transcript-messages)) tr
    (when (>= position (length messages))
      (error 'transcript-exhausted :index position))
    (let ((message (aref messages position)))
      (unless (and (typep message type)
                   (or (eq shape :any) (equal (message-shape message) shape)))
        (error 'message-shape-error :index position
                                    :expected (if (eq shape :any) type shape)
                                    :actual (message-shape message)))
      (incf position)
      message)))

(defun exchange (tr value type shape absorb)
  "Prover: record VALUE as a message of TYPE.  Verifier: take the next message, of TYPE and SHAPE.
Either way, absorb the message if ABSORB, and return it."
  (let ((message (if (verifier-p tr)
                     (take-message tr type shape)
                     (let ((message (ecase type
                                      (field-array (to-field-array value))
                                      (byte-message (to-octets value)))))
                       (vector-push-extend message (transcript-messages tr))
                       message))))
    (incf (transcript-sent-bytes tr) (message-size message))
    (when absorb (absorb tr (message-octets message)))
    message))

(defun bytes-shape (length)
  (if length (list :bytes length) :any))

(defun send (tr &optional value (shape :any))
  "Send and absorb a field array.  A verifier passes no VALUE, and the SHAPE it expects."
  (exchange tr value 'field-array shape t))

(defun send-bytes (tr &optional value length)
  "Send and absorb a byte message.  A verifier passes no VALUE, and the LENGTH it expects."
  (exchange tr value 'byte-message (bytes-shape length) t))

(defun record (tr &optional value (shape :any))
  "A field array that is part of the proof but not absorbed (the final query openings, after the
last challenge)."
  (exchange tr value 'field-array shape nil))

(defun record-bytes (tr &optional value length)
  "A byte message that is part of the proof but not absorbed."
  (exchange tr value 'byte-message (bytes-shape length) nil))

;;; ------------------------------------------------------------------ challenges

(defun squeeze (tr n)
  "N bytes of output; then absorb \"squeeze\" and the first min(32, N) of them."
  (let ((out (b3s-digest (transcript-state tr) n)))
    (b3s-update (transcript-state tr) (to-octets "squeeze"))
    (b3s-update (transcript-state tr) out :end (min 32 n))
    out))

(defun squeeze-field-elements (tr n)
  "The 31-bit words of N squeezed bytes that are below p."
  (loop with buffer = (squeeze tr n)
        for i from 0 below n by 4
        for word = (ldb (byte 31 0) (le-integer buffer i 4))
        when (< word +p+) collect word))

(defun challenge-base (tr k)
  "K base-field challenges, as a list."
  (let ((accepted '()))
    (loop while (< (length accepted) k)
          do (setf accepted (nconc accepted
                                   (squeeze-field-elements tr (* 4 (+ (- k (length accepted)) 4))))))
    (subseq accepted 0 k)))

(defun challenge (tr &optional (degree (transcript-degree tr)))
  "One challenge in F_p^DEGREE."
  (coerce-ext (challenge-base tr degree)))

(defun challenges (tr k &optional (degree (transcript-degree tr)))
  "K challenges in F_p^DEGREE, from DEGREE * K consecutive base challenges."
  (loop for tail on (challenge-base tr (* degree k)) by (lambda (l) (nthcdr degree l))
        collect (coerce-ext (subseq tail 0 degree))))

(defun challenge-bits (tr nbits k)
  "K integers of NBITS bits: the low bits of squeezed u64 LE words."
  (loop with buffer = (squeeze tr (* 8 k))
        for i below k
        collect (ldb (byte nbits 0) (le-integer buffer (* 8 i) 8))))

;;; ------------------------------------------------------------------ proof of work

(defun check-grind (base-digest nonce bits)
  "True when BLAKE3(BASE-DIGEST || NONCE as u64 LE), read as a u64 LE, is below 2^(64 - BITS)."
  (< (le-integer (blake3-hash (concatenate 'octets base-digest (le-bytes nonce)) 8))
     (expt 2 (- 64 bits))))

(defun grind (tr bits)
  "Prover, on the CPU (so for small BITS): send the smallest nonce that passes CHECK-GRIND."
  (let* ((base (b3s-digest (transcript-state tr) 32))
         (nonce (loop for n from 0 when (check-grind base n bits) return n)))
    (send-bytes tr (le-bytes nonce))
    nonce))

(defun verify-grind (tr bits)
  "Verifier: take the 8-byte nonce and check it; returns the nonce."
  (let* ((base (b3s-digest (transcript-state tr) 32))
         (nonce (le-integer (send-bytes tr nil 8))))
    (unless (check-grind base nonce bits)
      (reject :proof-of-work "nonce ~D does not meet ~D bits" nonce bits))
    nonce))

;;; ------------------------------------------------------------------ labels

(defun split-string (string separator)
  (loop for start = 0 then (1+ end)
        for end = (position separator string :start start)
        collect (subseq string start end)
        while end))

(defun strip-layer-prefix (segment)
  "SEGMENT without a leading layer index: \"12.mo\" -> \"mo\"."
  (let ((dot (position-if-not (lambda (c) (char<= #\0 c #\9)) segment)))
    (if (and dot (plusp dot) (char= (char segment dot) #\.))
        (subseq segment (1+ dot))
        segment)))

(defun label-name (name)
  "A domain or group name as absorbed in the transcript, without its layer prefixes:
\"12.mo\" -> \"mo\", \"3.oproj|3.rqo\" -> \"oproj|rqo\".  The layer index is absorbed once per
layer instead, so a layer's labels do not depend on the layer."
  (format nil "~{~A~^|~}" (mapcar #'strip-layer-prefix (split-string name #\|))))
