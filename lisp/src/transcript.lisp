;;;; transcript.lisp -- the Fiat-Shamir transcript (zk/transcript.py, host side).
;;;;
;;;; The hash is the sequential BLAKE3 mode of zk/cuda/b3s.h.  absorb(data) hashes the length of
;;;; DATA as a u64 LE, then DATA; a field array is absorbed as its canonical values, u32 LE, in
;;;; row-major order.  A squeeze of n bytes reads n output bytes, then absorbs "squeeze" and the
;;;; first min(32, n) of them.  Field challenges are 31-bit words below p (rejection sampling).
;;;;
;;;; A prover transcript records every message it sends; a verifier transcript is made from a
;;;; proof's messages (PROOFIO:READ-PROOF) and replays them in order, absorbing each exactly as the
;;;; prover did.  A message of the wrong kind or shape, or a missing one, rejects the proof.

(defpackage #:sable.transcript
  (:use #:cl #:sable.conditions #:sable.field #:sable.blake3)
  (:export #:transcript #:make-transcript #:transcript-d #:transcript-verifier-p
           #:transcript-messages #:transcript-position #:transcript-sent-bytes #:transcript-state
           #:absorb #:absorb-field #:squeeze #:send #:send-bytes #:record #:record-bytes
           #:challenge-base #:challenge #:challenges #:challenge-bits
           #:grind #:verify-grind #:check-grind #:transcript-finished-p #:label-name
           #:field-array-p #:byte-message-p #:field-array-octets #:+domain+))

(in-package #:sable.transcript)

(defparameter +domain+ "LLM-SUMCHECK-v6-O|"
  "The domain-separation prefix of every transcript label.")

(defun ascii (s)
  (if (stringp s) (sb-ext:string-to-octets s :external-format :utf-8) (coerce s 'octets)))

(defclass transcript ()
  ((d :initarg :d :reader transcript-d :documentation "The extension degree of challenges.")
   (state :initarg :state :accessor transcript-state :documentation "The B3S hash state.")
   (messages :initarg :messages :reader transcript-messages
             :documentation "Prover: messages sent so far.  Verifier: the proof's messages.")
   (position :initform 0 :accessor transcript-position)
   (verifier-p :initarg :verifier-p :reader transcript-verifier-p)
   (sent-bytes :initform 0 :accessor transcript-sent-bytes)))

(defun make-transcript (label d &key (messages nil verifier-p))
  "A transcript for LABEL (string or octets) whose challenges lie in F_p^D.  With MESSAGES (a
sequence of proof messages), a verifier transcript that replays them."
  (make-instance 'transcript
                 :d (check-degree d)
                 :state (b3s-init (octets-concat (ascii +domain+) (ascii label)))
                 :verifier-p verifier-p
                 :messages (if verifier-p
                               (coerce messages 'simple-vector)
                               (make-array 16 :adjustable t :fill-pointer 0))))

(defmethod print-object ((tr transcript) stream)
  (print-unreadable-object (tr stream :type t :identity t)
    (format stream "~:[prover~;verifier~] D=~D ~D/~D messages" (transcript-verifier-p tr)
            (transcript-d tr) (transcript-position tr) (length (transcript-messages tr)))))

;;; ------------------------------------------------------------------ messages

(defun byte-message-p (m)
  (typep m '(array (unsigned-byte 8))))

(defun field-array-p (m)
  (typep m '(array (unsigned-byte 32))))

(defun field-array-octets (a)
  "The canonical values of the field array A as u32 LE, row-major."
  (let* ((n (array-total-size a))
         (out (make-octets (* 4 n))))
    (dotimes (i n out)
      (let ((v (row-major-aref a i)))
        (setf (aref out (* 4 i)) (ldb (byte 8 0) v)
              (aref out (+ 1 (* 4 i))) (ldb (byte 8 8) v)
              (aref out (+ 2 (* 4 i))) (ldb (byte 8 16) v)
              (aref out (+ 3 (* 4 i))) (ldb (byte 8 24) v))))))

(defun canonical-field-array (x)
  "A field array of canonical values from X: an array of integers, a sequence, or an integer."
  (etypecase x
    (integer (let ((a (make-array '() :element-type '(unsigned-byte 32))))
               (setf (aref a) (fp-canon x))
               a))
    (array (let ((a (make-array (array-dimensions x) :element-type '(unsigned-byte 32))))
             (dotimes (i (array-total-size x) a)
               (setf (row-major-aref a i) (fp-canon (row-major-aref x i))))))
    (list (canonical-field-array (coerce x 'vector)))))

(defun next-message (tr)
  (let ((pos (transcript-position tr))
        (msgs (transcript-messages tr)))
    (when (>= pos (length msgs))
      (error 'transcript-exhausted :index pos))
    (setf (transcript-position tr) (1+ pos))
    (aref msgs pos)))

(defun take-field (tr shape)
  "Verifier: the next message, which must be a field array of dimensions SHAPE (or any shape if
SHAPE is :ANY)."
  (let* ((pos (transcript-position tr))
         (m (next-message tr)))
    (unless (and (field-array-p m)
                 (or (eq shape :any) (equal (array-dimensions m) shape)))
      (error 'message-shape-error :index pos :expected shape
                                  :actual (if (field-array-p m) (array-dimensions m) :bytes)))
    m))

(defun take-bytes (tr &optional length)
  (let* ((pos (transcript-position tr))
         (m (next-message tr)))
    (unless (and (byte-message-p m) (or (null length) (= (length m) length)))
      (error 'message-shape-error :index pos
                                  :expected (if length (list :bytes length) :bytes)
                                  :actual (if (byte-message-p m) (list :bytes (length m))
                                              (array-dimensions m))))
    m))

(defun transcript-finished-p (tr)
  "True when a verifier has consumed every message of the proof."
  (= (transcript-position tr) (length (transcript-messages tr))))

;;; ------------------------------------------------------------------ absorbing

(defun absorb (tr data)
  "Absorb DATA (octets or a string): its length as u64 LE, then the bytes."
  (let ((data (ascii data))
        (st (transcript-state tr)))
    (b3s-update st (u64-le (length data)))
    (b3s-update st data)
    tr))

(defun absorb-field (tr a)
  "Absorb the field array A (canonical values as u32 LE)."
  (absorb tr (field-array-octets (canonical-field-array a))))

(defun send (tr &optional arr (shape :any))
  "Prover: record and absorb the field array ARR.  Verifier: take the next message, which must be a
field array of dimensions SHAPE, absorb it and return it."
  (let ((a (if (transcript-verifier-p tr)
               (take-field tr shape)
               (let ((a (canonical-field-array arr)))
                 (vector-push-extend a (transcript-messages tr))
                 a))))
    (incf (transcript-sent-bytes tr) (* 4 (array-total-size a)))
    (absorb tr (field-array-octets a))
    a))

(defun send-bytes (tr &optional b length)
  "SEND for a byte message (LENGTH, if given, is checked by a verifier)."
  (let ((b (if (transcript-verifier-p tr)
               (take-bytes tr length)
               (let ((b (coerce b 'octets)))
                 (vector-push-extend b (transcript-messages tr))
                 b))))
    (incf (transcript-sent-bytes tr) (length b))
    (absorb tr b)
    b))

(defun record (tr &optional arr (shape :any))
  "A field-array message that is part of the proof but not absorbed (final query openings)."
  (let ((a (if (transcript-verifier-p tr)
               (take-field tr shape)
               (let ((a (canonical-field-array arr)))
                 (vector-push-extend a (transcript-messages tr))
                 a))))
    (incf (transcript-sent-bytes tr) (* 4 (array-total-size a)))
    a))

(defun record-bytes (tr &optional b length)
  "A byte message recorded without absorbing (see RECORD)."
  (let ((b (if (transcript-verifier-p tr)
               (take-bytes tr length)
               (let ((b (coerce b 'octets)))
                 (vector-push-extend b (transcript-messages tr))
                 b))))
    (incf (transcript-sent-bytes tr) (length b))
    b))

;;; ------------------------------------------------------------------ challenges

(defun squeeze (tr n)
  (let* ((st (transcript-state tr))
         (out (b3s-digest st n)))
    (b3s-update st (ascii "squeeze"))
    (b3s-update st out :end (min 32 n))
    out))

(defun challenge-base (tr k)
  "K base-field challenges, as a list."
  (let ((vals '()) (got 0))
    (loop while (< got k)
          do (let ((buf (squeeze tr (* 4 (+ (- k got) 4)))))
               (loop for i from 0 below (length buf) by 4
                     for w = (logand #x7FFFFFFF
                                     (logior (aref buf i) (ash (aref buf (+ i 1)) 8)
                                             (ash (aref buf (+ i 2)) 16) (ash (aref buf (+ i 3)) 24)))
                     when (and (< w +p+) (< got k))
                       do (push w vals) (incf got))))
    (nreverse vals)))

(defun challenge (tr &optional (d (transcript-d tr)))
  "One extension-field challenge, an EXT of degree D."
  (ext-coerce (challenge-base tr d)))

(defun challenges (tr k &optional (d (transcript-d tr)))
  "K extension-field challenges (a vector of EXTs), from D*K consecutive base challenges."
  (let ((flat (coerce (challenge-base tr (* d k)) 'simple-vector)))
    (coerce (loop for j below k collect (ext-coerce (subseq flat (* j d) (* (1+ j) d)))) 'simple-vector)))

(defun challenge-bits (tr nbits k)
  "K integers of NBITS bits each (the low bits of u64 LE words)."
  (let ((buf (squeeze tr (* 8 k))))
    (loop for j below k
          collect (let ((w 0))
                    (dotimes (i 8) (setf w (logior w (ash (aref buf (+ (* 8 j) i)) (* 8 i)))))
                    (ldb (byte nbits 0) w)))))

;;; ------------------------------------------------------------------ proof of work

(defun check-grind (base-digest nonce bits)
  "True when BLAKE3(BASE-DIGEST || NONCE as u64 LE), read as a u64 LE, is below 2^(64-BITS)."
  (let ((d (blake3-hash (octets-concat base-digest (u64-le nonce)) 8))
        (w 0))
    (dotimes (i 8) (setf w (logior w (ash (aref d i) (* 8 i)))))
    (< w (ash 1 (- 64 bits)))))

(defun grind (tr bits)
  "Prover (on the CPU, so only for small BITS): send the smallest nonce passing CHECK-GRIND."
  (let ((base (b3s-digest (transcript-state tr) 32)))
    (let ((nonce (loop for n from 0 when (check-grind base n bits) return n)))
      (send-bytes tr (u64-le nonce))
      nonce)))

(defun verify-grind (tr bits)
  "Verifier: take the 8-byte nonce, check it.  Returns the nonce; rejects the proof on failure."
  (let* ((base (b3s-digest (transcript-state tr) 32))
         (b (send-bytes tr nil 8))
         (nonce (loop for i below 8 sum (ash (aref b i) (* 8 i)))))
    (unless (check-grind base nonce bits)
      (reject :proof-of-work "nonce ~D does not meet ~D bits" nonce bits))
    nonce))

;;; ------------------------------------------------------------------ labels

(defun layer-prefix-end (s start end)
  "If S[START, END) begins with digits followed by a dot, the index after the dot; else START."
  (let ((i start))
    (loop while (and (< i end) (char<= #\0 (char s i) #\9)) do (incf i))
    (if (and (> i start) (< i end) (char= (char s i) #\.)) (1+ i) start)))

(defun label-name (name)
  "A domain or group name as absorbed in the transcript, without its layer prefixes:
\"12.mo\" -> \"mo\", \"3.oproj|3.rqo\" -> \"oproj|rqo\" (the layer index is absorbed once per layer)."
  (with-output-to-string (out)
    (let ((start 0) (n (length name)))
      (loop
        (let ((bar (or (position #\| name :start start) n)))
          (write-string name out :start (layer-prefix-end name start bar) :end bar)
          (when (>= bar n) (return))
          (write-char #\| out)
          (setf start (1+ bar)))))))
