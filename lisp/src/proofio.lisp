;;;; proofio.lisp -- the proof byte format (zk/proofio.py), so a verifier never deserialises code.
;;;;
;;;;   proof   = MAGIC | u32 count | message*
;;;;   message = u8 0 | u64 length | bytes                                  (a byte message)
;;;;           | u8 1 | u8 rank | u64 dim * rank | u32 value * prod(dims)  (a field array, values < p)
;;;;
;;;; All integers little-endian.  DECODE-PROOF checks every length against the remaining input and
;;;; every value against p, and signals MALFORMED-PROOF on any malformed proof.  It multiplies the
;;;; dimensions exactly, so a shape whose size overflows 64 bits is rejected as truncated.

(defpackage #:sable.proofio
  (:use #:cl #:sable.util #:sable.conditions #:sable.field #:sable.transcript)
  (:export #:*magic* #:+max-rank+ #:encode-proof #:decode-proof #:read-proof-file #:write-proof-file
           #:describe-proof))

(in-package #:sable.proofio)

(defparameter *magic* (to-octets "SABLEPF1") "The first bytes of every proof.")
(defconstant +max-rank+ 8)

;;; ------------------------------------------------------------------ encoding

(defun canonical-p (array)
  (loop for i below (array-total-size array)
        always (< (row-major-aref array i) +p+)))

(defun encode-message (message)
  "The bytes of MESSAGE, as a list of byte vectors."
  (etypecase message
    (byte-message
     (list #(0) (le-bytes (length message)) message))
    (field-array
     (let ((dimensions (array-dimensions message)))
       (check-parameter message (lambda (m) (<= (array-rank m) +max-rank+))
                        (format nil "an array of rank at most ~D" +max-rank+))
       (check-parameter message #'canonical-p "values below p")
       `(#(1) ,(vector (length dimensions))
         ,@(mapcar #'le-bytes dimensions)
         ,(message-octets message))))))

(defun encode-proof (messages)
  "MESSAGES (byte messages and field arrays) -> OCTETS."
  (join-octets (list* *magic*
                      (le-bytes (length messages) 4)
                      (loop for message in (coerce messages 'list)
                            append (encode-message message)))))

;;; ------------------------------------------------------------------ decoding

(defun decode-proof (octets &key (start 0) (end (length octets)))
  "Parse OCTETS[START, END) as a proof -> a vector of messages.  Signals MALFORMED-PROOF."
  (let ((octets (to-octets octets))
        (position start))
    (labels ((take (n)
               "Skip the next N bytes; return their offset."
               (when (> n (- end position))
                 (malformed :truncated :offset position :detail (format nil "~D bytes needed" n)))
               (shiftf position (+ position n)))
             (read-uint (size)
               (le-integer octets (take size) size))
             (read-bytes ()
               (let* ((length (read-uint 8))
                      (from (take length)))
                 (subseq octets from (+ from length))))
             (read-field-array (offset)
               (let ((rank (read-uint 1)))
                 (when (> rank +max-rank+)
                   (malformed :rank :offset offset :detail (format nil "rank ~D" rank)))
                 (let* ((dimensions (loop repeat rank collect (read-uint 8)))
                        (size (reduce #'* dimensions)))
                   (when (> (* 4 size) (- end position))
                     (malformed :truncated :offset offset :detail (format nil "~D values" size)))
                   (when (some (lambda (d) (>= d array-dimension-limit)) dimensions)
                     (malformed :dimension :offset offset :detail (format nil "~S" dimensions)))
                   (let ((from (take (* 4 size)))
                         (array (make-array dimensions :element-type '(unsigned-byte 32))))
                     (dotimes (i size array)
                       (let ((value (le-integer octets (+ from (* 4 i)) 4)))
                         (unless (< value +p+)
                           (malformed :value-range :offset (+ from (* 4 i))
                                                   :detail (format nil "value ~D" value)))
                         (setf (row-major-aref array i) value)))))))
             (read-message ()
               (let* ((offset position)
                      (tag (read-uint 1)))
                 (case tag
                   (0 (read-bytes))
                   (1 (read-field-array offset))
                   (t (malformed :unknown-tag :offset offset :detail (format nil "tag ~D" tag)))))))
      (unless (and (<= (length *magic*) (- end position))
                   (equalp *magic* (subseq octets position (+ position (length *magic*)))))
        (malformed :magic :offset position))
      (take (length *magic*))
      (let ((messages (loop repeat (read-uint 4) collect (read-message))))
        (unless (= position end)
          (malformed :trailing-bytes :offset position :detail (format nil "~D bytes" (- end position))))
        (coerce messages 'vector)))))

;;; ------------------------------------------------------------------ files

(defun read-file-octets (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((octets (make-octets (file-length in))))
      (subseq octets 0 (read-sequence octets in)))))

(defun read-proof-file (path)
  "The messages of the proof file PATH (e.g. the .proof written by BENCH_SAVE)."
  (decode-proof (read-file-octets path)))

(defun write-proof-file (path messages &key (if-exists :supersede))
  (with-open-file (out path :element-type '(unsigned-byte 8) :direction :output :if-exists if-exists)
    (write-sequence (encode-proof messages) out))
  path)

;;; ------------------------------------------------------------------ summary

(defun describe-proof (messages &optional (stream *standard-output*) (limit 25))
  "Print the number and total size of MESSAGES, and how many have each shape (the LIMIT most
common shapes)."
  (let ((shapes (make-hash-table :test 'equal)))
    (map nil (lambda (m) (incf (gethash (message-shape m) shapes 0))) messages)
    (flet ((total (type)
             (loop for m across (coerce messages 'vector)
                   when (typep m type) count t into n and sum (message-size m) into size
                   finally (return (values n size)))))
      (multiple-value-bind (nbytes byte-size) (total 'byte-message)
        (multiple-value-bind (nfield field-size) (total 'field-array)
          (format stream "~&~D messages: ~D byte messages (~:D bytes), ~D field arrays (~:D bytes)~%"
                  (length messages) nbytes byte-size nfield field-size))))
    (let ((rows (sort (loop for shape being the hash-keys of shapes using (hash-value count)
                            collect (cons shape count))
                      #'> :key #'cdr)))
      (format stream "~&~10@A  shape~%" "count")
      (loop for (shape . count) in rows
            repeat limit
            do (format stream "~10D  ~:[(~{~D~^ x ~})~;~{~(~A~) ~D~}~]~%"
                       count (eq (first shape) :bytes) shape))
      (when (> (length rows) limit)
        (format stream "~10@A  ... ~D more shapes~%" "" (- (length rows) limit))))))
