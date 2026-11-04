;;;; proofio.lisp -- the proof byte format (zk/proofio.py), so a verifier never deserialises code.
;;;;
;;;;   proof   = MAGIC | u32 count | message*
;;;;   message = u8 0 | u64 length | bytes                                  (a byte message)
;;;;           | u8 1 | u8 ndim | u64 dim * ndim | u32 value * prod(dims)  (a field array, values < p)
;;;;
;;;; All integers little-endian.  A byte message is an (ARRAY (UNSIGNED-BYTE 8)); a field array is an
;;;; (ARRAY (UNSIGNED-BYTE 32)) of rank 0 to 8.  READ-PROOF checks every length against the remaining
;;;; input and every value against p, and signals MALFORMED-PROOF on any malformed proof.  The
;;;; product of the dimensions is computed exactly, so a shape whose product overflows 64 bits is
;;;; rejected as truncated.

(defpackage #:sable.proofio
  (:use #:cl #:sable.conditions #:sable.field #:sable.blake3)
  (:import-from #:sable.transcript #:field-array-p #:byte-message-p #:field-array-octets)
  (:export #:+magic+ #:+max-ndim+ #:write-proof #:read-proof #:read-proof-file #:write-proof-file
           #:describe-proof))

(in-package #:sable.proofio)

(defparameter +magic+ (sb-ext:string-to-octets "SABLEPF1" :external-format :latin-1))
(defconstant +max-ndim+ 8)

(defun write-proof (messages)
  "MESSAGES (byte messages and field arrays) -> OCTETS."
  (let ((parts (list (u64-le (length messages) 4) +magic+)))     ; reversed, like the pushes below
    (map nil (lambda (m)
               (cond ((byte-message-p m)
                      (push (octets-concat '(0) (u64-le (length m))) parts)
                      (push (coerce m 'octets) parts))
                     ((field-array-p m)
                      (let ((dims (array-dimensions m)))
                        (when (> (length dims) +max-ndim+)
                          (error 'invalid-parameter :name 'array-rank :value (length dims)
                                                    :expected (format nil "<= ~D" +max-ndim+)))
                        (dotimes (i (array-total-size m))
                          (unless (< (row-major-aref m i) +p+)
                            (error 'invalid-parameter :name 'field-value :value (row-major-aref m i)
                                                      :expected "below p")))
                        (push (apply #'octets-concat (list 1 (length dims)) (mapcar #'u64-le dims)) parts)
                        (push (field-array-octets m) parts)))
                     (t (error 'invalid-parameter :name 'proof-message :value (type-of m)
                                                  :expected "a byte message or a field array"))))
         messages)
    (apply #'octets-concat (nreverse parts))))

(defun read-proof (bytes &key (start 0) (end (length bytes)))
  "Parse BYTES[START, END) as a proof -> a simple-vector of messages.  Signals MALFORMED-PROOF."
  (declare (type (simple-array (unsigned-byte 8) (*)) bytes) (type fixnum start end))
  (let ((pos start))
    (declare (type fixnum pos))
    (labels ((take (n)
               (when (or (minusp n) (> n (- end pos)))
                 (malformed :truncated :offset pos :detail (format nil "~D bytes needed" n)))
               (prog1 pos (incf pos n)))
             (uint (nbytes)
               (let ((at (take nbytes)) (v 0))
                 (dotimes (i nbytes v)
                   (setf v (logior v (ash (aref bytes (+ at i)) (* 8 i))))))))
      (unless (and (>= (- end pos) (length +magic+))
                   (not (mismatch +magic+ bytes :start2 pos :end2 (+ pos (length +magic+)))))
        (malformed :magic :offset pos))
      (incf pos (length +magic+))
      (let* ((count (uint 4))
             (msgs (make-array count :fill-pointer 0)))
        (dotimes (k count)
          (let* ((at pos)
                 (tag (uint 1)))
            (case tag
              (0 (let* ((n (uint 8))
                        (from (take n)))
                   (vector-push (subseq bytes from (+ from n)) msgs)))
              (1 (let ((ndim (uint 1)))
                   (when (> ndim +max-ndim+)
                     (malformed :rank :offset at :detail (format nil "rank ~D" ndim)))
                   (let* ((dims (loop repeat ndim collect (uint 8)))
                          (n (reduce #'* dims)))
                     (when (some (lambda (d) (>= d array-dimension-limit)) dims)
                       (malformed :dimension :offset at :detail (format nil "dimensions ~S" dims)))
                     ;; exact product: a shape that overflows 64 bits cannot fit in the input
                     (when (> n (floor (- end pos) 4))
                       (malformed :truncated :offset at
                                  :detail (format nil "array of ~D values" n)))
                     (let* ((from (take (* 4 n)))
                            (a (make-array dims :element-type '(unsigned-byte 32))))
                       (dotimes (i n)
                         (let* ((o (+ from (* 4 i)))
                                (v (logior (aref bytes o) (ash (aref bytes (+ o 1)) 8)
                                           (ash (aref bytes (+ o 2)) 16) (ash (aref bytes (+ o 3)) 24))))
                           (unless (< v +p+)
                             (malformed :value-range :offset o :detail (format nil "value ~D" v)))
                           (setf (row-major-aref a i) v)))
                       (vector-push a msgs)))))
              (t (malformed :unknown-tag :offset at :detail (format nil "tag ~D" tag))))))
        (unless (= pos end)
          (malformed :trailing-bytes :offset pos :detail (format nil "~D bytes" (- end pos))))
        (coerce msgs 'simple-vector)))))

(defun read-file-octets (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((buf (make-octets (file-length in))))
      (let ((n (read-sequence buf in)))
        (if (= n (length buf)) buf (subseq buf 0 n))))))

(defun read-proof-file (path)
  "Read and parse the proof file PATH (zk/proofio.py format, e.g. the .proof of BENCH_SAVE)."
  (read-proof (read-file-octets path)))

(defun write-proof-file (path messages &key (if-exists :supersede))
  (with-open-file (out path :element-type '(unsigned-byte 8) :direction :output :if-exists if-exists)
    (write-sequence (write-proof messages) out))
  path)

(defun describe-proof (messages &optional (stream *standard-output*))
  "A summary of a parsed proof: counts and sizes by message kind and shape."
  (let ((shapes (make-hash-table :test 'equal))
        (nbytes 0) (nfield 0) (byte-total 0) (field-total 0))
    (map nil (lambda (m)
               (if (byte-message-p m)
                   (progn (incf nbytes) (incf byte-total (length m))
                          (incf (gethash (list :bytes (length m)) shapes 0)))
                   (progn (incf nfield) (incf field-total (array-total-size m))
                          (incf (gethash (array-dimensions m) shapes 0)))))
         messages)
    (format stream "~&~D messages: ~D byte messages (~:D bytes), ~D field arrays (~:D values, ~:D bytes)~%"
            (length messages) nbytes byte-total nfield field-total (* 4 field-total))
    (let ((rows '()))
      (maphash (lambda (k v) (push (cons k v) rows)) shapes)
      (setf rows (sort rows #'> :key #'cdr))
      (format stream "~&~10@A  shape~%" "count")
      (loop for (shape . count) in rows
            repeat 25
            do (format stream "~10D  ~:[(~{~D~^ x ~})~;~{~(~A~) ~D~}~]~%" count (eq (car shape) :bytes) shape))
      (when (> (length rows) 25)
        (format stream "~10@A  ... ~D more shapes~%" "" (- (length rows) 25))))
    (values (length messages) nbytes nfield)))
