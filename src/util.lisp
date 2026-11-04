;;;; util.lisp -- octet vectors and little-endian integers.

(defpackage #:exacta.util
  (:use #:cl)
  (:export #:octets #:make-octets #:to-octets #:join-octets #:le-bytes #:le-integer))

(in-package #:exacta.util)

(deftype octets (&optional (length '*))
  `(simple-array (unsigned-byte 8) (,length)))

(defun make-octets (length)
  (make-array length :element-type '(unsigned-byte 8) :initial-element 0))

(defun to-octets (x)
  "A string (encoded as UTF-8) or a sequence of bytes, as OCTETS."
  (etypecase x
    (octets x)
    (string (sb-ext:string-to-octets x :external-format :utf-8))
    (sequence (coerce x 'octets))))

(defun join-octets (parts)
  "Concatenate a list of byte sequences into one OCTETS."
  (let ((out (make-octets (reduce #'+ parts :key #'length)))
        (pos 0))
    (dolist (part parts out)
      (replace out part :start1 pos)
      (incf pos (length part)))))

(defun le-bytes (n &optional (size 8))
  "The integer N as SIZE little-endian bytes."
  (let ((out (make-octets size)))
    (dotimes (i size out)
      (setf (aref out i) (ldb (byte 8 (* 8 i)) n)))))

(declaim (inline le-integer))
(defun le-integer (octets &optional (start 0) (size (- (length octets) start)))
  "The little-endian integer in OCTETS[START, START + SIZE)."
  (declare (type octets octets) (type (integer 0 8) size) (type fixnum start))
  (let ((n 0))
    (declare (type (unsigned-byte 64) n))
    (dotimes (i size n)
      (setf n (logior n (ash (aref octets (+ start i)) (* 8 i)))))))
