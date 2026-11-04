;;;; util.lisp -- octet vectors, little-endian integers, and DEFINE-CONSTANT.

(defpackage #:sable.util
  (:use #:cl)
  (:export #:octets #:make-octets #:to-octets #:join-octets #:le-bytes #:le-integer
           #:define-constant))

(in-package #:sable.util)

(deftype octets (&optional (length '*))
  `(simple-array (unsigned-byte 8) (,length)))

(defmacro define-constant (name value &optional documentation)
  "DEFCONSTANT for a value that is EQUALP but not EQL across evaluations (a vector, a string):
reloading the file keeps the existing value instead of signalling a redefinition."
  `(defconstant ,name
     (let ((new ,value))
       (if (and (boundp ',name) (equalp (symbol-value ',name) new))
           (symbol-value ',name)
           new))
     ,@(when documentation (list documentation))))

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
