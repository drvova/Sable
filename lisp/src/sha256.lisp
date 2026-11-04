;;;; sha256.lisp -- SHA-256 (FIPS 180-4), for the statement's digests (zk/llm.py uses hashlib).

(defpackage #:sable.sha256
  (:use #:cl #:sable.util)
  (:export #:sha256))

(in-package #:sable.sha256)

(defparameter *k*
  (coerce '(#x428a2f98 #x71374491 #xb5c0fbcf #xe9b5dba5 #x3956c25b #x59f111f1 #x923f82a4 #xab1c5ed5
            #xd807aa98 #x12835b01 #x243185be #x550c7dc3 #x72be5d74 #x80deb1fe #x9bdc06a7 #xc19bf174
            #xe49b69c1 #xefbe4786 #x0fc19dc6 #x240ca1cc #x2de92c6f #x4a7484aa #x5cb0a9dc #x76f988da
            #x983e5152 #xa831c66d #xb00327c8 #xbf597fc7 #xc6e00bf3 #xd5a79147 #x06ca6351 #x14292967
            #x27b70a85 #x2e1b2138 #x4d2c6dfc #x53380d13 #x650a7354 #x766a0abb #x81c2c92e #x92722c85
            #xa2bfe8a1 #xa81a664b #xc24b8b70 #xc76c51a3 #xd192e819 #xd6990624 #xf40e3585 #x106aa070
            #x19a4c116 #x1e376c08 #x2748774c #x34b0bcb5 #x391c0cb3 #x4ed8aa4a #x5b9cca4f #x682e6ff3
            #x748f82ee #x78a5636f #x84c87814 #x8cc70208 #x90befffa #xa4506ceb #xbef9a3f7 #xc67178f2)
          '(simple-array (unsigned-byte 32) (64)))
  "The round constants: the first 32 bits of the fractional parts of the cube roots of the first
64 primes.")

(defun sha256 (data)
  "The SHA-256 digest of the octets (or string, as UTF-8) DATA: 32 octets."
  (let* ((data (to-octets data))
         (bits (* 8 (length data)))
         ;; padding: 1, zeros to 56 mod 64 bytes, the length in bits as a big-endian u64
         (padded (join-octets (list data #(#x80)
                                    (make-octets (mod (- 55 (length data)) 64))
                                    (reverse (le-bytes bits 8)))))
         (h (list #x6a09e667 #xbb67ae85 #x3c6ef372 #xa54ff53a #x510e527f #x9b05688c #x1f83d9ab #x5be0cd19))
         (w (make-array 64 :element-type '(unsigned-byte 32))))
    (flet ((rotr (x n) (logior (ash x (- n)) (ldb (byte 32 0) (ash x (- 32 n)))))
           (add (&rest xs) (ldb (byte 32 0) (apply #'+ xs))))
      (loop for block from 0 below (length padded) by 64
            do (dotimes (i 16)
                 (setf (aref w i) (loop for j below 4
                                        sum (ash (aref padded (+ block (* 4 i) j)) (* 8 (- 3 j))))))
               (loop for i from 16 below 64
                     do (let* ((a (aref w (- i 15))) (b (aref w (- i 2)))
                               (s0 (logxor (rotr a 7) (rotr a 18) (ash a -3)))
                               (s1 (logxor (rotr b 17) (rotr b 19) (ash b -10))))
                          (setf (aref w i) (add (aref w (- i 16)) s0 (aref w (- i 7)) s1))))
               (destructuring-bind (a b c d e f g hh) h
                 (dotimes (i 64)
                   (let* ((s1 (logxor (rotr e 6) (rotr e 11) (rotr e 25)))
                          (ch (logxor (logand e f) (logand (logxor e #xffffffff) g)))
                          (t1 (add hh s1 ch (aref *k* i) (aref w i)))
                          (s0 (logxor (rotr a 2) (rotr a 13) (rotr a 22)))
                          (maj (logxor (logand a b) (logand a c) (logand b c)))
                          (t2 (add s0 maj)))
                     (setf hh g g f f e e (add d t1) d c c b b a a (add t1 t2))))
                 (setf h (mapcar #'add h (list a b c d e f g hh))))))
    (join-octets (mapcar (lambda (x) (reverse (le-bytes x 4))) h))))
