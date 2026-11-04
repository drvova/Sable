;;;; blake3.lisp -- BLAKE3, in two modes.
;;;;
;;;; 1. Standard BLAKE3 (the tree mode of the specification; same output as the `blake3' Python
;;;;    package and b3sum): BLAKE3-HASH, or MAKE-BLAKE3 / BLAKE3-UPDATE / BLAKE3-DIGEST.
;;;;
;;;; 2. The sequential mode of zk/cuda/b3s.h that the Fiat-Shamir transcript uses: the BLAKE3
;;;;    compression function with one chaining value updated per 64-byte block (flags 0, counter =
;;;;    block index); outputs come from the pending last block with its length and the flags
;;;;    CHUNK_END | ROOT.  B3S-INIT / B3S-UPDATE / B3S-COPY / B3S-DIGEST.  The state is a structure
;;;;    with the fields of struct B3S.

(defpackage #:sable.blake3
  (:use #:cl)
  (:export #:octets #:make-octets #:compress
           #:b3s #:b3s-init #:b3s-update #:b3s-copy #:b3s-digest
           #:make-blake3 #:blake3-update #:blake3-digest #:blake3-hash
           #:u64-le #:octets-concat))

(in-package #:sable.blake3)

(declaim (optimize (speed 3) (safety 1) (debug 1)))

(deftype octets () '(simple-array (unsigned-byte 8) (*)))
(deftype u32 () '(unsigned-byte 32))
(deftype words () '(simple-array (unsigned-byte 32) (*)))

(defconstant +chunk-start+ 1)
(defconstant +chunk-end+ 2)
(defconstant +parent+ 4)
(defconstant +root+ 8)
(defconstant +block-len+ 64)
(defconstant +chunk-len+ 1024)

(defun make-octets (n &optional (initial 0))
  (make-array n :element-type '(unsigned-byte 8) :initial-element initial))

(defun octets-concat (&rest parts)
  "Concatenate octet vectors (or lists of bytes) into one OCTETS."
  (let ((out (make-octets (reduce #'+ parts :key #'length)))
        (pos 0))
    (dolist (p parts out)
      (replace out p :start1 pos)
      (incf pos (length p)))))

(defun u64-le (n &optional (nbytes 8))
  "N as NBYTES little-endian bytes."
  (let ((out (make-octets nbytes)))
    (dotimes (i nbytes out)
      (setf (aref out i) (ldb (byte 8 (* 8 i)) n)))))

(declaim (type words +iv+))
(sb-ext:defglobal +iv+
    (make-array 8 :element-type '(unsigned-byte 32)
                  :initial-contents '(#x6A09E667 #xBB67AE85 #x3C6EF372 #xA54FF53A
                                      #x510E527F #x9B05688C #x1F83D9AB #x5BE0CD19)))

(declaim (type (simple-array (unsigned-byte 8) (16)) +perm+))
(sb-ext:defglobal +perm+
    (make-array 16 :element-type '(unsigned-byte 8)
                   :initial-contents '(2 6 3 10 7 0 4 13 1 11 12 5 9 14 15 8)))

(declaim (inline rotr32))
(defun rotr32 (x n)
  (declare (type u32 x) (type (integer 1 31) n))
  (logior (ash x (- n)) (ldb (byte 32 0) (ash x (- 32 n)))))

(defmacro u32+ (&rest xs)
  "A sum modulo 2^32."
  `(ldb (byte 32 0) (+ ,@xs)))

(defun compress (cv m counter block-len flags)
  "The BLAKE3 compression function: CV (8 words), M (16 words) -> 16 words."
  (declare (type words cv m) (type (unsigned-byte 64) counter) (type u32 block-len flags))
  (let ((v (make-array 16 :element-type 'u32))
        (m (copy-seq m))
        (tmp (make-array 16 :element-type 'u32))
        (out (make-array 16 :element-type 'u32)))
    (declare (type words v m tmp) (dynamic-extent v tmp))
    (dotimes (i 8)
      (setf (aref v i) (aref cv i)
            (aref v (+ i 8)) (aref +iv+ i)))
    (setf (aref v 12) (ldb (byte 32 0) counter)
          (aref v 13) (ldb (byte 32 32) counter)
          (aref v 14) block-len
          (aref v 15) flags)
    (macrolet ((g (a b c d mx my)
                 `(progn
                    (setf (aref v ,a) (u32+ (aref v ,a) (aref v ,b) ,mx)
                          (aref v ,d) (rotr32 (logxor (aref v ,d) (aref v ,a)) 16)
                          (aref v ,c) (u32+ (aref v ,c) (aref v ,d))
                          (aref v ,b) (rotr32 (logxor (aref v ,b) (aref v ,c)) 12)
                          (aref v ,a) (u32+ (aref v ,a) (aref v ,b) ,my)
                          (aref v ,d) (rotr32 (logxor (aref v ,d) (aref v ,a)) 8)
                          (aref v ,c) (u32+ (aref v ,c) (aref v ,d))
                          (aref v ,b) (rotr32 (logxor (aref v ,b) (aref v ,c)) 7)))))
      (dotimes (r 7)
        (g 0 4 8 12 (aref m 0) (aref m 1))
        (g 1 5 9 13 (aref m 2) (aref m 3))
        (g 2 6 10 14 (aref m 4) (aref m 5))
        (g 3 7 11 15 (aref m 6) (aref m 7))
        (g 0 5 10 15 (aref m 8) (aref m 9))
        (g 1 6 11 12 (aref m 10) (aref m 11))
        (g 2 7 8 13 (aref m 12) (aref m 13))
        (g 3 4 9 14 (aref m 14) (aref m 15))
        (dotimes (i 16) (setf (aref tmp i) (aref m (aref +perm+ i))))
        (replace m tmp)))
    (dotimes (i 8 out)
      (setf (aref out i) (logxor (aref v i) (aref v (+ i 8)))
            (aref out (+ i 8)) (logxor (aref v (+ i 8)) (aref cv i))))))

(defun block-words (block)
  "64 bytes -> 16 little-endian words."
  (declare (type octets block))
  (let ((w (make-array 16 :element-type 'u32)))
    (dotimes (i 16 w)
      (setf (aref w i) (logior (aref block (* 4 i))
                               (ash (aref block (+ 1 (* 4 i))) 8)
                               (ash (aref block (+ 2 (* 4 i))) 16)
                               (ash (aref block (+ 3 (* 4 i))) 24))))))

(defun emit-words (words out pos n)
  "Write the first N (<= 64) little-endian bytes of WORDS into OUT at POS."
  (declare (type words words) (type octets out) (type fixnum pos n))
  (dotimes (i n)
    (setf (aref out (+ pos i)) (ldb (byte 8 (* 8 (logand i 3))) (aref words (ash i -2))))))

(defun first8 (words)
  (declare (type words words))
  (subseq words 0 8))

;;; ------------------------------------------------------------------ sequential mode (b3s.h)

(defstruct (b3s (:constructor %make-b3s) (:copier nil))
  (cv (copy-seq +iv+) :type words)
  (block (make-octets 64) :type octets)
  (counter 0 :type (unsigned-byte 64))
  (block-len 0 :type (integer 0 64)))

(defun b3s-init (&optional data)
  "A fresh sequential-mode state, with DATA absorbed if given."
  (let ((s (%make-b3s)))
    (when data (b3s-update s data))
    s))

(defun b3s-copy (s)
  (%make-b3s :cv (copy-seq (b3s-cv s)) :block (copy-seq (b3s-block s))
             :counter (b3s-counter s) :block-len (b3s-block-len s)))

(defun b3s-update (s data &key (start 0) (end (length data)))
  "Absorb DATA[START, END).  A full pending block is compressed (flags 0, counter = its index) only
when more input arrives, so the last block, possibly empty, is always pending."
  (declare (type b3s s) (type (or octets (vector (unsigned-byte 8))) data) (type fixnum start end))
  (let ((blk (b3s-block s)))
    (loop while (< start end)
          do (when (= (b3s-block-len s) +block-len+)
               (setf (b3s-cv s) (first8 (compress (b3s-cv s) (block-words blk) (b3s-counter s) 64 0)))
               (incf (b3s-counter s))
               (setf (b3s-block-len s) 0)
               (fill blk 0))
             (let ((take (min (- end start) (- +block-len+ (b3s-block-len s)))))
               (replace blk data :start1 (b3s-block-len s) :start2 start :end2 (+ start take))
               (incf (b3s-block-len s) take)
               (incf start take))))
  s)

(defun b3s-digest (s n)
  "N output bytes for everything absorbed so far; S is unchanged.  64-byte output block i is the
compression of the pending block with counter i, its length and the flags CHUNK_END | ROOT."
  (declare (type b3s s) (type fixnum n))
  (let ((w (block-words (b3s-block s)))
        (out (make-octets n)))
    (loop for ctr of-type fixnum from 0
          for pos of-type fixnum from 0 by 64
          while (< pos n)
          do (emit-words (compress (b3s-cv s) w ctr (b3s-block-len s) (logior +chunk-end+ +root+))
                         out pos (min 64 (- n pos))))
    out))

;;; ------------------------------------------------------------------ standard BLAKE3 (tree mode)

(defstruct (chunk (:constructor make-chunk (cv counter)))
  (cv nil :type words)
  (counter 0 :type (unsigned-byte 64))
  (block (make-octets 64) :type octets)
  (block-len 0 :type (integer 0 64))
  (blocks-compressed 0 :type fixnum))

(declaim (inline chunk-len chunk-start-flag))
(defun chunk-len (c)
  (+ (* 64 (chunk-blocks-compressed c)) (chunk-block-len c)))

(defun chunk-start-flag (c)
  (if (zerop (chunk-blocks-compressed c)) +chunk-start+ 0))

(defun chunk-update (c data start end)
  (declare (type chunk c) (type fixnum start end))
  (loop while (< start end)
        do (when (= (chunk-block-len c) +block-len+)
             (setf (chunk-cv c) (first8 (compress (chunk-cv c) (block-words (chunk-block c))
                                                  (chunk-counter c) 64 (chunk-start-flag c))))
             (incf (chunk-blocks-compressed c))
             (setf (chunk-block-len c) 0)
             (fill (chunk-block c) 0))
           (let ((take (min (- end start) (- +block-len+ (chunk-block-len c)))))
             (replace (chunk-block c) data :start1 (chunk-block-len c) :start2 start :end2 (+ start take))
             (incf (chunk-block-len c) take)
             (incf start take))))

;; An "output": the inputs of a compression whose result is either a chaining value or root output.
(defstruct (output (:constructor make-output (input-cv words counter block-len flags)))
  input-cv words counter block-len flags)

(defun chunk-output (c)
  (make-output (chunk-cv c) (block-words (chunk-block c)) (chunk-counter c) (chunk-block-len c)
               (logior (chunk-start-flag c) +chunk-end+)))

(defun output-cv (o)
  "The chaining value of the output O."
  (first8 (compress (output-input-cv o) (output-words o) (output-counter o) (output-block-len o) (output-flags o))))

(defun parent-output (left right)
  (let ((w (make-array 16 :element-type 'u32)))
    (replace w left)
    (replace w right :start1 8)
    (make-output +iv+ w 0 64 +parent+)))

(defstruct (blake3 (:constructor %make-blake3))
  (chunk (make-chunk +iv+ 0) :type chunk)
  (stack nil :type list))

(defun make-blake3 () (%make-blake3))

(defun blake3-update (h data &key (start 0) (end (length data)))
  "Absorb DATA[START, END) into the standard BLAKE3 hasher H."
  (declare (type blake3 h) (type fixnum start end))
  (loop while (< start end)
        do (let ((c (blake3-chunk h)))
             (when (= (chunk-len c) +chunk-len+)
               (let ((cv (output-cv (chunk-output c)))
                     (total (1+ (chunk-counter c))))
                 ;; merge completed subtrees: one parent per trailing zero bit of the chunk count
                 (loop while (evenp total)
                       do (setf cv (output-cv (parent-output (pop (blake3-stack h)) cv))
                                total (ash total -1)))
                 (push cv (blake3-stack h))
                 (setf c (make-chunk +iv+ (1+ (chunk-counter c)))
                       (blake3-chunk h) c)))
             (let ((take (min (- end start) (- +chunk-len+ (chunk-len c)))))
               (chunk-update c data start (+ start take))
               (incf start take))))
  h)

(defun blake3-digest (h &optional (n 32))
  "N bytes of the root output of H (extendable output); H is unchanged."
  (let ((o (chunk-output (blake3-chunk h))))
    (dolist (cv (blake3-stack h))
      (setf o (parent-output cv (output-cv o))))
    (let ((out (make-octets n)))
      (loop for ctr from 0
            for pos from 0 by 64
            while (< pos n)
            do (emit-words (compress (output-input-cv o) (output-words o) ctr (output-block-len o)
                                     (logior (output-flags o) +root+))
                           out pos (min 64 (- n pos))))
      out)))

(defun blake3-hash (data &optional (n 32))
  "The BLAKE3 hash of DATA (octets), N bytes long."
  (blake3-digest (blake3-update (make-blake3) data) n))
