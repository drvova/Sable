;;;; blake3.lisp -- BLAKE3, in two modes.
;;;;
;;;; 1. Standard BLAKE3 (the tree mode of the specification; the output of the `blake3' Python
;;;;    package and of b3sum): BLAKE3-HASH, or MAKE-BLAKE3 / BLAKE3-UPDATE / BLAKE3-DIGEST.
;;;;
;;;; 2. The sequential mode of zk/cuda/b3s.h, the hash of the Fiat-Shamir transcript: one chaining
;;;;    value updated per 64-byte block (flags 0, counter = block index), outputs from the pending
;;;;    last block with its length and the flags CHUNK_END | ROOT.  MAKE-B3S / B3S-UPDATE /
;;;;    B3S-COPY / B3S-DIGEST.
;;;;
;;;; Both buffer input in 64-byte blocks and compress a full block only once more input arrives,
;;;; so the last block, possibly empty, is always pending; FILL-BLOCKS does that for both.

(defpackage #:sable.blake3
  (:use #:cl #:sable.util)
  (:export #:compress
           #:b3s #:make-b3s #:b3s-update #:b3s-copy #:b3s-digest
           #:make-blake3 #:blake3-update #:blake3-digest #:blake3-hash))

(in-package #:sable.blake3)

(deftype word () '(unsigned-byte 32))
(deftype words () '(simple-array (unsigned-byte 32) (*)))

(defconstant +chunk-start+ 1)
(defconstant +chunk-end+ 2)
(defconstant +parent+ 4)
(defconstant +root+ 8)
(defconstant +chunk-length+ 1024)

(define-constant +iv+
    (make-array 8 :element-type 'word
                  :initial-contents '(#x6A09E667 #xBB67AE85 #x3C6EF372 #xA54FF53A
                                      #x510E527F #x9B05688C #x1F83D9AB #x5BE0CD19)))

(define-constant +message-permutation+
    (make-array 16 :element-type '(unsigned-byte 4)
                   :initial-contents '(2 6 3 10 7 0 4 13 1 11 12 5 9 14 15 8)))

;;; ------------------------------------------------------------------ the compression function

(defun compress (cv m counter block-length flags)
  "The BLAKE3 compression function: chaining value CV (8 words), message block M (16 words) ->
16 words."
  (declare (type words cv m) (type (unsigned-byte 64) counter) (type word block-length flags)
           (optimize speed))
  (let ((v (make-array 16 :element-type 'word))
        (m (copy-seq m)))
    (declare (type words v m))
    (replace v cv)
    (replace v +iv+ :start1 8 :end2 4)
    (setf (aref v 12) (ldb (byte 32 0) counter)
          (aref v 13) (ldb (byte 32 32) counter)
          (aref v 14) block-length
          (aref v 15) flags)
    (macrolet ((add (place &rest terms)
                 `(setf ,place (ldb (byte 32 0) (+ ,place ,@terms))))
               (xor-rotate (place other n)
                 `(setf ,place (let ((w (logxor ,place ,other)))
                                 (logior (ash w (- ,n)) (ldb (byte 32 0) (ash w (- 32 ,n))))))))
      (flet ((g (a b c d x y)
               (declare (type (mod 16) a b c d) (type word x y))
               (add (aref v a) (aref v b) x)
               (xor-rotate (aref v d) (aref v a) 16)
               (add (aref v c) (aref v d))
               (xor-rotate (aref v b) (aref v c) 12)
               (add (aref v a) (aref v b) y)
               (xor-rotate (aref v d) (aref v a) 8)
               (add (aref v c) (aref v d))
               (xor-rotate (aref v b) (aref v c) 7)))
        (declare (inline g))
        (loop repeat 7 do
          (g 0 4 8 12 (aref m 0) (aref m 1))           ; columns
          (g 1 5 9 13 (aref m 2) (aref m 3))
          (g 2 6 10 14 (aref m 4) (aref m 5))
          (g 3 7 11 15 (aref m 6) (aref m 7))
          (g 0 5 10 15 (aref m 8) (aref m 9))          ; diagonals
          (g 1 6 11 12 (aref m 10) (aref m 11))
          (g 2 7 8 13 (aref m 12) (aref m 13))
          (g 3 4 9 14 (aref m 14) (aref m 15))
          (setf m (map 'words (lambda (i) (aref m i)) +message-permutation+)))))
    (dotimes (i 8 v)
      (let ((lo (aref v i))
            (hi (aref v (+ i 8))))
        (setf (aref v i) (logxor lo hi)
              (aref v (+ i 8)) (logxor hi (aref cv i)))))))

(defun block-words (block)
  "64 bytes -> 16 little-endian words."
  (let ((words (make-array 16 :element-type 'word)))
    (dotimes (i 16 words)
      (setf (aref words i) (le-integer block (* 4 i) 4)))))

(defun chaining-value (words)
  (subseq words 0 8))

(defun fill-blocks (block used data start end compress-full-block)
  "Copy DATA[START, END) into the 64-byte BLOCK, of which USED bytes are taken.  A full block is
passed to COMPRESS-FULL-BLOCK, and emptied, only when more input remains.  Returns the new USED."
  (loop while (< start end)
        do (when (= used 64)
             (funcall compress-full-block block)
             (fill block 0)
             (setf used 0))
           (let ((n (min (- end start) (- 64 used))))
             (replace block data :start1 used :start2 start :end2 (+ start n))
             (incf used n)
             (incf start n)))
  used)

(defun root-output (cv words block-length flags n)
  "N bytes of root output: 64-byte block i is the compression of WORDS with counter i and the
flags FLAGS | ROOT, as little-endian bytes."
  (let ((out (make-octets n)))
    (loop for counter from 0
          for pos from 0 below n by 64
          do (loop with block = (compress cv words counter block-length (logior flags +root+))
                   for i from pos below (min n (+ pos 64))
                   for j from 0
                   do (setf (aref out i) (ldb (byte 8 (* 8 (mod j 4))) (aref block (floor j 4))))))
    out))

;;; ------------------------------------------------------------------ sequential mode (b3s.h)

(defstruct (b3s (:constructor %make-b3s) (:copier nil))
  "The state of struct B3S: chaining value, pending block, blocks compressed so far."
  (cv (copy-seq +iv+) :type words)
  (block (make-octets 64) :type (octets 64))
  (used 0 :type (integer 0 64))
  (counter 0 :type (unsigned-byte 64)))

(defun b3s-update (s data &key (start 0) (end (length data)))
  "Absorb DATA[START, END) into the sequential-mode state S; returns S."
  (setf (b3s-used s)
        (fill-blocks (b3s-block s) (b3s-used s) data start end
                     (lambda (block)
                       (setf (b3s-cv s) (chaining-value (compress (b3s-cv s) (block-words block)
                                                                  (b3s-counter s) 64 0)))
                       (incf (b3s-counter s)))))
  s)

(defun make-b3s (&optional data)
  "A fresh sequential-mode state, with DATA absorbed if given."
  (let ((s (%make-b3s)))
    (if data (b3s-update s data) s)))

(defun b3s-copy (s)
  (%make-b3s :cv (copy-seq (b3s-cv s)) :block (copy-seq (b3s-block s))
             :used (b3s-used s) :counter (b3s-counter s)))

(defun b3s-digest (s n)
  "N output bytes for everything absorbed into S so far; S is unchanged."
  (root-output (b3s-cv s) (block-words (b3s-block s)) (b3s-used s) +chunk-end+ n))

;;; ------------------------------------------------------------------ standard BLAKE3 (tree mode)

(defstruct (chunk (:constructor make-chunk (counter)))
  (cv (copy-seq +iv+) :type words)
  (counter 0 :type (unsigned-byte 64))
  (block (make-octets 64) :type (octets 64))
  (used 0 :type (integer 0 64))
  (blocks-compressed 0 :type fixnum))

(defun chunk-length (c)
  (+ (* 64 (chunk-blocks-compressed c)) (chunk-used c)))

(defun chunk-start-flag (c)
  (if (zerop (chunk-blocks-compressed c)) +chunk-start+ 0))

(defun chunk-update (c data start end)
  (setf (chunk-used c)
        (fill-blocks (chunk-block c) (chunk-used c) data start end
                     (lambda (block)
                       (setf (chunk-cv c) (chaining-value (compress (chunk-cv c) (block-words block)
                                                                    (chunk-counter c) 64
                                                                    (chunk-start-flag c))))
                       (incf (chunk-blocks-compressed c))))))

;;; A node of the tree: the inputs of the compression that gives its chaining value or, at the
;;; root, the output.
(defstruct (node (:constructor make-node (cv words counter block-length flags)))
  cv words counter block-length flags)

(defun chunk-node (c)
  (make-node (chunk-cv c) (block-words (chunk-block c)) (chunk-counter c) (chunk-used c)
             (logior (chunk-start-flag c) +chunk-end+)))

(defun parent-node (left-cv right-cv)
  (make-node +iv+ (concatenate 'words left-cv right-cv) 0 64 +parent+))

(defun node-chaining-value (node)
  (chaining-value (compress (node-cv node) (node-words node) (node-counter node)
                            (node-block-length node) (node-flags node))))

(defstruct (blake3 (:constructor make-blake3 ()))
  "A standard BLAKE3 hasher: the current chunk, and the chaining values of completed subtrees,
most recent first."
  (chunk (make-chunk 0) :type chunk)
  (stack '() :type list))

(defun finish-chunk (h)
  "Push the chaining value of H's full chunk, merging completed subtrees (one parent per
trailing zero bit of the chunk count), and start the next chunk."
  (let* ((chunk (blake3-chunk h))
         (cv (node-chaining-value (chunk-node chunk))))
    (loop for chunks = (1+ (chunk-counter chunk)) then (ash chunks -1)
          while (evenp chunks)
          do (setf cv (node-chaining-value (parent-node (pop (blake3-stack h)) cv))))
    (push cv (blake3-stack h))
    (setf (blake3-chunk h) (make-chunk (1+ (chunk-counter chunk))))))

(defun blake3-update (h data &key (start 0) (end (length data)))
  "Absorb DATA[START, END) into the hasher H; returns H."
  (loop while (< start end)
        do (when (= (chunk-length (blake3-chunk h)) +chunk-length+)
             (finish-chunk h))
           (let ((n (min (- end start) (- +chunk-length+ (chunk-length (blake3-chunk h))))))
             (chunk-update (blake3-chunk h) data start (+ start n))
             (incf start n)))
  h)

(defun blake3-digest (h &optional (n 32))
  "N bytes of output (extendable) for everything absorbed into H; H is unchanged."
  (let ((root (reduce (lambda (node cv) (parent-node cv (node-chaining-value node)))
                      (blake3-stack h)
                      :initial-value (chunk-node (blake3-chunk h)))))
    (root-output (node-cv root) (node-words root) (node-block-length root) (node-flags root) n)))

(defun blake3-hash (data &optional (n 32))
  "The N-byte BLAKE3 hash of the octets DATA."
  (blake3-digest (blake3-update (make-blake3) data) n))
