;;;; poly.lisp -- polynomials over column symbols with base-field coefficients (zk/poly.py).
;;;;
;;;; A POLY maps monomials to coefficients mod p.  A monomial is a sorted list of column ids, with
;;;; repeats (() is the constant monomial).  Zero coefficients are never stored, so equal
;;;; polynomials have equal terms.  Ids below zero are layer parameters (LPARAM); RENAME keeps them.

(defpackage #:sable.poly
  (:use #:cl #:sable.conditions #:sable.field)
  (:export #:poly #:poly-p #:make-poly #:poly-terms #:poly-var #:poly-const
           #:poly+ #:poly- #:poly* #:poly-expt #:poly= #:poly-degree #:poly-vars #:poly-const-term
           #:poly-rename #:poly-eval #:+lparam-names+ #:lparam #:lparam-index))

(in-package #:sable.poly)

(defstruct (poly (:constructor %make-poly (table)))
  (table (make-hash-table :test 'equal) :type hash-table :read-only t))

(defun poly-terms (p)
  "The terms of P as an alist of (monomial . coefficient)."
  (loop for monomial being the hash-keys of (poly-table p) using (hash-value c)
        collect (cons monomial c)))

(defun make-poly (terms)
  "The POLY that sums TERMS, an alist of (monomial . coefficient); like terms are merged."
  (let ((table (make-hash-table :test 'equal)))
    (loop for (monomial . c) in terms
          for key = (sort (copy-list monomial) #'<)
          do (setf (gethash key table) (fadd (gethash key table 0) (modp c))))
    (loop for key being the hash-keys of table using (hash-value c)
          when (zerop c) do (remhash key table))
    (%make-poly table)))

(defun monomial< (a b)
  "Graded lexicographic order: by degree, then by ids."
  (or (< (length a) (length b))
      (and (= (length a) (length b))
           (let ((i (mismatch a b)))
             (and i (< (nth i a) (nth i b)))))))

(defmethod print-object ((p poly) stream)
  (print-unreadable-object (p stream :type t)
    (format stream "~:[0~;~:*~{~{~D~{*c~D~}~}~^ + ~}~]"
            (loop for (monomial . c) in (sort (poly-terms p) #'monomial< :key #'car)
                  collect (list c monomial)))))

(defun poly-var (id) (make-poly `(((,id) . 1))))
(defun poly-const (c) (make-poly `((() . ,c))))

(defun ensure-poly (x)
  (if (poly-p x) x (poly-const x)))

(defun poly+ (&rest summands)
  "The sum of polynomials and integers."
  (make-poly (mapcan (lambda (x) (poly-terms (ensure-poly x))) summands)))

(defun poly-scale (p c)
  (make-poly (loop for (monomial . d) in (poly-terms (ensure-poly p))
                   collect (cons monomial (fmul d (modp c))))))

(defun poly- (p &rest subtrahends)
  "P minus SUBTRAHENDS, or -P."
  (if subtrahends
      (apply #'poly+ p (mapcar (lambda (q) (poly-scale q -1)) subtrahends))
      (poly-scale p -1)))

(defun poly* (&rest factors)
  "The product of polynomials and integers."
  (flet ((mul (a b)
           (make-poly (loop for (m1 . c1) in (poly-terms a)
                            nconc (loop for (m2 . c2) in (poly-terms b)
                                        collect (cons (append m1 m2) (fmul c1 c2)))))))
    (reduce #'mul factors :key #'ensure-poly :initial-value (poly-const 1))))

(defun poly-expt (p e)
  (apply #'poly* (make-list e :initial-element p)))

(defun poly= (a b)
  (and (= (hash-table-count (poly-table a)) (hash-table-count (poly-table b)))
       (loop for (monomial . c) in (poly-terms a)
             always (eql c (gethash monomial (poly-table b))))))

(defun poly-degree (p)
  (loop for (monomial) in (poly-terms p) maximize (length monomial)))

(defun poly-vars (p)
  (sort (remove-duplicates (loop for (monomial) in (poly-terms p) append monomial)) #'<))

(defun poly-const-term (p)
  (gethash '() (poly-table p) 0))

(defun lookup-function (map)
  "A function of a column id from MAP: a function, hash table, alist or vector."
  (etypecase map
    (function map)
    (hash-table (lambda (id) (multiple-value-bind (value found) (gethash id map)
                               (if found value (error 'invalid-parameter :name 'column :value id
                                                                         :expected "a mapped column")))))
    (list (lambda (id) (let ((entry (assoc id map)))
                         (if entry (cdr entry) (error 'invalid-parameter :name 'column :value id
                                                                         :expected "a mapped column")))))
    (vector (lambda (id) (aref map id)))))

(defun poly-rename (p map)
  "P with each column id i >= 0 replaced by its image under MAP (see LOOKUP-FUNCTION)."
  (let ((rename (lookup-function map)))
    (make-poly (loop for (monomial . c) in (poly-terms p)
                     collect (cons (mapcar (lambda (id) (if (minusp id) id (funcall rename id))) monomial)
                                   c)))))

(defun poly-eval (p values)
  "P over F_p, with VALUES (see LOOKUP-FUNCTION) giving each column's value."
  (let ((value (lookup-function values)))
    (reduce #'fadd (loop for (monomial . c) in (poly-terms p)
                         collect (reduce #'fmul monomial :key (lambda (id) (modp (funcall value id)))
                                                         :initial-value c))
            :initial-value 0)))

;;; Layer parameters (per-layer format constants such as shift offsets) appear in constraint
;;; constant terms as variables with reserved ids below zero.
(defparameter +lparam-names+ '("Kco" "Kcd" "Dyn" "Dme"))

(defun lparam (name)
  (let ((index (position name +lparam-names+ :test #'string=)))
    (check-parameter index #'integerp (format nil "one of ~{~A~^, ~}" +lparam-names+))
    (poly-var (- -1 index))))

(defun lparam-index (id)
  (- -1 id))
