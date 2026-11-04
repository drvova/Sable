;;;; poly.lisp -- polynomials over column symbols with base-field coefficients (zk/poly.py).
;;;;
;;;; A POLY maps monomials to coefficients mod p.  A monomial is a sorted list of column ids
;;;; (repeats allowed; () is the constant monomial).  Zero coefficients are never stored, so two
;;;; equal polynomials have equal term sets.  Ids below zero are layer parameters (LPARAM), which
;;;; RENAME keeps.

(defpackage #:sable.poly
  (:use #:cl #:sable.conditions #:sable.field)
  (:export #:poly #:poly-terms #:make-poly #:poly-var #:poly-const #:poly-lift
           #:poly+ #:poly- #:poly* #:poly-neg #:poly-expt #:poly-degree #:poly-vars #:poly-const-term
           #:poly-rename #:poly-eval #:poly= #:+lparam-names+ #:lparam #:lparam-index))

(in-package #:sable.poly)

(defstruct (poly (:constructor %make-poly (terms)))
  "TERMS: an EQUAL hash table, monomial -> nonzero coefficient in [0, p)."
  (terms (make-hash-table :test 'equal) :type hash-table :read-only t))

(defun make-poly (&optional alist)
  "A POLY from an alist of (monomial . coefficient); monomials are sorted, like terms merged."
  (let ((tb (make-hash-table :test 'equal)))
    (loop for (mono . c) in alist
          do (let ((k (sort (copy-list mono) #'<)))
               (setf (gethash k tb) (fadd (gethash k tb 0) (fp-canon c)))))
    (maphash (lambda (k v) (when (zerop v) (remhash k tb))) tb)
    (%make-poly tb)))

(defun poly-alist (p)
  (let ((out '()))
    (maphash (lambda (k v) (push (cons k v) out)) (poly-terms p))
    out))

(defmethod print-object ((p poly) stream)
  (print-unreadable-object (p stream :type t)
    (let ((ts (sort (poly-alist p) (lambda (a b) (or (< (length (car a)) (length (car b)))
                                                     (and (= (length (car a)) (length (car b)))
                                                          (string< (prin1-to-string (car a))
                                                                   (prin1-to-string (car b)))))))))
      (if ts
          (format stream "~{~A~^ + ~}"
                  (loop for (mono . c) in ts
                        collect (format nil "~D~{*c~D~}" c mono)))
          (write-string "0" stream)))))

(defun poly-var (i) (make-poly (list (cons (list i) 1))))
(defun poly-const (c) (make-poly (list (cons '() c))))
(defun poly-lift (x) (if (poly-p x) x (poly-const x)))

(defun poly+ (&rest ps)
  (make-poly (loop for p in ps nconc (poly-alist (poly-lift p)))))

(defun poly-neg (p)
  (make-poly (mapcar (lambda (kv) (cons (car kv) (fneg (cdr kv)))) (poly-alist (poly-lift p)))))

(defun poly- (p &rest qs)
  (if qs
      (apply #'poly+ p (mapcar #'poly-neg qs))
      (poly-neg p)))

(defun poly*2 (a b)
  (make-poly (loop for (k1 . v1) in (poly-alist (poly-lift a))
                   nconc (loop for (k2 . v2) in (poly-alist (poly-lift b))
                               collect (cons (append k1 k2) (fmul v1 v2))))))

(defun poly* (&rest ps)
  (reduce #'poly*2 ps :initial-value (poly-const 1)))

(defun poly-expt (p e)
  (let ((r (poly-const 1)))
    (dotimes (i e r) (setf r (poly*2 r p)))))

(defun poly-degree (p)
  (loop for (k) in (poly-alist p) maximize (length k)))

(defun poly-vars (p)
  (sort (remove-duplicates (loop for (k) in (poly-alist p) append k)) #'<))

(defun poly-const-term (p)
  (gethash '() (poly-terms p) 0))

(defun poly= (a b)
  (let ((ta (poly-terms a)) (tb (poly-terms b)))
    (and (= (hash-table-count ta) (hash-table-count tb))
         (loop for k being the hash-keys of ta using (hash-value v)
               always (eql v (gethash k tb))))))

(defun poly-rename (p m)
  "Substitute column ids through M (a function, hash table or alist); ids < 0 are kept."
  (let ((f (etypecase m
             (function m)
             (hash-table (lambda (i) (multiple-value-bind (v ok) (gethash i m)
                                       (if ok v (error 'invalid-parameter :name 'column :value i
                                                                          :expected "a renamed column")))))
             (list (lambda (i) (or (cdr (assoc i m))
                                   (error 'invalid-parameter :name 'column :value i
                                                             :expected "a renamed column")))))))
    (make-poly (loop for (k . v) in (poly-alist p)
                     collect (cons (mapcar (lambda (i) (if (minusp i) i (funcall f i))) k) v)))))

(defun poly-eval (p vals)
  "P over F_p with VALS (a function, hash table or vector) giving each column's value."
  (let ((get (etypecase vals
               (function vals)
               (hash-table (lambda (i) (gethash i vals)))
               (vector (lambda (i) (aref vals i))))))
    (let ((s 0))
      (maphash (lambda (k v)
                 (let ((m v))
                   (dolist (i k) (setf m (fmul m (fp-canon (funcall get i)))))
                   (setf s (fadd s m))))
               (poly-terms p))
      s)))

;; Layer parameters (per-layer format constants such as shift offsets) appear in constraint
;; constant terms as variables with reserved ids < 0.
(defparameter +lparam-names+ '("Kco" "Kcd" "Dyn" "Dme"))

(defun lparam (name)
  (poly-var (- -1 (or (position name +lparam-names+ :test #'string=)
                      (error 'invalid-parameter :name 'layer-parameter :value name
                                                :expected +lparam-names+)))))

(defun lparam-index (var-id)
  (- -1 var-id))
