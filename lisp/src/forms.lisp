;;;; forms.lisp -- arithmetic over the field written as ordinary Lisp forms (zk/poly.py and the
;;;; sumcheck expressions of zk/sumcheck.py).
;;;;
;;;; A form is an integer, a variable (a symbol), or (+ f ...), (- f ...), (* f ...), (expt f k)
;;;; with k a non-negative integer:
;;;;
;;;;   (* (+ c0 3) (- c1 c0) c2)            a constraint over columns c0, c1, c2
;;;;   (* eq t0 t1)                         a sumcheck summand over tables t0, t1
;;;;
;;;; Forms are data: EXPAND puts one in a canonical sum of monomials, NORMALIZE writes that back as
;;;; a form, FORM-DEGREE and FORM-VARIABLES read it, and SUBLIS renames its variables.  They are
;;;; also code: ARITHMETIC-FORM turns one into Lisp code over F_p or F_p^D, which COMPILE-FORM
;;;; compiles (once per form) and EVALUATE runs.

(defpackage #:sable.forms
  (:use #:cl #:sable.conditions #:sable.field)
  (:export #:expand #:normalize #:form-degree #:form-variables #:constant-term #:form=
           #:arithmetic-form #:compile-form #:evaluate))

(in-package #:sable.forms)

(defun malformed-form (form)
  (error 'invalid-parameter :name 'form :value form
                            :expected "an integer, a symbol, or (+|-|* form...) or (expt form k)"))

;;; ------------------------------------------------------------------ forms as data

(defun variable< (a b)
  (string< (symbol-name a) (symbol-name b)))

(defun monomial< (a b)
  "Graded order: by degree, then by variables."
  (or (< (length a) (length b))
      (and (= (length a) (length b))
           (let ((i (mismatch a b)))
             (and i (variable< (nth i a) (nth i b)))))))

(defun collect-terms (terms)
  "TERMS, an alist (monomial . coefficient), with like terms merged, zero terms dropped, in
graded order."
  (let ((table (make-hash-table :test 'equal)))
    (loop for (monomial . c) in terms
          do (setf (gethash monomial table) (fadd (gethash monomial table 0) (modp c))))
    (sort (loop for monomial being the hash-keys of table using (hash-value c)
                unless (zerop c) collect (cons monomial c))
          #'monomial< :key #'car)))

(defun scale-terms (terms c)
  (loop for (monomial . d) in terms collect (cons monomial (fmul d (modp c)))))

(defun multiply-terms (a b)
  (collect-terms (loop for (m1 . c1) in a
                       nconc (loop for (m2 . c2) in b
                                   collect (cons (merge 'list (copy-list m1) (copy-list m2) #'variable<)
                                                 (fmul c1 c2))))))

(defun expand (form)
  "FORM as a sum of monomials over F_p: an alist (monomial . coefficient), each monomial a sorted
list of variables, like terms merged, zero terms dropped, in graded order.  Two forms are equal
polynomials exactly when their expansions are EQUAL."
  (typecase form
    (integer (collect-terms (list (cons '() form))))
    ((and symbol (not null)) (list (cons (list form) 1)))
    (cons
     (destructuring-bind (operator &rest arguments) form
       (case operator
         (+ (collect-terms (mapcan #'expand arguments)))
         (- (if (rest arguments)
                (collect-terms (append (expand (first arguments))
                                       (scale-terms (mapcan #'expand (rest arguments)) -1)))
                (scale-terms (expand (first arguments)) -1)))
         (* (reduce #'multiply-terms (mapcar #'expand arguments)
                    :initial-value (expand 1)))
         (expt (destructuring-bind (base k) arguments
                 (unless (typep k '(integer 0)) (malformed-form form))
                 (expand `(* ,@(make-list k :initial-element base)))))
         (t (malformed-form form)))))
    (t (malformed-form form))))

(defun normalize (form)
  "FORM as a canonical form: a sum of monomials in graded order, with coefficients in (-p/2, p/2]."
  (flet ((term (monomial c)
           (let ((c (to-signed c)))
             (cond ((null monomial) c)
                   ((/= c 1) `(* ,c ,@monomial))
                   ((rest monomial) `(* ,@monomial))
                   (t (first monomial))))))
    (let ((terms (loop for (monomial . c) in (expand form) collect (term monomial c))))
      (cond ((null terms) 0)
            ((rest terms) `(+ ,@terms))
            (t (first terms))))))

(defun form= (a b)
  "True when the forms A and B are the same polynomial."
  (equal (expand a) (expand b)))

(defun form-variables (form)
  "The variables of FORM's expansion, sorted."
  (sort (remove-duplicates (loop for (monomial) in (expand form) append monomial)) #'variable<))

(defun form-degree (form &optional (variables nil variables-p))
  "The total degree of FORM, counting only VARIABLES if given (e.g. the tables of a sumcheck)."
  (loop for (monomial) in (expand form)
        maximize (if variables-p
                     (count-if (lambda (v) (member v variables)) monomial)
                     (length monomial))))

(defun constant-term (form)
  (or (cdr (assoc '() (expand form))) 0))

;;; ------------------------------------------------------------------ forms as code

(defun arithmetic-form (form &key degree)
  "FORM as Lisp code over F_p, or over F_p^DEGREE if DEGREE is given: the operators become the
field's (FADD, EXT* ...), integers its constants; the variables are left free."
  (multiple-value-bind (add sub neg mul expt zero one constant variable)
      (if degree
          (values 'ext+ 'ext- 'ext-neg 'ext* 'ext-expt `(make-ext ,degree) `(ext-one ,degree)
                  (lambda (n) `(base->ext ,(modp n) ,degree))
                  #'identity)
          (values 'fadd 'fsub 'fneg 'fmul 'fpow 0 1 #'modp (lambda (v) `(modp ,v))))
    (labels ((fold (operator arguments empty)
               (if arguments
                   (reduce (lambda (a b) `(,operator ,a ,b)) (mapcar #'walk arguments))
                   empty))
             (walk (f)
               (typecase f
                 (integer (funcall constant f))
                 ((and symbol (not null)) (funcall variable f))
                 (cons (destructuring-bind (operator &rest arguments) f
                         (case operator
                           (+ (fold add arguments zero))
                           (* (fold mul arguments one))
                           (- (if (rest arguments)
                                  (fold sub arguments zero)
                                  `(,neg ,(walk (first arguments)))))
                           (expt (destructuring-bind (base k) arguments
                                   (unless (typep k '(integer 0)) (malformed-form f))
                                   `(,expt ,(walk base) ,k)))
                           (t (malformed-form f)))))
                 (t (malformed-form f)))))
      (walk form))))

(defvar *compiled-forms* (make-hash-table :test 'equal)
  "(form variables degree) -> the compiled function.")

(defun mentioned-variables (form)
  "The variables written in FORM (some may cancel out of its expansion)."
  (typecase form
    ((and symbol (not null)) (list form))
    (cons (remove-duplicates
           (loop for argument in (if (eq (first form) 'expt) (list (second form)) (rest form))
                 append (mentioned-variables argument))))))

(defun compile-form (form variables &key degree)
  "A function of VARIABLES (in that order) computing FORM over F_p, or F_p^DEGREE; compiled once
per form."
  (let ((free (set-difference (mentioned-variables form) variables)))
    (when free
      (error 'invalid-parameter :name 'free-variables :value free
                                :expected (format nil "variables among ~S" variables))))
  (let ((key (list form variables degree)))
    (or (gethash key *compiled-forms*)
        (setf (gethash key *compiled-forms*)
              (compile nil `(lambda ,variables
                              (declare (ignorable ,@variables)
                                       (sb-ext:muffle-conditions sb-ext:compiler-note))
                              ,(arithmetic-form form :degree degree)))))))

(defun evaluate (form bindings &key degree)
  "The value of FORM over F_p, or F_p^DEGREE, with BINDINGS an alist (variable . value)."
  (apply (compile-form form (mapcar #'car bindings) :degree degree)
         (mapcar #'cdr bindings)))
