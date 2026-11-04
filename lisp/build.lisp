;;;; build.lisp -- build the SBCL images.
;;;;
;;;;   sbcl --non-interactive --load build.lisp            -> sable       (standalone executable)
;;;;   sbcl --non-interactive --load build.lisp core       -> sable.core  (sbcl --core sable.core)
;;;;
;;;; Both images hold the sable system and its tests (with their vectors), and start in
;;;; SABLE-USER, so `sable '(run-tests)'' works anywhere.  The executable saves the runtime options
;;;; of the building SBCL (it parses no SBCL options of its own); give the heap size when building:
;;;;   sbcl --dynamic-space-size 16GB --non-interactive --load build.lisp

(require "asdf")

(let ((here (directory-namestring (or *load-truename* *default-pathname-defaults*))))
  (asdf:load-asd (merge-pathnames "sable.asd" here))
  (asdf:load-system "sable/tests")
  (use-package '#:sable.tests '#:sable-user)
  (let* ((args (rest sb-ext:*posix-argv*))        ; SBCL removes the options it processed
         (core-p (member "core" args :test #'string=))
         (out (merge-pathnames (if core-p "sable.core" "sable") here)))
    ;; check the image before saving it (this also compiles the tower code it uses)
    (unless (uiop:symbol-call '#:sable.tests '#:run-tests)
      (format *error-output* "~&build: tests failed, no image written~%")
      (sb-ext:exit :code 1))
    ;; compile the demo's kernels now, so the image's first (demo) does not
    (let ((*standard-output* (make-broadcast-stream)))
      (dolist (base '(nil t))
        (uiop:symbol-call '#:sable.sumcheck '#:demo :variables 6 :base base)))
    ;; forget where the sources were, so the image does not try to reload them
    (asdf:clear-configuration)
    (format t "~&build: writing ~A~%" out)
    (if core-p
        (progn
          (push (lambda () (setf *package* (find-package '#:sable-user))) sb-ext:*init-hooks*)
          (sb-ext:save-lisp-and-die out :executable nil))
        (sb-ext:save-lisp-and-die out :executable t
                                      :toplevel (symbol-function (uiop:find-symbol* '#:main '#:sable.toplevel))
                                      :save-runtime-options t))))
