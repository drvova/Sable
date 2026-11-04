;;;; build.lisp -- build the SBCL images.
;;;;
;;;;   sbcl --non-interactive --load build.lisp            -> sable       (standalone executable)
;;;;   sbcl --non-interactive --load build.lisp core       -> sable.core  (a core for the REPL:
;;;;                                                          sbcl --core sable.core)
;;;;
;;;; Both images hold the sable system and its tests (with their vectors), so `sable selftest'
;;;; runs anywhere.  The executable saves the runtime options of the building SBCL (it parses no
;;;; SBCL options of its own, so `sable --help' is Sable's help); give the heap size when building:
;;;;   sbcl --dynamic-space-size 16GB --non-interactive --load build.lisp

(require "asdf")

(let ((here (directory-namestring (or *load-truename* *default-pathname-defaults*))))
  (asdf:load-asd (merge-pathnames "sable.asd" here))
  (asdf:load-system "sable/tests")
  (let* ((args (rest sb-ext:*posix-argv*))        ; SBCL removes the options it processed
         (core-p (member "core" args :test #'string=))
         (out (merge-pathnames (if core-p "sable.core" "sable") here)))
    ;; check the image before saving it
    (unless (uiop:symbol-call '#:sable.tests '#:run-tests)
      (format *error-output* "~&build: tests failed, no image written~%")
      (sb-ext:exit :code 1))
    ;; forget where the sources were, so the image does not try to reload them
    (asdf:clear-configuration)
    (format t "~&build: writing ~A~%" out)
    (if core-p
        (sb-ext:save-lisp-and-die out :executable nil)
        (sb-ext:save-lisp-and-die out :executable t :toplevel (symbol-function (uiop:find-symbol* '#:main '#:sable.cli))
                                      :save-runtime-options t))))
