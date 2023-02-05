.. _sphinx-doc:

.. _Sphinx CheatSheet: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html

How to Use Sphinx for Writing Docs
===================================

How to have github build your documentation for you
----------------------------------------------------

Github can automatically build your documentation for you through the continuous integration pipeline.
After you submit a pull request with your .rst changes for documentation on `github-pages`_, 
the documentation will automatically get built. You will see a "build and deploy documentation" job
at the bottom of the pull request page. If this passes, your documentation will have been generated.

On the bottom left of the documentation page on `github-pages`_, you can select the branch/build
of the documentation, one of which should be the branch you wrote your changes on.

.. _github-pages: https://parthenon-hpc-lab.github.io/parthenon


Building documentation locally
------------------------------

While you can rely on the CI to build the documentation associated with your
branch, you can also very easily build sphinx documentation locally through
python. These instructions also *do not* require admin access and are usable
with shared machines or python distributions.

First, ensure that you are running a modern version of python (i.e. python 3 of
some flavor)

.. code-block:: bash

   $ python --version
   Python 3.9.7

Then, use pip to install :code:`spinx` and the RTD theme

.. code-block:: bash

   pip install --user sphinx sphinx-rtd-theme

Now, navigate to the :code:`../doc/sphinx` directory where a :code:`make help`
shows all of the available ways to build the documentation

.. code-block:: bash

   $ make help
   Sphinx v4.2.0
   Please use `make target' where target is one of
     html        to make standalone HTML files
     dirhtml     to make HTML files named index.html in directories
     singlehtml  to make a single large HTML file
     pickle      to make pickle files
     json        to make JSON files
     htmlhelp    to make HTML files and an HTML help project
     qthelp      to make HTML files and a qthelp project
     devhelp     to make HTML files and a Devhelp project
     epub        to make an epub
     latex       to make LaTeX files, you can set PAPER=a4 or PAPER=letter
     latexpdf    to make LaTeX and PDF files (default pdflatex)
     latexpdfja  to make LaTeX files and run them through platex/dvipdfmx
     text        to make text files
     man         to make manual pages
     texinfo     to make Texinfo files
     info        to make Texinfo files and run them through makeinfo
     gettext     to make PO message catalogs
     changes     to make an overview of all changed/added/deprecated items
     xml         to make Docutils-native XML files
     pseudoxml   to make pseudoxml-XML files for display purposes
     linkcheck   to check all external links for integrity
     doctest     to run all doctests embedded in the documentation (if enabled)
     coverage    to run coverage check of the documentation (if enabled)
     clean       to remove everything in the build directory

Making the documentation will create a new directory, :code:`_build` in the
:code:`sphinx` directory along with whichever type of documentation you wanted
to build.

For example, building the HTML documentation with :code:`make html` produces the
:code:`../doc/sphinx/_build/html` directory with an :code:`index.html` file that
you can point a browser to in order to view the documenation.


How to Get the Dependencies
---------------------------

Using Docker
^^^^^^^^^^^^

If you are using `Docker`_, then simply pull the docker image specified below:

.. _Docker: https://www.docker.com

.. code-block::

  image: sphinxdoc/sphinx-latexpdf

Then, after running :code:`docker run -it <docker-image-name> /bin/bash`, install the theme we are using with :code:`pip install sphinx_rtd_theme`

More Info.
----------

* `Sphinx Installation`_

.. _Sphinx Installation: https://www.sphinx-doc.org/en/master/usage/installation.html

* `Sphinx reStructuredText Documentation`_

.. _Sphinx reStructuredText Documentation: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
