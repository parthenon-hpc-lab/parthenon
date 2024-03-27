.. _solvers:

Solvers
=======

Parthenon does not yet provide an exhaustive set of solvers. Currently,
a few basic building blocks are provided and we hope to develop more
capability in the future.

Stencil
-------

This class provides a very simple and efficient means of storing a
sparse matrix with the special form that every row has identical entries
relative to the matrix diagonal. A good example of this is in the
straightforward finite difference discretization of the Poisson equation
(see `here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/poisson/poisson_package.cpp>`__ for example
usage). The ``Stencil`` object is extremely efficient at storing these
sparse matrices because it only has to store the matrix values and
offsets from the diagnonal for a single row. The ``Stencil`` class
provides member functions to compute matrix vector products (``MatVec``)
and Jacobi iterates (``Jacobi``). Both are designed to be called from
within kernels and operate on a single matrix row at a time.

SparseMatrixAccessor
--------------------

This is a helper class that allows one to store a more general sparse
matrix than ``Stencil`` provides. Like ``Stencil``, the
``SparseMatrixAccessor`` class assumes that the location of the nonzero
matrix elements have fixed offsets from the diagonal in every row. Here,
though, the values of the matrix elements can be different from row to
row. The sparse matrix itself can be stored in a normal
:ref:`cell var` with the number of components
equal to the number of nonzero elements in a row of the matrix. The
``SparseMatrixAccessor`` class than associates each of these components
with a particular matrix element. Like ``Stencil``, the
``SparseMatrixAccessor`` class provides ``MatVec`` and ``Jacobi`` member
functions. A simple demonstration of usage can be found in the `Poisson
example <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/poisson/poisson_package.cpp>`__.
