Batched Linear Algebra
======================

Parthenon provides a small set of dense linear algebra routines in
``src/batched_linear_algebra/`` (namespace
``parthenon::batched_linear_algebra``). They are meant for many small,
independent problems, for example one matrix per cell or per meshblock. Each
routine can be called from host code, by a single thread in a flat kernel, or
by a Kokkos team in a hierarchical kernel (see `Ways to call the routines`_).

.. note::
   These routines are designed for small matrices (up to a few tens of rows
   and columns). They are not a replacement for a distributed or vendor
   BLAS/LAPACK for large problems.

Available decompositions
------------------------

All routines work in place and overwrite the input matrix. Optional output
factors are passed as pointers and may be ``nullptr`` if they are not
needed.

* ``QRDecomposition::execute(tm, pA, pQ, scratch)``: Householder QR of an
  :math:`m \times n` matrix with :math:`m \geq n`. On exit ``*pA`` holds the
  upper-trapezoidal :math:`R`, and all entries below the diagonal are exactly
  zero. ``*pQ`` may be a full (:math:`m \times m`) or thin
  (:math:`m \times n`) matrix.
* ``LQDecomposition::execute(tm, pA, pQ, scratch)``: LQ of a wide matrix
  (:math:`m \leq n`), computed by applying QR to the transpose.
* ``SquareSVD::execute(tm, pA, pU, pV, sings, scratch, iscratch)``: SVD of a
  matrix with :math:`n_{rows} \geq n_{cols}`, computed by Householder
  bidiagonalization followed by implicit-shift QR iteration. The singular
  values are returned non-negative but are not sorted.
* ``SymmetricEVD::execute(tm, pA, pQ, eigs, scratch, iscratch)``: eigen
  decomposition of a symmetric matrix, computed by Householder
  tridiagonalization followed by implicit QR. The eigenvalues are not sorted.

Linear solvers
--------------

The solvers overwrite both the matrix and the right-hand sides. They do not
check for singular or rank-deficient matrices, which produce inf/NaN in the
solution.

* ``QRSolve::execute(tm, pA, pB, scratch)``: solves :math:`A X = B` for an
  :math:`m \times n` matrix with :math:`m \geq n` and an :math:`m \times k`
  ``*pB``, giving the least-squares solution when :math:`m > n`. Each
  Householder reflector is applied to :math:`B` as it is built, so :math:`Q` is
  never formed. On exit the first :math:`n` rows of ``*pB`` hold :math:`X`, and
  the remaining rows hold the trailing rows of :math:`Q^T B`, whose column norms
  are the least-squares residuals. The upper triangle of ``*pA`` holds
  :math:`R`, and the entries below the diagonal are unspecified.
* ``QRSolveRight::execute(tm, pA, pB, scratch)``: solves :math:`X A = B` for
  an :math:`n \times m` matrix with :math:`n \leq m` and a :math:`k \times m`
  ``*pB``, by applying ``QRSolve`` to the transposed problem. On exit the first
  :math:`n` columns of ``*pB`` hold :math:`X`.
* ``TriangularSolve::execute(tm, R, pB)``: back substitution for
  :math:`R X = B` with :math:`R` upper triangular. Only the upper triangle of
  the leading :math:`n \times n` block of ``R`` is read, and the first
  :math:`n` rows of ``*pB`` are overwritten with :math:`X`. It needs no
  workspace.

The workspace size functions of ``QRSolve`` and ``QRSolveRight`` take the
dimensions of ``A`` and the number of right-hand sides,
``double_scratch_size(nrows, ncols, nrhs)``.

Row selection
-------------

* ``Maxvol::execute(tm, A, pB, I, scratch, initialize_indices, tau,
  max_iters)``: for an :math:`n \times r` matrix ``A`` with :math:`n \geq r`,
  finds :math:`r` rows ``I`` whose submatrix is :math:`\tau`-dominant, meaning
  every entry of :math:`B = A A(I,:)^{-1}` is at most :math:`\tau` in absolute
  value. Such a submatrix has close to the largest :math:`|\det|` of all
  :math:`r \times r` submatrices. ``A`` is only read. On exit ``*pB``
  (:math:`n \times r`) holds :math:`B`, and ``I`` (an ``int`` array of length
  :math:`r`) holds the rows.

  If ``initialize_indices`` is true (the default), the starting rows are chosen
  greedily by eliminating the largest entry of each column in turn. Otherwise
  ``I`` must hold a starting set with :math:`A(I,:)` nonsingular, which allows
  warm starts. Each iteration finds the largest :math:`|B(i,j)|`, replaces row
  ``I[j]`` with row :math:`i`, and updates :math:`B` by a rank-1 correction, until
  :math:`\max |B| \leq \tau` (default 1.05; it should be greater than 1) or
  ``max_iters`` (default 100) swaps. The return value is the number of swaps.
  Columns of a wide matrix can be selected by passing its transpose through
  ``matrix_transpose_view_t``. Rank-deficient ``A`` is not checked for.

Low-rank approximation
----------------------

* ``MatrixCross::execute(tm, A, pC, pR, I, J, rank, scratch,
  initialize_indices, max_sweeps, tau)``: rank-:math:`r` cross approximation
  :math:`A \approx C A(I,:)` of an :math:`n \times m` matrix, with
  :math:`C = A(:,J) A(I,J)^{-1}`. The approximation is exact if
  :math:`\operatorname{rank} A = r`. Only the :math:`(n + m) r` entries of
  :math:`A(:,J)` and :math:`A(I,:)` are read per sweep, so ``A`` can be any type
  satisfying the matrix requirements below, including a functor that computes
  entries on demand.

  Each sweep orthogonalizes the column fiber :math:`A(:,J)` by QR and selects
  ``I`` with ``Maxvol`` on its :math:`Q`, then does the same on
  :math:`A(I,:)^T` to select ``J``. Each ``Maxvol`` call is warm-started from
  the current indices, and the sweeps stop once a sweep changes neither index
  set, or after ``max_sweeps`` (default 10). On exit ``*pC``
  (:math:`n \times r`) holds :math:`C`, and ``I`` and ``J`` (``int`` arrays of
  length :math:`r`) hold the indices. ``pR`` may be ``nullptr`` (with a pointer
  type, e.g. ``static_cast<decltype(pC)>(nullptr)``). If it is not null,
  ``*pR`` (:math:`r \times m`) is filled with :math:`A(I,:)`.

  If ``initialize_indices`` is true (the default), ``J`` starts as :math:`r`
  evenly spaced columns. Otherwise ``I`` and ``J`` are a warm start, for example
  from a previous call, with :math:`A(I,J)` nonsingular. The return value is the
  number of sweeps. The rank is fixed, and no error estimate is made.

``SquareSVD`` and ``SymmetricEVD`` return the number of QR iterations they
performed. ``Maxvol`` returns the number of row swaps and ``MatrixCross`` the
number of sweeps. ``QRDecomposition``, ``LQDecomposition`` and the solvers
return 0.

Workspace
---------

Besides the matrices themselves, each routine needs a ``double`` workspace
and, for ``SquareSVD`` and ``SymmetricEVD``, a ``std::size_t`` workspace.
Each class reports the required lengths through ``double_scratch_size(...)``
and ``sizet_scratch_size(...)``. ``total_shmem_scratch_size(...)`` returns
the number of bytes to request from ``par_for_outer`` for both workspaces.

Matrix types
------------

The routines are templated on the matrix type. Any type works if it provides
``operator()(int r, int c)`` and has ``GetNrows``/``GetNcols`` overloads
that can be found by argument-dependent lookup. Rank-2 Kokkos views work
directly. ``matrix_wrapper_t<T>`` wraps a row-major pointer, for example
into scratch memory, and ``matrix_transpose_wrapper_t`` and the
permuted-row/column wrappers in ``matrix_utils.hpp`` provide views without
copies.

Ways to call the routines
-------------------------

The first argument of ``execute``, ``tm``, is the execution handle. It
selects one of three ways to use the library.

On the host
~~~~~~~~~~~

Host-only overloads omit ``tm`` and the workspace arguments and allocate
their own workspace. They are convenient for setup code and testing:

.. code-block:: cpp

   using namespace parthenon::batched_linear_algebra;
   QRDecomposition::execute(&A, &Q); // host only, allocates scratch
   QRDecomposition::execute(&A);     // R only

In a flat kernel (one thread per matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``serial_tm_t()`` runs the whole decomposition on the calling
thread, so it can be called from an ordinary ``par_for`` with one matrix per
iteration. This suits large batches of very small matrices, where one thread
per matrix already exposes enough parallelism. There is no team scratch in a
flat kernel, so the matrices and workspace live either in per-thread local
arrays, when the sizes are known at compile time, or in slices of
preallocated device arrays:

.. code-block:: cpp

   using namespace parthenon::batched_linear_algebra;
   constexpr int m = 6, n = 3;

   parthenon::par_for(
       "BatchedQRFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
         constexpr std::size_t kWorkSize = QRDecomposition::double_scratch_size(m, n);
         double a_data[m * n], q_data[m * n], work[kWorkSize];
         matrix_wrapper_t<double> A(a_data, m, n);
         matrix_wrapper_t<double> Q(q_data, m, n);

         // Fill A for this batch entry, e.g. from cell data
         for (int r = 0; r < m; ++r) {
           for (int c = 0; c < n; ++c) {
             A(r, c) = data(b, r, c);
           }
         }

         QRDecomposition::execute(serial_tm_t(), &A, &Q, work);

         // A now holds R and Q holds the thin Q factor; use them here
       });

In a hierarchical kernel (one team per matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing a ``team_mbr_t`` spreads each decomposition over the threads of a
team, which suits larger matrices or smaller batches. All threads of the team
must call ``execute`` together. The matrices and workspace would usually be
allocated in team scratch. ``execute`` does not end with a team barrier, so
add one before reading the results:

.. code-block:: cpp

   using namespace parthenon::batched_linear_algebra;
   using parthenon::ScratchPad1D;
   using parthenon::ScratchPad2D;
   const int m = 12, n = 6;
   const std::size_t nscratch = QRDecomposition::double_scratch_size(m, n);
   // Workspace plus scratch for A (m x n) and a thin Q (m x n)
   const std::size_t scratch_bytes = QRDecomposition::total_shmem_scratch_size(m, n) +
                                     2 * ScratchPad2D<double>::shmem_size(m, n);
   constexpr int scratch_level = 0;

   parthenon::par_for_outer(
       "BatchedQR", scratch_bytes, scratch_level, 0, nbatch - 1,
       KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
         ScratchPad1D<double> work(member.team_scratch(scratch_level), nscratch);
         ScratchPad2D<double> A(member.team_scratch(scratch_level), m, n);
         ScratchPad2D<double> Q(member.team_scratch(scratch_level), m, n);

         // Fill A for this batch entry, e.g. from cell data
         parthenon::par_for_inner(member, 0, m - 1, 0, n - 1,
                                  [&](const int r, const int c) { A(r, c) = data(b, r, c); });
         member.team_barrier();

         QRDecomposition::execute(member, &A, &Q, work.data());
         member.team_barrier();

         // A now holds R and Q holds the thin Q factor; use them here
       });
