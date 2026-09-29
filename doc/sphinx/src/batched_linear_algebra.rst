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

``SquareSVD`` and ``SymmetricEVD`` return the number of QR iterations they
performed. ``QRDecomposition`` and ``LQDecomposition`` return 0.

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
         double a_data[m * n], q_data[m * n];
         double work[QRDecomposition::double_scratch_size(m, n)];
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
