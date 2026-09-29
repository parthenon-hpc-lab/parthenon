Batched Linear Algebra
======================

Parthenon provides a small set of dense linear algebra routines in
``src/batched_linear_algebra/`` (namespace
``parthenon::batched_linear_algebra``). They are meant for many small,
independent problems, for example one matrix per cell or per meshblock. Each
routine can be called from host code or by a single Kokkos team inside a
kernel, so a batch is processed by launching one team per matrix.

.. note::
   These routines are designed for small matrices (up to a few tens of rows
   and columns) that fit in team scratch memory. They are not a replacement
   for a distributed or vendor BLAS/LAPACK for large problems.

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

Execution contexts and scratch
------------------------------

The first argument ``tm`` selects how the work is parallelized:

* ``serial_tm_t()`` runs serially, for example on the host or inside a flat
  ``par_for``.
* A ``team_mbr_t`` distributes the inner loops over the team members.

Each class provides ``double_scratch_size(...)`` (and ``sizet_scratch_size``
where integer scratch is needed), plus ``total_shmem_scratch_size(...)``,
which returns the number of bytes to request from ``par_for_outer``. For
convenience, host-only overloads that omit ``tm`` and the scratch arguments
allocate their own workspace:

.. code-block:: cpp

   using namespace parthenon::batched_linear_algebra;
   QRDecomposition::execute(&A, &Q); // host only, allocates scratch
   QRDecomposition::execute(&A);     // R only

Matrix types
------------

The routines are templated on the matrix type. Any type works if it provides
``operator()(int r, int c)`` and has ``GetNrows``/``GetNcols`` overloads
that can be found by argument-dependent lookup. Rank-2 Kokkos views work
directly. ``matrix_wrapper_t<T>`` wraps a row-major pointer, for example
into scratch memory, and ``matrix_transpose_wrapper_t`` and the
permuted-row/column wrappers in ``matrix_utils.hpp`` provide views without
copies.

Example: batched QR in a team kernel
------------------------------------

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
