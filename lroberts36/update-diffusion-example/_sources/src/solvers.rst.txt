.. _solvers:

Solvers
=======

Parthenon provides a number of linear solvers, including a geometric multigrid
solver, a CG solver, a BiCGSTAB solver, and multigrid preconditioned versions
of the latter two solvers. 

Solvers are templated on a type defining the system of equations they are solving. 
The type defining the system of equations must provide two methods and a ``TypeList`` 
of all of the fields that make up the vector space:

.. code:: c++

  class MySystemOfEquations {
    using IndependentVars = parthenon::TypeList<var1_t, var2_t>;

    TaskId Ax(TaskList &tl, TaskID depends_on,
              std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
              std::shared_ptr<parthenon::MeshData<Real>> &md_in,
              std::shared_ptr<parthenon::MeshData<Real>> &md_out);

    TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                           std::shared_ptr<parthenon::MeshData<Real>> &md_diag)
  };

The routine ``Ax`` must calculate the matrix vector product ``y <- A.x`` by taking a container
``md_mat`` which contains all of the fields required to reconstruct the matrix ``A`` associated
with the system of linear equations, the container ``md_in`` which will store the vector ``x`` 
in the fields in the typelist ``IndependentVars``, and ``md_out`` which will hold the vector ``y``. 

The routine ``SetDiagonal`` takes the same container ``md_mat`` as ``Ax`` and returns the
(approximate) diagonal of ``A`` in the container ``md_diag``. This only needs to be approximate
since it is only used in preconditioners/smoothers.

With such a class defining a linear system of equations, one can then define and use a solver with 
code along the lines of:

.. code:: c++ 

  std::string base_cont_name = "base";
  std::string u_cont_name = "u";
  std::string rhs_cont_name = "rhs";

  MySystemOfEquations eqs(....);
  std::shared_ptr<SolverBase> psolver = std::make_shared<BiCGSTABSolver<MySystemOfEquations>>(
      base_cont_name, u_cont_name, rhs_cont_name, pin, "location/of/solver_params", eqs);

  ...

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int partition = 0; partition < num_partitions; ++partition) {
    TaskList &tl = region[partition];
    auto &md = pmesh->mesh_data.Add(base_cont_name, partitions[partition]);
    auto &md_u = pmesh->mesh_data.Add(u_cont_name, md);
    auto &md_rhs = pmesh->mesh_data.Add(rhs_cont_name, md);

    // Do some stuff to fill the base container with information necessary to define A
    // if it wasn't defined during initialization or something
    
    // Do some stuff to fill the rhs container

    auto setup = psolver->AddSetupTasks(tl, dependence, partition, pmesh);
    auto solve = psolver->AddTasks(tl, setup, partition, pmesh); 

    // Do some stuff with the solution stored in md_u
  }

Some notes: 

- All solvers inherit from ``SolverBase``, so the best practice is to stash a shared pointer to a 
  ``SolverBase`` object in params during initialization and pull this solver out while building a 
  task list. This should make switching between solvers trivial.
- For any solver involving geometric multigrid, the input parameter 
  ``parthenon/mesh/multigrid`` must be set to ``true``. This tells the ``Mesh``
  to build the coarsened blocks associated with the multi-grid hierarchy.
- For geometric multigrid based solvers, it is possible to define block interior prolongation 
  operators that are separate from the standard prolongation machinery in Parthenon. This allows 
  for defining boundary aware prolongation operators and having different prolongation operators
  in the ghost cells of blocks from the prolongation operators used in their interiors. Users can 
  easily define their own prolongation operators. The prolongation functor is passed as a template 
  argument to the multi-grid solver class. An example of using these interior prolongation
  operators is contained in the ``poisson_gmg`` example.



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

.. _parthenon_gmg:

Geometric Multigrid
-------------------

[This documentation was generated with assistance from generative AI]

Parthenon implements a geometric multigrid (GMG) solver on a forest-of-octrees
adaptive mesh. The multigrid hierarchy is constructed from two independent
coarsening mechanisms:

1. AMR (tree) refinement, indexed by ``logical_level``.
2. In-block geometric coarsening, indexed by ``block_coarsenings``.

The GMG hierarchy is realized explicitly: for each multigrid level,
Parthenon constructs (or reuses) a concrete ``MeshBlock`` for every participating
pair

::

   (LogicalLocation, block_coarsenings).

Each ``MeshBlock`` in the GMG hierarchy therefore corresponds to a specific
location in the refinement tree and a specific internal coarsening level.

Some details about our AMR geometric multi-grid can be found in 
`these notes <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/doc/latex/main.pdf>`_.
Note that they only focus on hierarchies with only two-level composite grids.

AMR Refinement and In-Block Coarsening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMR refinement follows the Athena++ convention:

- Larger ``logical_level`` corresponds to finer spatial resolution.
- For a domain of size :math:`L` in a given direction, the physical block size is

  .. math::

     L_{\text{block}} = \frac{L}{2^{\text{logical\_level}}}.

Changing ``logical_level`` corresponds to moving up or down the octree.

Independently of AMR refinement, geometric coarsening can be applied inside
each block. If a block has base resolution :math:`N^3`, then after
``block_coarsenings = k`` the resolution becomes

  .. math::

     N_{\text{block}} = \frac{N}{2^k}.

This reduces the number of zones per block without changing its physical
extent.


Grids and GridIdentifier
~~~~~~~~~~~~~~~~~~~~~~~~

Each multigrid level is described by a ``GridIdentifier``. A
``GridIdentifier`` records the structural properties of a grid:

- the grid type (leaf or two-level composite),
- the number of ``block_coarsenings``,
- the ``logical_level`` (for two-level composite grids),
- the ``multigrid_level`` index,
- whether the grid participates in the GMG hierarchy.

The ``GridIdentifier`` does not contain pointers to blocks and does not own
data. It is purely a description of which blocks belong to a given multigrid
level. The actual ``MeshBlock`` instances for a grid are stored separately.


Leaf Grids
~~~~~~~~~~

A leaf grid consists of all leaf nodes in the AMR refinement tree, at a fixed
value of ``block_coarsenings``.

In an adaptive mesh, leaf nodes may exist at multiple ``logical_level`` values.
A leaf grid therefore contains blocks spanning potentially many AMR levels.
However, all blocks in a leaf grid share the same in-block coarsening level.

Equivalently, a leaf grid is the collection of all AMR leaf logical locations,
each paired with a single specified value of ``block_coarsenings``.


Two-Level Composite Grids
~~~~~~~~~~~~~~~~~~~~~~~~~

A two-level composite grid is anchored at a specific AMR level
``logical_level = ℓ`` and contains, at a fixed ``block_coarsenings``:

- all leaf blocks at level ``ℓ``,
- all internal-node blocks at level ``ℓ`` that are parents of finer leaf blocks,
- all leaf blocks at level ``ℓ − 1``, which provide boundary conditions for
  the level-``ℓ`` blocks.

Two-level composite grids are required to correctly couple coarse and fine
regions of the AMR mesh while preserving geometric consistency of the
multigrid operators.


Allowed Multigrid Hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GMG hierarchy is not arbitrary. There is a finite family of admissible
hierarchies determined by the base block size and the parameter

::

   parthenon/mesh/base_block_coarsenings

which specifies the number of initial leaf-only multigrid levels.

The hierarchy always proceeds in three distinct phases.


Initial Leaf-Only Phase
^^^^^^^^^^^^^^^^^^^^^^^

The hierarchy may begin with zero or more leaf-only multigrid levels.
The number of such levels is controlled by the input parameter
``parthenon/mesh/base_block_coarsenings``.

During this phase:

- The grid consists of all AMR leaf logical locations.
- The value of ``block_coarsenings`` increases by one at each successive
  multigrid level.
- The AMR tree structure is not traversed.

The maximum number of such levels is limited by the base block resolution,
since in-block coarsening cannot exceed the number of times the block
dimensions may be divided by two.


Tree-Traversal Phase
^^^^^^^^^^^^^^^^^^^^

After the leaf-only phase, the hierarchy transitions to two-level composite
grids.

In this phase:

- The value of ``block_coarsenings`` remains fixed at the value reached
  at the end of the leaf-only phase.
- The hierarchy moves upward through the AMR tree by decreasing
  ``logical_level`` one level at a time.
- At each step, a two-level composite grid is constructed that couples
  blocks at levels ``ℓ`` and ``ℓ − 1``.

This phase continues until ``logical_level = 0`` is reached.


Root-Level In-Block Coarsening Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the hierarchy reaches the root of the AMR tree (``logical_level = 0``),
further coarsening is achieved exclusively through additional in-block
coarsening.

In this phase:

- The AMR structure remains fixed at the root.
- The value of ``block_coarsenings`` increases by one at each successive
  multigrid level.

At this stage, there is a single logical location per tree, and multigrid
coarsening proceeds purely by reducing the number of zones inside the root
blocks.


Summary of Hierarchy Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every admissible GMG hierarchy in Parthenon has the following structure:

1. Zero or more leaf-only grids at increasing in-block coarsening.
2. A sequence of two-level composite grids that traverse the AMR tree from
   the finest level to the root, at fixed in-block coarsening.
3. Zero or more additional two-level composite grids at the root with
   further in-block coarsening.

For each grid in this sequence, Parthenon constructs (or reuses) the
corresponding ``MeshBlock`` instances for all participating
``(LogicalLocation, block_coarsenings)`` pairs.

This construction cleanly separates AMR refinement from in-block resolution
changes while providing a consistent geometric multigrid hierarchy on
fully adaptive forest-of-octrees meshes.
