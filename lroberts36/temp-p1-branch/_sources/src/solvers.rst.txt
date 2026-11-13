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

Some implementation notes about geometric multi-grid can be found in 
`these notes <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/doc/latex/main.pdf>`_.

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
