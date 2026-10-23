.. _sphinx-doc:

Boundary Conditions
===================

Built-in boundary conditions
----------------------------

Natively, Parthenon supports three kinds of boundary conditions:

- ``periodic``
- ``outflow``
- ``reflecting``

which are all imposed on variables with the ``Metadata::FillGhost``
metadata flag. To set the boundaries in each direction, set the
``*x*_bc`` variables under ``parthenon/mesh`` in the input file. e.g.,

::

   <parthenon/mesh>
   ix1_bc = outflow
   ox1_bc = outflow

   ix2_bc = reflecting
   ox2_bc = reflecting

   ix3_bc = periodic
   ox3_bc = periodic

sets outflow boundary conditions in the ``X1`` direction, reflecting in
``X2``, and periodic in ``X3``.

User-defined boundary conditions.
---------------------------------

You can provide your own boundary conditions by setting the the
``boundary_conditions`` function pointers in the ``ApplicationInput``
for your ``parthenon_manager``. e.g.,

.. code:: c++

   pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] = MyBoundaryInnerX1;

where ``BoundaryFace`` is an enum defined in ``defs.hpp`` as

.. code:: c++

   // identifiers for all 6 faces of a MeshBlock
   constexpr int BOUNDARY_NFACES = 6;
   enum BoundaryFace {
     undef = -1,
     inner_x1 = 0,
     outer_x1 = 1,
     inner_x2 = 2,
     outer_x2 = 3,
     inner_x3 = 4,
     outer_x3 = 5
   };

You can then set this boundary condition via the ``user`` flag in the
input file:

::

   <parthenon/mesh>
   ix1_bc = user

Boundary conditions so defined should look roughly like

.. code:: c++

   // an implementation of outflow conditions
   void MyBoundaryInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
     std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
     auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
     int ref = bounds.GetBoundsI(IndexDomain::interior).s;
     auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
     auto nb = IndexRange{0, q.GetDim(4) - 1};
     pmb->par_for_bndry(
         "OutflowInnerX1", nb, IndexDomain::inner_x1, coarse,
         KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
           q(l, k, j, i) = q(l, k, j, ref);
         });
   }

Important things to note:

- The signature is a ``std::shared_ptr<MeshBlockData<Real>>`` and a boolean.
- The boolean determines whether the boundary condition is applied over ghost cells on
  a coarse buffer (for mesh refinement) or for a fine buffer.
- You can use the ``MeshBlock::par_for_bndry`` abstraction to loop over the
  appropriate cells by specifying the ``IndexDomain`` for the ghost region
  and whether or not the loop is coarse. For more information on the
  ``IndexDomain`` object, see :ref:`here <domain>`.
- You can pack over all the coarse or fine buffers of a variable with the
  ``PackVariables`` optional ``coarse`` boolean as seen here.

Other than these requirements, the ``Boundary`` object can do whatever
you like. Reference implementations of the standard boundary conditions
are available `here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/src/bvals/boundary_conditions.cpp>`__.


Per package user-defined boundary conditions.
---------------------------------

In addition to user defined *global* boundary conditions, Parthenon also supports 
registration of boundary conditions at the *per package* level. These per package 
boundary conditions are *not* controlled by parameter input in `<parthenon/mesh>`,
and they are always applied after the chosen global boundary conditions have 
been applied during `ApplyBoundaryConditions*`.

A `StateDescriptor` defining a package contains a member `UserBoundaryFunctions`, 
which is an array with an element for each boundary direction consisting of a vector 
of boundary function pointers. When the packages are resolved and associated with 
a `Mesh` object, these boundary conditions are called in the order in which they 
are registered within a package. For example, to register a boundary condition for 
your package you could do something like (see `examples/poisson_gmg/poisson_package.cpp` 
for a more complete example):

.. code:: c++

  template <CoordinateDirection DIR, BCSide SIDE>
  auto GetMyBC() {
    return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
      // Implementation of BC here
    };
  }
  
  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
    ...
    using BF = parthenon::BoundaryFace;
    pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetMyBC<X1DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetMyBC<X2DIR, BCSide::Inner>());
    ...
  }