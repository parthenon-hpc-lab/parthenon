.. _sparse_packs:

Packing Variables
=================

Starting from a high level, at the beginning of a simulation a Parthenon based code defines fields on the ``Mesh`` by adding them to a ``StateDescriptor``. The ``StateDescriptor``\ s for all packages are then passed to the ``Mesh`` constructor. A constructed ``Mesh`` includes  a collection of ``MeshBlock``\ s that cover the domain of the ``Mesh``. The ``MeshBlock``\ s contain information about the coordinate region they cover, their relationship to neighboring ``MeshBlock``\ s, and *importantly* a container holding ``MeshBlockData`` objects. These ``MeshBlockData`` objects in turn hold ``Variable`` objects corresponding to the fields defined on the ``Mesh`` [1]_. Each of the ``MeshBlockData`` objects stored in a given ``MeshBlock`` is labeled as a *stage*. By construction, every ``MeshBlock`` in a Parthenon contains the same stages. Memory for storing a field on a block on a stage is only actually allocated (on device) and held within a ``Variable`` object in a ``ParArray`` [2]_.   Putting the storage for fields in a separate object from ``MeshBlock``, the ``MeshBlockData``, easily allows for multiple storage locations for a given field on a given block within the mesh, e.g. to store multiple Runge-Kutta stages. Further, for performance reasons, downstream Parthenon codes generally should work with ``MeshData`` objects, which hold pointers to groups of ``MeshBlockData`` objects across different blocks but on the same stage. 

As a result of this somewhat complicated structure, it is impractical to access variable storage by following through these different objects in a downstream code [3]_. Therefore, Parthenon defines ``SparsePack``\ s and ``SwarmPack``\ s to allow seamless access to variables within compute kernels. Essentially, packs are objects that contain a ``ParArray`` of references to the ``ParArray``\ s stored within ``Variable`` over a chosen set of fields on a given set of blocks. Said differently, given a ``SparsePack pack`` built from ``MeshData md`` and a set of fields ``var1``, ``var2``,... , a sparse pack allows one to access the field ``var1`` on block ``b`` of ``md`` at position ``(k, j, i)`` using syntax like

.. code:: c++

   Real &my_val = pack(b, var1_t(), k, j, i); // Pull out a reference to the value of var1 on block b in cell (k, j, i)
   ParArray3D<Real> var1 = pack(b, var1_t()); // Pull out a reference to the 3D array containing var1 on block b 

etc. within a kernel.

``SparsePack``\ s *work for all types of variables (both dense and sparse). They were originally implemented to support sparse variables and to supersede the older* ``VariablePack``\ s *and* ``VariableFluxPack``\ s *and picked up the* ``Sparse`` *modifier to differentiate them. The latter have not been removed from ``Parthenon`` because some downstream codes still rely on them, but they are deprecated and will be removed eventually.*

*If you want to deal with particle fields, you will need to use* ``SwarmPack``\ *s, which are described at* :ref:`swarm_packs`.

Type-based Packing
------------------

Parthenon provides functionality for accessing fields in a pack via a type that is associated with a field name [4]_. As an example, if a downstream code includes a field with the name ``"var1"`` the code could also define a type

.. code:: c++

  struct var1_t : public parthenon::variable_names::base_t<false> {            
    template <class... Ts>                                                      
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
    static std::string name() { return "var1"; }                      
  }
 
which inherits from ``parthenon::variable_names::base_t`` [5]_. Rather than write this boilerplate out for every variable name in a downstream code, it is often easier to define a macro that expands to this class for a given variable name argument. Additionally, fields can be added to a ``StateDescriptor`` using the variable name type directly using

.. code:: c++

  StateDescriptor pkg;
  Metadata metadata;
  pkg->AddField<var1_t>(metadata);

The advantage of using types instead of just strings to denote field names is that the types are accessible within kernels on device. Under the hood, ``SparsePack::operator()`` is overloaded on each of the type list of variable name types used to create the pack, so an instance of the variable name type can be used to access desired field within a pack.

Building and Using a ``SparsePack``
-----------------------------------

``SparsePack``\ s are built in two stages, first a ``PackDescriptor`` is built that defines the set of fields to include in the pack using one of the overloaded ``MakePackDescriptor`` functions. Then ``PackDescriptor::GetPack(...)`` is called on a given ``MeshData`` or ``MeshBlockData`` object to return an actual pack. In practice, this will look like 

.. code:: c++

   parthenon::TaskStatus my_task(MeshData *md) {
     // Pull out indices, etc.
     std:::vector<MetadataFlags> md_flags; // Optional argument below
     std::set<PDOpt> options{PDOpt::WithFluxes}; // Optional argument below
     auto desc = parthenon::MakePackDescriptor<var1_t, var2_t>(pmd, md_flags, options);
     auto pack = desc.GetPack(md);
     parthenon::par_for(/*index ranges, etc. go here */,
         KOKKOS_LAMBDA(int b, int k, int j, int i) {
       for (int c = 0; c < ncomponents_2; ++c)
         pack(b, var1_t(), k, j, i) += 2.0 * pack(b, var2_t(c), k, j, i);
       pack.flux(b, X1DIR, var1_t(), k, j, i) = pack(b, var1_t(), k, j, i) * pack(b, var2_t(), k, j, i);
     });
     return TaskStatus::complete;
   }

``PackDescriptor``\ s can be somewhat expensive to build because they require searching through all fields in simulation. Therefore, they are automatically cached in the ``StateDescriptor`` where possible. Additionally, it is often possible to declare ``PackDescriptors`` that are created in task functions to be ``static``.  

``PackDescriptor`` takes a ``std::set`` of ``PDOpt`` options to determine what to include in the pack:

.. list-table:: Pack Descriptor Options
   :widths: 8 20
   :header-rows: 0

   * - ``PDOpt::WithFluxes``
     - Fluxes associated with variables in the pack are included in the pack and accessible through ``pack.flux(...)``
   * - ``PDOpt::Coarse``
     - Pack the coarse buffers for the fields rather than the normal resolution buffers.
   * - ``PDOpt::Flatten``
     - Packs all blocks across all fields into the variable index so that the pack looks like it has a single block.

``SparsePack::operator()``
^^^^^^^^^^^^^^^^^^^^^^^^^^
There are a number of different overloads for ``operator()`` in sparse packs that allow accessing field data:

* ``template <class var_t> Real &operator()(int b, TopologicalElement te, const var_t &t, int k, int j, int i)`` and ``template <class var_t> ParArray3D operator()(int b, TopologicalElement te, const var_t &t)`` : Returns the value on block ``b`` for topological element ``te`` of ``var_t``. The first call returns a reference to the value at position ``(k, j, i)`` while the second returns a ``ParArray3D`` (which obeys reference semantics) containing that component of the field on the block. ``var_t`` must be in the list of types used to create the pack.
*  ``template <class var_t> Real &operator()(int b, const var_t &t, int k, int j, int i)`` and ``template <class var_t> Real &operator()(int b, const var_t &t)``: Same as above, but with the topological element defaulted to cell-centered.
* ``Real &operator()(int b, TopologicalElement te, int idx, int k, int j, int i)`` and ``Real &operator()(int b, int idx, int k, int j, int i)`` and ``Real &operator()(int b, TopologicalElement te, int idx)`` and ``Real &operator()(int b, int idx)``: Same as above, but directly accesses the field at position ``idx`` in the pack. This should be used with the bounds returned from ``SparsePack::GetLowerBound(...)`` and ``SparsePack::GetUpperBound(...)``. 
* ``Real &operator()(int b, TopologicalElement te, PackIdx idx, int k, int j, int i)`` and ``Real &operator()(int b, PackIdx idx, int k, int j, int i)`` and ``Real &operator()(int b, TopologicalElement te, PackIdx idx)`` and ``Real &operator()(int b, PackIdx idx)``: Same as above, but access the field using ``PackIdx idx``. This only works for packs that were built using a list of names (as opposed to a list of types), see [4]_.  

Other ``SparsePack`` Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``Coordinates_t &GetCoordinates(const int b = 0)``: Returns a reference to the coordinates object associated with block ``b``.
* ``template <class var_t> int GetLowerBound(int b, var_t)``: Returns the first index in the pack where a field corresponding to ``var_t`` is stored. Returns ``-1`` if ``var_t`` is not allocated on block ``b``. A similar functions exist for ``PackIdx``. 
* ``template <class var_t> int GetUpperBound(int b, var_t)``: Returns the last index in the pack where a field corresponding to ``var_t`` is stored. Returns ``-2`` if ``var_t`` is not allocated on block ``b``. A similar functions exist for ``PackIdx``.
* ``int GetLevel(int b, int off3, int off2, int off1)``: Returns the logical level of neighbor block(s) of block ``b`` offset in direction ``(off3, off2, off1)``.
* ``bool IsPhysicalBoundary(int b, int off3, int off2, int off1)``: Returns if block ``b`` has a physical boundary in offset direction ``(off3, off2, off1)``. 
* ``template <class var_t> bool Contains(const int b, const var_t t)``: Returns if ``var_t`` is allocated on block ``b``.
* ``template <class var_t> Real &flux(int b, int dir, const var_t &t, int k, int j, int i)``: Gets the flux in direction ``dir`` associated with variable ``var_t``.

``SparsePack``\ s with Sparse Fields
------------------------------------

A given sparse field may or may not be allocated on each block within a pack. To safely access the fields in a given pack, ``SparsePack``\ s provide checks on whether or not a given sparse variable/pool is allocated 

.. code:: c++
  
   parthenon::TaskStatus my_task(MeshData *md) {
     // Pull out indices, etc.
     auto desc = parthenon::MakePackDescriptor<sparse_var_t>(pmd);
     auto pack = desc.GetPack(md);
     parthenon::par_for(/*index ranges, etc. go here */,
         KOKKOS_LAMBDA(int b, int k, int j, int i) {
         // For a single sparse field
         if (pack.Contains(b, sparse_var_t())) {
           // The sparse field is allocated on block b, and so is safe to access
         }
         // This loop will go over all allocated sparse fields in the sparse_var sparse pool
         for (int idx = pack.GetLowerBound(b, sparse_var_t()); idx <= pack.GetUpperBound(b, sparse_var_t()); ++idx) {
           pack(b, idx, k, j, i) = 0.0;
         }
     });
     return TaskStatus::complete;
   }

.. [1] In practice, there are ways of selecting subsets of fields for inclusion in a given ``MeshBlockData`` instance.
.. [2] ``ParArray``\ s are lite wrappers around ``Kokkos::View``\ s and therefore obey reference semantics.
.. [3] Additionally, because many ``std::`` library containers don't work on device, the chain from field name through ``MeshBlockData`` to ``Variable`` to the underlying ``ParArray`` cannot be legally followed within a kernel.
.. [4] It is also possible to use ``SparsePack``\ s created with a vector of string field names. In this case, the ``PackDescriptor`` can return a map from variable name strings to ``PackIdx``\ s using the function ``PackDescriptor::GetMap()``. These indices need to be pulled out from the map outside of the kernel, since ``std::unordered_map`` is unavailable on device. Then inside the kernel, packs are accessed using ``PackIdx var_idx`` in calls to the pack ``pack(b, var_idx, k, j, i)``, etc. 
.. [5] The template arguments for ``parthenon::variable_names::base_t`` are ``<bool REGEX, int... NCOMP>`` where the ``REGEX`` boolean determines if the returned name sure be interpreted as a simple string or as a regex that possibly selects multiple variable names. The variadic integer pack ``NCOMP...`` defines the tensor shape of the field, so that when the ``var1_t(m, l)`` constructor is called the correct component of the field is returned from a call to ``SparsePack::operator()`` (assuming we have a two component field).
