.. _sparse_packs:

Packing Variables
=================

Starting from a high level, at the beginning of a simulation a Parthenon based code defines fields on the ``Mesh`` by adding them to a ``StateDescriptor``. The ``StateDescriptor``\ s for all packages are then passed to the ``Mesh`` constructor. A constructed ``Mesh`` includes  a collection of ``MeshBlock``\ s that cover the domain of the ``Mesh``. The ``MeshBlock``\ s contain information about the coordinate region they cover, their relationship to neighboring ``MeshBlock``\ s, and *importantly* a container holding ``MeshBlockData`` objects. These ``MeshBlockData`` objects in turn hold ``Variable`` objects corresponding to the fields defined on the ``Mesh`` [1]_. Each of the
``MeshBlockData`` objects stored in a given ``MeshBlock`` is labeled as a *stage*. By construction, every ``MeshBlock`` in a Parthenon contains the same stages. Memory for storing a field on a block on a stage is only actually allocated (on device) and held within a ``Variable`` object in a ``ParArray`` [2]_.   Putting the storage for fields in a separate object from ``MeshBlock``, the ``MeshBlockData``, easily allows for multiple storage locations for a given field on a given block within the mesh, e.g. to store multiple Runge-Kutta stages. Further, for performance reasons, downstream Parthenon codes generally should work with ``MeshData`` objects, which hold pointers to groups of ``MeshBlockData`` objects across different blocks but on the same stage. 

As a result of this somewhat complicated structure, it is impractical to access variable storage by following through these different objects in a downstream code [3]_. Therefore, Parthenon defines ``SparsePack``\ s and ``SwarmPack``\ s to allow seamless access to variables within compute kernels. Essentially, packs are objects that contain a ``ParArray`` of references to the ``ParArray``\ s stored within ``Variable`` over a chosen set of fields on a given set of blocks. Said differently, given a sparse pack ``pack`` built from ``MeshData md`` and a set of fields ``var1``, ``var2``,... , a sparse pack allows one to access the field ``var1`` on block ``b`` of ``md`` at position ``(k, j, i)`` using syntax like

.. code:: c++

   Real &my_val = pack(b, var1_t(), k, j, i); // Pull out a reference to the value of var1 on block b in cell (k, j, i)
   ParArray3D<Real> var1 = pack(b, var1_t()); // Pull out a reference to the 3D array containing var1 on block b 

etc. within a kernel. 

Type-based Packing
------------------


Although it is not necessary, many downstream codes have found it useful to define classes associated with each field that is registered in a ``StateDescriptor``.



Building a ``SparsePack``
-------------------------
.. code:: c++

   Real &my_val = pack(b, var1(), k, j, i);
   pack(b, var1(), k, j, ) = 1.0;



.. [1] In practice, there are ways of selecting subsets of fields for inclusion in a given ``MeshBlockData`` instance.
.. [2] ``ParArray``\ s are lite wrappers around ``Kokkos::View``\ s and therefore obey reference semantics.
.. [3] Additionally, because many ``std::`` library containers don't work on device, the chain from field name through ``MeshBlockData`` to ``Variable`` to the underlying ``ParArray`` cannot be legally followed within a kernel.
