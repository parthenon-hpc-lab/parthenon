.. _containers:

Containers
==========

``MeshBlockData``
-----------------

A ``MeshBlockData`` container owns ``Variable``\ s. Each
``Variable`` is named, and the ``MeshBlockData`` container knows about
various types of ``Variable``\ s, such as whether the ``Variable`` is
for cell-centered data, face-centered data, sparse data, or dense data.
(For more details on anonymous variables, see :ref:`here <metadata>`\ .)
``Variable``\ s in a ``MeshBlockData`` container can be different
shapes, e.g., scalar, tensor, etc. A ``Variable`` can be added to a
``MeshBlockData`` container as:

.. code:: c++

   parthenon::MeshBlockData.Add(name, metadata, shape)

where the name is a string, the metadata is a ``std::vector`` of
metadata flags, and the shape is a ``std::vector`` of integers
specifying the dimensions.

Note that if a location, such as ``Metadata::Cell`` or
``Metadata::Face``, then shape is the shape of the variable at a given
point. If you have a scalar, you do not need to specify the shape in
``Add``. If you want a vector field with 3 components, you would use

.. code:: c++

   shape = std::vector<int>({3});

and if you wanted a 3x3 rank-2 tensor field, you would use

.. code:: c++

   shape = std::vector<int>({3,3});

If you *do not* specify a location on the grid which the variable lives,
then shape is the shape of the full array. I.e., a ``11x12x13x14`` array
would be added with a shape of

.. code:: c++

   shape = std::vector<int>({11,12,13,14});

It is often desirable to extract from a ``MeshBlockData`` container a
specific set of ``Variable``\ s that have desired names, sparse ids, or
conform to specific metadata flags. This set of ``Variable``\ s must be
collected in such a way that it can be accessed easily and performantly
on a GPU, and such that one can index into the collection in a known
way. This capability is provided by *``VariablePack``\ s*.

To extract a ``VariablePack`` for ``Variable``\ s with a set of names,
call

.. code:: c++

   meshblock_data.PackVariables(names, map)

where ``names`` is a ``std::vector`` of strings and ``map`` is an
instance of a ``parthenon::PackIndexMap``. This will return a
``VariablePack`` object, which is essentially a ``Kokkos::View`` of
``parthenon::ParArray3D``\ s. The map will be filled by reference as a
map from ``Variable`` names to indices in the ``VariablePack``.

Similar methods are available for metadata and sparse IDs:

.. code:: c++

   meshblock_data.PackVariables(metadata, ids, map)
   meshblock_data.PackVariables(metadata, map)

If you would like all ``Variable``\ s in a ``MeshBlockData`` container,
you can ommit the metadata or name arguments:

.. code:: c++

   meshblock_data.PackVariables(map)

If you do not care about indexing into ``Variable``\ s by name, you can
ommit the ``map`` argument in any of the above calls.

For examples of use, see
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/unit/test_meshblock_data_iterator.cpp>`__.

Coordinates
~~~~~~~~~~~

Variable packs contain a ``coords`` object, which is the coordinates on
the meshblock for which the variables were packed. It can be accessed
via

.. code:: c++

   pack.GetCoords()

or

.. code:: c++

   pack.GetCoords(i);

for any ``i``. This latter API is used for consistency with
``MeshBlockPack``\ s.

``MeshData`` and ``MeshBlockPack``\ s
-------------------------------------

``Kokkos`` kernel launches come with an overhead (e.g., about 6
microsecond on a V100). For small kernels that perform little work
(e.g., because of the simplicity of the kernel itself or the small
number of cells per ``MeshBlock``, say 163 or smaller), this can be a
performance bottleneck when each kernel is launched per ``MeshBlock``.
Parthenon therefore provides the capability to combine variables into a
single data structure that spans some number of meshblocks, the
``MeshBlockPack``.

``MeshBlockPack``\ s created automatically and accessed transparently
through ``MeshData`` objects. These ``MeshData`` objects are stored as a
``DataCollection`` of shared pointers in the ``Mesh`` object.

*IMPORTANT*, ``MeshData`` and ``MeshBlockPack`` are considered to be
higher level representations of lower level data, i.e., the data used in
the simulation itself always needs to be registered as ``MeshBlockData``
first before it can be accessed through ``MeshData`` and
``MeshBlockPacks``.

Registering ``MeshData``
~~~~~~~~~~~~~~~~~~~~~~~~

``MeshData`` is a lightweight object that aggregates multiple
``MeshBlock``\ s. Therefore, it needs to be setup/registered with some
number of ``MeshBlock``\ s (at least one and at most all), which is
referred to as partitioning.

The ``Partition`` machinery is implemented in
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/src/utils/partition_stl_containers.hpp>`__.

Registration and partitioning can be controlled manually or
automatically (recommended in multi-stage drivers).

Manual registration
^^^^^^^^^^^^^^^^^^^

The following steps (used in the ``calculate_pi`` example
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/calculate_pi/pi_driver.cpp>`__ ) are needed to
manually register and fill a ``MeshData`` object.

.. code:: cpp

   // Number of MeshBlocks per Partition
   const int pack_size = pmesh->DefaultPackSize();
   // Partition all blocks of the Mesh into separate partitions containing pack_size MeshBlocks
   auto partitions = partition::ToSizeN(pmesh->block_list, pack_size);
   // Register a MeshData object for each partition (collection of blocks) using the partition
   // number as label and containing references to the data stored in the "base" MeshBlockPack
   for (int i = 0; i < partitions.size(); i++) {
     const std::string label = std::to_string(i);
     auto mesh_data = pmesh->mesh_data.Add(label);
     // assign MeshBlocks of partitions[i] and data stored in "base" MeshBlockPack to MeshData object
     mesh_data->Set(partitions[i], "base");
   }

There are two partitioning functions:

.. code:: c++

   // Splits container into N equally sized partitions
   template <typename T, typename Container_t>
   Partition_t<T> ToNPartitions(Container_t<T> &container, const int N);

   // Splits container into partitions of size N
   template <typename T, typename Container_t>
   std::vector<std::vector<T>> ToSizeN(Container_t<T> &container, const int N);

Both functions live within the namespace ``parthenon::partition`` and
``Partition_t`` is defined as:

.. code:: c++

   template<typename T>
   using Partition_t = std::vector<std::vector<T>>

The ``pmesh->DefaultPackSize()`` is controlled via the ``pack_size``
variable in a ``parthenon`` input file under the ``parthenon/mesh``
input block. e.g.,

::

   <parthenon/mesh>
   pack_size = 6

A ``pack_size < 1`` in the input file indicates the entire mesh (per MPI
rank) should be contained within a single pack.

The registered ``MeshData`` can then later be accessed, for example, via
the ``Get(label)`` function:

.. code:: cpp

   auto &md = pmesh->mesh_data.Get(std::to_string(i));

Automatic registration
^^^^^^^^^^^^^^^^^^^^^^

For ease of use, the steps illustrated in the manual registration are
automated in the
``mesh_data.GetOrAdd(string MeshBlockData_label, int partition_id)``
function (e.g., used in the ``advection`` example
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/advection/advection_driver.cpp>`__ ). Here, the
partitioning in the background uses the default Mesh partition size
(``pack_size``) and the the total number of partition is accesse through
``pmesh->DefaultNumPartitions();``. Thus, a sample usage in a driver
that executes Tasks on multiple partitions in parallel may look like

.. code:: cpp

   TaskID no_dependency(0); // no dependency
   const int num_partitions = pmesh->DefaultNumPartitions();
   TaskRegion &single_tasklist_per_partition = tc.AddRegion(num_partitions);
   for (int i = 0; i < num_partitions; i++) {
     auto &tl = single_tasklist_per_partition[i];
     // "base" MeshBlockData of blocks in partition i
     auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
     // MeshBlockData of the previous stage of blocks in partition i
     auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
     // MeshBlockData of the current stage of blocks in partition i
     auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);

     auto my_task = tl.AddTask(no_dependency, MyTaskFunction, mbase, mc0, mc1);
   }

``MeshBlockPack`` Access and Data Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MeshBlockPack`` is indexable as a five-dimensional
``Kokkos::View``. The slowest moving index indexes into a 4-dimensional
``VariablePack``. The next slowest indexes into a ``Variable``. The
fastest three index into the cells on a meshblock. They are accessed
from existing ``MeshData`` objects.

For example:

.. code:: c++

   // MeshData object must exists (see Registering above)
   auto &meshdata_base = pmesh->mesh_data.Get("base");

   // Pack all "independent" variables (of MeshBlockData)
   std::vector<MetadataFlag> flags({Metadata::Independent});
   auto meshblockpack = in_obj->PackVariables(flags);

   // If access to the "fluxes" of the Variable is required use PackVariableAndFluxes
   //auto meshblockpack = in_obj->PackVariablesAndFluxes(flags);

   auto variablepack = meshblockpack(b); // Indexes into the b'th meshblock
   auto var = meshblockpack(b,n); // Indexes into the n'th variable on the b'th MB
   // The n'th variable in the i,j,k'th cell of the b'th meshblock
   Real r = meshblockpack(b,n,k,j,i);

For convenience, ``MeshBlockPack`` also includes the following methods
and fields:

.. code:: c++

   // An accessor method for the coords object on each meshblock
   auto &coords = meshblockpack.GetCoords(m); // gets the Coordinates_t object on the m'th MB

   // The dimensionality of the simulation. Will be 1, 2, or 3.
   // This is needed because components of the flux vector
   // are only allocated for dimensions in use.
   int ndim = meshblockpack.GetNdim();

   // Get the sparse index of the n'th sparse variable in the pack.
   int sparse = meshblockpack.GetSparse(n);

   // The size of the n'th dimension of the pack
   int dim = meshblockpack.GetDim(n);

For an example using all these methods see the ``FluxDivergence``
function in `update.cpp <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/src/interface/update.cpp>`__.

Type
~~~~

The types for packs are:

.. code:: c++

   MeshBlockVarPack<T>

and

.. code:: c++

   MeshBlockVarFluxPack<T>

which correspond to packs over ``MeshBlock``\ s that contain just
variables or contain variables and fluxes.
