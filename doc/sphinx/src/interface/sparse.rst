Sparse implementation
=====================

This file describes the implementation of sparse variables in Parthenon.
Parthenon also provides the capability to use **sparse** fields, where
“sparse” really refers to two distinct and orthogonal concepts, namely
(a) a family or pool of distinct and separate variable fields that share
a common base name and are distinguished by a sparse ID (**sparse
naming**), and (b) variables that are only allocated on some but not
necessarily all mesh blocks in the domain (**sparse fields**). In
principle, both of these “sparse” concepts can be used independently,
i.e.  we could make a family of dense variables that share a base name
and are distinguished by sparse IDs, or we could simply have a single
variable with a label without a sparse ID that is only allocated on some
blocks but not others. However, in practice these two concepts are often
used together. As a result, the current implementation of sparse fields
in parthenon only allows fields to be sparse if they are sparsely named.
This should be relaxed in the future. These two concepts are addressed
in separate sections below. ## Sparse Fields

For computation and memory savings, it may be desirable to explicitly
evolve certain fields only on certain sub-regions of the grid. In
regions of the grid where certain fields are not evolved, they are
assumed to take some default value (usually zero) and storage for these
fields does not need to be allocated in those sub-regions. Fields that
are explicitly evolved on only certain sub-regions of the grid are
termed “sparse fields.”

Some examples of where sparse fields might be desirable: - Properties
(density, temperature, etc.) of materials that are only present in
certain parts of the simulation. One does not want to allocate memory
that contains all zeros and perform operations on zeros in regions where
the material is not present. - A self-gravitating fluid in a vacuum.
Often, astrophysics codes will model this by including fluid throughout
the entire simulation domain but set a floor on the density and
temperature of the material to create an effective “atmosphere” around
the actual self-gravitating system. Regions that are set to the floor do
not necessarily need to be explicitly tracked and the fluid quantities
could be treated as sparse fields. - Material that is transitioning
between non-equilibrium nuclear burning and nuclear statistical
equilibrium (NSE). In the non-equilibrium regions, the abundances of all
nuclear species are required to define the composition of the material
while in the regions that are in NSE only one field (the electron
fraction) is required to define the composition.

In Parthenon, sparse fields are implemented at the block level, meaning
that a field is allocated everywhere or nowhere on each block in the
simulation. The set of possible operations for a sparse field are:

-  *Allocation on initialization:* Sparse variables need to be allocated
   by hand in the ``ProblemGenerator`` in regions where they are
   non-zero.
-  *Allocation due to local changes:* For some types of sparse fields
   (e.g. nuclear abundances fields for material transitioning in and out
   of NSE), it may be desirable to allocate them based on the local
   state on the block (e.g. allocate abundances if the temperature is
   anywhere on the block below some threshold). This is an operation
   that needs to be defined in a task by the downstream code, since it
   is problem specific. **Currently, Parthenon does not have this
   capability. It will probably be necessary to define some hooks for
   checking blocks for the necessity of allocation and having some
   internal Parthenon functions for actually performing the
   allocation.**
-  *Allocation due to neighbors:* If a sparse field on a neighbor block
   passes ghost data that is anywhere above the
   ``allocation_threshold``, the receiving block will allocate the
   sparse field if it is unallocated before loading the ghost data.
   ``allocation_threshold`` is set in the ``Metadata`` of a field. A
   detailed description of sparse ghost zone communication is given
   `here <../sparse_boundary_communication.md>`__.
-  *Allocation due to other fields:* In some instances, we may want to
   allocate or deallocate a sparse field when another sparse field is
   allocated or deallocated, not when the field itself changes state
   (i.e. we would like to allocate a sparse field describing a dependent
   variable whenever a corresponding inedendent sparse variable is
   allocated). For this, we introduce the notion of *controlling* and
   *controlled* fields. A controlling field has a list of field names
   associated with it and whenever allocation or deallocation of the
   controlling field occurs on a block, it also goes through the list of
   controlled fields associated with it and allocates or deallocates
   them. To make a controlled sparse field, a sparse pool can be created
   using ``SparsePool(base_name, metadata, controller_base_name, ...)``
   where ``controller_base_name`` is base name of the controlling field.
   Fields associated with this sparse pool will only be allocated or
   deallocated based on the corresponding fields in the
   ``controller_base_name`` sparse pool. **There are no checks performed
   that the two pools contain the same set of sparse indices.**
-  *Initialization after allocation:* Depending on what a particular
   sparse field is modeling, its values on a block may need to be
   initialized to values that are non-zero and/or depend on the state of
   other fields on the block. Since this requirement is problem
   dependent, downstream codes using Parthenon must define tasks that
   perform this initialization. If no task is defined, a newly
   initialized field is everywhere on the block set to the
   ``default_value`` defined in its ``Metadata``. **The implementation
   of this needs to be fleshed out and more extensively tested and this
   description needs more detail about how to register initialization
   functions. Also needs to describe how to check for sub-regions of a
   block where uninitialized data has been passed in. In the downstream
   use cases so far, this has not been a necessary feature.**
-  *Deallocation due to local changes:* Currently, the only way for a
   sparse field to be deallocated is if its absolute value falls below
   ``deallocation_threshold`` everywhere on the interior of a block (or
   if its controlling field is deallocated). The
   ``deallocation_threshold`` is set in the sparse fields metadata.
-  *Deallocation due to other fields:* If a field is controlled, it is
   deallocated whenever its controlling field is deallocated.
-  *Access if allocated:* Since fields are mainly accessed through packs
   in Parthenon based codes (a ``VariablePack``, ``MeshBlockPack``, or
   ``SparsePack<...>``), packs need to carry around information about
   the allocation status of the requested fields in the pack. A “dense
   sparse packing” strategy is used in ``SparsePack<...>``, where only
   allocated fields are included in the index space of the pack. Here,
   the index space means ``(block, field, k, j, i)`` and for sparse
   fields the number of allocated fields can change from block to block
   which means the index space is logically ragged for dense sparse
   packing. For each block, the range of indices corresponding to a
   particular field in the pack can be accessed on device and iterated
   over (if the range has positive size, a negative size for the range
   indicates that none of the corresponding fields are allocated).
   Looping over fields in these type of packs generally requires
   hierarchichal parallelism. Currently, ``VariablePack`` and
   ``MeshBlockPack`` employ a “sparse sparse packing” strategy, where
   all fields are included in the index space of the pack but the
   allocation status of ``(block, field)`` must be checked before
   accessing ``(block, field, k, j, i)`` since this is not guaranteed to
   point to valid memory. \*There is “dense sparse pack” implementation
   of ``VariablePack`` and ``MeshBlockPack`` in the branch
   ``lroberts36/merge-sparse-with-jdolence-sparse`` that is being used
   in ``Riot``. This should probably be brought into ``develop``, since
   the “sparse sparse pack” access pattern is probably not desirable.

In comparison to a sparse field, a dense field only requires the
operation *Access*.

**To set the thresholds for a sparse field, after creating the
``Metadata`` object that will be used for the field, call
``Metadata::SetSparseThresholds(allocation_threshold, deallocation_threshold,  default_value)``.**

Turning off sparse
------------------

The sparse allocation feature can be turned off at run- or compile-time.
The sparse naming feature cannot be turned off.

Run-time
~~~~~~~~

Setting ``enable_sparse`` to ``false`` (default is ``true``) in the
``parthenon/sparse`` block of the input file turns on the “fake sparse”
mode. In this mode, all variables are always allocated on all blocks,
just if they were all dense, and they will not be automatically
deallocated. Thus the fake sparse mode produces the same results as if
all variables were declared dense, but the infrastructure will still
perform ``IsAllocated`` checks, so this mode does not remove the sparse
infrastructure overhead, but it is useful to debug issues arising with
the usage of sparse variables.

Compile-time
~~~~~~~~~~~~

Turning on the CMake option ``PARTHENON_DISABLE_SPARSE`` turns on fake
sparse mode (see above) and also replaces all the ``IsAllocated``
functions with essentially
``constexpr bool IsAllocated() const { return true; }`` so that they
should all be optimized out and thus the sparse infrastructure overhead
should be removed, which will be useful for measuring the performance
impact of the sparse overhead. Note however, that there will still be
some overhead due to the sparse implementation on the host. For example,
the allocation status of the variables will still be part of variable
pack caches and will be checked when retrieving packs from the cache.
However, since fake sparse is enabled, the allocation statuses will
always be all true, thus not resulting in any additional cache misses.

If sparse is compile-time disabled, this information is passed through
to the regression test suite, which will adjust its comparison to gold
results accordingly. ## Sparse naming

Of the two sparse concepts described above, sparse naming is much
simpler to implement, because it is essentially just a convenient front
end to the machinery provided by the state descriptor, containers, and
other parts of the Parthenon infrastructure, all of which don’t need to
know anything about sparse naming. Once a family or pool of variables
sharing the same base name but having different sparse IDs is added to
the state descriptor, they are treated exactly like ordinary, unrelated
variables that all have distinct labels. The only exception is functions
that take a set of flags or labels to pull out a list of variables.
These functions are aware that multiple variables can share the same
base name, and it will match all those variables if the base name is
given in a list of labels, furthermore, many of such functions take an
optional list of sparse IDs, which can be used to restrict the variable
selection to specific sparse IDs. But again, these are just front end
conveniences. Once the list of variables is assembled, all the variables
are treated as completely independently and unrelated, just like dense
variables.

Sparse naming is implemented through the ``SparsePool`` class, which can
be added to a state descriptor via ``AddSparsePool``. A ``SparsePool``
consists of: (i) a base name, (ii) a shared ``Metadata`` instance, and
(iii) a list of sparse IDs, which may be used. Note that the list of
sparse IDs must be specified when the sparse pool is created and once
its added to the state descriptor, that list cannot be changed. This
limitation drastically simplifies the sparse naming implementation,
because it means that we know the complete list of variables at the
beginning and that list is always the same on all mesh blocks. The
individual ``CellVariable`` instances that are created for each sparse
ID have a label of the form ``<base name>_<sparse index>`` and the have
the same metadata as the shared metadata of the pool, with two
exceptions: (i) the shape of the variable can be set per sparse ID
(i.e. some ID could be a scalar, another a vector of length 2, another a
vector of length 12, another a rank-3 tensor, etc.), and (ii) related to
the shape, the ``Metadata::Vector`` and ``Metadata::Tensor`` flags can
be individually set per sparse ID as well.

The sparse ID can be any integer (positive, negative, and zero) except
the smallest possible integer (``std::numeric_limits<int>::min()``),
which is reserved to mean an invalid sparse ID. It is not allowed to add
a dense variable with a label that is used as a base name for a
``SparsePool`` or vice versa.

When a sparse pool is added to the state descriptor, it simply adds a
separate variable for each of its sparse IDs with the appropriate
metadata and composite label (as described above). After this point, the
rest of the infrastructure treats those variables like any other
unrelated variables, with the following exception.When one specifies a
variable label in a list of labels, for example in the ``PackVariable``
or ``PackVariablesAndFluxes`` functions, one can simply specify the base
name in the list of labels, which will add all sparse variables with
that base name to the resulting list of variables. Furthermore, the
``Pack*`` functions also take an optional argument to specify a list of
sparse IDs. If such a list is present, then only sparse variables with
an ID from that list will be added to the pack. However, when using a
label to refer to a single variable, one must specify the full label
(base name plus sparse ID) to refer to a particular sparse variable.

Sparse allocation and deallocation implementation
-------------------------------------------------

*This section has not been completely updated from the original sparse
implementation and is kept here as a reference for developers.*

Implementing the sparse allocation capability requires deep changes in
the entire infrastructure, because the entire infrastructure assumed
that all variables are always allocated on all blocks. It also raises
the question of how to handle the case when one block has a sparse
variable allocated and its neighbor doesn’t. Under what circumstances
will the neighboring block have to allocate that sparse variable and how
will this be communicated? Furthermore, the use of MPI to communicate
boundary and other data between blocks on different MPI ranks requires
that the sending and receiving ranks both call send and receive
functions for each message passed between them, which complicates the
situation where two neighboring blocks don’t have the same sparse
variables allocated and thus would like to communicate data for
different sets of variables.

Before describing the bigger infrastructure changes to handle the
boundary communication for sparse variables, here are some smaller
changes that are necessary for sparse variables to work.

-  ``CellVariable`` tracks its allocation status and has member
   functions to allocate and deallocate its data (``data``, ``flux``,
   and ``coarse_s``).
-  A ``CellVariable`` now knows its dimensions and coarse dimensions.
   Because the ``ParArrayND<T> data`` member holding the actual variable
   data is not necessarily allocated (i.e. it has a size of 0), we can
   no longer use its size to get the dimension of the ``CellVariable``,
   but we still need to know its dimensions when it’s unallocated, for
   example when adding it to a pack. Similarly, the ``coarse_s`` member
   used to be queried to get the coarse dimensions, but that is also not
   always allocated, thus ``CellVariable`` also directly knows its
   coarse dimensions.
-  ``CellVariable``, ``MeshBlock``, ``MeshBlockData``, variable packs,
   and mesh block packs, all have new member functions ``IsAllocated``
   to query whether a particular variable is allocated or not. Generally
   speaking, whenever the data or fluxes of a variable are accessed,
   such accesses need to be guarded with ``IsAllocated`` checks.
-  The ``pvars_cc_`` field of the ``MeshRefinement`` class is now a
   ``std::vector<std::shared_ptr<CellVariable<Real>>>`` instead of a
   ``std::vector<std::tuple<ParArrayND<Real>, ParArrayND<Real>>>``. The
   problem with storing (shallow) copies of the ``ParArrayND``\ s
   ``data`` and ``coarse_s`` is that they don’t point to the newly
   allocated views if a variable is initially unallocated and then gets
   allocated during the evolution. Storing a pointer to the
   ``CellVariable`` instance works because that one remains the same
   when it gets allocated.
-  The caching mechanisms for variable packs, mesh block packs, send
   buffers, receive (i.e. set) buffers, and restrict buffers now all
   include the allocation status of all the contained variables (as a
   ``std::vector<int>`` because it’s only used on the host). When a pack
   or buffers collection is requested, the allocation status of the
   cached entity is compared to the current allocation status of the
   variables and if they don’t match, the pack or buffer collection is
   recreated.
-  The ``Globals`` namespace contains some global sparse settings
   (whether sparse is enabled, allocation/deallocation thresholds, and
   deallocation count).

Below follows a detailed description of the main sparse allocation
implementation.

Allocation status
~~~~~~~~~~~~~~~~~

Every ``CellVariable`` is either allocated or deallocated at all times.
Furthermore, the ``CellVariable``\ s with the same label but
corresponding to different stages (i.e. ``MeshBlockData`` instances) of
the same ``MeshBlock`` are always either allocated or deallocated on all
stages of the mesh block. This is enforced by the fact that the only
public methods to (de)allocate a variable is through the mesh block. The
``MeshBlock::AllocateSparse`` and ``MeshBlock::AllocSparseID`` functions
are meant to be used in the user code to specifically allocate a sparse
variable on a given block (usually, this would be done in the problem
generator). They are also used internally by the infrastructure to
allocate a sparse variable on a block if it receives non-zero boundary
data for that block, see `Boundary exchange <#boundary-exchange>`__ for
details. The infrastructure can also automatically deallocate sparse
variables on a block, see `Deallocation <#deallocation>`__.

When a ``CellVariable`` is allocated, its ``data``, ``flux``, and
``coarse_s`` fields are allocated. When the variable is deallocated,
those fields are reset to ``ParArrayND``\ s of size 0.

Deallocation
~~~~~~~~~~~~

There is a new task called ``SparseDealloc`` in
``src/interface/update.cpp`` taking a ``MeshData`` pointer. It is meant
to be run after the update task for the last stage (of course, it does
not have to be run every time step). On every block, it checks the
values of all sparse variables. If the maximum absolute value is below
the user-defined deallocation threshold, the variable is flagged for
deallocation on that block. The variable is only actually deallocated if
it has been flagged for deallocation a certain number of times in a row
(if any of the values exceeds the deallocation threshold, the counter is
reset to 0). That number is the deallocation count, which is also
settable by the user in the input file.

Boundary exchange
~~~~~~~~~~~~~~~~~

Boundary communication can trigger allocation of a field on the
receiving block if the communicated ghost data is above the allocation
threshold. Otherwise sparse boundary communication is the same as dense
boundary communication. A detailed description of the boundary
communication and flux correction implementation in Parthenon is given
`here <../sparse_boundary_communication.md>`__.

AMR and load balancing
~~~~~~~~~~~~~~~~~~~~~~

The sparse implementation for AMR and load balancing is quite straight
forward. For AMR, when we create new mesh blocks, we allocate the same
variables on them as there were allocated on the old mesh blocks the new
ones are created from.

For the load balancing, we need to send the allocation statuses of the
variables together with their data. So we add flags at the beginning of
the send/receive buffers to indicate the allocation statuses. There is
one flag per variable. The rest of the buffer is unchanged and always
includes space for all variables regardless whether they are allocated
or not. This simplifies the implementation drastically, because all the
MPI messages have the same size and the sender and receiver know what
that size is without needing the know the allocation status of the other
block. The remaining changes are as follows:

-  In ``Mesh::PrepareSendSameLevel`` we only fill the send buffer (using
   ``BufferUtility::PackData``) if the variable is allocated, otherwise
   we simply skip that region of the buffer (and leave its values
   uninitialized, since they won’t be read) so that the data for each
   variable is in the same place as if all variables were allocated.
-  In ``Mesh::PrepareSendCoarseToFineAMR`` and
   ``Mesh::PrepareSendFineToCoarseAMR`` we do the same as above, but
   instead of leaving regions of the buffer belonging to unallocated
   variables uninitialized, we fill them with zeros (using
   ``BufferUtility::PackZero``) since the target block may have the
   variable allocated even if the sender doesn’t (actually, I think this
   can only happen for fine-to-coarse and not for coarse-to-fine).
-  In ``Mesh::FillSameRankFineToCoarseAMR`` when filling in the
   destination data, we write zeros if the fine source block doesn’t
   have the variable allocated. Whereas in
   ``Mesh::FillSameRankCoarseToFineAMR`` we make sure the source and
   destination blocks have the same allocation status for each variable
   and we simply skip unallocated variables.
-  In all three types of ``Mesh::FinishRecv*`` functions, we read the
   allocation flags for all variables from the buffer, and we allocate
   it on the receiving block if the sending block had it allocated but
   it’s not yet allocated on the receiving block. We then proceed to
   read the buffer only if the variable is allocated on the receiving
   block.
