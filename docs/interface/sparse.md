# Sparse implementation

This file describes the implementation of sparse variables in Parthenon. It is mostly addressed at
developers and focuses on the implementation details, but it also briefly summarizes the available
features and how to use them.


## Sparse features overview

By default, fields added to a state descriptor via the [`AddField`
function](state.md#StateDescriptor) are dense, which means that every mesh block in the entire
domain allocates data for this field (in addition to buffers for fluxes, ghost zones, and coarse
values, depending on the metadata assigned to the field, see [here](./Metadata.md)).

Parthenon also provides the capability to use **sparse** fields, where "sparse" really refers to two
distinct and orthogonal concepts, namely (a) a family or pool of distinct and separate variable
fields that share a common base name and are distinguished by a sparse ID (**sparse naming**),
and (b) variables that are only allocated on some but not necessarily all mesh blocks in the domain
(**sparse allocation**). In principle, both of
these "sparse" concepts can be used independently, i.e. we could make a family of dense
variables that share a base name and are distinguished by sparse IDs, or we could simply have a
single variable with a label without a sparse ID that is only allocated on some blocks but not
others. However, in practice these two concepts are usually used together. For example, if we use
sparse IDs to enumerate different materials, we might want to have a density variable for each
material, which naturally forms a family of related variables all called "density" but they are
distinguished by the material (or sparse) ID (sparse naming). Furthermore, a particular material is
probably only present in some part of the domain, so it would be a waste to allocate
data consisting of all zeros in the parts of the domain where that material is not present (sparse
allocation).


## Sparse naming

Of the two sparse concepts described above, sparse naming is much simpler to implement, because it
is essentially just a convenient front end to the existing machinery provided by the state
descriptor, containers, and other parts of the Parthenon infrastructure, all of which don't need to
know anything about sparse naming. Once a family or pool of variables sharing the same base name but
having different sparse IDs is added to the state descriptor, they are treated exactly like
ordinary, unrelated variables that all have distinct labels. The only exception is functions that take
a set of flags or labels to pull out a list of variables. These functions are aware that multiple
variables can share the same base name, and it will match all those variables if the base name is
given in a list of labels, furthermore, many of such functions take an optional list of sparse
IDs, which can be used to restrict the variable selection to specific sparse IDs. But again,
these are just front end conveniences. Once the list of variables is assembled, all the variables are
treated as completely independently and unrelated, just like dense variables.

Sparse naming is implemented through the `SparsePool` class, which can be added to a state
descriptor via `AddSparsePool`. A `SparsePool` consists of: (i) a base name, (ii) a shared
`Metadata` instance, and (iii) a list of sparse IDs, which may be used. Note that the list of
sparse IDs must be specified when the sparse pool is created and once its added to the state
descriptor, that list cannot be changed. This limitation drastically simplifies the sparse naming
implementation, because it means that we know the complete list of variables at the beginning and
that list is always the same on all mesh blocks. The individual `CellVariable` instances that are
created for each sparse ID have a label of the form `<base name>_<sparse index>` and the have the
same metadata as the shared metadata of the pool, with two exceptions: (i) the shape of the variable
can be set per sparse ID (i.e. some ID could be a scalar, another a vector of length 2,
another a vector of length 12, another a rank-3 tensor, etc.), and (ii) related to the shape, the
`Metadata::Vector` and `Metadata::Tensor` flags can be individually set per sparse ID as well.

The sparse ID can be any integer (positive, negative, and zero) except the smallest possible
integer (`std::numeric_limits<int>::min()`), which is reserved to mean an invalid sparse ID. It is
not allowed to add a dense variable with a label that is used as a base name for a `SparsePool` or
vice versa.

When a sparse pool is added to the state descriptor, it simply adds a separate variable for each of
its sparse IDs with the appropriate metadata and composite label (as described above). After this
point, the rest of the infrastructure treats those variables like any other unrelated variables,
with the following exception.When one specifies a variable label in a list of labels, for
example in the `PackVariable` or `PackVariablesAndFluxes` functions, one can simply specify the base
name in the list of labels, which will add all sparse variables with that base name to the resulting
list of variables. Furthermore, the `Pack*` functions also take an optional argument to specify a
list of sparse IDs. If such a list is present, then only sparse variables with an ID from that list
will be added to the pack. However, when using a label to refer to a single variable, one must
specify the full label (base name plus sparse ID) to refer to a particular sparse variable.


## Sparse allocation

Implementing the sparse allocation capability requires deep changes in the entire infrastructure,
because the entire infrastructure assumed that all variables are always allocated on all blocks. It
also raises the question of how to handle the case when one block has a sparse variable allocated
and its neighbor doesn't. Under what circumstances will the neighboring block have to allocate that
sparse variable and how will this be communicated? Furthermore, the use of MPI to communicate
boundary and other data between blocks on different MPI ranks requires that the sending and
receiving ranks both call send and receive functions for each message passed between them, which
complicates the situation where two neighboring blocks don't have the same sparse variables
allocated and thus would like to communicate data for different sets of variables.

Before describing the bigger infrastructure changes to handle the boundary communication for sparse
variables, here are some smaller changes that are necessary for sparse variables to work.

- `CellVariable` tracks its allocation status and has member functions to allocate and deallocate
  its data (`data`, `flux`, and `coarse_s`).
- A `CellVariable` now knows its dimensions and coarse dimensions. Because the `ParArrayND<T> data` member holding
  the actual variable data is not necessarily allocated (i.e. it has a size of 0), we can no longer
  use its size to get the dimension of the `CellVariable`, but we still need to know its dimensions
  when it's unallocated,  for example when adding it to a pack. Similarly, the `coarse_s` member
  used to be queried to get the coarse dimensions, but that is also not always allocated, thus
  `CellVariable` also directly knows its coarse dimensions.
- `CellVariable`, `MeshBlock`, `MeshBlockData`, variable packs, and mesh block packs, all have new
  member functions `IsAllocated` to query whether a particular variable is allocated or not.
  Generally speaking, whenever the data or fluxes of a variable are accessed, such accesses need to
  be guarded with `IsAllocated` checks.
- The `pvars_cc_` field of the `MeshRefinement` class is now a
  `std::vector<std::shared_ptr<CellVariable<Real>>>` instead of a
  `std::vector<std::tuple<ParArrayND<Real>, ParArrayND<Real>>>`. The problem with storing (shallow)
  copies of the `ParArrayND`s `data` and `coarse_s` is that they don't point to the newly allocated views if a
  variable is initially unallocated and then gets allocated during the evolution. Storing a pointer
  to the `CellVariable` instance works because that one remains the same when it gets allocated.
- The caching mechanisms for variable packs, mesh block packs, send buffers, receive (i.e. set) buffers, and
  restrict buffers now all include the allocation status of all the contained variables (as a
  `std::vector<bool>` because it's only used on the host and provides efficient comparison). When a
  pack or buffers collection is requested, the allocation status of the cached entity is compared to
  the current allocation status of the variables and if they don't match, the pack or buffer
  collection is recreated.
- The `Globals` namespace contains some global sparse settings (whether sparse is enabled,
  allocation/deallocation thresholds, and deallocation count).

Below follows a detailed description of the main sparse allocation implementation.


### Allocation status
Every `CellVariable` is either allocated or deallocated at all times. Furthermore, the
`CellVariable`s with the same label but corresponding to different stages (i.e. `MeshBlockData`
instances) of the same `MeshBlock` are always either allocated or deallocated on all stages of the
mesh block. This is enforced by the fact that the only public methods to (de)allocate a variable is
through the mesh block. The `MeshBlock::AllocateSparse` and `MeshBlock::AllocSparseID` functions are
meant to be used in the user code to specifically allocate a sparse variable on a given block
(usually, this would be done in the problem generator). They are also used internally by the
infrastructure to allocate a sparse variable on a block if it receives non-zero boundary data for
that block, see [Boundary exchange](#boundary-exchange) for details. The infrastructure can also
automatically deallocate sparse variables on a block, see [Deallocation](#deallocation).

When a `CellVariable` is allocated, its `data`, `flux`, and `coarse_s` fields are allocated. When
the variable is deallocated, those fields are reset to `ParArrayND`s of size 0. Note, however, that
the `CellCenteredBoundaryVariable` (field `vbvar`) is always allocated, but it only holds shallow
copies of the `ParArrayND`s owned by the `CellVariable` that may are may not be allocated. Note also
that all the `CellVariable`s with the same label share one instance of
`CellCenteredBoundaryVariable` between the different stages of a mesh block (and the shallow copies
of the `data`, `flux`, and `coarse_s` arrays in `CellCenteredBoundaryVariable` point to the ones of
the `CellVariable` belonging to the base stage). This may be changed in the future so that each
`CellVariable` has its own `CellCenteredBoundaryVariable`.

Note that since `CellCenteredBoundaryVariable` is always allocated, the `BoundaryData` instances it
contains are also always allocated (regardless whether the `CellVariable` is allocated).
The `BoundaryData` instances (one for boundary exchange and one for flux exchange) contain buffers
to communicate the boundary/flux correction data. So these buffers are always allocated for all
variables. (However, as noted above, there is only one `CellCenteredBoundaryVariable` per variable
label per mesh block that is shared between all stages.) This is not strictly necessary, the sparse
implementation could be improved to not allocate the `CellCenteredBoundaryVariable` for local
neighbors of unallocated variables, thus saving some memory. However, this would make the sparse
boundary exchange mechanism (see [Boundary exchange](#boundary-exchange)) more complex. For that
reason, the current implementation keeps things simpler by always allocating the
`CellCenteredBoundaryVariable`.

### Deallocation
There is a new task called `SparseDealloc` taking a `MeshData` pointer. It is meant to be run after
the update task for the last stage (of course, it does not have to be run every time step). On every
block, it checks the values of all sparse variables. If the maximum absolute value is below the
user-defined deallocation threshold, the variable is flagged for deallocation on that block. The
variable is only actually deallocated if it has been flagged for deallocation a certain number of
times in a row (if any of the values exceeds the deallocation threshold, the counter is reset to 0).
That number is the deallocation count, which is also settable by the user in the input file. Note
that currently the deallocation threshold is the same for all variables, a future improvement is to
add the capability of setting a different threshold per sparse pool.

### Boundary exchange


### Flux corrections


### AMR and load balancing

