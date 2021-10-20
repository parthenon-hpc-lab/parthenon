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
The boundary exchange with sparse follows the same pattern as without sparse, but with the following
additional complexity:

- The `BoundaryVariable` abstract base class now has an additional field `std::array<bool,
  NMAX_NEIGHBORS> local_neighbor_allocated`, which tracks if this particular variable is allocated
  on each neighboring block. This is used to determine if boundary buffers should be sent to local
  neighboring blocks on the same rank. It is not used to keep track of neighbor allocation status
  for neighboring blocks on a different MPI rank. The local neighbor allocation status is updated at
  the beginning of `MeshBlockData<T>::StartReceiving`, when the send boundary buffers are recreated
  in `cell_centered_bvars::ResetSendBufferBoundaryInfo` and at the end of running the [`SparseDealloc`
  task](#deallocation).
- In `cell_centered_bvars::ResetSendBufferBoundaryInfo` for local neighbors we only point the send
  buffer directly at the neighbor's receive buffer if the neighbor has the variable allocated.
  Otherwise, we use the separate send buffer just like for non-local neighbors.
- The send and receive buffers of `CellCenteredBoundaryVariable` have been extended by one element
  at the end of the buffer space (after the packed data) to convey if non-zero values are being sent
  in the buffer. Note that this is not the same as the allocation status of the variable on the
  sending block. If the sending block doesn't have the variable allocated, then the flag will be
  false (meaning only zero values are being sent), but if the variable is allocated on the sending
  block, the value of the flag will depend on whether the values in the send buffer are all below
  the allocation threshold (i.e. considered to be zero) or not. More details in the next item.
- In `cell_centered_bvars::SendBoundaryBuffers` when the buffers are being filled, we check if the
  absolute value of a value written to the buffer is above the allocation threshold (this threshold
  is currently shared between all variables, a future improvement will be to make this different per
  variable). If at least one absolute value is above the allocation threshold, the flag indicating
  that non-zero values are being sent is set to true (see item above). Note that currently the
  buffers for unallocated variables are filled with all zeros. This is a simplification and not
  necessary, but it does help to ensure that MPI and non-MPI results are exactly the same. In the
  future, this could be optimized if one is willing to allow different results (up to the allocation
  tolerance with and without MPI and depending on the number of ranks).
- In `cell_centered_bvars::SendAndNotify` is where the magic for local neighbors happens.
  Previously, there was nothing to do in `SendAndNotify` for local neighbors other than setting  the
  boundary status flag of th neighbor to arrived. Now we check if the neighbor needs to newly
  allocate the variable. If we are sending a non-zero buffer (see previous item) and the neighbor
  does not have this variable allocated, then we allocate that variable on the neighbor in
  `SendAndNotify` and we perform a `Kokkos::deep_copy` from the send buffer of the sending block to
  the receive buffer of the target block (this is not necessary for local neighbors that already had
  the the variable allocated, because in that case we bypassed the send buffer altogether and
  `cell_centered_bvars::SendBoundaryBuffers` directly filled in the receive buffer).
- For non-local neighbors, `cell_centered_bvars::SendAndNotify` always calls `MPI_Start` regardless
  whether the variable is allocated or not. That is because the receiving block does not know the
  allocation status of the sending block, so it always waits for a message (again regardless whether
  it has the variable allocated or not) and thus we always have to send the message.
- In `BoundaryVariable::ReceiveBoundaryBuffers` we only wait for the boundary status to be flagged
  as arrived for local neighbors if the variable is allocated (otherwise there is no need to wait).
  For non-local neighbors, we always wait until the MPI message is received. Once the MPI message is
  received and the receiving block doesn't have the variable allocated, we check the flag at the end
  of the receive buffer to see if the sender sent any non-zero values in the buffer. If so, we
  allocate the variable on the receiving mesh block.
- Finally, in `cell_centered_bvars::SetBoundaries` we only read the buffer the variable is allocated
  (otherwise there is nowhere to write to) and if the flag indicating that non-zero values were sent
  is false, we simply write zeros instead of reading the buffer. This is again to make sure that the
  results are the same regardless of the number of MPI ranks used.

### Flux corrections
Flux corrections work essentially the same as for dense variables, except that no flux corrections
will be exchanged if either the send or the receiver doesn't have the variable allocated (for
non-local neighbors, the MPI send/receive calls are still performed) and if a block receives flux
corrections from a neighbor that has the variable allocated but the receiving block does not have
the variable allocated, it will never result in a variable allocation, unlike in the boundary
exchange.

Specifically the implementation details are as follows:

- Just like for the boundary buffers, the flux corrections buffers carry one additional value to
  indicate the allocation status on the sending block. However, for flux corrections that flag is
  actually the first element in the buffer and not the last (because of how the flux corrections are
  written and read, it is easier to put that flag at the beginning). And also, this flag just
  indicates the allocation status of the variable on the sending block, not if it is sending
  non-zero values.
- In `CellCenteredBoundaryVariable::SendFluxCorrection` if the neighbor is local and the variable is
  not allocated on **both* the sending and receiving block, the function does nothing. Because we'd
  either send all zeros (if not allocated on sender) or we'd ignore the flux corrections (if not
  allocated on receiver). If the neighbor is non-local we always perform the `MPI_Start` call
  regardless whether the sender has the variable allocated or not. But, of course, the buffer can
  only be filled with actual flux corrections if the sender has it allocated.
- In `CellCenteredBoundaryVariable::ReceiveFluxCorrection`, we wait to receive flux corrections from
  a local neighbor only if both the sender and receiver have the variable allocated (otherwise the
  sender won't be sending anything). For non-local neighbors, we always wait to receive the MPI
  message. Once the flux corrections are received (local or non-local), we ignore them if the
  receiving block does not have the variable allocated. Otherwise (receiver has variable allocated),
  we check if the flag in the buffer indicates that the sender has the variable allocated. If yes,
  we read the buffer, otherwise we write zeros into the fluxes of the variable.

### AMR and load balancing
The sparse implementation for AMR and load balancing is quite straight forward. For AMR, when we
create new mesh blocks, we allocate the same variables on them as there were allocated on the old
mesh blocks the new ones are created from.

For the load balancing, we need to send the allocation statuses of the variables together with their
data. So similarly to the boundary exchange and flux corrections buffers, we add flags at the
beginning of the send/receive buffers to indicate the allocation statuses. There is one flag per
variable. The rest of the buffer is unchanged and always includes space for all variables regardless
whether they are allocated or not. This simplifies the implementation drastically, because all the
MPI messages have the same size and the sender and receiver know what that size is without needing
the know the allocation status of the other block. The remaining changes are as follows:

- In `Mesh::PrepareSendSameLevel` we only fill the send buffer (using `BufferUtility::PackData`) if
  the variable is allocated, otherwise we simply skip that region of the buffer (and leave its
  values uninitialized, since they won't be read) so that the data for each variable is in the same
  place as if all variables were allocated.
- In `Mesh::PrepareSendCoarseToFineAMR` and `Mesh::PrepareSendFineToCoarseAMR` we do the same as
  above, but instead of leaving regions of the buffer belonging to unallocated variables
  uninitialized, we fill them with zeros (using `BufferUtility::PackZero`) since the target block
  may have the variable allocated even if the sender doesn't (actually, I think this can only happen
  for fine-to-coarse and not for coarse-to-fine).
- In `Mesh::FillSameRankFineToCoarseAMR` when filling in the destination data, we write zeros if the
  fine source block doesn't have the variable allocated. Whereas in
  `Mesh::FillSameRankCoarseToFineAMR` we make sure the source and destination blocks have the same
  allocation status for each variable and we simply skip unallocated variables.
- In all three types of `Mesh::FinishRecv*` functions, we read the allocation flags for all
  variables from the buffer, and we allocate it on the receiving block if the sending block had it
  allocated but it's not yet allocated on the receiving block. We then proceed to read the buffer
  only if the variable is allocated on the receiving block.


### Turning off sparse

The sparse allocation feature can be turned off at run- or compile-time. The sparse naming feature
cannot be turned off.

#### Run-time
Setting `enable_sparse` to `false` (default is `true`) in the `parthenon/sparse` block of the input
file turns on the "fake sparse" mode. In this mode, all variables are always allocated on all
blocks, just if they were all dense, and they will not be automatically deallocated. Thus the fake
sparse mode produces the same results as if all variables were declared dense, but the
infrastructure will still perform `IsAllocated` checks, so this mode does not remove the sparse
infrastructure overhead, but it is useful to debug issues arising with the usage of sparse
variables.

#### Compile-time
Turning on the CMake option `PARTHENON_DISABLE_SPARSE` turns on fake sparse mode (see above) and
also replaces all the `IsAllocated` functions with essentially `constexpr bool IsAllocated() const
{ return true; }` so that they should all be optimized out and thus the sparse infrastructure
overhead should be removed, which will be useful for measuring the performance impact of the sparse
overhead. Note however, that there will still be some overhead due to the sparse implementation on
the host. For example, the allocation status of the variables will still be part of variable pack
caches and will be checked when retrieving packs from the cache. However, since fake sparse is
enabled, the allocation statuses will always be all true, thus not resulting in any additional cache
misses.

If sparse is compile-time disabled, this information is passed through to the regression test suite,
which will adjust its comparison to gold results accordingly.
