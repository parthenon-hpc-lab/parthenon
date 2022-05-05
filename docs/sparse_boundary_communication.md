# Sparse boundary communication 

Communication between meshblocks at all stages is required to synchronize fields that include ghost zones. For each pair of meshblocks **a** and **b** that share a boundary, we define two *communication channel*s (**a**->**b** and **b**->**a**) for each field that communicates ghost zone data. This communication channel can be shared amongst all stages contained in the `MeshBlock` field `DataCollection<MeshBlockData<Real>> meshblock_data`, since the communication during different stages doesn't overlap. A communication channel needs to contain some state that indicates if information has been sent from block **a** but not received on block **b**, or if the information has been received. Additionally, the communication channel should contain storage if data is being communicated (which may not always be the case for sparse variables). 

When the sender is sparse and unallocated, no storage communication buffer storage is required since the ghost zones of the receiver should be filled with default values. When the sender is allocated, we always allocate buffer storage  When sparse variables are considered, there should be five possible states of the buffer:

- `sending`: The sender has filled the buffer with data meets the threshold for sparse allocation.
- `sending_null`: The sender is either unallocated, so that it just contains default data, or the sender is allocated and but all of the values in the communicated region fall below the sparse threshold. 
- `received`: The receiver has successfully queried that information has been sent and the buffer contains data that is above the sparse threshold. 
- `received_null`: The receiver has successfully queried that information has been sent and the buffer contains data that is above the sparse threshold. 
- `stale`: The receiver no longer needs the information currently stored by the communication channel. 

Information about the indices of the data being copied from the sender and where they are copied to in the receiver is kept separate from the communication channel in the current implementation. The only information the communication channel has is the size of buffer required. 

## Utilities classes for boundary communication 

### `class CommBuffer<T>`
This general idea is implemented in the `CommBuffer<T>` class template, which contains a field `T buf_` and access to that field through the method `T& buffer()`. `T` is assumed to be some type that can act as a buffer for storing the communicated data that has methods `size()` and `data()`, the latter of which must give access to the underlying storage that can be used by the MPI communication calls. `CommBuffer` has the public methods: 

- `T& buffer()`: Gives access to the the actual data buffer for filling and reading. 
- `void Allocate()`: Which allocates storage for the buffer and can be called internally by `TryReceive()` when non-null data is being communicated by a block on a different rank. 
- `void Send()`: If the underlying storage buffer is allocated, sets the state of the buffer to `sending`. If the underlying storage buffer is unallocated, set the state of the buffer to `sending_null`. Also starts asynchronous MPI send if sender and receiver are on separate ranks.  
- `void SendNull()`: Sets the buffer state to `sending_null`. Also starts asynchronous MPI send of a zero length buffer if sender and receiver are on separate ranks. 
- `bool TryReceive()`: If on same rank, checks if state is `sending` or `sending_null` and sets to `received` or `received_null`, respectively, and returns `true`. If on different ranks, checks to see if MPI message is available, returns `false` if it is not. If it is, it checks the size of the incoming message, allocates buffer storage if necessary, sets the state of the `CommBuffer` to `received` or `received_null` and returns `true`.
- `Stale()`: Sets the state to `stale`; 

as well as copy constructors, assignment operators, etc. The constructor of `CommBuffer` is called as 
```c++
CommBuffer<T>(mpi_message_tag, sender_rank, receiver_rank, mpi_communicator,
            [...capture necessary stuff...](){ 
              return ...allocated object of type T that has the desired size...; 
            });
```
The lambda passed to the constructor is stored as a field in the class and is called when the internal storage buffer needs to be allocated (see `BuildSparseBoundaryBuffers` in `sparse_bvals_cc_in_one.cpp` for an example usage). Aside from during construction, there should be no difference in useage between a same rank to same rank `CommBuffer` and a separate rank `CommBuffer`. 

### `class ObjectPool<T>` 

An `ObjectPool` hands out reference counted objects that publicly inherit from `T`, which is assumed to be something like a Kokkos view which has an assignment operator and copy ctor that perform shallow copies. Rather than creating a new object each time one is requested, an `ObjectPool` recycles previously created objects that have been released back to the pool to limit the number of times that objects need to be created. When the class method `Get()` is called a pre-allocated, free resource is handed out if one is available, otherwise a new resource is created. 

An object pool has two types of objects it can hand out, 
```c++
class ObjectPool<T>::weak_t : public T {...};
class ObjectPool<T>::owner_t : public weak_t {...};
```
both of which contain a pointer to the object pool that handed them out and a key that identifies the unique resource in the pool that they reference. An `owner_t` object contributes to the reference count for a particular resource. When the reference count goes to zero in an `owner_t` dtor, the underlying resource is added to the stack of available objects and the key associated with the resource is invalidated. A `weak_t` object does not contribute to the reference count and its underlying resource can become invalid. The validity of an object can be checked with the member method `bool IsValid()`.  

The mechanism by which new pool objects are created is specified by a lambda passed to the `ObjectPool` constructor:
```c++
template <class T>
using dev_arr_t = typename Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::CudaSpace>;
int N = 100;
Pool<dev_arr_t<double>>([N](){ return dev_arr_t<double>("test pool", N); });
```
A lot of this functionality could be replicated with `shared_ptr` I think, but it is somewhat useful for these objects to be able to exist on device (even though the reference counting doesn't work there). 

## Sparse boundary communication tasks 
The tasks for sparse boundary communication pretty closely mirror the `bvals_in_one` tasks but allow 
for allocation and deallocation of the communication buffers and do not reference the `BoundaryVariable` associated classes. We also add an object to `Mesh` mapping from communication channel key (denoted by a tuple{send_gid, receive_gid, var_name}) to the associated `CommBuffer` associated with that channel. We also add a map of object pools containing pools for various buffer sizes.
```c++
using channel_key_t = std::tuple<int, int, std::string>;
std::unordered_map<channel_key_t, CommBuffer<buf_pool_t<Real>::owner_t>> boundary_comm_map;

template <typename T>
using buf_pool_t = ObjectPool<BufArray1D<T>>
std::unordered_map<int, buf_pool_t<Real>> pool_map;
``` 

Note that every stage shares the same `CommBuffer`s. 

This works with old code with the replacements 
```c++
cell_centered_bvars::SendBoundaryBuffers -> cell_centered_bvars::LoadAndSendSparseBoundaryBuffers
cell_centered_bvars::ReceiveBoundaryBuffers -> cell_centered_bvars::ReceiveSparseBoundaryBuffers
cell_centered_bvars::SetBoundaries -> cell_centered_bvars::SetInternalSparseBoundaryBuffers
```
and for dense variables gives bitwise agreement. 

### `BuildSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>>& md)`
Iterates over communication channels sending or receiving from blocks in `md`. For every sending channel it creates a communication channel for each in the `Mesh::boundary_comm_map`. For receiving channels where the blocks are on different ranks, it also creates a receiving channel in `Mesh::boundary_comm_map` since the sender will not add this channel on the current rank. Also creates new `buf_pool_t`s for the required buffer sizes if they don't already exist. Note that no memory is saved for the communication buffers at this point. 

This is called during `Mesh::Initialize(...)` and during `EvolutionDriver::InitializeBlockTimeStepsAndBoundaries()` and before this task is called `Mesh::boundary_comm_map` is cleared. This should not be called in downstream code. 

### `LoadAndSendSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>>& md)`
- Allocates buffers if necessary based on allocation status of block fields and checks if `MeshData::send_bnd_info` objects are stale. 
- Rebuilds the `MeshData::send_bnd_info` objects if they are stale 
- Restricts where necessary 
- Launches kernels to load data from fields into buffers, checks whether any of the data is above the sparse allocation threshold. 
- Calls `Send()` or `SendNull()` from all of the boundary buffers depending on their status.   

### `ReceiveSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>>& md)` 
- Tries to receive from each of the receive channels associated with `md`. 
- If the receive is succesful, and allocation of the associated field is required, allocate it 

### `SetInternalSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>>& md)` 
- Check if `MeshData::recv_bnd_info` needs to be rebuilt because of changed allocation status. 
- Rebuild `MeshData::recv_bnd_info` if necessary. 
- Launch kernels to copy from buffers into fields or copy default data into fields if sending null. 
- Stale the communication buffers. 