# Packing Variables Accross the Entire Mesh

`Kokkos` kernel launches have a 6 microsecond overhead. For small
kernels that perform little work, this can be a perforamnce bottleneck
when each kernel is launched per meshblock. Parthenon therefore
provides the capability to combine variables into a single data
structure that spans some number of meshblocks, the `MeshBlockPack`.

## Creating a MeshBlockPack

There are two methods for creating mesh packs, which are analogous to the `VariablePack` and `VariableFluxPack` available in [containers](../interface/containers.md).
```C++
template <typename T, typename... Args>
auto PackVariablesOnMesh(T &blocks, const std::string &container_name,
                         Args &&... args)
```
and
```C++
template <typename T, typename... Args>
auto PackVariablesAndFluxesOnMesh(T &blocks, const std::string &container_name,
                                  Args &&... args)
```
The former packs only the variables, the latter packs in the variables and associated flux vectors.

Here `T` can be the mesh or any standard template container that contains meshblocks.

The variatic arguments take exactly the same arguments as
`container.PackVariables` and `container.PackVariablesAndFluxes`. You
can pass in a metadata vector, a vector of names, and optionally IDs
and a map from names to indices. See
[here](../interface/containers.md) for more details.

### Packing over a piece of the mesh

Instead of packing over the whole mesh, you can pack over only a piece
of it by using the `Partition` machinery found in
`utils/partition_stl_containers.hpp`. For example, to break the mesh
into four evenly sized meshpacks, do
```C++
using parthenon::MeshBlock;
auto partitions = parthenon::partition::ToNPartitions(mesh->block_list, 4);
MeshBlockPack<VariablePack<Real>> packs[4];
for (int i = 0; i < partitions.size() {
  packs[i] = PackVariablesOnMesh(partitions[i], "base");
}
```
To pack only the variables "var1" and "var2" in the container named "mycontainer", do:
```C++
std::vector<std::string>> vars = {"var1", "var2"};
for (int i = 0; i < partitions.size() {
  packs[i] = PackVariablesOnMesh(partitions[i], "myContainer", vars);
}
```

There are two partitioning functions:
```C++
// Splits container into N equally sized partitions
template <typename T, typename Container_t>
Partition_t<T> ToNPartitions(Container_t<T> &container, const int N);

// Splits container into partitions of size N
template <typename T, typename Container_t>
std::vector<std::vector<T>> ToSizeN(Container_t<T> &container, const int N);
```
Both functions live within the namespace `parthenon::partition` and `Partition_t` 
is defined as:
```C++
templat<typename T>
using Parition_t = std::vector<std::vector<T>>
```

### Data Layout

The Mesh Pack is indexable as a five-dimensional `Kokkos` view. The
slowest moving index indexes into a 4-dimensional `VariablePack`. The
next slowest indexes into a variable. The fastest three index into the
cells on a meshblock.

For example:
```C++
// pack all variables on the base container accross the whole mesh
auto meshpack = PackVariablesAndFluxes(pmesh, "base");
auto variablepack = meshpack(m); // Indexes into the m'th meshblock
auto var = meshpack(m,l); // Indexes into the k'th variable on the m'th MB
// The l'th variable in the i,j,k'th cell of the m'th meshblock
Real r = meshpack(m,l,k,j,i); 
```

For convenience, `MeshBlockPack` also includes the following methods and fields:
```C++
// the IndexShape object describing the bounds of all blocks in the pack
IndexShape bounds = meshpack.cellbounds; 

// a 1D array of coords objects
auto coords = meshpack.coords(m); // gets the Coordinates_t object on the m'th MB

// The dimensionality of the simulation. Will be 1, 2, or 3.
// This is needed because components of the flux vector
// are only allocated for dimensions in use.
int ndim = meshpack.GetNdim();

// Get the sparse index of the n'th sparse variable in the pack.
int sparse = meshpack.GetSparse(n);

// The size of the n'th dimension of the pack
int dim = meshpack.GetDim(n);
```

### Type

The types for packs are:
```C++
MeshBlockVarPack<T>
```
and
```C++
MeshBlockVarFluxPack<T>
```
which correspond to packs over meshblocks that contain just variables
or contain variables and fluxes.

## Registering MeshBlockPacks

`MeshBlockPack`s can be named and cached on the `Mesh`, or a piece of
the mesh. A complication is that a `MeshBlockPack` must be regenerated
after each remeshing or load balancing step. Therefore, a user
registers a *function* to generate a `MeshBlockPack`, rather than a
pack itself. The relevant functions are
```C++
void RegisterMeshBlockPack(const std::string &package, const std::string &name,
    const VarPackingFunc<Real> &func);
void RegisterMeshBlockPack(const std::string &package, const std::string &name,
    const FluxPackingFunc<Real> &func);
```
Where the `package` variable is a package name and is a namespace, the
`name` variable is the name of the `MeshBlockPack`, and
```C++
template <typename T>
using VarPackingFunc = std::function<std::vector<MeshBlockVarPack<T>>(Mesh *)>;
template <typename T>
using FluxPackingFunc = std::function<std::vector<MeshBlockVarFluxPack<T>>(Mesh *)>;
```
is the function signature of the function that generates a
pack. `VarPackingFunc` and `FluxPackingFunc` should take a `Mesh`
pointer and return either a
`std::vector<MeshBlockPack<VariablePack<Real>>>` or a
std::vector<MeshBlockPack<VariableFluxPack<Real>>>`, respectively.

An example registration might look like this:
```C++
int pack_size = 4; // The number of meshblocks in a pack
pmesh->RegisterMeshBlockPack("default", "fill_ghost", [pack_size, metadata](Mesh *pmesh) {
    // Add containers if not already present
    for (auto &pmb : pmesh->block_list) {
        auto &base = pmb->real_containers.Get();
        pmb->real_containers.Add("new",base);
    }
    // Partition mesh block list
    std::vector<MeshBlockVarPack<Real>> packs;
    auto partitions = partition::ToSizeN(pmesh->block_list, pack_size);
    packs.resize(partitions.size());
    // Generate packs
    for (int i = 0; i < partitions.size(); i++) {
        packs[i] = PackVariablesOnMesh(partitions[i], "new", metadata);
    }
    return packs;
});
```
for a `Mesh* pmesh`.

Note that the packing function expects you to prepare the state of the
mesh for the pack. This may mean you need to create new containers if
they are not available in your container collection. It may also mean
you must partition the your meshblocks if you want to pack over a
piece of the mesh, as in the example above.

The user can also register these functions from within an individual
package initialization via the `StateDescriptor`. The state descriptor
supports `AddMeshBlock`, which takes a `MeshBlockPack` name and a
packing function. It is overloaded to accept either `VarPackingFunc`
or `FluxPackingFunc`. For an example of this, see the `Initialize`
function in the `calculate_pi` example,
[here](../../example/calculate_pi/calculate_pi.cpp).

Packs are regenerated appropriately and automatically when the mesh is
generated or changes. However, you can manually regenerate them with
```C++
pmesh->BuildMeshBlockPacks();
```
for a `Mesh* pmesh`.

### Accessing Registered MeshBlockPacks

You can access a `MeshBlockPack` that you registered via the public
fields in the `Mesh`:
```C++
Mesh::real_varpacks;
Mesh::real_fluxpacks;
```

The former contains `MeshBlockPack<VariablePack<Real>>`s and the
latter contains `MeshBlockPack<VariableFluxPack<Real>>`s. The actual
pack must be accessed with two strings, and an integer, a package
namespace, a pack name, and an index. For example, to access one of
the packs registed above, you would call
```C++
auto &pack = pmesh->real_varpacks["default"]["fill_ghost"][0];
```
for a `Mesh* pmesh`. To access all of them, one might call
```C++
auto &packs = pmesh->real_varpacks["default"]["fill_ghost"];
for (int i = 0; i < packs.size(); i++) {
    do_something_with(packs[i]);
}
```
For an example of this in use, see the `MakeTasks` function in the
`Driver` for `calculate_pi`
[here](../../example/calculate_pi/pi_driver.cpp).

### Default MeshBlockPacks

The `Mesh` automatically generates some commonly used `MeshBlockPacks`
based on `Metadata`. These live in the "default" namespace. These
packs contain `pack_size` `MeshBlocks` per pack. This variable
can be set in a `parthenon` input file under the `parthenon/mesh`
input block. e.g.,
```
<parthenon/mesh>
pack_size = 6
```
A `pack_size < 1` in the input file indicates the entire mesh (per MPI rank)
should be contained within a single pack. This can be accessed within
the program via
```C++
pmesh->DefaultPackSize();
```
The default packs available are:

| Namespace | Name        | Contains Fluxes | Metadata Condition     |
| --------- | ----------- | --------------- | ---------------------- |
| default   | fill_ghosts | No              | `Metadata::FillGhosts` |
