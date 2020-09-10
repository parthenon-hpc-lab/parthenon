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

## Packing over a piece of the mesh

Instead of packing over the whole mesh, you can pack over only a piece
of it by using the `Partition` machinery found in
`utils/partition_stl_containers.hpp`. For example, to break the mesh
into four evenly sized meshpacks, do
```C++
using parthenon::MeshBlock;
using parthenon::Partition::Partition_t;
Partition_t<MeshBlock> partitions;
parthenon::Partition::ToNPartitions(mesh->block, 4, partitions);
MeshBlockPack<VariablePack<Real>> packs[4];
for (int i = 0; i < partitions.size() {
  packs[i] = PackVariablesOnMesh(partitions[i], "base");
}
```

There are two partitioning functions:
```C++
// Splits container into N equally sized partitions
template <typename Container_t, typename T>
void ToNPartitions(Container_t &container, const int N, Partition_t<T> &partitions);

// Splits container into partitions of size N
template <typename Container_t, typename T>
void ToSizeN(Container_t &container, const int N, Partition_t<T> &partitions);
```
Both functions live within the namespace `parthenon::Partition`.

## Data Layout

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

## Type

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
