# Containers, Container Iterators, and Variable Packs

## Containers

A *container* owns *variables.* Each variable is named, and the
container knows about various types of variables, such as whether the
variable is for cell-centered data, face-centered data, sparse data,
or dense data. (For more details on anonymous variables, see
[here](Metadata.md).) Variables in a container can be different
shapes, e.g., scalar, tensor, etc.A variable can be added to a
container as:
```C++
parthenon::MeshBlockData.Add(name, metadata, shape)
```
where the name is a string, the metadata is a `std::vector` of
metadata flags, and the shape is a `std::vector` of integers
specifying the dimensions.

Note that if a location, such as `Metadata::Cell` or `Metadata::Face`,
then shape is the shape of the variable at a given point. If you have
a scalar, you do not need to specify the shape in `Add`. If you want a
vector field with 3 components, you would use
```C++
shape = std::vector<int>({3});
```
and if you wanted a 3x3 rank-2 tensor field, you would use
```C++
shape = std::vector<int>({3,3});
```
If you *do not* specify a location on the grid which the variable
lives, then shape is the shape of the full array. I.e., a
`11x12x13x14` array would be added with a shape of
```C++
shape = std::vector<int>({11,12,13,14});
```

It is often desirable to extract from a container a specific set of
variables that have desired names, sparse ids, or conform to specific
metadata flags. This set of variables must be collected in such a way
that it can be accessed easily and performantly on a GPU, and such
that one can index into the collection in a known way. This capability
is provided by *variable packs*.

To extract a variable pack for variables with a set of names, call 
```C++
container.PackVariables(names, map)
```
where `names` is a `std::vector` of strings and `map` is an 
instance of a `parthenon::PackIndexMap`. 
This will return a `VariablePack` object, which 
is essentially a `Kokkos::view` of `parthenon::ParArray3D`s. 
The map will be filled by reference as a map from 
variable names to indices in the `VariablePack`.

Similar methods are available for metadata and sparse IDs:
```C++
container.PackVariables(metadata, ids, map)
container.PackVariables(metadata, map)
```
If you would like all variables in a container, 
you can ommit the metadata or name arguments:
```C++
container.PackVariables(map)
```
If you do not care about indexing into variables by name, 
you can ommit the `map` argument in any of the above calls.

For examples of use, see [here](../../tst/unit/test_meshblock_data_iterator.cpp).
