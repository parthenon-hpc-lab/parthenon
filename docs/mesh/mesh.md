# Mesh

The mesh object represents the mesh on a given processor/MPI
rank. There is one mesh per processor. The `Mesh` object owns all
`MeshBlock`s on a given processor.

## Looping over MeshBlocks

`MeshBlock`s are stored in a variable `Mesh::block_list`, which is an object of type
```C++
using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;
```
and so to get the predicted time step for each mesh block, you can call:
```C++
for (auto &pmb : pmesh->block_list) {
    std::cout << pmb->NewDt() << std::endl;
}
```
where `pmesh` is a pointer to a `Mesh` object. This paradigm may appear, 
for example, in an application driver.
