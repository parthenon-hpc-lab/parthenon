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

## Mesh Functions Controlling Evolulution and Diagnostics

Some mesh functions are stored as `std::function` members and are called by the `EvolutionDriver`. They may be overridden by the application by reassigning the default values of the `std::function` member.
* `PreStepUserWorkInLoop(Mesh*, ParameterInput*, SimTime const&)` is called to perform mesh-wide work _before_ the time-integration advance (the default is a no-op)
* `PostStepUserWorkInLoop(Mesh*, ParameterInput*, SimTime const&)` is called to perform mesh-wide work _after_ the time-integration advance (the default is a no-op)
* `PreStepUserDiagnosticsInLoop(Mesh*, ParameterInput*, SimTime const&)` is called to perform diagnostic calculations and/or edits _before_ the time-integration advance.  The default behavior calls to each package's (StateDesrcriptor's) `PreStepDiagnostics` method which, in turn, delegates to a `std::function` member that defaults to a no-op.
* `PostStepUserDiagnosticsInLoop(Mesh*, ParameterInput*, SimTime const&)` is called to perform diagnostic calculations and/or edits _after_ the time-integration advance.  The default behavior calls to each package's (StateDesrcriptor's) `PreStepDiagnostics` method which, in turn, delegates to a `std::function` member that defaults to a no-op.
