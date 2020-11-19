# Changelog

## Current develop

### Added (new features/APIs/variables/...)
- [[PR 335]](https://github.com/lanl/parthenon/pull/335) Support for project-relative `MACHINE_CFG` with `@PAR_ROOT@`
- [[PR 328]](https://github.com/lanl/parthenon/pull/328) New `MeshBlock` packing interface using `DataCollection`s of `MeshData` and `MeshBlockData`.

### Changed (changing behavior/API/variables/...)

### Fixed (not changing behavior/API/variables/...)
- [[PR 353]](https://github.com/lanl/parthenon/pull/353) Fixed small error in input\_parameter logic
- [[PR 352]](https://github.com/lanl/parthenon/pull/352) Code compiles cleanly (no warnings) with nvcc_wrapper

### Infrastructure (changes irrelevant to downstream codes)
- [[PR 335]](https://github.com/lanl/parthenon/pull/335) New machine configuration file for LANL's Darwin cluster
- [[PR 200]](https://github.com/lanl/parthenon/pull/200) Adds support for running ci on power9 nodes. 
- [[PR 347]](https://github.com/lanl/parthenon/pull/347) Speed up darwin ci by using pre installed spack packages from project space
- [[PR 368]](https://github.com/lanl/parthenon/pull/368) Fixes false positive in ci.

### Removed (removing behavior/API/varaibles/...)

## Release 0.3.0
Date: 10/29/2020

### Added (new features/APIs/variables/...)
- [[PR 317]](https://github.com/lanl/parthenon/pull/317) Add initial support for particles (no MPI support)
- [[PR 311]](https://github.com/lanl/parthenon/pull/311) Bugfix::Restart. Fixed restart parallel bug and also restart bug for simulations with reflecting boundary conditions.  Added ability to write restart files with or without ghost cells by setting `ghost_zones` in the output block similar to other output formats.
- [[PR 314]](https://github.com/lanl/parthenon/pull/314) Generalized `par_for` abstractions to provide for reductions with a consistent interface.
- [[PR 308]](https://github.com/lanl/parthenon/pull/308) Added the ability to register and name `MeshBlockPack`s in the `Mesh` or in package initialization.
- [[PR 285]](https://github.com/lanl/parthenon/pull/285) Parthenon can now be linked in CMake as `Parthenon::parthenon` when used as a subdirectory, matching install.

### Changed (changing behavior/API/variables/...)
- [[PR 303]](https://github.com/lanl/parthenon/pull/303) Changed `Mesh::BlockList` from a `std::list<MeshBlock>` to a `std::vector<std::shared_ptr<MeshBlock>>`, making `FindMeshBlock` run in constant, rather than linear, time. Loops over `block_list` in application drivers must be cahnged accordingly.
- [[PR 300]](https://github.com/lanl/parthenon/pull/300): Changes to `AddTask` function signature. Requires re-ordering task dependency argument to front.
- [[PR 307]](https://github.com/lanl/parthenon/pull/307) Changed back-pointers in mesh structure to weak pointers. Cleaned up `MeshBlock` constructor and implemented `MeshBlock` factory function.

### Fixed (not changing behavior/API/variables/...)
- [[PR 293]](https://github.com/lanl/parthenon/pull/293) Changed `VariablePack` and related objects to use `ParArray1D` objects instead of `ParArrayND` objects under the hood to reduce the size of the captured objects.
- [[PR 313]](https://github.com/lanl/parthenon/pull/313) Add include guards for Kokkos in cmake.
- [[PR 321]](https://github.com/lanl/parthenon/pull/321) Make inner loop pattern tags constexpr

### Infrastructure (changes irrelevant to downstream codes)
- [[PR 336]](https://github.com/lanl/parthenon/pull/336) Automated testing now checks for extraneous HtoD or DtoH copies.
- [[PR 325]](https://github.com/lanl/parthenon/pull/325) Fixes regression in convergence tests with multiple MPI ranks.
- [[PR 310]](https://github.com/lanl/parthenon/pull/310) Fix Cuda 11 builds.
- [[PR 281]](https://github.com/lanl/parthenon/pull/281) Allows one to run regression tests with more than one cuda device, Also improves readability of regression tests output.
- [[PR 330]](https://github.com/lanl/parthenon/pull/330) Fixes restart regression test.


## Release 0.2.0
Date: 9/12/2020

### Added
- [[PR 250]](https://github.com/lanl/parthenon/pull/250) Feature::Restart. If output file format 'rst' is specified restart files are written using independent variables and those marked with Restart metadata flag.  Simulations can be restarted with a '-r \<restartFile\>' argument to the code.
- [[PR 263]](https://github.com/lanl/parthenon/pull/263) Added MeshBlockPack, a mechanism for looping over the whole mesh at once within a `Kokkos` kernel. See [documentation](docs/mesh/packing.md)
- [[PR 267]](https://github.com/lanl/parthenon/pull/267) Introduced TaskRegions and TaskCollections to allow for task launches on multiple blocks.
- [[PR 287]](https://github.com/lanl/parthenon/pull/287) Added machine configuration file for compile options, see [documentation](https://github.com/lanl/parthenon/blob/develop/docs/building.md#default-machine-configurations)
- [[PR 290]](https://github.com/lanl/parthenon/pull/290) Added per cycle performance output diagnostic.
- [[PR 298]](https://github.com/lanl/parthenon/pull/298) Introduced Partition, a tiny utility for partitioning STL containers. Used for MeshBlockPacks, to enable packing over a fraction of the mesh.

### Changed
- [\#68](https://github.com/lanl/parthenon/issues/68) Moved default `par_for` wrappers to `MeshBlock` 
- [[PR 243]](https://github.com/lanl/parthenon/pull/243) Automatically find/check Python version used in regression tests. Bumps CMake minimum version to 3.12
- [[PR 266]](https://github.com/lanl/parthenon/pull/266): It is no longer necessary to specify Kokkos_ENABLE_OPENMP this is by default enabled, to turn off one can specify PARTHENON_DISABLE_OPENMP.

### Fixed
- [[PR 271]](https://github.com/lanl/parthenon/issues/256): Fix setting default CXX standard.
- [[PR 262]](https://github.com/lanl/parthenon/pull/262) Fix setting of "coverage" label in testing. Automatically applies coverage tag to all tests not containing "performance" label.
- [[PR 276]](https://github.com/lanl/parthenon/pull/276) Decrease required Python version from 3.6 to 3.5.
- [[PR 283]](https://github.com/lanl/parthenon/pull/283) Change CI to extended nightly develop tests and short push tests.
- [[PR 291]](https://github.com/lanl/parthenon/pull/291) Adds Task Diagram to documentation.

### Removed
- [[PR 282]](https://github.com/lanl/parthenon/pull/282) Integrated MeshBlockPack and tasking in pi example
- [[PR 294]](https://github.com/lanl/parthenon/pull/294) Fix `IndexShape::GetTotal(IndexDomain)` - previously was returning opposite of expected domain result.

## Release 0.1.0
Date: 8/4/2020

Initial release of Parthenon AMR infrastructure.

### Changed
- [[PR 214]](https://github.com/lanl/parthenon/pull/214): The weak linked routines for user-specified parthenon behavior have been removed in favor of a more portable approach. See [the documentation](docs/README.md#user-specified-internal-functions).
