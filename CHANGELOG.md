# Changelog

## Current develop

### Added (new features/APIs/variables/...)

### Changed (changing behavior/API/variables/...)
- [\#68](https://github.com/lanl/parthenon/issues/68) Moved default `par_for` wrappers to `MeshBlock` 
- [[PR 243]](https://github.com/lanl/parthenon/pull/243) Automatically find/check Python version used in regression tests. Bumps CMake minimum version to 3.12

### Fixed (not changing behavior/API/variables/...)
- [[PR 262]](https://github.com/lanl/parthenon/pull/262) Fix setting of "coverage" label in testing. Automatically applies coverage tag to all tests not containing "performance" label.

### Removed

## Release 0.1.0
Date: 8/4/2020

Initial release of Parthenon AMR infrastructure.

### Changed
- [[PR 214]](https://github.com/lanl/parthenon/pull/214): The weak linked routines for user-specified parthenon behavior have been removed in favor of a more portable approach. See [the documentation](docs/README.md#user-specified-internal-functions).
