# Changelog

## Current develop

### Added (new features/APIs/variables/...)

### Changed (changing behavior/API/variables/...)
[[PR 266]](https://github.com/lanl/parthenon/pull/266): It is no longer necessary to specify Kokkos_ENABLE_OPENMP this is by default enabled, to turn off one can specify PARTHENON_DISABLE_OPENMP.

### Fixed (not changing behavior/API/variables/...)

### Removed

## Release 0.1.0
Date: 8/4/2020

Initial release of Parthenon AMR infrastructure.

### Changed
[[PR 214]](https://github.com/lanl/parthenon/pull/214): The weak linked routines for user-specified parthenon behavior have been removed in favor of a more portable approach. See [the documentation](docs/README.md#user-specified-internal-functions).

