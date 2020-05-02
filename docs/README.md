# Parthenon documentation

This file provides an overview of different (not necessarily all) features in Parthenon
and how to use them.

## Description of examples

- [Calculate &pi](../example/calculate_pi)
- [Average face-centered variables to cell centers](../example/face_fields)
    
## Short feature description

Short feature descriptions may directly go in here with, for example, a link to a (unit)
test illustrating its usage.

### ParthenonManager

This class provides a streamlined capability to write new applications by providing a simple interface to initialize and finalize a simulation.  It's usage is straightforward and demonstrated in the &pi; [example](../example/calculate_pi/calculate_pi.cpp).

There are several weakly linked member functions that applications can (and often should) override to specialize:
* ParthenonManager::ProcessProperties
  * Constructs and returns a Properties_t object that is often filled with runtime specified (i.e. determined from the input file) settings and parameters.  For example, this might hold an equation of state.
* ParthenonManager::ProcessPackages
  * Constructs and returns a Packages_t object that contains a listing of all the variables and their metadata associated with each package.
* ParthenonManager::SetFillDerivedFunctions
  * Each package can register a function pointer in the Packages_t object that provides a callback mechanism for derived quantities (e.g. velocity, from momentum and mass) to be filled.  Additionally, this function provides a mechanism to register functions to fill derived quantities before and/or after all the individual package calls are made.  This is particularly useful for derived quantities that are shared by multiple packages.


## Long feature description

For features that require more detailed documentation a short paragraph or sentence here
is sufficient with a link to a more detailed description in a separate [file](feature.md).

### Kokkos/Wrapper related

- `par_for` wrappers use inclusive bounds, i.e., the loop will include the last index given
- `AthenaArrayND` arrays by default allocate on the *device* using default precision configured
- To create an array on the host with identical layout to the device array either use
  - `auto arr_host = Kokkos::create_mirror(arr_dev);` to always create a new array even if the device is associated with the host (e.g., OpenMP) or
  - `auto arr_host = Kokkos::create_mirror_view(arr_dev);` to create an array on the host if the HostSpace != DeviceSpace or get another reference to arr_dev through arr_host if HostSpace == DeviceSpace
- `par_for` and `Kokkos::deep_copy` by default use the standard stream (on Cuda devices) and are discouraged from use. Use `mb->par_for` and `mb->deep_copy` instead where `mb` is a `MeshBlock` (explanation: each `MeshBlock` has an `ExecutionSpace`, which may be changed at runtime, e.g., to a different stream, and the wrapper within a `MeshBlock` offer transparent access to the parallel region/copy where the `MeshBlock`'s `ExecutionSpace` is automatically used).

An arbitrary-dimensional wrapper for `Kokkos::Views` is available as
`ParArrayND`. See documentation [here](parthenon_arrays.md).

### State Management
[Full Documentation](interface/state.md)

Parthenon provides a convenient means of managing simulation data. Variables can be registered
with Parthenon to have the framework automatically manage the field, including
updating ghost cells, prolongation, restriction, and I/O.

### Application Drivers

A description of the Parthenon-provided classes that facilitate developing the high-level functionality of an application (e..g. time stepping) can be found [here](driver.md).

### Adaptive Mesh Refinement

A description of how to enable and extend the AMR capabilities of Parthenon is provided [here](amr.md).

### Tasks

The tasking capabilities in Parthenon are documented [here](tasks.md).

### Graphics

Check [here](graphics.md) for a description of how to get data out of Parthenon and how to visualize it.

