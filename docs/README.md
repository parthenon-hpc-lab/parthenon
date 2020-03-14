# Parthenon documentation

This file provides an overview of different (not necessarily all) features in Parthenon
and how to use them.

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

## Kokkos/Wrapper related

- `par_for` wrappers use inclusive bounds, i.e., the loop will include the last index given
- `AthenaArrayND` arrays by default allocate on the *device* using default precision configured
- To create an array on the hosti with identical layout to the device array either use
  - `auto arr_host = Kokkos::create_mirror(arr_dev);` to always create a new array even if the device is associated with the host (e.g., OpenMP) or
  - `auto arr_host = Kokkos::create_mirror_view(arr_dev);` to create an array on the host if the HostSpace != DeviceSpace or get another reference to arr_dev through arr_host if HostSpace == DeviceSpace
- `par_for` and `Kokkos::deep_copy` by default use the standard stream (on Cuda devices) and are discouraged from use. Use `mb->par_for` and `mb->deep_copy` instead.
