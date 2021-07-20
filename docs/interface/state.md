# State Management

Parthenon manages simulation data through a hierarchy of classes designed to provide convenient state management but also high-performance in low-level, performance critical kernels.  This page gives an overview of the basic classes involved in state management.

# Metadata

The `Metadata` class provides a means of defining self-describing variables within Parthenon.  It's documentation can be found [here](Metadata.md).

# StateDescriptor

The `StateDescriptor` class is intended to be used to inform Parthenon about the needs of an application and store relevant parameters that control application-specific behavior at runtime.  The class provides several useful features and functions.
* `bool AddDenseField(const std::string& field_name, Metadata& m)` provides the means to add new
  dense variables to a Parthenon-based application with associated `Metadata`.  This function does
  not allocate any storage or create any of the objects below, it simply adds the name and
  `Metadata` to a list so that those objects can be populated at the appropriate time.
* `bool AddSparsePool(...)` either adds a given `SparsePool` or forwards the arguments to the
  `SparsePool` constructor. A `SparsePool` is a collection of sparse variable fields that share the
  same base name and `Metadata`, except that the shape, `Vector`/`Tensor` metadata flags, and
  component names can be specified per sparse id. Currently, sparse variables are allocated on all
  blocks just like dense variables, however, in a future upgrade, they will only be allocated on
  those blocks where the user explicitly allocates them or non-zero values are advected into.
* `void AddParam<T>(const std::string& key, T& value)` adds a parameter (e.g. a timestep control coefficient, refinement tolerance, etc.) with name `key` and value `value`.
* `void UpdateParam<T>(const std::string& key, T& value)`updates a parameter (e.g. a timestep control coefficient, refinement tolerance, etc.) with name `key` and value `value`. A parameter of the same type must exist.
* `const T& Param(const std::string& key)` provides the getter to access parameters previously added by `AddParam`.
* `void FillDerivedBlock(MeshBlockData<Real>* rc)` delgates to the `std::function` member `FillDerivedBlock` if set (defaults to `nullptr` and therefore a no-op) that allows an application to provide a function that fills in derived quantities from independent state per `MeshBlockData<Real>`.
* `void FillDerivedMesh(MeshData<Real>* rc)` delgates to the `std::function` member `FillDerivedMesh` if set (defaults to `nullptr` and therefore a no-op) that allows an application to provide a function that fills in derived quantities from independent state per `MeshData<Real>`.
* `Real EstimateTimestepBlock(MeshBlockData<Real>* rc)` delgates to the `std::function` member `EstimateTimestepBlock` if set (defaults to `nullptr` and therefore a no-op) that allows an application to provide a means of computing stable/accurate timesteps for a mesh block.
* `Real EstimateTimestepMesh(MeshData<Real>* rc)` delgates to the `std::function` member `EstimateTimestepBlock` if set (defaults to `nullptr` and therefore a no-op) that allows an application to provide a means of computing stable/accurate timesteps for a mesh block.
* `AmrTag CheckRefinement(MeshBlockData<Real>* rc)` delegates to the `std::function` member `CheckRefinementBlock` if set (defaults to `nullptr` and therefore a no-op) that allows an application to define an application-specific refinement/de-refinement tagging function.
* `void PreStepDiagnostics(SimTime const &simtime, MeshData<Real> *rc)` deletgates to the `std::function` member `PreStepDiagnosticsMesh` if set (defaults to `nullptr` an therefore a no-op) to print diagnostics before the time-integration advance
* `void PostStepDiagnostics(SimTime const &simtime, MeshData<Real> *rc)` deletgates to the `std::function` member `PostStepDiagnosticsMesh` if set (defaults to `nullptr` an therefore a no-op) to print diagnostics after the time-integration advance

The reasoning for providing `FillDerived*` and `EstimateTimestep*` function pointers appropriate for usage with both `MeshData` and `MeshBlockData` is to allow downstream applications better control over task/kernel granularity.  If, for example, the functionality needed in a package's `FillDerived*` function is minimal (e.g., computing velocity from momentum and mass), better performance may be acheived by making use of the `FillDerivedMesh` interface.  Note that applications and even individual packages can make simultaneous usage of _both_ `*Mesh` and `*Block` functions, so long as the appropriate tasks are called as needed by the application driver.

In Parthenon, each `Mesh` and `MeshBlock` owns a `Packages_t` object, which is a `std::map<std::string, std::shared_ptr<StateDescriptor>>`.  The object is intended to be populated with a `StateDescriptor` object per package via an `Initialize` function as in the advection example [here](../example/advection/advection.cpp).  When Parthenon makes use of the `Packages_t` object, it iterates over all entries in the `std::map`.  Note that it's often useful to add a `StateDescriptor` to the `Packages_t` object for the overall application, allowing for a convenient way to define global parameters, for example.

## History output

Parthenon allows packages to enroll an arbitrary number of "history" functions that are all
called at the interval according to the input parameters,
see [output documention](../outputs.md#History-Files).

To enroll functions create a list of callback function with the appropriate reduction operation:

```c++
// List (vector) of HistoryOutputVar that will all be enrolled as output variables
parthenon::HstVar_list hst_vars = {};

// Add a callback function
hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, MyHstFunction, "my label"));

// add callbacks for HST output identified by the `hist_param_key`
pkg->AddParam<>(parthenon::hist_param_key, hst_vars);
```

Here, `HistoryOutputVar` is a `struct` containing the global (over all blocks of all ranks) reduction operation, `MyHstFunction` is a callback function (see below), and `"my label"` is the string to
be used as the column heading of the output file.

Currently supported reductions are

- `UserHistoryOperation::sum`
- `UserHistoryOperation::min`
- `UserHistoryOperation::max`

which all match their respective MPI counterpart.
*Note*, in case of volume weighting being desired (e.g., to calculate the total value in
the simulation domain of some density) the volume weighting need to be done within the callback
function, see the [advection example](../../example/advection/advection_package.cpp).

Callback functions need to have the following signature
```c++
Real MyHstFunction(MeshData<Real> *md);
```
i.e., they will always work on `MeshData`.
*Note*, currently history output will always be calculated for the "base" container.
More specifically, the output machinery will automatically use (or create if non existent)
a single "base" `MeshData` object containing *all* blocks of a rank.
This simplifies the the logic for reductions over all blocks of a rank and also (generally)
resuls in better performance as the number of kernel calls is reduced.
However, this also implies the expectation that the "base" container holds the most recent
data at the end of a timestep.

# ParArrayND

This provides a light wrapper around `Kokkos::View` with some convenience features.  It is described fully [here](../parthenon_arrays.md).

# CellVariable

The `CellVariable` class collects several associated objects that are needed to store, describe, and update simulation data.  `CellVariable` is templated on type `T` and includes the following member data (names preceded by `_` have private scope):

| Member Data              | Description                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ParArrayND<T> data`     | Storage for the cell-centered associated with the object.                                                                                        |
| `ParArrayND<T> flux[3]`  | Storage for the face-centered intercell fluxes in each direction.<br>Only allocated for fields registered with the `Metadata::Independent` flag. |
| `ParArrayND<T> coarse_s` | Storage for coarse buffers need for multilevel setups.                                                                                           |
| `Metadata m_`            | See [here](Metadata.md).                                                                                                                         |

Additionally, the class overloads the `()` operator to provide convenient access to the `data` array, though this may be less efficient than operating directly on `data` or a reference/copy of that array.

Finally, the `bool IsSet(const MetadataFlag bit)` member function provides a convenient mechanism to query whether a particular `Metadata` flag is set for the `CellVariable`.

# FaceVariable (Work in progress...)

# EdgeVariable (Work in progress...)

# Sparse fields

Sparse fields can be added via the `StateDescriptor::AddSparsePool` function. A `SparsePool` is a
collection of sparse fields that share a common base name and metadata (except shape), but each
sparse ID produces a distinct `CellVariable`. For example, a `SparsePool` with base name `sparse`
and sparse IDs `{3, 10, 11, 2097}` will produce four `CellVariable`s: `sparse_3`, `sparse_10`,
`sparse_11`, and `sparse_2097`. These variables can be accessed either via their full name or the
combination of base name and sparse ID. Furthermore, in a future upgrade, the sparse fields will not
be allocated on all blocks but can be allocated only on specific blocks with a custom prescription
on how to handle when they advect to neighboring blocks.

The `SparseVariable` class is designed to support multi-component state where not all components may be present and therefore need to be stored.  At its core, the data is represented using a map that associates an integer ID to a `std::shared_ptr<CellVariable<T>>`.  Since all `CellVariable` entries are assumed to have identical `Metadata` flags, the class provides an `IsSet` member function identical to the `CellVariable` class that applies to all variables stored in the map.  The `Get` method takes an integer ID as input and returns a reference to the associated `CellVariable`, or throws a `std::invalid_argument` error if it does not exist.  The `GetVector` method returns a dense `std::vector`, eliminating the sparsity but also the association to particular IDs.  The `GetIndex` method provides the index in this vector associated with a given sparse ID, and returns -1 if the ID does not exist.

# MeshBlockData

The `MeshBlockData` class provides a means of organizing and accessing simulation data.  New `Variable`s are added to a `MeshBlockData` container via the `Add` member function and accessed via various `Get*` functions.  These `Get*` functions provide access to the various kinds of `Variable` objects described above, typically by name.

# DataCollection

The `DataCollection` class is the highest level abstraction in Parthenon's state management.  Each `MeshBlock` in a simulation owns a `DataCollection` that through the classes just described, manages all of the simulation data.  Every `DataCollection` is initialized with a `MeshBlockData` container named `"base"`.  The `Get` function, when invoked without arguments, returns a reference to this base `MeshBlockData` container which is intended to contain all of the simulation data that persists between timesteps (if applicable).

The `Add(const std::string& label, MeshBlockData<T>& src)` member function creates a new `MeshBlockData` container with the provided label.  This new `MeshBlockData` container is populated with all of the variables in `src`.  When a variable has the `Metadata::OneCopy` flag set, the variables in the new `MeshBlockData` container are just shallow copies from `src`, i.e. no new storage for data is allocated, the `std::shared_ptr` to the variable is just copied.  For variables that do not have `Metadata::OneCopy` set, new storage is allocated.  Once created, these new containers are accesible by calling `Get` with the name of the desired `MeshBlockData` container as an argument.  NOTE: The `Add` function checks if a `MeshBlockData` container by the name `label` already exists in the collection, immediately returning if one is found (or throwing a `std::runtime_error` if the new and pre-existing containers are not equivalent).  Therefore, adding a `MeshBlockData` container to the collection multiple times results in a single new container, with the remainder of the calls no-ops.

The overload `Add(const std::string &label, MeshBlockData<T> &src, const std::vector<std::string> &names)` provides the same functionality as the above `Add` function, but for a subset of variables provided in the vector of names.  This feature allows downstream applications to allocate storage in a more targeted fashion, as might be desirable to hold source terms for particular equations, for example.

Two simple examples of usage of these new containers are 1) to provide storage for multistage integration schemes and 2) to provide a mechanism to allocate storage for right hand sides, deltas, etc.  Both of these usages are demonstrated in the advection example that ships with Parthenon.
