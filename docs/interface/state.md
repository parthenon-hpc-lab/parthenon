# State Management

Parthenon manages simulation data through a hierarchy of classes designed to provide convenient state management but also high-performance in low-level, performance critical kernels.  This page gives an overview of the basic classes involved in state management.

# Metadata

The ```Metadata``` class provides a means of defining self-describing variables within Parthenon.  It's documentation can be found [here](Metadata.md).

# StateDescriptor

The ```StateDescriptor``` class is intended to be used to inform Parthenon about the needs of an application and store relevant parameters that control application-specific behavior at runtime.  The class provides several useful features and functions.
* ```bool AddField(const std::string& field_name, Metadata& m, DerivedOwnership owner=DerivedOwnership::unique)```
Provides the means to add new variables to a Parthenon-based application with associated ```Metadata```.  This function does not allocate any storage or create any of the objects below, it simply adds the name and ```Metadata``` to a list so that those objects can be populated at the appropriate time.
* ```void AddParam<T>(const std::string& key, T& value)``` adds a parameter (e.g. a timestep control coefficient, refinement tolerance, etc.) with name ```key``` and value ```value```.
* ```const T& Param(const std::string& key)``` provides the getter to access parameters previously added by ```AddParam```.
* ```void AddMeshBlockPack(const std::string &pack_name, const Function &packer)``` registers a callback that generates `MeshBlockPack`s on the `Mesh` as described [here](../mesh/packing.md)
* ```std::vector<std::shared_ptr<AMRCriteria>> amr_criteria``` holds a vector of criteria that Parthenon will make use of when tagging cells for refinement and derefinement.
* ```void (*FillDerived)(MeshBlockData<Real>& rc)``` is a function pointer (defaults to ```nullptr``` and therefore a no-op) that allows an application to provide a function that fills in derived quantities from independent state.
* ```Real (*EstimateTimestep)(MeshBlockData<Real>& rc)``` is a function pointer (defaults to ```nullptr``` and therefore a no-op) that allows an application to provide a means of computing stable/accurate timesteps.
* ```AmrTag (*CheckRefinement)(MeshBlockData<Real>& rc)``` is a function pointer (defaults to ```nullptr``` and therefore a no-op) that allows an application to define an application-specific refinement/de-refinement tagging function. 

In Parthenon, each ```MeshBlock``` owns a ```Packages_t``` object, which is a ```std::map<std::string, std::shared_ptr<StateDescriptor>>```.  The object is intended to be populated with a ```StateDescriptor``` object per package via an ```Initialize``` function as in the advection example [here](../example/advection/advection.cpp).  When Parthenon makes use of the ```Packages_t``` object, it iterates over all entries in the ```std::map```.



# ParArrayND

This provides a light wrapper around ```Kokkos::View``` with some convenience features.  It is described fully [here](../parthenon_arrays.md).

# CellVariable

The ```CellVariable``` class collects several associated objects that are needed to store, describe, and update simulation data.  ```CellVariable``` is templated on type ```T``` and includes the following member data (names preceded by ```_``` have private scope):

| Member Data | Description |
|-|-|
| ```ParArrayND<T> data``` | Storage for the cell-centered associated with the object. |
| ```ParArrayND<T> flux[3]``` | Storage for the face-centered intercell fluxes in each direction.<br>Only allocated for fields registered with the ```Metadata::Independent``` flag. |
| ```ParArrayND<T> coarse_s``` | Storage for coarse buffers need for multilevel setups. |
| ```Metadata m_``` | See [here](Metadata.md). |

Additionally, the class overloads the ```()``` operator to provide convenient access to the ```data``` array, though this may be less efficient than operating directly on ```data``` or a reference/copy of that array.

Finally, the ```bool IsSet(const MetadataFlag bit)``` member function provides a convenient mechanism to query whether a particular ```Metadata``` flag is set for the ```CellVariable```.

# FaceVariable (Work in progress...)

# EdgeVariable (Work in progress...)

# SparseVariable

The ```SparseVariable``` class is designed to support multi-component state where not all components may be present and therefore need to be stored.  At its core, the data is represented using a map that associates an integer ID to a ```std::shared_ptr<CellVariable<T>>```.  Since all ```CellVariable``` entries are assumed to have identical ```Metadata``` flags, the class provides an ```IsSet``` member function identical to the ```CellVariable``` class that applies to all variables stored in the map.  The ```Get``` method takes an integer ID as input and returns a reference to the associated ```CellVariable```, or throws a ```std::invalid_argument``` error if it does not exist.  The ```GetVector``` method returns a dense ```std::vector```, eliminating the sparsity but also the association to particular IDs.  The ```GetIndex``` method provides the index in this vector associated with a given sparse ID, and returns -1 if the ID does not exist.

# Container

The ```Container``` class provides a means of organizing and accessing simulation data.  New variables are added to a container via the ```Add``` member function and accessed via various ```Get*``` functions.  These ```Get*``` functions provide access to the various kinds of ```Variable``` objects described above, typically by name.

# ContainerCollection

The ```ContainerCollection``` class is the highest level abstraction in Parthenon's state management.  Each ```MeshBlock``` in a simulation owns a ```ContainerCollection``` that through the classes just described, manages all of the simulation data.  Every ```ContainerCollection``` is initialized with a ```Container``` named ```"base"```.  The ```Get``` function, when invoked without arguments, returns a reference to this base ```Container``` which is intended to contain all of the simulation data that persists between timesteps (if applicable).

The ```Add(const std::string& label, MeshBlockData<T>& src)``` member function creates a new ```Container``` with the provided label.  This new ```Container``` is populated with all of the variables in ```src```.  When a variable has the ```Metadata::OneCopy``` flag set, the variables in the new ```Container``` are are just shallow copies from ```src```, i.e. no new storage for data is allocated, the ```std::shared_ptr``` to the variable is just copied.  For variables that do not have ```Metadata::OneCopy``` set, new storage is allocated.  Once created, these new containers are accesible by calling ```Get``` with the name of the desired ```Container``` as an argument.  NOTE: The ```Add``` function checks if a ```Container``` by the name ```label``` already exists in the collection, immediately returning if one is found (or throwing a ```std::runtime_error``` if the new and pre-existing containers are not equivalent).  Therefore, adding a ```Container``` to the collection multiple times results in a single new container, with the remainder of the calls no-ops.

Two simple examples of usage of these new containers are 1) to provide storage for multistage integration schemes and 2) to provide a mechanism to allocate storage for right hand sides, deltas, etc.  Both of these usages are demonstrated in the advection example that ships with Parthenon.
