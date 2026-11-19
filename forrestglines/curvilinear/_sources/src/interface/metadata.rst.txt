.. _metadata:

Metadata
========

Variables can be tagged with a variety of ``MetadataFlag`` values. These
flags primarily allow an application to tell Parthenon to apply certain
behaviors to each field.

Dependency Management
---------------------

Several packages may add variables of the same name. These metadata
flags tell ``Parthenon`` how to resolve conflicts.

-  ``Private``: private metadata means a variable is package specific.
   The package name is prepended to the variable name, like a namespace.
   Two different variables can have the same name if they live in
   different packages.
-  ``Provides``: variables with this metadata are shared, and demand
   they be the only ones to provide a variable. The metadata of these
   variables takes priority. If two packages provide the same variable
   with the Provides metadata, an error is raised at runtime. Provides
   is on by default unless overriden by another flag.
-  ``Requires``: variables with this metadata flag are not added by a
   package. The package simply declares it expects the variable to exist
   and if it doesn’t, an error is raised.
-  ``Overridable``: Variables with this metadata flag are provided by
   the package that registers them unless another package provides said
   variable (with the Provides flag), in which case, that other package
   determines what happens to that variable. If two packages request an
   Overridable variable, but it is not provided, it's undefined
   behaviour and Parthenon warns as such.

Variable Topology
-----------------

Topology essentially specifies on which place on the finite volume grid
the variable is defined. These fields specify what index space the
variable is allocated over. E.g. ``Metadata::Cell`` specifies that the
variable is allocated over the cell index space in each block.

The following fields specify the topology of the variable, and are
mutually exclusive:

-  ``Metadata::None``: no topology specified. The variable could be
   anywhere, or location is not a meaningful concept for this variable.
-  ``Metadata::Cell``: The variable is defined on cell centers. The
   variable is likely volume-averaged.
-  ``Metadata::Face``: The variable is defined on cell faces. The
   variable is likely area-averaged.
-  ``Metadata::Edge``: The variable is defined on cell edges. The
   variable is likely length-averaged.
-  ``Metadata::Node``: The variable is defined at nodes, i.e.,
   cell-corners. The variable might be volume-averaged, or defined
   pointwise.

Variable Behaviors
------------------

These flags can be used to tell an application code how to treat a
variable in relation to the problem.

-  ``Metadata::Advected`` implies a variable is advected with the flow
   of another variable, e.g., a velocity.
-  ``Metadata::Conserved`` implies a variable obeys a conservation law.
-  ``Metadata::Intensive`` implies that the value of a variable does not
   scale with volume.
-  ``Metadata::Sparse`` implies that the variable is sparse and hence it
   may not be allocated on all blocks.

Output
------

These flags specify how a variable interacts with I/O. Enable them to
enable output properties.

-  ``Metadata::Restart`` implies a variable is required in restart files

Tensor properties and boundaries
--------------------------------

For multidimensional variables, these flags specify how to treat the
individual components at boundaries. For concreteness, we will discuss
reflecting boundaries. But this may apply more broadly. A variable with
no flag set is assumed to be a *Scalar*. Scalars obey `Dirichlet
boundary
conditions <https://en.wikipedia.org/wiki/Dirichlet_boundary_condition>`__
at reflecting boundaries and are set to a constant value. The following
flags are mutually exclusive.

-  ``Metadata::Vector`` implies the variable transforms as a *vector* at
   reflecting boundaries. And so i-th component is flipped for a
   boundary in the i-th direction.
-  ``Metadata::Tensor`` is the generalization of the vector boundary
   condition, but for tensor quantities.

Independent/Derived
-------------------

These flags specify to an application code, and the infrastructure,
whether or not a variable is part of independent state. Derived
quantities can be calculated from the set of independent quantities,
while independent quantities cannot. The following flags are mutually
exclusive and required. All variables should be either independent or
derived.

-  ``Metadata::Independent`` implies the variable is part of independent
   state. In particular, implies data is in restart files and is
   prolongated/restricted during remeshing. Buffers for a coarse grid
   are allocated for independent variables.
-  ``Metadata::Derived`` implies the variable can be calculated, given
   the independent state. This is the default.

Communication
-------------

These flags specify both how ghost zones are treated, and whether
variables are copied or not in multiple stages.

-  ``Metadata::OneCopy`` are shared between stages. They are only ever
   allocated once.
-  ``Metadata::FillGhost`` specifies that ghost zones for this variable
   must be filled via communication or boundary conditions. This is not
   always required. ``OneCopy`` variables, for example, may not need
   this.

Ghost Zones, Communication, and Fluxes
--------------------------------------

Depending on a combination of flags, extra communication buffers and
classes may be allocated. The behaviours are the following:

-  If ``Metadata::FillGhosts`` is set, boundary conditions data is set,
   MPI communication buffers are allocated, and buffers for a coarse
   grid are allocated. These buffers are *one-copy*, meaning they are
   shared between all instances of a variable in all ``Containers`` in a
   ``DataCollection``.

-  If ``Metadata::WithFluxes`` is set, the flux vector for the variable
   is allocated. Note that it is necessary to set both
   ``Metadata::WithFluxes`` and ``Metadata::FillGhosts`` to send flux
   corrections across meshblocks.

-  If ``Metadata::ForceRemeshComm`` is set, the variable is communicated
   between ranks during remeshing. Variables with
   ``Metadata::Independent`` and/or ``Metadata::FillGhost`` are also
   automatically communicated when a block is communicated from one
   process to another. Other variables **are not** communicated across
   ranks, since the Parthenon model assumes that all fields are either
   ``Independent`` or ``Derived`` and that the ``Derived`` fields can be
   reconstructed from only the ``Independent`` fields by calling
   ``FillDerived``. Nevertheless, it is sometimes useful to pass certain
   ``Derived`` fields between ranks during remeshing rather than rebuild
   them (e.g. the initial guesses for a root find that may converge
   slowly or not at all without a good initial guess). *This flag should
   be used with caution, since it has the possibility the possibility to
   mask errors in the ``FillDerived`` implementation in downstream
   codes.*

Application Metadata Flags
---------------------------

Applications can allocate their own flags by calling
``Metadata::AllocateNewFlag("FlagName")``. For example:

.. code:: cpp

   using parthenon::Metadata;
   using parthenon::MetadataFlag;

   MetadataFlag const my_app_flag = Metadata::AllocateNewFlag("MyAppFlag");

These can be used in all the same contexts that the built-in metadata
flags are used. Parthenon will not interpret them in any way - it’s up
to the application to interpret them.

A user-registered metadata flag can be retrieved from the
infrastructure by, for example:

.. code:: cpp

   MetadataFlag const my_app_flag = Metadata::GetUserFlag("MyAppFlag");

Note that this call will return an error if a flag is requested that
hasn't been registered.

Flag Collections
-----------------

The ``Metadata::FlagCollection`` class provides a way to express a desire for
a collection of ``Parthenon`` fields that satisfy some combinations of
``MetadataFlag``\ s. In particular, a ``FlagCollection`` specifies for a
desire for fields with:

- At least **one** of the flags in the ``Unions`` property of the ``FlagCollection``

- **All** of the flags in the ``Intersections`` property of the ``FlagCollection``

- **None** of the flags in the ``Exclusions`` property of the ``FlagCollection``

Flag collections can be constructed from a C++
standard library container of ``MetadataFlag`` objects, or simply a
comma separated list of them. For example:

.. code:: cpp

   using parthenon::Metadata;
   using parthenon::MetadataFlag;
   using FS_t = Metadata::FlagCollection
   // Constructor from a container
   FS_t set1(std::vector<MetadataFlag>{Metadata::Cell, Metadata::Face});
   // Constructor from a comma separated list
   FS_t set2(Metadata::Requires, Metadata::Overridable);

By default constructor arguments go into the ``Intersections`` property
of the ``FlagCollection``. However, if a container is passed into the
constructor, you can also pass in an optional boolean flag to specify
whether or not you want to match **any** flags instead of **all**
flag. This adds the constructor arguments to the ``Unions``
property of the ``FlagCollection``.

.. code:: cpp

   // Implicit construction from a container, which
   // requests EITHER the following flags instead of BOTH
   FS_t set2({Metadata::Independent, Metadata::FillGhost}, true);

The flags contained in the ``Unions``, ``Intersections``, and
``Exclusions`` properties of the ``FlagCollection`` can be extracted via
equivalently named accessors, which return a ``std::set``. For
example:

.. code:: cpp

   const std::set<MetadataFlag> &u = set1.Unions();
   const std::set<MetadataFlag> &i = set1.Intersections();
   const std::set<MetadataFlag> &e = set1.Exclusions();

For the most part, you should not need these accessors. They are used
by Parthenon internal functions, such as variable and meshblock
packing, to compute the correct variables to pack.

You can add flags to these property fields with the ``TakeUnion``,
``TakeIntersection``, and ``Exclude`` methods. These methods take
either a standard library container of metadata flags, or another
``FlagCollection`` instance. For example, you could write:

.. code:: cpp

   FS_t my_set;
   my_set.TakeUnion(std::vector<MetadataFlag>{Flag1, Flag2});
   my_set.TakeIntersection(Flag3, Flag4);
   my_set.Exclude(Flag5, Flag6);

which expresses a desire for particles/fields with EITHER Flag1 or
Flag2 AND Flag3 AND Flag4 and NOT Flag5 or Flag6. Note that these
methods accept standard library containers as well as simple
comma-separated lists.

The ``FlagCollection`` class supports algebraic operations, although they are
not entirely consistent with standard arithmetic order of
operations. In particular:

.. code:: cpp

   // this could also be auto s = set1 || set2;
   auto s = set1 + set2;

produces a set s with the a unions field which is the set union of the
union fields of set1 and set2. However,

.. code:: cpp

   // this could also be s = set1 && set2;
   auto s = set1 * set2;

Produces a set s with a "unions" field of set1 and an intersections
field containing the original intersections of set1, and the
intersections of set2.

.. code:: cpp

   auto s = Set1 - Set2

Produces a set s with the "unions" and "intersections" fields of set1
and an exclusion field containing set1's exlcusion field as well as
ALL THREE fields (union, intersection, exclusion) contained by set2.

This feels unintuitive, but it makes expressions like

.. code:: cpp

   auto s = FS_t({Flag1, Flag2},true) * FS_t({Flag3, Flag4}) - FS_t({Flag5, Flag6})

behave in an intuitive way. This translates to a desire for
particles/fields with EITHER Flag1 or Flag2 AND Flag3 AND Flag4 and
NOT Flag5 or Flag6.

When in doubt about arithmetic with FlagCollections, aggressively use
parenthesis to enforce the order of operations you expect.

Note that the unary inverse operator is **not** supported.
