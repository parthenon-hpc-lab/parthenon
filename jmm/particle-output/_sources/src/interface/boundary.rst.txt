Boundary related classes
========================

BoundaryData
------------

**Contains**: ``BoundaryStatus``, buffers, send/receive buffers and
requests, all PER NEIGHBOR

BoundaryCommunication
---------------------

Pure abstract base class, defines interfaces for managing
``BoundaryStatus`` flags and MPI requests

BoundaryBuffer
--------------

Pure abstract base class, defines interfaces for managing MPI
send/receive and loading/storing data from communication buffers.

BoundaryVariable
----------------

**Derived from**: ``BoundaryCommunication`` and ``BoundaryBuffer``

**Contains**: ``BoundaryData`` for variable and flux correction.

**Knows about**: ``MeshBlock`` and ``Mesh``

Still abstract base class, but implements some methods for sending and
receiving buffers.

BoundaryBase
------------

**Contains**: ``NeighborIndexes`` and ``NeighborBlock`` PER NEIGHBOR,
number of neighbors, neighbor levels

**Knows about**: ``MeshBlock``

Implements ``SearchAndSetNeighbors``

BoundaryValues
--------------

**Derived from**: ``BoundaryBase`` and ``BoundaryCommunication``

**Knows about**: ``MeshBlock``, all the ``BoundaryVariable`` connected
to variables of this block

Central class to interact with individual variable boundary data. Owned
by ``MeshBlock``.

CellCenteredBoundaryVariable
----------------------------

**Derived from**: ``BoundaryVariable``

**Contains**: Shallow copies of variable data, coarse buffer, and fluxes
(owned by ``CellVariable``)

Owned by ``CellVariable``, implements loading and setting boundary data,
sending and receiving flux corrections, and more.
