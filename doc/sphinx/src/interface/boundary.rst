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
