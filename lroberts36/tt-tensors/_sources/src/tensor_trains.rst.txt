Tensor Trains
=============

Parthenon provides tensor-train (TT) types and operations in ``src/tensors/``
(namespace ``parthenon::tensor``). This page describes the data model: what each
type owns, what copying a train does, and when a device pack stays valid. The
operations themselves are documented in ``tt_operations.hpp``.

.. note::
   This page is for developers working on the tensor-train layer itself.
   Tensor-train fields on the mesh are meant to hide these details from
   application code.

A tensor train is a sequence of cores. Core :math:`c` is a three-index array of
shape :math:`r_{c-1} \times n_c \times r_c`, where :math:`n_c` is the physical
dimension of the core and :math:`r_{c-1}` and :math:`r_c` are its left and right
bond ranks. In the code these are ``LR()``, ``DD()`` and ``RR()``. A train whose
outer bonds are both one is *closed* and represents a scalar-valued tensor. A
train with dangling outer bonds is *open*. Sums and rounding require closed
trains.

.. tip::
   In application code, hold trains through ``std::shared_ptr<TensorTrain>`` and
   pass them around by pointer. Avoid copying ``TensorTrain`` or
   ``TensorCoreHost`` objects. A copy is neither a deep copy nor a clean alias
   (see `Copy semantics`_), and the rules below only matter if you make one. Use
   ``DeepCopy()`` when you need an independent train. Host packs take
   ``std::vector<std::shared_ptr<train_t>>`` directly.

Types
-----

All types are templated on a traits class ``TTraits``, and each has a default
alias that uses ``DefaultTTraits``.

=========================================  ===========================  ======================================
Type                                       Default alias                Role
=========================================  ===========================  ======================================
``TensorTraits<Device, Real, bool>``       ``DefaultTTraits``           Device, scalar type and storage model
``TensorTrainT<TTraits>``                  ``TensorTrain``              Owns a ``std::vector`` of host cores
``TensorCoreHostT<TTraits, Storage>``      ``TensorCoreHost``           Owns the storage of one core
``TensorCoreDeviceT<TTraits, Storage>``    ``TensorCoreDevice``         Non-owning core descriptor for kernels
``TensorTrainHostPackT<TTraits>``          ``TensorTrainHostPack``      Non-owning (var, block) grid of trains
``TensorPackT<TTraits>``                   ``TensorPack``               Device view of core descriptors
=========================================  ===========================  ======================================

``TensorCoreHostT`` is the host-side object that owns
device data. ``TensorCoreDeviceT`` is a small value type that a kernel can
capture and index as ``core(l, d, r)``.

Ownership runs in one direction:

.. code-block:: text

   std::shared_ptr<TensorTrain>          (application)
     -> TensorTrain                       owns its cores
        -> TensorCoreHost                 owns its storage
           -> storage (fibers or buffer)  owns the numbers
   TensorTrainHostPack                    points at trains, owns nothing
     -> TensorPack                        owns a device array of descriptors,
        -> TensorCoreDevice               which point into the storage

Storage models
--------------

``TensorTraits``' boolean parameter ``DFastestMoving`` selects one of two storage
models. The two exist so they can be benchmarked against each other. Kernels
index cores only through ``core(l, d, r)`` and the unfoldings in ``TTraits``, so
they compile for both.

* **Fiber storage** (``d_fastest_moving = true``, ``FiberTTraits``, the default).
  A core is an :math:`r_{c-1} \times r_c` grid of *fibers*, and each fiber is a
  separately allocated device array of length :math:`n_c`. Access along the
  physical index is contiguous, which coalesces on the large spatial core.
  Changing a core's ranks can reuse existing fibers by handle instead of
  copying data. For example, the fiber ``DestructiveSum`` assembles the
  block-diagonal result without running a kernel.
* **r-fastest storage** (``d_fastest_moving = false``, ``ContiguousTTraits``). A
  core is one ``[lr][dd][rr]`` buffer with the right rank index fastest. Changing
  a core's ranks allocates a new buffer and copies into it.

The layout is always ``Kokkos::LayoutRight``. The r-fastest indexing, the
scratch cores and the unfoldings all assume it.

The fiber-core invariant
~~~~~~~~~~~~~~~~~~~~~~~~

A fiber core involves four levels:

1. **Fiber data:** managed 1-D device views, one per :math:`(l, r)`. These hold
   the numbers.
2. **host_fibers:** a host view of managed fiber handles. This table keeps the
   fibers alive through their reference counts.
3. **device_managed_fibers:** a managed device view of *unmanaged* fiber views.
   It owns only its own memory. Its pointers are valid only while a
   ``host_fibers`` table that holds the same fibers is alive.
4. **FiberStorageDevice:** an unmanaged view of level 3, owning nothing. This is
   what a ``TensorCoreDevice`` in a pack holds.

``host_fibers`` and ``device_managed_fibers`` always describe the same grid of
the same fibers. Only two routines change them. ``RebuildOuterViews`` builds both
into freshly allocated views, and ``Release()`` empties both. Neither table is
ever modified in place. Every operation that changes a fiber core's ranks goes
through ``RebuildOuterViews``. This includes ``ReduceSize``, ``RebuildFibers``
and the fiber ``DestructiveSum``.

Copy semantics
--------------

``TensorTrain`` and ``TensorCoreHost`` have *reference semantics*. Copying one
makes a new handle to the same numbers:

* **Copy construction or assignment** shares the numerical data. For fiber
  storage it shares the fibers (level 1) and gives the copy its own levels 2
  and 3. For r-fastest storage it shares the buffer.
* **Move** transfers everything and leaves the source empty.
* **DeepCopy()** duplicates the data. ``DeepCopyTrains`` does the same for a
  vector of trains.

A copy is not a full alias. It shares the numbers but not the structure (the
ranks and the tables). After a copy:

* Writes through either object's device pack are seen by both, as long as
  neither object has been reshaped.
* Reshaping one object rebinds only that object. Reshaping includes
  ``BuildFresh``, ``ReduceSize``, ``Clear``, assigning a new train, and every op
  that changes ranks. The other object keeps its old shape and data.
* An in-place op on one object can therefore leave the other inconsistent:

  * ``RoundGramSVD`` and ``RoundOseledetsSVD`` overwrite the existing cores and
    then shrink them. A copy taken before rounding keeps the old ranks over data
    that has been partly overwritten.
  * With fiber storage, ``DestructiveSum(A, B)`` moves A's and B's fibers into
    A's new cores. A copy of B taken beforehand still points at fibers that now
    belong to A. With r-fastest storage the result is a new buffer and the copy
    is unaffected. The behaviour differs between the storage models.

These semantics exist so that ``std::vector<TensorTrain>`` can copy and move
its elements safely. Holding trains through ``std::shared_ptr`` avoids the
question entirely, because there is only ever one train object.

Packs and lifetimes
-------------------

``TensorTrainHostPackT`` is a host-side ``(var, block)`` grid of non-owning
``train_t *``. It can be built from:

* a ``std::vector<train_t>&``
* a ``const std::vector<train_t*>&``
* a ``std::vector<std::shared_ptr<train_t>>&``
* a ``VarBlockGrid<train_t*>``, for several variables

Every operation takes host packs. The operation reshapes the pointed-to trains
in place, so the owner sees the result without rebinding. Each operation then
builds its own device packs.

``TensorPackT`` holds a device array of ``TensorCoreDevice`` descriptors indexed
``(block, var, core)``. It is built by ``MakeDevicePack()``. The descriptors
point into the trains' storage without owning it, so a device pack is valid only
while:

* every train it was built from is alive, and
* no core of those trains has been reshaped.

After any reshape, build a new device pack. Never cache one across an operation
that may change ranks.

Device-side unfoldings (``TTraits::GetHorizontalUnfolding`` and the others) hold
a reference to a core. Build them inside the kernel. One built on the host and
captured by a lambda would dangle.

.. note::
    This file was made in part with generative AI.
