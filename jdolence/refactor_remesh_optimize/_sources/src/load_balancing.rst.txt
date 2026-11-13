.. _load_balancing:

Load Balancing
==============

Parthenon supports some load balancing capabilities. By default load
balancing is done "round robin" so that each MPI rank has a roughly
equal number of ``MeshBlock`` s. However, this behaviour can be
modified.

On a per ``MeshBlock`` basis, you call the
function

.. code:: cpp

   void MeshBlock::SetCostForLoadBalancing(double cost);

where the cost is any positive real number. For example, you might
call the above function in an application driver like:

.. code:: cpp

   for (int pmb : pmy_mesh->block_list) {
     pmb->SetCostForLoadBalancing(SomeHeuristicFunction(pmb));
   }

Then, if you set the following block in your input file:

::

   <parthenon/loadbalancing>
   balancer = manual

parthenon will try to give each MPI rank a set of meshblocks with
equal total cost. To disable this functionality and recover default
behaviour, set the ``balancer`` option to ``default``.

Remesh placement
----------------

AMR remeshing plans costs and ownership before changing the active forest or
constructing destination blocks. The default ``remesh_cost_policy = inherit``
preserves the existing behavior: unchanged blocks and refined children inherit
their source block cost, while a coarsened parent receives the mean cost of its
children. ``remesh_cost_policy = parthenon`` selects a deterministic framework
heuristic based on planned field storage and exact future particle counts.
Load-balancing-only migrations use the same inventory, placement, and transfer
pipeline with an identity topology. A load balancer supplies costs and invokes
placement; it does not define or modify data-transfer behavior.

Applications can select ``remesh_cost_policy = application`` and register a
descriptor-only callback. The callback runs during planning and must return a
finite positive cost without allocating destination data or communicating:

.. code:: cpp

   pman.app_input.EstimateRemeshBlockCost =
       [](const parthenon::remesh::CostContext &context) {
         const auto particles = context.inventory.swarms.front().active_count;
         return parthenon::remesh::CostEstimate{1.0 + particles, {}};
       };

The callback is a planning-only hook. It receives immutable descriptors, must
produce the same finite positive result on every rank, and must not allocate
destination data, communicate, or transfer fields or particles. Transfer
classification, layouts, sparse allocation closure, and numerical refinement
operators remain framework-owned sources of truth.

Optional ``remesh_host_memory_budget_mib`` and
``remesh_device_memory_budget_mib`` values constrain the contiguous SFC
placement and reject a remesh before destination construction when the planned
peak exceeds either per-rank budget. Current inventory and transfer-overhead
accounting is explicitly an estimate. Setting
``remesh_require_hard_memory_guarantee = true`` therefore rejects planning
until all application, finalization, transfer, and runtime storage has a
conservative declaration; Parthenon does not silently present an estimate as a
hard cap.

Remesh lifecycle guarantees
---------------------------

After topology and placement are accepted, Parthenon constructs the planned
destination blocks, applies planned sparse allocations, transfers fields, and
then performs finalization in one ordered path. Finalization completes shared
prolongation, redistributes swarms, updates active location/rank/cost lists,
rebuilds GMG block lists, invokes ``InitMeshBlockUserData`` for the active
blocks, corrects shared non-cell-centered values, rebuilds neighbors and
communication buffers, fills ghosts and derived quantities, and finally
recreates the base ``MeshData`` object. Application code must not retain
pre-remesh block, ``MeshData``, boundary-buffer, or swarm-cache references
across this boundary.

``parthenon/mesh/remesh_max_payload_count`` bounds the number of ``Real`` values
staged in one field-transfer wave. Records are never split internally, so the
limit must be at least as large as the largest indivisible field payload.
``parthenon/mesh/remesh_do_coalesced_comms`` controls aggregation of field records
within those bounded waves independently of ordinary boundary communication
coalescing. It defaults to ``false``.

.. note::

   Parthenon does not currently support timer based load balancing,
   however this is a planned feature.
