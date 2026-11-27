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

Applications can select ``remesh_cost_policy = application`` and register a
descriptor-only callback. The callback runs during planning and must return a
finite positive cost without allocating destination data or communicating:

.. code:: cpp

   pman.app_input.EstimateRemeshBlockCost =
       [](const parthenon::remesh::CostContext &context) {
         const auto particles = context.inventory.swarms.front().active_count;
         return parthenon::remesh::CostEstimate{1.0 + particles, {}};
       };

Optional ``remesh_host_memory_budget_mib`` and
``remesh_device_memory_budget_mib`` values constrain the contiguous SFC
placement and reject a remesh before destination construction when the planned
peak exceeds either per-rank budget. Current inventory and transfer-overhead
accounting is explicitly an estimate. Setting
``remesh_require_hard_memory_guarantee = true`` therefore rejects planning
until all application, finalization, transfer, and runtime storage has a
conservative declaration; Parthenon does not silently present an estimate as a
hard cap.

.. note::

   Parthenon does not currently support timer based load balancing,
   however this is a planned feature.
