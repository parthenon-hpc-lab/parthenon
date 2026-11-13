.. _load_balancing:

Load Balancing
==============

Parthenon supports some load balancing capabilities. By default load
balancing is done "round robin" so that each MPI rank has a roughly
equal number of ``MeshBlock`` s. However, this behaviour can be
modified.

On a per ``MeshBlock`` basis, you call the
function

.. cpp:function::

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

.. note::

   Parthenon does not currently support timer based load balancing,
   however this is a planned feature.
