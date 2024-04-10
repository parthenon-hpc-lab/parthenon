Mesh
====

The mesh object represents the mesh on a given processor/MPI rank. There
is one mesh per processor. The ``Mesh`` object owns all ``MeshBlock``\ s
on a given processor.

Looping over MeshBlocks
-----------------------

``MeshBlock``\ s are stored in a variable ``Mesh::block_list``, which is
an object of type

.. code:: c++

   using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;

and so to get the predicted time step for each mesh block, you can call:

.. code:: c++

   for (auto &pmb : pmesh->block_list) {
       std::cout << pmb->NewDt() << std::endl;
   }

where ``pmesh`` is a pointer to a ``Mesh`` object. This paradigm may
appear, for example, in an application driver.

Mesh Functions Controlling Evolulution and Diagnostics
------------------------------------------------------

Some mesh functions are stored as ``std::function`` members and are
called by the ``EvolutionDriver``. They may be overridden by the
application by reassigning the default values of the ``std::function``
member.

- ``PreStepUserWorkInLoop(Mesh*, ParameterInput*, SimTime const&)`` is
  called to perform mesh-wide work *before* the time-integration advance
  (the default is a no-op)
- ``PostStepUserWorkInLoop(Mesh*, ParameterInput*, SimTime const&)`` is
  called to perform mesh-wide work *after* the time-integration advance
  (the default is a no-op)
- ``PreStepUserDiagnosticsInLoop(Mesh*, ParameterInput*, SimTime const&)``
  is called to perform diagnostic calculations and/or edits *before* the
  time-integration advance. The default behavior calls to each package's
  (StateDesrcriptor's) ``PreStepDiagnostics`` method which, in turn,
  delegates to a ``std::function`` member that defaults to a no-op.
- ``PostStepUserDiagnosticsInLoop(Mesh*, ParameterInput*, SimTime const&)``
  is called to perform diagnostic calculations and/or edits *after* the
  time-integration advance. The default behavior calls to each package's
  (StateDesrcriptor's) ``PreStepDiagnostics`` method which, in turn,
  delegates to a ``std::function`` member that defaults to a no-op.
- ``UserMeshWorkBeforeOutput(Mesh*, ParameterInput*, SimTime const&)``
  is called to perform mesh-wide work immediately before writing an output
  (the default is a no-op). The most likely use case is to fill derived
  fields with updated values before writing them out to disk (or passing
  them to Ascent for in-situ analysis).

Multi-grid Grids Stored in ``Mesh``
-----------------------------------

If the parameter ``parthenon/mesh/multigrid`` is set to ``true``, the ``Mesh``
constructor and AMR routines populate both 
``std::vector<LogicalLocMap_t> Mesh::gmg_grid_locs`` and 
``std::vector<BlockList_t> gmg_block_lists``, where each entry into the vectors 
describes one level of the of the geometric multi-grid (GMG) mesh. For refined 
meshes, each GMG level only includes blocks that are at a given logical level 
(starting from the finest logical level on the grid and including both internal 
and leaf nodes in the refinement tree) as well as leaf blocks on the next coarser 
level that are neighbors to finer blocks, which implies that below the root grid 
level the blocks may not cover the entire mesh. For levels above the root grid, 
blocks may change shape so that they only cover the domain of the root grid. Note 
that leaf blocks may be contained in multiple blocklists, and the lists all point
to the same block (not a separate copy). To be explicit, when ``parthenon/mesh/multigrid`` is set to ``true`` blocks corresponding to *all* internal nodes of the refinement tree are created, in addition to the leaf node blocks that are normally created.

*GMG Implementation Note:*
The reason for including two levels in the GMG block lists is for dealing with 
accurately setting the boundary values of the fine blocks. Convergence can be poor 
or non-exististent if the fine-coarse boundaries of a given level are not 
self-consistently updated (since the boundary prolongation from the coarse grid to 
the fine grid also depends on interior values of the fine grid that are being updated 
by a smoothing operation). This means that each smoothing step, boundary communication 
must occur between all blocks corresponding to all internal and leaf nodes at a given 
level in the tree and with all leaf nodes at the next coarser level which abut blocks 
at the current level. Therefore, the GMG block lists contain blocks at two levels, but 
smoothing operations etc. should usually only occur on the subset of those blocks that 
are at the fine level.

To work with these GMG levels, ``MeshData`` objects containing these blocks can 
be recovered from a ``Mesh`` pointer using 

.. code:: c++

  auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", partition_idx);

This ``MeshData`` will include blocks at the current level and possibly some 
blocks at the next coarser level. Often, one will only want to operate on blocks
on the finer level (the coarser blocks are required mainly for boundary 
communication). To make packs containing only a subset of blocks from a 
GMG ``MeshData`` pointer ``md``, one would use 

.. code:: c++

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  for (int b = 0; b < nblocks; ++b)
    include_block[b] =
        (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());

  auto desc = parthenon::MakePackDescriptor<in, out>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);

In addition to creating the ``LogicalLocation`` and block lists for the GMG levels, 
``Mesh`` fills neighbor arrays in ``MeshBlock`` for intra- and inter-GMG block list 
communication (i.e. boundary communication and internal prolongation/restriction, 
respectively). Communication within and between GMG levels can be done by calling 
boundary communication routines with the boundary tags ``gmg_same``, 
``gmg_restrict_send``, ``gmg_restrict_recv``, ``gmg_prolongate_send``, 
``gmg_prolongate_recv`` (see :ref:`boundary_comm_tasks`). 

