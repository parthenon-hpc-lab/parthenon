Input Parameters
================

Most Parthenon behavior is controlled via an input file, generally named
``parthinput.<problem-name>``. Parameters in this file take the form
``param = value``, with a particular type for each parameter set in the
Parthenon code.

Parameters are split into blocks, denoted by bracketed names
(``<blockname>``). All Parthenon-specific parameters are in blocks named
``<parthenon/x>``. Below is an incomplete list of some parameters
Parthenon accepts, split by block.

``<parthenon/job>``
-------------------

General parthenon options such as problem name and parameter handling.

+-----------------------------------------+--------+-----+-----------+
| Option                                  | D      | T   | De        |
|                                         | efault | ype | scription |
+=========================================+========+=====+===========+
| name                                    | none   | str | Name of   |
|                                         |        | ing | this      |
|                                         |        |     | problem   |
|                                         |        |     | or        |
|                                         |        |     | initia    |
|                                         |        |     | lization, |
|                                         |        |     | prefixed  |
|                                         |        |     | to output |
|                                         |        |     | files.    |
+-----------------------------------------+--------+-----+-----------+
| archive_parameters                      | false  | str | Produce a |
|                                         |        | ing | parameter |
|                                         |        |     | file      |
|                                         |        |     | c         |
|                                         |        |     | ontaining |
|                                         |        |     | all       |
|                                         |        |     | p         |
|                                         |        |     | arameters |
|                                         |        |     | known to  |
|                                         |        |     | P         |
|                                         |        |     | arthenon. |
|                                         |        |     | Set to    |
|                                         |        |     | ``true``  |
|                                         |        |     | for an    |
|                                         |        |     | output    |
|                                         |        |     | file      |
|                                         |        |     | named     |
|                                         |        |     | ``par     |
|                                         |        |     | thinput.a |
|                                         |        |     | rchive``. |
|                                         |        |     | Set to    |
|                                         |        |     | ``ti      |
|                                         |        |     | mestamp`` |
|                                         |        |     | for a     |
|                                         |        |     | file with |
|                                         |        |     | a name    |
|                                         |        |     | c         |
|                                         |        |     | ontaining |
|                                         |        |     | a         |
|                                         |        |     | t         |
|                                         |        |     | imestamp. |
+-----------------------------------------+--------+-----+-----------+

``<parthenon/time>``
--------------------

Options related to time-stepping and printing of diagnostic data.

+-----------------------------------------+--------+-----+-----------+
| Option                                  | D      | T   | De        |
|                                         | efault | ype | scription |
+=========================================+========+=====+===========+
| tlim                                    | none   | fl  | Stop      |
|                                         |        | oat | criterion |
|                                         |        |     | on        |
|                                         |        |     | s         |
|                                         |        |     | imulation |
|                                         |        |     | time.     |
+-----------------------------------------+--------+-----+-----------+
| nlim                                    | -1     | int | Stop      |
|                                         |        |     | criterion |
|                                         |        |     | on total  |
|                                         |        |     | number of |
|                                         |        |     | steps     |
|                                         |        |     | taken.    |
|                                         |        |     | Ignored   |
|                                         |        |     | if < 0.   |
+-----------------------------------------+--------+-----+-----------+
| perf_cycle_offset                       | 0      | int | Skip the  |
|                                         |        |     | first N   |
|                                         |        |     | cycles    |
|                                         |        |     | when      |
|                                         |        |     | ca        |
|                                         |        |     | lculating |
|                                         |        |     | the final |
|                                         |        |     | pe        |
|                                         |        |     | rformance |
|                                         |        |     | (e.g.,    |
|                                         |        |     | zone-cy   |
|                                         |        |     | cles/wall |
|                                         |        |     | _second). |
|                                         |        |     | Allows to |
|                                         |        |     | hide the  |
|                                         |        |     | initi     |
|                                         |        |     | alization |
|                                         |        |     | overhead  |
|                                         |        |     | in        |
|                                         |        |     | P         |
|                                         |        |     | arthenon, |
|                                         |        |     | which     |
|                                         |        |     | usually   |
|                                         |        |     | takes     |
|                                         |        |     | place in  |
|                                         |        |     | the first |
|                                         |        |     | cycles    |
|                                         |        |     | when      |
|                                         |        |     | C         |
|                                         |        |     | ontainers |
|                                         |        |     | are       |
|                                         |        |     | a         |
|                                         |        |     | llocated, |
|                                         |        |     | etc.      |
+-----------------------------------------+--------+-----+-----------+
| ncycle_out                              | 1      | int | Number of |
|                                         |        |     | cycles    |
|                                         |        |     | between   |
|                                         |        |     | short     |
|                                         |        |     | d         |
|                                         |        |     | iagnostic |
|                                         |        |     | output to |
|                                         |        |     | standard  |
|                                         |        |     | out       |
|                                         |        |     | co        |
|                                         |        |     | ntaining, |
|                                         |        |     | e.g.,     |
|                                         |        |     | current   |
|                                         |        |     | time, dt, |
|                                         |        |     | zone-upd  |
|                                         |        |     | ate/wsec. |
|                                         |        |     | Default:  |
|                                         |        |     | 1 (i.e,   |
|                                         |        |     | every     |
|                                         |        |     | cycle).   |
+-----------------------------------------+--------+-----+-----------+
| ncycle_out_mesh                         | 0      | int | Number of |
|                                         |        |     | cycles    |
|                                         |        |     | between   |
|                                         |        |     | printing  |
|                                         |        |     | the mesh  |
|                                         |        |     | structure |
|                                         |        |     | (e.g.,    |
|                                         |        |     | total     |
|                                         |        |     | number of |
|                                         |        |     | M         |
|                                         |        |     | eshBlocks |
|                                         |        |     | and       |
|                                         |        |     | number of |
|                                         |        |     | M         |
|                                         |        |     | eshBlocks |
|                                         |        |     | per       |
|                                         |        |     | level) to |
|                                         |        |     | standard  |
|                                         |        |     | out. Use  |
|                                         |        |     | a         |
|                                         |        |     | negative  |
|                                         |        |     | number to |
|                                         |        |     | also      |
|                                         |        |     | print     |
|                                         |        |     | every     |
|                                         |        |     | time the  |
|                                         |        |     | mesh was  |
|                                         |        |     | modified. |
|                                         |        |     | Default:  |
|                                         |        |     | 0 (i.e,   |
|                                         |        |     | off).     |
+-----------------------------------------+--------+-----+-----------+
| ncrecv_bdry_buf_timeout_sec             | -1.0   | R   | Timeout   |
|                                         |        | eal | in        |
|                                         |        |     | seconds   |
|                                         |        |     | for the   |
|                                         |        |     | ``        |
|                                         |        |     | cell_cent |
|                                         |        |     | ered_bvar |
|                                         |        |     | s::Receiv |
|                                         |        |     | eBoundary |
|                                         |        |     | Buffers`` |
|                                         |        |     | tasks.    |
|                                         |        |     | Disabed   |
|                                         |        |     | (         |
|                                         |        |     | negative) |
|                                         |        |     | by        |
|                                         |        |     | default.  |
|                                         |        |     | Typically |
|                                         |        |     | no need   |
|                                         |        |     | in        |
|                                         |        |     | p         |
|                                         |        |     | roduction |
|                                         |        |     | runs.     |
|                                         |        |     | Useful    |
|                                         |        |     | for       |
|                                         |        |     | debugging |
|                                         |        |     | MPI       |
|                                         |        |     | calls.    |
+-----------------------------------------+--------+-----+-----------+

``<parthenon/mesh>``
--------------------

See the `AMR <amr.md>`__ documentation for details of the required
parameters in ``<parthenon/mesh>`` and ``<parthenon/meshblock>``.

+-----------------------------------------+--------+-----+-----------+
| Option                                  | D      | T   | De        |
|                                         | efault | ype | scription |
+=========================================+========+=====+===========+
| nghost                                  | 2      | int | Number of |
|                                         |        |     | ghost     |
|                                         |        |     | cells for |
|                                         |        |     | each mesh |
|                                         |        |     | block on  |
|                                         |        |     | each      |
|                                         |        |     | side.     |
+-----------------------------------------+--------+-----+-----------+

``<parthenon/sparse>``
----------------------

See the `Sparse <interface/sparse.md>`__ documentation for details.

+-----------------------------------------+--------+-----+-----------+
| Option                                  | D      | T   | De        |
|                                         | efault | ype | scription |
+=========================================+========+=====+===========+
| enable_sparse                           | ``     | b   | If set to |
|                                         | true`` | ool | false,    |
|                                         |        |     | sparse    |
|                                         |        |     | variables |
|                                         |        |     | will      |
|                                         |        |     | always be |
|                                         |        |     | a         |
|                                         |        |     | llocated, |
|                                         |        |     | see also  |
|                                         |        |     | `sparse   |
|                                         |        |     | do        |
|                                         |        |     | c <interf |
|                                         |        |     | ace/spars |
|                                         |        |     | e.md#run- |
|                                         |        |     | time>`__. |
+-----------------------------------------+--------+-----+-----------+
| alloc_threshold                         | 1e-12  | fl  | Global    |
|                                         |        | oat | (for all  |
|                                         |        |     | sparse    |
|                                         |        |     | v         |
|                                         |        |     | ariables) |
|                                         |        |     | threshold |
|                                         |        |     | to        |
|                                         |        |     | trigger   |
|                                         |        |     | a         |
|                                         |        |     | llocation |
|                                         |        |     | of a      |
|                                         |        |     | variable  |
|                                         |        |     | if cells  |
|                                         |        |     | in the    |
|                                         |        |     | receiving |
|                                         |        |     | ghost     |
|                                         |        |     | cells are |
|                                         |        |     | above     |
|                                         |        |     | this      |
|                                         |        |     | value.    |
+-----------------------------------------+--------+-----+-----------+
| dealloc_threshold                       | 1e-14  | fl  | Global    |
|                                         |        | oat | (for all  |
|                                         |        |     | sparse    |
|                                         |        |     | v         |
|                                         |        |     | ariables) |
|                                         |        |     | threshold |
|                                         |        |     | to        |
|                                         |        |     | trigger   |
|                                         |        |     | dea       |
|                                         |        |     | llocation |
|                                         |        |     | if all    |
|                                         |        |     | active    |
|                                         |        |     | cells of  |
|                                         |        |     | a         |
|                                         |        |     | variable  |
|                                         |        |     | in a      |
|                                         |        |     | block are |
|                                         |        |     | below     |
|                                         |        |     | this      |
|                                         |        |     | value.    |
+-----------------------------------------+--------+-----+-----------+
| dealloc_count                           | 5      | int | First     |
|                                         |        |     | d         |
|                                         |        |     | eallocate |
|                                         |        |     | a sparse  |
|                                         |        |     | variable  |
|                                         |        |     | if the    |
|                                         |        |     | ``d       |
|                                         |        |     | ealloc_th |
|                                         |        |     | reshold`` |
|                                         |        |     | has been  |
|                                         |        |     | met in    |
|                                         |        |     | this      |
|                                         |        |     | number of |
|                                         |        |     | co        |
|                                         |        |     | nsecutive |
|                                         |        |     | cycles.   |
+-----------------------------------------+--------+-----+-----------+
