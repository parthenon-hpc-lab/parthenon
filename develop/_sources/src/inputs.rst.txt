.. _inputs:

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

+---------------------+---------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option              | Default | Type    | Description                                                                                                                                                                                            |
+=====================+=========+=========+========================================================================================================================================================================================================+
|| archive_parameters || false  || string || Produce a parameter file containing all parameters known to Parthenon. Set to `true` for an output file named `parthinput.archive`. Set to `timestamp` for a file with a name containing a timestamp. |
|| name               || none   || string || Name of this problem or initialization, prefixed to output files.                                                                                                                                     |
+---------------------+---------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


``<parthenon/time>``
--------------------

Options related to time-stepping and printing of diagnostic data.

+------------------------------+---------+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                       | Default | Type   | Description                                                                                                                                                           |
+==============================+=========+========+=======================================================================================================================================================================+
|| dt_ceil                     || none   || Real  || The maximum allowed timestep.                                                                                                                                        |
|| dt_factor                   || 2.0    || Real  || The maximum allowed relative increase of the timestep over the previous value.                                                                                       |
|| dt_floor                    || none   || Real  || The minimum allowed timestep.                                                                                                                                        |
|| dt_force                    || none   || Real  || Force the timestep to this value, ignoring all other conditions.                                                                                                     |
|| dt_init                     || none   || Real  || The maximum allowed timestep during the first cycle.                                                                                                                 |
|| dt_init_force               || none   || bool  || If set to true, force the first cycle's timestep to the value given by dt_init.                                                                                      |
|| dt_min                      || none   || Real  || If the timestep falls below dt_min for dt_min_cycle_limit cycles, Parthenon fatals.                                                                                  |
|| dt_min_cycle_limit          || 10     || int   || The maximum number of cycles the timestep can be below dt_min.                                                                                                       |
|| dt_max                      || none   || Real  || If the timestep falls below dt_max for dt_max_cycle_limit cycles, Parthenon fatals.                                                                                  |
|| dt_max_cycle_limit          || 1      || int   || The maximum number of cycles the timestep an be above dt_max.                                                                                                        |
|| dt_user                     || none   || Real  || Set a global timestep limit.                                                                                                                                         |
|| ncrecv_bdry_buf_timeout_sec || -1.0   || Real  || Timeout in seconds for the `ReceiveBoundaryBuffers` tasks. Disabed (negative) by default. Typically no need in production runs. Useful for debugging MPI calls.      |
|| ncycle_out                  || 1      || int   || Number of cycles between short diagnostic output to standard out containing, e.g., current time, dt, zone-update/wsec. Default: 1 (i.e, every cycle).                |
|| ncycle_out_mesh             || 0      || int   || Number of cycles between printing the mesh structure to standard out. Use a negative number to also print every time the mesh was modified. Default: 0 (i.e, off).   |
|| nlim                        || -1     || int   || Stop criterion on total number of steps taken. Ignored if < 0.                                                                                                       |
|| perf_cycle_offset           || 0      || int   || Skip the first N cycles when calculating the final performance (e.g., zone-cycles/wall_second). Allows to hide the initialization overhead in Parthenon.             |
|| tlim                        || none   || Real  || Stop criterion on simulation time.                                                                                                                                   |
+------------------------------+---------+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+


``<parthenon/mesh>``
--------------------

See the :ref:`amr` documentation for details of the required
parameters in ``<parthenon/mesh>`` and ``<parthenon/meshblock>``.

+--------+---------+------+---------------------------------------------------------+
| Option | Default | Type | Description                                             |
+========+=========+======+=========================================================+
| nghost | 2       | int  | Number of ghost cells for each mesh block on each side. |
+--------+---------+------+---------------------------------------------------------+


``<parthenon/sparse>``
----------------------

See the :ref:`sparse impl` documentation for details.

+--------------------+---------+--------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Option             | Default | Type   | Description                                                                                                                                  |
+====================+=========+========+==============================================================================================================================================+
|| alloc_threshold   || 1e-12  || float || Global (for all sparse variables) threshold to trigger allocation of a variable if cells in the receiving ghost cells are above this value. |
|| dealloc_count     || 5      || int   || First deallocate a sparse variable if the `dealloc_threshold` has been met in this number of consecutive cycles.                            |
|| dealloc_threshold || 1e-14  || float || Global (for all sparse variables) threshold to trigger deallocation if all active cells of a variable in a block are below this value.      |
|| enable_sparse     || `true` || bool  || If set to false, sparse variables will always be allocated, see also :ref:`sparse run-time`                                                 |
+--------------------+---------+--------+----------------------------------------------------------------------------------------------------------------------------------------------+

