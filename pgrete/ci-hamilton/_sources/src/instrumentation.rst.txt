.. _instrumentation:

Performance Instrumentation
===========================

Parthenon provides several macros that make instrumenting your code simple.  For now,
these macros instantiate Kokkos profiling regions via calls to
``Kokkos::Profiling::pushRegion`` and ``Kokkos::Profiling::popRegion``, meaning all the
Kokkos profiling tools should work straightforwardly with Parthenon-based applications.

- ``PARTHENON_INSTRUMENT``: Instantiates an object that pushes a profiling region on
  construction and pops the region on destruction.  The name of the region is
  auto-generated and takes the form ``"file_name::line_number::function_name"``.  The region
  being profiled is controlled by invoking the macro at the appropriate scope.
- ``PARTHENON_INSTRUMENT_REGION(name)``: Same as ``PARTHENON_INSTRUMENT``, but uses the
  provided name instead of the auto-generated name.
- ``PARTHENON_INSTRUMENT_REGION_PUSH``: A trivial wrapper around ``pushRegion`` where
  the name is auto-generated as above.
- ``PARTHENON_INSTRUMENT_REGION_POP``: A trivial wrapper around ``popRegion``.

In addition to these macros, Parthenon provides the ``PARTHENON_AUTO_LABEL`` macro which
can be used to provide a label to kernels (e.g. through the various ``par_for``
functions).  The auto-generated name is the same as was described above.

Though not required, the use of the auto-generated names is highly recommended.  In
addition to avoiding possible name collisions, the auto-generated names provide a simple
structure that is amenable to post-processing profiling results to ease analysis.  For
example, the ``process_timer.py`` script that ships with Parthenon post-processes the
results of the Kokkos simple kernel timer output to provide a convenient view of the data.