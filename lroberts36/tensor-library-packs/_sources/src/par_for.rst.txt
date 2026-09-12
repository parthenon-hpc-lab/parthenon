.. par_for:

Parallelism
===========

The loop wrappers documented here abstract the ``Kokkos::parallel_*`` parallel launches. The wrappers
simplify the use of Kokkos `execution policies <https://kokkos.org/kokkos-core-wiki/API/core/Execution-Policies.html>`_
for multidimensional loops through a common interface using loop pattern tags. 

Additionally there is a provided ``parthenon::seq_for`` wrapper that uses a similar interface to perform
multidimensional sequential loops.

An example of usage can be found in `the unit
test <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/unit/kokkos_abstraction.cpp>`__

.. list-table:: parallel launches
   :widths: 25 25
   :header-rows: 1

   * - Parthenon
     - Kokkos
   * - ``par_for``
     - ``parallel_for``
   * - ``par_reduce``
     - ``parallel_reduce``
   * - ``par_scan``
     - ``parallel_scan``

Parallel launches are passed a string label, a set of inclusive loop bounds, a functor, and any extra arguments needed 
for parallel reductions/scans. Optionally a loop pattern tag and an execution space may be provided.
When ommitted the ``DEFAULT_LOOP_PATTERN`` is used.

.. code:: cpp

   parthenon::par_for(
       loop_pattern_tag, exec_space, PARTHENON_AUTO_LABEL, ks, ke, js, je, is, ie,
       KOKKOS_LAMBDA(const int k, const int j, const int i) {
         data(k, j, i) += 1.;
       });

.. list-table:: parallel launch parameters
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     -
   * - loop_pattern_tag
     - Determines the execution policy. See table below.
   * - exec_space
     - kokkos execution space
   * - loop bounds
     - inclusive start/end pairs for the multidimensional loop. Supported types are ``integral`` and ``parthenon::IndexRange``.
       Can be extended to accept other types (see below).
   * - functor
     - Defines the body of the parallel loop. 
       See `Kokkos programming guide <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#functors>`_
       for more.

.. list-table:: Loop Pattern tags
   :widths: 40 60
   :header-rows: 1

   * - Tag
     - Execution Policy
   * - ``loop_pattern_flatrange_tag``
     - Flattens all of the loops into a single ``Kokkos::RangePolicy``
   * - ``loop_pattern_simdfor_tag``
     - Maps to two C-style loops. The innermost gets decorated with a ``#pragma omp simd`` and the remaining
       loops are flattened into a single C-style for loop. Only supported on CPU.
   * - ``loop_pattern_mdrange_tag``
     - Maps all the loop bounds onto a ``Kokkos::MDRangePolicy``
   * - ``LoopPatternTeamThreadVec<Nt, Nv>()``
     - Maps onto a hierarchical parallel loop. The ``Nv`` inner loops are flattened onto a ``VectorRange`` policy,
       the next ``Nt`` onto a ``ThreadRange`` policy, and the remaining loops are 
       flattened into an outer ``TeamThreadRange``. The specializations ``loop_pattern_[tpttr|tptvr|tpttrtvr]_tag`` correspond
       to ``<1,0>``, ``<0,1>``, ``<1,1>`` respectively.


Cmake Options
-------------

``PAR_LOOP_LAYOUT`` controls the ``DEFAULT_LOOP_PATTERN`` macro.

.. list-table:: ``PAR_LOOP_LAYOUT`` options.
   :widths: 25 25
   :header-rows: 1

   * - ``PAR_LOOP_LAYOUT``
     -  Pattern Tag
   * - "MANUAL1D_LOOP"
     -  loop_pattern_flatrange_tag
   * - "SIMDFOR_LOOP"
     -  loop_pattern_simdfor_tag
   * - "MDRANGE_LOOP"
     -  loop_pattern_mdrange_tag
   * - "TP_TTR_LOOP"
     -  loop_pattern_tpttr_tag
   * - "TP_TVR_LOOP"
     -  loop_pattern_tptvr_tag
   * - "TPTTRTVR_LOOP"
     -  loop_pattern_tpttrtvr_tag

Adding New Loop Patterns
------------------------

All of the ``par_for*`` overloads get processed into the ``par_dispatch_impl`` struct that 
determines the types of the loop pattern, functor, functor arguments, loop bounds, and any
extra arguments need for scans/reductions. The struct implements overloads of the
``par_dispatch_impl::dispatch_impl`` method that are tagged using the ``PatternTag`` ``enum``
to specialize the ``LoopPatternTag`` struct. New loop patterns need to extend this enum and
provide an additional overload.

There is a chance that the requested loop pattern passed through ``parthenon::par_for``, for
example a ``loop_pattern_simdfor_tag`` ``DEFAULT_LOOP_PATTERN`` being used in a ``par_reduce``,
resulting in a conflict. For this reason the ``DispatchType`` type trait provides the
``DispatchType::GetPatternTag()`` method that processes the requested loop pattern and returns
a ``PatternTag`` and provides sensible fallbacks for the loop pattern if there are any conflicts.
In this way ``DEFAULT_LOOP_PATTERN`` can be reliably used.

Adding New Loop Bound Types
---------------------------

All of the loop bounds provided to any parallel wrapper gets processed by the ``LoopBoundTranslator`` 
to determine the rank of the multidimensional loop and translate the start/end pairs into an array
of ``IndexRange`` s. Each bound type gets processed individually and allows the flexibility to mix
loop bound types as long as they are supported.

New types can be provided by specializing the ``ProcessLoopBound`` struct in the ``parthenon`` namespace. 
These structs need to provide a ``GetNumBounds`` method to count the number of start/end bounds contained
in the type, as well as a ``GetIndexRanges`` method to fill the ``IndexRange`` bounds used in the
parallel dispatch.

