.. _analysis:

Analysis Initialization
=======================

Parthenon applications can load a previously written HDF5 or openPMD output and run
application-native diagnostics without advancing the simulation. This *analysis mode*
reuses the application's packages, mesh data structures, boundary communication, and
output system, but it is distinct from a normal restart.

Analysis mode is useful when a diagnostic is easier to express with the application's
existing C++ kernels than with an external postprocessing tool. Typical examples include
computing derived fields, histograms, reductions, or converted output formats after a
simulation has completed.

Running an analysis
-------------------

Invoke an application with both an analysis source and a separate analysis input file:

.. code-block:: console

   ./my-application -a /path/to/source.phdf -i parthinput.analysis

The ``-a`` and ``-r`` options are mutually exclusive. Unlike ``-r``, ``-a`` requires an
input file because the analysis input defines the diagnostics and outputs to perform.

The supported source formats depend on the configured I/O backends:

* ``.phdf`` and ``.rhdf`` files require HDF5 support.
* ``.bp`` openPMD series require openPMD and ADIOS2 support. A trailing slash on the
  series directory is accepted.

An unsupported suffix, or a source whose backend was disabled at build time, produces a
startup error.

Analysis and restart differ in several important ways:

* Analysis does not set ``Globals::is_restart``. Applications can use
  ``ParthenonManager::IsAnalysis()`` when they need to distinguish analysis from both a
  fresh run and a restart.
* The source mesh hierarchy and selected field data are loaded, but problem generators
  are not called.
* The evolution loop and timestep initialization are skipped, so the analysis run does
  not advance the source cycle or time.
* Boundary communication is performed after the selected fields are loaded.
* ``UserWorkBeforeLoop`` runs once, followed by the output pass. Output callbacks run if
  at least one output is selected.

Selecting source fields
-----------------------

The optional ``fields`` entry in ``<parthenon/analysis>`` limits which mesh fields are
loaded:

.. code-block:: text

   <parthenon/analysis>
   fields = density, pressure, tracers_7

Names are logical Parthenon field labels, including the sparse ID suffix for an
individual sparse field. They are not HDF5 dataset paths or openPMD mesh-record names.

If ``fields`` is omitted or empty, Parthenon requests every field discoverable in the
source. If it is supplied, Parthenon loads the intersection of the requested names and
the source catalog. Unknown names are ignored. A field must also be registered by the
application's packages before it can be included in the resolved analysis state.

Current HDF5 and openPMD outputs store an ``OutputDatasetNames`` catalog containing the
logical labels. Legacy HDF5 files fall back to compatible root-dataset discovery.
OpenPMD format-version-2 files can recover labels from mesh-record names. Format-version-1
openPMD files do not contain an unambiguous catalog, so ``fields`` must be supplied
explicitly for those files.

Analysis-only fields and swarms
-------------------------------

Packages can register data needed only while postprocessing by adding
``Metadata::Analysis``. The flag is supported for dense fields, sparse fields, and
swarms. Such data is included in the resolved analysis state even when it is absent from
the source file.

For example:

.. code-block:: cpp

   Metadata diagnostic({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                        Metadata::Analysis});
   package->AddField("diagnostic", diagnostic);

   Metadata sparse_diagnostic({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                               Metadata::Sparse, Metadata::Analysis});
   package->AddSparsePool("diagnostic_pool", sparse_diagnostic,
                          std::vector<int>{7});

``Metadata::Analysis`` controls state construction; it does not initialize values.
Applications must populate analysis-only data in a callback that runs during analysis.
``UserWorkBeforeOutput`` is usually the appropriate location because it executes after
the source data has been loaded and communicated and immediately before output is
written. Applications must allocate analysis-only sparse fields before writing them.

File-backed fields are converted into restart-loaded derived fields for analysis. Their
shape, component labels, and applicable refinement operations are retained, while
independent ownership and flux relationships are removed. They participate in the
boundary communication required to make the loaded data usable by diagnostics.

Input precedence and source time
--------------------------------

Analysis parameters are assembled in this order, with each later source overriding
earlier values:

#. The input embedded in the source output.
#. The source iteration's actual ``start_time``, ``dt``, and ``ncycle`` metadata.
#. The required analysis input supplied with ``-i``.
#. Command-line assignments.

This allows an analysis to inherit the source mesh and package configuration while
changing boundaries, diagnostic settings, output definitions, or time parameters. For
example:

.. code-block:: console

   ./my-application -a source.out0.final.phdf -i parthinput.analysis \
       parthenon/time/ncycle=100 Advection/cfl=0.5

The source mesh geometry must remain compatible with the stored hierarchy. Boundary
conditions may be overridden by the analysis input and are applied when the loaded data
is communicated.

Selecting analysis outputs
--------------------------

Set ``analysis_output = true`` on output blocks that must be triggered by the analysis
signal:

.. code-block:: text

   <parthenon/job>
   problem_id = analysis

   <parthenon/analysis>
   fields = density, pressure

   <parthenon/output0>
   file_type = hdf5
   dt = 1.0
   analysis_output = true
   variables = density, pressure, diagnostic

The output block must still be enabled with a nonnegative ``dt`` or ``dn``. The
``analysis_output`` setting forces the block to run for the analysis signal regardless
of whether its normal cadence is due. Other enabled output blocks continue to obey their
ordinary scheduling conditions, so use negative ``dt`` and ``dn`` values to disable
blocks that should never run during this invocation.

Analysis output uses the normal numbered-output path rather than the ``final`` label.
The initial number can reflect counters stored in the source file. No additional final
output is written after the analysis pass.

``UserWorkBeforeOutput`` is called before the first selected non-history output, so it
can compute analysis-only fields and diagnostics. Analysis can read one supported
backend and write another; for example, an openPMD source can produce HDF5 diagnostics
when both backends are enabled.

.. warning::

   An output written during analysis contains only the resolved analysis state. Parthenon
   marks this state in the embedded input and rejects using the resulting file as a
   normal restart source.

Complete example
----------------

The advection example includes an analysis input that loads two fields, computes any
application diagnostics in ``UserWorkBeforeOutput``, and writes one HDF5 analysis
output:

.. literalinclude:: ../../../example/advection/parthinput.analysis
   :language: text

Run it against an existing output:

.. code-block:: console

   ./advection-example -a advection.out0.00010.phdf \
       -i ../../example/advection/parthinput.analysis

Change ``file_type`` to ``openpmd`` to write an openPMD result when that backend is
enabled. The source and destination backends do not need to match.

The regression fixtures in ``tst/regression/test_suites/analysis_data`` and
``tst/regression/test_suites/restart_opmd`` provide maintained HDF5 and openPMD examples.

Troubleshooting
---------------

No analysis output is produced
   Confirm that the output block has a nonnegative ``dt`` or ``dn`` and
   ``analysis_output = true``.

A requested field is missing
   Confirm that the logical label is present in the source and registered by the
   application's packages. For legacy openPMD format-version-1 files, list every desired
   field explicitly.

Ghost values are not suitable for the diagnostic
   Ensure the field metadata supports the required refinement and boundary operations,
   and request ghost zones in the output if they must be inspected afterward.

An analysis output cannot be restarted
   This is intentional. Restart from the original simulation output rather than from a
   file produced by analysis mode.
