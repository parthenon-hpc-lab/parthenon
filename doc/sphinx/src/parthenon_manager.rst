.. _parthenonmanager:

Parthenon Manager
=================

The ``ParthenonManager`` class helps set up a parthenon-based
application. An instance of ``ParthenonManager`` owns pointers a
number of sub-objects:

* The ``ApplicationInput`` struct, which lets users set things like
  the ``ProcessPackages`` and ``ProblemGenerator`` function pointers.
* The ``ParameterInput`` class, which populates input parameters from
  the input file and command line
* The ``Mesh`` object

The ``ParthenonManager`` has two important methods that usually must
be called in the ``main`` function of a parthenon-based app. The
function

.. code:: cpp

   ParthenonStatus ParthenonManager::ParthenonInitEnv(int argc, char *argv);

reads the input deck and populates the ``ParameterInput`` object
pointer ``pman.pinput``, and sets up the ``MPI``, and ``Kokkos``
runtimes. The function

.. code:: cpp

   void ParthenonManager::ParthenonInitPackagesAndMesh();

Calls the ``Initialize(ParameterInput *pin)`` function of all packages
to be utilized and creates the grid hierarchy, including the ``Mesh``
and ``MeshBlock`` objects. For fresh initialization it also calls the
``ProblemGenerator``, ``PostProblemGenerator``, and ``PostInitialization``
routines. Restart and analysis initialization load their state from an output instead.

The reason these functions are split out is to enable decisions to be
made by the application between reading the input deck and setting up
the grid. For example, during problem initialization, ``ProblemGenerator``
may be used to be the user-facing API to describe initial conditions,
whereas, ``PostProblemGenerator`` could use those user-specified fields
to sync *all* fields prior to entering communication routines.
``PostInitialization`` could be used to make any last initialization changes
after the initialization hierarchy has converged, regardless of whether that
required AMR, SMR, or no refinement at all. A common use-case is:

.. code:: cpp

  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = MyProcessPackages;
  std::string prob = pman.pinput->GetString("app", "problem");
  if (prob == "problem1") {
    pman.app_input->ProblemGenerator = Problem1Generator;
    pman.app_input->PostProblemGenerator = Problem1PostProblemGenerator;
  } else {
    pman.app_input->ProblemGenerator = Problem2Generator;
  }

  pman.ParthenonInitPackagesAndMesh();

Analysis initialization
-----------------------

When the command line contains ``-a``, ``ParthenonInitEnv`` constructs the reader for
the selected HDF5 or openPMD source and assembles the input parameters before package
initialization. The precedence order is:

#. Input embedded in the source output.
#. The source iteration's actual time, timestep, and cycle.
#. The analysis input supplied with ``-i``.
#. Command-line assignments.

``ParthenonInitPackagesAndMesh`` then resolves an analysis-specific package state,
constructs the mesh hierarchy from the source, loads the selected fields, and performs
the analysis initialization path. Problem generators, derived-fill callbacks, and
post-problem-generator callbacks are not run during this path. Boundary communication is
performed so loaded fields can be consumed by analysis callbacks.

Applications can query the manager after ``ParthenonInitEnv``:

.. code:: cpp

  if (pman.IsAnalysis()) {
    // Configure callbacks or application behavior for analysis mode.
  }

``IsAnalysis()`` is distinct from ``Globals::is_restart``. Analysis mode deliberately
leaves ``Globals::is_restart`` false so existing application logic does not treat a
filtered analysis state as a complete restart state. See :ref:`analysis` for the full
workflow.

.. note::
    This file was made in part with generative AI.
