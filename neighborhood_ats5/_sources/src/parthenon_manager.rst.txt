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
pointer ``pman.pin``, and sets up the ``MPI``, and ``Kokkos``
runtimes. The function

.. code:: cpp

   void ParthenonManager::ParthenonInitPackagesAndMesh();

Calls the ``Initialize(ParameterInput *pin)`` function of all packages
to be utilized and creates the grid hierarchy, including the ``Mesh``
and ``MeshBlock`` objects, and calls the ``ProblemGenerator``
initialization routines.

The reason these functions are split out is to enable decisions to be
made by the application between reading the input deck and setting up
the grid. For example, a common use-case is:

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
  std::string prob = pman.pin->GetString("app", "problem");
  if (prob == "problem1") {
    pman.app_input->ProblemGenerator = Problem1Generator;
  } else {
    pman.app_input->ProblemGenerator = Problem2Generator;
  }

  pman.ParthenonInitPackagesAndMesh();
