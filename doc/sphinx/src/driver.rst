.. _driver:

Application Drivers
===================

Parthenon provides some basic functionality for coordinating various
types of calculation in the form of the ``Driver`` class and others that
derive from it.

Driver
------

``Driver`` is an abstract base class that owns pointers to a
``ParameterInput`` object, a ``Mesh`` object, and an ``Outputs`` object.
It has a single pure virtual member function called ``Execute`` that
must be defined by a derived class and is intended to be called from
``main()``. A simple example of defining an application based on
inheriting from this class can be found
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/calculate_pi/pi_driver.hpp>`__.

EvolutionDriver
---------------

The ``EvolutionDriver`` class derives from ``Driver``, defining the
``Execute`` function to carry out the

.. code:: cpp

   while (t < tmax) {
       // step the solution through time
   }

loop, including periodic outputs. It has a single pure virtual member
function called ``Step`` which a derived class must define and which
will be called during each pass of the loop above. The
``SetGlobalTimeStep`` and ``OutputCycleDiagnostics`` functions have
default implementations, but can be overridden for flexibility.

MultiStageDriver
----------------

The ``MultiStageDriver`` derives from the ``EvolutionDriver``, extending
it in two important ways. First, it is templated on an ``Integrator`` type and
holds a ``std::unique_ptr<Integrator>`` object which includes members for
the number of stages, the stage weights, and the names of the stages.
More details on integrators can be found on the :ref:`integrators` page.
Second, it defines a ``Step()`` function, which is reponsible for taking
a timestep by looping over stages and calling the
``ConstructAndExecuteTaskLists`` function which builds and executes the
tasks for each stage. The task list called here has a timeout set by the
parameter ``parthenon/mesh/task_collection_timeout_in_seconds``, which
defaults to five minutes. Applications that derive from ``MultiStageDriver``
are responsible for defining a ``MakeTaskCollection`` function that
makes a ``TaskCollection`` given a ``BlockList_t &`` and an integer
stage. The advection example
(`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/advection/advection_driver.hpp>`__) demonstrates the
use of this capability.

MultiStageBlockTaskDriver
-------------------------

The ``MultiStageBlockTaskDriver`` derives from the ``MultiStageDriver``,
providing a slightly different interface to downstream codes.
Application drivers that derive from this class must define a
``MakeTaskList`` function that builds a task list for a given
``MeshBlock *`` and integer stage. This is less flexible than the
``MultiStageDriver``, but does simplify the code necessary to stand up a
new capability.

PreExecute/PostExecute
----------------------

The ``Driver`` class defines two virtual void member functions,
``PreExecute`` and ``PostExecute``, that provide simple reporting
including timing data for the application. Application drivers that
derive from ``Driver`` are intended to call these functions at the
beginning and end of their ``Execute`` function if such reporting is
desired. In addition, applications can extend these capabilities as
desired, as demonstrated in the example
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/calculate_pi/pi_driver.cpp>`__.

The ``EvolutionDriver``, which already defines the ``Execute`` function,
extends the ``PostExecute`` function with additional reporting and calls
both methods.

Analysis execution
------------------

The standard ``EvolutionDriver`` recognizes the analysis signal installed by
``ParthenonManager``. In analysis mode it skips block timestep initialization, global
timestep selection, and every call to ``Step``. The driver instead performs the
following sequence once:

#. ``PreExecute``.
#. Application and package ``UserWorkBeforeLoop`` callbacks.
#. The output pass, including ``UserWorkBeforeOutput`` for selected non-history outputs.
#. ``UserWorkAfterLoop``.
#. ``PostExecute``.

The ordinary final-output pass is skipped because the analysis output pass has already
been performed. The source cycle and time therefore remain unchanged unless the
analysis input or command line overrides them. See :ref:`analysis` for output selection
and initialization details.

Initialization callback summary
--------------------------------

.. list-table::
   :header-rows: 1

   * - Operation
     - Fresh
     - Restart
     - Analysis
   * - Problem generator
     - yes
     - no
     - no
   * - Post-problem-generator callbacks
     - yes
     - no
     - no
   * - Pre-communication derived fills
     - yes
     - yes
     - no
   * - Boundary communication
     - yes
     - yes
     - yes
   * - Post-communication derived fills
     - yes
     - yes
     - no
   * - Post-initialization callbacks
     - yes
     - no
     - no
   * - ``UserWorkBeforeLoop``
     - yes
     - yes
     - yes
   * - Evolution ``Step``
     - yes
     - yes
     - no
   * - ``UserWorkBeforeOutput``
     - when selected
     - when selected
     - when selected
