.. _tasks:

Tasks
=====

TaskList
--------

The ``TaskList`` class implements methods to build and execute a set of
tasks with associated dependencies. The class implements a few public
facing member functions that provide useful functionality for downstream
apps:

AddTask
~~~~~~~

``AddTask`` is a templated variadic function that takes the task
function to be executed, the task dependencies (see ``TaskID`` below),
and the arguments to the task function as it’s arguments. All arguments
are captured by value in a lambda for later execution.

When adding functions that are non-static class member functions, a
slightly different interface is required. The first argument should be
the class-name-scoped name of the function. For example, for a function
named ``DoSomething`` in class ``SomeClass``, the first argument would
be ``&SomeClass::DoSomething``. The second argument should be a pointer
to the object that should invoke this member function. Finally, the
dependencies and function arguments should be provided as described
above.

Examples of both ``AddTask`` calls can be found in the advection example
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/advection/advection_driver.cpp>`__.

AddIteration
~~~~~~~~~~~~

``AddIteration`` provides a means of grouping a set of tasks together
that will be executed repeatedly until stopping criteria are satisfied.
``AddIteration`` returns an ``IterativeTasks`` object which provides
overloaded ``AddTask`` functions as described above, but internally
handles the bookkeeping necessary to maintain the association of all the
tasks associated with the iterative process. A special function
``SetCompletionTask``, which behaves identically to ``AddTask``, allows
a task to be defined that evaluates the stopping criteria. The maximum
number of iterations can be controlled through the ``SetMaxIterations``
member function and the number of iterations between evaluating the
stopping criteria can be set with the ``SetCheckInterval`` function.

DoAvailable
~~~~~~~~~~~

``DoAvailable`` loops over the task list once, executing all tasks whose
dependencies are satisfied. Completed tasks are removed from the task
list.

TaskID
------

The ``TaskID`` class implements methods that allow Parthenon to keep
track of tasks, their dependencies, and what remains to be completed.
The main way application code will interact with this object is as a
returned object from ``TaskList::AddTask`` and as an argument to
subsequent calls to ``TaskList::AddTask`` as a dependency for other
tasks. When used as a dependency, ``TaskID`` objects can be combined
with the bitwise or operator (``|``) to specify multiple dependencies.

TaskRegion
----------

``TaskRegion`` is a lightweight class that wraps
``std::vector<TaskList>``, providing a little extra functionality.
During task execution (described below), all task lists in a
``TaskRegion`` can be operated on concurrently. For example, a
``TaskRegion`` can be used to construct independent task lists for each
``MeshBlock``. Occasionally, it is useful to have a task not be
considered complete until that task completes in all lists of a region.
For example, a global iterative solver cannot be considered complete
until the stopping criteria are satisfied everywhere, which may require
evaluating those criteria in tasks that live in different lists within a
region. An example of this use case is
shown `here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/poisson/poisson_driver.cpp>`__. The mechanism
to mark a task so that dependent tasks will wait until all lists have
completed it is to call ``AddRegionalDependencies``, as shown in the
Poisson example.

TaskCollection
--------------

A ``TaskCollection`` contains a
``std::vector<TaskRegion>``, i.e. an ordered list of ``TaskRegion``\ s.
Importantly, each ``TaskRegion`` will be executed to completion before
subsequent ``TaskRegion``\ s, introducing a notion of sequential
execution and enabling flexibility in task granularity. For example, the
following code fragment uses the ``TaskCollection`` and ``TaskRegion``
abstractions to express work that can be done asynchronously across
blocks, followed by a bulk synchronous task involving all blocks, and
finally another round of asynchronous work.

.. code:: cpp

  TaskCollection tc;
  TaskRegion &tr1 = tc.AddRegion(nmb);
  for (int i = 0; i < nmb; i++) {
    auto task_id = tr1[i].AddTask(dep, foo, args, blocks[i]);
  }

  {
    TaskRegion &tr2 = tc.AddRegion(1);
    auto sync_task = tr2[0].AddTask(dep, bar, args, blocks);
  }

  TaskRegion &tr3 = tc.AddRegion(nmb);
  for (int i = 0; i < nmb; i++) {
    auto task_id = tr3[i].AddTask(dep, foo, args, blocks[i]);
  }

A diagram illustrating the relationship between these different classes
is shown below.

.. figure:: figs/TaskDiagram.png
   :alt: Task Diagram

``TaskCollection`` provides two member functions, ``AddRegion`` and
``Execute``.

AddRegion
~~~~~~~~~

``AddRegion`` simply adds a new ``TaskRegion`` to the back of the
collection and returns it as a reference. The integer argument
determines how many task lists make up the region.

Execute
~~~~~~~

Calling the ``Execute`` method on the ``TaskCollection`` executes all
the tasks that have been added to the collection, processing each
``TaskRegion`` in the order they were added, and allowing tasks in
different ``TaskList``\ s but the same ``TaskRegion`` to be executed
concurrently.
