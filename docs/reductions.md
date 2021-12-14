# Task-based reductions

Many codes require the ability to do global reductions.  In a task-based environment where each rank may be executing multiple tasks lists operating on independent sub-domains, orchestrating these reductions turns out to be nontrivial.  Here, we document a Parthenon-ic way of expressing a global reduction.  The basic strategy follows:

1. Initialize a variable that will capture the *local* (just a given MPI rank's value) reduced value.
2. Launch a task in each list which updates the *local* value.  For example, if the reduction is a sum, each task will add its contribution to this shared variable.  Since task launching is not threaded, there is no concern over race conditions.
3. Mark the task which accumulates the *local* reduction using the `TaskRegion` member function `AddRegionalDependencies`.  This will ensure that tasks that require a complete local reduction will not launch until that *local* value is available.
4. One task list on each rank launches a non-blocking reduction operation.
5. All task lists launch a task which checks the status of the reduction, returning `TaskStatus::complete` once the value of the global reduction is set.

To facilitate this pattern, parthenon provides an `AllReduce` struct, described below.  Examples of the pattern above and the usage of `AllReduce` are provided [here](../example/poisson/poisson_driver.cpp).

## AllReduce

`AllReduce` is a struct templated on the type of value that needs to be reduced (e.g. `int`, `Real`, `std::vector<Real>`, etc.).  It manages the storage in a member variable `val` which is of the type provided as a template argument.  `val` must be appropriately initialized by the user.  The functionality in `AllReduce` (described above) is exposed through two member functions, `StartReduce ` and `CheckReduce`.  `StartReduce` requires a single argument which is the MPI reduction operator (e.g. `MPI_SUM`, `MPI_MAX`, etc.).  Both of these tasks are non-blocking (i.e. they call `MPI_Iallreduce` and `MPI_Test`).

## Reduce

Same as `AllReduce` except `MPI_Ireduce` is called and the root rank of the reduction must be provided in `StartReduce`
