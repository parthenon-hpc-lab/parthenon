# Tasks

## TaskList
The `TaskList` class implements methods to build and execute a set of tasks with associated dependencies.  The main functionality of the class is implemented in two member functions:

### AddTask
`AddTask` is a templated variadic function that takes the task function to be executed, the task dependencies (see `TaskID` below), and the arguments to the task function as it's arguments.  All arguments are captured by value in a lambda for later execution.

When adding functions that are non-static class member functions, a slightly different interface is required.  The first argument should be the class-name-scoped name of the function.  For example, for a function named `DoSomething` in class `SomeClass`, the first argument would be `&SomeClass::DoSomething`.  The second argument should be a pointer to the object that should invoke this member function.  Finally, the dependencies and function arguments should be provided as described above.

### DoAvailable
`DoAvailable` loops over the task list once, executing all tasks whose dependencies are satisfied.  The function returns either `TaskListStatus::complete` if all tasks have been executed (and the task list is therefore empty) or `TaskListStatus::running` if tasks remain to be completed.

## TaskID
The `TaskID` class implements methods that allow Parthenon to keep track of tasks, their dependencies, and what remains to be completed.  The main way application code will interact with this object is as a returned object from `TaskList::AddTask` and as an argument to subsequent calls to `TaskList::AddTask` as a dependency for other tasks.  When used as a dependency, `TaskID` objects can be combined with the bitwise or operator (`|`) to specify multiple dependencies.