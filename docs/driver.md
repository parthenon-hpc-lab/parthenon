# Application Drivers

Parthenon provides some basic functionality for coordinating various types of calculation in the form of the ```Driver``` class and others that derive from it.

## Driver

```Driver``` is an abstract base class that owns pointers to a ```ParameterInput``` object, a ```Mesh``` object, and an ```Outputs``` object.  It has a single pure virtual member function called ```Execute``` that must be defined by a derived class and is intended to be called from ```main()```.  A simple example of defining an application based on inheriting from this class can be found [here](../example/calculate_pi/pi.hpp).

## EvolutionDriver

The ```EvolutionDriver``` class derives from ```Driver```, defining the ```Execute``` function to carry out the
```c++
while (t < tmax) {
    // step the solution through time
}
```
loop, including periodic outputs.  It has a single pure virtual member function called ```Step``` which a derived class must define and which will be called during each pass of the loop above.

## MultiStageDriver

The ```MultiStageDriver``` derives from the ```EvolutionDriver```, extending it with two new data members.  These include a vector of ```std::string``` names for the stages of a multi-stage integration scheme and a pointer to an ```Integrator``` object which includes members for the number of stages and the stage weights.

## MultiStageBlockTaskDriver

The ```MultiStageBlockTaskDriver``` derives from the ```MultiStageDriver```, defining the ```Step``` function to loop over the stages in a step, constructing and executing task lists per ```MeshBlock```.  This class includes a single pure virtual member function called ```MakeTaskList``` which must be defined by an application and is responsible for constructing a ```TaskList``` for a given ```MeshBlock``` and ```Stage```.  The driver for the advection example (found [here](../example/advection/advection.hpp)) derives from this class, demonstrating how a simple application based on a multi-stage Runge-Kutta scheme can be built. 