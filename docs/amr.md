# Adaptive Mesh Refinement

There are two ways to control AMR in parthenon.  First, built-in refinemnt criteria can be activated at runtime via the input file.  Second, package developers can create a package-specific refinement tagging function to allow for more tailored criteria of less generic applicability.

## Enable AMR
To enable AMR, the following lines are required in the `<parthenon/mesh>` block of your input file.
```c++
refinement = adaptive    # enable adaptive mesh refinement
numlevel = 5             # how many refined levels can parthenon produce
```

## Built-in 
Parthenon includes the ability to tag cells for refinement/derefinement based on predefined criteria that can be enabled at runtime in the input file.  Multiple criteria can be enabled simultaneously, in which case the most refined criteria wins.  If ``refinement=adaptive`` has been specified as above, parthenon will initialize your AMR choices by looking for blocks with names ``<parthenon/refinement#>`` where ``#`` is a zero-based sequential indexing of Refinement criteria.  An input file might looks like
```c++
<parthenon/mesh>
refinement = adaptive
...

<parthenon/refinement0>
...

<parthenon/efinement1>
...

<parthenon/refinement2>
...
```
In each refinement block, you are required to provide a ``method`` which is a string that selects among the provided critera (listed below).  Additionally, you are required to provide a ``field`` which must be a valid variable name in the application.  Optionally, you can provide a ``refine_tol`` value (defaults to 0.5) indicating that a block should be tagged for refinement if the criteria selected evaluates to a value above this threshold anywhere on the block.  Similarly, the ``derefine_tol`` value (default 0.05) determines when derefinement can occur (all values of the criteria function must be less than this).  Finally, an integer ``max_level`` value can be specified that limits refinement triggered by this criteria to no greater than this level.  The default is to allow each criteria to refine to ``numlevel``, which is the global maximum number of refinement levels specified in the ``<parthenon/mesh>`` block.

### Predefined Criteria
| Method | Description |
|--------|-------------|
| derivative_order_1 | ![formula](https://render.githubusercontent.com/render/math?math=\|dlnq\/dlnx\|) |
| derivative_order_2 | $$\frac{\delta x^2}{4\|q\|} \left\| \frac{\partial^2 q}{\partial x^2} \right\| = \frac{ \| q_{i-1} - 2 q_{i} + q_{i+1} \| }{ 2\| q_{i} \| + \| q_{i-1} + q_{i+1} \| } $$ Note that this quantity is bounded by by $\[0,1\]$. |

where q is the user selected variable.

## Package-specific Criteria
As a package developer, you can define a tagging function that takes a ``Container`` as an argument and returns an integer in {-1,0,1} to indicate the block should be derefined, left alone, or refined, respectively.  This function should be registered in a ``StateDescriptor`` object by assigning the ``CheckRefinement`` function pointer to point at the packages function.  An example is demonstrated [here](../example/calculate_pi/pi.cpp).

## Ensuring your data is consistent after re-meshing

When re-meshing happens, a few operations happen, which can be plugged in to in various ways. The operations performed (in order) are:
- The function `InitMeshBlockUserData` is called. This function can be set by setting it in the `ApplicationInputs` field of the problem generator:
```C++
void MyInitMeshBlockUserData(MeshBlock *pmb, ParameterInput *pin) {
  // Do something on a meshblock
}
pman.app_input.InitMeshBlockUserData = MyInitMeshBlockUserData;
// continue with initialization...
```
- When the mesh is being generated at initialization, the problem generator is called after every re-meshing.
- Prolongation, restriction, physical boundaries, and ghost zone communication are performed
- The `FillDerived` functions set per-package and per-application are called.

If you have a function that you would like called every cycle, you may wish to put it in `FillDerived`.
If you have a function you would like performed only at re-meshing, you may wish to put it in `InitMeshBlockUserData`.
