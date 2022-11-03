## Guide to Parthenon-VIBE: A benchmark that solves the Vector Inviscid Burgers' Equation on a block-AMR mesh

### Basic Description

This benchmark solves the inviscid Burgers' equation

$$
\begin{equation*}
\partial_t \mathbf{u} + \nabla\cdot\left(\frac{1}{2}\mathbf{u} \mathbf{u}\right) = 0
\end{equation*}
$$

and evolves one or more passive scalar quantities $q^i$ according to

$$
\begin{equation*}
\partial_t q^i + \nabla \cdot \left( q^i \mathbf{u} \right) = 0
\end{equation*}
$$

as well as computing an auxiliary quantity $d$ that resemebles a kinetic energy

$$
\begin{equation*}
d = \frac{1}{2} q^0 \mathbf{u}\cdot\mathbf{u}\;.
\end{equation*}
$$

Parthenon-VIBE makes use of a Godunov-type finite volume scheme with options for slope-limited linear ore WENO5 reconstruction, HLL fluxes, and second order Runge-Kutta time integration.

### Parameters

The benchmark includes an input file _burgers.pin_ that specifies the base (coarsest level) mesh size, the size of a mesh block, the number of levels, and a variety of other parameters that control the behavior of Parthenon and the benchmark problem configuration.  The table below is an incomplete list of the available parameters, default values, and possible options.

| Block             | Paramter    | Default    | Options                        | Description | 
| -----                 | --------    | :------:   | :-----:                        | ----------- |
| <parthenon/mesh>      | refinement  | *adaptive* | {*adaptive*, *static*, *none*} | Is the mesh adaptively refined/derefined, use statically refined, or uniform? *static* requires specifying refined regions separately, and is not recoommended for this benchmark.|
|                       | numlevel    | 3          | 1+                             | The total number of levels, including the base level |
|                       | deref_count | 10         | 1+                             | The number of time steps between possible derefinement operations. |
|                       | nghost      | 4          | {2,4}                          | The number of ghost cells on each side of a mesh block.  WENO5 reconstruction requires this to be set to 4, while linear can use 2. |
|                       | nx1         | 64         | >= nghost                      | The number of cells in the x-direction on the coarsest level |
|                       | nx2         | 64         | >= nghost                      | The number of cells in the y-direction on the coarsest level |
|                       | nx3         | 64         | >= nghost                      | The number of cells in the z-direction on the coarsest level |
| <parthenon/meshblock> | nx1         | 16         | must evenly divide mesh/nx1    | The number of cells in the x-direction in each mesh block |
|                       | nx2         | 16         | must evenly divide mesh/nx2    | The number of cells in the y-direction in each mesh block |
|                       | nx3         | 16         | must evenly divide mesh/nx3    | The number of cells in the z-direction in each mesh block |
| <parthenon/time>      | nlim        | -1         | any integer                    | The maximum number of time steps.  Negative values imply no bound. |
|                       | tlim        | 0.4        | any float                      | The total amount of simulation time to evolve the solution |
|                       | perf_cycle_offset | 0    | >= 0                           | Number of time steps at start up to skip before performance measurement begins |
| <parthenon/output0>   | dt          | -0.4        | any float                      | Simulated time between HDF5 dumps.  Setting this to a negative value disables HDF5 dumps, which is required if the benchmark was built without HDF5 support. |
| \<burgers>             | num_scalars | 1          | > 0                           | The number of scalar conservation laws to evolve, in addition to Burgers' equation. |
|                        | recon       | weno5      | {weno5, linear}               | Reconstruction method to define states of faces for Riemann solves.  weno5 uses a higher order function (5pt stencil, requires nghost = 4), while linear does a simple linear function (3pt stencil, requires only nghost = 2). |



