# Outputs

Outputs from Parthenon are controled via ```<parthenon/output*>``` blocks, where ```*``` should be replaced by a unique integer for each block.

## HDF5

Parthenon allows users to select which fields are captured in the HDF5 (```.phdf```) dumps at runtime.  In the input file, include a ```<parthenon/output*>``` block, list of variables, and specify ```file_type = hdf5```.  A ```dt``` parameter controls the frequency of outputs for simulations involving evolution. A ```<parthenon/output*>``` block might look like
```
<parthenon/output1>
file_type = hdf5
variables = density, velocity, & # comments are still ok
            energy               # notice the & continuation character
                                 # for multiline lists
dt = 1.0
```
This will produce an hdf5 (`.phdf`) output file every 1 units of
simulation time containing the density, velocity, and energy of each
cell.

## Python scripts

The ```scripts/python``` folder includes scripts that may be useful for visualizing or analyzing data in the ```.phdf``` files.  The ```phdf.py``` file defines a class to read in and query data.  The ```movie2d.py``` script shows an example of using this class, and also provides a convenient means of making movies of 2D simulations.  The script can be invoked as
```
python3 /path/to/movie2d.py name_of_variable *.phdf
```
which will produce a ```png``` image per dump suitable for encoding into a movie.

## Visualization software

Both [ParaView](https://www.paraview.org/) and [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/) are capable of opening and visualizing Parthenon graphics dumps.  In both cases, the ```.xdmf``` files should be opened.  In ParaView, select the "XDMF Reader" when prompted.
