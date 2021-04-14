# Outputs

Outputs from Parthenon are controled via ```<parthenon/output*>``` blocks, where ```*``` should be replaced by a unique integer for each block.

To disable an output block without removing it from the intput file set the block's `dt < 0.0`.

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

## Restart Files

Parthenon allows users to output restart files for restarting a simulation.  The restart file captures the input file, so no input file is required to be specified.  Parameters for the input can be overriden in the usual way from teh command line.  At a future date we will allow for users the ability to extensively edit the parameters stored within the restart file. 

In the input file, include a ```<parthenon/output*>``` block and specify ```file_type = rst```.  A ```dt``` parameter controls the frequency of outputs for simulations involving evolution. A ```<parthenon/output*>``` block might look like
```
<parthenon/output7>
file_type = rst
dt = 1.0
```
This will produce an hdf5 (`.rhdf`) output file every 1 units of
simulation time that can be used for restarting the simulation.

To use this restart file, simply specify the restart file with a ```-r <restart.rhdf>``` at the command line.  It is an error to specify an input file with the ```-i``` flag when using the restart option.

For physics developers: The fields to be output are automatically selected as all the variables that have either the ```Independent``` or ```Restart``` ```Metadata``` flags specifiec.  No other intervention is required by the developer.

## History Files

In the input file, include a ```<parthenon/output*>``` block and specify ```file_type = hst```.  A ```dt``` parameter controls the frequency of outputs for simulations involving evolution. A ```<parthenon/output*>``` block might look like
```
<parthenon/output8>
file_type = hst
dt = 1.0
```
This will produce a text file (`.hst`) output file every 1 units of simulation time.
The content of the file is determined by the functions enrolled by a specific package,
see the [interface doc](interface/state.md#history-output).

## Python scripts

The ```scripts/python``` folder includes scripts that may be useful for visualizing or analyzing data in the ```.phdf``` files.  The ```phdf.py``` file defines a class to read in and query data.  The ```movie2d.py``` script shows an example of using this class, and also provides a convenient means of making movies of 2D simulations.  The script can be invoked as
```
python3 /path/to/movie2d.py name_of_variable *.phdf
```
which will produce a ```png``` image per dump suitable for encoding into a movie.

## Visualization software

Both [ParaView](https://www.paraview.org/) and [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/) are capable of opening and visualizing Parthenon graphics dumps.  In both cases, the ```.xdmf``` files should be opened.  In ParaView, select the "XDMF Reader" when prompted.
