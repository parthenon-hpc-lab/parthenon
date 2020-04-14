# Graphics

Parthenon allows users to select which fields are captured in the HDF5 (```.phdf```) graphics dumps at runtime.  In the input file, include a ```<Graphics>``` block and list of variables, such as
```
<Graphics>
variables = density, velocity, & # comments are still ok
            energy               # notice the & continuation character
                                 # for multiline lists
```

## Python scripts

The ```vis/python``` folder includes scripts that may be useful for visualizing or analyzing data in the ```.phdf``` files.  The ```phdf.py``` file defines a class to read in and query data.  The ```movie2d.py``` script shows an example of using this class, and also provides a convenient means of making movies of 2D simulations.  The script can be invoked as
```
python3 /path/to/movie2d.py name_of_variable *.phdf
```
which will produce a ```png``` image per dump suitable for encoding into a movie.

## Visualization software

Both [ParaView](https://www.paraview.org/) and [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/) are capable of opening and visualizing Parthenon graphics dumps.  In both cases, the ```.xdmf``` files should be opened.  In ParaView, select the "XDMF Reader" when prompted.