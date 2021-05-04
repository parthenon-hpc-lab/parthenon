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

## Preparing outputs for ```yt```

Parthenon HDF5 outputs can be read with the python visualization library
[yt](https://yt-project.org/) as certain variables are named when adding
fields via ```StateDescriptor::AddField```. Variable names are added as a
```std::vector<std::string>``` in the variable metadata. These labels are
optional and are only used for output to HDF5. 4D variables are named with a
list of names for each row while 3D variables are named with a single name.
For example, the following configurations are acceptable:

```c++
auto pkg = std::make_shared<StateDescriptor>("Hydro");

/* ... */
const int nhydro = 5;
std::vector<std::string> cons_labels(nhydro);
cons_labels[0]="Density";
cons_labels[1]="MomentumDensity1";
cons_labels[2]="MomentumDensity2";
cons_labels[3]="MomentumDensity3";
cons_labels[4]="TotalEnergyDensity";
Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
           std::vector<int>({nhydro}), cons_labels);
pkg->AddField("cons", m);

const int ndensity = 1;
std::vector<std::string> density_labels(ndensity);
density_labels[0]="Density";
m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({ndensity}), density_labels);
pkg->AddField("dens", m);

const int nvelocity = 3;
std::vector<std::string> velocity_labels(nvelocity);
velocity_labels[0]="Velocity1";
velocity_labels[1]="Velocity2";
velocity_labels[2]="Velocity3";
m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nvelocity}), velocity_labels);
pkg->AddField("vel", m);

const int npressure = 1;
std::vector<std::string> pressure_labels(npressure);
pressure_labels[0]="Pressure";
m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({npressure}), pressure_labels);
pkg->AddField("pres", m);
```

The `yt` frontend needs either the hydrodynamic conserved variables or
primitive compute derived quantities. The conserved variables must have the
names ```"Density"```, ```"MomentumDensity1"```, ```"MomentumDensity2"```,
```"MomentumDensity3"```, ```"TotalEnergyDensity"``` while the primitive
variables must have the names ```"Density"```, ```"Velocity1"```,
```"Velocity2"```, ```"Velocity3"```, ```"Pressure"```. Either of these sets of
variables must be named and present in the output, with the primitive variables
taking precedence over the conserved variables when computing derived
quantities such as specific thermal energy. In the above example, including
either ```"cons"``` or ```"dens"```, ```"vel"```, and ```"pres"``` in the  HDF5
output would allow ```yt``` to read the data.

Additional parameters can also be packaged into the HDF5 file to help ```yt```
interpret the data, namely adiabatic index and code unit information. These are
identified by passing ```true``` as an optional boolean argument when adding
parameters via ```StateDescriptor::AddParam```. For example, 
```c++
pkg->AddParam<double>("CodeLength", 100,true);
pkg->AddParam<double>("CodeMass", 1000,true);
pkg->AddParam<double>("CodeTime", 1,true);
pkg->AddParam<double>("AdibaticIndex", 5./3.,true);

pkg->AddParam<int>("IntParam", 0,true);
pkg->AddParam<std::string>("EquationOfState", "Adiabatic",true);
```
adds the parameters ```CodeLength```, ```CodeMass```, ```CodeTime```,
```AdiabaticIndex```, ```IntParam```, and ```EquationOfState``` to the HDF5
output. Currently, only ```int```, ```float```, and ```std::string```
parameters can be included with the HDF5.

Code units can be defined for ```yt``` by including the parameters
```CodeLength```, ```CodeMass```, and ```CodeTime```, which specify the code
units used by Parthenon in terms of centimeters, grams, and seconds by writing
the parameters.  In the above example, these parameters dictate ```yt``` to
interpret code lengths in the data in units of 100 centimeters (or 1 meter per
code unit), code masses in units of 1000 grams (or 1 kilogram per code units)
and code times in units of seconds (or 1 second per code time).
Alternatively, this unit information can also be supplied to the ```yt```
frontend when loading the data. If code units are not defined in the HDF5 file
or at load time, ```yt``` will assume that the data is in ```CGS```.

The adiabatic index can also be specified via the parameter
```AdiabaticIndex```, defined at load time for ```yt```, or left as its default
```5./3.```.

For example, the following methods are valid to load data with ```yt```
```python
filename = "parthenon.out0.00000.phdf"

#Read units and adiabatic index from the HDF5 file or use defaults
ds = yt.load(filename)

#Specify units and adiabatic index explicitly
units_override = {"length_unit" : (100, "cm"),
                  "time_unit"   : (1,   "s"),
                  "mass_unit"   : (1000,"g")}

ds = yt.load(filename,units_override=units_override,gamma=5./3.)
```

Currently, the ```yt``` frontend for Parthenon is hosted on the
```athenapk-frontend``` [on this ```yt```
fork](https://github.com/forrestglines/yt/tree/athenapk-frontend). In the
future, the Parthenon frontend will be included in the main ```yt``` repo.
