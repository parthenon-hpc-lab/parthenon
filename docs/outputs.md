# Outputs

Outputs from Parthenon are controled via `<parthenon/output*>` blocks, where `*` should be replaced by a unique integer for each block.

To disable an output block without removing it from the intput file set the block's `dt < 0.0`.

In addition to time base outputs, two additional options to trigger outputs
(applies to HDF5 and restart outputs) exist.

- Signaling: If `Parthenon` catches a signal, e.g., `SIGALRM` which is often sent by
schedulers such as Slurm to signal a job of exceeding the job's allocated walltime,
`Parthenon` will gracefully terminate and write output files with a `final` id rather
than a number.
This also applies to the `Parthenon` internal walltime limit, e.g., when executing an
application with the `-t HH:MM:SS` parameter on the command line.
- File trigger: If a user places a file with the name `output_now` in the working
directory of a running application, `Parthenon` will write output files with a `now` id
rather than a number.
After the output is being written the `output_now` file is removed and the simulation
continues normally.
The user can repeat the process any time by creating a new `output_now` file.

Note, in both cases the original numbering of the output will be unaffected and the
`final` and `now` files will be overwritten each time without warning.
## HDF5

Parthenon allows users to select which fields are captured in the HDF5 (`.phdf`) dumps at
runtime.  In the input file, include a `<parthenon/output*>` block, list of variables, and
specify `file_type = hdf5`.  A `dt` parameter controls the frequency of outputs for
simulations involving evolution. If the optional parameter `single_precision_output` is set to
`true`, all variable data will be written in single precision.
A `<parthenon/output*>` block might look like
```
<parthenon/output1>
file_type = hdf5
variables = density, velocity, & # comments are still ok
            energy               # notice the & continuation character
                                 # for multiline lists
dt = 1.0
file_number_width = 6 # default: 5
use_final_label = true # default: true
```
This will produce an hdf5 (`.phdf`) output file every 1 units of simulation time containing the
density, velocity, and energy of each cell. The files will be identified by a 6-digit ID, and the
output file generated upon completion of the simulation will be labeled `*.final.*` rather than
with the integer ID.

HDF5 and restart files write variable field data with inline compression by default. This is
especially helpful when there are sparse variables allocated only in a few blocks, because all other
blocks would write zeros of these variables, which can drastically increase output file size (and
decrease I/O performance) without compression. The optional parameter `hdf5_compression_level` can
be used to set the compression level (between 1 and 9, default is 5). Compression can be disabled
altogether with the CMake build option `PARTHENON_DISABLE_HDF5_COMPRESSION`. See the [build
doc](building.md) for more details.

## Gridh5 Files

Gridh5 files are a special, lightweight subclass of HDF5 files, where
only the grid structure is output. No variables are output, nor are
coordinate values. Instead, the bounding box of each meshblock is
output, along with all the metadata normally in a `phdf` file. A
relevant block might look like:
```
<parthenon/output2>
file_type = gridh5
dt = 1.0
```
All options (except for the list of output variables) supported by
`hdf5` are supported. The `plot_mesh.py` script in the
`parthenon_tools` library found in the `scripts/python` folder can
generate a constant `X3` slice of the mesh based on an `hdf5` or
`gridh5` output file.

## Tuning HDF5 Performance

Tuning IO parameters can be passed to Parthenon through the use of environment variables. Available environment variables are:

|  Environment Variable | Initial State | Value Type | Description |
|---|---|---|---|
| H5_sieve_buf_size | disabled | int | Sets the maximum size of the data sieve buffer, in bytes. The value should be equal to a multiple of the disk block size. If no value is set then the default is 256 KiB. |
| H5_meta_block_size | disabled | int | Sets the minimum metadata block size, in bytes. If no value is set then the default is 8 MiB. May help performance if enabled. |
| H5_alignment_threshold | disabled | int | The threshold value, in bytes, of H5Pset_alignment. Setting to 0 forces everything to be aligned. If a value is not set then the default is 0. Setting the environment variable automatically enables alignment. |
| H5_alignment_alignment | disabled | int | The alignment value, in bytes, of H5Pset_alignment. If a value is not set then the default is 8 MiB.  Setting the environment variable automatically enables alignment.  H5Pset_alignment sets the alignment properties of a file access property list. Choose an alignment that is a multiple of the disk block size, enabling this usually shows better performance on parallel file systems. However, enabling may increase the file size significantly. |
| H5_defer_metadata_flush | disabled | int | Value of 1 enables deferring metadata flush. Value of 0 disables. Experiment with before using. |
| MPI_access_style | enabled | string | Specifies the manner in which the file will be accessed until the file is closed. Default is "write_once" |
| MPI_collective_buffering | disabled | int | Value of 1 enables MPI collective buffering. Value of 0 disables. Experiment with before using. |
| MPI_cb_block_size | N/A | int | Sets the block size, in bytes, to be used for collective buffering file access. Default is 1 MiB. |
| MPI_cb_buffer_size | N/A | int | Sets the total buffer space, in bytes, that can be used for collective buffering on each target node,  usually a multiple of cb_block_size. Default is 4 MiB. |

## Restart Files

Parthenon allows users to output restart files for restarting a simulation.  The restart file captures the input file, so no input file is required to be specified.  Parameters for the input can be overriden in the usual way from the command line.  At a future date we will allow for users the ability to extensively edit the parameters stored within the restart file.

In the input file, include a `<parthenon/output*>` block and specify `file_type = rst`.  A `dt` parameter controls the frequency of outputs for simulations involving evolution. A `<parthenon/output*>` block might look like
```
<parthenon/output7>
file_type = rst
dt = 1.0
```
This will produce an hdf5 (`.rhdf`) output file every 1 units of
simulation time that can be used for restarting the simulation.

To use this restart file, simply specify the restart file with a `-r <restart.rhdf>` at the command line.  It is an error to specify an input file with the `-i` flag when using the restart option.

For physics developers: The fields to be output are automatically selected as all the variables that have either the `Independent` or `Restart` `Metadata` flags specified.  No other intervention is required by the developer.

## History Files

In the input file, include a `<parthenon/output*>` block and specify `file_type = hst`.  A `dt` parameter controls the frequency of outputs for simulations involving evolution. A `<parthenon/output*>` block might look like
```
<parthenon/output8>
file_type = hst
dt = 1.0
```
This will produce a text file (`.hst`) output file every 1 units of simulation time.
The content of the file is determined by the functions enrolled by a specific package,
see the [interface doc](interface/state.md#history-output).

## Python scripts

The `scripts/python` folder includes scripts that may be useful for visualizing or analyzing data in the `.phdf` files.  The `phdf.py` file defines a class to read in and query data.  The `movie2d.py` script shows an example of using this class, and also provides a convenient means of making movies of 2D simulations.  The script can be invoked as
```
python3 /path/to/movie2d.py name_of_variable *.phdf
```
which will produce a `png` image per dump suitable for encoding into a movie.

## Visualization software

Both [ParaView](https://www.paraview.org/) and [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/) are capable of opening and visualizing Parthenon graphics dumps.  In both cases, the `.xdmf` files should be opened.  In ParaView, select the "XDMF Reader" when prompted.

## Preparing outputs for `yt`

Parthenon HDF5 outputs can be read with the python visualization library
[yt](https://yt-project.org/) as certain variables are named when adding
fields via `StateDescriptor::AddField` and `StateDescriptor::AddSparsePool`.
Variable names are added as a
`std::vector<std::string>` in the variable metadata. These labels are
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
names `"Density"`, `"MomentumDensity1"`, `"MomentumDensity2"`,
`"MomentumDensity3"`, `"TotalEnergyDensity"` while the primitive
variables must have the names `"Density"`, `"Velocity1"`,
`"Velocity2"`, `"Velocity3"`, `"Pressure"`. Either of these sets of
variables must be named and present in the output, with the primitive variables
taking precedence over the conserved variables when computing derived
quantities such as specific thermal energy. In the above example, including
either `"cons"` or `"dens"`, `"vel"`, and `"pres"` in the  HDF5
output would allow `yt` to read the data.

Additional parameters can also be packaged into the HDF5 file to help `yt`
interpret the data, namely adiabatic index and code unit information. These are
identified by passing `true` as an optional boolean argument when adding
parameters via `StateDescriptor::AddParam`. For example,
```c++
pkg->AddParam<double>("CodeLength", 100,true);
pkg->AddParam<double>("CodeMass", 1000,true);
pkg->AddParam<double>("CodeTime", 1,true);
pkg->AddParam<double>("AdibaticIndex", 5./3.,true);

pkg->AddParam<int>("IntParam", 0,true);
pkg->AddParam<std::string>("EquationOfState", "Adiabatic",true);
```
adds the parameters `CodeLength`, `CodeMass`, `CodeTime`,
`AdiabaticIndex`, `IntParam`, and `EquationOfState` to the HDF5
output. Currently, only `int`, `float`, and `std::string`
parameters can be included with the HDF5.

Code units can be defined for `yt` by including the parameters
`CodeLength`, `CodeMass`, and `CodeTime`, which specify the code
units used by Parthenon in terms of centimeters, grams, and seconds by writing
the parameters.  In the above example, these parameters dictate `yt` to
interpret code lengths in the data in units of 100 centimeters (or 1 meter per
code unit), code masses in units of 1000 grams (or 1 kilogram per code units)
and code times in units of seconds (or 1 second per code time).
Alternatively, this unit information can also be supplied to the `yt`
frontend when loading the data. If code units are not defined in the HDF5 file
or at load time, `yt` will assume that the data is in `CGS`.

The adiabatic index can also be specified via the parameter
`AdiabaticIndex`, defined at load time for `yt`, or left as its default
`5./3.`.

For example, the following methods are valid to load data with `yt`
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

Currently, the `yt` frontend for Parthenon is hosted on the
`athenapk-frontend` [on this `yt`
fork](https://github.com/forrestglines/yt/tree/athenapk-frontend). In the
future, the Parthenon frontend will be included in the main `yt` repo.
