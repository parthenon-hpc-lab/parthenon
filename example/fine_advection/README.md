This example implements upwind advection of a cell-centered scalar variable defined
on the regular grid and for another cell-centered variable on the fine grid (which
is twice the resolution and is selected using Metadata::Fine). The newer type-based
`SparsePack`s are used throughout and machinery for doing a generalized Stoke's
theorem based update is included.

## Running the example

The example can be run with either a traditional text input file or a Python input file:

```bash
# Using text input file
./fine_advection-example -i parthinput.advection

# Using Python input file (requires -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON)
# Must set PYTHONPATH to find parthenon.so and parthenon_tools
# From the repo root:
PYTHONPATH=build/src/pybind:scripts/python/packages/parthenon_tools \
  build/example/fine_advection/fine_advection-example -i example/fine_advection/parthinput.advection.py

# Or export PYTHONPATH once:
export PYTHONPATH=/path/to/parthenon/build/src/pybind:/path/to/parthenon/scripts/python/packages/parthenon_tools
./fine_advection-example -i parthinput.advection.py
```

### PYTHONPATH requirements

Python input files require two modules:
- `parthenon.so` - Python bindings (built in `build/src/pybind/`)
- `parthenon_tools` - Input file helpers (in `scripts/python/packages/parthenon_tools/`)

These paths must be in `PYTHONPATH` when running. After `make install`, they will be in
`CMAKE_INSTALL_PREFIX/lib/python/` and no longer need manual `PYTHONPATH` configuration.

### MPI considerations

Python input files run **independently on every MPI rank**. Each rank:
- Starts its own Python interpreter
- Executes the entire script
- Configures its own ParameterInput object

For rank-specific operations (e.g., printing summary info only once), check `parthenon.my_rank`:

```python
if parthenon.my_rank == 0:
    print(f"Configured {ndim}D problem with resolution {nx}")
```

Best practices:
- Keep scripts deterministic (same parameters on all ranks)
- Guard print statements with rank checks to avoid output spam
- Avoid file I/O unless coordinated (or use rank-specific filenames)
- Don't use random numbers without setting seed based on rank

### Python input advantages

The Python input file (`parthinput.advection.py`) demonstrates several advantages over text files:
- **Command line arguments**: Pass Python-style flags like `--nx=128 --ndim=3`
- **Dimensionality control**: Change `ndim` to easily switch between 1D, 2D, or 3D
- **Calculated parameters**: Automatically derive `derefine_tol` from `refine_tol`
- **Variables**: Define resolution once, use everywhere
- **Type safety**: Use native Python types (lists, not comma-separated strings)
- **Documentation**: Inline comments explaining parameter choices

### Command line arguments

Python input files support both Python-style and Parthenon-style arguments:

```bash
# Python-style arguments (parsed by the script)
./fine_advection-example -i parthinput.advection.py --nx=128 --ndim=3

# Parthenon-style overrides (processed by C++ after Python runs)
./fine_advection-example -i parthinput.advection.py parthenon/time/tlim=0.5

# Both can be combined
./fine_advection-example -i parthinput.advection.py --nx=128 parthenon/time/tlim=0.5
```

Python scripts use `argparse.parse_known_args()` to parse their own flags while ignoring
Parthenon-style overrides, which are applied by C++ after the script completes.