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
PYTHONPATH=../../build/src/pybind:../../scripts/python/packages/parthenon_tools \
  ./fine_advection-example -i parthinput.advection.py
```

### Python input advantages

The Python input file (`parthinput.advection.py`) demonstrates several advantages over text files:
- **Dimensionality control**: Change `ndim` variable to easily switch between 1D, 2D, or 3D
- **Calculated parameters**: Automatically derive `derefine_tol` from `refine_tol`
- **Variables**: Define resolution once, use everywhere
- **Type safety**: Use native Python types (lists, not comma-separated strings)
- **Documentation**: Inline comments explaining parameter choices