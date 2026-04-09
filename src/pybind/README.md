# Python Bindings for Parthenon Parameter Input

This directory contains pybind11 bindings for Parthenon's `ParameterInput` class, enabling typed parameter input from Python without string parsing.

## Overview

The Python input system provides a two-stage approach:

1. **Stage 1: Build mutable parameter structure in Python**
   - Use `InputFile` class to accumulate parameters
   - Full mutability - modify parameters before transfer
   - Cleaner syntax than manual text editing

2. **Stage 2: Transfer to C++ with type preservation**
   - Call `to_parameter_input()` to create typed C++ `ParameterInput`
   - Automatic type dispatch based on Python types
   - No string parsing - direct typed transfer

## Building

To enable Python bindings:

```bash
cmake -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON ..
make
```

The build system will:
1. Try to find system-installed pybind11
2. If not found, automatically fetch from GitHub
3. Build the `parthenon_py` Python module

## Usage

```python
from parthenon_input import InputFile

# Build parameter structure
inp = InputFile()
mesh = inp.block("parthenon/mesh", nx1=64, nx2=64, nx3=64)
mesh.params["x1min"] = 0.0  # Can modify after creation

inp.block("parthenon/time", tlim=1.0, nlim=100)
inp.block("problem", velocity=[1.0, 0.5, 0.0], periodic=True)

# Transfer to C++ with type preservation
pi = inp.to_parameter_input()
```

## Python API

**Adding parameters** (use these in Python input scripts):
- `add_int(block, name, value)`
- `add_real(block, name, value)`
- `add_bool(block, name, value)`
- `add_string(block, name, value)`
- `add_int_vector(block, name, list)`
- `add_real_vector(block, name, list)`
- `add_bool_vector(block, name, list)`
- `add_string_vector(block, name, list)`
- `add_unresolved(block, name, string)` - for lazy conversion

**Querying structure** (safe, const methods):
- `does_parameter_exist(block, name)` - Check if parameter exists
- `does_block_exist(block)` - Check if block exists
- `get_parameter_names(block)` - List parameters in a block
- `get_blocks_with_prefix(prefix)` - Find blocks matching prefix

**Note**: Parameter value retrieval (`Get` methods) is intentionally not exposed to prevent premature finalization. Python scripts should only **add** parameters, not query their values.

## Installation

After building, add the Python module and packages to your PYTHONPATH:

```bash
export PYTHONPATH=/path/to/parthenon/build/lib/python:$PYTHONPATH
```

The build system automatically creates `build/lib/python/` with:
- The compiled `parthenon` module
- Symlinks to `parthenon_input` and `parthenon_tools` packages

Or install system-wide:

```bash
make install
```

## Example

See `example/fine_advection/parthinput.advection.py` for a complete working example.

## Dependencies

- pybind11 (automatically fetched if not found)
- Python 3.6+
- parthenon_input Python package
