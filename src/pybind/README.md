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
from parthenon_tools import InputFile

# Build parameter structure
inp = InputFile()
mesh = inp.block("parthenon/mesh", nx1=64, nx2=64, nx3=64)
mesh.params["x1min"] = 0.0  # Can modify after creation

inp.block("parthenon/time", tlim=1.0, nlim=100)
inp.block("problem", velocity=[1.0, 0.5, 0.0], periodic=True)

# Transfer to C++ with type preservation
pi = inp.to_parameter_input()
```

## Type Mapping

Python → C++ type dispatch:

- `int` → `Set<int>()`
- `float` → `Set<Real>()`
- `bool` → `Set<bool>()`
- `str` → `Set<std::string>()`
- `list[int]` → `Set<std::vector<int>>()`
- `list[float]` → `Set<std::vector<Real>>()`
- `list[bool]` → `Set<std::vector<bool>>()`
- `list[str]` → `Set<std::vector<std::string>>()`

## Installation

After building, add the Python module to your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/parthenon/build/lib/python
```

Or install the module system-wide:

```bash
make install
```

## Examples

See `scripts/python/packages/parthenon_tools/examples/python_input_example.py` for a complete example.

## Dependencies

- pybind11 (automatically fetched if not found)
- Python 3.6+
- parthenon_tools Python package
