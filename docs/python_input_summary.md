# Python Input Support - Implementation Summary

## Overview

This feature enables Python scripts as input files (`.py` instead of `.pin`), allowing programmatic parameter generation. The primary use case is native Python command-line argument parsing with full argparse support (choices, validation, help text, type conversion). The implementation is intentionally minimal: the core only provides a Python interpreter and bindings to `ParameterInput` methods. Everything else (helper classes, JSON parsers, etc.) is optional user-level tooling.

## Key Capabilities

**1. Python Command Line Arguments**
```bash
./app -i input.py --ndim=3 --nx=128 --problem=blast --cfl=0.3 --help
```

Python scripts can use argparse for configuration:
- Type checking: `--nx=128` (enforces int)
- Choices: `--problem` in {blast, linear_wave, kh}
- Help text: `--help` shows all available options
- Defaults: fallback values if not specified
- Custom validation: complex constraints on parameter combinations

**2. Programmatic Configuration**
- Loops, conditionals, math expressions in parameter definitions
- Dimension-agnostic setups (ndim=1/2/3 from command line)
- Load parameters from JSON/YAML/HDF5
- Generate parameter sweeps from environment variables

**3. Flexible Abstractions**
- Use provided helper classes (`InputFile`, `Block`)
- Write your own JSON/YAML parsers
- Direct API usage with no abstractions
- Application-specific parameter generators

## Design Philosophy

**Minimal Core, Flexible Tooling**: The C++ infrastructure only provides:
1. Embedded Python interpreter (via pybind11)
2. ParameterInput bindings (`add_int`, `add_real`, `add_bool`, `add_string`, `add_*_vector`)
3. A way to retrieve the injected ParameterInput object

Users can write their own Python abstractions to suit their needs:
- Helper classes like `InputFile` and `Block` (provided as an example)
- JSON/YAML parsers that populate ParameterInput
- Direct scripting without abstractions
- Application-specific parameter generators

## Architecture

### C++ Side (Required Core)

```
┌─────────────────────────────────────────────┐
│  parthenon_manager.cpp                      │
│  - Detect .py extension                     │
│  - Call LoadParameterInputFromPython()      │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│  parameter_parsers/python_parser.cpp        │
│  - Start embedded Python interpreter       │
│  - Set sys.argv for script                  │
│  - Inject ParameterInput into globals       │
│  - Execute user's .py file                  │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│  pybind/parameter_input_bindings.cpp        │
│  - Expose add_int, add_real, add_bool, ...  │
│  - Expose add_*_vector methods              │
│  - Expose query methods (does_exist, etc.)  │
│  - get_parameter_input() → injected object  │
└─────────────────────────────────────────────┘
```

### Python Side (Optional Tooling)

```
┌─────────────────────────────────────────────┐
│  User's input.py script                     │
│  - import parthenon                         │
│  - pi = parthenon.get_parameter_input()     │
│  - pi.add_int("block", "param", value)      │
│                                             │
│  Optional: use helper abstractions          │
│  - from parthenon_input import InputFile    │
│  - inp = InputFile()                        │
│  - inp.block("mesh", nx1=64, nx2=64)        │
│  - inp.to_parameter_input(pi)               │
└─────────────────────────────────────────────┘
```

## Core Implementation

### File Extension Detection

```cpp
// parthenon_manager.cpp
bool is_python_input = (fs::path(arg.input_filename).extension() == ".py");

#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS
  if (is_python_input) {
    pinput = LoadParameterInputFromPython(arg.input_filename, argc, argv);
  }
#else
  if (is_python_input) {
    PARTHENON_FAIL("Python input detected but not enabled at build time");
  }
#endif
```

### Python Interpreter Lifecycle

```cpp
// parameter_parsers/python_parser.cpp
std::unique_ptr<ParameterInput> LoadParameterInputFromPython(
    const char *python_filename, int argc, char *argv[]) {

  auto pinput = std::make_unique<ParameterInput>();

  py::scoped_interpreter guard{};  // Start interpreter

  // Import parthenon module (makes bindings available)
  py::module_::import("parthenon");

  // Build sys.argv for the script (includes arguments after -i)
  py::list py_argv;
  py_argv.append(python_filename);
  // ... add remaining arguments ...
  py::module_::import("sys").attr("argv") = py_argv;

  // Inject ParameterInput into globals (accessible via get_parameter_input())
  py::globals()["__parthenon_pi__"] =
      py::cast(pinput.get(), py::return_value_policy::reference);

  // Execute user's script
  py::eval_file(python_filename, py::globals());

  return pinput;  // Interpreter destroyed, ParameterInput returned to C++
}
```

### Python Bindings (Minimal API)

```cpp
// pybind/parameter_input_bindings.cpp
PYBIND11_MODULE(parthenon, m) {
  py::class_<parthenon::ParameterInput>(m, "ParameterInput")
      // Add methods (typed, parser interface)
      .def("add_int", &parthenon::ParameterInput::AddParsedParameter<int>)
      .def("add_real", &parthenon::ParameterInput::AddParsedParameter<Real>)
      .def("add_bool", &parthenon::ParameterInput::AddParsedParameter<bool>)
      .def("add_string", &parthenon::ParameterInput::AddParsedParameter<std::string>)
      .def("add_int_vector", &parthenon::ParameterInput::AddParsedParameter<std::vector<int>>)
      .def("add_real_vector", &parthenon::ParameterInput::AddParsedParameter<std::vector<Real>>)
      .def("add_bool_vector", &parthenon::ParameterInput::AddParsedParameter<std::vector<bool>>)
      .def("add_string_vector", &parthenon::ParameterInput::AddParsedParameter<std::vector<std::string>>)
      .def("add_unresolved", /* for parameters from nested .pin files */)

      // Query methods (safe during parsing, don't trigger finalization)
      .def("does_parameter_exist", &parthenon::ParameterInput::DoesParameterExist)
      .def("does_block_exist", &parthenon::ParameterInput::DoesBlockExist)
      .def("get_parameter_names", &parthenon::ParameterInput::GetParameterNames)
      .def("get_blocks_with_prefix", &parthenon::ParameterInput::GetBlocksWithPrefix);

  // Retrieve injected ParameterInput from globals
  m.def("get_parameter_input", []() {
    return py::globals()["__parthenon_pi__"].cast<parthenon::ParameterInput*>();
  });

  // Note: Get methods (get_int, get_real, etc.) are NOT exposed.
  // They trigger FinalizeParsing(), which would break ModifyFromCmdline().
  // Python scripts should only ADD parameters, not query their values.
}
```

## Usage Patterns

### Pattern 1: Direct API (No Helper Classes)

```python
#!/usr/bin/env python3
import parthenon

pi = parthenon.get_parameter_input()

# Add parameters directly
pi.add_int("parthenon/mesh", "nx1", 64)
pi.add_int("parthenon/mesh", "nx2", 64)
pi.add_int("parthenon/mesh", "nx3", 1)
pi.add_real("parthenon/mesh", "x1min", 0.0)
pi.add_real("parthenon/mesh", "x1max", 1.0)

pi.add_real("parthenon/time", "tlim", 1.0)
pi.add_int("parthenon/time", "nlim", 100)
```

### Pattern 2: With Helper Classes (Optional)

```python
#!/usr/bin/env python3
from parthenon_input import InputFile
import parthenon

inp = InputFile()
inp.block("parthenon/mesh", nx1=64, nx2=64, nx3=1)
inp.block("parthenon/time", tlim=1.0, nlim=100)

# Transfer to C++
pi = parthenon.get_parameter_input()
inp.to_parameter_input(pi)
```

### Pattern 3: From JSON (User-Written)

```python
#!/usr/bin/env python3
import json
import parthenon

with open("config.json") as f:
    config = json.load(f)

pi = parthenon.get_parameter_input()
for block_name, params in config.items():
    for key, value in params.items():
        if isinstance(value, int):
            pi.add_int(block_name, key, value)
        elif isinstance(value, float):
            pi.add_real(block_name, key, value)
        # ... etc
```

### Pattern 4: Programmatic Generation

```python
#!/usr/bin/env python3
import argparse
import parthenon

parser = argparse.ArgumentParser()
parser.add_argument("--ndim", type=int, default=2)
parser.add_argument("--nx", type=int, default=64)
args, _ = parser.parse_known_args()

pi = parthenon.get_parameter_input()

# Set mesh based on dimensionality
pi.add_int("parthenon/mesh", "nx1", args.nx)
pi.add_int("parthenon/mesh", "nx2", args.nx if args.ndim >= 2 else 1)
pi.add_int("parthenon/mesh", "nx3", args.nx if args.ndim >= 3 else 1)
```

## Command Line Argument Handling

Python scripts receive arguments via `sys.argv`:

```bash
./myapp -i input.py --nx=128 parthenon/mesh/refinement=static
```

```python
# input.py sees: ["input.py", "--nx=128", "parthenon/mesh/refinement=static"]
import argparse
import parthenon

# Parse Python-style arguments
parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=64)
args, remaining = parser.parse_known_args()  # remaining = ["parthenon/mesh/refinement=static"]

pi = parthenon.get_parameter_input()
pi.add_int("parthenon/mesh", "nx1", args.nx)
# Script completes...

# C++ then processes remaining Parthenon-style arguments via ModifyFromCmdline()
# This sets parthenon/mesh/refinement = "static" (overriding any Python value)
```

## Provided Tooling (Optional)

The `parthenon_input` package provides **example** helper classes:

### InputFile Class

```python
class InputFile:
    """Accumulator for parameter blocks."""
    def block(self, name, **params):
        """Add a parameter block."""
        blk = Block(name, **params)
        self.blocks.append(blk)
        return blk

    def to_parameter_input(self, pi=None):
        """Transfer to C++ ParameterInput with type preservation."""
        for block in self.blocks:
            for key, value in block.params.items():
                # Dispatch based on Python type
                if isinstance(value, bool):
                    pi.add_bool(block.name, key, value)
                elif isinstance(value, int):
                    pi.add_int(block.name, key, value)
                # ... etc
```

**Note**: This is just one possible abstraction. Users can write their own.

## Build System

```cmake
# src/pybind/CMakeLists.txt
pybind11_add_module(parthenon_py parameter_input_bindings.cpp)
set_target_properties(parthenon_py PROPERTIES
    OUTPUT_NAME parthenon
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python)

# Create symlinks to Python packages for convenient PYTHONPATH
add_custom_command(TARGET parthenon_py POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${PYTHON_PACKAGES_DIR}/parthenon_input
    ${PYTHON_LIB_DIR}/parthenon_input)
```

Single PYTHONPATH entry:
```bash
export PYTHONPATH=/path/to/build/lib/python:$PYTHONPATH
```

## Integration with Core Refactor

The Python parser integrates seamlessly with the parser separation refactor:

1. Python script executes, populates ParameterInput via `add_*` methods
2. Interpreter shuts down, returns populated ParameterInput to C++
3. C++ applies command line overrides via `ModifyFromCmdline()`
4. C++ calls `FinalizeParsing()` to mark parsing complete
5. Application queries parameters via `Get<T>()` and `GetOrAdd<T>()`

The Python script runs **before** `FinalizeParsing()`, so it uses the same parser interface as text files.

## Why Get Methods Are Not Exposed

The Python bindings intentionally **do not** expose `Get<T>()` methods:

```cpp
// NOT exposed:
// .def("get_int", &parthenon::ParameterInput::Get<int>)
```

**Reason**: `Get<T>()` triggers `FinalizeParsing()`, which would prevent `ModifyFromCmdline()` from working. Python scripts should only **add** parameters, not query them. If a script needs conditional logic based on existing parameters, use `does_parameter_exist()` and maintain state in Python variables.

## Build Requirements

- CMake option: `-DPARTHENON_ENABLE_PYTHON_BINDINGS=ON`
- pybind11 (found via `find_package(pybind11)`)
- Python 3 development headers

Without Python support:
- `.py` input files trigger clear error message
- No runtime dependency on Python
- All code guarded by `#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS`

## File Organization

```
src/
  parameter_parsers/
    python_parser.hpp         # LoadParameterInputFromPython() declaration
    python_parser.cpp         # Embedded interpreter logic
  pybind/
    CMakeLists.txt           # Python module build
    parameter_input_bindings.cpp  # pybind11 bindings
    README.md                # Python API documentation

scripts/python/packages/
  parthenon_input/           # Optional helper classes (example tooling)
    __init__.py
    input_generator.py       # InputFile, Block classes

example/fine_advection/
  parthinput.advection.py   # Example Python input file
```

## Lines of Code

**C++ Core (~250 lines total)**:
- `python_parser.cpp`: ~90 lines (interpreter lifecycle)
- `parameter_input_bindings.cpp`: ~135 lines (pybind11 bindings)
- `parthenon_manager.cpp`: ~5 lines (call LoadParameterInputFromPython)
- `CMakeLists.txt`: ~30 lines (build configuration)

**Python Tooling (~420 lines, optional)**:
- `input_generator.py`: ~377 lines (InputFile, Block classes)
- `__init__.py`: ~40 lines (exports, mpi_print helper)

Most complexity is in **optional** Python tooling, not the C++ core.

## Extensibility

The minimal core enables diverse use cases:

### JSON Input
```python
import json, parthenon
config = json.load(open("config.json"))
# ... populate ParameterInput from config dict ...
```

### YAML Input
```python
import yaml, parthenon
config = yaml.safe_load(open("config.yaml"))
# ... populate ParameterInput from config dict ...
```

### Parameter Sweeps
```python
import parthenon, os
run_id = int(os.environ.get("RUN_ID", 0))
pi = parthenon.get_parameter_input()
pi.add_real("problem", "amplitude", 0.1 * (run_id + 1))
```

### Application-Specific Abstractions
```python
# User writes their own abstractions
from my_app_utils import PhysicsSetup

setup = PhysicsSetup(eos="ideal", gamma=1.4)
setup.configure_parameter_input()  # Populates ParameterInput internally
```

## Testing

- Unit tests: Python bindings tested via pytest (if desired)
- Regression tests: Example `parthinput.advection.py` runs with fine_advection
- Build tests: Both `-DPARTHENON_ENABLE_PYTHON_BINDINGS=ON` and `OFF` configurations

## Summary

This implementation provides a **minimal, flexible foundation** for Python input support:

1. **Core (C++)**: Just enough to embed Python and expose ParameterInput methods
2. **Tooling (Python)**: Optional abstractions that users can replace or extend
3. **No vendor lock-in**: Users can write their own JSON parsers, abstractions, etc.
4. **Clean integration**: Works seamlessly with parser separation refactor
5. **Backward compatible**: Text input still works, Python is purely additive

The philosophy is: provide the plumbing, let users build their own faucets.
