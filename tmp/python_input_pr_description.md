## PR Summary

## Summary

This PR adds support for Python scripts as input files (`.py` instead of `.pin`), building on the parser separation infrastructure from #1385. Python input files enable programmatic parameter generation with native command-line argument parsing, loops, conditionals, and integration with external data sources. This is obviously inspired by the recent features added to Riot, so we probably want to think about if/how these two things fit together.

**Primary Use Case**: Python's argparse for rich command-line interfaces with type validation, choices, help text, and custom constraints.

**Philosophy**: Minimal C++ core (~250 lines) that embeds Python and exposes `ParameterInput` methods. Optional Python helper classes (~420 lines) demonstrate usage but can be replaced with user-specific abstractions.

## Key Features

### 1. Python Command Line Arguments

```bash
./app -i input.py --ndim=3 --nx=128  --cfl=0.3 --help
```

Python scripts can use argparse for validation:
- Type checking: `--nx=128` (enforces int)
- Help text: `--help` shows all available options
- Defaults: fallback values if not specified
- Complex validation: parameter interdependencies

### 2. Programmatic Configuration

```python
import argparse

def parthenon_init_parameters(pin):
    """Configure parameters based on command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndim", type=int, choices=[1,2,3], default=2)
    parser.add_argument("--nx", type=int, default=64)
    args, _ = parser.parse_known_args()

    pin.add_int("parthenon/mesh", "nx1", args.nx)
    pin.add_int("parthenon/mesh", "nx2", args.nx if args.ndim >= 2 else 1)
    pin.add_int("parthenon/mesh", "nx3", args.nx if args.ndim >= 3 else 1)
```

### 3. Flexible Abstractions

Users can choose their approach:
- **Direct API**: Call `pi.add_int()`, `pi.add_real()`, etc. directly
- **Helper classes**: Use provided `InputFile`/`Block` classes (optional)
- **Custom parsers**: Write JSON/YAML parsers that populate ParameterInput
- **Application-specific**: Build domain-specific parameter generators

## Implementation

### C++ Core (~250 lines)

**New files**:
- `src/parameter_parsers/python_parser.{hpp,cpp}` (~120 lines)
  - Embeds Python interpreter via pybind11
  - Executes user's `.py` script to load function definitions
  - Calls `parthenon_init_parameters(pin)` function with ParameterInput object
  - Returns populated ParameterInput to C++

- `src/pybind/parameter_input_bindings.cpp` (~135 lines)
  - Exposes `add_int`, `add_real`, `add_bool`, `add_string`, `add_*_vector` methods
  - Exposes query methods: `does_parameter_exist`, `get_parameter_names`, etc.
  - **Intentionally does NOT expose `Get<T>()` methods during parsing** (would trigger `FinalizeParsing()` and break command-line overrides)

- `src/pybind/CMakeLists.txt` (~80 lines)
  - Builds `parthenon.so` Python module
  - Creates symlinks for single PYTHONPATH: `export PYTHONPATH=build/lib/python:$PYTHONPATH`

**Modified files**:
- `src/parthenon_manager.cpp`: Detect `.py` extension and call `LoadParameterInputFromPython()`
- `src/config.hpp.in`: Add `PARTHENON_ENABLE_PYTHON_BINDINGS` define
- `src/CMakeLists.txt`: Add `parameter_parsers/*.{cpp,hpp}` to library

### Python Tooling (~420 lines, optional)

**New package**: `scripts/python/packages/parthenon_input/`
- `input_generator.py`: `InputFile` and `Block` classes for structured parameter building
- `__init__.py`: Exports and `mpi_print()` helper for rank 0 printing

**Example**: `example/fine_advection/parthinput.advection.py` (151 lines)
- Demonstrates argparse for ndim-agnostic configuration
- Shows programmatic parameter generation
- Updated README with Python usage instructions

### Documentation

**New files**:
- `docs/python_input_summary.md` (~450 lines)
  - Architecture overview (minimal core philosophy)
  - Multiple usage patterns (direct API, helper classes, JSON, programmatic)
  - Integration with parser separation refactor
  - Extensibility examples

- `src/pybind/README.md` (~100 lines)
  - Python API reference
  - PYTHONPATH setup instructions
  - Example usage patterns

## Integration with Core Refactor (#1385)

Python input integrates seamlessly with parser separation:

1. Python file executes to load function definitions
2. C++ calls `parthenon_init_parameters(pin)` which populates `ParameterInput` via `add_*()` methods (uses `AddParsedParameter()` interface)
3. Function returns, interpreter shuts down, populated `ParameterInput` returned to C++
4. C++ applies command line overrides via `ModifyFromCmdline()` (Parthenon-style `block/param=value`)
5. C++ calls `FinalizeParsing()` to mark parsing complete
6. Application queries parameters via `Get<T>()` and `GetOrAdd<T>()`

Python scripts run **before** `FinalizeParsing()`, using the same parser interface as text files. The explicit function call pattern (`parthenon_init_parameters(pin)`) provides a clear entry point with no magic global variables.

## Build Requirements

```cmake
-DPARTHENON_ENABLE_PYTHON_BINDINGS=ON  # Enable Python input support
```

**Dependencies**:
- pybind11 (found via `find_package(pybind11)`)
- Python 3 development headers

**Without Python support**:
- `.py` input files trigger clear error message: "Python input detected but not enabled at build time"
- No runtime Python dependency
- All code guarded by `#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS`

## Usage Example

```bash
# Build with Python support
cmake -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON ..
make

# Set PYTHONPATH
export PYTHONPATH=/path/to/build/lib/python:$PYTHONPATH

# Run with Python input
./fine_advection -i parthinput.advection.py --ndim=2 --nx=128
```

## Command Line Argument Flow

```bash
./app -i input.py --nx=128 parthenon/mesh/refinement=static
```

1. C++ executes Python file to load function definitions
2. C++ calls `parthenon_init_parameters(pin)`
3. Inside function, Python sees: `sys.argv = ["input.py", "--nx=128", "parthenon/mesh/refinement=static"]`
4. Python uses `parse_known_args()` to consume `--nx=128`, ignores rest
5. Function populates ParameterInput based on parsed arguments and returns
6. C++ receives populated ParameterInput
7. C++ applies `ModifyFromCmdline()` to override `parthenon/mesh/refinement=static`

Both Python-style (`--flag=value`) and Parthenon-style (`block/param=value`) arguments work together.

## Testing

- **Unit tests**: Parameter input bindings tested via existing C++ test infrastructure
- **Regression test**: `example/fine_advection/parthinput.advection.py` demonstrates full workflow
- **Build configurations**: Both `PARTHENON_ENABLE_PYTHON_BINDINGS=ON` and `OFF` tested

## Lines of Code

| Component | Lines | Required |
|-----------|-------|----------|
| C++ core (parameter_parsers, pybind) | ~250 | Yes |
| Python tooling (parthenon_input package) | ~420 | No (example) |
| Documentation (docs, README) | ~550 | - |
| Example (parthinput.advection.py) | ~150 | No (example) |

**Total new code**: ~1370 lines (~250 required, ~1120 optional/documentation)

## Future Extensions

The minimal core enables diverse use cases:

**JSON/YAML input**:
```python
import json

def parthenon_init_parameters(pin):
    """Load configuration from JSON file."""
    config = json.load(open("config.json"))
    for block, params in config.items():
        for key, value in params.items():
            if isinstance(value, int):
                pin.add_int(block, key, value)
            elif isinstance(value, float):
                pin.add_real(block, key, value)
            # ... etc
```

**Parameter sweeps**:
```python
import os

def parthenon_init_parameters(pin):
    """Configure parameters based on environment variable."""
    run_id = int(os.environ["RUN_ID"])
    pin.add_real("problem", "amplitude", 0.1 * run_id)
```

**Application-specific abstractions**:
```python
from my_app import Setup

def parthenon_init_parameters(pin):
    """Use application-specific configuration helper."""
    setup = Setup(param1="value", param2=1.4)
    setup.configure_parameter_input(pin)  # User's custom logic
```

## Breaking Changes

None. This is a pure addition:
- Existing `.pin` files work unchanged
- Python support is opt-in at build time
- No changes to existing public APIs

## Depends On

- #1385 (Parser separation refactor) - must be merged first



## PR Checklist

<!-- Note that some of these check boxes may not apply to all pull requests -->

- [x] Code passes cpplint
- [x] New features are documented.
- [ ] Adds a test for any bugs fixed. Adds tests for new features.
- [x] Code is formatted
- [x] Changes are summarized in CHANGELOG.md
- [ ] Change is breaking (API, behavior, ...)
  - [ ] Change is *additionally* added to CHANGELOG.md in the breaking section
  - [ ] PR is marked as breaking
  - [ ] Short summary API changes at the top of the PR (plus optionally with an automated update/fix script)
- [ ] CI has been triggered on [Darwin](https://re-git.lanl.gov/eap-oss/parthenon/-/pipelines) for performance regression tests.
- [ ] Docs build
- [x] Any contribution that was created or modified with the assistance of generative AI must have a comment disclosing this such as `// This file was made in part with generative AI.`
- [x] (@lanl.gov employees) Update copyright on changed files
