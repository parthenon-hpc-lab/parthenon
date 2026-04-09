# Parameter Parser Separation - Implementation Summary

## Overview

This refactoring separates parameter **parsing** from parameter **storage** in Parthenon's ParameterInput system, enabling multiple input formats (text, Python, TOML, etc.) to coexist without breaking existing functionality.

## Key Design Principles

1. **Parser → Storage Separation**: Any parser can populate parameters through a generic interface
2. **Type Preservation**: Parsers can provide typed values OR unresolved strings (lazy conversion)
3. **Explicit Resolution**: Applications control when parsing is complete
4. **Backward Compatibility**: Existing code works unchanged

## Architecture

### Three Layers

```
┌─────────────────────────────────────────────┐
│  Parser Layer                               │
│  - Text (.pin files)                        │
│  - Python (pybind11)                        │
│  - TOML (future)                            │
│  - Rummy bytecode (future)                  │
└─────────────────┬───────────────────────────┘
                  │ AddParsedParameter(block, name, ParamValue)
                  ↓
┌─────────────────────────────────────────────┐
│  Storage Layer                              │
│  - param_storage_: vector-of-vectors        │
│    (preserves insertion order, O(n) lookup) │
│  - UnresolvedString: lazy type conversion   │
└─────────────────┬───────────────────────────┘
                  │ Get<T>(), GetOrAdd<T>()
                  ↓
┌─────────────────────────────────────────────┐
│  Query Layer                                │
│  - QueryRecord: default value checking      │
│  - Origin tracking (Input/Default/SetInCode)│
│  - Docstring/allowed values validation      │
└─────────────────────────────────────────────┘
```

## Public API

### Parser Interface

```cpp
// Generic interface for any parser
void AddParsedParameter(const std::string &block,
                       const std::string &name,
                       const ParamValue &value);

// Explicitly mark parsing complete
void MarkResolved();
```

### ParamValue Type

```cpp
// Public types for parser interface
struct UnresolvedString {
  std::string value;  // Lazy conversion on first access
};

using ParamValue = std::variant<
  UnresolvedString,      // From file (needs conversion)
  bool, int, Real, std::string,
  std::vector<bool>, std::vector<int>,
  std::vector<Real>, std::vector<std::string>
>;
```

## Usage Patterns

### Pattern 1: Text Input (Existing)

```cpp
pin->LoadFromStream(file);         // Populates linked list
pin->ModifyFromCmdline(argc, argv); // Overrides from command line
pin->MarkResolved();                // Done with parsing

int val = pin->Get<int>(...);      // Query parameters
```

### Pattern 2: Python Input

```python
# Python side
inp = InputFile()
inp.block("parthenon/mesh", nx1=64, nx2=64)
pi = inp.to_parameter_input()  # Calls add_int(), add_real(), etc.
```

```cpp
// C++ side
pin->ModifyFromCmdline(argc, argv);  // Can still override!
pin->MarkResolved();
int val = pin->Get<int>(...);
```

### Pattern 3: Mixed Parsing

```cpp
pin->LoadFromStream(base_file);      // Base config
pin->LoadFromStream(override_file);  // Override some values
// Python or other parser could add more here
pin->ModifyFromCmdline(argc, argv);  // Final overrides
pin->MarkResolved();
```

## Implementation Details

### AddParsedParameter Flow

1. Check `!parsing_resolved_` (can't add after resolution)
2. Find or create Block in `param_storage_` vector
3. Add or update Parameter with value (typed or UnresolvedString)
4. **Does NOT create QueryRecord** (deferred until first Get/GetOrAdd)

### Type Handling

- **UnresolvedString**: Lazy conversion on first access (like text files)
- **Typed values**: Direct storage (Python, future parsers)
- **Conversion caching**: Once converted, typed value replaces UnresolvedString

### QueryRecord Creation

QueryRecords are created on **first access** (Get/GetOrAdd), not at parse time:
- `origin_type = Input` (default) for file/Python parameters
- `origin_type = Default` when GetOrAdd adds missing parameter
- `origin_type = SetInCode` when Set<T>() explicitly sets value

This ensures:
- Default value consistency checking works correctly
- Command line overrides behave like input file values
- Python-typed parameters don't get special SetInCode treatment

## Python Bindings

### Methods

```python
pi = parthenon.ParameterInput()

# Add parameters (parser interface)
pi.add_unresolved("block", "name", "value")  # From file
pi.add_int("block", "name", 42)              # Typed
pi.add_real("block", "name", 3.14)
pi.add_bool("block", "name", True)
pi.add_string("block", "name", "value")
pi.add_int_vector("block", "name", [1, 2, 3])
# ... similar for real_vector, bool_vector, string_vector

# Mark complete (optional - done automatically on first Get)
pi.mark_resolved()

# Query parameters (normal API)
val = pi.get_int("block", "name")
```

### High-Level API

```python
from parthenon_tools import InputFile

inp = InputFile.from_file("base.pin")  # Load existing
mesh = inp.get_block("parthenon/mesh")
mesh.set(nx1=128, nx2=128)             # Override with types

pi = inp.to_parameter_input()         # Transfer to C++
```

## Benefits

1. **Flexibility**: Any parser can feed parameters through `AddParsedParameter`
2. **Type Safety**: Python can provide typed values, avoiding string conversion
3. **Backward Compatible**: Existing text files and code work unchanged
4. **Explicit Control**: Applications decide when parsing is complete
5. **Origin Tracking**: Parameters maintain correct origin (Input vs SetInCode)
6. **Command Line Support**: Overrides work regardless of parser

## Future Extensions

With this separation, new parsers are straightforward:

### TOML Parser
```cpp
void LoadFromTOML(const std::string& file) {
  auto table = toml::parse(file);
  for (auto& [block, params] : table) {
    for (auto& [name, value] : params) {
      ParamValue pv = ConvertTOMLValue(value);
      AddParsedParameter(block, name, pv);
    }
  }
}
```

### Rummy Bytecode
```cpp
void LoadFromRummy(const std::string& file) {
  RummyDeck deck(file);
  deck.compile();
  for (auto& param : deck.parameters()) {
    AddParsedParameter(param.suit(), param.card(), param.value());
  }
}
```

## Storage Implementation

The storage layer uses a vector-of-vectors approach that maintains backward
compatibility with the original linked list implementation:
- `std::vector<Block>` where each `Block` contains `std::vector<Parameter>`
- Preserves insertion order (blocks and parameters within blocks)
- Ensures restart files and parameter dumps maintain consistent ordering
- Maps would provide O(log n) lookups vs O(n) linear search, but would break
  ordering and thus backward compatibility with restart files

## Testing

All existing tests pass:
- ✓ Parameter hashing
- ✓ Delete parameters
- ✓ MarkResolved
- ✓ Python input generation
- ✓ Type dispatch
- ✓ All regression tests (restart files, parameter order)

No breaking changes to existing code.
