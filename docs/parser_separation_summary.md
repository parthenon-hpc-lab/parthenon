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
│  - Future: Python, TOML, Rummy bytecode     │
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
void FinalizeParsing();
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
pin->FinalizeParsing();                // Done with parsing

int val = pin->Get<int>(...);      // Query parameters
```

### Pattern 2: Mixed Parsing

```cpp
pin->LoadFromStream(base_file);      // Base config
pin->LoadFromStream(override_file);  // Override some values
// Future parsers could add more parameters here via AddParsedParameter
pin->ModifyFromCmdline(argc, argv);  // Final overrides
pin->FinalizeParsing();
```

## Implementation Details

### AddParsedParameter Flow

1. Check `!parsing_resolved_` (can't add after resolution)
2. Find or create Block in `param_storage_` vector
3. Add or update Parameter with value (typed or UnresolvedString)
4. **Does NOT create QueryRecord** (deferred until first Get/GetOrAdd)

### Type Handling

- **UnresolvedString**: Lazy conversion on first access (used by text file parser)
- **Typed values**: Direct storage (for future parsers that provide typed data)
- **Conversion caching**: Once converted, typed value replaces UnresolvedString

### QueryRecord Creation

QueryRecords are created on **first access** (Get/GetOrAdd), not at parse time:
- `origin_type = Input` (default) for parameters from any parser (text files, future parsers)
- `origin_type = Default` when GetOrAdd adds missing parameter
- `origin_type = SetInCode` when Set<T>() explicitly sets value

This ensures:
- Default value consistency checking works correctly
- Command line overrides behave like input file values
- Typed parameters from future parsers maintain Input origin

## Benefits

1. **Flexibility**: Any parser can feed parameters through `AddParsedParameter`
2. **Type Safety**: Future parsers can provide typed values, avoiding string conversion
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
- ✓ FinalizeParsing enforcement
- ✓ AddParsedParameter with typed values
- ✓ UnresolvedString lazy conversion
- ✓ Parameter ordering preservation
- ✓ Type dispatch
- ✓ All regression tests (restart files, parameter order)

No breaking changes to existing code.
