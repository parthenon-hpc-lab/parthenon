# Anonymous Variables

## Metadata
Variables can be tagged with a variety of `MetadataFlag` values. These flags
primarily allow an application to tell Parthenon to apply certain behaviors to
each field.

### Field Topology
The following fields specify the topology of the variable, and are exclusive:
- `Metadata::None`
- `Metadata::Cell`
- `Metadata::Face`
- `Metadata::Edge`
- `Metadata::Node`

These fields specify what index space the variable is allocated over. E.g.
`Metadata::Cell` specifies that the variable is allocated over the cell index
space in each block.

### Variable Behaviors
TODO
- `Metadata::Advected`
- `Metadata::Conserved`
- `Metadata::Intensive`

### Output
TODO
- `Metadata::Restart`
- `Metadata::Graphics`

### Tensors
TODO
- `Metadata::Vector`
- `Metadata::Tensor`

### Independent/Derived
TODO
- `Metadata::Independent`
- `Metadata::Derived`

### Communication
TODO
- `Metadata::OneCopy`
- `Metadata::FillGhost`
- `Metadata::SharedComms`

### Application Metadata Flags
Applications can allocate their own flags by calling
`Metadata::AllocateNewFlag("FlagName")`. For example:
```c++
using parthenon::Metadata;
using parthenon::MetadataFlag;

MetadataFlag const my_app_flag = Metadata::AllocateNewFlag("MyAppFlag");
```

These can be used in all the same contexts that the built-in metadata flags
are used. Parthenon will not interpret them in any way - it's up to the
application to interpret them.