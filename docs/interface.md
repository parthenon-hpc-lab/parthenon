# State Management

Parthenon manages simulation data through a hierarchy of classes designed to provide convenient state management but also high-performance in low-level, performance critical kernels.  This page gives an overview of the basic classes involved in state management.

# Metadata

The metadata class provides a mechanism to inform parthenon about an application's needs with respect to each variable.  It's documentation can be found [here](Metadata.md).  An object is associated with each variable and is queried as needed.

<table>
  <tr>
    <td><b>Flag</b></td>
    <td><b>Description</b></td>
  </tr>
  <tr>
    <td colspan=2 align="center"><i>Flags to control variable topology</i></td>
  </tr>
  <tr>
    <td>cell</td>
    <td>Indicates this is a cell-centered variable</td>
  </tr>
  <tr>
    <td>*face</td>
    <td>Face-centered variable</td>
  </tr>
  <tr>
    <td>*edge</td>
    <td>Edge-centered</td>
</table>

| Flag | Description
|-|-
<td colspan=1>Flags to control variable topology </td>
| cell | Indicates this is a cell-centered variable |
| face | Face-centered variable |
| edge | Edge-centered |
| node | node-centered |

# WIP ParthenonArray

This may end up as a light wrapper around ```Kokkos::View```.  We'll see...

# Variable

The ```Variable``` class collects several associated objects that are needed to store, describe, and update simulation data.

# Container

# ContainerCollection