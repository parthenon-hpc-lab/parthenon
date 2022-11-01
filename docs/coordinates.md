# Coordinates

*Parthenon currently provides a coordinates class `UniformCartesian` for a
uniform Cartesian coordinate system. In the near future we will provide uniform
spherical and uniform cylindrical coordinate systems. Other coordinate systems
may be implemented in downstream codes. Alternatively, coordinate systems can
be incorporated in the fluid equations such as is done in
[Phoebus](https://github.com/lanl/phoebus).*

Coordinate objects under the `Coordinates_t` type are created for each
meshblock. Coordinate information such as positions of different elements
within each cell (cell centers, face centers, edge centers, and nodes),
distance between elements, and integration elements (cell widths, edge centers,
face areas, and cell volumes) can be accessed from the coordinate object for
each meshblock. The coordinates object for meshblock `b` in a variable pack
`var_pack` can be accessed via `var_pack.GetCoords(b)`.

The interface for coordinate systems is derived from Athena++ but with new
names and with directions, faces, and edges specified by either template
parameters at compile time or arguments at run time instead of with different
function names. For ease of conversion from Athena++, we summarize the new
coordinate functions with their Athena++ predecessors.

Position of elements (replacing `x1v`, `x1f`, `x1s2` etc. in Athena++)
```
Xc<int dir>(const int idx); //Cell Centers
Xf<int face, int dir>(const int idx); //Face Centers
Xe<int edge, int dir>(const int idx); //Edge Centers
Xn< int dir>(const int idx); //Cell Nodes
```
where `1<=dir<=3` is which component of the direction being queried (i.e.
x,y,z), `face` and `edge` specify which face/edge is being queried, and `idx`
is the index along `dir`.  For the function `Xf`, when `dir==face` it gives the
same face position as `x1f`, `x2f`, and `x3f` in Athena++. When `dir!=face`, it
returns the face-average position, lying within the plane of the face, which
replaces `x1s2`, `x1s3`, `x2s1`, `x2s3`, `x3s1`, and `x3s2` in Athena++. `Xe`
and `Xn` have no equivalents in Athena++ put return the positions of
off-cell-center locations consistent with these positions computed from
cell-centers and face-centers in Athena++.

Distance between elements (previously `dx1v`, `dx1f` etc. )
```
Dxc<int dir>(const int idx); //Distance between cell centers along dir
Dxf<int dir, int face>(const int idx); //Distance between face centers along dir
Dxe<int dir, int edge>(const int idx); //Distance between edge centers along dir
Dxn< int dir>(const int idx); //Distance between cell Nodes along dir
```
Likewise, `face`, `edge`, `Dxe`, and `Dxn` are new to get distances between
different elements along different dimensions.

Integration elements (replacing `CenterWidth1`, `Edge1Length`, `Face1Area`, and `CellVolume`)
```
CellWidth<int dir>( const int k, const int j, const int i); //Cell Width at cell center
FaceArea<int face>( const int k, const int j, const int i); //Area of Face
EdgeLength<int edge>( const int k, const int j, const int i); //Length of edge
CellVolume( const int k, const int j, const int i); //CellVolume
```
These functions take all three indices to accommodate extension to spherical
and cylindrical.

## Compile Time vs. Run Time Parameters

For each of these functions that take parameters `dir`, `face`, and/or `edge`
as template parameters at compile time we use uppercase first-letter names. As
needed we also introduce versions with runtime parameters using lowercase function names. For example,
```
xc( const int dir, const int idx); //Cell Centers
da( const int dir, const int k, const int j, const int i); //Face Areas
```
This convention avoids naming conflicts between simplified functions and
runtime functions. These lowercase versions are implemented on an as-needed
basis.

*This page will be expanded with the implementation of spherical and cylindrical coordinates.*
