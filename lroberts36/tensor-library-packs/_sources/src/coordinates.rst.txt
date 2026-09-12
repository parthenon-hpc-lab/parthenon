.. _coordinates:

Coordinates
===========

*Parthenon currently provides three coordinates classes ``UniformCartesian``,
``UniformCylindrical``, and ``UniformSpherical``.  With small extensions,
Parthenon could support other coordinate systems defined in downstream codes.
Alternatively, coordinate systems can be incorporated in the fluid
equations such as is done
in*\ `Phoebus <https://github.com/lanl/phoebus>`__\ *.*

Coordinates are selected at compile time by passing the appropriate option to cmake.
The default coordinate system is ``UniformCartesian``.  To select an alternative, or
just to explicitly select the default, pass `-DPARTHENON_COORDINATES=OPTION` where
`OPTION` is one of `{UniformCartesian, UniformCylindrical, UniformSpherical}`.

Coordinate objects under the ``Coordinates_t`` type are created for each
meshblock. Coordinate information such as positions of different
elements within each cell (cell centers, face centers, edge centers, and
nodes), distance between elements, and integration elements (cell
widths, edge centers, face areas, and cell volumes) can be accessed from
the coordinate object for each meshblock. The coordinates object for
meshblock ``b`` in a ``SparsePack`` ``var_pack`` can be accessed via
``var_pack.GetCoordinates(b)``.

Coordinate objects provide the following API:

+-----------------------------------------------------------------------------+-----------------------------------------------------+
| Function                                                                    | Description                                         |
+=============================================================================+=====================================================+
| | ``Real Dx<dir>()``                                                        |                                                     |
| | ``Real Dx(int dir)``                                                      | Coordinate spacing between grid lines               |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Dxc<dir>(int idx)``                                                |                                                     |
| | ``Real Dxc<dir>(int k, int j, int i)``                                    |                                                     |
| | ``Real Dxc(int dir, int k, int j, int i)``                                | Spacing between volume centroids                    |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Dxf<dir>(int idx)``                                                |                                                     |
| | ``Real Dxf<dir>(int k, int j, int i)``                                    | Spacing between faces                               |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Xc<dir>(int idx)``                                                 |                                                     |
| | ``Real Xc<dir>(int k, int j, int i)``                                     | Volume centroid position                            |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Xf<dir, face>(int idx)``                                           |                                                     |
| | ``Real Xf<dir, face>(int k, int j, int i)``                               |                                                     |
| | ``Real Xf<dir>(int k,int j,int i) = Xf<dir,dir>(k,j,i)``                  | Face centroid position                              |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real X<dir, TopologicalElement>(int idx)``                              |                                                     |
| | ``Real X<dir, TopologicalElement>(int k, int j, int i)``                  | Position associated with `TopologicalElement`       |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Scale<dir, TopologicalElement>(int k, int j, int i)``              |                                                     |
| | ``Real Scale<TopologicalElement>(int dir, int k, int j, int i)``          | Scale factor                                        |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real CellWidth<dir>(int k, int j, int i)``                              |                                                     |
| | ``Real CellWidth(int dir, int k, int j, int i)``                          | Physical width of cell                              |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real EdgeLength<dir>(int k, int j, int i)``                             |                                                     |
| | ``Real EdgeLength(int dir, int k, int j, int i)``                         | Physical length of edge                             |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real FaceArea<dir>(int k, int j, int i)``                               |                                                     |
| | ``Real FaceArea(int dir, int k, int j, int i)``                           | Area of face                                        |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| ``Real CellVolume(int k, int j, int i)``                                    | Volume of cell                                      |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| | ``Real Volume<TopologicalElement>(int k, int j, int i)``                  |                                                     |
| | ``Real Volume(CellLevel cl, TopologicalElement el, int k, int j, int i)`` | Generalized volume                                  |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| ``std::array<Real, 3> GetXmin()``                                           | Minimum coordinates of block, including ghost cells |
+-----------------------------------------------------------------------------+-----------------------------------------------------+
| ``std::array<int, 3> GetStartIndex()``                                      | Number of ghost cells                               |
+-----------------------------------------------------------------------------+-----------------------------------------------------+

Here, `dir` and `face` specify a direction and should be `X1DIR`, `X2DIR`, or `X3DIR`.  ``TopologicalElement`` is an enum class with elements ``CC``, ``F1``, ``F2``, ``F3``, ``E1``, ``E2``, ``E3``, ``NN``, corresponding to cells, faces, edges, and nodes, with appropriate numbers to indicate directions.  For Parthenon's restricted set of supported coordinate systems, functions that take a single integer index `idx` are understood to correspond with the appropriate dimension's index, i.e. `idx` -> `i` for `dir = X1DIR`, `idx` -> `j` for `dir = X2DIR`, and `idx` -> `k` for `dir = X3DIR`.

