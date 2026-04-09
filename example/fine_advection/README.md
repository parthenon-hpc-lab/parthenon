This example implements upwind advection of a cell-centered scalar variable defined 
on the regular grid and for another cell-centered variable on the fine grid (which
is twice the resolution and is selected using Metadata::Fine). The newer type-based
`SparsePack`s are used throughout and machinery for doing a generalized Stoke's 
theorem based update is included.