### List of input file parameters (incomplete)

   |             Option                    | Default  | Type   | Description |
   | ------------------------------------: | :------- | :----- | :---------- |
   | <parthenon/time><br>perf_cycle_offset | 0        | int    | Skip the first N cycles when calculating the final performance (e.g., zone-cycles/wall_second). Allows to hide the initialization overhead in Parthenon, which usually takes place in the first cycles when Containers are allocated, etc. | 
   | <parthenon/time><br>ncycle_out        | 1        | int    | Number of cycles between short diagnostic output to standard out containing, e.g., current time, dt, zone-update/wsec. Default: 1 (i.e, every cycle).|
   | <parthenon/time><br>ncycle_out_mesh   | 0        | int    | Number of cycles between printing the mesh structure (e.g., total number of MeshBlocks and number of MeshBlocks per level) to standard out. Use a negative number to also print every time the mesh was modified. Default: 0 (i.e, off). |
   | <parthenon/mesh><br>nghost            | 2        | int    | Number of ghost cells for each mesh block on each side. | 
