
### List of input file parameters (incomplete)

   |             Option               | Default  | Type   | Description |
   | -------------------------------: | :------- | :----- | :---------- |
   | parthenon/mesh/num_threads       | 1        | int    | Number of (parallel) threads to be used on the host per process. |
   | parthenon/mesh/num_streams       | 1        | int    | Number of (Cuda) streams to be used per GPU. |
   | parthenon/time/perf_cycle_offset | 0        | int | Skip the first N cycles when calculating the final performance (e.g., zone-cycles/wall_second). Allows to hide the initialization overhead in Parthenon, which usually takes place in the first cycles when Containers are allocated, etc. | 