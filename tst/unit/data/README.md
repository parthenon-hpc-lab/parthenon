# CalcIndices gold data

This directory holds the gold file for the `test_calc_indices_gold` unit test
(`[CalcIndices]`), which checks the output of `CalcIndices` for a few small meshes
against a known correct set. Similarly to the regression test gold files, this data
is not stored directly in the git repository. 

## Regenerating and publishing a new version

1. Build the unit tests against an HDF5-enabled configuration.
2. Regenerate locally (writes `calc_indices_gold_v<N>.h5` into this directory):
   ```
   PARTHENON_REGEN_GOLD=1 mpirun -np 1 <build>/tst/unit/unit_tests "[CalcIndices]"
   ```
3. Bump `CALC_INDICES_GOLD_VER` in the top-level `CMakeLists.txt`, add an entry below, and
   update `CALC_INDICES_GOLD_HASH` with the new SHA-512 (`shasum -a 512 <file>`).
4. Upload the file as a release asset:
   ```
   gh release create calc-indices-gold-v<N> calc_indices_gold_v<N>.h5 \
       --title "CalcIndices gold v<N>" --notes "See tst/unit/data/README.md"
   ```

## Version history

- 1: initial golden master. Pins CalcIndices boxes + ownership over 1D/2D/3D
  statically-refined periodic multigrid meshes (leaf + all GMG neighbor relationships)
  and a 2D two-tree forest mesh with a non-trivial (rotation + flip) coordinate
  transformation between trees.
