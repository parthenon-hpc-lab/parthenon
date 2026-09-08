# CalcIndices golden-master data

This directory holds the golden-master file for the `test_calc_indices_gold` unit test
(`[CalcIndices]`), which pins the output of `CalcIndices` (boundary index boxes + ownership
masks) so the upcoming boundary-communication refactor cannot silently change boundary
geometry.

## The gold file is not committed

Like the regression gold standard (`tst/regression/gold_standard/`), the gold data is
**not** stored in the repository. It is a single HDF5 file
(`calc_indices_gold_v<N>.h5`) published as a GitHub release asset and downloaded at CMake
configure time when unit tests are enabled. The download is pinned by
`CALC_INDICES_GOLD_VER` and verified against `CALC_INDICES_GOLD_HASH` (SHA-512) in the
top-level `CMakeLists.txt`. Set `CALC_INDICES_GOLD_SYNC=OFF` to disable the download.

If the file is absent (e.g. offline build, or a new version not yet uploaded), the test
skips with a warning rather than failing.

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
  statically-refined periodic multigrid meshes (leaf + all GMG neighbor relationships).
