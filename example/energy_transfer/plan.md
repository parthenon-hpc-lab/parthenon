# Energy Transfer Analysis — Application Plan

## Purpose

GPU-portable shell-to-shell energy transfer analysis for turbulent MHD flows,
built on the Parthenon framework's FFTManager (HeFFTe backend). Replaces the
CPU-only Python implementation in `external/energy-transfer-analysis/`.

## File Structure

```
example/energy_transfer/
├── energy_transfer_driver.hpp   -- EnergyTransferDriver class (inherits Driver)
├── energy_transfer_driver.cpp   -- main(), ProcessPackages(), ProblemGenerator(), Execute()
├── CMakeLists.txt               -- builds "energy-transfer" target, links ADIOS2
├── parthinput.example           -- sample input deck (64^3, periodic, linear bins)
└── plan.md                      -- this file

external/energy-transfer-analysis/testing/
└── enzo_to_bp5.py              -- Python script: Enzo data (via yt) → ADIOS2/bp5
```

Registered in `example/CMakeLists.txt` via `add_subdirectory(energy_transfer)`.

## Architecture

Follows the standard Parthenon driver pattern:

1. `main()` — ParthenonManager setup, register callbacks, init, execute, finalize
2. `ProcessPackages()` — conditionally registers mesh fields (only when not reading
   from file). Uses `DoesParameterExist("energy_transfer", "input_file")` to decide.
3. `ProblemGenerator()` — no-op (data loading handled in Execute)
4. `EnergyTransferDriver::Execute()` — full analysis workflow

## Data Input

Two mutually exclusive modes, selected by presence of `input_file` in the
`<energy_transfer>` input block:

### Mode 1: ADIOS2/bp5 file (`input_file = path/to/data.bp`)

- Reads directly into flat device arrays, bypassing meshblock fields entirely
- File must end in `.bp`; validated at runtime
- File dimensions validated against mesh configuration (`[Nz, Ny, Nx]`)
- Always reads as `double` from file; converts to `Real` if built with single precision
- Uses ADIOS2 deferred mode: all variables queued, single `PerformGets()` flush
- Each rank reads its local chunk via `SetSelection` based on `UniformGridHelper::LocalMeshBox`
- Python conversion script: `external/energy-transfer-analysis/testing/enzo_to_bp5.py`

### Mode 2: Meshblock fields (no `input_file` parameter)

- Registers `rho`, `vel`, `mag`, `acc`, `pres` fields in `ProcessPackages`
- Gathers from meshblocks into flat arrays via `UniformGridHelper::FlatIndex`
- Intended for use when coupled to a running simulation or custom `ProblemGenerator`

Both paths feed into a shared `ComputeW` kernel: `W = sqrt(rho) * U`.

## Algorithm

### Phase 1: Setup

- Read configuration from `<energy_transfer>` input block
- Build shell edge array (linear or logarithmic binning)
- Load fields (ADIOS2 or meshblock gather, see above)
- Compute derived fields: `W = sqrt(rho) * U`, `b = B / sqrt(rho)`
- Forward FFT: `FT_W`, `FT_U`, `FT_B`, `FT_b`, `FT_P`, `FT_Acc` (conditionally)
- Precompute `div(U)` spectrally (single IFFT)

### Phase 2: Shell-to-Shell Transfer (double loop)

```
For each Q shell:
  Shell-filter Q-dependent fields (IFFT of masked Fourier coefficients)
  Compute Q-dependent derivatives (fused shell-filter + spectral derivative)
    - UdotGradW_Q = sum_j U_j * d(W_Q_i)/dx_j      (9 IFFTs, if UU)
    - UdotGradB_Q = sum_j U_j * d(B_Q_i)/dx_j      (9 IFFTs, if BB)
    - bDotGradB_Q = sum_j b_j * d(B_Q_i)/dx_j      (9 IFFTs, if BUT)
    - bDotGradW_Q = sum_j b_j * d(W_Q_i)/dx_j      (9 IFFTs, if UBTb)
    - grad(B.B_Q)/(2*sqrt(rho))                    (4 FFTs, if BUPbb)
    - div(W_Q/(2*sqrt(rho)))                       (4 FFTs, if UBPbb)
    - gradP_Q     = grad(P_Q) / sqrt(rho)           (3 IFFTs, if PU)
    - Acc_Q       = shell-filtered acceleration      (3 IFFTs, if FU)

  For each K shell:
    Shell-filter W_K (3 IFFTs)
    Shell-filter B_K (3 IFFTs, if BB)

    Compute inner products (parallel_reduce + MPI_Allreduce):
      UUA(K,Q) = -sum(W_K * UdotGradW_Q)
      UUC(K,Q) = -0.5 * sum(W_K * W_Q * DivU)
      BBA(K,Q) = -sum(B_K * UdotGradB_Q)
      BBC(K,Q) = -0.5 * sum(B_K * B_Q * DivU)
      BUT(K,Q) = +sum(W_K * bDotGradB_Q)
      UBTb(K,Q) = +sum(B_K * (bDotGradW_Q + W_Q * Divb))
      BUPbb(K,Q) = -sum(W_K * grad(B.B_Q)/(2*sqrt(rho)))
      UBPbb(K,Q) = -sum(B_K * B * div(W_Q/(2*sqrt(rho))))
      PU(K,Q)  = -sum(W_K * gradP_Q)
      FU(K,Q)  = +sum(W_K * sqrt(rho) * Acc_Q)
```

### Phase 3: Output

Single ADIOS2/bp5 file via openPMD. Each transfer term stored as a named
2D mesh dataset. Shell edges and metadata stored as iteration attributes.

### Optional Direct Hydrodynamic Cross-Scale Flux

`compute_flux_U` and `compute_flux_W` independently enable direct fluxes for
`X = U` and `X = W = sqrt(rho) U`, respectively. Both default to false. To run
only the direct flux calculation, set `compute_UU = false` (its default is true)
and leave all other shell-transfer switches false. Disable all `compute_spec_*`
switches as well when only fluxes are needed. No transfer matrices are allocated
and the double loop is skipped when all shell-transfer switches are false.

At cutoff `c = shell_edges[s]`, define `X_<` using modes in
`(shell_edges.front(), c]` and `X_>` using `(c, shell_edges.back()]`. Then compute

```
flux_X/advection   = -sum_x X_> . (U . grad) X_<
flux_X/compression = -0.5 sum_x X_> . X_< div(U)
flux_X/total       = advection + compression
```

The advecting velocity and its divergence are unfiltered. The unweighted option
uses the same advection/compression decomposition with `W` replaced by `U`;
for incompressible velocity the compression contribution vanishes. This is the
hydrodynamic redistribution term; pressure work and forcing are not included.

Positive flux denotes transfer from lower to higher wavenumbers. Values are
spatial sums, matching the transfer matrices; divide by `Nx*Ny*Nz` for spatial
averages. In particular, the weighted flux at index `s` equals
`sum(UU[s:, :s])`, and its components equal the corresponding sums of `UUA`
and `UUC`. The unweighted result equals this construction with `W` replaced by
`U`. At uniform density `rho0`, weighted flux is `rho0` times unweighted flux.

The mode range deliberately matches the configured shells, including their
exclusion of the mean mode and any truncation at the last edge. The mean velocity
is still included in the advecting field. Each output contains `n_shells + 1`
values, one per shell edge; the first and last are exactly zero because one
filtered set is empty. Shell edges must be strictly increasing.

The direct calculation needs 15 inverse transforms per interior cutoff per
enabled field (six for the two filtered vectors and nine for derivatives),
with linear work in the number of cutoffs and fixed-size device scratch arrays.
It does not sum or construct individual shell transfers. Device and MPI
reductions, and flux output, use double precision even in single-precision builds.

Output meshes `flux_u` and `flux_w` each have the components `advection`,
`compression`, and `total`. Iteration attributes `flux_cutoffs`,
`flux_sign`, `flux_normalization`, and `flux_mode_range` describe the conventions.
For example:

```python
s = io.Series("transfer.%05T.bp", io.Access.read_only)
it = s.iterations[0]
cutoffs = it.get_attribute("flux_cutoffs")
flux_w = it.meshes["flux_w"]["total"].load_chunk()
s.flush()
```

`test_flux.py` checks against an independent NumPy shell-transfer calculation
and against the driver's weighted matrices. It covers varying and constant
density, primitive and conserved inputs, each flux enabled alone, linear/log/test
bins, and a single shell. It requires `numpy`, `adios2`, and `openpmd_api`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python example/energy_transfer/test_flux.py \
    build-test/example/energy_transfer/energy-transfer
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python example/energy_transfer/test_flux.py \
    build-test/example/energy_transfer/energy-transfer --ranks 4
# Add --single when testing a Real=float build.
```

When regression testing is enabled and these Python modules are available at
configuration time, the serial check is also registered as the CTest test
`energy_transfer_flux`.

### Optional Power Spectra

The driver can also compute shell-averaged power spectra with
`utils::fft::CalcSpectrum`, reusing the flat fields already loaded for the
transfer analysis. These spectra use FFTManager's normalized forward transform,
so the summed spectral power is comparable to the volume-averaged real-space
power.

Available spectra:

| Switch | Input field | Output prefix | Power column |
|--------|-------------|---------------|--------------|
| `compute_spec_U` | `U` | `spec/u` | squared velocity Fourier amplitude |
| `compute_spec_rho` | `rho` | `spec/rho` | squared density Fourier amplitude |
| `compute_spec_W` | `W = sqrt(rho) * U` | `spec/w` | squared density-weighted velocity Fourier amplitude |
| `compute_spec_B` | `B` | `spec/b` | squared magnetic-field Fourier amplitude |

Each spectrum output prefix contains three 1D datasets:

- `pow_sum`: shell-summed Fourier power
- `k_sum`: shell-summed wavenumber magnitude
- `count_sum`: weighted mode count

`compute_spec_U` also performs a runtime sanity check comparing
`sum_x |U|^2 / N` against the summed velocity spectrum and checking the mean
mode. The other spectra do not perform sanity checks.

## Implemented Transfer Terms

| Term | Formula | Description |
|------|---------|-------------|
| UUA  | `-W_K * (U . grad)W_Q` | Kinetic advection |
| UUC  | `-0.5 * W_K * W_Q * div(U)` | Kinetic compression |
| UU   | UUA + UUC | Total kinetic |
| BBA  | `-B_K * (U . grad)B_Q` | Magnetic advection |
| BBC  | `-0.5 * B_K * B_Q * div(U)` | Magnetic compression |
| BB   | BBA + BBC | Total magnetic |
| BUT  | `+W_K * (b . grad)B_Q` | Magnetic tension -> KE |
| UBTb | `+B_K * div(b W_Q)` | KE -> magnetic tension |
| BUPbb | `-W_K * grad(B.B_Q)/(2*sqrt(rho))` | Magnetic pressure -> KE |
| UBPbb | `-B_K * B * div(W_Q/(2*sqrt(rho)))` | KE -> magnetic pressure |
| PU   | `-W_K * (1/sqrt(rho)) * grad(P_Q)` | Pressure -> KE |
| FU   | `+W_K * sqrt(rho) * Acc_Q` | Forcing -> KE |

## Key Design Decisions

- **Spectral derivatives** instead of real-space finite differences (more accurate,
  no ghost cell communication needed)
- **Fused shell-filter + derivative**: single Fourier-space kernel followed by one
  IFFT, halving the FFT count for derivative terms
- **Conditional allocation**: arrays sized to 0 when their term is disabled
- **Kokkos::fence()** after every Backward FFT (HeFFTe is async on GPU)
- **HostArray2D** for transfer matrices (dynamically sized, no arbitrary cap)
- **openPMD/ADIOS2** output for self-describing, Python-friendly I/O
- **Dual input modes**: direct ADIOS2 file read OR meshblock gather, selected by
  presence of `input_file` parameter (not a sentinel value)
- **Type-safe I/O**: always reads doubles from ADIOS2, converts to Real if needed
- **Dimension validation**: file shape checked against mesh at startup
- **Batched I/O**: ADIOS2 deferred mode with single PerformGets for all variables

## Configuration (`<energy_transfer>` block)

```
input_file = data.bp    # ADIOS2/bp5 input (must end in .bp); omit for meshblock mode
binning = lin|log|test  # shell edge distribution
num_shells = 20         # number of shells
compute_UU = true       # kinetic transfer
compute_flux_U = false  # direct hydrodynamic flux using U
compute_flux_W = false  # direct hydrodynamic flux using W=sqrt(rho)*U
compute_BB = false      # magnetic transfer
compute_BUT = false     # magnetic tension
compute_UBTb = false    # magnetic tension
compute_BUPbb = false   # magnetic pressure
compute_UBPbb = false   # magnetic pressure
compute_PU = false      # pressure
compute_FU = false      # forcing
compute_spec_U = true   # velocity power spectrum
compute_spec_rho = false # density power spectrum
compute_spec_W = false  # density-weighted velocity power spectrum, W=sqrt(rho)*U
compute_spec_B = false  # magnetic-field power spectrum
output_file = transfer  # output filename base (produces transfer.%05T.bp)
output_number = 0       # openPMD iteration/file number
```

## Python Conversion Script

`external/energy-transfer-analysis/testing/enzo_to_bp5.py`

Converts Enzo simulation data to the expected ADIOS2/bp5 format:

```bash
python enzo_to_bp5.py DD0024/data0024 --output enzo_data.bp --gamma 1.001
python enzo_to_bp5.py DD0024/data0024 --output enzo_data_64.bp --res 64 --gamma 1.001
```

- Reads via yt (`covering_grid`), transposes to `[k, j, i]` order
- Stores all fields as float64 in shape `[Nz, Ny, Nx]`
- Fields: `rho`, `vel_{x,y,z}`, `mag_{x,y,z}`, `acc_{x,y,z}` (optional), `pres` (optional)
- Supports downsampling via `--res` (volume averaging)
- Attributes: `resolution`, `domain_left`, `domain_right`, `gamma`

## Output Format

ADIOS2/bp5 via openPMD. Reading in Python:

```python
import openpmd_api as io

s = io.Series("transfer.bp", io.Access.read_only)
it = s.iterations[0]
shell_edges = it.get_attribute("shell_edges")
n_shells = it.get_attribute("n_shells")
UU = it.meshes["UU"][io.Mesh_Record_Component.SCALAR].load_chunk()
u_power = it.meshes["spec/u/pow_sum"][io.Mesh_Record_Component.SCALAR].load_chunk()
s.flush()
```

Power spectra are stored under `spec/<field>/` with `pow_sum`, `k_sum`, and
`count_sum` components. The magnetic spectrum is raw squared magnetic-field
Fourier amplitude, not magnetic energy, so no `1/2` factor is applied.

## Dependencies

- Parthenon with HeFFTe enabled (`PARTHENON_ENABLE_HEFFTE`)
- ADIOS2 with CXX and MPI components (linked explicitly in CMakeLists.txt)
- openPMD with ADIOS2 backend (for output)
- Uniform grid only (`refinement = none`, `pack_size = -1`)
- Periodic boundary conditions in all directions

## Known Limitations / TODO

- The existing power-spectrum output uses slash-containing scalar mesh names,
  which the openPMD 0.17.1 ADIOS2 backend rejects. Keep `compute_spec_* = false`
  for flux-only runs with this backend. The new flux meshes use record components
  and do not have this restriction.
- Assumes isotropic (cubic) domain: spectral derivatives use `2*pi/Lx` for all directions
- Missing terms: SS (internal energy), UBT, nuU, etaB (dissipation)
- No runtime check that domain is actually cubic when non-cubic would give wrong results
- Python script requires `yt` and `adios2` Python packages
