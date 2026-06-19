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
compute_BB = false      # magnetic transfer
compute_BUT = false     # magnetic tension
compute_UBTb = false    # magnetic tension
compute_BUPbb = false   # magnetic pressure
compute_UBPbb = false   # magnetic pressure
compute_PU = false      # pressure
compute_FU = false      # forcing
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
s.flush()
```

## Dependencies

- Parthenon with HeFFTe enabled (`PARTHENON_ENABLE_HEFFTE`)
- ADIOS2 with CXX and MPI components (linked explicitly in CMakeLists.txt)
- openPMD with ADIOS2 backend (for output)
- Uniform grid only (`refinement = none`, `pack_size = -1`)
- Periodic boundary conditions in all directions

## Known Limitations / TODO

- Assumes isotropic (cubic) domain: spectral derivatives use `2*pi/Lx` for all directions
- Missing terms: SS (internal energy), UBT, nuU, etaB (dissipation)
- No runtime check that domain is actually cubic when non-cubic would give wrong results
- Python script requires `yt` and `adios2` Python packages
