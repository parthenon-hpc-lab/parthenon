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
├── CMakeLists.txt               -- builds "energy-transfer" target
├── parthinput.example           -- sample input deck (64^3, periodic, linear bins)
└── plan.md                      -- this file
```

Registered in `example/CMakeLists.txt` via `add_subdirectory(energy_transfer)`.

## Architecture

Follows the standard Parthenon driver pattern:

1. `main()` — ParthenonManager setup, register callbacks, init, execute, finalize
2. `ProcessPackages()` — registers mesh fields: `rho` (scalar), `vel` (vector-3),
   `mag` (vector-3), `acc` (vector-3), `pres` (scalar)
3. `ProblemGenerator()` — placeholder filling test data (data I/O to be added)
4. `EnergyTransferDriver::Execute()` — full analysis workflow

## Algorithm

### Phase 1: Setup

- Read configuration from `<energy_transfer>` input block
- Build shell edge array (linear or logarithmic binning)
- Gather fields from meshblocks into flat device arrays via `UniformGridHelper`
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

## Configuration (`<energy_transfer>` block)

```
binning = lin|log       # shell edge distribution
num_shells = 20         # number of shells
compute_UU = true       # kinetic transfer
compute_BB = false      # magnetic transfer
compute_BUT = false     # magnetic tension
compute_PU = false      # pressure
compute_FU = false      # forcing
output_file = transfer  # output filename base (produces transfer.bp)
```

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
- openPMD with ADIOS2 backend (default in this branch)
- Uniform grid only (`refinement = none`, `pack_size = -1`)
- Periodic boundary conditions in all directions

## Known Limitations / TODO

- Data reading not implemented (ProblemGenerator is a placeholder)
- Assumes isotropic (cubic) domain: spectral derivatives use `2*pi/Lx` for all directions
- Missing terms: SS (internal energy), UBT, nuU, etaB (dissipation)
- No runtime check that domain is actually cubic when non-cubic would give wrong results
