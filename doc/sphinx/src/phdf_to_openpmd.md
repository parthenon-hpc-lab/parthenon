<!-- This file was made in part with generative AI. -->

# Converting PHDF snapshots to openPMD

Analysis mode can read saved PHDF fields and write them through the openPMD
backend without advancing the simulation. This example describes the conversion
tested with AthenaPK turbulence snapshots; it is not a general validation of all
applications, devices, file encodings, or mesh layouts.

## Requirements

Use a Parthenon-based application compatible with the source snapshot's packages
and field definitions. In the example below, `my-application` means that
application's executable (for our tests, `athenaPK`), not a standalone Parthenon
conversion utility.

The build needs HDF5 input and openPMD/ADIOS2 output support. Relevant CMake
settings for our build were:

```bash
-DPARTHENON_DISABLE_HDF5=OFF \
-DPARTHENON_DISABLE_OPENPMD=OFF \
-DPARTHENON_USE_SYSTEM_OPENPMD=OFF \
-DADIOS2_DIR=/path/to/adios2/lib/cmake/adios2 \
-DopenPMD_USE_HDF5=OFF
```

The last setting disables openPMD-api's own HDF5 backend, not Parthenon's PHDF
reader. Other backend configurations may also work; these are the settings used
in our conversion tests, not a complete machine-specific build command.

## Example analysis input

Save the following as `parthinput.analysis`. It assumes the application registers
the logical fields `prim` and `acc` and that both are present in the source.
Replace them with the appropriate logical field labels for another application.

```text
<parthenon/job>
problem_id = converted_turbulence

<parthenon/analysis>
fields = prim, acc

# Disable output blocks inherited from this particular source snapshot.
<parthenon/output1>
dt = -1
dn = -1

<parthenon/output2>
dt = -1
dn = -1

<parthenon/output3>
dt = -1
dn = -1

<parthenon/output99>
file_type = openpmd
output_type = data
variables = prim, acc
id = converted
dt = 1
dn = -1
analysis_output = true
single_precision_output = false
ghost_zones = false

# AthenaPK turbulence options; omit for applications that do not use them.
<problem/turbulence>
calc_temperature = false
calc_vorticity_mag = false
```

The analysis input overrides parameters embedded in the source snapshot. Inspect
that snapshot's input and disable every inherited output block that should not
run. Blocks 1--3 were sufficient for our files; other snapshots can contain
different blocks. Choose an unused block number for the conversion output.

`analysis_output = true` selects output99 for the analysis pass. Its nonnegative
`dt` enables it; `dn = -1` disables cycle-based scheduling. No timestep is taken.
The `fields` list controls loading, whereas `variables` selects the output
fields. Missing or unregistered requested fields are not guaranteed to cause an
error, so inspect the resulting field inventory.

## Run the conversion

Use a fresh output directory and absolute source, executable, and input paths:

```bash
mkdir conversion-smoke
cd conversion-smoke

/path/to/my-application \
  -a /path/to/snapshot.prim.00150.phdf \
  -i /path/to/parthinput.analysis \
  > conversion.out 2> conversion.err
```

Inside our Tuolumne GPU allocation, the launch instead used Flux with one MPI
rank and one GPU:

```bash
flux run -N 1 -n 1 -c 1 -g 1 \
  /path/to/athenaPK \
  -a /path/to/snapshot.prim.00150.phdf \
  -i /path/to/parthinput.analysis \
  > conversion.out 2> conversion.err
```

The launcher and runtime environment depend on the machine and build. A CPU
build or another MPI launcher does not use these GPU-specific Flux arguments.

For this input, our tests produced
`converted_turbulence.converted.00000.bp` and a companion `.pmd` file. Output
numbering can reflect counters inherited from the source. The BP directory is
the openPMD/ADIOS2 data product; preserve its contents together.

## Application callbacks and output meaning

Analysis mode loads the saved mesh and selected fields, communicates boundaries,
and runs the analysis/output callbacks without evolving or remeshing the state.
Application callbacks must be compatible with that selected analysis state.
For example, our AthenaPK turbulence application needed to skip callbacks that
expected forcing-phase fields absent from the selected `prim, acc` state. The
application can use `ParthenonManager::IsAnalysis()` to distinguish this path.

This is a selected-field analysis output, not a replacement simulation
checkpoint. Parthenon rejects analysis-state outputs as normal restart sources.
Fields not saved in the original snapshot cannot be recovered by changing the
destination format or output precision.

## Validation and limits

For the contribution build, we converted one uniform-grid 64³ AthenaPK
turbulence snapshot on an AMD MI300A GPU using one MPI rank. We independently
compared all eight primitive and three acceleration components against the
source PHDF, reconstructed in global grid order. All values matched exactly,
and the snapshot time was preserved exactly. Source fields were float32 and the
converted fields were float64: widening preserves saved values but does not
restore precision discarded by the original output.

The tested environment used Kokkos 5.1.1, ROCm 6.4.3, Cray MPICH 9.0.1, parallel
HDF5 1.14.3.7, openPMD-api 0.17.1, and ADIOS2 2.11.0. Earlier versions of the
same fixes also supported our uniform-grid 256³ and 512³ conversions.

Other architectures, byte orders, refinement layouts, application scenarios,
and particle-removal behavior were not systematically tested. Check the output
field inventory, shapes, values, precision, and source time for a new use case.

The tested driver printed `Driver timed out. Restart to continue.` even though
the analysis output was complete and validated. In this path the evolution loop
is skipped while the final status still considers the inherited simulation time
limit. Do not use that line alone as evidence of conversion success or failure;
check the process status, errors, and output data.
