#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "energy_transfer_driver.hpp"
#include <parthenon/driver.hpp>

#include <adios2.h>
#include <openPMD/openPMD.hpp>

using namespace parthenon::driver::prelude;
using energy_transfer::EnergyTransferDriver;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = ProblemGenerator;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  pman.ParthenonInitPackagesAndMesh();
  {
    EnergyTransferDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver.Execute();
  }
  pman.ParthenonFinalize();
  return 0;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  auto package = std::make_shared<parthenon::StateDescriptor>("energy_transfer");

  // Only register mesh fields when not reading from an ADIOS2/bp5 file
  const bool read_from_file =
      pin->DoesParameterExist("energy_transfer", "input_file");
  if (!read_from_file) {
    parthenon::Metadata m_scalar({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                  parthenon::Metadata::OneCopy});
    parthenon::Metadata m_vector({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                  parthenon::Metadata::OneCopy, parthenon::Metadata::Vector},
                                 std::vector<int>{3});

    package->AddField("rho", m_scalar);
    package->AddField("vel", m_vector);
    package->AddField("mag", m_vector);
    package->AddField("acc", m_vector);
    package->AddField("pres", m_scalar);
  }

  packages.Add(package);
  return packages;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {}

// ============================================================================
// Helper: Shell-filter a field in Fourier space and IFFT to real space.
// Extracts modes with k_low < |k| <= k_high.
// FT_field: input Fourier coefficients (n_comp * fft_size_outbox)
// real_out: output real-space field (n_comp * fft_size_inbox)
// FT_scratch: working array (n_comp * fft_size_outbox)
// ============================================================================
static void ShellFilter(parthenon::FFTManager *fft_mgr, int n_comp,
                        const parthenon::ParArray1D<std::complex<Real>> &FT_field,
                        parthenon::ParArray1D<std::complex<Real>> &FT_scratch,
                        parthenon::ParArray1D<Real> &real_out, Real k_low, Real k_high) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();
  const auto fft_size_inbox = fft_mgr->size_real_space_box();

  auto FT_in = reinterpret_cast<const Kokkos::complex<Real> *>(FT_field.data());
  auto FT_out = reinterpret_cast<Kokkos::complex<Real> *>(FT_scratch.data());
  const int nc = n_comp;

  parthenon::par_for(
      "ShellFilter", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] +
                                       k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        for (int n = 0; n < nc; n++) {
          FT_out[idx + n * fft_size_outbox] =
              in_shell ? FT_in[idx + n * fft_size_outbox]
                       : Kokkos::complex<Real>(0.0, 0.0);
        }
      });

  for (int n = 0; n < n_comp; n++) {
    fft_mgr->Backward(
        reinterpret_cast<std::complex<Real> *>(FT_scratch.data()) + n * fft_size_outbox,
        real_out.data() + n * fft_size_inbox);
  }
  Kokkos::fence();
}

// ============================================================================
// Helper: Fused shell-filter + spectral derivative.
// Computes d(field_Q)/dx_dir for modes in shell (k_low, k_high].
// Result stored in deriv_out (single component, fft_size_inbox).
// FT_field_comp: single component of Fourier field (fft_size_outbox)
// ============================================================================
static void ShellFilterDerivative(
    parthenon::FFTManager *fft_mgr,
    const parthenon::ParArray1D<std::complex<Real>> &FT_field_full, int comp_offset,
    parthenon::ParArray1D<std::complex<Real>> &FT_scratch, int scratch_offset,
    parthenon::ParArray1D<Real> &deriv_out, int out_offset, Real k_low, Real k_high,
    int dir, Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();

  auto FT_in = reinterpret_cast<const Kokkos::complex<Real> *>(FT_field_full.data());
  auto FT_out = reinterpret_cast<Kokkos::complex<Real> *>(FT_scratch.data());
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);
  const int d = dir;
  const Real scale = two_pi_over_L;
  const std::size_t in_off = comp_offset;
  const std::size_t out_off = scratch_offset;

  parthenon::par_for(
      "ShellFilterDeriv", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] +
                                       k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        Real k_phys = scale * k_vec[d];
        FT_out[idx + out_off] =
            in_shell ? imag_unit * k_phys * FT_in[idx + in_off]
                     : Kokkos::complex<Real>(0.0, 0.0);
      });

  fft_mgr->Backward(
      reinterpret_cast<std::complex<Real> *>(FT_scratch.data()) + scratch_offset,
      deriv_out.data() + out_offset);
  Kokkos::fence();
}

// ============================================================================
// Helper: Spectral divergence of a 3-component vector field.
// FT_vec: 3 * fft_size_outbox complex values
// div_out: fft_size_inbox real values
// ============================================================================
static void SpectralDivergence(parthenon::FFTManager *fft_mgr,
                               const parthenon::ParArray1D<std::complex<Real>> &FT_vec,
                               parthenon::ParArray1D<std::complex<Real>> &FT_scratch,
                               parthenon::ParArray1D<Real> &div_out,
                               Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();

  auto FT_in = reinterpret_cast<const Kokkos::complex<Real> *>(FT_vec.data());
  auto FT_out = reinterpret_cast<Kokkos::complex<Real> *>(FT_scratch.data());
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);
  const Real scale = two_pi_over_L;

  parthenon::par_for(
      "SpectralDiv", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        // div = i*(kx*Fx + ky*Fy + kz*Fz)
        auto sum = Kokkos::complex<Real>(0.0, 0.0);
        for (int d = 0; d < 3; d++) {
          Real k_phys = scale * k_vec[d];
          sum += k_phys * FT_in[idx + d * fft_size_outbox];
        }
        FT_out[idx] = imag_unit * sum;
      });

  fft_mgr->Backward(reinterpret_cast<std::complex<Real> *>(FT_scratch.data()),
                     div_out.data());
  Kokkos::fence();
}

// ============================================================================
// Main Execute method
// ============================================================================
parthenon::DriverStatus EnergyTransferDriver::Execute() {
  PreExecute();

  // --- Configuration ---
  const auto binning =
      pinput->GetOrAddString("energy_transfer", "binning", "lin");
  const auto num_shells = pinput->GetOrAddInteger("energy_transfer", "num_shells", 20);
  const auto compute_UU = pinput->GetOrAddBoolean("energy_transfer", "compute_UU", true);
  const auto compute_BB = pinput->GetOrAddBoolean("energy_transfer", "compute_BB", false);
  const auto compute_BUT =
      pinput->GetOrAddBoolean("energy_transfer", "compute_BUT", false);
  const auto compute_PU = pinput->GetOrAddBoolean("energy_transfer", "compute_PU", false);
  const auto compute_FU = pinput->GetOrAddBoolean("energy_transfer", "compute_FU", false);
  const auto output_file =
      pinput->GetOrAddString("energy_transfer", "output_file", "transfer");

  auto mesh_size = pmesh->mesh_size;
  const auto Nx = mesh_size.nx(parthenon::X1DIR);
  const auto Ny = mesh_size.nx(parthenon::X2DIR);
  const auto Nz = mesh_size.nx(parthenon::X3DIR);
  const Real Lx = pinput->GetReal("parthenon/mesh", "x1max") -
                  pinput->GetReal("parthenon/mesh", "x1min");
  const Real two_pi_over_L = 2.0 * M_PI / Lx;

  // Build shell edges
  std::vector<Real> shell_edges;
  if (binning == "lin") {
    shell_edges.push_back(0.5);
    for (int i = 1; i < num_shells; i++) {
      shell_edges.push_back(0.5 + i);
    }
    shell_edges.push_back(Real(Nx) / 2.0 * std::sqrt(3.0));
  } else if (binning == "log") {
    shell_edges.push_back(0.0);
    const Real resolution_exp =
        std::log(Real(Nx) / 8.0) / std::log(2.0) * 4.0 + 1.0;
    const int n_log_bins = static_cast<int>(resolution_exp) + 1;
    for (int i = 0; i <= n_log_bins; i++) {
      shell_edges.push_back(4.0 * std::pow(2.0, (Real(i) - 1.0) / 4.0));
    }
  } else {
    PARTHENON_FAIL("Unknown binning type: " + binning);
  }
  const int n_shells = static_cast<int>(shell_edges.size()) - 1;

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Energy transfer analysis: " << n_shells << " shells, binning=" << binning
              << std::endl;
    std::cout << "Shell edges: ";
    for (const auto &e : shell_edges)
      std::cout << e << " ";
    std::cout << std::endl;
  }

  // --- Get FFT infrastructure ---
  PARTHENON_REQUIRE_THROWS(pmesh->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 currently supported for energy transfer.");
  auto FFTMgr = pmesh->GetFFTManager();
  auto UniformGridHelper = pmesh->GetUniformGridHelper();
  const auto fft_size_inbox = FFTMgr->size_real_space_box();
  const auto fft_size_outbox = FFTMgr->size_fourier_space_box();

  // --- Allocate real-space working arrays (only what's needed) ---
  const bool need_mag = compute_BB || compute_BUT;

  parthenon::ParArray1D<Real> rho_flat("rho_flat", fft_size_inbox);
  parthenon::ParArray1D<Real> vel_flat("vel_flat", 3 * fft_size_inbox);
  parthenon::ParArray1D<Real> mag_flat("mag_flat", need_mag ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> W_flat("W_flat", 3 * fft_size_inbox);
  parthenon::ParArray1D<Real> pres_flat("pres_flat", compute_PU ? fft_size_inbox : 0);
  parthenon::ParArray1D<Real> acc_flat("acc_flat", compute_FU ? 3 * fft_size_inbox : 0);

  // --- Load fields: either from ADIOS2/bp5 file or from existing meshblock data ---
  const bool read_from_file =
      pinput->DoesParameterExist("energy_transfer", "input_file");

  if (read_from_file) {
    const auto input_file =
        pinput->GetString("energy_transfer", "input_file");
    PARTHENON_REQUIRE_THROWS(
        input_file.size() >= 3 &&
            input_file.substr(input_file.size() - 3) == ".bp",
        "input_file must be an ADIOS2/bp5 file (ending in .bp), got: " + input_file);

    // Read directly from ADIOS2/bp5 file into flat arrays
    const auto &local_box = UniformGridHelper->LocalMeshBox;
    const adios2::Dims start = {static_cast<std::size_t>(local_box.low[2]),
                                static_cast<std::size_t>(local_box.low[1]),
                                static_cast<std::size_t>(local_box.low[0])};
    const adios2::Dims count = {static_cast<std::size_t>(local_box.size[2]),
                                static_cast<std::size_t>(local_box.size[1]),
                                static_cast<std::size_t>(local_box.size[0])};

    adios2::ADIOS adios(MPI_COMM_WORLD);
    adios2::IO io = adios.DeclareIO("InputReader");
    adios2::Engine reader = io.Open(input_file, adios2::Mode::Read);
    reader.BeginStep();

    // Validate that file dimensions match the mesh
    {
      auto var = io.InquireVariable<double>("rho");
      PARTHENON_REQUIRE_THROWS(var,
                               "Variable 'rho' not found in " + input_file);
      const auto shape = var.Shape();
      PARTHENON_REQUIRE_THROWS(
          shape.size() == 3 &&
              static_cast<int>(shape[0]) == Nz &&
              static_cast<int>(shape[1]) == Ny &&
              static_cast<int>(shape[2]) == Nx,
          "ADIOS2 file dimensions [" + std::to_string(shape[0]) + ", " +
              std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
              "] do not match mesh dimensions [" + std::to_string(Nz) + ", " +
              std::to_string(Ny) + ", " + std::to_string(Nx) + "]");
    }

    // Collect variable names to read, each gets its own host buffer slot
    struct ReadRequest {
      std::string name;
      parthenon::ParArray1D<Real> *dest;
      std::size_t offset;
    };
    std::vector<ReadRequest> requests;
    requests.push_back({"rho", &rho_flat, 0});
    requests.push_back({"vel_x", &vel_flat, 0});
    requests.push_back({"vel_y", &vel_flat, fft_size_inbox});
    requests.push_back({"vel_z", &vel_flat, 2 * fft_size_inbox});
    if (need_mag) {
      requests.push_back({"mag_x", &mag_flat, 0});
      requests.push_back({"mag_y", &mag_flat, fft_size_inbox});
      requests.push_back({"mag_z", &mag_flat, 2 * fft_size_inbox});
    }
    if (compute_PU) {
      requests.push_back({"pres", &pres_flat, 0});
    }
    if (compute_FU) {
      requests.push_back({"acc_x", &acc_flat, 0});
      requests.push_back({"acc_y", &acc_flat, fft_size_inbox});
      requests.push_back({"acc_z", &acc_flat, 2 * fft_size_inbox});
    }

    // Allocate one host buffer per variable for deferred reads
    const auto n_vars = requests.size();
    std::vector<std::vector<double>> host_bufs(n_vars,
                                               std::vector<double>(fft_size_inbox));

    // Issue all deferred Gets
    for (std::size_t v = 0; v < n_vars; v++) {
      auto var = io.InquireVariable<double>(requests[v].name);
      PARTHENON_REQUIRE_THROWS(
          var, "Variable '" + requests[v].name + "' not found in " + input_file);
      var.SetSelection({start, count});
      reader.Get(var, host_bufs[v].data(), adios2::Mode::Deferred);
    }

    // Single I/O flush for all variables
    reader.PerformGets();

    // Copy from host buffers to device arrays
    for (std::size_t v = 0; v < n_vars; v++) {
      auto dest_sub = Kokkos::subview(
          *requests[v].dest,
          Kokkos::make_pair(requests[v].offset, requests[v].offset + fft_size_inbox));
      if constexpr (std::is_same_v<Real, double>) {
        auto host_view =
            Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                host_bufs[v].data(), fft_size_inbox);
        Kokkos::deep_copy(dest_sub, host_view);
      } else {
        std::vector<Real> conv_buf(fft_size_inbox);
        for (std::size_t i = 0; i < fft_size_inbox; i++) {
          conv_buf[i] = static_cast<Real>(host_bufs[v][i]);
        }
        auto host_view =
            Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                conv_buf.data(), fft_size_inbox);
        Kokkos::deep_copy(dest_sub, host_view);
      }
    }

    reader.EndStep();
    reader.Close();

  } else {
    // Gather from existing Parthenon meshblock fields
    auto &md = pmesh->mesh_data.Get();
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    auto rho_var = md->PackVariables(std::vector<std::string>{"rho"});
    auto vel_var = md->PackVariables(std::vector<std::string>{"vel"});

    auto helper = UniformGridHelper->GetKernelHelper();
    const int num_blocks = pmesh->GetNumMeshBlocksThisRank();

    parthenon::par_for(
        "GatherFields", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto idx = helper.FlatIndex(b, k, j, i);
          rho_flat(idx) = rho_var(b, 0, k, j, i);
          for (int n = 0; n < 3; n++) {
            vel_flat(n * fft_size_inbox + idx) = vel_var(b, n, k, j, i);
          }
        });

    if (need_mag) {
      auto mag_var = md->PackVariables(std::vector<std::string>{"mag"});
      parthenon::par_for(
          "GatherMag", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto idx = helper.FlatIndex(b, k, j, i);
            for (int n = 0; n < 3; n++) {
              mag_flat(n * fft_size_inbox + idx) = mag_var(b, n, k, j, i);
            }
          });
    }

    if (compute_PU) {
      auto pres_var = md->PackVariables(std::vector<std::string>{"pres"});
      parthenon::par_for(
          "GatherPres", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto idx = helper.FlatIndex(b, k, j, i);
            pres_flat(idx) = pres_var(b, 0, k, j, i);
          });
    }

    if (compute_FU) {
      auto acc_var = md->PackVariables(std::vector<std::string>{"acc"});
      parthenon::par_for(
          "GatherAcc", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto idx = helper.FlatIndex(b, k, j, i);
            for (int n = 0; n < 3; n++) {
              acc_flat(n * fft_size_inbox + idx) = acc_var(b, n, k, j, i);
            }
          });
    }
  }

  // Compute W = sqrt(rho) * U
  parthenon::par_for(
      "ComputeW", std::size_t(0), fft_size_inbox - 1,
      KOKKOS_LAMBDA(const std::size_t idx) {
        const Real sqrt_rho = Kokkos::sqrt(rho_flat(idx));
        for (int n = 0; n < 3; n++) {
          W_flat(n * fft_size_inbox + idx) = sqrt_rho * vel_flat(n * fft_size_inbox + idx);
        }
      });

  // --- Forward FFT fields that are needed ---
  parthenon::ParArray1D<std::complex<Real>> FT_W("FT_W", 3 * fft_size_outbox);
  parthenon::ParArray1D<std::complex<Real>> FT_U("FT_U", 3 * fft_size_outbox);
  parthenon::ParArray1D<std::complex<Real>> FT_B("FT_B",
                                                  need_mag ? 3 * fft_size_outbox : 0);

  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(W_flat.data() + n * fft_size_inbox,
                    FT_W.data() + n * fft_size_outbox);
    FFTMgr->Forward(vel_flat.data() + n * fft_size_inbox,
                    FT_U.data() + n * fft_size_outbox);
  }
  if (need_mag) {
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(mag_flat.data() + n * fft_size_inbox,
                      FT_B.data() + n * fft_size_outbox);
    }
  }

  // Forward FFT pressure and acceleration if needed
  parthenon::ParArray1D<std::complex<Real>> FT_P("FT_P",
                                                  compute_PU ? fft_size_outbox : 0);
  if (compute_PU) {
    FFTMgr->Forward(pres_flat.data(), FT_P.data());
  }

  parthenon::ParArray1D<std::complex<Real>> FT_Acc("FT_Acc",
                                                    compute_FU ? 3 * fft_size_outbox : 0);
  if (compute_FU) {
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(acc_flat.data() + n * fft_size_inbox,
                      FT_Acc.data() + n * fft_size_outbox);
    }
  }

  // Compute FT of b = B/sqrt(rho) if needed for BUT
  parthenon::ParArray1D<std::complex<Real>> FT_b("FT_b",
                                                  compute_BUT ? 3 * fft_size_outbox : 0);
  if (compute_BUT) {
    parthenon::ParArray1D<Real> b_flat("b_flat", 3 * fft_size_inbox);
    parthenon::par_for(
        "ComputeSmallB", std::size_t(0), fft_size_inbox - 1,
        KOKKOS_LAMBDA(const std::size_t idx) {
          const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho_flat(idx));
          for (int n = 0; n < 3; n++) {
            b_flat(n * fft_size_inbox + idx) =
                mag_flat(n * fft_size_inbox + idx) * inv_sqrt_rho;
          }
        });
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(b_flat.data() + n * fft_size_inbox,
                      FT_b.data() + n * fft_size_outbox);
    }
  }

  // --- Precompute div(U) (needed for UU and BB compression terms) ---
  parthenon::ParArray1D<Real> DivU("DivU",
                                   (compute_UU || compute_BB) ? fft_size_inbox : 0);
  parthenon::ParArray1D<std::complex<Real>> FT_scratch("FT_scratch", 3 * fft_size_outbox);
  if (compute_UU || compute_BB) {
    SpectralDivergence(FFTMgr, FT_U, FT_scratch, DivU, two_pi_over_L);
  }

  // --- Allocate transfer matrices (host 2D views) ---
  parthenon::HostArray2D<Real> UUA_matrix("UUA", n_shells, n_shells);
  parthenon::HostArray2D<Real> UUC_matrix("UUC", n_shells, n_shells);
  parthenon::HostArray2D<Real> BBA_matrix("BBA", n_shells, n_shells);
  parthenon::HostArray2D<Real> BBC_matrix("BBC", n_shells, n_shells);
  parthenon::HostArray2D<Real> BUT_matrix("BUT", n_shells, n_shells);
  parthenon::HostArray2D<Real> PU_matrix("PU", n_shells, n_shells);
  parthenon::HostArray2D<Real> FU_matrix("FU", n_shells, n_shells);

  // --- Working arrays for shell-filtered fields (only allocate what's needed) ---
  parthenon::ParArray1D<Real> W_Q("W_Q",
                                  compute_UU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> W_K("W_K",
                                  (compute_UU || compute_BUT || compute_PU || compute_FU)
                                      ? 3 * fft_size_inbox
                                      : 0);
  parthenon::ParArray1D<Real> B_Q("B_Q",
                                  need_mag ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> B_K("B_K",
                                  compute_BB ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> UdotGradW_Q("UdotGradW_Q",
                                          compute_UU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> UdotGradB_Q("UdotGradB_Q",
                                          compute_BB ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> bDotGradB_Q("bDotGradB_Q",
                                          compute_BUT ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> gradP_Q("gradP_Q",
                                      compute_PU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> Acc_Q("Acc_Q",
                                    compute_FU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> dW_Q_dx("dW_Q_dx", fft_size_inbox);

  // --- Main double loop ---
  for (int q = 0; q < n_shells; q++) {
    const Real Q_low = shell_edges[q];
    const Real Q_high = shell_edges[q + 1];

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Processing Q shell " << q << "/" << n_shells << " [" << Q_low << ", "
                << Q_high << "]" << std::endl;
    }

    // Shell-filter W_Q
    if (compute_UU) {
      ShellFilter(FFTMgr, 3, FT_W, FT_scratch, W_Q, Q_low, Q_high);

      // Compute (U dot grad) W_Q spectrally: for each W component i,
      // UdotGradW_Q[i] = sum_j U_j * d(W_Q_i)/dx_j
      // First zero UdotGradW_Q
      Kokkos::deep_copy(
          Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
              UdotGradW_Q.data(), 3 * fft_size_inbox),
          0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          // Compute d(W_Q_i)/dx_j via fused shell filter + derivative
          ShellFilterDerivative(FFTMgr, FT_W, comp_i * fft_size_outbox, FT_scratch,
                                0, dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);

          // UdotGradW_Q[i] += U_j * d(W_Q_i)/dx_j
          const std::size_t vel_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "AccumAdvection", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                UdotGradW_Q(out_offset + idx) += vel_flat(vel_offset + idx) * dW_Q_dx(idx);
              });
        }
      }
    }

    // Shell-filter B_Q and compute (U dot grad) B_Q
    if (compute_BB) {
      ShellFilter(FFTMgr, 3, FT_B, FT_scratch, B_Q, Q_low, Q_high);

      Kokkos::deep_copy(
          Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
              UdotGradB_Q.data(), 3 * fft_size_inbox),
          0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          ShellFilterDerivative(FFTMgr, FT_B, comp_i * fft_size_outbox, FT_scratch,
                                0, dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);
          const std::size_t vel_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "AccumAdvectionB", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                UdotGradB_Q(out_offset + idx) +=
                    vel_flat(vel_offset + idx) * dW_Q_dx(idx);
              });
        }
      }
    }

    // Compute (b dot grad) B_Q for BUT term
    if (compute_BUT) {
      if (!compute_BB) {
        ShellFilter(FFTMgr, 3, FT_B, FT_scratch, B_Q, Q_low, Q_high);
      }

      Kokkos::deep_copy(
          Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
              bDotGradB_Q.data(), 3 * fft_size_inbox),
          0.0);

      // b dot grad B_Q: use FT_B for the shell-filtered derivative, multiplied by b
      // (b is in real space, stored in b_flat — but we freed it. Recompute from mag/rho.)
      // Actually, we need b in real space. Let's use mag_flat / sqrt(rho_flat).
      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          ShellFilterDerivative(FFTMgr, FT_B, comp_i * fft_size_outbox, FT_scratch,
                                0, dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);
          // bDotGradB_Q[i] += b_j * d(B_Q_i)/dx_j
          // b_j = mag_j / sqrt(rho)
          const std::size_t mag_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "AccumTension", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                const Real b_j =
                    mag_flat(mag_offset + idx) / Kokkos::sqrt(rho_flat(idx));
                bDotGradB_Q(out_offset + idx) += b_j * dW_Q_dx(idx);
              });
        }
      }
    }

    // Compute (1/sqrt(rho)) * grad(P_Q) for PU term
    if (compute_PU) {
      for (int dir_j = 0; dir_j < 3; dir_j++) {
        ShellFilterDerivative(FFTMgr, FT_P, 0, FT_scratch, 0, gradP_Q,
                              dir_j * fft_size_inbox, Q_low, Q_high, dir_j,
                              two_pi_over_L);
      }
      // Scale by 1/sqrt(rho)
      parthenon::par_for(
          "ScaleGradP", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho_flat(idx));
            for (int n = 0; n < 3; n++) {
              gradP_Q(n * fft_size_inbox + idx) *= inv_sqrt_rho;
            }
          });
    }

    // Shell-filter Acc_Q for FU term
    if (compute_FU) {
      ShellFilter(FFTMgr, 3, FT_Acc, FT_scratch, Acc_Q, Q_low, Q_high);
    }

    // Inner K loop
    for (int kk = 0; kk < n_shells; kk++) {
      const Real K_low = shell_edges[kk];
      const Real K_high = shell_edges[kk + 1];

      // Shell-filter W_K
      if (compute_UU || compute_BUT || compute_PU || compute_FU) {
        ShellFilter(FFTMgr, 3, FT_W, FT_scratch, W_K, K_low, K_high);
      }

      // Shell-filter B_K
      if (compute_BB) {
        ShellFilter(FFTMgr, 3, FT_B, FT_scratch, B_K, K_low, K_high);
      }

      // --- UU advection: -sum(W_K * UdotGradW_Q) ---
      if (compute_UU) {
        Real local_sum_adv = 0.0;
        Kokkos::parallel_reduce(
            "UUA_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              sum += W_K(idx) * UdotGradW_Q(idx);
            },
            Kokkos::Sum<Real>(local_sum_adv));

        Real global_sum_adv = local_sum_adv;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_adv, &global_sum_adv, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        UUA_matrix(kk, q) = -global_sum_adv;

        // UU compression: -0.5 * sum(W_K * W_Q * DivU)
        Real local_sum_comp = 0.0;
        Kokkos::parallel_reduce(
            "UUC_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              const auto comp = idx / fft_size_inbox;
              const auto local_idx = idx % fft_size_inbox;
              sum += W_K(idx) * W_Q(idx) * DivU(local_idx);
            },
            Kokkos::Sum<Real>(local_sum_comp));

        Real global_sum_comp = local_sum_comp;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_comp, &global_sum_comp, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        UUC_matrix(kk, q) = -0.5 * global_sum_comp;
      }

      // --- BB advection: -sum(B_K * UdotGradB_Q) ---
      if (compute_BB) {
        Real local_sum_adv = 0.0;
        Kokkos::parallel_reduce(
            "BBA_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              sum += B_K(idx) * UdotGradB_Q(idx);
            },
            Kokkos::Sum<Real>(local_sum_adv));

        Real global_sum_adv = local_sum_adv;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_adv, &global_sum_adv, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        BBA_matrix(kk, q) = -global_sum_adv;

        // BB compression: -0.5 * sum(B_K * B_Q * DivU)
        Real local_sum_comp = 0.0;
        Kokkos::parallel_reduce(
            "BBC_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += B_K(idx) * B_Q(idx) * DivU(local_idx);
            },
            Kokkos::Sum<Real>(local_sum_comp));

        Real global_sum_comp = local_sum_comp;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_comp, &global_sum_comp, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        BBC_matrix(kk, q) = -0.5 * global_sum_comp;
      }

      // --- BUT: +sum(W_K * bDotGradB_Q) ---
      if (compute_BUT) {
        Real local_sum = 0.0;
        Kokkos::parallel_reduce(
            "BUT_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              sum += W_K(idx) * bDotGradB_Q(idx);
            },
            Kokkos::Sum<Real>(local_sum));

        Real global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        BUT_matrix(kk, q) = global_sum;
      }

      // --- PU: -sum(W_K * (1/sqrt(rho)) * gradP_Q) ---
      if (compute_PU) {
        Real local_sum = 0.0;
        Kokkos::parallel_reduce(
            "PU_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              sum += W_K(idx) * gradP_Q(idx);
            },
            Kokkos::Sum<Real>(local_sum));

        Real global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        PU_matrix(kk, q) = -global_sum;
      }

      // --- FU: +sum(W_K * sqrt(rho) * Acc_Q) ---
      if (compute_FU) {
        Real local_sum = 0.0;
        Kokkos::parallel_reduce(
            "FU_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, Real &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += W_K(idx) * Kokkos::sqrt(rho_flat(local_idx)) * Acc_Q(idx);
            },
            Kokkos::Sum<Real>(local_sum));

        Real global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        FU_matrix(kk, q) = global_sum;
      }

    } // end K loop
  }   // end Q loop

  // --- Write output via openPMD/ADIOS2 ---
  // Python reading example:
  //   import openpmd_api as io
  //   s = io.Series("transfer.bp", io.Access.read_only)
  //   it = s.iterations[0]
  //   shell_edges = it.get_attribute("shell_edges")
  //   UU = it.meshes["UU"][io.Mesh_Record_Component.SCALAR].load_chunk()
  //   s.flush()
  {
    std::string fname = output_file + ".bp";
    openPMD::Series series(fname, openPMD::Access::CREATE,
#ifdef MPI_PARALLEL
                           MPI_COMM_WORLD,
#endif
                           "{}");

    auto it = series.iterations[0];
    it.setAttribute("shell_edges", shell_edges);
    it.setAttribute("n_shells", n_shells);
    it.setAttribute("binning", binning);

    auto write_matrix = [&](const std::string &name,
                            const parthenon::HostArray2D<Real> &matrix) {
      auto mesh = it.meshes[name];
      auto comp = mesh[openPMD::MeshRecordComponent::SCALAR];
      openPMD::Extent extent = {static_cast<uint64_t>(n_shells),
                                static_cast<uint64_t>(n_shells)};
      comp.resetDataset(
          openPMD::Dataset(openPMD::determineDatatype<Real>(), extent));
      comp.storeChunkRaw(matrix.data(), {0, 0}, extent);
    };

    if (compute_UU) {
      write_matrix("UUA", UUA_matrix);
      write_matrix("UUC", UUC_matrix);
      parthenon::HostArray2D<Real> UU_matrix("UU", n_shells, n_shells);
      for (int kk = 0; kk < n_shells; kk++)
        for (int q = 0; q < n_shells; q++)
          UU_matrix(kk, q) = UUA_matrix(kk, q) + UUC_matrix(kk, q);
      write_matrix("UU", UU_matrix);
    }
    if (compute_BB) {
      write_matrix("BBA", BBA_matrix);
      write_matrix("BBC", BBC_matrix);
      parthenon::HostArray2D<Real> BB_matrix("BB", n_shells, n_shells);
      for (int kk = 0; kk < n_shells; kk++)
        for (int q = 0; q < n_shells; q++)
          BB_matrix(kk, q) = BBA_matrix(kk, q) + BBC_matrix(kk, q);
      write_matrix("BB", BB_matrix);
    }
    if (compute_BUT) {
      write_matrix("BUT", BUT_matrix);
    }
    if (compute_PU) {
      write_matrix("PU", PU_matrix);
    }
    if (compute_FU) {
      write_matrix("FU", FU_matrix);
    }

    series.close();
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Wrote " << fname << std::endl;
    }
  }

  Driver::PostExecute(DriverStatus::complete);
  return DriverStatus::complete;
}
