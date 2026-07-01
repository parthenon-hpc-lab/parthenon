#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <adios2.h>
#include <openPMD/openPMD.hpp>

#include "energy_transfer_driver.hpp"
#include <parthenon/driver.hpp>
#include <utils/calc_spectrum.hpp>

using namespace parthenon::driver::prelude;
using energy_transfer::EnergyTransferDriver;

// Keep shell-transfer accumulations, MPI collectives, and output matrices in double
// precision even when the application is built with Real=float.
using TransferReal = double;

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
    EnergyTransferDriver driver(pman.pinput.get(), pman.app_input.get(),
                                pman.pmesh.get());
    driver.Execute();
  }
  pman.ParthenonFinalize();
  return 0;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  auto package = std::make_shared<parthenon::StateDescriptor>("energy_transfer");

  // Only register mesh fields when not reading from an ADIOS2/bp5 file
  const bool read_from_file = pin->DoesParameterExist("energy_transfer", "input_file");
  if (!read_from_file) {
    parthenon::Metadata m_scalar({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                  parthenon::Metadata::OneCopy});
    parthenon::Metadata m_vector({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                                  parthenon::Metadata::OneCopy,
                                  parthenon::Metadata::Vector},
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

KOKKOS_INLINE_FUNCTION
static int ComponentWavenumber(const std::array<int, 3> &kji_vec, const int dir) {
  return kji_vec[2 - dir];
}

// ============================================================================
// Helper: Shell-filter a field in Fourier space and IFFT to real space.
// Extracts modes with k_low < |k| <= k_high.
// FT_field: input Fourier coefficients (n_comp * fft_size_outbox)
// real_out: output real-space field (n_comp * fft_size_inbox)
// FT_scratch: working array (n_comp * fft_size_outbox)
// ============================================================================
static void ShellFilter(parthenon::FFTManager *fft_mgr, int n_comp,
                        const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field,
                        parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                        parthenon::ParArray1D<Real> &real_out, Real k_low, Real k_high) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();
  const auto fft_size_inbox = fft_mgr->size_real_space_box();

  auto FT_in = FT_field.data();
  auto FT_out = FT_scratch.data();
  const int nc = n_comp;

  parthenon::par_for(
      "ShellFilter", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(
            Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        for (int n = 0; n < nc; n++) {
          FT_out[idx + n * fft_size_outbox] = in_shell ? FT_in[idx + n * fft_size_outbox]
                                                       : Kokkos::complex<Real>(0.0, 0.0);
        }
      });

  for (int n = 0; n < n_comp; n++) {
    fft_mgr->Backward(FT_scratch.data() + n * fft_size_outbox,
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
    const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_field_full, int comp_offset,
    parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch, int scratch_offset,
    parthenon::ParArray1D<Real> &deriv_out, int out_offset, Real k_low, Real k_high,
    int dir, Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();

  auto FT_in = FT_field_full.data();
  auto FT_out = FT_scratch.data();
  const Kokkos::complex<Real> imag_unit(0.0, 1.0);
  const int d = dir;
  const Real scale = two_pi_over_L;
  const std::size_t in_off = comp_offset;
  const std::size_t out_off = scratch_offset;

  parthenon::par_for(
      "ShellFilterDeriv", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0],
      fb.high[0], KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = Kokkos::sqrt(
            Real(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]));
        auto idx = kernel_helper.FourierFlatIndex(k, j, i);
        bool in_shell = (k_mag > k_low) && (k_mag <= k_high);
        Real k_phys = scale * ComponentWavenumber(k_vec, d);
        FT_out[idx + out_off] = in_shell ? imag_unit * k_phys * FT_in[idx + in_off]
                                         : Kokkos::complex<Real>(0.0, 0.0);
      });

  fft_mgr->Backward(FT_scratch.data() + scratch_offset, deriv_out.data() + out_offset);
  Kokkos::fence();
}

// ============================================================================
// Helper: Spectral divergence of a 3-component vector field.
// FT_vec: 3 * fft_size_outbox complex values
// div_out: fft_size_inbox real values
// ============================================================================
static void SpectralDivergence(parthenon::FFTManager *fft_mgr,
                               const parthenon::ParArray1D<Kokkos::complex<Real>> &FT_vec,
                               parthenon::ParArray1D<Kokkos::complex<Real>> &FT_scratch,
                               parthenon::ParArray1D<Real> &div_out, Real two_pi_over_L) {
  auto fb = fft_mgr->fourier_space_box();
  auto kernel_helper = fft_mgr->GetKernelHelper();
  const auto fft_size_outbox = fft_mgr->size_fourier_space_box();

  auto FT_in = FT_vec.data();
  auto FT_out = FT_scratch.data();
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
          Real k_phys = scale * ComponentWavenumber(k_vec, d);
          sum += k_phys * FT_in[idx + d * fft_size_outbox];
        }
        FT_out[idx] = imag_unit * sum;
      });

  fft_mgr->Backward(FT_scratch.data(), div_out.data());
  Kokkos::fence();
}

// ============================================================================
// Main Execute method
// ============================================================================
parthenon::DriverStatus EnergyTransferDriver::Execute() {
  PreExecute();

  // --- Configuration ---
  const auto binning = pinput->GetOrAddString("energy_transfer", "binning", "lin");
  const auto num_shells = pinput->GetOrAddInteger("energy_transfer", "num_shells", 20);
  const auto compute_UU = pinput->GetOrAddBoolean("energy_transfer", "compute_UU", true);
  const auto compute_BB = pinput->GetOrAddBoolean("energy_transfer", "compute_BB", false);
  const auto compute_BUT =
      pinput->GetOrAddBoolean("energy_transfer", "compute_BUT", false);
  const auto compute_UBTb =
      pinput->GetOrAddBoolean("energy_transfer", "compute_UBTb", false);
  const auto compute_BUPbb =
      pinput->GetOrAddBoolean("energy_transfer", "compute_BUPbb", false);
  const auto compute_UBPbb =
      pinput->GetOrAddBoolean("energy_transfer", "compute_UBPbb", false);
  const auto compute_PU = pinput->GetOrAddBoolean("energy_transfer", "compute_PU", false);
  const auto compute_FU = pinput->GetOrAddBoolean("energy_transfer", "compute_FU", false);

  // energy spectra config
  const auto compute_spec_U =
      pinput->GetOrAddBoolean("energy_transfer", "compute_spec_U", true);

  // Input data config
  const bool read_from_file = pinput->DoesParameterExist("energy_transfer", "input_file");
  const auto input_quantity_type =
      read_from_file
          ? pinput->GetOrAddString("energy_transfer", "input_quantity_type", "primitive")
          : std::string("primitive");
  PARTHENON_REQUIRE_THROWS(input_quantity_type == "primitive" ||
                               input_quantity_type == "conserved",
                           "energy_transfer/input_quantity_type must be 'primitive' "
                           "or 'conserved'");
  const bool input_conserved = input_quantity_type == "conserved";
  const Real gamma = input_conserved && compute_PU
                         ? pinput->GetOrAddReal("energy_transfer", "gamma", 5.0 / 3.0)
                         : 0.0;
  const auto output_file =
      pinput->GetOrAddString("energy_transfer", "output_file", "transfer");
  const auto output_number =
      pinput->GetOrAddInteger("energy_transfer", "output_number", 0);
  if (output_number < 0) {
    PARTHENON_FAIL("energy_transfer/output_number must be non-negative");
  }

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
    const Real resolution_exp = std::log(Real(Nx) / 8.0) / std::log(2.0) * 4.0 + 1.0;
    const int n_log_bins = static_cast<int>(resolution_exp) + 1;
    for (int i = 0; i <= n_log_bins; i++) {
      shell_edges.push_back(4.0 * std::pow(2.0, (Real(i) - 1.0) / 4.0));
    }
  } else if (binning == "test") {
    shell_edges = {0.5, 1.5, 2.5, 16.0, 26.5, 28.5, 32.0};
  } else {
    PARTHENON_FAIL("Unknown binning type: " + binning);
  }
  const int n_shells = static_cast<int>(shell_edges.size()) - 1;

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Energy transfer analysis: " << n_shells
              << " shells, binning=" << binning << std::endl;
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
  const bool need_mag =
      compute_BB || compute_BUT || compute_UBTb || compute_BUPbb || compute_UBPbb;
  const bool need_b_flat = compute_BUT || compute_UBTb;
  const bool need_FT_b = compute_UBTb;
  const bool need_DivU = compute_UU || compute_BB;
  const bool need_scalar_scratch = compute_UBTb || compute_BUPbb;
  const bool need_mag_loaded = need_mag || (input_conserved && compute_PU);

  parthenon::ParArray1D<Real> rho_flat("rho_flat", fft_size_inbox);
  parthenon::ParArray1D<Real> vel_flat("vel_flat", 3 * fft_size_inbox);
  parthenon::ParArray1D<Real> mag_flat("mag_flat",
                                       need_mag_loaded ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> W_flat("W_flat", 3 * fft_size_inbox);
  parthenon::ParArray1D<Real> pres_flat("pres_flat", compute_PU ? fft_size_inbox : 0);
  parthenon::ParArray1D<Real> acc_flat("acc_flat", compute_FU ? 3 * fft_size_inbox : 0);

  // --- Load fields: either from ADIOS2/bp5 file or from existing meshblock data ---
  if (read_from_file) {
    const auto input_file = pinput->GetString("energy_transfer", "input_file");
    PARTHENON_REQUIRE_THROWS(
        input_file.size() >= 3 && input_file.substr(input_file.size() - 3) == ".bp",
        "input_file must be an ADIOS2/bp5 file (ending in .bp), got: " + input_file);

    // Read directly from ADIOS2/bp5 file into flat arrays
    const auto &local_box = UniformGridHelper->local_mesh_box;
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

    enum class InputRealType { Float, Double };

    auto get_input_type_and_shape = [&](const std::string &name,
                                        adios2::Dims &shape) -> InputRealType {
      auto var_double = io.InquireVariable<double>(name);
      if (var_double) {
        shape = var_double.Shape();
        return InputRealType::Double;
      }
      auto var_float = io.InquireVariable<float>(name);
      if (var_float) {
        shape = var_float.Shape();
        return InputRealType::Float;
      }
      PARTHENON_FAIL("Variable '" + name + "' not found as float or double in " +
                     input_file);
      return InputRealType::Double;
    };

    const auto input_variable_prefix =
        pinput->GetOrAddString("energy_transfer", "input_variable_prefix", "");
    auto join_input_name = [&](const std::string &mesh,
                               const std::string &field) -> std::string {
      std::string name = input_variable_prefix;
      auto append = [&](const std::string &part) {
        if (part.empty()) return;
        if (!name.empty() && name.back() != '/') name += "/";
        name += part;
      };
      append(mesh);
      append(field);
      return name;
    };
    auto input_name = [&](const std::string &mesh_param, const std::string &field_param,
                          const std::string &flat_default,
                          const std::string &component_default) -> std::string {
      const auto mesh =
          pinput->GetOrAddString("energy_transfer", mesh_param, std::string(""));
      const auto field_default = mesh.empty() ? flat_default : component_default;
      const auto field =
          pinput->GetOrAddString("energy_transfer", field_param, field_default);
      return join_input_name(mesh, field);
    };

    auto validate_shape = [&](const std::string &name, const adios2::Dims &shape) {
      PARTHENON_REQUIRE_THROWS(
          shape.size() == 3 && static_cast<int>(shape[0]) == Nz &&
              static_cast<int>(shape[1]) == Ny && static_cast<int>(shape[2]) == Nx,
          "ADIOS2 variable '" + name + "' dimensions [" + std::to_string(shape[0]) +
              ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
              "] do not match mesh dimensions [" + std::to_string(Nz) + ", " +
              std::to_string(Ny) + ", " + std::to_string(Nx) + "]");
    };

    auto read_field = [&](const std::string &name, parthenon::ParArray1D<Real> &dest,
                          const std::size_t offset) {
      adios2::Dims shape;
      const auto type = get_input_type_and_shape(name, shape);
      validate_shape(name, shape);
      auto dest_sub =
          Kokkos::subview(dest, Kokkos::make_pair(offset, offset + fft_size_inbox));
      auto dest_view = dest;
      const auto dest_offset = offset;
      if (type == InputRealType::Double) {
        auto var = io.InquireVariable<double>(name);
        PARTHENON_REQUIRE_THROWS(var,
                                 "Variable '" + name + "' not found in " + input_file);
        std::vector<double> buffer(fft_size_inbox);
        var.SetSelection({start, count});
        reader.Get(var, buffer.data(), adios2::Mode::Deferred);
        reader.PerformGets();
#if SINGLE_PRECISION_ENABLED
        parthenon::ParArray1D<double> input_double("input_double", fft_size_inbox);
        auto host_view =
            Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                buffer.data(), fft_size_inbox);
        Kokkos::deep_copy(input_double, host_view);
        parthenon::par_for(
            "ConvertInputDoubleToReal", std::size_t(0), fft_size_inbox - 1,
            KOKKOS_LAMBDA(const std::size_t idx) {
              dest_view(dest_offset + idx) = static_cast<Real>(input_double(idx));
            });
        Kokkos::fence();
#else
        auto host_view = Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            buffer.data(), fft_size_inbox);
        Kokkos::deep_copy(dest_sub, host_view);
#endif
      } else {
        auto var = io.InquireVariable<float>(name);
        PARTHENON_REQUIRE_THROWS(var,
                                 "Variable '" + name + "' not found in " + input_file);
        std::vector<float> buffer(fft_size_inbox);
        var.SetSelection({start, count});
        reader.Get(var, buffer.data(), adios2::Mode::Deferred);
        reader.PerformGets();
#if SINGLE_PRECISION_ENABLED
        auto host_view = Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            buffer.data(), fft_size_inbox);
        Kokkos::deep_copy(dest_sub, host_view);
#else
        parthenon::ParArray1D<float> input_float("input_float", fft_size_inbox);
        auto host_view =
            Kokkos::View<float *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                buffer.data(), fft_size_inbox);
        Kokkos::deep_copy(input_float, host_view);
        parthenon::par_for(
            "ConvertInputFloatToReal", std::size_t(0), fft_size_inbox - 1,
            KOKKOS_LAMBDA(const std::size_t idx) {
              dest_view(dest_offset + idx) = static_cast<Real>(input_float(idx));
            });
        Kokkos::fence();
#endif
      }
    };

    const auto rho_name =
        input_name("input_rho_mesh", "input_rho_field", "rho", "SCALAR");
    read_field(rho_name, rho_flat, 0);

    if (input_conserved) {
      read_field(
          input_name("input_momentum_mesh", "input_momentum_x_field", "mom_x", "x"),
          vel_flat, 0);
      read_field(
          input_name("input_momentum_mesh", "input_momentum_y_field", "mom_y", "y"),
          vel_flat, fft_size_inbox);
      read_field(
          input_name("input_momentum_mesh", "input_momentum_z_field", "mom_z", "z"),
          vel_flat, 2 * fft_size_inbox);
    } else {
      read_field(
          input_name("input_velocity_mesh", "input_velocity_x_field", "vel_x", "x"),
          vel_flat, 0);
      read_field(
          input_name("input_velocity_mesh", "input_velocity_y_field", "vel_y", "y"),
          vel_flat, fft_size_inbox);
      read_field(
          input_name("input_velocity_mesh", "input_velocity_z_field", "vel_z", "z"),
          vel_flat, 2 * fft_size_inbox);
    }

    if (need_mag_loaded) {
      read_field(
          input_name("input_magnetic_mesh", "input_magnetic_x_field", "mag_x", "x"),
          mag_flat, 0);
      read_field(
          input_name("input_magnetic_mesh", "input_magnetic_y_field", "mag_y", "y"),
          mag_flat, fft_size_inbox);
      read_field(
          input_name("input_magnetic_mesh", "input_magnetic_z_field", "mag_z", "z"),
          mag_flat, 2 * fft_size_inbox);
    }

    if (compute_PU) {
      if (input_conserved) {
        read_field(input_name("input_total_energy_mesh", "input_total_energy_field",
                              "total_energy", "SCALAR"),
                   pres_flat, 0);
      } else {
        read_field(
            input_name("input_pressure_mesh", "input_pressure_field", "pres", "SCALAR"),
            pres_flat, 0);
      }
    }

    if (compute_FU) {
      read_field(input_name("input_acceleration_mesh", "input_acceleration_x_field",
                            "acc_x", "x"),
                 acc_flat, 0);
      read_field(input_name("input_acceleration_mesh", "input_acceleration_y_field",
                            "acc_y", "y"),
                 acc_flat, fft_size_inbox);
      read_field(input_name("input_acceleration_mesh", "input_acceleration_z_field",
                            "acc_z", "z"),
                 acc_flat, 2 * fft_size_inbox);
    }

    if (input_conserved) {
      parthenon::par_for(
          "ConvertMomentumToVelocity", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            const Real inv_rho = 1.0 / rho_flat(idx);
            for (int n = 0; n < 3; n++) {
              vel_flat(n * fft_size_inbox + idx) *= inv_rho;
            }
          });
      Kokkos::fence();

      if (compute_PU) {
        const Real gm1 = gamma - 1.0;
        parthenon::par_for(
            "ConvertTotalEnergyToPressure", std::size_t(0), fft_size_inbox - 1,
            KOKKOS_LAMBDA(const std::size_t idx) {
              Real v2 = 0.0;
              Real b2 = 0.0;
              for (int n = 0; n < 3; n++) {
                const Real v = vel_flat(n * fft_size_inbox + idx);
                const Real b = mag_flat(n * fft_size_inbox + idx);
                v2 += v * v;
                b2 += b * b;
              }
              pres_flat(idx) =
                  gm1 * (pres_flat(idx) - 0.5 * rho_flat(idx) * v2 - 0.5 * b2);
            });
        Kokkos::fence();
        if (!need_mag) {
          mag_flat = parthenon::ParArray1D<Real>();
        }
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
          W_flat(n * fft_size_inbox + idx) =
              sqrt_rho * vel_flat(n * fft_size_inbox + idx);
        }
      });
  Kokkos::fence();

  // --- Forward FFT fields that are needed ---
  parthenon::ParArray1D<Real> DivU("DivU", need_DivU ? fft_size_inbox : 0);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scratch("FT_scratch",
                                                          3 * fft_size_outbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_W("FT_W", 3 * fft_size_outbox);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_U("FT_U",
                                                    need_DivU ? 3 * fft_size_outbox : 0);

  for (int n = 0; n < 3; n++) {
    FFTMgr->Forward(W_flat.data() + n * fft_size_inbox,
                    FT_W.data() + n * fft_size_outbox);
    if (need_DivU) {
      FFTMgr->Forward(vel_flat.data() + n * fft_size_inbox,
                      FT_U.data() + n * fft_size_outbox);
    }
  }
  W_flat = parthenon::ParArray1D<Real>();
  if (need_DivU) {
    SpectralDivergence(FFTMgr, FT_U, FT_scratch, DivU, two_pi_over_L);
    FT_U = parthenon::ParArray1D<Kokkos::complex<Real>>();
  }

  parthenon::ParArray1D<Kokkos::complex<Real>> FT_B("FT_B",
                                                    need_mag ? 3 * fft_size_outbox : 0);
  if (need_mag) {
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(mag_flat.data() + n * fft_size_inbox,
                      FT_B.data() + n * fft_size_outbox);
    }
  }

  // Forward FFT pressure and acceleration if needed
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_P("FT_P",
                                                    compute_PU ? fft_size_outbox : 0);
  if (compute_PU) {
    FFTMgr->Forward(pres_flat.data(), FT_P.data());
    pres_flat = parthenon::ParArray1D<Real>();
  }

  parthenon::ParArray1D<Kokkos::complex<Real>> FT_Acc(
      "FT_Acc", compute_FU ? 3 * fft_size_outbox : 0);
  if (compute_FU) {
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(acc_flat.data() + n * fft_size_inbox,
                      FT_Acc.data() + n * fft_size_outbox);
    }
    acc_flat = parthenon::ParArray1D<Real>();
  }

  // Compute b = B/sqrt(rho) and its FT for magnetic tension terms.
  parthenon::ParArray1D<Real> b_flat("b_flat", need_b_flat ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_b("FT_b",
                                                    need_FT_b ? 3 * fft_size_outbox : 0);
  if (need_b_flat) {
    parthenon::par_for(
        "ComputeSmallB", std::size_t(0), fft_size_inbox - 1,
        KOKKOS_LAMBDA(const std::size_t idx) {
          const Real inv_sqrt_rho = 1.0 / Kokkos::sqrt(rho_flat(idx));
          for (int n = 0; n < 3; n++) {
            b_flat(n * fft_size_inbox + idx) =
                mag_flat(n * fft_size_inbox + idx) * inv_sqrt_rho;
          }
        });
    Kokkos::fence();
  }
  if (need_FT_b) {
    for (int n = 0; n < 3; n++) {
      FFTMgr->Forward(b_flat.data() + n * fft_size_inbox,
                      FT_b.data() + n * fft_size_outbox);
    }
  }

  // --- Allocate transfer matrices (host 2D views) ---
  parthenon::HostArray2D<TransferReal> UUA_matrix("UUA", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> UUC_matrix("UUC", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> BBA_matrix("BBA", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> BBC_matrix("BBC", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> BUT_matrix("BUT", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> UBTb_matrix("UBTb", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> UBTbA_matrix("UBTbA", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> UBTbC_matrix("UBTbC", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> BUPbb_matrix("BUPbb", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> UBPbb_matrix("UBPbb", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> PU_matrix("PU", n_shells, n_shells);
  parthenon::HostArray2D<TransferReal> FU_matrix("FU", n_shells, n_shells);

  // --- Working arrays for shell-filtered fields (only allocate what's needed) ---
  parthenon::ParArray1D<Real> W_Q(
      "W_Q", (compute_UU || compute_UBTb || compute_UBPbb) ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> W_K(
      "W_K", (compute_UU || compute_BUT || compute_BUPbb || compute_PU || compute_FU)
                 ? 3 * fft_size_inbox
                 : 0);
  parthenon::ParArray1D<Real> B_Q("B_Q", need_mag ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> B_K(
      "B_K", (compute_BB || compute_UBTb || compute_UBPbb) ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> UdotGradW_Q("UdotGradW_Q",
                                          compute_UU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> UdotGradB_Q("UdotGradB_Q",
                                          compute_BB ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> bDotGradB_Q("bDotGradB_Q",
                                          compute_BUT ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> DivbW_Q("DivbW_Q", compute_UBTb ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> bDotGradW_Q("bDotGradW_Q",
                                          compute_UBTb ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> Divb("Divb", compute_UBTb ? fft_size_inbox : 0);
  parthenon::ParArray1D<Real> scalar_scratch("scalar_scratch",
                                             need_scalar_scratch ? fft_size_inbox : 0);
  parthenon::ParArray1D<Real> gradBdotBQ("gradBdotBQ",
                                         compute_BUPbb ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_scalar_scratch(
      "FT_scalar_scratch", need_scalar_scratch ? fft_size_outbox : 0);
  parthenon::ParArray1D<Real> W_Q_over_sqrt_rho("W_Q_over_sqrt_rho",
                                                compute_UBPbb ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> DivWQOverSqrtRho("DivWQOverSqrtRho",
                                               compute_UBPbb ? fft_size_inbox : 0);
  parthenon::ParArray1D<Kokkos::complex<Real>> FT_vector_scratch(
      "FT_vector_scratch", compute_UBPbb ? 3 * fft_size_outbox : 0);
  parthenon::ParArray1D<Real> gradP_Q("gradP_Q", compute_PU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> Acc_Q("Acc_Q", compute_FU ? 3 * fft_size_inbox : 0);
  parthenon::ParArray1D<Real> dW_Q_dx("dW_Q_dx", fft_size_inbox);

  if (compute_UBTb) {
    SpectralDivergence(FFTMgr, FT_b, FT_scratch, Divb, two_pi_over_L);
    FT_b = parthenon::ParArray1D<Kokkos::complex<Real>>();
  }

  // --- Main double loop ---
  for (int q = 0; q < n_shells; q++) {
    const Real Q_low = shell_edges[q];
    const Real Q_high = shell_edges[q + 1];

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Processing Q shell " << q << "/" << n_shells << " [" << Q_low << ", "
                << Q_high << "]" << std::endl;
    }

    // Shell-filter W_Q
    if (compute_UU || compute_UBTb || compute_UBPbb) {
      ShellFilter(FFTMgr, 3, FT_W, FT_scratch, W_Q, Q_low, Q_high);
    }

    if (compute_UU) {
      // Compute (U dot grad) W_Q spectrally: for each W component i,
      // UdotGradW_Q[i] = sum_j U_j * d(W_Q_i)/dx_j
      // First zero UdotGradW_Q
      Kokkos::deep_copy(Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
                            UdotGradW_Q.data(), 3 * fft_size_inbox),
                        0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          // Compute d(W_Q_i)/dx_j via fused shell filter + derivative
          ShellFilterDerivative(FFTMgr, FT_W, comp_i * fft_size_outbox, FT_scratch, 0,
                                dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);

          // UdotGradW_Q[i] += U_j * d(W_Q_i)/dx_j
          const std::size_t vel_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "AccumAdvection", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                UdotGradW_Q(out_offset + idx) +=
                    vel_flat(vel_offset + idx) * dW_Q_dx(idx);
              });
        }
      }
    }

    // Shell-filter B_Q and compute (U dot grad) B_Q
    if (compute_BB || compute_BUT || compute_BUPbb) {
      ShellFilter(FFTMgr, 3, FT_B, FT_scratch, B_Q, Q_low, Q_high);
    }

    if (compute_BB) {
      Kokkos::deep_copy(Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
                            UdotGradB_Q.data(), 3 * fft_size_inbox),
                        0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          ShellFilterDerivative(FFTMgr, FT_B, comp_i * fft_size_outbox, FT_scratch, 0,
                                dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);
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
      Kokkos::deep_copy(Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
                            bDotGradB_Q.data(), 3 * fft_size_inbox),
                        0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          ShellFilterDerivative(FFTMgr, FT_B, comp_i * fft_size_outbox, FT_scratch, 0,
                                dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);
          // bDotGradB_Q[i] += b_j * d(B_Q_i)/dx_j
          // b_j = mag_j / sqrt(rho)
          const std::size_t mag_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "AccumTension", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                const Real b_j = mag_flat(mag_offset + idx) / Kokkos::sqrt(rho_flat(idx));
                bDotGradB_Q(out_offset + idx) += b_j * dW_Q_dx(idx);
              });
        }
      }
    }

    if (compute_UBTb) {
      Kokkos::deep_copy(Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
                            DivbW_Q.data(), 3 * fft_size_inbox),
                        0.0);
      Kokkos::deep_copy(Kokkos::View<Real *, Kokkos::DefaultExecutionSpace::memory_space>(
                            bDotGradW_Q.data(), 3 * fft_size_inbox),
                        0.0);

      for (int comp_i = 0; comp_i < 3; comp_i++) {
        for (int dir_j = 0; dir_j < 3; dir_j++) {
          const std::size_t b_offset = dir_j * fft_size_inbox;
          const std::size_t out_offset = comp_i * fft_size_inbox;
          parthenon::par_for(
              "ComputeBWQ", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                scalar_scratch(idx) = b_flat(b_offset + idx) * W_Q(out_offset + idx);
              });
          FFTMgr->Forward(scalar_scratch.data(), FT_scalar_scratch.data());
          ShellFilterDerivative(FFTMgr, FT_scalar_scratch, 0, FT_scratch, 0, dW_Q_dx, 0,
                                -1.0, Real(Nx + Ny + Nz), dir_j, two_pi_over_L);
          parthenon::par_for(
              "AccumDivbWQ", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                DivbW_Q(out_offset + idx) += dW_Q_dx(idx);
              });

          ShellFilterDerivative(FFTMgr, FT_W, comp_i * fft_size_outbox, FT_scratch, 0,
                                dW_Q_dx, 0, Q_low, Q_high, dir_j, two_pi_over_L);
          parthenon::par_for(
              "AccumbDotGradW", std::size_t(0), fft_size_inbox - 1,
              KOKKOS_LAMBDA(const std::size_t idx) {
                bDotGradW_Q(out_offset + idx) += b_flat(b_offset + idx) * dW_Q_dx(idx);
              });
        }
      }
    }

    if (compute_BUPbb) {
      parthenon::par_for(
          "ComputeBdotBQ", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            Real bdotbq = 0.0;
            for (int n = 0; n < 3; n++) {
              bdotbq +=
                  mag_flat(n * fft_size_inbox + idx) * B_Q(n * fft_size_inbox + idx);
            }
            scalar_scratch(idx) = bdotbq;
          });
      FFTMgr->Forward(scalar_scratch.data(), FT_scalar_scratch.data());
      for (int dir_j = 0; dir_j < 3; dir_j++) {
        ShellFilterDerivative(FFTMgr, FT_scalar_scratch, 0, FT_scratch, 0, gradBdotBQ,
                              dir_j * fft_size_inbox, -1.0, Real(Nx + Ny + Nz), dir_j,
                              two_pi_over_L);
      }
      parthenon::par_for(
          "ScaleGradBdotBQ", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            const Real scale = 0.5 / Kokkos::sqrt(rho_flat(idx));
            for (int n = 0; n < 3; n++) {
              gradBdotBQ(n * fft_size_inbox + idx) *= scale;
            }
          });
    }

    if (compute_UBPbb) {
      parthenon::par_for(
          "ComputeWQOverSqrtRho", std::size_t(0), fft_size_inbox - 1,
          KOKKOS_LAMBDA(const std::size_t idx) {
            const Real scale = 0.5 / Kokkos::sqrt(rho_flat(idx));
            for (int n = 0; n < 3; n++) {
              W_Q_over_sqrt_rho(n * fft_size_inbox + idx) =
                  scale * W_Q(n * fft_size_inbox + idx);
            }
          });
      for (int n = 0; n < 3; n++) {
        FFTMgr->Forward(W_Q_over_sqrt_rho.data() + n * fft_size_inbox,
                        FT_vector_scratch.data() + n * fft_size_outbox);
      }
      SpectralDivergence(FFTMgr, FT_vector_scratch, FT_scratch, DivWQOverSqrtRho,
                         two_pi_over_L);
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
      if (compute_UU || compute_BUT || compute_BUPbb || compute_PU || compute_FU) {
        ShellFilter(FFTMgr, 3, FT_W, FT_scratch, W_K, K_low, K_high);
      }

      // Shell-filter B_K
      if (compute_BB || compute_UBTb || compute_UBPbb) {
        ShellFilter(FFTMgr, 3, FT_B, FT_scratch, B_K, K_low, K_high);
      }

      // --- UU advection: -sum(W_K * UdotGradW_Q) ---
      if (compute_UU) {
        TransferReal local_sum_adv = 0.0;
        Kokkos::parallel_reduce(
            "UUA_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(W_K(idx)) *
                     static_cast<TransferReal>(UdotGradW_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_adv));

        TransferReal global_sum_adv = local_sum_adv;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_adv, &global_sum_adv, 1, MPI_DOUBLE,
                                          MPI_SUM, MPI_COMM_WORLD));
#endif
        UUA_matrix(kk, q) = -global_sum_adv;

        // UU compression: -0.5 * sum(W_K * W_Q * DivU)
        TransferReal local_sum_comp = 0.0;
        Kokkos::parallel_reduce(
            "UUC_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += static_cast<TransferReal>(W_K(idx)) *
                     static_cast<TransferReal>(W_Q(idx)) *
                     static_cast<TransferReal>(DivU(local_idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_comp));

        TransferReal global_sum_comp = local_sum_comp;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_comp, &global_sum_comp, 1,
                                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
        UUC_matrix(kk, q) = -0.5 * global_sum_comp;
      }

      // --- BB advection: -sum(B_K * UdotGradB_Q) ---
      if (compute_BB) {
        TransferReal local_sum_adv = 0.0;
        Kokkos::parallel_reduce(
            "BBA_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(UdotGradB_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_adv));

        TransferReal global_sum_adv = local_sum_adv;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_adv, &global_sum_adv, 1, MPI_DOUBLE,
                                          MPI_SUM, MPI_COMM_WORLD));
#endif
        BBA_matrix(kk, q) = -global_sum_adv;

        // BB compression: -0.5 * sum(B_K * B_Q * DivU)
        TransferReal local_sum_comp = 0.0;
        Kokkos::parallel_reduce(
            "BBC_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(B_Q(idx)) *
                     static_cast<TransferReal>(DivU(local_idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_comp));

        TransferReal global_sum_comp = local_sum_comp;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_comp, &global_sum_comp, 1,
                                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
        BBC_matrix(kk, q) = -0.5 * global_sum_comp;
      }

      // --- BUT: +sum(W_K * bDotGradB_Q) ---
      if (compute_BUT) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "BUT_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(W_K(idx)) *
                     static_cast<TransferReal>(bDotGradB_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
#endif
        BUT_matrix(kk, q) = global_sum;
      }

      if (compute_UBTb) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "UBTb_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(DivbW_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal local_sum_adv = 0.0;
        Kokkos::parallel_reduce(
            "UBTbA_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(bDotGradW_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_adv));

        TransferReal local_sum_comp = 0.0;
        Kokkos::parallel_reduce(
            "UBTbC_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(W_Q(idx)) *
                     static_cast<TransferReal>(Divb(local_idx));
            },
            Kokkos::Sum<TransferReal>(local_sum_comp));

        TransferReal global_sum = local_sum;
        TransferReal global_sum_adv = local_sum_adv;
        TransferReal global_sum_comp = local_sum_comp;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_adv, &global_sum_adv, 1, MPI_DOUBLE,
                                          MPI_SUM, MPI_COMM_WORLD));
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum_comp, &global_sum_comp, 1,
                                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
        UBTbA_matrix(kk, q) = global_sum_adv;
        UBTbC_matrix(kk, q) = global_sum_comp;
        UBTb_matrix(kk, q) = global_sum;
      }

      if (compute_BUPbb) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "BUPbb_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(W_K(idx)) *
                     static_cast<TransferReal>(gradBdotBQ(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
#endif
        BUPbb_matrix(kk, q) = -global_sum;
      }

      if (compute_UBPbb) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "UBPbb_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += static_cast<TransferReal>(B_K(idx)) *
                     static_cast<TransferReal>(mag_flat(idx)) *
                     static_cast<TransferReal>(DivWQOverSqrtRho(local_idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
#endif
        UBPbb_matrix(kk, q) = -global_sum;
      }

      // --- PU: -sum(W_K * (1/sqrt(rho)) * gradP_Q) ---
      if (compute_PU) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "PU_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              sum += static_cast<TransferReal>(W_K(idx)) *
                     static_cast<TransferReal>(gradP_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
#endif
        PU_matrix(kk, q) = -global_sum;
      }

      // --- FU: +sum(W_K * sqrt(rho) * Acc_Q) ---
      if (compute_FU) {
        TransferReal local_sum = 0.0;
        Kokkos::parallel_reduce(
            "FU_reduce", Kokkos::RangePolicy<>(0, 3 * fft_size_inbox),
            KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum) {
              const auto local_idx = idx % fft_size_inbox;
              sum += static_cast<TransferReal>(W_K(idx)) *
                     Kokkos::sqrt(static_cast<TransferReal>(rho_flat(local_idx))) *
                     static_cast<TransferReal>(Acc_Q(idx));
            },
            Kokkos::Sum<TransferReal>(local_sum));

        TransferReal global_sum = local_sum;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                                          MPI_COMM_WORLD));
#endif
        FU_matrix(kk, q) = global_sum;
      }

    } // end K loop
  } // end Q loop

  // Calculate the power spectra
  parthenon::HostArray2D<TransferReal> spectra_h;
  if (compute_spec_U) {
    auto spectra = parthenon::utils::fft::CalcSpectrum(pmesh, vel_flat, 3);
    spectra_h = spectra.GetHostMirrorAndCopy();

    // Sanity checks
    // Power in real space
    Kokkos::Array<TransferReal, 4> sums{{0.0, 0.0, 0.0, 0.0}};
    Kokkos::parallel_reduce(
        "U2_sum", Kokkos::RangePolicy<>(0, fft_size_inbox),
        KOKKOS_LAMBDA(const std::size_t idx, TransferReal &sum_usqr, TransferReal &sum_u1,
                      TransferReal &sum_u2, TransferReal &sum_u3) {
          const auto u1 = static_cast<TransferReal>(vel_flat(idx));
          const auto u2 = static_cast<TransferReal>(vel_flat(idx + fft_size_inbox));
          const auto u3 = static_cast<TransferReal>(vel_flat(idx + 2 * fft_size_inbox));
          sum_u1 += u1;
          sum_u2 += u2;
          sum_u3 += u3;
          sum_usqr += SQR(u1) + SQR(u2) + SQR(u3);
        },
        sums[0], sums[1], sums[2], sums[3]);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
    sums[0] /= Nx * Ny * Nz;
    sums[1] /= Nx * Ny * Nz;
    sums[2] /= Nx * Ny * Nz;
    sums[3] /= Nx * Ny * Nz;

    TransferReal spec_sum = 0.0;
    for (int i = 0; i < spectra_h.extent(0); i++) {
      spec_sum += spectra_h(i, 0);
    }
    if (parthenon::Globals::my_rank == 0) {
      std::cerr << "sum u^2=" << sums[0] << " sum uhat^2=" << spec_sum
                << " <u>^2=" << SQR(sums[1]) + SQR(sums[2]) + SQR(sums[3])
                << " uhat(0)^2=" << spectra_h(0, 0) << " sum u_1=" << sums[1]
                << " sum u_2=" << sums[2] << " sum u_3=" << sums[3] << "\n";
    }
  }

  // --- Write output via openPMD/ADIOS2 ---
  // Python reading example:
  //   import openpmd_api as io
  //   s = io.Series("transfer.%05T.bp", io.Access.read_only)
  //   it = s.iterations[output_number]
  //   shell_edges = it.get_attribute("shell_edges")
  //   UU = it.meshes["UU"][io.Mesh_Record_Component.SCALAR].load_chunk()
  //   s.flush()
  {
    std::string fname = output_file;
    const auto has_iteration_pattern = [](const std::string &name) {
      for (std::size_t pos = name.find('%'); pos != std::string::npos;
           pos = name.find('%', pos + 1)) {
        auto digit_pos = pos + 1;
        while (digit_pos < name.size() && name[digit_pos] >= '0' &&
               name[digit_pos] <= '9') {
          digit_pos++;
        }
        if (digit_pos < name.size() && name[digit_pos] == 'T') {
          return true;
        }
      }
      return false;
    }(fname);
    const auto has_bp_suffix =
        fname.size() >= 3 && fname.substr(fname.size() - 3) == ".bp";
    if (!has_iteration_pattern) {
      if (has_bp_suffix) {
        fname.insert(fname.size() - 3, ".%05T");
      } else {
        fname += ".%05T";
      }
    }
    if (fname.size() < 3 || fname.substr(fname.size() - 3) != ".bp") {
      fname += ".bp";
    }
    openPMD::Series series(fname, openPMD::Access::CREATE,
#ifdef MPI_PARALLEL
                           MPI_COMM_WORLD,
#endif
                           "{}");

    series.setIterationEncoding(openPMD::IterationEncoding::fileBased);

    auto it = series.iterations[static_cast<uint64_t>(output_number)];
    it.open();
    it.setAttribute("shell_edges", shell_edges);
    it.setAttribute("n_shells", n_shells);
    it.setAttribute("binning", binning);

    auto write_matrix = [&](const std::string &name,
                            const parthenon::HostArray2D<TransferReal> &matrix) {
      auto mesh = it.meshes[name];
      auto comp = mesh[openPMD::MeshRecordComponent::SCALAR];
      openPMD::Extent extent = {static_cast<uint64_t>(n_shells),
                                static_cast<uint64_t>(n_shells)};
      comp.resetDataset(
          openPMD::Dataset(openPMD::determineDatatype<TransferReal>(), extent));
      comp.storeChunkRaw(matrix.data(), {0, 0}, extent);
      it.seriesFlush();
    };

    if (compute_UU) {
      write_matrix("UUA", UUA_matrix);
      write_matrix("UUC", UUC_matrix);
      parthenon::HostArray2D<TransferReal> UU_matrix("UU", n_shells, n_shells);
      for (int kk = 0; kk < n_shells; kk++)
        for (int q = 0; q < n_shells; q++)
          UU_matrix(kk, q) = UUA_matrix(kk, q) + UUC_matrix(kk, q);
      write_matrix("UU", UU_matrix);
    }
    if (compute_BB) {
      write_matrix("BBA", BBA_matrix);
      write_matrix("BBC", BBC_matrix);
      parthenon::HostArray2D<TransferReal> BB_matrix("BB", n_shells, n_shells);
      for (int kk = 0; kk < n_shells; kk++)
        for (int q = 0; q < n_shells; q++)
          BB_matrix(kk, q) = BBA_matrix(kk, q) + BBC_matrix(kk, q);
      write_matrix("BB", BB_matrix);
    }
    if (compute_BUT) {
      write_matrix("BUT", BUT_matrix);
    }
    if (compute_UBTb) {
      write_matrix("UBTb", UBTb_matrix);
      write_matrix("UBTbA", UBTbA_matrix);
      write_matrix("UBTbC", UBTbC_matrix);
      parthenon::HostArray2D<TransferReal> UBTbTot_matrix("UBTbTot", n_shells, n_shells);
      for (int kk = 0; kk < n_shells; kk++)
        for (int q = 0; q < n_shells; q++)
          UBTbTot_matrix(kk, q) = UBTbA_matrix(kk, q) + UBTbC_matrix(kk, q);
      write_matrix("UBTbTot", UBTbTot_matrix);
    }
    if (compute_BUPbb) {
      write_matrix("BUPbb", BUPbb_matrix);
    }
    if (compute_UBPbb) {
      write_matrix("UBPbb", UBPbb_matrix);
    }
    if (compute_PU) {
      write_matrix("PU", PU_matrix);
    }
    if (compute_FU) {
      write_matrix("FU", FU_matrix);
    }

    // Write the power spectra
    auto write_vector_from_matrix =
        [&](const std::string &name, const parthenon::HostArray2D<TransferReal> &matrix,
            const int idx) {
          auto mesh = it.meshes[name];
          auto comp = mesh[openPMD::MeshRecordComponent::SCALAR];

          const auto num_bins = matrix.extent(0);
          std::vector<TransferReal> outdata(num_bins);
          for (int i = 0; i < num_bins; i++) {
            outdata.at(i) = matrix(i, idx);
          }

          openPMD::Extent extent = {static_cast<uint64_t>(num_bins)};
          // Result have been reduced to rank 0, so only rank 0 writes
          if (parthenon::Globals::my_rank == 0) {
            comp.resetDataset(
                openPMD::Dataset(openPMD::determineDatatype<TransferReal>(), extent));

            comp.storeChunkRaw(outdata.data(), {0}, extent);
          }

          it.seriesFlush();
        };

    if (compute_spec_U) {
      std::string spec_prefix = "spec/u";
      write_vector_from_matrix(spec_prefix + "/en_sum", spectra_h, 0);
      write_vector_from_matrix(spec_prefix + "/k_sum", spectra_h, 1);
      write_vector_from_matrix(spec_prefix + "/count_sum", spectra_h, 2);
    }

    series.close();
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Wrote " << fname << " with iteration " << output_number << std::endl;
    }
  }

  Driver::PostExecute(DriverStatus::complete);
  return DriverStatus::complete;
}
