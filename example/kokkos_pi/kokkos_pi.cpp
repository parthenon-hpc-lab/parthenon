//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#include "Kokkos_Core.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/container.hpp"
#include "interface/container_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using Real = double;
using View1D = Kokkos::View<Real *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using View3D = Kokkos::View<Real ***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using View4D =
    Kokkos::View<Real ****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using ViewOfView3D = Kokkos::View<View3D *>;

using parthenon::CellVariable;
using parthenon::CellVariableVector;
using parthenon::Container;
using parthenon::ContainerIterator;
using parthenon::DevExecSpace;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::Metadata;
using parthenon::MetadataFlag;
using parthenon::PackVariables;
using parthenon::par_for;
using parthenon::ParArray4D;
using parthenon::ParArrayND;
using parthenon::Real;

// Test wrapper to run a function multiple times
template <typename PerfFunc>
double kernel_timer_wrapper(const int n_burn, const int n_perf, PerfFunc perf_func) {

  // Initialize the timer and test
  Kokkos::Timer timer;

  for (int i_run = 0; i_run < n_burn + n_perf; i_run++) {

    if (i_run == n_burn) {
      // Burn in time is over, start timing
      Kokkos::fence();
      timer.reset();
    }

    // Run the function timing performance
    perf_func();
  }

  // Time it
  Kokkos::fence();
  double perf_time = timer.seconds();

  return perf_time;
}

// Test wrapper for timing a container on the square array test
double container_test_wrapper(const int n_burn, const int n_perf,
                              Container<Real> &container_in,
                              Container<Real> &container_out) {
  // Setup a variable pack
  auto var_view_in = PackVariables<Real>(container_in, {Metadata::Independent});
  auto var_view_out = PackVariables<Real>(container_out, {Metadata::Independent});

  // Test performance of view of views VariablePack implementation
  return kernel_timer_wrapper(n_burn, n_perf, [&]() {
    par_for(
        "Flat Container Array Perf", DevExecSpace(), 0, var_view_in.GetDim(4) - 1, 0,
        var_view_in.GetDim(3) - 1, 0, var_view_in.GetDim(2) - 1, 0,
        var_view_in.GetDim(1) - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          var_view_out(l, k, j, i) = 2. * var_view_in(l, k, j, i);
        });
  });
}

// Test wrapper for timing a container on the square array test
double container_always_pack_test_wrapper(const int n_burn, const int n_perf,
                                          Container<Real> &container_in,
                                          Container<Real> &container_out) {

  // Test performance of view of views VariablePack implementation
  return kernel_timer_wrapper(n_burn, n_perf, [&]() {
    // Setup a variable pack
    auto var_view_in = PackVariables<Real>(container_in, {Metadata::Independent});
    auto var_view_out = PackVariables<Real>(container_out, {Metadata::Independent});

    par_for(
        "Flat Container Array Perf", DevExecSpace(), 0, var_view_in.GetDim(4) - 1, 0,
        var_view_in.GetDim(3) - 1, 0, var_view_in.GetDim(2) - 1, 0,
        var_view_in.GetDim(1) - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          var_view_out(l, k, j, i) = 2. * var_view_in(l, k, j, i);
        });
  });
}

void usage(std::string program) {
  std::cout << std::endl
            << "    Usage: " << program << " n_vars n_vector n_side n_run" << std::endl
	    << std::endl
            << "              n_vars = total number of columns" << std::endl
            << "              n_vector = number of columns in each vector." << std::endl
            << "                         Note that n_vars%n_vector must be 0" << std::endl
            << "              n_side = number of cells on each side of block" << std::endl
            << "              n_run = number of iterations to time" << std::endl
            << std::endl;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::size_t pos;

    // ensure we have correct number of arguments
    if (argc != 5) {
      usage(argv[0]);
      exit(1);
    }

    const int n_var = std::stoi(argv[1], &pos);
    const int n_vector = std::stoi(argv[2], &pos);
    const int n_side = std::stoi(argv[3], &pos);
    const int n_run = std::stoi(argv[4], &pos);

    if (n_var % n_vector != 0) {
      std::cerr << "Error! n_var \% n_vector =" << n_var % n_vector << " is not zero"
                << std::endl;
      exit(0);
    }

    // Order of iteration, fastest moving to slowest moving:
    // x (full n_side), y (full n_side), z (only n_buf), var (n_var)
    const int n_side2 = n_side * n_side;
    const int n_side3 = n_side * n_side * n_side;
    const int n_grid = n_side3;

    //////////////////////////////////////////////////
    // Do some plain Kokkos tests
    //////////////////////////////////////////////////

    auto policy = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0,
                                        n_var * n_grid, Kokkos::ChunkSize(512));

    // Setup a raw 4D view
    View4D view4d_in("view4d_in", n_var, n_side, n_side, n_side);
    View4D view4d_out("view4d_out", n_var, n_side, n_side, n_side);

    double time_view4d = kernel_timer_wrapper(n_run, n_run, [&]() {
      Kokkos::parallel_for(
          "View4D Loop", policy, KOKKOS_LAMBDA(const int &idx) {
            const int v_var = idx / n_side3;
            const int k_grid = (idx - v_var * n_side3) / n_side2;
            const int j_grid = (idx - v_var * n_side3 - k_grid * n_side2) / n_side;
            const int i_grid = idx - v_var * n_side3 - k_grid * n_side2 - j_grid * n_side;

            view4d_out(v_var, k_grid, j_grid, i_grid) =
                2. * view4d_in(v_var, k_grid, j_grid, i_grid);
          });
    });

    // Setup a view of views
    ViewOfView3D view_of_view3d_in("view_of_view3d_in", n_var);
    ViewOfView3D view_of_view3d_out("view_of_view3d_out", n_var);

    auto h_view_of_view3d_in = Kokkos::create_mirror_view(view_of_view3d_in);
    auto h_view_of_view3d_out = Kokkos::create_mirror_view(view_of_view3d_out);

    for (int i = 0; i < n_var; i++) {
      h_view_of_view3d_in[i] = View3D("view3d_in", n_side, n_side, n_side);
      h_view_of_view3d_out[i] = View3D("view3d_out", n_side, n_side, n_side);
    }
    Kokkos::deep_copy(view_of_view3d_in, h_view_of_view3d_in);
    Kokkos::deep_copy(view_of_view3d_out, h_view_of_view3d_out);

    //////////////////////////////////////////////////
    // Do some plain Kokkos tests
    //////////////////////////////////////////////////

    double time_view_of_view3d = kernel_timer_wrapper(n_run, n_run, [&]() {
      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(const int &idx) {
            const int v_var = idx / n_side3;
            const int k_grid = (idx - v_var * n_side3) / n_side2;
            const int j_grid = (idx - v_var * n_side3 - k_grid * n_side2) / n_side;
            const int i_grid = idx - v_var * n_side3 - k_grid * n_side2 - j_grid * n_side;

            // Get the 3D views
            auto view3d_in = view_of_view3d_in(v_var);
            auto view3d_out = view_of_view3d_out(v_var);

            view3d_out(k_grid, j_grid, i_grid) = 2. * view3d_in(k_grid, j_grid, i_grid);
          });
    });

    //////////////////////////////////////////////////
    // Test Parthenon infrastructure
    //////////////////////////////////////////////////

    Metadata m_in({Metadata::Independent});
    Metadata m_out({Metadata::Independent});
    std::vector<int> scalar_block_size{n_side, n_side, n_side};
    std::vector<int> vector_block_size{n_side, n_side, n_side, n_vector};
    std::vector<int> n_var_block_size{n_side, n_side, n_side, n_var};

    // Time a container with n_var scalars
    Container<Real> container_scalars_in;
    Container<Real> container_scalars_out;
    for (int i = 0; i < n_var; i++) {
      container_scalars_in.Add(std::string("s_in") + std::to_string(i), m_in,
                               scalar_block_size);
      container_scalars_out.Add(std::string("s_out") + std::to_string(i), m_out,
                                scalar_block_size);
    }
    double time_scalars =
        container_test_wrapper(n_run, n_run, container_scalars_in, container_scalars_out);

    // Time a container with n_vector-vectors to have total n_var fields
    Container<Real> container_vectors_in;
    Container<Real> container_vectors_out;
    for (int i = 0; i < n_var; i += n_vector) {
      container_vectors_in.Add(std::string("v_in") + std::to_string(i), m_in,
                               vector_block_size);
      container_vectors_out.Add(std::string("v_out") + std::to_string(i), m_out,
                                vector_block_size);
    }
    double time_vectors =
        container_test_wrapper(n_run, n_run, container_vectors_in, container_vectors_out);

    // Time a container with a mix of vectors and scalars, every other one, scalars to
    // backfill
    Container<Real> container_mix_in;
    Container<Real> container_mix_out;
    {
      int i = 0;
      bool add_scalar = true;
      while (i < n_var) {
        if (add_scalar) {
          container_mix_in.Add(std::string("s_in") + std::to_string(i), m_in,
                               scalar_block_size);
          container_mix_out.Add(std::string("s_out") + std::to_string(i), m_out,
                                scalar_block_size);
          i++;
          if (i + n_vector <= n_var) { // Add a vector if it will fit
            add_scalar = false;
          }
        } else {
          container_mix_in.Add(std::string("v_in") + std::to_string(i), m_in,
                               vector_block_size);
          container_mix_out.Add(std::string("v_out") + std::to_string(i), m_out,
                                vector_block_size);
          i += n_vector;
        }
      }
    }
    double time_mix =
        container_test_wrapper(n_run, n_run, container_mix_in, container_mix_out);

    // Time a container with one big vector
    Container<Real> container_one_vector_in;
    Container<Real> container_one_vector_out;
    container_one_vector_in.Add("v_in", m_in, n_var_block_size);
    container_one_vector_out.Add("v_out", m_out, n_var_block_size);
    double time_one_vector = container_test_wrapper(n_run, n_run, container_one_vector_in,
                                                    container_one_vector_out);

    // Time a container with n_var scalars, always packing
    Container<Real> container_always_pack_in;
    Container<Real> container_always_pack_out;
    for (int i = 0; i < n_var; i++) {
      container_always_pack_in.Add("s_in", m_in, scalar_block_size);
      container_always_pack_out.Add("s_out", m_out, scalar_block_size);
    }
    double time_always_pack = container_always_pack_test_wrapper(
        n_run, n_run, container_always_pack_in, container_always_pack_out);

    double timings[] = {time_view4d, time_view_of_view3d, time_scalars,    time_vectors,
                        time_mix,    time_one_vector,     time_always_pack};

    std::cout << n_var << " " << n_side << " " << n_run << " " << n_vector;
    for (double timing : timings) {
      std::cout << " " << timing << " "
                << static_cast<double>(n_grid) * static_cast<double>(n_run) / timing;
    }
    std::cout << std::endl;
  }
  Kokkos::finalize();
}
