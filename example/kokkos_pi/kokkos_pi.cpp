//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

// This is a simple example that uses par_for() to compute whether cell
// centers sit within a unit sphere or not.  Adding up all the
// cells that lie within a unit sphere gives us a way to compute pi.
//
// Note: The goal is to use different methods of iterating through
// mesh blocks to see what works best for different architectures.
// While this code could be sped up by checking the innermost and
// outermost points of a mesh block, that would defeat the purpose of
// this program, so please do not make that change.
//
// Since the mesh infrastructure is not yet usable on GPUs, we create
// mesh blocks and chain them manually.  The cell coordinates are
// computed based on the origin of the mesh block and given cell
// sizes.  Once we have a canonical method of using a mesh on the GPU,
// this code will be changed to reflect that.
//
// Usage: examples/kokkos_pi/kokkos-pi N_Block N_Mesh N_iter
//          N_Block = size of each mesh block on each edge
//           N_Mesh = Number of mesh blocks along each axis
//           N_Iter = Number of timing iterations to run
//         [Radius] = Optional: Radius of sphere (size of cube).
//                    Defaults to 1.0
//
// The unit sphere is actually a unit octant that sits within a unit
// square which runs from (0,0,0) to (1,1,1).  Hence, in the perfect
// case, the sum of the interior would be pi/6. Our unit cube has
// N_Mesh*N_Block cells that span from [0,1] which gives us a
// dimension of 1.0/(N_Mesh*N_Block) for each side of the cell and the
// rest can be computed accordingly.  The coordinates of each cell
// within the block can be computed as:
//      (x0+dx*i_grid,y0+dy*j_grid,z0+dx*k_grid).
//
// We plan to explore using a flat range and a MDRange within par_for
// and using flat range and MDRange in Kokkos
//

#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Kokkos_Core.hpp"

// Get most commonly used parthenon package includes
#include "kokkos_abstraction.hpp"
#include "parthenon/driver.hpp"
#include "parthenon/package.hpp"
using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

using View2D = Kokkos::View<Real **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

const int nghost = 2;

// The result struct contains results of different tests
typedef struct result_t {
  std::string name; // The name of this test
  Real64 pi;          // the value of pi calculated
  Real64 t;           // time taken to run test
  int iops;         // number of integer ops
  int fops;         // number of floating point ops
} result_t;

// simple giga-ops calculator
static double calcGops(const int &nops, const double &t, const int &n_block3,
                       const int &n_mesh3, const int &n_iter) {
  return (static_cast<Real>(nops * n_iter) / t / 1.0e9 * static_cast<Real>(n_block3) *
          static_cast<Real>(n_mesh3));
}

// Test wrapper to run a function multiple times
template <typename PerfFunc>
static double kernel_timer_wrapper(const int n_burn, const int n_perf,
                                   PerfFunc perf_func) {
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

static void usage(std::string program) {
  std::cout << std::endl
            << "    Usage: " << program << " n_block n_mesh n_iter" << std::endl
            << std::endl
            << "             n_block = size of each mesh block on each axis" << std::endl
            << "              n_mesh = number mesh blocks along each axis" << std::endl
            << "              n_iter = number of iterations to time" << std::endl
            << "            [Radius] = Optional: Radius of sphere" << std::endl
            << "                                 Defaults to 1.0" << std::endl
            << std::endl;
}

static Real64 sumArray(BlockList_t &blocks, const int &n_block) {
  // This policy is over one block
  const int n_block2 = n_block * n_block;
  const int n_block3 = n_block * n_block * n_block;
  auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                           Kokkos::ChunkSize(512));
  Real64 theSum = 0.0;
  // reduce the sum on the device
  // I'm pretty sure I can do this better, but not worried about performance for this
  for (auto &pmb : blocks) {
    auto &base = pmb->meshblock_data.Get();
    auto inOrOut = base->PackVariables({Metadata::Independent});
    Real64 oneSum;
    Kokkos::parallel_reduce(
        "Reduce Sum", policyBlock,
        KOKKOS_LAMBDA(const int &idx, Real64 &mySum) {
          const int k_grid = idx / n_block2;
          const int j_grid = (idx - k_grid * n_block2) / n_block;
          const int i_grid = idx - k_grid * n_block2 - j_grid * n_block;
          mySum += inOrOut(0, k_grid + nghost, j_grid + nghost, i_grid + nghost);
        },
        oneSum);
    Kokkos::fence();
    theSum += oneSum;
  }
  // calculate Pi
  return theSum;
}

static BlockList_t setupMesh(const int &n_block, const int &n_mesh, const double &radius,
                             View2D &xyz, const int NG = 0) {
  // *** Kludge warning ***
  // Since our mesh is not GPU friendly we set up a hacked up
  // collection of mesh blocks.  The hope is that when our mesh is
  // up to par we will replace this code with the mesh
  // infrastructure.

  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  auto h_xyz = Kokkos::create_mirror_view(xyz);

  // Set up state descriptor.
  Metadata myMetadata({Metadata::Independent, Metadata::WithFluxes, Metadata::Cell});
  auto pgk = std::make_shared<StateDescriptor>("Pi");
  pgk->AddField("in_or_out", myMetadata);

  // Set up our mesh.
  BlockList_t block_list;
  block_list.reserve(n_mesh * n_mesh * n_mesh);

  // compute an offset due to ghost cells
  double delta = dxyzCell * static_cast<Real>(NG);

  int idx = 0; // an index into Block coordinate array
  for (int k_mesh = 0; k_mesh < n_mesh; k_mesh++) {
    for (int j_mesh = 0; j_mesh < n_mesh; j_mesh++) {
      for (int i_mesh = 0; i_mesh < n_mesh; i_mesh++, idx++) {
        // get a new meshblock and insert into chain
        block_list.push_back(std::make_shared<MeshBlock>(n_block, 3));
        auto &pmb = block_list.back();
        // set coordinates of first cell center
        h_xyz(0, idx) = dxyzCell * (static_cast<Real>(i_mesh * n_block) + 0.5) - delta;
        h_xyz(1, idx) = dxyzCell * (static_cast<Real>(j_mesh * n_block) + 0.5) - delta;
        h_xyz(2, idx) = dxyzCell * (static_cast<Real>(k_mesh * n_block) + 0.5) - delta;
        // Add variable for in_or_out
        pmb->meshblock_data.Get()->Initialize(pgk, pmb);
      }
    }
  }
  // copy our coordinates over to Device and wait for completion
  Kokkos::deep_copy(xyz, h_xyz);
  Kokkos::fence();

  return block_list;
}

result_t naiveKokkos(int n_block, int n_mesh, int n_iter, double radius) {
  // creates a mesh and rusn a basic Kokkos implementation for looping through blocks.

  // Setup auxilliary variables
  const int n_block2 = n_block * n_block;
  const int n_block3 = n_block * n_block * n_block;
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  const double radius2 = radius * radius;
  const double radius3 = radius * radius * radius;
  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  const Real dVol = radius3 / static_cast<Real>(n_mesh3) / static_cast<Real>(n_block3);

  // allocate space for origin coordinates and set up the mesh
  View2D xyz("xyzBlocks", 3, n_mesh3);
  auto blocks = setupMesh(n_block, n_mesh, radius, xyz);

  // first A  naive Kokkos loop over the mesh
  // This policy is over one block
  auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                           Kokkos::ChunkSize(512));

  double time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
    auto pmb = blocks.begin();
    for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb++) {
      auto &base = (*pmb)->meshblock_data.Get();
      auto inOrOut = base->PackVariables({Metadata::Independent});
      // iops = 8  fops = 11
      Kokkos::parallel_for(
          "Compute In Or Out", policyBlock, KOKKOS_LAMBDA(const int &idx) {
            const int k_grid = idx / n_block2;                             // iops = 1
            const int j_grid = (idx - k_grid * n_block2) / n_block;        // iops = 3
            const int i_grid = idx - k_grid * n_block2 - j_grid * n_block; // iops = 4
            const Real x =
                xyz(0, iMesh) + dxyzCell * static_cast<Real>(i_grid); // fops = 2
            const Real y =
                xyz(1, iMesh) + dxyzCell * static_cast<Real>(j_grid); // fops = 2
            const Real z =
                xyz(2, iMesh) + dxyzCell * static_cast<Real>(k_grid); // fops = 2
            const Real myR2 = x * x + y * y + z * z;                  // fops = 5
            inOrOut(0, k_grid + nghost, j_grid + nghost,
                    i_grid + nghost) = (myR2 < radius2 ? 1.0 : 0.0); // iops = 3
          });
    }
  });
  Kokkos::fence();

  // formulate result struct
  constexpr int niops = 8;
  constexpr int nfops = 11;
  auto r = result_t{"Naive_Kokkos", (6.0 * sumArray(blocks, n_block) * dVol / radius3),
                    time_basic, niops, nfops};

  return r;
}

result_t naiveParFor(int n_block, int n_mesh, int n_iter, double radius) {
  // creates a mesh and rusn a basic par_for implementation for looping through blocks.

  // Setup auxilliary variables
  const int n_block3 = n_block * n_block * n_block;
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  const double radius2 = radius * radius;
  const double radius3 = radius * radius * radius;
  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  const Real dVol = radius3 / static_cast<Real>(n_mesh3) / static_cast<Real>(n_block3);

  // allocate space for origin coordinates and set up the mesh
  View2D xyz("xyzBlocks", 3, n_mesh3);
  auto blocks = setupMesh(n_block, n_mesh, radius, xyz, nghost);

  double time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
    auto pmb = blocks.begin();
    for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb++) {
      auto &base = (*pmb)->meshblock_data.Get();
      auto inOrOut = base->PackVariables({Metadata::Independent});
      // iops = 0  fops = 11
      par_for(
          DEFAULT_LOOP_PATTERN, "par_for in or out", DevExecSpace(), 0,
          inOrOut.GetDim(4) - 1, nghost, inOrOut.GetDim(3) - nghost - 1, nghost,
          inOrOut.GetDim(2) - nghost - 1, nghost, inOrOut.GetDim(1) - nghost - 1,
          KOKKOS_LAMBDA(const int l, const int k_grid, const int j_grid,
                        const int i_grid) {
            const Real x =
                xyz(0, iMesh) + dxyzCell * static_cast<Real>(i_grid); // fops = 2
            const Real y =
                xyz(1, iMesh) + dxyzCell * static_cast<Real>(j_grid); // fops = 2
            const Real z =
                xyz(2, iMesh) + dxyzCell * static_cast<Real>(k_grid); // fops = 2
            const Real myR2 = x * x + y * y + z * z;                  // fops = 5
            inOrOut(l, k_grid, j_grid, i_grid) = (myR2 < radius2 ? 1.0 : 0.0);
          });
    }
  });
  Kokkos::fence();

  // formulate result struct
  constexpr int niops = 0;
  constexpr int nfops = 11;
  auto r = result_t{"Naive_ParFor", (6.0 * sumArray(blocks, n_block) * dVol / radius3),
                    time_basic, niops, nfops};

  return r;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  do {
    // ensure we have correct number of arguments
    if (!(argc == 4 || argc == 5)) {
      std::cout << "argc=" << argc << std::endl;
      usage(argv[0]);
      break;
    }

    std::size_t pos;
    Real radius = 1.0;

    // Read command line input
    const int n_block = std::stoi(argv[1], &pos);
    const int n_mesh = std::stoi(argv[2], &pos);
    const int n_iter = std::stoi(argv[3], &pos);
    if (argc >= 5) {
      radius = static_cast<Real>(std::stod(argv[4], &pos));
    } else {
      radius = 1.0;
    }

    // Run tests
    // A result vector
    std::vector<struct result_t> results;

    // Run Naive Kokkos Implementation
    results.push_back(naiveKokkos(n_block, n_mesh, n_iter, radius));
    results.push_back(naiveParFor(n_block, n_mesh, n_iter, radius));

    // print all results
    const int64_t n_block3 = n_block * n_block * n_block;
    const int64_t n_mesh3 = n_mesh * n_mesh * n_mesh;

    printf("\nname,t(s),cps,GFlops,pi\n");
    int64_t iterBlockMesh = static_cast<int64_t>(n_iter) * static_cast<int64_t>(n_mesh3) *
                            static_cast<int64_t>(n_block3);
    for (auto &test : results) {
      double cps = static_cast<Real>(iterBlockMesh) / test.t;
      printf("%s,%.8lf,%10g,%.4lf,%.14lf\n", test.name.c_str(), test.t, cps,
             calcGops(test.fops, test.t, n_block3, n_mesh3, n_iter), test.pi);
    }
  } while (0);
  Kokkos::finalize();
}
