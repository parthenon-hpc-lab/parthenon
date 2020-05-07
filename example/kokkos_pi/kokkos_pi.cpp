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
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"

using Real = double;

using parthenon::CellVariable;
using parthenon::CellVariableVector;
using parthenon::Container;
using parthenon::ContainerIterator;
using parthenon::DevExecSpace;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::MetadataFlag;
using parthenon::par_for;
using parthenon::ParArray4D;
using parthenon::ParArrayND;
using parthenon::Real;

using View1D = Kokkos::View<Real *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using View2D = Kokkos::View<Real **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using View3D = Kokkos::View<Real ***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using View4D =
    Kokkos::View<Real ****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using ViewOfView3D = Kokkos::View<View3D *>;
using ViewMeshBlock1D =
    Kokkos::View<MeshBlock *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
using ViewMesh =
    Kokkos::View<ViewMeshBlock1D, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

// simple giga-ops calculator
double calcGops(const int &nops, const double &t, const int &n_block3, const int &n_mesh3,
                const int &n_iter) {
  return (static_cast<Real>(nops * n_iter) / t / 1.0e9 * static_cast<Real>(n_block3) *
          static_cast<Real>(n_mesh3));
};

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

void usage(std::string program) {
  std::cout << std::endl
            << "    Usage: " << program << " n_block n_mesh n_iter" << std::endl
            << std::endl
            << "             n_block = size of each mesh block on each axis" << std::endl
            << "              n_mesh = number mesh blocks alogn each axis" << std::endl
            << "              n_iter = number of iterations to time" << std::endl
            << "            [Radius] = Optional: Radius of sphere" << std::endl
            << "                                 Defaults to 1.0" << std::endl
            << std::endl;
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
    if (argc == 5) {
      radius = static_cast<Real>(std::stod(argv[4], &pos));
    }

    // Setup auxilliary variables
    const int n_block2 = n_block * n_block;
    const int n_block3 = n_block * n_block * n_block;
    const int n_mesh3 = n_mesh * n_mesh * n_mesh;
    const double radius2 = radius * radius;
    const double radius3 = radius * radius * radius;
    const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
    const Real dVol = radius3 / static_cast<Real>(n_mesh3 * n_block3);

    // allocate space for origin coordinates and create host mirror view
    View2D xyz("xyzBlocks", 3, n_mesh3);
    auto h_xyz = Kokkos::create_mirror_view(xyz);

    // *** Kludge warning ***
    // Since our mesh is not GPU friendly we set up a hacked up
    // collection of mesh blocks.  The hope is that when our mesh is
    // up to par we will replace this code with the mesh
    // infrastructure.

    // Set up our mesh.
    Metadata myMetadata({Metadata::Independent, Metadata::Cell});

    MeshBlock *lastBlock = nullptr;
    MeshBlock *allBlocks[n_mesh3]; // probably not needed, but creating it so that we can
                                   // launch many blocks

    std::cout << "Begin setup " << std::endl;
    int idxMesh = 0; // an index into AllBlocks array
    for (int k_mesh = 0; k_mesh < n_mesh; k_mesh++) {
      for (int j_mesh = 0; j_mesh < n_mesh; j_mesh++) {
        for (int i_mesh = 0; i_mesh < n_mesh; i_mesh++, idxMesh++) {
          // get a new meshblock
          auto *pmb = new MeshBlock(n_block, 3);
          // if not first block then add to chain
          if (lastBlock) {
            // set pointers accordingly
            lastBlock->next = pmb;
            pmb->prev = lastBlock;
          }

          // store block away
          lastBlock = allBlocks[idxMesh] = pmb;
          // set coordinates of first cell center
          h_xyz(0, idxMesh) = dxyzCell * (static_cast<Real>(i_mesh * n_block) + 0.5);
          h_xyz(1, idxMesh) = dxyzCell * (static_cast<Real>(j_mesh * n_block) + 0.5);
          h_xyz(2, idxMesh) = dxyzCell * (static_cast<Real>(k_mesh * n_block) + 0.5);
          // Add our variable for in_or_out
          Container<Real> &base = pmb->real_containers.Get();
          base.setBlock(pmb);
          base.Add("in_or_out", myMetadata);
        }
      }
    }
    // copy our coordinates over to Device and wait for completion
    Kokkos::deep_copy(xyz, h_xyz);
    Kokkos::fence();

    // first let us do a stupid loop over the mesh
    // This policy is over entire mesh

    // This policy is over entire mesh
    auto policy = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0,
                                        n_block3 * n_mesh3, Kokkos::ChunkSize(512));

    // This policy is over one block
    auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                             Kokkos::ChunkSize(512));

    std::cout << "Begin basic timing " << std::endl;
    double time_basic;
    {
      MeshBlock *pStart = allBlocks[0];
      time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
        MeshBlock *pmb = pStart;
        for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb = pmb->next) {
          Container<Real> &base = pmb->real_containers.Get();
          auto inOrOut = base.PackVariables({Metadata::Independent});
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
                inOrOut(0, k_grid + NGHOST, j_grid + NGHOST, i_grid + NGHOST) =
                    (myR2 < radius2 ? 1.0 : 0.0); // iops = 3
              });
        }
      });
      // wait for work to finish, redundant due to timing loop
      Kokkos::fence();
    }

    std::cout << "Begin reduce sum" << std::endl;
    double myPi = 0.0;
    {
      // reduce the sum on the device
      // I'm pretty sure I can do this better, but not worried about performance for this
      MeshBlock *pmb = allBlocks[0];
      for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb = pmb->next) {
        Container<Real> &base = pmb->real_containers.Get();
        auto inOrOut = base.PackVariables({Metadata::Independent});
        double onePi;
        Kokkos::parallel_reduce(
            "Reduce Sum", policyBlock,
            KOKKOS_LAMBDA(const int &idx, double &mySum) {
              const int k_grid = idx / n_block2;
              const int j_grid = (idx - k_grid * n_block2) / n_block;
              const int i_grid = idx - k_grid * n_block2 - j_grid * n_block;
              mySum += inOrOut(0, k_grid + NGHOST, j_grid + NGHOST, i_grid + NGHOST);
            },
            onePi);
        Kokkos::fence();
        myPi += onePi;
      }

      // wait for work to finish
      Kokkos::fence();

      // calculate Pi
      myPi = 6.0 * myPi * dVol / radius3;
    }

    constexpr int niops = 8;
    constexpr int nfops = 11;

    printf("Calculated value of pi=%.16lf in %.6lf seconds; GFlops="
           "%.16lf GIops=%.16lf\n",
           myPi, time_basic, calcGops(nfops, time_basic, n_block3, n_mesh3, n_iter),
           calcGops(niops, time_basic, n_block3, n_mesh3, n_iter));

  } while (0);
  Kokkos::finalize();
}
