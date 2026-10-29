//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

#include "lukewarm/lukewarm_relocate.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "interface/mesh_data.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_comms.hpp"
#include "lukewarm/lukewarm_container.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "lukewarm/lukewarm_pack.hpp"
#include "mesh/mesh.hpp"
#include "pack/default_names.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

namespace {
// The MeshData that own lukewarms, one per default block partition
std::vector<MeshData<Real> *> LukewarmMeshData(Mesh *pmesh) {
  std::vector<MeshData<Real> *> mds;
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    mds.push_back(pmesh->mesh_data.Add("base", partition).get());
  }
  return mds;
}
} // namespace

int LocateLukewarm(MeshData<Real> *md, const std::string &name) {
  PARTHENON_INSTRUMENT
  auto &container = md->GetLukewarmContainer();
  auto &lukewarm = container.GetLukewarm(name);
  auto &comms = container.GetComms(name);
  PARTHENON_REQUIRE_THROWS(comms.nleaving == 0, "Lukewarm " + name +
                                                    " still has particles waiting for "
                                                    "SendLukewarm()");
  // A new round removes particles, which may move the ones received in the last round
  comms.received = IndexRange{0, -1};
  const int n = lukewarm.NumParticles();
  if (n == 0) return 0;

  // Locate particles exactly as a kernel using the pack would
  const auto &geometry = container.GetGeometry();
  const LukewarmPack<lukewarm::x> pack(lukewarm, geometry);
  auto marks = lukewarm.GetRemovalMarks();
  auto integer = lukewarm.GetData<int>();
  auto counts = comms.counts;
  Kokkos::deep_copy(counts, 0);
  // The neighbor each particle is headed to, if it leaves this MeshData
  comms.Reserve(n);
  auto destinations = comms.destinations;
  int nlost = 0;
  par_reduce(
      PARTHENON_AUTO_LABEL, 0, n - 1,
      KOKKOS_LAMBDA(const int i, int &lost) {
        destinations(i) = -1;
        if (marks(i) != 0) return;
        const auto loc = pack.Locate(i);
        if (loc.InOtherMeshData()) {
          const int nb = geometry.NeighborIndex(loc.Gid());
          if (nb < 0) {
            lost += 1;
            return;
          }
          // Until it is delivered, a particle's block is the gid of the block it is
          // headed to
          integer(Lukewarm::block_row, i) = loc.Gid();
          destinations(i) = nb;
          Kokkos::atomic_add(&counts(nb), 1);
        } else if (loc.LeftMesh()) {
          marks(i) = 1;
        } else if (loc.Lost()) {
          lost += 1;
        }
      },
      Kokkos::Sum<int>(nlost));
  PARTHENON_REQUIRE_THROWS(nlost == 0, std::to_string(nlost) + " particles of lukewarm " +
                                           name + " could not be located");

  Kokkos::deep_copy(comms.counts_h, counts);
  int nleaving = 0;
  for (int nb = 0; nb < comms.counts_h.extent_int(0); ++nb) {
    nleaving += comms.counts_h(nb);
  }
  if (nleaving == 0) {
    lukewarm.RemoveMarked();
    return 0;
  }

  // Leaving particles are removed once they are sent, along with the marked ones
  auto leaving = comms.leaving;
  Kokkos::parallel_scan(
      PARTHENON_AUTO_LABEL, Kokkos::RangePolicy<DevExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, int &update, const bool final) {
        if (destinations(i) < 0) return;
        if (final) leaving(update) = i;
        update += 1;
      });
  comms.nleaving = nleaving;
  return nleaving;
}

TaskStatus SendLukewarm(MeshData<Real> *md, const std::string &name) {
  auto &container = md->GetLukewarmContainer();
  return SendLukewarmParticles(container.GetLukewarm(name), container.GetComms(name))
             ? TaskStatus::complete
             : TaskStatus::incomplete;
}

TaskStatus StartReceiveLukewarm(MeshData<Real> *md, const std::string &name) {
  StartReceiveLukewarmParticles(md->GetLukewarmContainer().GetComms(name));
  return TaskStatus::complete;
}

TaskStatus ReceiveLukewarm(MeshData<Real> *md, const std::string &name) {
  auto &container = md->GetLukewarmContainer();
  return ReceiveLukewarmParticles(container.GetLukewarm(name), container.GetComms(name),
                                  container.GetGeometry())
             ? TaskStatus::complete
             : TaskStatus::incomplete;
}

IndexRange GetLastReceived(MeshData<Real> *md, const std::string &name) {
  auto &container = md->GetLukewarmContainer();
  const auto &received = container.GetComms(name).received;
  PARTHENON_REQUIRE_THROWS(received.e < container.GetLukewarm(name).NumParticles(),
                           "Particles of lukewarm " + name +
                               " were removed since they were received");
  return received;
}

void RelocateLukewarm(Mesh *pmesh, const std::string &name) {
  PARTHENON_INSTRUMENT
  const auto mds = LukewarmMeshData(pmesh);
  // Each round moves a particle at least one MeshData further along its way, and as in
  // Locate(), it visits each block of the mesh at most a few times
  const int max_rounds = 4 * pmesh->nbtotal + 16;
  for (int round = 0;; ++round) {
    PARTHENON_REQUIRE_THROWS(round <= max_rounds,
                             "Particles of lukewarm " + name + " never settled");
    int nleaving = 0;
    for (auto *md : mds) {
      nleaving += LocateLukewarm(md, name);
    }
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, &nleaving, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
#endif
    if (nleaving == 0) break;

    for (auto *md : mds) {
      StartReceiveLukewarm(md, name);
    }
    for (auto *md : mds) {
      while (SendLukewarm(md, name) == TaskStatus::incomplete) {
      }
    }
    std::vector<bool> received(mds.size(), false);
    for (int nreceived = 0; nreceived < mds.size();) {
      for (int p = 0; p < mds.size(); ++p) {
        if (received[p]) continue;
        received[p] = ReceiveLukewarm(mds[p], name) == TaskStatus::complete;
        nreceived += received[p];
      }
    }
  }
}

TaskID AddLukewarmExchangeTasks(TaskID dependency, TaskList &tl, MeshData<Real> *md,
                                const std::string &name) {
  auto start_recv = tl.AddTask(TaskID(0), TF(StartReceiveLukewarm), md, name);
  auto locate = tl.AddTask(dependency, "LocateLukewarm", [md, name]() {
    LocateLukewarm(md, name);
    return TaskStatus::complete;
  });
  auto send = tl.AddTask(locate, TF(SendLukewarm), md, name);
  return tl.AddTask(start_recv | send, TF(ReceiveLukewarm), md, name);
}

} // namespace parthenon
