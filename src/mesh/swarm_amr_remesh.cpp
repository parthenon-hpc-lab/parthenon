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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "defs.hpp"
#include "globals.hpp"
#include "interface/swarm.hpp"
#include "mesh/mesh.hpp"
#include "mesh/swarm_amr_remesh.hpp"
#include "pack/default_names.hpp"
#include "parthenon_arrays.hpp"
#include "utils/byte_utils.hpp"
#include "utils/error_checking.hpp"
#include "utils/mpi_types.hpp"

namespace parthenon {

namespace {

// Each old MeshBlock produces one local send plan. The payload buffer is contiguous and
// rank-partitioned so it can be concatenated into one rank-partitioned send buffer later.
struct BlockSendPlan {
  ParArray1D<char> buffer;
  std::vector<int> send_counts;
  std::vector<int> send_offsets;
};

// Remesh communication is sparse: only ranks that actually exchange remeshed particles
// need MPI messages. This helper computes the possible neighbor ranks from AMR topology
// alone.
//
// The distinction between "possible" and "actual" matters:
// - topology tells us which rank pairs might communicate after remesh
// - particle counts tell us which of those pairs actually have nonzero payloads
//
// We use the possible neighbor ranks only for the fixed-size count exchange. That avoids
// an all-rank handshake while still guaranteeing that every posted receive has a matching
// send, even when the actual count later turns out to be zero.
struct RemeshNeighborRanks {
  std::vector<int> send_ranks;
  std::vector<int> recv_ranks;
};

//----------------------------------------------------------------------------------------
// Refinement is the only remesh case where particle coordinates matter. The topology is
// still simple, though: one old block can only send particles to its daughters. Build a
// compact lookup table mapping daughter orientation bits -> new gid once on host, then
// let a device kernel classify particles into those daughters.
std::array<int, 8> GetRefinedDestinationGids(
    const Mesh *pmesh, const LogicalLocation &old_loc,
    const std::unordered_map<LogicalLocation, int> &new_gid_by_loc) {
  std::array<int, 8> child_gids{};
  for (int ox3 = 0; ox3 <= (pmesh->ndim > 2); ++ox3) {
    for (int ox2 = 0; ox2 <= (pmesh->ndim > 1); ++ox2) {
      for (int ox1 = 0; ox1 <= 1; ++ox1) {
        const auto child = old_loc.GetDaughter(ox1, ox2, ox3);
        const auto it = new_gid_by_loc.find(child);
        PARTHENON_REQUIRE(it != new_gid_by_loc.end(),
                          "Failed to find child block for swarm AMR remesh.");
        child_gids[ox1 + 2 * ox2 + 4 * ox3] = it->second;
      }
    }
  }
  return child_gids;
}

//----------------------------------------------------------------------------------------
// Build one rank-partitioned send plan for one old MeshBlock.
//
// The AMR bookkeeping already tells us the topological successor of each old block:
// same-level block, parent block after derefinement, or first daughter after
// refinement. Only the refinement case needs particle-by-particle routing. Everything is
// packed into one contiguous byte buffer so the remesh exchange can stay as one compact
// MPI_BYTE payload per peer rank.
BlockSendPlan
BuildBlockSendPlan(const std::shared_ptr<Swarm> &swarm, const Mesh *pmesh,
                   const SwarmRemeshContext &context, const int old_gid,
                   const std::unordered_map<LogicalLocation, int> &new_gid_by_loc) {
  const int nranks = Globals::nranks;
  BlockSendPlan plan{ParArray1D<char>(), std::vector<int>(nranks, 0),
                     std::vector<int>(nranks, 0)};

  // A swarm may exist on this block but currently contain no active particles. In that
  // case there is nothing to communicate and the empty plan is returned immediately.
  if (swarm->GetNumActive() == 0) return plan;

  const int new_gid = context.NewGid(old_gid);
  const auto &old_loc = context.OldLoc(old_gid);
  const auto &new_loc = context.NewLoc(new_gid);
  const bool refined = new_loc.level() > old_loc.level();
  // This format stores one leading destination gid followed by the full swarm record for
  // that particle in byte form.
  const int record_size = sizeof(int) + swarm->GetRecordSize();

  // Particle activity lives on device. Ownership decisions below are branchy and rely on
  // topology metadata, so we pull the mask to host once and build a simple host-side send
  // schedule before doing one device packing kernel.
  auto mask_h = swarm->GetMask().GetHostMirrorAndCopy();

  // The remesh exchange is rank-partitioned. For this one old block we first collect
  // "which source particle goes to which destination gid on which rank" and only later
  // flatten that routing plan into one byte buffer.
  std::vector<std::vector<int>> source_indices_by_rank(nranks);
  std::vector<std::vector<int>> dest_gids_by_rank(nranks);

  if (!refined) {
    // Same-level remaps and derefinement both have one unique topological destination for
    // every particle in this old block. There is no need to inspect positions.
    const int dest_rank = context.NewRank(new_gid);
    auto &source_indices = source_indices_by_rank[dest_rank];
    auto &dest_gids = dest_gids_by_rank[dest_rank];
    source_indices.reserve(swarm->GetNumActive());
    dest_gids.reserve(swarm->GetNumActive());
    for (int n = 0; n <= swarm->GetMaxActiveIndex(); ++n) {
      if (!mask_h(n)) continue;
      source_indices.push_back(n);
      dest_gids.push_back(new_gid);
    }
  } else {
    // Refinement is the only case where particles from one old block fan out to several
    // new leaf blocks. Compute that daughter ownership on device and only copy back the
    // compact destination-gid result, rather than mirroring full coordinate arrays.
    const auto block = pmesh->GetBlockSize(old_loc);
    const Real x_mid = 0.5 * (block.xmin(X1DIR) + block.xmax(X1DIR));
    const Real y_mid = 0.5 * (block.xmin(X2DIR) + block.xmax(X2DIR));
    const Real z_mid = 0.5 * (block.xmin(X3DIR) + block.xmax(X3DIR));

    auto child_gids_h = GetRefinedDestinationGids(pmesh, old_loc, new_gid_by_loc);
    ParArray1D<int> child_gids("swarm_amr_remesh_child_gids", child_gids_h.size());
    auto child_gids_host = Kokkos::View<const int *, Kokkos::HostSpace,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        child_gids_h.data(), child_gids_h.size());
    Kokkos::deep_copy(child_gids.KokkosView(), child_gids_host);

    ParArray1D<int> refined_dest_gids("swarm_amr_remesh_refined_dest_gids",
                                      swarm->GetMaxActiveIndex() + 1);
    auto mask = swarm->GetMask();
    auto x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto z = swarm->Get<Real>(swarm_position::z::name()).Get();
    const int ndim = pmesh->ndim;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
          if (!mask(n)) return;
          const int ox1 = x(n) > x_mid;
          // Inactive mesh directions do not participate in refinement, so they always
          // contribute daughter bit 0 even if particles carry nontrivial coordinates in
          // those directions.
          const int ox2 = ndim > 1 ? y(n) > y_mid : 0;
          const int ox3 = ndim > 2 ? z(n) > z_mid : 0;
          refined_dest_gids(n) = child_gids(ox1 + 2 * ox2 + 4 * ox3);
        });

    auto refined_dest_gids_h = refined_dest_gids.GetHostMirrorAndCopy();
    for (int n = 0; n <= swarm->GetMaxActiveIndex(); ++n) {
      if (!mask_h(n)) continue;
      const int dest_gid = refined_dest_gids_h(n);
      const int dest_rank = context.NewRank(dest_gid);
      source_indices_by_rank[dest_rank].push_back(n);
      dest_gids_by_rank[dest_rank].push_back(dest_gid);
    }
  }

  int total_particles = 0;
  for (int rank = 0; rank < nranks; ++rank) {
    // MPI counts/offsets are expressed in units of the communicated datatype. Here that
    // datatype is MPI_BYTE, so send_counts and send_offsets are byte counts.
    plan.send_offsets[rank] = total_particles * record_size;
    total_particles += source_indices_by_rank[rank].size();
    const auto count =
        static_cast<std::size_t>(source_indices_by_rank[rank].size()) * record_size;
    PARTHENON_REQUIRE(count <= std::numeric_limits<int>::max(),
                      "Swarm remesh send buffer too large for MPI.");
    plan.send_counts[rank] = static_cast<int>(count);
  }

  if (total_particles == 0) return plan;

  // Allocate the packed byte payload plus compact device arrays that map packed-record
  // index -> original particle index / destination gid. Those two arrays let the device
  // kernel pack records without consulting host-side STL containers.
  plan.buffer =
      ParArray1D<char>("swarm_amr_remesh_block_send", total_particles * record_size);
  ParArray1D<int> source_indices("swarm_amr_remesh_source_indices", total_particles);
  ParArray1D<int> dest_gids("swarm_amr_remesh_dest_gids", total_particles);
  auto source_indices_h = source_indices.GetHostMirror();
  auto dest_gids_h = dest_gids.GetHostMirror();

  // Flatten the per-rank STL vectors into contiguous arrays. The order chosen here is the
  // order used later by send_offsets, so each rank already occupies one contiguous byte
  // region inside the block-local payload.
  int offset = 0;
  for (int rank = 0; rank < nranks; ++rank) {
    for (std::size_t i = 0; i < source_indices_by_rank[rank].size(); ++i, ++offset) {
      source_indices_h(offset) = source_indices_by_rank[rank][i];
      dest_gids_h(offset) = dest_gids_by_rank[rank][i];
    }
  }
  source_indices.DeepCopy(source_indices_h);
  dest_gids.DeepCopy(dest_gids_h);

  // Swarm variables are stored as separate typed arrays. PackVariables gives flat typed
  // accessors into those arrays, including all components of vector/tensor particle
  // variables, so one kernel can serialize the full particle record.
  PackIndexMap real_imap, int_imap, uint64_imap;
  const auto vreal =
      swarm->PackVariables<Real>(swarm->GetVariableNames<Real>(), real_imap);
  const auto vint = swarm->PackVariables<int>(swarm->GetVariableNames<int>(), int_imap);
  const auto vuint64 = swarm->PackVariables<std::uint64_t>(
      swarm->GetVariableNames<std::uint64_t>(), uint64_imap);
  const int real_pack_dim = vreal.GetDim(2);
  const int int_pack_dim = vint.GetDim(2);
  const int uint64_pack_dim = vuint64.GetDim(2);

  auto buffer = plan.buffer;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, total_particles - 1,
      KOKKOS_LAMBDA(const int n) {
        // n indexes packed records, not swarm particle ids directly. source_indices maps
        // back to the source particle chosen on host above.
        const int src_idx = source_indices(n);
        int buffer_idx = n * record_size;

        // The packed record layout is:
        //   [destination gid][all Real vars][all int vars][all uint64 vars]
        //
        // The first entry is routing metadata. Everything after that is the actual swarm
        // state that must survive remeshing.
        byte_utils::PackValue(buffer, buffer_idx, dest_gids(n));
        for (int i = 0; i < real_pack_dim; ++i) {
          byte_utils::PackValue(buffer, buffer_idx, vreal(i, src_idx));
        }
        for (int i = 0; i < int_pack_dim; ++i) {
          byte_utils::PackValue(buffer, buffer_idx, vint(i, src_idx));
        }
        for (int i = 0; i < uint64_pack_dim; ++i) {
          byte_utils::PackValue(buffer, buffer_idx, vuint64(i, src_idx));
        }
      });

  return plan;
}

//----------------------------------------------------------------------------------------
// Unpack a subset of received particle records into one destination swarm.
//
// The rank-level MPI exchange produces one shared receive buffer. record_indices selects
// the records belonging to one destination gid inside that buffer. This routine allocates
// new particle slots in the destination swarm and restores the typed swarm variables into
// those slots.
void UnpackReceivedParticles(const std::shared_ptr<Swarm> &swarm,
                             const ParArray1D<char> &buffer,
                             const std::vector<int> &record_indices,
                             const int record_size) {
  const int count = record_indices.size();
  if (count == 0) return;

  // At this point ownership is already known: every record_indices entry belongs to this
  // destination swarm. The only remaining job is to create new particle slots and decode
  // the byte payload into those slots.
  auto new_particles = swarm->AddEmptyParticles(count);

  ParArray1D<int> record_indices_d("swarm_amr_remesh_record_indices", count);
  auto record_indices_h = record_indices_d.GetHostMirror();

  // record_indices was assembled on host after regrouping the rank-wide receive buffer
  // by gid. Copy it to device so the unpack kernel can jump directly to the selected
  // records.
  for (int i = 0; i < count; ++i) {
    record_indices_h(i) = record_indices[i];
  }
  record_indices_d.DeepCopy(record_indices_h);

  PackIndexMap real_imap, int_imap, uint64_imap;
  auto vreal = swarm->PackVariables<Real>(swarm->GetVariableNames<Real>(), real_imap);
  auto vint = swarm->PackVariables<int>(swarm->GetVariableNames<int>(), int_imap);
  auto vuint64 = swarm->PackVariables<std::uint64_t>(
      swarm->GetVariableNames<std::uint64_t>(), uint64_imap);
  const int real_pack_dim = vreal.GetDim(2);
  const int int_pack_dim = vint.GetDim(2);
  const int uint64_pack_dim = vuint64.GetDim(2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, count - 1,
      KOKKOS_LAMBDA(const int n) {
        // AddEmptyParticles returns a compact context object that can be queried directly
        // on device, so there is no need to materialize a second array of destination
        // particle indices here.
        const int particle_idx = new_particles.GetNewParticleIndex(n);

        // Skip the leading destination gid because ownership has already been resolved by
        // the caller. What remains is the serialized swarm state.
        int buffer_idx = record_indices_d(n) * record_size + sizeof(int);
        for (int i = 0; i < real_pack_dim; ++i) {
          vreal(i, particle_idx) = byte_utils::UnpackValue<Real>(buffer, buffer_idx);
        }
        for (int i = 0; i < int_pack_dim; ++i) {
          vint(i, particle_idx) = byte_utils::UnpackValue<int>(buffer, buffer_idx);
        }
        for (int i = 0; i < uint64_pack_dim; ++i) {
          vuint64(i, particle_idx) =
              byte_utils::UnpackValue<std::uint64_t>(buffer, buffer_idx);
        }
      });
}

//----------------------------------------------------------------------------------------
// Convert per-rank counts into starting offsets and return the total size.
int BuildOffsets(const std::vector<int> &counts, std::vector<int> &offsets) {
  int total = 0;
  for (int rank = 0; rank < counts.size(); ++rank) {
    offsets[rank] = total;
    total += counts[rank];
  }
  return total;
}

//----------------------------------------------------------------------------------------
// Return the possible destination ranks for particles that start in one old block.
//
// Same-level remaps and derefinement have one unique topological destination. Refinement
// is the only case that can fan particles out to multiple blocks, and even then the
// possible destinations are just the daughters of the old block.
std::vector<int> GetCandidateDestinationRanks(
    const Mesh *pmesh, const SwarmRemeshContext &context, const int old_gid,
    const std::unordered_map<LogicalLocation, int> &new_gid_by_loc) {
  const int new_gid = context.NewGid(old_gid);
  const auto &old_loc = context.OldLoc(old_gid);
  const auto &new_loc = context.NewLoc(new_gid);

  std::vector<int> ranks;
  ranks.reserve(1 << (pmesh->ndim - 1 + 1));

  if (new_loc.level() <= old_loc.level()) {
    ranks.push_back(context.NewRank(new_gid));
    return ranks;
  }

  for (int ox3 = 0; ox3 <= (pmesh->ndim > 2); ++ox3) {
    for (int ox2 = 0; ox2 <= (pmesh->ndim > 1); ++ox2) {
      for (int ox1 = 0; ox1 <= 1; ++ox1) {
        const auto child = old_loc.GetDaughter(ox1, ox2, ox3);
        const auto it = new_gid_by_loc.find(child);
        PARTHENON_REQUIRE(it != new_gid_by_loc.end(),
                          "Failed to find child block for swarm AMR remesh.");
        const int rank = context.NewRank(it->second);
        if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
          ranks.push_back(rank);
        }
      }
    }
  }

  return ranks;
}

//----------------------------------------------------------------------------------------
// Derive the sparse rank communication pattern implied by the AMR topology.
//
// Every rank has the full old/new block topology, so it can independently determine
// which remote ranks it may need to exchange remeshed particles with. This lets the
// remesh path stay in the same point-to-point style as the rest of Parthenon instead of
// requiring an all-rank communication operation.
RemeshNeighborRanks
BuildRemeshNeighborRanks(const Mesh *pmesh, const SwarmRemeshContext &context,
                         const std::unordered_map<LogicalLocation, int> &new_gid_by_loc) {
  const int nranks = Globals::nranks;
  const int my_rank = Globals::my_rank;
  std::vector<bool> may_send_to(nranks, false);
  std::vector<bool> may_recv_from(nranks, false);

  for (int old_gid = 0; old_gid < context.NumOldBlocks(); ++old_gid) {
    const int old_rank = context.OldRank(old_gid);
    const auto candidate_ranks =
        GetCandidateDestinationRanks(pmesh, context, old_gid, new_gid_by_loc);

    if (old_rank == my_rank) {
      for (const int rank : candidate_ranks) {
        if (rank != my_rank) may_send_to[rank] = true;
      }
    } else {
      for (const int rank : candidate_ranks) {
        if (rank == my_rank) {
          may_recv_from[old_rank] = true;
          break;
        }
      }
    }
  }

  RemeshNeighborRanks neighbors;
  for (int rank = 0; rank < nranks; ++rank) {
    if (may_send_to[rank]) neighbors.send_ranks.push_back(rank);
    if (may_recv_from[rank]) neighbors.recv_ranks.push_back(rank);
  }
  return neighbors;
}

} // namespace

//----------------------------------------------------------------------------------------
// Remesh all swarms after the field AMR remap has finished.
//
// Fields already use Parthenon's native same-level / prolongation / restriction AMR
// paths. Swarms are different because particle ownership after remesh is geometric: a
// particle may belong to an arbitrary new leaf block after refinement or derefinement.
// This routine therefore performs a separate ownership-based remap for each swarm.
void RemeshSwarms(const std::shared_ptr<StateDescriptor> &resolved_packages,
                  const BlockList_t &old_block_list, Mesh *pmesh,
                  const SwarmRemeshContext &context) {
  // Parthenon packages may define no swarms at all. In that case the field AMR remap is
  // sufficient and the swarm-specific pass is skipped entirely.
  if (resolved_packages->AllSwarms().empty()) return;

  const int nranks = Globals::nranks;

  // The new leaf mesh has already been built by mesh-amr_loadbalance before this routine
  // is called. Cache a logical-location -> gid lookup once so refined particles can map
  // daughter logical locations to concrete destination blocks cheaply.
  std::unordered_map<LogicalLocation, int> new_gid_by_loc;
  new_gid_by_loc.reserve(context.NumNewBlocks());
  for (int gid = 0; gid < context.NumNewBlocks(); ++gid) {
    new_gid_by_loc.emplace(context.NewLoc(gid), gid);
  }

  // The possible rank communication pattern depends only on AMR topology, not on swarm
  // contents. Compute it once here and reuse it for each swarm's count exchange.
  const auto neighbors = BuildRemeshNeighborRanks(pmesh, context, new_gid_by_loc);

  for (const auto &swarm_pair : resolved_packages->AllSwarms()) {
    const auto &swarm_name = swarm_pair.first;

    // The remesh exchange is performed independently per swarm. That keeps the record
    // schema simple because each swarm can have its own set of variables.
    std::vector<BlockSendPlan> block_plans;
    std::vector<int> send_counts(nranks, 0);
    std::vector<int> send_offsets(nranks, 0);

    for (int on = context.old_gid_first; on <= context.old_gid_last; ++on) {
      const int nn = context.NewGid(on);

      // If a block stayed on the same rank and refinement level, mesh-amr_loadbalance
      // already preserved that MeshBlock in-place. Its swarm storage is still valid, so
      // remesh communication is only needed for blocks whose ownership or level changed.
      const bool same_rank_same_level =
          (context.OldRank(on) == context.NewRank(nn)) &&
          (context.OldLoc(on).level() == context.NewLoc(nn).level());
      if (same_rank_same_level) continue;

      auto &old_pmb = old_block_list[on - context.old_gid_first];
      auto &old_swarm_container = old_pmb->meshblock_data.Get()->GetSwarmData();
      if (!old_swarm_container->Contains(swarm_name)) continue;

      auto swarm = old_swarm_container->Get(swarm_name);
      auto plan = BuildBlockSendPlan(swarm, pmesh, context, on, new_gid_by_loc);
      if (plan.buffer.size() == 0) continue;

      for (int rank = 0; rank < nranks; ++rank) {
        send_counts[rank] += plan.send_counts[rank];
      }
      block_plans.emplace_back(std::move(plan));
    }

    // Each old block produced its own packed payload for clarity. MPI wants one payload
    // per rank, so concatenate those block-local payloads into one byte stream here.
    const int send_total = BuildOffsets(send_counts, send_offsets);

    ParArray1D<char> send_buffer;
    if (send_total > 0) {
      send_buffer = ParArray1D<char>("swarm_amr_remesh_send", send_total);
      auto next_offsets = send_offsets;
      for (const auto &plan : block_plans) {
        for (int rank = 0; rank < nranks; ++rank) {
          // plan.send_offsets/counts already describe byte ranges for this rank inside
          // the block-local payload. Copy those byte ranges into the final rank-global
          // send buffer without unpacking/repacking anything.
          const int count = plan.send_counts[rank];
          if (count == 0) continue;
          const int src_begin = plan.send_offsets[rank];
          const int dst_begin = next_offsets[rank];
          auto src = Kokkos::subview(plan.buffer.KokkosView(),
                                     std::make_pair(src_begin, src_begin + count));
          auto dst = Kokkos::subview(send_buffer.KokkosView(),
                                     std::make_pair(dst_begin, dst_begin + count));
          Kokkos::deep_copy(dst, src);
          next_offsets[rank] += count;
        }
      }
    }

    std::vector<int> recv_counts(nranks, 0);
    std::vector<int> recv_offsets(nranks, 0);
    int recv_total = 0;

#ifdef MPI_PARALLEL
    const auto swarm_comm = pmesh->GetMPIComm(swarm_name);
    constexpr int count_tag = 0;
    std::vector<MPI_Request> count_recv_reqs;
    std::vector<MPI_Request> count_send_reqs;
    count_recv_reqs.reserve(neighbors.recv_ranks.size());
    count_send_reqs.reserve(neighbors.send_ranks.size());

    // Same-rank traffic never touches MPI. The byte count is already known locally.
    recv_counts[Globals::my_rank] = send_counts[Globals::my_rank];

    // Exchange one fixed-size integer count with each possible peer rank. Zero counts are
    // sent too, because the topology can say "communication is possible" even when this
    // particular swarm instance has no particles taking that path.
    for (const int rank : neighbors.recv_ranks) {
      MPI_Request req;
      PARTHENON_MPI_CHECK(
          MPI_Irecv(&recv_counts[rank], 1, MPI_INT, rank, count_tag, swarm_comm, &req));
      count_recv_reqs.push_back(req);
    }
    for (const int rank : neighbors.send_ranks) {
      MPI_Request req;
      PARTHENON_MPI_CHECK(
          MPI_Isend(&send_counts[rank], 1, MPI_INT, rank, count_tag, swarm_comm, &req));
      count_send_reqs.push_back(req);
    }

    WaitAll(count_recv_reqs);
    WaitAll(count_send_reqs);
    recv_total = BuildOffsets(recv_counts, recv_offsets);
#else
    recv_counts = send_counts;
    recv_offsets = send_offsets;
    recv_total = send_total;
#endif

    ParArray1D<char> recv_buffer;
    if (recv_total > 0) {
      recv_buffer = ParArray1D<char>("swarm_amr_remesh_recv", recv_total);
    }

    // The packed payload is produced on device. Fence once before exposing raw pointers
    // to MPI, otherwise the host may race ahead of the packing kernels.
    Kokkos::fence();
#ifdef MPI_PARALLEL
    constexpr int payload_tag = 1;
    std::vector<MPI_Request> payload_recv_reqs;
    std::vector<MPI_Request> payload_send_reqs;
    payload_recv_reqs.reserve(neighbors.recv_ranks.size());
    payload_send_reqs.reserve(neighbors.send_ranks.size());

    // Same-rank payloads stay entirely on-device and bypass MPI. This preserves the
    // direct device-resident path while avoiding unnecessary trips through the MPI stack.
    if (recv_counts[Globals::my_rank] > 0) {
      const int src_begin = send_offsets[Globals::my_rank];
      const int dst_begin = recv_offsets[Globals::my_rank];
      const int count = recv_counts[Globals::my_rank];
      auto src = Kokkos::subview(send_buffer.KokkosView(),
                                 std::make_pair(src_begin, src_begin + count));
      auto dst = Kokkos::subview(recv_buffer.KokkosView(),
                                 std::make_pair(dst_begin, dst_begin + count));
      Kokkos::deep_copy(dst, src);
    }

    // Remote payloads use the usual sparse point-to-point pattern Parthenon already uses
    // for field and boundary communication. The buffers themselves remain device
    // resident, so GPU-aware MPI can still move particle data directly device-to-device.
    for (const int rank : neighbors.recv_ranks) {
      if (recv_counts[rank] == 0) continue;
      MPI_Request req;
      PARTHENON_MPI_CHECK(MPI_Irecv(recv_buffer.data() + recv_offsets[rank],
                                    recv_counts[rank], MPI_BYTE, rank, payload_tag,
                                    swarm_comm, &req));
      payload_recv_reqs.push_back(req);
    }
    for (const int rank : neighbors.send_ranks) {
      if (send_counts[rank] == 0) continue;
      MPI_Request req;
      PARTHENON_MPI_CHECK(MPI_Isend(send_buffer.data() + send_offsets[rank],
                                    send_counts[rank], MPI_BYTE, rank, payload_tag,
                                    swarm_comm, &req));
      payload_send_reqs.push_back(req);
    }

    WaitAll(payload_recv_reqs);
    WaitAll(payload_send_reqs);
#else
    if (recv_total > 0) {
      // In the non-MPI build the same byte stream can just be copied locally.
      Kokkos::deep_copy(recv_buffer, send_buffer);
    }
#endif

    if (recv_total == 0) continue;

    // Every received record has the same byte length because the swarm schema is fixed.
    // That lets us recover the number of particle records with one division.
    auto prototype_swarm =
        pmesh->block_list.front()->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
    const int record_size = sizeof(int) + prototype_swarm->GetRecordSize();
    PARTHENON_REQUIRE(recv_total % record_size == 0,
                      "Swarm remesh receive buffer has invalid size.");

    const int num_recv_particles = recv_total / record_size;
    ParArray1D<int> recv_gids("swarm_amr_remesh_recv_gids", num_recv_particles);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, num_recv_particles - 1, KOKKOS_LAMBDA(const int n) {
          // The first bytes of each record store the destination gid. Read only that
          // here; the actual swarm state stays in the shared byte buffer until records
          // have been regrouped by destination block.
          int offset = n * record_size;
          recv_gids(n) = byte_utils::UnpackValue<int>(recv_buffer, offset);
        });
    auto recv_gids_h = recv_gids.GetHostMirrorAndCopy();

    // MPI delivers one rank-wide stream. Convert that into gid -> list of record indices
    // so each local MeshBlock can unpack only the particles that now belong to it.
    std::unordered_map<int, std::vector<int>> records_by_gid;
    for (int n = 0; n < num_recv_particles; ++n) {
      records_by_gid[recv_gids_h(n)].push_back(n);
    }

    for (const auto &[gid, record_indices] : records_by_gid) {
      // Find the destination MeshBlock in the newly remeshed mesh and append just the
      // records addressed to that gid.
      auto pmb = pmesh->FindMeshBlock(gid);
      auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
      UnpackReceivedParticles(swarm, recv_buffer, record_indices, record_size);
    }
  }
}

//----------------------------------------------------------------------------------------
// MeshData and MeshBlockData may still hold cached swarm pack views built before remesh.
// Once particle ownership changes, those cached views are stale and must be rebuilt on
// next access.
void ClearSwarmCachesAfterRemesh(Mesh *pmesh, const BlockList_t &block_list) {
  for (const auto &pmb : block_list) {
    for (const auto &[label, mbd] : pmb->meshblock_data.Stages()) {
      mbd->ClearSwarmCaches();
    }
  }
  for (const auto &[label, md] : pmesh->mesh_data.Stages()) {
    md->ClearSwarmCaches();
  }
}

} // namespace parthenon
