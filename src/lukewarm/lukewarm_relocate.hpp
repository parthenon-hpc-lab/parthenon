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
#ifndef LUKEWARM_LUKEWARM_RELOCATE_HPP_
#define LUKEWARM_LUKEWARM_RELOCATE_HPP_

// This file was made in part with generative AI.

#include <string>

#include "basic_types.hpp"
#include "tasks/tasks.hpp"

namespace parthenon {

class Mesh;
template <typename T>
class MeshData;

// Particles that leave a MeshData travel to the neighbor that owns their new block, on
// this rank or another, in rounds:
//   1. LocateLukewarm() on every MeshData. If no particle anywhere has left its
//      MeshData, every particle is settled and the rounds are over.
//   2. StartReceiveLukewarm() and SendLukewarm() on every MeshData.
//   3. ReceiveLukewarm() on every MeshData. Delivered particles may still have to travel
//      further, so go back to 1.
// RelocateLukewarm() runs these rounds to the end.

// Move every particle of lukewarm `name` on md into the block containing its position,
// following it across as many blocks of md as it takes. Particles marked for removal
// and particles that leave the mesh through a non-periodic boundary are removed.
// Particles that reach a block of another MeshData wait for SendLukewarm(), and until it
// runs no particles may be created or removed on md. Returns how many are waiting.
int LocateLukewarm(MeshData<Real> *md, const std::string &name);

// Send the particles waiting on md to the neighbors they reached, and remove them. Every
// neighbor gets a message, even if no particles go to it. Incomplete while md's last
// messages are still in flight.
TaskStatus SendLukewarm(MeshData<Real> *md, const std::string &name);

// Start receiving particles from md's neighbors
TaskStatus StartReceiveLukewarm(MeshData<Real> *md, const std::string &name);

// Incomplete until every neighbor's message has arrived. Then append the particles to md
// in their new blocks.
TaskStatus ReceiveLukewarm(MeshData<Real> *md, const std::string &name);

// The particles the last ReceiveLukewarm() appended to md, at the end of its storage,
// or an empty range if none arrived. Creating particles keeps them in place, but
// removing particles may move them, so this is cleared by the next LocateLukewarm() and
// must not be used after Lukewarm::RemoveMarked(). After a round in which every
// particle that stayed on md was finished, as when a kernel moves each particle until it
// is done or leaves md, these are the only particles still to be worked on:
//   const auto range = first ? IndexRange{0, pack.GetNParticles() - 1}
//                            : GetLastReceived(md, name);
// RelocateLukewarm() ends with a LocateLukewarm(), so it leaves this empty.
IndexRange GetLastReceived(MeshData<Real> *md, const std::string &name);

// Locate, send, and receive until every particle of lukewarm `name` is settled in its
// block. This is collective over all ranks.
void RelocateLukewarm(Mesh *pmesh, const std::string &name);

// Add one round of locating, sending, and receiving the particles of lukewarm `name` on
// md to a task list, as AddBoundaryExchangeTasks() does for fields. Every MeshData of
// the mesh has to take part, e.g., from the task lists of one TaskRegion. A particle
// is delivered to the first block of another MeshData on its way, so after one round
// every particle is settled as long as none moved past a whole block, as when moves are
// limited by a CFL condition. Otherwise use RelocateLukewarm().
TaskID AddLukewarmExchangeTasks(TaskID dependency, TaskList &tl, MeshData<Real> *md,
                                const std::string &name);

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_RELOCATE_HPP_
