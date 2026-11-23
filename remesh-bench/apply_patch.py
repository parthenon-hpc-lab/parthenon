#!/usr/bin/env python3
"""Insert benchmark-only remesh timers into a parthenon source tree.

Works on develop (point-to-point remesh MPI), the bvals-remesh branch, and its merge
with faster-coalesced-comms by matching anchors instead of applying a fixed diff. Idempotent: refuses to patch twice.
Every mark is placed outside rank-dependent conditionals, so BENCH_BARRIER=1 is safe.

usage: apply_patch.py <parthenon-src-dir>
"""
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
target = src / "src/mesh/mesh-amr_loadbalance.cpp"
text = target.read_text()
if "bench_remesh_timer.hpp" in text:
    sys.exit(f"{target} is already patched")

is_pr = "SetRemeshNeighbors_" in text
# The merge branch waits for its remesh sends in ExchangeBoundaryBuffers itself; the PR
# only does so with the BENCH_WAIT_SENDS experiment inserted below
has_send_wait = "Wait for our own sends as well" in text

if is_pr:
    # Split the exchange into post / pack+send / wait / unpack, with the marks outside
    # the per-rank have_send/have_recv conditions
    old = ("  if (have_recv) StartReceiveBoundBufs<RECV>(md_recv);\n"
           "  std::int64_t iters = 0;\n"
           "  if (have_send) {\n"
           "    while (SendBoundBufs<SEND>(md_send) != TaskStatus::complete && ++iters < max_it) {\n"
           "    }\n"
           "  }\n"
           "  if (have_recv) {\n"
           "    while (ReceiveBoundBufs<RECV>(md_recv) != TaskStatus::complete && ++iters < max_it) {\n"
           "    }\n"
           "    SetBounds<RECV>(md_recv);\n"
           "  }\n")
    new = ("  if (have_recv) StartReceiveBoundBufs<RECV>(md_recv);\n"
           "  bench::MarkCurrent(bench::p_post);\n"
           "  std::int64_t iters = 0;\n"
           "  if (have_send) {\n"
           "    while (SendBoundBufs<SEND>(md_send) != TaskStatus::complete && ++iters < max_it) {\n"
           "    }\n"
           "  }\n"
           "  bench::MarkCurrent(bench::p_send);\n"
           "  if (have_recv) {\n"
           "    while (ReceiveBoundBufs<RECV>(md_recv) != TaskStatus::complete && ++iters < max_it) {\n"
           "    }\n"
           "  }\n"
           "  bench::MarkCurrent(bench::p_recv);\n"
           "  if (have_recv) SetBounds<RECV>(md_recv);\n"
           "  bench::MarkCurrent(bench::p_unpack);\n")
    if has_send_wait:
        # Merge branch: the wait follows the receive block, so close it with a mark
        req = '  PARTHENON_REQUIRE(iters < max_it, "Remesh communication did not complete.");\n'
        if text.count(req) != 1:
            sys.exit("ExchangeBoundaryBuffers completion check not found verbatim")
        text = text.replace(req, "  bench::MarkCurrent(bench::p_sendwait);\n" + req)
    else:
        new += ("  if (bench::WaitSendsRequested()) {\n"
                "    Mesh *pmesh = md_send->GetMeshPointer();\n"
                "    bool done;\n"
                "    do {\n"
                "      done = true;\n"
                "      for (auto &[k, comm] : pmesh->boundary_comm_map)\n"
                "        if (comm.IsPureSender()) done = comm.IsAvailableForWrite() && done;\n"
                "      if (pmesh->do_coalesced_comms)\n"
                "        done = pmesh->pcoalesced_comms->IsAvailableForWrite(md_send.get(), SEND) &&\n"
                "               done;\n"
                "    } while (!done);\n"
                "  }\n"
                "  bench::MarkCurrent(bench::p_sendwait);\n")
    if text.count(old) != 1:
        sys.exit("ExchangeBoundaryBuffers body not found verbatim")
    text = text.replace(old, new)
    # Accessor needed by the BENCH_WAIT_SENDS experiment
    cb = src / "src/utils/communication_buffer.hpp"
    cbt = cb.read_text()
    anchor = "  bool IsActive() const { return active_; }\n"
    if not has_send_wait and "IsPureSender" not in cbt:
        if cbt.count(anchor) != 1:
            sys.exit("CommBuffer::IsActive anchor not found")
        cb.write_text(cbt.replace(anchor, anchor + "  bool IsPureSender() const { "
                                  "return *comm_type_ == BuffCommType::sender; }\n"))

lines = text.splitlines(keepends=True)


def find(pattern, start=0):
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    sys.exit(f"anchor not found: {pattern!r} (after line {start + 1})")


inserts = []  # (index to insert before, text)


def mark_before(i, span, indent="    "):
    inserts.append((i, f"{indent}bench_timer.Mark(bench::{span});\n"))


def mark_after(i, span, indent="    "):
    inserts.append((i + 1, f"{indent}bench_timer.Mark(bench::{span});\n"))


i = find(r'^#include "mesh/mesh.hpp"')
inserts.append((i + 1, '#include "mesh/bench_remesh_timer.hpp"\n'))

f = find(r"^void Mesh::RedistributeAndRefineMeshBlocks\(")
i = find(r"^  PARTHENON_INSTRUMENT$", f)
inserts.append((i + 1, "  bench::RemeshTimer bench_timer;\n"))

i = find(r"^  (const )?int nbe = nbs \+ nblist\[Globals::my_rank\] - 1;", f)
inserts.append((i + 1, "  bench_timer.Mark(bench::lb);\n"
                       "  bench_timer.Count(loclist, newloc, ranklist, newrank, newtoold);\n"))

if is_pr:
    i = find(r"^  \} // AMR Construct new MeshBlockList region$", f)
    mark_after(i, "build", "  ")
    i = find(r"^    // Prolongation onto newly refined blocks uses", f)
    mark_before(i, "p_nbrs")
    t = find(r"^    // Transfer \(and restrict where needed\) the field data", f)
    mark_before(t, "p_halo")
    i = find(r"^    ResetCommunicationInfrastructure\(this\);$", t)
    mark_after(i, "p_reset")
    i = find(r"^    tag_map.ResolveMap\(\);$", t)
    mark_after(i, "p_tagmap")
    i = find(r"^    BuildRemeshBoundaryBuffers\(md_old, md_new\);$", t)
    mark_after(i, "p_bufs")
    i = find(r"^    ExchangeCoalescedBufferInfo\(this\);$", t)
    mark_after(i, "p_cinfo")
    a = find(r"ExchangeRemeshCounters_\(block_list", t)
    i = find(r"^    Kokkos::fence\(\);$", a)
    mark_after(i, "p_counters")
else:
    r = find(r"^  // Restrict fine to coarse buffers$", f)
    i = find(r"^  Kokkos::fence\(\);$", r)
    mark_after(i, "d_restrict", "  ")
    s = find(r"^  \} // AMR Send region$", f)
    i = find(r"^#endif // MPI_PARALLEL$", s)
    mark_after(i, "d_send", "  ")
    i = find(r"^  \} // AMR Construct new MeshBlockList region$", f)
    mark_after(i, "build", "  ")
    a = find(r"// Fence here to be careful that all communication is finished", f)
    i = find(r"^    Kokkos::fence\(\);$", a)
    mark_after(i, "d_recv")

# Prolongation ends where swarm handling starts
i = find(r"^    // Field data is transferred by the", f)
mark_before(i, "prolong")

# Shared finalize steps
i = find(r"^    SetMeshBlockNeighbors\(this, GridIdentifier::leaf\(\), block_list, ranklist\);$",
         f)
mark_before(i, "f_misc")
i = find(r"^    if \(noncc_names.size\(\) == 0", f)
mark_before(i, "f_nbrs")
i = find(r"^    // Call to fill ghosts", f)
mark_before(i, "f_bufs")
i = find(r"^    FillDerived\(\);$", f)
mark_after(i, "f_comm")

# Last ResetLoadBalanceVariables() in the function; Finish() closes f_rest
i = find(r"^  ResetLoadBalanceVariables\(\);$", f)
inserts.append((i, "  bench_timer.Finish();\n"))

for idx, txt in sorted(inserts, key=lambda x: x[0], reverse=True):
    lines.insert(idx, txt)
target.write_text("".join(lines))
shutil.copy(Path(__file__).with_name("bench_remesh_timer.hpp"), src / "src/mesh/")
layout = ("merge" if has_send_wait else "PR") if is_pr else "develop"
print(f"patched {target} ({layout} layout)")
