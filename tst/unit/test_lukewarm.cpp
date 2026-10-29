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
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_container.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "lukewarm/lukewarm_pack.hpp"
#include "lukewarm/lukewarm_relocate.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pack/default_names.hpp"
#include "parameter_input.hpp"
#include "parthenon_mpi.hpp"
#include "utils/mpi_types.hpp"

using parthenon::AmrTag;
using parthenon::ApplicationInput;
using parthenon::CoordinateDirection;
using parthenon::Lukewarm;
using parthenon::LukewarmContainer;
using parthenon::LukewarmGeometry;
using parthenon::MakeLukewarmPackDescriptor;
using parthenon::Mesh;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::ParArray1DRaw;
using parthenon::ParArray2DRaw;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace test {
PAR_VAR(test, weight);
PAR_TYPED_VAR(int, test, tag);
PAR_TYPED_VAR(std::uint64_t, test, key);
} // namespace test

namespace wrong {
PAR_VAR(test, missing);
PAR_VAR(test, tag); // a Real name for an Integer field
} // namespace wrong

namespace {

template <typename View>
auto ToHost(const View &v) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
}

// Write value(c, n) into every component and particle of a field
template <typename T, typename F>
void Fill(parthenon::LukewarmVariable<T> &var, F value) {
  auto h = Kokkos::create_mirror_view(var.data);
  for (int c = 0; c < h.extent_int(0); ++c) {
    for (int n = 0; n < h.extent_int(1); ++n) {
      h(c, n) = value(c, n);
    }
  }
  Kokkos::deep_copy(var.data, h);
}

// An 8x8 mesh on [0, 1]^2 of four 4x4 blocks, periodic unless asked otherwise. Refining
// splits the lower left block into four, to give neighbors at different levels. An
// adaptive mesh may gain one level, and blocks derefine as soon as they ask to.
std::shared_ptr<ParameterInput> MakePin(const int pack_size,
                                        const bool periodic_x1 = true,
                                        const bool refined = false,
                                        const bool adaptive = false) {
  const std::string x1_bc = periodic_x1 ? "periodic" : "outflow";
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  if (refined) is << "refinement = static\n";
  if (adaptive) {
    is << "refinement = adaptive\n";
    is << "numlevel = 2\n";
    is << "derefine_count = 1\n";
  }
  is << "nx1 = 8\n";
  is << "x1min = 0.0\n";
  is << "x1max = 1.0\n";
  is << "nx2 = 8\n";
  is << "x2min = 0.0\n";
  is << "x2max = 1.0\n";
  is << "nx3 = 1\n";
  is << "x3min = -0.5\n";
  is << "x3max = 0.5\n";
  is << "ix1_bc = " << x1_bc << "\n";
  is << "ox1_bc = " << x1_bc << "\n";
  is << "ix2_bc = periodic\n";
  is << "ox2_bc = periodic\n";
  is << "ix3_bc = periodic\n";
  is << "ox3_bc = periodic\n";
  is << "pack_size = " << pack_size << "\n";
  is << "<parthenon/meshblock>\n";
  is << "nx1 = 4\n";
  is << "nx2 = 4\n";
  is << "nx3 = 1\n";
  if (refined) {
    is << "<parthenon/static_refinement0>\n";
    is << "level = 1\n";
    is << "x1min = 0.1\n";
    is << "x1max = 0.2\n";
    is << "x2min = 0.1\n";
    is << "x2max = 0.2\n";
    is << "x3min = -0.5\n";
    is << "x3max = 0.5\n";
  }
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return pin;
}

// Values of the fields of the particle with a given id, of every type and with several
// components, so that every field can be followed from block to block and rank to rank
Real Weight(const int c, const std::uint64_t id) { return (c == 0 ? 10.0 : -0.5) * id; }
int Tag(const int c, const std::uint64_t id) {
  return c == 0 ? id % 3 : -static_cast<int>(id);
}
// Beyond 32 bits, to catch truncation
std::uint64_t Key(const int c, const std::uint64_t id) {
  return (std::uint64_t{1} << (40 + c)) + 7 * id;
}

// Create particles at the given positions in the plane, filling in the other fields
void CreateAt(Lukewarm &lw, const std::vector<std::pair<Real, Real>> &positions) {
  const auto range = lw.Create(positions.size());
  auto x = ToHost(lw.Get<Real>(lukewarm::x::name()).data);
  auto y = ToHost(lw.Get<Real>(lukewarm::y::name()).data);
  auto z = ToHost(lw.Get<Real>(lukewarm::z::name()).data);
  auto weight = ToHost(lw.Get<Real>(test::weight::name()).data);
  auto tag = ToHost(lw.Get<int>(test::tag::name()).data);
  auto key = ToHost(lw.Get<std::uint64_t>(test::key::name()).data);
  auto id = ToHost(lw.Get<std::uint64_t>(lukewarm::id::name()).data);
  for (int n = range.s; n <= range.e; ++n) {
    x(0, n) = positions[n - range.s].first;
    y(0, n) = positions[n - range.s].second;
    z(0, n) = 0.0;
    for (int c = 0; c < 2; ++c) {
      weight(c, n) = Weight(c, id(0, n));
      tag(c, n) = Tag(c, id(0, n));
      key(c, n) = Key(c, id(0, n));
    }
  }
  Kokkos::deep_copy(lw.Get<Real>(lukewarm::x::name()).data, x);
  Kokkos::deep_copy(lw.Get<Real>(lukewarm::y::name()).data, y);
  Kokkos::deep_copy(lw.Get<Real>(lukewarm::z::name()).data, z);
  Kokkos::deep_copy(lw.Get<Real>(test::weight::name()).data, weight);
  Kokkos::deep_copy(lw.Get<int>(test::tag::name()).data, tag);
  Kokkos::deep_copy(lw.Get<std::uint64_t>(test::key::name()).data, key);
}

// Every rank's values, in rank order, on every rank
template <typename T>
std::vector<T> AllGather(const std::vector<T> &local) {
#ifdef MPI_PARALLEL
  const int nranks = parthenon::Globals::nranks;
  const int n = local.size();
  std::vector<int> counts(nranks), offsets(nranks, 0);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 1; r < nranks; ++r) {
    offsets[r] = offsets[r - 1] + counts[r - 1];
  }
  std::vector<T> all(offsets.back() + counts.back());
  const auto type = parthenon::MPITypeMap<T>::type();
  MPI_Allgatherv(local.data(), n, type, all.data(), counts.data(), offsets.data(), type,
                 MPI_COMM_WORLD);
  return all;
#else
  return local;
#endif
}

int SumOverRanks(int n) {
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  return n;
}

// Ids of the particles created by rank 0, in order
std::uint64_t RootId(const int k) { return k * parthenon::Globals::nranks; }

std::vector<MeshData<Real> *> LukewarmMeshData(Mesh *mesh) {
  std::vector<MeshData<Real> *> mds;
  for (auto &partition : mesh->GetDefaultBlockPartitions()) {
    mds.push_back(mesh->mesh_data.Add("base", partition).get());
  }
  return mds;
}

// Check that every particle of tracers lies in the block it claims, with its fields
// intact, and return the position of each by id, over all ranks
std::map<std::uint64_t, std::pair<Real, Real>> CheckSettled(Mesh *mesh) {
  std::vector<std::uint64_t> ids;
  std::vector<Real> xs, ys;
  for (auto *md : LukewarmMeshData(mesh)) {
    auto &lw = md->GetLukewarm("tracers");
    REQUIRE(md->GetLukewarmContainer().GetComms("tracers").nleaving == 0);
    auto x = ToHost(lw.Get<Real>(lukewarm::x::name()).data);
    auto y = ToHost(lw.Get<Real>(lukewarm::y::name()).data);
    auto block = ToHost(lw.GetData<int>());
    auto weight = ToHost(lw.Get<Real>(test::weight::name()).data);
    auto tag = ToHost(lw.Get<int>(test::tag::name()).data);
    auto key = ToHost(lw.Get<std::uint64_t>(test::key::name()).data);
    auto id = ToHost(lw.Get<std::uint64_t>(lukewarm::id::name()).data);
    for (int n = 0; n < lw.NumParticles(); ++n) {
      const int b = block(Lukewarm::block_row, n);
      REQUIRE(b >= 0);
      REQUIRE(b < md->NumBlocks());
      const auto &size = md->GetBlockData(b)->GetBlockPointer()->block_size;
      REQUIRE(size.xmin(X1DIR) <= x(0, n));
      REQUIRE(x(0, n) < size.xmax(X1DIR));
      REQUIRE(size.xmin(X2DIR) <= y(0, n));
      REQUIRE(y(0, n) < size.xmax(X2DIR));
      for (int c = 0; c < 2; ++c) {
        REQUIRE(weight(c, n) == Weight(c, id(0, n)));
        REQUIRE(tag(c, n) == Tag(c, id(0, n)));
        REQUIRE(key(c, n) == Key(c, id(0, n)));
      }
      ids.push_back(id(0, n));
      xs.push_back(x(0, n));
      ys.push_back(y(0, n));
    }
  }
  ids = AllGather(ids);
  xs = AllGather(xs);
  ys = AllGather(ys);
  std::map<std::uint64_t, std::pair<Real, Real>> where;
  for (int n = 0; n < ids.size(); ++n) {
    REQUIRE(where.count(ids[n]) == 0);
    where[ids[n]] = {xs[n], ys[n]};
  }
  return where;
}

Packages_t MakePackages() {
  Packages_t packages;
  auto pkg = std::make_shared<StateDescriptor>("provider");
  pkg->AddLukewarm("tracers", Metadata({Metadata::Provides}));
  pkg->AddLukewarmField<test::weight>("tracers",
                                      Metadata({Metadata::Real}, std::vector<int>{2}));
  pkg->AddLukewarmField<test::tag>("tracers",
                                   Metadata({Metadata::Integer}, std::vector<int>{2}));
  pkg->AddLukewarmField<test::key>("tracers",
                                   Metadata({Metadata::UInt64}, std::vector<int>{2}));
  packages.Add(pkg);
  return packages;
}

Real Wrap(const Real x) { return x - std::floor(x); }

} // namespace

TEST_CASE("Declaring lukewarms", "[Lukewarm]") {
  StateDescriptor pkg("pkg");
  REQUIRE(pkg.AddLukewarm("tracers", Metadata()));
  REQUIRE_FALSE(pkg.AddLukewarm("tracers", Metadata()));
  REQUIRE(pkg.LukewarmPresent("tracers"));
  REQUIRE_FALSE(pkg.LukewarmPresent("other_tracers"));
  REQUIRE_THROWS(pkg.AddLukewarm("bad name", Metadata()));

  SECTION("every lukewarm carries positions and an id") {
    const auto &fields = pkg.AllLukewarmFields("tracers");
    REQUIRE(fields.size() == 4);
    REQUIRE(fields.at(lukewarm::x::name()).Type() == Metadata::Real);
    REQUIRE(fields.at(lukewarm::y::name()).Type() == Metadata::Real);
    REQUIRE(fields.at(lukewarm::z::name()).Type() == Metadata::Real);
    REQUIRE(fields.at(lukewarm::id::name()).Type() == Metadata::UInt64);
    for (const auto &[name, m] : fields) {
      REQUIRE(m.IsSet(Metadata::Particle));
    }
  }

  SECTION("fields") {
    REQUIRE(pkg.AddLukewarmField<test::weight>("tracers", Metadata({Metadata::Real})));
    REQUIRE_FALSE(
        pkg.AddLukewarmField<test::weight>("tracers", Metadata({Metadata::Real})));
    REQUIRE(pkg.AddLukewarmField<test::tag>(
        "tracers", Metadata({Metadata::Integer}, std::vector<int>{3})));
    REQUIRE(pkg.LukewarmFieldPresent("tracers", test::tag::name()));
    REQUIRE(pkg.AllLukewarmFields("tracers")
                .at(test::weight::name())
                .IsSet(Metadata::Particle));
    REQUIRE_THROWS(pkg.AddLukewarmField("other_tracers", "weight", Metadata()));
    REQUIRE_THROWS(
        pkg.AddLukewarmField("tracers", "flag", Metadata({Metadata::Boolean})));
  }
}

TEST_CASE("Resolving lukewarms across packages", "[Lukewarm]") {
  Packages_t packages;
  auto provider = std::make_shared<StateDescriptor>("provider");
  provider->AddLukewarm("tracers", Metadata({Metadata::Provides}));
  provider->AddLukewarmField<test::weight>("tracers", Metadata({Metadata::Real}));
  provider->AddLukewarm("scratch", Metadata({Metadata::Private}));
  packages.Add(provider);

  auto user = std::make_shared<StateDescriptor>("user");
  user->AddLukewarm("tracers", Metadata({Metadata::Requires}));
  packages.Add(user);

  auto resolved = StateDescriptor::CreateResolvedStateDescriptor(packages);
  REQUIRE(resolved->LukewarmPresent("tracers"));
  REQUIRE(resolved->LukewarmFieldPresent("tracers", test::weight::name()));
  REQUIRE(resolved->LukewarmFieldPresent("tracers", lukewarm::id::name()));
  REQUIRE(resolved->LukewarmPresent("provider::scratch"));
  REQUIRE_FALSE(resolved->LukewarmPresent("scratch"));
  REQUIRE(resolved->Lukewarms().size() == 2);

  SECTION("a required lukewarm must be provided") {
    Packages_t lonely;
    auto pkg = std::make_shared<StateDescriptor>("lonely");
    pkg->AddLukewarm("tracers", Metadata({Metadata::Requires}));
    lonely.Add(pkg);
    REQUIRE_THROWS(StateDescriptor::CreateResolvedStateDescriptor(lonely));
  }
}

TEST_CASE("Lukewarm storage", "[Lukewarm]") {
  Lukewarm lw("tracers", Metadata());
  lw.Add(lukewarm::x::name(), Metadata({Metadata::Real}));
  lw.Add(test::weight::name(), Metadata({Metadata::Real}, std::vector<int>{2}));
  lw.Add(test::tag::name(), Metadata({Metadata::Integer}));
  lw.Add(lukewarm::id::name(), Metadata({Metadata::UInt64}));
  REQUIRE_THROWS(lw.Add(test::weight::name(), Metadata({Metadata::Real})));

  REQUIRE(lw.NumParticles() == 0);
  REQUIRE(lw.Capacity() == 0);
  REQUIRE(lw.Contains(test::weight::name()));
  REQUIRE(lw.GetFieldNames().size() == 4);
  REQUIRE(lw.NumComponents<Real>() == 3);
  // Each particle's block is an int row of its own, ahead of every field
  REQUIRE(lw.NumComponents<int>() == 2);
  REQUIRE(lw.Get<int>(test::tag::name()).GetOffset() == Lukewarm::block_row + 1);
  REQUIRE(lw.NumComponents<std::uint64_t>() == 1);

  // Fields of one type are packed in the order they were added
  REQUIRE(lw.Get<Real>(lukewarm::x::name()).GetOffset() == 0);
  REQUIRE(lw.Get<Real>(test::weight::name()).GetOffset() == 1);
  REQUIRE(lw.Get<Real>(test::weight::name()).NumComponents() == 2);
  REQUIRE(lw.GetVariables<Real>().size() == 2);
  REQUIRE_THROWS(lw.Get<int>(test::weight::name()));
  REQUIRE_THROWS(lw.Get<Real>("missing"));

  lw.Reserve(5);
  REQUIRE(lw.Capacity() == 5);
  REQUIRE(lw.GetData<Real>().extent_int(0) == 3);
  REQUIRE(lw.GetData<Real>().extent_int(1) == 5);
  auto &weight = lw.Get<Real>(test::weight::name());
  REQUIRE(weight.data.extent_int(0) == 2);
  REQUIRE(weight.data.extent_int(1) == 5);

  Fill(weight, [](int c, int n) { return 10.0 * c + n; });
  Fill(lw.Get<int>(test::tag::name()), [](int c, int n) { return 7 * n; });

  SECTION("a field is a view of the shared storage") {
    auto all = ToHost(lw.GetData<Real>());
    REQUIRE(all(1, 3) == 3.0);
    REQUIRE(all(2, 4) == 14.0);
  }

  SECTION("growing keeps existing values") {
    lw.Reserve(3); // never shrinks
    REQUIRE(lw.Capacity() == 5);
    lw.Reserve(16);
    REQUIRE(lw.Capacity() == 16);
    auto e = ToHost(lw.Get<Real>(test::weight::name()).data);
    REQUIRE(e.extent_int(1) == 16);
    REQUIRE(e(0, 4) == 4.0);
    REQUIRE(e(1, 2) == 12.0);
    auto s = ToHost(lw.Get<int>(test::tag::name()).data);
    REQUIRE(s(0, 4) == 28);
  }

  SECTION("adding a field keeps existing values") {
    lw.Add("weight", Metadata({Metadata::Real}));
    REQUIRE(lw.NumComponents<Real>() == 4);
    REQUIRE(lw.Get<Real>("weight").GetOffset() == 3);
    auto e = ToHost(lw.Get<Real>(test::weight::name()).data);
    REQUIRE(e(1, 4) == 14.0);
    REQUIRE(lw.Get<Real>("weight").data.extent_int(1) == 5);
  }
}

TEST_CASE("Lukewarms are owned by MeshData", "[Lukewarm][MPI]") {
  Packages_t packages;
  auto pkg = std::make_shared<StateDescriptor>("provider");
  pkg->AddLukewarm("tracers", Metadata({Metadata::Provides}));
  pkg->AddLukewarmField<test::weight>("tracers", Metadata({Metadata::Real}));
  pkg->AddLukewarm("other_tracers", Metadata({Metadata::Provides}));
  packages.Add(pkg);

  // 4 blocks, in partitions of 2 or all together
  const int pack_size = GENERATE(-1, 1, 2);
  auto pin = MakePin(pack_size);
  ApplicationInput app_in;
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  REQUIRE(mesh->nbtotal == 4);

  mesh->BuildLukewarms();
  const auto &partitions = mesh->GetDefaultBlockPartitions();
  const int nparts = partitions.size();
  REQUIRE(nparts == mesh->DefaultNumPartitions());

  std::vector<Lukewarm *> seen;
  for (auto &partition : partitions) {
    auto &md = mesh->mesh_data.Add("base", partition);
    REQUIRE(md->HasLukewarms());
    auto &container = md->GetLukewarmContainer();
    REQUIRE(container.GetLukewarms().size() == 2);
    auto &tracers = md->GetLukewarm("tracers");
    REQUIRE(tracers.Contains(test::weight::name()));
    REQUIRE(tracers.Contains(lukewarm::x::name()));
    REQUIRE(tracers.Contains(lukewarm::id::name()));
    REQUIRE(tracers.NumParticles() == 0);
    // x, y, z, and id. The block is internal, so it is not a field.
    REQUIRE(md->GetLukewarm("other_tracers").GetFieldNames().size() == 4);
    REQUIRE_THROWS(md->GetLukewarm("missing"));
    seen.push_back(&tracers);
  }
  // Every partition owns its own particles
  for (int i = 0; i < seen.size(); ++i) {
    for (int j = i + 1; j < seen.size(); ++j) {
      REQUIRE(seen[i] != seen[j]);
    }
  }

  // Ids are unique across partitions and ranks
  for (auto *lw : seen) {
    lw->Create(3);
  }
  std::vector<std::uint64_t> ids;
  for (auto *lw : seen) {
    auto id = ToHost(lw->Get<std::uint64_t>(lukewarm::id::name()).data);
    for (int n = 0; n < lw->NumParticles(); ++n) {
      ids.push_back(id(0, n));
    }
  }
  ids = AllGather(ids);
  std::sort(ids.begin(), ids.end());
  REQUIRE(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
  REQUIRE(ids.size() == 3 * SumOverRanks(nparts));

  // Rebuilding leaves existing lukewarms alone
  auto &md0 = mesh->mesh_data.Add("base", partitions[0]);
  md0->GetLukewarm("tracers").Reserve(8);
  mesh->BuildLukewarms();
  REQUIRE(&md0->GetLukewarm("tracers") == seen[0]);
  REQUIRE(md0->GetLukewarm("tracers").Capacity() == 8);

  // Other stages do not own lukewarms
  auto &stage = mesh->mesh_data.Add("stage", md0);
  REQUIRE_FALSE(stage->HasLukewarms());
  REQUIRE_THROWS(stage->GetLukewarm("tracers"));
}

TEST_CASE("Creating and removing lukewarm particles", "[Lukewarm]") {
  auto next_id = std::make_shared<std::uint64_t>(100);
  Lukewarm lw("tracers", Metadata(), next_id);
  lw.Add(lukewarm::id::name(), Metadata({Metadata::UInt64}));
  lw.Add(test::weight::name(), Metadata({Metadata::Real}, std::vector<int>{2}));
  lw.Add(test::tag::name(), Metadata({Metadata::Integer}));

  auto range = lw.Create(3);
  REQUIRE(range.s == 0);
  REQUIRE(range.e == 2);
  REQUIRE(lw.NumParticles() == 3);
  REQUIRE(lw.Capacity() >= 3);
  REQUIRE(*next_id == 103);

  // Capacity grows geometrically, so repeated small emissions rarely reallocate
  range = lw.Create(1);
  REQUIRE(range.s == 3);
  REQUIRE(lw.Capacity() >= 6);
  REQUIRE(lw.Create(0).size() == 0);
  REQUIRE_THROWS(lw.Create(-1));

  SECTION("creating from per-item counts") {
    std::vector<int> counts_h = {2, 0, 3, 1};
    ParArray1DRaw<int> counts("counts", counts_h.size());
    auto counts_m = Kokkos::create_mirror_view(counts);
    for (int i = 0; i < counts_h.size(); ++i) {
      counts_m(i) = counts_h[i];
    }
    Kokkos::deep_copy(counts, counts_m);
    auto offsets = ToHost(lw.Create(counts));
    REQUIRE(offsets.extent_int(0) == 5);
    const std::vector<int> expected = {4, 6, 6, 9, 10};
    for (int i = 0; i < expected.size(); ++i) {
      REQUIRE(offsets(i) == expected[i]);
    }
    REQUIRE(lw.NumParticles() == 10);
  }

  SECTION("ids are unique and sequential") {
    // Ranks take turns, so ids are unique across ranks without communication
    lw.Create(5);
    auto id = ToHost(lw.Get<std::uint64_t>(lukewarm::id::name()).data);
    const std::uint64_t nranks = parthenon::Globals::nranks;
    const std::uint64_t rank = parthenon::Globals::my_rank;
    for (int n = 0; n < lw.NumParticles(); ++n) {
      REQUIRE(id(0, n) == (100 + n) * nranks + rank);
    }
  }

  SECTION("removal keeps storage dense") {
    lw.Create(6); // 10 particles
    Fill(lw.Get<Real>(test::weight::name()), [](int c, int n) { return 100.0 * c + n; });
    Fill(lw.Get<int>(test::tag::name()), [](int c, int n) { return -n; });

    const std::vector<int> doomed = {0, 3, 4, 9};
    auto marks = lw.GetRemovalMarks();
    auto marks_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), marks);
    for (const int n : doomed) {
      marks_m(n) = 1;
    }
    Kokkos::deep_copy(marks, marks_m);
    lw.RemoveMarked();
    REQUIRE(lw.NumParticles() == 6);

    // Every survivor appears once and its fields still agree with each other
    auto weight = ToHost(lw.Get<Real>(test::weight::name()).data);
    auto tag = ToHost(lw.Get<int>(test::tag::name()).data);
    auto id = ToHost(lw.Get<std::uint64_t>(lukewarm::id::name()).data);
    std::vector<int> survivors;
    for (int n = 0; n < lw.NumParticles(); ++n) {
      const int original = -tag(0, n);
      survivors.push_back(original);
      REQUIRE(weight(0, n) == original);
      REQUIRE(weight(1, n) == 100.0 + original);
      REQUIRE(id(0, n) == (100 + original) * parthenon::Globals::nranks +
                              parthenon::Globals::my_rank);
    }
    std::sort(survivors.begin(), survivors.end());
    REQUIRE(survivors == std::vector<int>{1, 2, 5, 6, 7, 8});

    // Marks are cleared, so a second pass removes nothing
    auto marks_after = ToHost(lw.GetRemovalMarks());
    for (int n = 0; n < lw.Capacity(); ++n) {
      REQUIRE(marks_after(n) == 0);
    }
    lw.RemoveMarked();
    REQUIRE(lw.NumParticles() == 6);
  }

  SECTION("removing everything") {
    auto marks = lw.GetRemovalMarks();
    Kokkos::deep_copy(marks, 1);
    lw.RemoveMarked();
    REQUIRE(lw.NumParticles() == 0);
    REQUIRE(lw.Create(2).s == 0);
  }
}

TEST_CASE("Packing lukewarm fields for device access", "[Lukewarm][MPI]") {
  Packages_t packages;
  auto pkg = std::make_shared<StateDescriptor>("provider");
  pkg->AddLukewarm("tracers", Metadata({Metadata::Provides}));
  pkg->AddLukewarmField<test::weight>("tracers",
                                      Metadata({Metadata::Real}, std::vector<int>{2}));
  pkg->AddLukewarmField<test::tag>("tracers", Metadata({Metadata::Integer}));
  packages.Add(pkg);

  auto pin = MakePin(2);
  ApplicationInput app_in;
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  mesh->BuildLukewarms();
  auto psd = mesh->resolved_packages.get();

  SECTION("descriptors check fields and types up front") {
    REQUIRE_THROWS(MakeLukewarmPackDescriptor<test::weight>(psd, "missing"));
    REQUIRE_THROWS(MakeLukewarmPackDescriptor<wrong::missing>(psd, "tracers"));
    REQUIRE_THROWS(MakeLukewarmPackDescriptor<wrong::tag>(psd, "tracers"));
  }

  SECTION("kernels read and write fields and mark removals") {
    auto desc =
        MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::id, test::weight, test::tag>(
            psd, "tracers");
    auto &md = mesh->mesh_data.Add("base", mesh->GetDefaultBlockPartitions()[0]);
    auto &tracers = md->GetLukewarm("tracers");
    tracers.Create(8);

    auto pack = desc.GetPack(md.get());
    REQUIRE(pack.GetNParticles() == 8);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, pack.GetNParticles() - 1, KOKKOS_LAMBDA(const int n) {
          pack(lukewarm::x(), n) = 0.5 * n;
          pack(test::weight(0), n) = 1.0 * pack(lukewarm::id(), n);
          pack(test::weight(1), n) = -1.0;
          pack(test::tag(), n) = n % 3;
          if (n % 2 == 1) pack.MarkForRemoval(n);
        });

    auto x = ToHost(tracers.Get<Real>(lukewarm::x::name()).data);
    auto e = ToHost(tracers.Get<Real>(test::weight::name()).data);
    auto s = ToHost(tracers.Get<int>(test::tag::name()).data);
    auto id = ToHost(tracers.Get<std::uint64_t>(lukewarm::id::name()).data);
    for (int n = 0; n < 8; ++n) {
      REQUIRE(x(0, n) == 0.5 * n);
      REQUIRE(e(0, n) == static_cast<Real>(id(0, n)));
      REQUIRE(e(1, n) == -1.0);
      REQUIRE(s(0, n) == n % 3);
    }

    // A later kernel sees the marked particles until they are removed, and can skip them
    REQUIRE(tracers.NumParticles() == 8);
    int unmarked = 0;
    parthenon::par_reduce(
        PARTHENON_AUTO_LABEL, 0, pack.GetNParticles() - 1,
        KOKKOS_LAMBDA(const int n, int &sum) {
          if (!pack.IsMarkedForRemoval(n)) sum += 1;
        },
        Kokkos::Sum<int>(unmarked));
    REQUIRE(unmarked == 4);

    tracers.RemoveMarked();
    REQUIRE(tracers.NumParticles() == 4);
    pack = desc.GetPack(md.get());
    REQUIRE(pack.GetNParticles() == 4);
    int odd = 0;
    int marked = 0;
    parthenon::par_reduce(
        PARTHENON_AUTO_LABEL, 0, pack.GetNParticles() - 1,
        KOKKOS_LAMBDA(const int n, int &sum, int &nmarked) {
          sum += static_cast<int>(2.0 * pack(lukewarm::x(), n)) % 2;
          nmarked += pack.IsMarkedForRemoval(n);
        },
        Kokkos::Sum<int>(odd), Kokkos::Sum<int>(marked));
    REQUIRE(odd == 0);
    REQUIRE(marked == 0);
  }
}

TEST_CASE("Relocating lukewarm particles", "[Lukewarm][MPI]") {
  const int pack_size = GENERATE(-1, 1, 2);
  const bool refined = GENERATE(false, true);
  auto pin = MakePin(pack_size, true, refined);
  ApplicationInput app_in;
  auto packages = MakePackages();
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  REQUIRE(mesh->nbtotal == (refined ? 7 : 4));
  mesh->BuildLukewarms();
  const auto mds = LukewarmMeshData(mesh.get());

  // All the particles start on the first MeshData of rank 0, scattered over the mesh and
  // its periodic images, some several mesh lengths away
  std::vector<std::pair<Real, Real>> positions;
  for (int k = 0; k < 40; ++k) {
    positions.push_back({-1.3 + 0.091 * k, 2.1 - 0.137 * k});
  }
  positions.push_back({0.5, 0.5});   // on a corner shared by four blocks
  positions.push_back({0.0, 0.999}); // on the lower face of the mesh
  if (parthenon::Globals::my_rank == 0)
    CreateAt(mds[0]->GetLukewarm("tracers"), positions);

  RelocateLukewarm(mesh.get(), "tracers");
  auto where = CheckSettled(mesh.get());
  REQUIRE(where.size() == positions.size());
  for (int k = 0; k < positions.size(); ++k) {
    REQUIRE(where[RootId(k)].first == Approx(Wrap(positions[k].first)));
    REQUIRE(where[RootId(k)].second == Approx(Wrap(positions[k].second)));
  }

  SECTION("Settled particles stay put") {
    int nleaving = 0;
    for (auto *md : mds) {
      nleaving += LocateLukewarm(md, "tracers");
    }
    REQUIRE(SumOverRanks(nleaving) == 0);
    REQUIRE(CheckSettled(mesh.get()) == where);
  }

  SECTION("The particles received in a round are the last ones stored") {
    // RelocateLukewarm() ends by locating particles that are all settled
    for (auto *md : mds) {
      REQUIRE(GetLastReceived(md, "tracers").size() == 0);
    }

    std::vector<std::set<std::uint64_t>> before(mds.size());
    static auto desc =
        MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y>(mds[0], "tracers");
    for (int p = 0; p < mds.size(); ++p) {
      auto id = ToHost(
          mds[p]->GetLukewarm("tracers").Get<std::uint64_t>(lukewarm::id::name()).data);
      for (int n = 0; n < mds[p]->GetLukewarm("tracers").NumParticles(); ++n) {
        before[p].insert(id(0, n));
      }
      auto pack = desc.GetPack(mds[p]);
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "move", parthenon::DevExecSpace(), 0,
          pack.GetNParticles() - 1, KOKKOS_LAMBDA(const int n) {
            pack(lukewarm::x(), n) += 0.61;
            pack(lukewarm::y(), n) -= 0.83;
          });
    }

    // One round, by hand
    int nleaving = 0;
    for (auto *md : mds) {
      nleaving += LocateLukewarm(md, "tracers");
    }
    for (auto *md : mds) {
      StartReceiveLukewarm(md, "tracers");
    }
    for (auto *md : mds) {
      while (SendLukewarm(md, "tracers") == parthenon::TaskStatus::incomplete) {
      }
    }
    for (auto *md : mds) {
      while (ReceiveLukewarm(md, "tracers") == parthenon::TaskStatus::incomplete) {
      }
    }

    // A particle is in the received range exactly when it was not on this MeshData
    // before the round
    int nreceived = 0;
    for (int p = 0; p < mds.size(); ++p) {
      auto &lw = mds[p]->GetLukewarm("tracers");
      const auto range = GetLastReceived(mds[p], "tracers");
      if (range.size() > 0) REQUIRE(range.e == lw.NumParticles() - 1);
      nreceived += range.size();
      auto id = ToHost(lw.Get<std::uint64_t>(lukewarm::id::name()).data);
      for (int n = 0; n < lw.NumParticles(); ++n) {
        const bool in_range = range.s <= n && n <= range.e;
        REQUIRE(in_range == (before[p].count(id(0, n)) == 0));
      }
    }
    REQUIRE(SumOverRanks(nreceived) == SumOverRanks(nleaving));
    if (pack_size != -1 || parthenon::Globals::nranks > 1) {
      REQUIRE(SumOverRanks(nleaving) > 0);
    }

    // Creating particles leaves the range in place
    for (auto *md : mds) {
      auto &lw = md->GetLukewarm("tracers");
      const auto range = GetLastReceived(md, "tracers");
      const auto created = lw.Create(3);
      REQUIRE(GetLastReceived(md, "tracers").s == range.s);
      REQUIRE(GetLastReceived(md, "tracers").e == range.e);
      auto marks = lw.GetRemovalMarks();
      parthenon::par_for(
          PARTHENON_AUTO_LABEL, created.s, created.e,
          KOKKOS_LAMBDA(const int n) { marks(n) = 1; });
      lw.RemoveMarked();
    }

    // The next round starts afresh
    for (auto *md : mds) {
      LocateLukewarm(md, "tracers");
      REQUIRE(GetLastReceived(md, "tracers").size() == 0);
    }
  }

  SECTION("Removing received particles invalidates the range") {
    if (parthenon::Globals::my_rank == 0) {
      auto *md = mds[0];
      auto &lw = md->GetLukewarm("tracers");
      // Fake a delivery: received ranges are only set by ReceiveLukewarm()
      auto &comms = md->GetLukewarmContainer().GetComms("tracers");
      comms.received = lw.Append(2);
      REQUIRE(GetLastReceived(md, "tracers").size() == 2);
      auto marks = lw.GetRemovalMarks();
      const int last = lw.NumParticles() - 1;
      parthenon::par_for(
          PARTHENON_AUTO_LABEL, 0, 0, KOKKOS_LAMBDA(const int) { marks(last) = 1; });
      lw.RemoveMarked();
      REQUIRE_THROWS(GetLastReceived(md, "tracers"));
    }
  }

  SECTION("Particles follow their motion from block to block in one kernel") {
    // Move every particle across a block and then some, and locate it on device as a
    // kernel that pushes particles would
    std::map<std::uint64_t, std::pair<Real, Real>> expected;
    for (const auto &[id, p] : where) {
      expected[id] = {Wrap(p.first + 0.61), Wrap(p.second - 0.83)};
    }
    static auto desc =
        MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y>(mds[0], "tracers");
    int nsettled = 0;
    for (auto *md : mds) {
      auto pack = desc.GetPack(md);
      int settled = 0;
      parthenon::par_reduce(
          parthenon::loop_pattern_flatrange_tag, "move", parthenon::DevExecSpace(), 0,
          pack.GetNParticles() - 1,
          KOKKOS_LAMBDA(const int n, int &sum) {
            pack(lukewarm::x(), n) += 0.61;
            pack(lukewarm::y(), n) -= 0.83;
            sum += pack.Locate(n).InMeshData();
          },
          Kokkos::Sum<int>(settled));
      nsettled += settled;
    }
    // With a single MeshData, nothing ever has to be exchanged
    if (pack_size == -1 && parthenon::Globals::nranks == 1) {
      REQUIRE(nsettled == where.size());
    }

    RelocateLukewarm(mesh.get(), "tracers");
    auto moved = CheckSettled(mesh.get());
    REQUIRE(moved.size() == expected.size());
    for (const auto &[id, p] : expected) {
      REQUIRE(moved[id].first == Approx(p.first));
      REQUIRE(moved[id].second == Approx(p.second));
    }
  }

  SECTION("Particles know their block and cell") {
    static auto desc = MakeLukewarmPackDescriptor<lukewarm::x>(mds[0], "tracers");
    for (auto *md : mds) {
      auto pack = desc.GetPack(md);
      const int np = pack.GetNParticles();
      ParArray1DRaw<int> located("located", np), cached("cached", np);
      ParArray1DRaw<int> ci("i", np), cj("j", np), ck("k", np);
      parthenon::par_for(
          parthenon::loop_pattern_flatrange_tag, "cells", parthenon::DevExecSpace(), 0,
          np - 1, KOKKOS_LAMBDA(const int n) {
            cached(n) = pack.GetBlockIndex(n);
            located(n) = pack.Locate(n).Block();
            const auto [k, j, i] = pack.GetCellIndices(located(n), n);
            ck(n) = k;
            cj(n) = j;
            ci(n) = i;
          });
      auto x = ToHost(md->GetLukewarm("tracers").Get<Real>(lukewarm::x::name()).data);
      auto y = ToHost(md->GetLukewarm("tracers").Get<Real>(lukewarm::y::name()).data);
      auto located_h = ToHost(located), cached_h = ToHost(cached);
      auto i_h = ToHost(ci), j_h = ToHost(cj), k_h = ToHost(ck);
      for (int n = 0; n < np; ++n) {
        REQUIRE(located_h(n) == cached_h(n));
        const auto &coords = md->GetBlockData(located_h(n))->GetBlockPointer()->coords;
        REQUIRE(coords.Xf<1>(i_h(n)) <= x(0, n));
        REQUIRE(x(0, n) < coords.Xf<1>(i_h(n) + 1));
        REQUIRE(coords.Xf<2>(j_h(n)) <= y(0, n));
        REQUIRE(y(0, n) < coords.Xf<2>(j_h(n) + 1));
        REQUIRE(k_h(n) == 0);
      }
    }
  }

  SECTION("Lost particles are an error") {
    // On every rank, since a rank that finds none goes on to wait for the others
    CreateAt(mds.back()->GetLukewarm("tracers"), {{std::nan(""), 0.5}});
    REQUIRE_THROWS(RelocateLukewarm(mesh.get(), "tracers"));
  }
}

TEST_CASE("Lukewarm particles leave through non-periodic boundaries", "[Lukewarm][MPI]") {
  const int pack_size = GENERATE(-1, 1);
  auto pin = MakePin(pack_size, false);
  ApplicationInput app_in;
  auto packages = MakePackages();
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  mesh->BuildLukewarms();
  const auto mds = LukewarmMeshData(mesh.get());

  if (parthenon::Globals::my_rank == 0) {
    CreateAt(
        mds.back()->GetLukewarm("tracers"),
        {{0.25, 0.25}, {-0.1, 0.5}, {1.2, 1.7}, {0.75, -0.6}, {1.0, 0.5}, {0.0, 0.5}});
  }
  RelocateLukewarm(mesh.get(), "tracers");
  auto where = CheckSettled(mesh.get());
  REQUIRE(where.size() == 3);
  REQUIRE(where.count(RootId(0)) == 1);
  REQUIRE(where.count(RootId(3)) == 1);
  REQUIRE(where.count(RootId(5)) == 1);
  REQUIRE(where[RootId(3)].second == Approx(0.4));
}

TEST_CASE("New lukewarm particles that move out of their MeshData are delivered in one "
          "round",
          "[Lukewarm][MPI]") {
  const int pack_size = GENERATE(-1, 4);
  auto pin = MakePin(pack_size);
  // A 4x4 grid of blocks, so that a MeshData can have blocks far from a neighbor
  pin->SetInteger("parthenon/mesh", "nx1", 16);
  pin->SetInteger("parthenon/mesh", "nx2", 16);
  ApplicationInput app_in;
  auto packages = MakePackages();
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  REQUIRE(mesh->nbtotal == 16);
  mesh->BuildLukewarms();
  const auto mds = LukewarmMeshData(mesh.get());

  // Every block creates particles a quarter block beyond each of its faces and corners,
  // some across periodic boundaries, and moves on before they are ever located. Their
  // coordinates are multiples of 1/16, so they wrap exactly.
  for (auto *md : mds) {
    std::vector<std::pair<Real, Real>> positions;
    for (int b = 0; b < md->NumBlocks(); ++b) {
      const auto &size = md->GetBlockData(b)->GetBlockPointer()->block_size;
      const Real cx = 0.5 * (size.xmin(X1DIR) + size.xmax(X1DIR));
      const Real cy = 0.5 * (size.xmin(X2DIR) + size.xmax(X2DIR));
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx != 0 || dy != 0)
            positions.push_back({cx + dx * 0.1875, cy + dy * 0.1875});
        }
      }
    }
    CreateAt(md->GetLukewarm("tracers"), positions);
  }

  // One round, as AddLukewarmExchangeTasks() does it
  for (auto *md : mds) {
    LocateLukewarm(md, "tracers");
  }
  for (auto *md : mds) {
    StartReceiveLukewarm(md, "tracers");
  }
  for (auto *md : mds) {
    while (SendLukewarm(md, "tracers") == parthenon::TaskStatus::incomplete) {
    }
  }
  for (auto *md : mds) {
    while (ReceiveLukewarm(md, "tracers") == parthenon::TaskStatus::incomplete) {
    }
  }

  std::vector<std::pair<Real, Real>> found;
  for (const auto &[id, position] : CheckSettled(mesh.get())) {
    found.push_back(position);
  }
  std::vector<std::pair<Real, Real>> expected;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) continue;
          expected.push_back({Wrap(0.25 * i + 0.125 + dx * 0.1875),
                              Wrap(0.25 * j + 0.125 + dy * 0.1875)});
        }
      }
    }
  }
  std::sort(found.begin(), found.end());
  std::sort(expected.begin(), expected.end());
  REQUIRE(found == expected);
}

TEST_CASE("Lukewarm geometry", "[Lukewarm][MPI]") {
  auto pin = MakePin(-1, true, true);
  ApplicationInput app_in;
  auto packages = MakePackages();
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  mesh->BuildLukewarms();
  auto *md = LukewarmMeshData(mesh.get())[0];
  const auto geometry = md->GetLukewarmGeometry();
  const int nblocks = md->NumBlocks();
  REQUIRE(geometry.NumBlocks() == nblocks);
  REQUIRE(SumOverRanks(nblocks) == 7);

  // Walk from every block to points across the mesh and check against a direct search
  std::vector<std::pair<Real, Real>> targets = {{0.05, 0.05}, {0.2, 0.1},  {0.3, 0.45},
                                                {0.6, 0.2},   {0.9, 0.95}, {0.1, 0.7},
                                                {0.26, 0.24}, {0.0, 0.0},  {0.499, 0.5}};
  const int nt = targets.size();
  ParArray1DRaw<Real> tx("tx", nt), ty("ty", nt);
  auto htx = Kokkos::create_mirror_view(tx);
  auto hty = Kokkos::create_mirror_view(ty);
  for (int t = 0; t < nt; ++t) {
    htx(t) = targets[t].first;
    hty(t) = targets[t].second;
  }
  Kokkos::deep_copy(tx, htx);
  Kokkos::deep_copy(ty, hty);
  // Points outside this MeshData, which there are only with several ranks, are left
  // for a neighbor
  int nwrong = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "walk", parthenon::DevExecSpace(), 0,
      nblocks - 1, 0, nt - 1,
      KOKKOS_LAMBDA(const int b, const int t, int &wrong) {
        Real x1 = tx(t), x2 = ty(t), x3 = 0.0;
        const auto loc = geometry.Locate(b, x1, x2, x3);
        const int expected = geometry.FindBlock(tx(t), ty(t), 0.0);
        if (expected >= 0) {
          wrong += !loc.InMeshData() || loc.Block() != expected ||
                   loc.Gid() != geometry.Gid(expected) || x1 != tx(t) || x2 != ty(t);
        } else {
          wrong += !loc.InOtherMeshData() || geometry.LocalIndex(loc.Gid()) >= 0 ||
                   geometry.NeighborIndex(loc.Gid()) < 0;
        }
      },
      Kokkos::Sum<int>(nwrong));
  REQUIRE(nwrong == 0);

  // Local and neighbor indices agree with the blocks of this MeshData
  ParArray1DRaw<int> indices("indices", nblocks);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1,
      KOKKOS_LAMBDA(const int b) { indices(b) = geometry.LocalIndex(geometry.Gid(b)); });
  auto indices_h = ToHost(indices);
  for (int b = 0; b < nblocks; ++b) {
    REQUIRE(indices_h(b) == b);
  }

  // Block faces match the block sizes
  ParArray2DRaw<Real> faces("faces", nblocks, 6);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1, KOKKOS_LAMBDA(const int b) {
        const CoordinateDirection dirs[3] = {X1DIR, X2DIR, X3DIR};
        for (int d = 0; d < 3; ++d) {
          faces(b, 2 * d) = geometry.Xmin(b, dirs[d]);
          faces(b, 2 * d + 1) = geometry.Xmax(b, dirs[d]);
        }
      });
  auto faces_h = ToHost(faces);
  for (int b = 0; b < nblocks; ++b) {
    const auto &block_size = md->GetBlockData(b)->GetBlockPointer()->block_size;
    const CoordinateDirection dirs[3] = {X1DIR, X2DIR, X3DIR};
    for (int d = 0; d < 3; ++d) {
      REQUIRE(faces_h(b, 2 * d) == block_size.xmin(dirs[d]));
      REQUIRE(faces_h(b, 2 * d + 1) == block_size.xmax(dirs[d]));
    }
  }
}

TEST_CASE("Lukewarm particles survive remeshing", "[Lukewarm][MPI]") {
  const int pack_size = GENERATE(-1, 1);
  auto pin = MakePin(pack_size, true, false, true);
  ApplicationInput app_in;
  auto packages = MakePackages();
  auto mesh = std::make_shared<Mesh>(pin.get(), &app_in, packages, 0);
  mesh->BuildLukewarms();
  REQUIRE(mesh->nbtotal == 4);

  // Particles in each daughter of the lower left block, on its midplanes, and elsewhere,
  // created unsettled in one MeshData
  const std::vector<std::pair<Real, Real>> positions = {
      {0.1, 0.1},  {0.4, 0.1},   {0.1, 0.4}, {0.4, 0.4}, {0.25, 0.25},
      {0.25, 0.1}, {0.75, 0.25}, {0.6, 0.9}, {0.2, 0.7}, {0.999, 0.001}};
  if (parthenon::Globals::my_rank == 0) {
    CreateAt(LukewarmMeshData(mesh.get()).back()->GetLukewarm("tracers"), positions);
  }
  // Made before the remesh and used after it, as a static descriptor would be
  const auto desc =
      MakeLukewarmPackDescriptor<lukewarm::x>(LukewarmMeshData(mesh.get())[0], "tracers");

  auto check = [&]() {
    const auto where = CheckSettled(mesh.get());
    REQUIRE(where.size() == positions.size());
    for (int k = 0; k < positions.size(); ++k) {
      REQUIRE(where.at(RootId(k)) == positions[k]);
    }
    // Packs read the new MeshData, and every particle is where it was put
    int nwrong = 0;
    for (auto *md : LukewarmMeshData(mesh.get())) {
      const auto pack = desc.GetPack(md);
      int wrong = 0;
      parthenon::par_reduce(
          PARTHENON_AUTO_LABEL, 0, pack.GetNParticles() - 1,
          KOKKOS_LAMBDA(const int n, int &sum) {
            const int b = pack.GetBlockIndex(n);
            const auto loc = pack.Locate(n);
            sum += !loc.InMeshData() || loc.Block() != b;
          },
          Kokkos::Sum<int>(wrong));
      nwrong += wrong;
    }
    REQUIRE(nwrong == 0);
    // The channels of the new MeshData work
    RelocateLukewarm(mesh.get(), "tracers");
    REQUIRE(CheckSettled(mesh.get()) == where);
  };

  // Refine the lower left block
  for (auto &pmb : mesh->block_list) {
    if (pmb->block_size.xmax(X1DIR) <= 0.5 && pmb->block_size.xmax(X2DIR) <= 0.5) {
      pmb->pmr->SetRefinement(AmrTag::refine);
    }
  }
  mesh->LoadBalancingAndAdaptiveMeshRefinement(pin.get(), &app_in);
  REQUIRE(mesh->nbtotal == 7);
  check();

  // And derefine it again
  for (auto &pmb : mesh->block_list) {
    pmb->pmr->SetRefinement(AmrTag::derefine);
  }
  mesh->LoadBalancingAndAdaptiveMeshRefinement(pin.get(), &app_in);
  REQUIRE(mesh->nbtotal == 4);
  check();
}
