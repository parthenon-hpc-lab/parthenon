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

#include "lukewarm/lukewarm.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/default_names.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

namespace {
// Copy every component of particle srcs(k) over particle dsts(k), for k in [0, n)
template <typename T>
void CopyParticles(const ParArray2DRaw<T> &data, const ParArray1DRaw<int> &srcs,
                   const ParArray1DRaw<int> &dsts, const int n) {
  const int ncomp = data.extent_int(0);
  if (ncomp == 0) return;
  par_for(
      PARTHENON_AUTO_LABEL, 0, ncomp - 1, 0, n - 1,
      KOKKOS_LAMBDA(const int c, const int k) { data(c, dsts(k)) = data(c, srcs(k)); });
}
} // namespace

Lukewarm::Lukewarm(const std::string &label, const Metadata &m, LukewarmIdCounter next_id)
    : label_(label), m_(m), next_id_(std::move(next_id)),
      marks_("Lukewarm::" + label + "::marks", 0) {
  PARTHENON_REQUIRE_THROWS(next_id_ != nullptr, "Lukewarm needs an id counter");
  // The block row comes before every field
  GetStorage_<int>().ncomp = block_row + 1;
  LukewarmTypes::IterateTypes([&](auto t) {
    using T = decltype(t);
    constexpr auto idx = LukewarmTypes::GetIdx<T>();
    auto &storage = GetStorage_<T>();
    storage.data = ParArray2DRaw<T>(
        "Lukewarm::" + label_ + "::data" + std::to_string(idx), storage.ncomp, 0);
  });
}

void Lukewarm::Add(const std::string &label, const Metadata &m) {
  PARTHENON_REQUIRE_THROWS(!Contains(label),
                           "Lukewarm " + label_ + " already has field " + label);
  const auto type = m.Type();
  if (type == Metadata::Real) {
    Add_<Real>(label, m);
  } else if (type == Metadata::Integer) {
    Add_<int>(label, m);
  } else if (type == Metadata::UInt64) {
    Add_<std::uint64_t>(label, m);
  } else {
    PARTHENON_THROW("Lukewarm field " + label + " must be Real, Integer, or UInt64");
  }
  types_.emplace(label, type);
}

template <typename T>
void Lukewarm::Add_(const std::string &label, const Metadata &m) {
  auto &storage = GetStorage_<T>();
  auto var = std::make_shared<LukewarmVariable<T>>(label, m, storage.ncomp);
  storage.index[label] = storage.vars.size();
  storage.vars.push_back(var);
  storage.ncomp += var->NumComponents();
  Reallocate_();
}

std::vector<std::string> Lukewarm::GetFieldNames() const {
  std::vector<std::string> names;
  names.reserve(types_.size());
  for (const auto &pair : types_) {
    names.push_back(pair.first);
  }
  return names;
}

void Lukewarm::Reserve(const int n) {
  if (n <= capacity_) return;
  capacity_ = n;
  Reallocate_();
}

IndexRange Lukewarm::Append(const int n) {
  PARTHENON_REQUIRE_THROWS(n >= 0, "Cannot append a negative number of particles");
  const IndexRange range{num_particles_, num_particles_ + n - 1};
  if (n == 0) return range;
  // Doubling keeps the cost of many small appends linear in the particle count
  const int needed = num_particles_ + n;
  if (needed > capacity_) Reserve(std::max(needed, 2 * capacity_));
  num_particles_ = needed;
  return range;
}

IndexRange Lukewarm::Create(const int n) {
  const IndexRange range = Append(n);
  if (n == 0) return range;

  Kokkos::deep_copy(
      Kokkos::subview(GetData<int>(), block_row, std::make_pair(range.s, range.e + 1)),
      -1);
  if (Contains(lukewarm::id::name())) {
    auto id = Get<std::uint64_t>(lukewarm::id::name()).data;
    const std::uint64_t first = *next_id_;
    const std::uint64_t nranks = Globals::nranks;
    const std::uint64_t rank = Globals::my_rank;
    const int s = range.s;
    par_for(
        PARTHENON_AUTO_LABEL, range.s, range.e,
        KOKKOS_LAMBDA(const int i) { id(0, i) = (first + (i - s)) * nranks + rank; });
  }
  *next_id_ += n;
  return range;
}

ParArray1DRaw<int> Lukewarm::Create(const ParArray1DRaw<int> &counts) {
  const int nitems = counts.extent_int(0);
  ParArray1DRaw<int> offsets("Lukewarm::" + label_ + "::create offsets", nitems + 1);
  const int start = num_particles_;
  int total = 0;
  Kokkos::parallel_scan(
      PARTHENON_AUTO_LABEL, Kokkos::RangePolicy<DevExecSpace>(0, nitems),
      KOKKOS_LAMBDA(const int i, int &update, const bool final) {
        if (final) offsets(i) = start + update;
        update += counts(i);
      },
      total);
  Kokkos::deep_copy(Kokkos::subview(offsets, nitems), start + total);
  Create(total);
  return offsets;
}

void Lukewarm::RemoveMarked() {
  const int n = num_particles_;
  if (n == 0) return;
  auto marks = marks_;
  int nremove = 0;
  par_reduce(
      PARTHENON_AUTO_LABEL, 0, n - 1,
      KOKKOS_LAMBDA(const int i, int &sum) { sum += marks(i) != 0; },
      Kokkos::Sum<int>(nremove));
  if (nremove == 0) return;

  // Survivors end up in [0, m). Holes are marked particles below m and movers are
  // unmarked particles at or above m. They are equal in number, and the k-th mover
  // fills the k-th hole.
  const int m = n - nremove;
  ParArray1DRaw<int> holes("Lukewarm::" + label_ + "::holes", nremove);
  ParArray1DRaw<int> movers("Lukewarm::" + label_ + "::movers", nremove);
  int nholes = 0;
  Kokkos::parallel_scan(
      PARTHENON_AUTO_LABEL, Kokkos::RangePolicy<DevExecSpace>(0, m),
      KOKKOS_LAMBDA(const int i, int &update, const bool final) {
        if (marks(i) == 0) return;
        if (final) holes(update) = i;
        update += 1;
      },
      nholes);
  if (nholes > 0) {
    Kokkos::parallel_scan(
        PARTHENON_AUTO_LABEL, Kokkos::RangePolicy<DevExecSpace>(m, n),
        KOKKOS_LAMBDA(const int i, int &update, const bool final) {
          if (marks(i) != 0) return;
          if (final) movers(update) = i;
          update += 1;
        });
    LukewarmTypes::IterateTypes([&](auto t) {
      using T = decltype(t);
      CopyParticles(GetStorage_<T>().data, movers, holes, nholes);
    });
  }
  Kokkos::deep_copy(Kokkos::subview(marks_, std::make_pair(0, n)), 0);
  num_particles_ = m;
}

void Lukewarm::Clear() {
  Kokkos::deep_copy(Kokkos::subview(marks_, std::make_pair(0, num_particles_)), 0);
  num_particles_ = 0;
}

void Lukewarm::Reallocate_() {
  if (marks_.extent_int(0) != capacity_) Kokkos::resize(marks_, capacity_);
  LukewarmTypes::IterateTypes([&](auto t) {
    using T = decltype(t);
    auto &storage = GetStorage_<T>();
    if (storage.data.extent_int(0) != storage.ncomp ||
        storage.data.extent_int(1) != capacity_) {
      Kokkos::resize(storage.data, storage.ncomp, capacity_);
    }
    for (auto &var : storage.vars) {
      var->data = Kokkos::subview(
          storage.data,
          std::make_pair(var->GetOffset(), var->GetOffset() + var->NumComponents()),
          Kokkos::ALL());
    }
  });
}

} // namespace parthenon
