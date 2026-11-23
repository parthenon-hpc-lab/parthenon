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

#ifndef TENSORS_TT_SERIALIZE_HPP_
#define TENSORS_TT_SERIALIZE_HPP_

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kokkos_abstraction.hpp"
#include "tensors/tt_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace tensor {

// Serialization of tensor trains into flat real_t buffers, for shipping trains between
// ranks. These routines know nothing about meshes or channels, so boundary comm (and
// eventually remesh) can share them.
//
// Wire format of one serialized train, starting at its offset in the buffer:
//
//   [ ncores, r_0, r_1, ..., r_ncores | core 0 | core 1 | ... ]
//
// The header holds integers stored exactly as real_t; r_c is the bond rank to the left
// of core c (so r_0 is the first core's left bond and r_ncores the last core's right
// bond). Each core c follows in fiber order: for l in [0, r_c), for r in [0, r_{c+1}),
// DD(c) contiguous values. Physical indexers are not sent; the receiver supplies them
// from its own field. A serialized train is self-describing given its expected core
// count, so payloads can sit back to back in one larger buffer.
//
// Every routine takes a list of (buffer, offset) targets and does all of its copying in
// a single batched kernel, so a whole set of trains costs one launch regardless of how
// many cores or fibers they have.

// A location in a flat buffer that a serialized train starts at.
template <class BufView>
struct SerialTarget {
  BufView buf;
  std::size_t offset{0};
};

// Number of real_t values in the header of a serialized ncores-core train.
inline std::size_t SerializedHeaderSize(int ncores) {
  return static_cast<std::size_t>(ncores) + 2;
}

// Total number of real_t values needed to serialize train.
template <class TTraits>
std::size_t SerializedSize(const TensorTrainT<TTraits> &train) {
  std::size_t size = SerializedHeaderSize(train.NCores());
  for (int c = 0; c < train.NCores(); ++c) {
    const auto &core = train(c);
    size += static_cast<std::size_t>(core.LR()) * core.RR() * core.DD();
  }
  return size;
}

namespace impl {

// One contiguous copy of n values from src to dst.
template <class real_t>
struct SerialCopy {
  real_t *src;
  real_t *dst;
  int n;
};

// Run every copy in entries in one launch: one team per copy, threads over its values.
// Fences before returning, so the caller may release any staging storage.
template <class TTraits>
void RunSerialCopies(
    const std::vector<SerialCopy<typename TTraits::real_t>> &entries_in) {
  using real_t = typename TTraits::real_t;
  using copy_t = SerialCopy<real_t>;
  const int ncopies = static_cast<int>(entries_in.size());
  if (ncopies == 0) return;
  Kokkos::View<copy_t *, typename TTraits::memory_space> entries("tt_serial_copies",
                                                                 ncopies);
  auto entries_h = Kokkos::create_mirror_view(entries);
  for (int e = 0; e < ncopies; ++e)
    entries_h(e) = entries_in[e];
  Kokkos::deep_copy(entries, entries_h);

  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0, ncopies - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int e) {
        const copy_t ce = entries(e);
        parthenon::par_for_inner(member, 0, ce.n - 1,
                                 [&](const int i) { ce.dst[i] = ce.src[i]; });
      });
  Kokkos::fence();
}

// Header integers must be exactly representable in real_t.
template <class real_t>
real_t HeaderValue(int value) {
  constexpr long long max_exact = 1LL << std::numeric_limits<real_t>::digits;
  PARTHENON_REQUIRE(value >= 0 && value < max_exact,
                    "TT serialization: header value not exactly representable.");
  return static_cast<real_t>(value);
}

// Append one copy per fiber of train, mapping between the fibers and the serialized core
// data starting at data. to_buffer selects the copy direction.
template <class TTraits>
void AddFiberCopies(const TensorTrainT<TTraits> &train, typename TTraits::real_t *data,
                    bool to_buffer,
                    std::vector<SerialCopy<typename TTraits::real_t>> &entries) {
  std::size_t offset = 0;
  for (int c = 0; c < train.NCores(); ++c) {
    const auto &core = train(c);
    for (int l = 0; l < core.LR(); ++l) {
      for (int r = 0; r < core.RR(); ++r) {
        auto *fiber = core.GetFiber(l, r).data();
        auto *wire = data + offset;
        if (to_buffer) {
          entries.push_back({fiber, wire, core.DD()});
        } else {
          entries.push_back({wire, fiber, core.DD()});
        }
        offset += core.DD();
      }
    }
  }
}

} // namespace impl

// Serialize trains[t] into targets[t]. Each target buffer must hold at least
// SerializedSize(*trains[t]) values past its offset.
template <class TTraits, class BufView>
void SerializeTrains(const std::vector<const TensorTrainT<TTraits> *> &trains,
                     const std::vector<SerialTarget<BufView>> &targets) {
  static_assert(TTraits::d_fastest_moving,
                "TT serialization is only implemented for fiber storage.");
  using real_t = typename TTraits::real_t;
  static_assert(std::is_same_v<std::remove_cv_t<typename BufView::value_type>, real_t>,
                "TT serialization buffers must hold the train's real_t.");
  PARTHENON_REQUIRE(trains.size() == targets.size(),
                    "SerializeTrains: need one target per train.");
  const int ntrains = static_cast<int>(trains.size());

  // Stage every header on the device so they ship in the same copy kernel as the data.
  std::size_t nheader = 0;
  for (const auto *train : trains)
    nheader += SerializedHeaderSize(train->NCores());
  Kokkos::View<real_t *, typename TTraits::memory_space> headers("tt_serial_headers",
                                                                 nheader);
  auto headers_h = Kokkos::create_mirror_view(headers);

  std::vector<impl::SerialCopy<real_t>> entries;
  std::size_t hoff = 0;
  for (int t = 0; t < ntrains; ++t) {
    const auto &train = *trains[t];
    const int ncores = train.NCores();
    PARTHENON_REQUIRE(targets[t].offset + SerializedSize(train) <=
                          static_cast<std::size_t>(targets[t].buf.size()),
                      "SerializeTrains: target buffer is too small.");
    headers_h(hoff) = impl::HeaderValue<real_t>(ncores);
    headers_h(hoff + 1) = impl::HeaderValue<real_t>(ncores > 0 ? train(0).LR() : 0);
    for (int c = 0; c < ncores; ++c)
      headers_h(hoff + 2 + c) = impl::HeaderValue<real_t>(train(c).RR());

    real_t *dst = targets[t].buf.data() + targets[t].offset;
    const int hsize = static_cast<int>(SerializedHeaderSize(ncores));
    entries.push_back({headers.data() + hoff, dst, hsize});
    impl::AddFiberCopies(train, dst + hsize, /*to_buffer=*/true, entries);
    hoff += hsize;
  }
  Kokkos::deep_copy(headers, headers_h);
  impl::RunSerialCopies<TTraits>(entries);
}

// Read the bond ranks (r_0, ..., r_ncores) of each serialized train in sources, which
// are expected to have ncores[t] cores. All headers come back in one device->host copy.
template <class TTraits, class BufView>
std::vector<std::vector<int>>
ReadSerializedBondRanks(const std::vector<SerialTarget<BufView>> &sources,
                        const std::vector<int> &ncores) {
  using real_t = typename TTraits::real_t;
  PARTHENON_REQUIRE(sources.size() == ncores.size(),
                    "ReadSerializedBondRanks: need one core count per source.");
  const int ntrains = static_cast<int>(sources.size());

  std::size_t nheader = 0;
  for (const int n : ncores)
    nheader += SerializedHeaderSize(n);
  Kokkos::View<real_t *, typename TTraits::memory_space> headers("tt_serial_headers",
                                                                 nheader);
  std::vector<impl::SerialCopy<real_t>> entries;
  std::size_t hoff = 0;
  for (int t = 0; t < ntrains; ++t) {
    const int hsize = static_cast<int>(SerializedHeaderSize(ncores[t]));
    PARTHENON_REQUIRE(sources[t].offset + hsize <=
                          static_cast<std::size_t>(sources[t].buf.size()),
                      "ReadSerializedBondRanks: buffer is too small for its header.");
    entries.push_back(
        {sources[t].buf.data() + sources[t].offset, headers.data() + hoff, hsize});
    hoff += hsize;
  }
  impl::RunSerialCopies<TTraits>(entries);
  auto headers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), headers);

  auto to_int = [](real_t x) { return static_cast<int>(std::lround(x)); };
  std::vector<std::vector<int>> ranks(ntrains);
  hoff = 0;
  for (int t = 0; t < ntrains; ++t) {
    PARTHENON_REQUIRE(to_int(headers_h(hoff)) == ncores[t],
                      "ReadSerializedBondRanks: serialized train has the wrong number "
                      "of cores.");
    ranks[t].resize(ncores[t] + 1);
    for (int c = 0; c <= ncores[t]; ++c)
      ranks[t][c] = to_int(headers_h(hoff + 1 + c));
    hoff += SerializedHeaderSize(ncores[t]);
  }
  return ranks;
}

// Rebuild out[t] from the serialized train in sources[t], using indexers[t] as the
// per-core physical layout. Each out[t] is replaced by a freshly allocated train.
template <class TTraits, class BufView>
void DeserializeTrains(const std::vector<SerialTarget<BufView>> &sources,
                       const std::vector<std::vector<Indexer6D>> &indexers,
                       const std::vector<TensorTrainT<TTraits> *> &out) {
  static_assert(TTraits::d_fastest_moving,
                "TT serialization is only implemented for fiber storage.");
  using real_t = typename TTraits::real_t;
  PARTHENON_REQUIRE(sources.size() == indexers.size() && sources.size() == out.size(),
                    "DeserializeTrains: need one indexer set and output per source.");
  const int ntrains = static_cast<int>(sources.size());

  std::vector<int> ncores(ntrains);
  for (int t = 0; t < ntrains; ++t)
    ncores[t] = static_cast<int>(indexers[t].size());
  const auto bond_ranks = ReadSerializedBondRanks<TTraits>(sources, ncores);

  std::vector<impl::SerialCopy<real_t>> entries;
  for (int t = 0; t < ntrains; ++t) {
    const auto &br = bond_ranks[t];
    const std::vector<int> internal(br.begin() + 1, br.end() - 1);
    *out[t] = TensorTrainT<TTraits>(indexers[t], internal, br.front(), br.back());
    PARTHENON_REQUIRE(sources[t].offset + SerializedSize(*out[t]) <=
                          static_cast<std::size_t>(sources[t].buf.size()),
                      "DeserializeTrains: buffer is too small for its train.");
    real_t *src =
        sources[t].buf.data() + sources[t].offset + SerializedHeaderSize(ncores[t]);
    impl::AddFiberCopies(*out[t], src, /*to_buffer=*/false, entries);
  }
  impl::RunSerialCopies<TTraits>(entries);
}

} // namespace tensor
} // namespace parthenon

#endif // TENSORS_TT_SERIALIZE_HPP_
