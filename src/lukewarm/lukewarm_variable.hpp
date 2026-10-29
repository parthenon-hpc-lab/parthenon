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
#ifndef LUKEWARM_LUKEWARM_VARIABLE_HPP_
#define LUKEWARM_LUKEWARM_VARIABLE_HPP_

// This file was made in part with generative AI.

#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

// Data types a lukewarm field may hold
using LukewarmTypes = TypeList<Real, int, std::uint64_t>;

template <typename T>
constexpr MetadataFlag LukewarmTypeFlag() {
  static_assert(LukewarmTypes::IsIn<T>(), "Unsupported lukewarm field type");
  if constexpr (std::is_same_v<T, Real>) {
    return Metadata::Real;
  } else if constexpr (std::is_same_v<T, int>) {
    return Metadata::Integer;
  } else {
    return Metadata::UInt64;
  }
}

// A single per-particle field of a lukewarm. A lukewarm stores all fields of the same
// data type together, and this field's components are rows
// [GetOffset(), GetOffset() + NumComponents()) of that storage. data is a view of just
// those rows, with extents (NumComponents(), capacity) and the particle index contiguous.
template <typename T>
class LukewarmVariable {
 public:
  LukewarmVariable(const std::string &label, const Metadata &m, const int offset)
      : label_(label), m_(m), offset_(offset) {}

  const std::string &label() const { return label_; }
  const Metadata &metadata() const { return m_; }
  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  const std::vector<int> &GetShape() const { return m_.Shape(); }
  int NumComponents() const {
    const auto &shape = m_.Shape();
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }
  int GetOffset() const { return offset_; }

  // Replaced whenever the owning lukewarm reallocates, so don't hold on to it across
  // operations that add particles or fields
  ParArray2DRaw<T> data;

 private:
  std::string label_;
  Metadata m_;
  int offset_;
};

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_VARIABLE_HPP_
