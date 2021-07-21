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

#include "interface/metadata.hpp"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

using parthenon::Metadata;
using parthenon::MetadataFlag;

namespace parthenon {

// Must declare the flag values for ODR-uses
#define PARTHENON_INTERNAL_FOR_FLAG(name) constexpr MetadataFlag Metadata::name;

PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

namespace internal {

class UserMetadataState {
 public:
  UserMetadataState() {
#define PARTHENON_INTERNAL_FOR_FLAG(name)                                                \
  flag_name_map_.push_back(#name);                                                       \
  flag_names_.insert(#name);

    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG

#undef PARTHENON_INTERNAL_FOR_FLAG
  }

  MetadataFlag AllocateNewFlag(std::string &&name) {
    if (flag_names_.find(name) != flag_names_.end()) {
      std::stringstream ss;
      ss << "MetadataFlag with name '" << name << "' already exists.";
      throw std::runtime_error(ss.str());
    }

    auto const flag = flag_name_map_.size();
    flag_names_.insert(name);
    flag_name_map_.push_back(std::move(name));

    auto flag_obj = MetadataFlag(static_cast<int>(flag));

    return flag_obj;
  }

  std::string const &FlagName(MetadataFlag flag) { return flag_name_map_.at(flag.flag_); }

  const auto &AllFlags() { return flag_name_map_; }

 private:
  std::vector<std::string> flag_name_map_;
  std::unordered_set<std::string> flag_names_;
};

} // namespace internal
} // namespace parthenon

parthenon::internal::UserMetadataState metadata_state;

MetadataFlag Metadata::AllocateNewFlag(std::string &&name) {
  return metadata_state.AllocateNewFlag(std::move(name));
}

std::string const &MetadataFlag::Name() const { return metadata_state.FlagName(*this); }

namespace parthenon {
std::ostream &operator<<(std::ostream &os, const parthenon::Metadata &m) {
  bool first = true;
  auto &flags = metadata_state.AllFlags();
  for (int i = 0; i < flags.size(); ++i) {
    auto flag = MetadataFlag(i);
    if (m.IsSet(flag)) {
      if (!first) {
        os << ",";
      } else {
        first = false;
      }
      auto &flag_name = flags[i];
      os << flag_name;
    }
  }
  return os;
}

std::vector<MetadataFlag> Metadata::Flags() const {
  std::vector<MetadataFlag> set_flags;
  const auto &flags = metadata_state.AllFlags();
  for (int i = 0; i < flags.size(); ++i) {
    const auto flag = MetadataFlag(i);
    if (IsSet(flag)) {
      set_flags.push_back(flag);
    }
  }

  return set_flags;
}

std::array<int, 6> Metadata::GetArrayDims(std::weak_ptr<MeshBlock> wpmb) const {
  std::array<int, 6> arrDims;

  const auto &shape = shape_;
  const int N = shape.size();

  if (IsMeshTied()) {
    // Let the FaceVariable, EdgeVariable, and NodeVariable
    // classes add the +1's where needed.  They all expect
    // these dimensions to be the number of cells in each
    // direction, NOT the size of the arrays
    assert(N >= 1 && N <= 3);
    PARTHENON_REQUIRE_THROWS(!wpmb.expired(),
                             "Cannot determine array dimensions for mesh-tied entity "
                             "without a valid meshblock");
    auto pmb = wpmb.lock();
    arrDims[0] = pmb->cellbounds.ncellsi(IndexDomain::entire);
    arrDims[1] = pmb->cellbounds.ncellsj(IndexDomain::entire);
    arrDims[2] = pmb->cellbounds.ncellsk(IndexDomain::entire);
    for (int i = 0; i < N; i++)
      arrDims[i + 3] = shape[i];
    for (int i = N; i < 3; i++)
      arrDims[i + 3] = 1;
  } else {
    // This variable is not necessarily tied to any specific
    // mesh element, so dims will be used as the actual array
    // size in each dimension
    assert(N >= 1 && N <= 6);
    for (int i = 0; i < N; i++)
      arrDims[i] = shape[i];
    for (int i = N; i < 6; i++)
      arrDims[i] = 1;
  }

  return arrDims;
}

} // namespace parthenon
