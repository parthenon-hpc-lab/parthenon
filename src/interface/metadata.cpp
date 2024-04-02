//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

using parthenon::Metadata;
using parthenon::MetadataFlag;

namespace parthenon {

// Must declare the flag values for ODR-uses
#define PARTHENON_INTERNAL_FOR_FLAG(name) constexpr MetadataFlag Metadata::name;

PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

int Metadata::num_flags = static_cast<int>(internal::MetadataInternal::Max);

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

  MetadataFlag AllocateNewFlag(const std::string &name) {
    if (flag_names_.find(name) != flag_names_.end()) {
      std::stringstream ss;
      ss << "MetadataFlag with name '" << name << "' already exists.";
      throw std::runtime_error(ss.str());
    }

    auto const flag = flag_name_map_.size();
    flag_names_.insert(name);
    flag_name_map_.push_back(name);

    auto flag_obj = MetadataFlag(static_cast<int>(flag));
    names_to_flags_.emplace(name, flag_obj);

    return flag_obj;
  }

  std::string const &FlagName(MetadataFlag flag) { return flag_name_map_.at(flag.flag_); }

  const auto &AllFlags() { return flag_name_map_; }
  const auto &NamesToFlags() { return names_to_flags_; }

 private:
  std::unordered_map<std::string, MetadataFlag> names_to_flags_;
  std::vector<std::string> flag_name_map_;
  std::unordered_set<std::string> flag_names_;
};

} // namespace internal
} // namespace parthenon

parthenon::internal::UserMetadataState metadata_state;

MetadataFlag Metadata::AddUserFlag(const std::string &name) {
  num_flags++;
  return metadata_state.AllocateNewFlag(name);
}

std::string const &MetadataFlag::Name() const { return metadata_state.FlagName(*this); }

bool Metadata::FlagNameExists(const std::string &flagname) {
  return (metadata_state.NamesToFlags().count(flagname) > 0);
}
MetadataFlag Metadata::GetUserFlag(const std::string &flagname) {
  return (metadata_state.NamesToFlags().at(flagname));
}

namespace parthenon {
Metadata::Metadata(const std::vector<MetadataFlag> &bits, const std::vector<int> &shape,
                   const std::vector<std::string> &component_labels,
                   const std::string &associated,
                   const refinement::RefinementFunctions_t ref_funcs_)
    : shape_(shape), component_labels_(component_labels), associated_(associated) {
  // set flags
  for (const auto f : bits) {
    DoBit(f, true);
  }

  // set defaults
  if (CountSet({None, Node, Edge, Face, Cell}) == 0) {
    DoBit(None, true);
  }
  if (CountSet({Private, Provides, Requires, Overridable}) == 0) {
    DoBit(Provides, true);
  }
  if (CountSet({Boolean, Integer, Real}) == 0) {
    DoBit(Real, true);
  }
  if (CountSet({Independent, Derived}) == 0) {
    DoBit(Derived, true);
  }
  // If variable is refined, set a default prolongation/restriction op
  // TODO(JMM): This is dangerous. See Issue #844.
  if (IsRefined()) {
    refinement_funcs_ = ref_funcs_;
  }

  // check if all flag constraints are satisfied, throw if not
  IsValid(true);

  // check shape is valid
  // TODO(JL) Should we be extra pedantic and check that shape matches Vector/Tensor
  // flags?
  if (IsMeshTied()) {
    PARTHENON_REQUIRE_THROWS(
        shape_.size() <= 3,
        "Variables tied to mesh entities can only have a shape of rank <= 3");

    int num_comp = 1;
    for (auto s : shape) {
      num_comp *= s;
    }

    PARTHENON_REQUIRE_THROWS(component_labels.size() == 0 ||
                                 (component_labels.size() == num_comp),
                             "Must provide either 0 component labels or the same "
                             "number as the number of components");
  }

  // Set the allocation and deallocation thresholds
  // TODO(JMM): This is dangerous. See Issue #844.
  if (IsSet(Sparse)) {
    allocation_threshold_ = Globals::sparse_config.allocation_threshold;
    deallocation_threshold_ = Globals::sparse_config.deallocation_threshold;
    default_value_ = 0.0;
  } else {
    // Not sparse, so set to zero so we are guaranteed never to deallocate
    allocation_threshold_ = 0.0;
    deallocation_threshold_ = 0.0;
    default_value_ = 0.0;
  }
}

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

std::array<int, MAX_VARIABLE_DIMENSION>
Metadata::GetArrayDims(std::weak_ptr<MeshBlock> wpmb, bool coarse) const {
  std::array<int, MAX_VARIABLE_DIMENSION> arrDims;
  const auto &shape = shape_;
  const int N = shape.size();

  if (IsMeshTied()) {
    // Let the FaceVariable, EdgeVariable, and NodeVariable
    // classes add the +1's where needed.  They all expect
    // these dimensions to be the number of cells in each
    // direction, NOT the size of the arrays
    assert(N >= 0 && N <= 3);
    PARTHENON_REQUIRE_THROWS(!wpmb.expired(),
                             "Cannot determine array dimensions for mesh-tied entity "
                             "without a valid meshblock");
    auto pmb = wpmb.lock();
    const auto bnds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    arrDims[0] = bnds.ncellsi(IndexDomain::entire);
    arrDims[1] = bnds.ncellsj(IndexDomain::entire);
    arrDims[2] = bnds.ncellsk(IndexDomain::entire);
    for (int i = 0; i < N; i++)
      arrDims[i + 3] = shape[i];
    for (int i = N; i < 3; i++)
      arrDims[i + 3] = 1;
    if (IsSet(Cell) || (IsSet(Face) && IsSet(Flux))) {
      arrDims[MAX_VARIABLE_DIMENSION - 1] = 1; // Only one cell center per cell
    } else if (IsSet(Face) || IsSet(Edge)) {
      arrDims[MAX_VARIABLE_DIMENSION - 1] = 3; // Three faces and edges per cell
      arrDims[0]++;
      if (arrDims[1] > 1) arrDims[1]++;
      if (arrDims[2] > 1) arrDims[2]++;
    } else if (IsSet(Node)) {
      arrDims[MAX_VARIABLE_DIMENSION - 1] = 1; // Only one lower left node per cell
      arrDims[0]++;
      if (arrDims[1] > 1) arrDims[1]++;
      if (arrDims[2] > 1) arrDims[2]++;
    }
  } else if (IsSet(Particle)) {
    assert(N >= 0 && N <= MAX_VARIABLE_DIMENSION - 1);
    arrDims[0] = 1; // To be updated by swarm based on pool size before allocation
    for (int i = 0; i < N; i++)
      arrDims[i + 1] = shape[i];
    for (int i = N; i < MAX_VARIABLE_DIMENSION - 1; i++)
      arrDims[i + 1] = 1;
  } else if (IsSet(Swarm)) {
    // No dimensions
    // TODO(BRR) This will be replaced in the swarm packing PR, but is currently required
    // since swarms carry metadata but do not have an array size.
  } else {
    // This variable is not necessarily tied to any specific
    // mesh element, so dims will be used as the actual array
    // size in each dimension
    assert(N >= 1 && N <= MAX_VARIABLE_DIMENSION);
    for (int i = 0; i < N; i++)
      arrDims[i] = shape[i];
    for (int i = N; i < MAX_VARIABLE_DIMENSION; i++)
      arrDims[i] = 1;
  }

  return arrDims;
}

namespace MetadataUtils {
bool MatchFlags(const Metadata::FlagCollection &flags, Metadata m) {
  const auto &intersections = flags.GetIntersections();
  const auto &unions = flags.GetUnions();
  const auto &exclusions = flags.GetExclusions();

  return m.AllFlagsSet(intersections) && (unions.empty() || m.AnyFlagsSet(unions)) &&
         m.NoFlagsSet(exclusions);
}
} // namespace MetadataUtils

} // namespace parthenon
