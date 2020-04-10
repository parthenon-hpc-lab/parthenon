//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_HPP_
#define INTERFACE_SWARM_HPP_

///
/// A Variable type for Placebo-K.
/// Builds on AthenaArrays
/// Date: August 21, 2019
///
///
/// The variable class typically contains state data for the
/// simulation but can also hold non-mesh-based data such as physics
/// parameters, etc.  It inherits the AthenaArray class, which is used
/// for actural data storage and generation

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "bvals/cc/bvals_cc.hpp"
#include "Metadata.hpp"

namespace parthenon {
class MeshBlock;

class Swarm {
  public:
    Swarm(const std::string label, const Metadata &metadata) :
      _label(label),
      _m(metadata),
      mpiStatus(true) {}

  ///< Assign label for variable
  void setLabel(const std::string label) { _label = label; }

  ///< retrieve label for variable
  std::string label() const { return _label; }

  ///< retrieve metadata for variable
  const Metadata metadata() const { return _m; }

  std::string getAssociated() { return _m.getAssociated(); }

  /// return information string
  std::string info() { return std::string("Default information"); }

  bool mpiStatus;

 private:
  Metadata _m;
  std::string _label;
};

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
