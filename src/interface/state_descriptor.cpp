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

#include <array>
#include <stringstream>
#include <unordered_map>
#include <unordered_set>

#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"

namespace parthenon {

bool StateDescriptor::AddSwarmValue(const std::string &value_name,
                                    const std::string &swarm_name, Metadata &m) {
  if (swarmMetadataMap_.count(swarm_name) == 0) {
    throw std::invalid_argument("Swarm " + swarm_name + " does not exist!");
  }
  if (swarmValueMetadataMap_[swarm_name].count(value_name) > 0) {
    throw std::invalid_argument("Swarm value " + value_name + " already exists!");
  }
  swarmValueMetadataMap_[swarm_name][value_name] = m;

  return true;
}

bool StateDescriptor::AddField(const std::string &field_name, Metadata &m) {
  if (m.IsSet(Metadata::Sparse)) {
    auto miter = sparseMetadataMap_.find(field_name);
    if (miter != sparseMetadataMap_.end()) {
      miter->second.push_back(m);
    } else {
      sparseMetadataMap_[field_name] = {m};
    }
  } else {
    const std::string &assoc = m.getAssociated();
    if (!assoc.length()) m.Associate(field_name);
    auto miter = metadataMap_.find(field_name);
    if (miter != metadataMap_.end()) { // this field has already been added
      Metadata &mprev = miter->second;
      return false;
    } else {
      metadataMap_[field_name] = m;
      m.Associate("");
    }
  }
  return true;
}

bool StateDescriptor::FlagsPresent(std::vector<MetadataFlag> const &flags,
                                   bool matchAny) {
  for (auto &pair : metadataMap_) {
    auto &metadata = pair.second;
    if (metadata.FlagsSet(flags, matchAny)) return true;
  }
  for (auto &pair : sparseMetadataMap_) {
    auto &sparsevec = pair.second;
    for (auto &metadata : sparsevec) {
      if (metadata.FlagsSet(flags, matchAny)) return true;
    }
  }
  return false;
}

std::ostream &operator<<(ostream &os, const StateDescriptor &sd) {
  os << "# Package: " << sd.label() << "\n"
     << "# ---------------------------------------------------\n"
     << "# Variables:\n"
     << "# Name\tMetadata flags\n"
     << "# ---------------------------------------------------\n";
  for (auto &pair : sd.metadataMap_) {
    auto &var = pair.first;
    auto &metadata = pair.second;
    os << var << "\t" << metadata << "\n";
  }
  os << "# ---------------------------------------------------\n"
     << "# Sparse Variables:\n"
     << "# Name\tsparse id\tMetadata flags\n"
     << "# ---------------------------------------------------\n";
  for (auto &pair : sd.sparseMetadataMap_) {
    auto &var = pair.first;
    auto &mvec = pair.second;
    os << var << "\n";
    for (auto &metadata : mvec) {
      os << "    \t" << metadata.GetSparseID() << "\t" << metadata << "\n";
    }
  }
  os << "# ---------------------------------------------------\n"
     << "# Swarms:\n"
     << "# Swarm\tValue\tmetadata\n"
     << "# ---------------------------------------------------\n";
  for (auto &pair : sd.swarmValueMetadataMap_) {
    auto &swarm = pair.first;
    auto &svals = pair.second;
    os << swarm << "\n";
    for (auto &p2 : svals) {
      auto &val = pair.first;
      auto &metadata = pair.second;
      os << "     \t" << val << "\t" << metadata << "\n";
    }
  }
  return os;
}

std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages) {
  auto state = std::make_shared<StateDescriptor>();
  std::unordered_set<std::string> provided_vars;
  std::unordered_set<std::string> depends_vars;
  std::unordered_map<std::string, std::string> overridable_vars;

  auto sort_var = [&](const std::string package, const std::string &var,
                      Metadata &metadata) {
    auto dependency = metadata.Dependency();
    if (dependency == Metadata::None) {
      metadata.Set(Metadata::Provides);
    }
    switch (dependency) {
    case Metadata::Private:
      AddField(package + "::" + var, m);
      break;
    case Metadata::Provides:
      auto it = provided_vars.find(var);
      if (provided_vars.count(var) > 0) {
        std::stringstream ss;
        ss < "Variable " << var << " Provided by multiple packages" << std::endl;
        PARTHENON_THROW(ss);
      }
      provided_vars.insert(var);
      AddField(var, m);
      break;
    case Metadata::Depends:
      depends_vars.insert(var);
      break;
    case Metadata::Overridable:
      overridable_vars[package] = var;
      break;
    default:
      PARTHENON_THROW("Unknown dependency");
      break;
    }
  };

  for (auto & package : packages) {
    for (auto & pair : package->AllFields) {
      auto &var = pair.first;
      auto &metadata = pair.second;
      sort_var(package->label(), var, metadata);
    }
    for (auto & pair : package->AllSparseFields) {
      auto &var = pair.first;
      auto &mvec = pair.second;
      for (auto & metadata : mvec) {
        sort_var(package->label(), var
      }
    }
  }
}

} // namespace parthenon
