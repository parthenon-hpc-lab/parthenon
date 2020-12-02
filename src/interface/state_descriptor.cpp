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

template <typename T>
struct DependencyTracker {
  std::unordered_set<T> provided_vars;
  std::unordered_set<T> depends_vars;
  std::unordered_map<T, int> overridable_vars;
  std::unordered_map<T, Metadata> overridable_meta;

  void Sort(const std::string &package, const T &var, Metadata &metadata) {
    auto dependency = metadata.Dependency();
    if (dependency == Metadata::None) {
      metadata.Set(Metadata::Provides);
    }
    switch (dependency) {
    case Metadata::None:
      PARTHENON_THROW("Unknown dependency");
      break;
    case Metadata::Provides:
      auto it = provided_vars.find(var);
      if (provided_vars.count(var) > 0) {
        std::stringstream ss;
        ss < "Variable " << var << " Provided by multiple packages" << std::endl;
        PARTHENON_THROW(ss);
      }
      provided_vars.insert(var);
      break;
    case Metadata::Depends:
      depends_vars.insert(var);
      break;
    case Metadata::Overridable:
      if (overridable_meta[var] = metadata;
      overridable_vars[var] += 1; // using value initalization of ints = 0
    default:
      break;
    }
  }

  template <typename Collection c>
  void Sort(const std::string &package, const Collection &c) {
    for (auto &pair : c) {
      auto &var = pair.first;
      auto &metadata = pair.second;
      Sort(package, var, metadata);
    }
  }

  void CheckDepends() {
    for (auto &v : depends_vars) {
      if (provided_vars.count(v) == 0) {
        std::stringstream ss;
        ss << "Variable " << var
           << " registered as required, but not provided by any package!" << std::endl;
        PARTHENON_THROW(ss);
      }
    }
  }

  template <typename Adder>
  void CheckOverridable(Adder add_to_state) {
    std::unordered_set<T> cache;
    for (auto &pair : overridable_vars) {
      auto &var = pair.first;
      auto &count = pair.second;
      if (provided_vars.count(var) == 0) {
        if (cout > 1) {
          std::stringstream ss;
          ss << "Variable " << var
             << " registered as overridable multiple times, but never provided."
             << " This results in undefined behaviour as to which package will provide "
                "it."
             << std::endl;
          PARTHENON_DEBUG_WARN(ss);
        }
        auto metadata = overridable_meta[var];
        add_to_state(var, metadata);
      }
    }
  }

  bool Provided(const std::string &var) { return provided_vars.count(var) > 0; }
};

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

  // First walk through packages and fill trackers
  DependencyTracker<std::string> var_tracker;
  DependencyTracker<std::pair<std::string, int>> sparse_tracker;
  DependencyTracker<std::string> swarm_tracker;
  for (auto &package : packages) {
    var_tracker.Sort(package->label(), package->AllFields());
    for (auto &pair : package->AllSparseFields()) {
      auto &var = pair.first;
      auto &mvec = pair.second;
      for (auto &metadata : mvec) {
        int id = metadata.GetSparseId();
        sparse_tracker.Sort(package->label(), std::make_pair(var, id), metadata);
      }
    }
    swarm_tracker.Sort(package->label(), package->AllSwarms());
  }

  // check that dependent variables are provided somewhere
  var_tracker.CheckDepends();
  sparse_tracker.CheckDepends();
  swarm_tracker.CheckDepends();

  // Treat overridable vars:
  // If a var is overridable and provided, do nothing.
  // If a var is overridable and unique, add it to the state.
  // If a var is overridable and not unique, add one to the state
  // and optionally throw a warning.
  var_tracker.CheckOverridable([&](const std::string &var, Metadata &metadata) {
    state->AddField(var, metadata);
  });
  sparse_tracker.CheckOverridable(
      [&](std::pair<std::string, int> &var, Metadata &metadata) {
        state->AddField(var.first, metadata);
      });
  swarm_tracker.CheckOverridable([&](const std::string &swarm, Metadata &metadata) {
      state->AddSwarm(swarm, metadata);
      for (auto package : packages) {
        if (package->SwarmPresent(swarm)) {
          for (auto & pair : package->AllSwarmValues(swarm)) {
            state->AddSwarmValue(pair.first, swarm, pair.second);
          }
          return;
        }
      }
    });

  // Second walk through packages to fill state
  auto AddSwarm = [&](StateDescriptor *package, const std::string &swarm,
                      const std::string &swarm_name, Metadata &metadata) {
    state->AddSwarm(swarm_name, metadata);
    for (auto &p : package->AllSwarmValues(swarm)) {
      auto &val_name = p.first;
      auto &val_meta = p.second;
      state->AddSwarmValue(val_name, swarm_name, val_meta);
    }
  };
  for (auto &package : packages) {
    // dense fields
    for (auto &pair : package->AllFields()) {
      auto &var = pair.first;
      auto &metadata = pair.second;
      auto dependency = metadata.Dependency();
      switch (dependency) {
      case Metadata::Private:
        state->AddField(package->label() + "::" + var, metadata);
        break;
      case Metadata::Provides: // safe to just add this, as already validated
        state->AddField(var, metadata);
        break;
      default: 
        // Depends, Overridable, other already dealt with
        break;
      }
    }
    // sparse fields
    for (auto &pair : package->AllSparseFields()) {
      auto &var = pair.first;
      auto &mvec = pair.second;
      for (auto &metadata : mvec) {
        auto id = metadata.GetSparseId();
        auto dependency = metadata.Dependency();
        switch (dependency) {
        case Metadata::Private:
          state->AddField(package->label() + "::" + var, metadata);
          break;
        case Metadata::Provides:
          state->AddField(var, metadata);
          break;
        default:
          break;
        }
      }
    }
    // swarms
    for (auto &pair : package->AllSwarms()) {
      auto &swarm = pair.first;
      auto &metadata = pair.second;
      auto dependency = metadata.Dependency();
      switch (dependency) {
      case Metadata::Private:
        auto swarm_name = package->label() + "::" + swarm;
        AddSwarm(package.get(), swarm, swarm_name, metadata);
        break;
      case Metadata::Provides:
        AddSwarm(package.get(), swarm, swarm, metadata);
        break;
      default:
        break;
      }
    }
  }
  return state;
}

} // namespace parthenon
