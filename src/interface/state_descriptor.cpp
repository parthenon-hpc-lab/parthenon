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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

void Packages_t::Add(const std::shared_ptr<StateDescriptor> &package) {
  const auto &name = package->label();
  PARTHENON_REQUIRE_THROWS(packages_.count(name) == 0,
                           "Package name " + name + " must be unique.");
  packages_[name] = package;
  return;
}

class VariableProvider {
 public:
  virtual void AddPrivate(const std::string &package, const VarID &vid,
                          const Metadata &metadata) = 0;
  virtual void AddProvides(const std::string &package, const VarID &vid,
                           const Metadata &metadata) = 0;
  virtual void AddOverridable(const VarID &vid, Metadata &metadata) = 0;
};

// Helper class for ResolvePackages
class DependencyTracker {
 public:
  bool Provided(const VarID &vid) { return provided_vars.count(vid) > 0; }

  void Categorize(const std::string &package, const VarID &vid, const Metadata &metadata,
                  VariableProvider *pvp) {
    auto dependency = metadata.Role();
    if (dependency == Metadata::Private) {
      pvp->AddPrivate(package, vid, metadata);
    } else if (dependency == Metadata::Provides) {
      if (Provided(vid)) {
        PARTHENON_THROW("Variable " + vid.label() + " provided by multiple packages");
      }
      provided_vars.insert(vid);
      pvp->AddProvides(package, vid, metadata);
    } else if (dependency == Metadata::Requires) {
      depends_vars.insert(vid);
    } else if (dependency == Metadata::Overridable) {
      if (overridable_meta.count(vid) == 0) {
        overridable_meta[vid] = {metadata};
      }
      // only update overridable_vars count once
      if (overridable_meta.at(vid).size() == 1) {
        overridable_vars[vid] += 1; // using value initialization of ints = 0
      }
    } else {
      PARTHENON_THROW("Unknown dependency");
    }
  }

  template <typename Collection>
  void CategorizeCollection(const std::string &package, const Collection &c,
                            VariableProvider *pvp) {
    for (auto &pair : c) {
      const auto &vid = pair.first;
      auto &metadata = pair.second;
      Categorize(package, VarID(vid), metadata, pvp);
    }
  }

  void CheckRequires() {
    for (auto &v : depends_vars) {
      if (!Provided(v) && overridable_vars.count(v) == 0) {
        std::stringstream ss;
        ss << "Variable " << v.label()
           << " registered as required, but not provided by any package!" << std::endl;
        PARTHENON_THROW(ss);
      }
    }
  }

  void CheckOverridable(VariableProvider *pvp) {
    std::unordered_set<std::string> cache;
    for (auto &pair : overridable_vars) {
      const auto &vid = pair.first;
      auto &count = pair.second;
      if (!Provided(vid)) {
        if (count > 1) {
          std::stringstream ss;
          ss << "Variable " << vid.label()
             << " registered as overridable multiple times, but never provided."
             << " This results in undefined behaviour as to which package will provide"
             << " it." << std::endl;
          PARTHENON_DEBUG_WARN(ss);
        }
        auto &mvec = overridable_meta[vid];
        for (auto &metadata : mvec) {
          pvp->AddOverridable(vid, metadata);
        }
      }
    }
  }

 private:
  std::unordered_set<VarID, VarIDHasher> provided_vars;
  std::unordered_set<VarID, VarIDHasher> depends_vars;

  std::unordered_map<VarID, int, VarIDHasher> overridable_vars;
  std::unordered_map<VarID, std::vector<Metadata>, VarIDHasher> overridable_meta;
};

// Helper functions for adding vars
// closures by reference
class FieldProvider : public VariableProvider {
 public:
  explicit FieldProvider(std::shared_ptr<StateDescriptor> &sd) : state_(sd) {}
  void AddPrivate(const std::string &package, const VarID &vid,
                  const Metadata &metadata) {
    state_->AddFieldImpl(VarID(package + "::" + vid.base_name, vid.sparse_id), metadata);
  }
  void AddProvides(const std::string &package, const VarID &vid,
                   const Metadata &metadata) {
    state_->AddFieldImpl(vid, metadata);
  }
  void AddOverridable(const VarID &vid, Metadata &metadata) {
    state_->AddFieldImpl(vid, metadata);
  }

 private:
  std::shared_ptr<StateDescriptor> &state_;
};

class SwarmProvider : public VariableProvider {
 public:
  SwarmProvider(Packages_t &packages, std::shared_ptr<StateDescriptor> &sd)
      : packages_(packages), state_(sd) {}
  void AddPrivate(const std::string &package, const VarID &vid,
                  const Metadata &metadata) {
    AddSwarm_(packages_.Get(package).get(), vid.label(), package + "::" + vid.label(),
              metadata);
  }
  void AddProvides(const std::string &package, const VarID &vid,
                   const Metadata &metadata) {
    AddSwarm_(packages_.Get(package).get(), vid.label(), vid.label(), metadata);
  }
  void AddOverridable(const VarID &vid, Metadata &metadata) {
    state_->AddSwarm(vid.label(), metadata);
    for (auto &pair : packages_.AllPackages()) {
      auto &package = pair.second;
      if (package->SwarmPresent(vid.label())) {
        for (auto &pair : package->AllSwarmValues(vid.label())) {
          state_->AddSwarmValue(pair.first, vid.label(), pair.second);
        }
        return;
      }
    }
  }

 private:
  void AddSwarm_(StateDescriptor *package, const std::string &swarm,
                 const std::string &swarm_name, const Metadata &metadata) {
    state_->AddSwarm(swarm_name, metadata);
    for (auto &p : package->AllSwarmValues(swarm)) {
      auto &val_name = p.first;
      auto &val_meta = p.second;
      state_->AddSwarmValue(val_name, swarm_name, val_meta);
    }
  }
  std::shared_ptr<StateDescriptor> &state_;
  Packages_t &packages_;
};

bool StateDescriptor::AddSwarmValue(const std::string &value_name,
                                    const std::string &swarm_name, const Metadata &m) {
  if (swarmMetadataMap_.count(swarm_name) == 0) {
    throw std::invalid_argument("Swarm " + swarm_name + " does not exist!");
  }
  if (swarmValueMetadataMap_[swarm_name].count(value_name) > 0) {
    throw std::invalid_argument("Swarm value " + value_name + " already exists!");
  }
  swarmValueMetadataMap_[swarm_name][value_name] = m;

  return true;
}

bool StateDescriptor::AddFieldImpl(const VarID &vid, const Metadata &m_in) {
  Metadata m = m_in; // Force const correctness

  const std::string &assoc = m.getAssociated();
  if (m.getAssociated() != "") {
    m.Associate(vid.label());
  }
  if (metadataMap_.count(vid) > 0) {
    return false; // this field has already been added
  } else {
    metadataMap_[vid] = m;
  }

  return true;
}

bool StateDescriptor::FlagsPresent(std::vector<MetadataFlag> const &flags,
                                   bool matchAny) {
  for (auto &pair : metadataMap_)
    if (pair.second.FlagsSet(flags, matchAny)) return true;

  for (auto &pair : swarmMetadataMap_)
    if (pair.second.FlagsSet(flags, matchAny)) return true;

  // TODO(JL): What about swarmValueMetadataMap_?

  return false;
}

std::ostream &operator<<(std::ostream &os, const StateDescriptor &sd) {
  os << "# Package: " << sd.label() << "\n"
     << "# ---------------------------------------------------\n"
     << "# Variables:\n"
     << "# Name\tMetadata flags\n"
     << "# ---------------------------------------------------\n";
  for (auto &pair : sd.metadataMap_) {
    auto &metadata = pair.second;
    if (!metadata.IsSet(Metadata::Sparse))
      os << std::left << std::setw(25) << pair.first.label() << " " << metadata << "\n";
  }
  os << "# ---------------------------------------------------\n"
     << "# Sparse Variables:\n"
     << "# Name\tsparse id\tMetadata flags\n"
     << "# ---------------------------------------------------\n";
  for (auto &pair : sd.metadataMap_) {
    auto &metadata = pair.second;
    if (metadata.IsSet(Metadata::Sparse))
      os << std::left << std::setw(25) << pair.first.label() << " " << metadata << "\n";
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
      auto &val = p2.first;
      auto &metadata = p2.second;
      os << std::left << std::setw(25) << ("    \t" + val + " ") << metadata << "\n";
    }
  }
  return os;
}

// Takes all packages and combines them into a single state descriptor
// containing all variables with conflicts resolved.  Note the new
// state descriptor DOES not have any of its function pointers set.
std::shared_ptr<StateDescriptor>
StateDescriptor::CreateResolvedStateDescriptor(Packages_t &packages) {
  auto state = std::make_shared<StateDescriptor>("parthenon::resolved_state");

  // The workhorse data structure. Uses sets to cache which variables
  // are of what type.
  DependencyTracker var_tracker;
  DependencyTracker swarm_tracker;
  // closures that provide functions for DependencyTracker
  FieldProvider field_provider(state);
  SwarmProvider swarm_provider(packages, state);

  // Add private/provides variables. Check for conflicts among those.
  // Track dependent and overridable variables.
  for (auto &pair : packages.AllPackages()) {
    const auto &name = pair.first;
    auto &package = pair.second;
    // sort
    var_tracker.CategorizeCollection(name, package->AllFields(), &field_provider);
    swarm_tracker.CategorizeCollection(name, package->AllSwarms(), &swarm_provider);

    // add sparse ID pools
    for (const auto &pool_itr : package->AllSparseIdPools()) {
      const auto &base_name = pool_itr.first;

      const auto itm = state->sparseIdPool_.find(base_name);
      if (itm != state->sparseIdPool_.end()) {
        // we already have this sparse base name in the resolved state, check if the pool
        // of sparse ids is the same, if they are the same, move on
        if (itm->second != pool_itr.second) {
          std::stringstream err;
          err << "Package '" << name << "' tried to add a sparse ID pool '" << base_name
              << "' to the resolved state, but a different pool of this name already "
                 "exists";
          PARTHENON_THROW(err);
        }
      } else {
        // add this pool to the resolved state
        state->sparseIdPool_.insert(pool_itr);
      }
    }
  }

  // check that dependent variables are provided somewhere
  var_tracker.CheckRequires();
  swarm_tracker.CheckRequires();

  // Treat overridable vars:
  // If a var is overridable and provided, do nothing.
  // If a var is overridable and unique, add it to the state.
  // If a var is overridable and not unique, add one to the state
  // and optionally throw a warning.
  var_tracker.CheckOverridable(&field_provider);   // works on both dense and sparse
  swarm_tracker.CheckOverridable(&swarm_provider); // special for swarms

  return state;
}

} // namespace parthenon
