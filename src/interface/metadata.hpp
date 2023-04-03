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
#ifndef INTERFACE_METADATA_HPP_
#define INTERFACE_METADATA_HPP_

#include <algorithm>
#include <bitset>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "prolong_restrict/pr_ops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

/// The point of this macro is to generate code for each built-in flag using the
/// `PARTHENON_INTERNAL_FOR_FLAG` macro. This is to accomplish the following goals:
/// - Generate a unique value for each flag using an enum
/// - Wrap each value in a `MetadataFlag` type that can only be instantiated within
///   parthenon
/// - Make it possible to return a user-friendly string for each flag.
///
/// Having this macro means you only need to add a new flag in one place and have each of
/// these properties automatically be updated.
#define PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG                                          \
  /**  bit 0 is ignored */                                                               \
  PARTHENON_INTERNAL_FOR_FLAG(Ignore)                                                    \
  /************************************************/                                     \
  /** TOPOLOGY: Exactly one must be specified (default is None) */                       \
  /** no topology specified */                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(None)                                                      \
  /** cell variable */                                                                   \
  PARTHENON_INTERNAL_FOR_FLAG(Cell)                                                      \
  /** face variable */                                                                   \
  PARTHENON_INTERNAL_FOR_FLAG(Face)                                                      \
  /** edge variable */                                                                   \
  PARTHENON_INTERNAL_FOR_FLAG(Edge)                                                      \
  /** node variable */                                                                   \
  PARTHENON_INTERNAL_FOR_FLAG(Node)                                                      \
  /** particle variable */                                                               \
  PARTHENON_INTERNAL_FOR_FLAG(Particle)                                                  \
  /** swarm */                                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(Swarm)                                                     \
  /************************************************/                                     \
  /** ROLE: Exactly one must be specified (default is Provides) */                       \
  /** Private to a package */                                                            \
  PARTHENON_INTERNAL_FOR_FLAG(Private)                                                   \
  /** Provided by a package */                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(Provides)                                                  \
  /** Not created by a package, assumes available from another package */                \
  PARTHENON_INTERNAL_FOR_FLAG(Requires)                                                  \
  /** does nothing if another package provides the variable */                           \
  PARTHENON_INTERNAL_FOR_FLAG(Overridable)                                               \
  /************************************************/                                     \
  /** SHAPE: Neither or one (but not both) can be specified */                           \
  /** a vector quantity, i.e. a rank 1 contravariant tensor */                           \
  PARTHENON_INTERNAL_FOR_FLAG(Vector)                                                    \
  /** a rank-2 tensor */                                                                 \
  PARTHENON_INTERNAL_FOR_FLAG(Tensor)                                                    \
  /************************************************/                                     \
  /** DATATYPE: Exactly one must be specified (default is Real) */                       \
  /** Boolean-valued quantity */                                                         \
  PARTHENON_INTERNAL_FOR_FLAG(Boolean)                                                   \
  /** Integer-valued quantity */                                                         \
  PARTHENON_INTERNAL_FOR_FLAG(Integer)                                                   \
  /** Real-valued quantity */                                                            \
  PARTHENON_INTERNAL_FOR_FLAG(Real)                                                      \
  /************************************************/                                     \
  /** INDEPENDENT: Exactly one must be specified (default is Derived) */                 \
  /** is an independent, evolved variable */                                             \
  PARTHENON_INTERNAL_FOR_FLAG(Independent)                                               \
  /** is a derived quantity (ignored) */                                                 \
  PARTHENON_INTERNAL_FOR_FLAG(Derived)                                                   \
  /************************************************/                                     \
  /** OTHER: All the following flags can be turned on or off independently */            \
  /** advected variable */                                                               \
  PARTHENON_INTERNAL_FOR_FLAG(Advected)                                                  \
  /** conserved variable */                                                              \
  PARTHENON_INTERNAL_FOR_FLAG(Conserved)                                                 \
  /** intensive variable */                                                              \
  PARTHENON_INTERNAL_FOR_FLAG(Intensive)                                                 \
  /** added to restart dump */                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(Restart)                                                   \
  /** is a sparse variable */                                                            \
  PARTHENON_INTERNAL_FOR_FLAG(Sparse)                                                    \
  /** should this variable minimize buffer use during communication */                   \
  PARTHENON_INTERNAL_FOR_FLAG(SparseCommunication)                                       \
  /** only one copy even if multiple stages */                                           \
  PARTHENON_INTERNAL_FOR_FLAG(OneCopy)                                                   \
  /** Do boundary communication */                                                       \
  PARTHENON_INTERNAL_FOR_FLAG(FillGhost)                                                 \
  /** does variable have fluxes */                                                       \
  PARTHENON_INTERNAL_FOR_FLAG(WithFluxes)                                                \
  /** the variable needs to be communicated across ranks during remeshing */             \
  PARTHENON_INTERNAL_FOR_FLAG(ForceRemeshComm)
namespace parthenon {

namespace internal {
enum class MetadataInternal {
// declare all the internal flags in an enum so that their values are unique and kept
// up to date
#define PARTHENON_INTERNAL_FOR_FLAG(name) name,
  PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG Max
#undef PARTHENON_INTERNAL_FOR_FLAG
};

class UserMetadataState;

} // namespace internal

class MeshBlock;
class Metadata;

class TensorShape {
 public:
  TensorShape() {}
  explicit TensorShape(int scalar) {}
  TensorShape(int rank, int *size) { shape_.insert(shape_.end(), size, size + rank); }
  std::vector<int> shape_;
};

class MetadataFlag {
  // This allows `Metadata` and `UserMetadataState` to instantiate `MetadataFlag`.
  friend class Metadata;
  friend class internal::UserMetadataState;

 public:
  constexpr bool operator==(MetadataFlag const &other) const {
    return flag_ == other.flag_;
  }

  constexpr bool operator!=(MetadataFlag const &other) const {
    return flag_ != other.flag_;
  }

  constexpr bool operator<(const MetadataFlag &other) const {
    return flag_ < other.flag_;
  }

  std::string const &Name() const;

#ifdef CATCH_VERSION_MAJOR
  // Should never be used for application code - only exposed for testing.
  constexpr int InternalFlagValue() const { return flag_; }
#endif

  friend std::ostream &operator<<(std::ostream &os, const Metadata &m);
  friend std::ostream &operator<<(std::ostream &os, const MetadataFlag &flag) {
    os << flag.Name();
    return os;
  }

 private:
  // MetadataFlag can only be instantiated by Metadata
  constexpr explicit MetadataFlag(int flag) : flag_(flag) {}

  int flag_;
};

/// @brief
///
/// The metadata class is a descriptor for variables in the
/// simulation.
///
///  Can set or query attributes specifed in flags.
///
class Metadata {
 public:
  /// The flags refer to the different attributes that a variable can
  /// have.  These include the topology, IO, advection, conservation,
  /// whether it sparse, etc.  This is designed to be easily extensible.

  // this wraps all the built-in flags in the `MetadataFlag` type so that using these
  // flags is type-safe.
#define PARTHENON_INTERNAL_FOR_FLAG(name)                                                \
  constexpr static MetadataFlag name =                                                   \
      MetadataFlag(static_cast<int>(::parthenon::internal::MetadataInternal::name));

  PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

  // TODO(JMM): Kind of treating Metadata as a namespace here
  using FlagVec = std::vector<MetadataFlag>;

  // FlagCollection is a collection object that encapsulates the desire for
  // target variables/swarms with:
  // - At least ONE of the "unions" flags,
  // - ALL of the "intersections" flags,
  // - and NONE of the "exclusions" flags
  class FlagCollection {
   public:
    FlagCollection() = default;
    template <typename T,
              REQUIRES(std::is_same<typename T::value_type, MetadataFlag>::value)>
    explicit FlagCollection(const T &flags, bool take_union = false) {
      if (take_union) {
        unions_.insert(flags.begin(), flags.end());
      } else { // intersection
        intersections_.insert(flags.begin(), flags.end());
      }
    }
    // Constructor that takes a brace-enclosed initializer list
    // Required since the type of {...} cannot be deduced since it
    // could be the type of any object that could be initialized by
    // an initializer list
    // TODO(JMM): The cast to to a vector here implies some extra
    // copies which aren't great. Don't do this too much I guess.
    FlagCollection(std::initializer_list<MetadataFlag> flags, bool take_union = false)
        : FlagCollection(FlagVec(std::move(flags)), take_union) {}
    // Constructor from a comma-separated list. Default is union.
    // Force correct type inferrence by making the first arg a flag
    template <typename... Args>
    FlagCollection(MetadataFlag first, Args... args)
        : FlagCollection({first, std::forward<Args>(args)...}, false) {}
    // Check if set empty
    bool Empty() const { return (unions_.empty() && intersections_.empty()); }
    // Union
    template <template <class...> class Container_t, class... extra>
    void TakeUnion(const Container_t<MetadataFlag, extra...> &flags) {
      unions_.insert(flags.begin(), flags.end());
    }
    template <typename... Args>
    void TakeUnion(MetadataFlag first, Args... args) {
      TakeUnion(FlagVec({first, std::forward<Args>(args)...}));
    }
    void TakeUnion(const FlagCollection &other) {
      unions_.insert(other.unions_.begin(), other.unions_.end());
      exclusions_.insert(other.exclusions_.begin(), other.exclusions_.end());
    }
    FlagCollection operator+(const FlagCollection &other) {
      FlagCollection s(*this);
      s.TakeUnion(other);
      return s;
    }
    FlagCollection operator||(const FlagCollection &other) { return (*this) + other; }
    // Intersection
    template <template <class...> class Container_t, class... extra>
    void TakeIntersection(const Container_t<MetadataFlag, extra...> &flags) {
      intersections_.insert(flags.begin(), flags.end());
    }
    template <typename... Args>
    void TakeIntersection(MetadataFlag first, Args... args) {
      TakeIntersection(FlagVec({first, std::forward<Args>(args)...}));
    }
    void TakeIntersection(const FlagCollection &other) {
      intersections_.insert(other.intersections_.begin(), other.intersections_.end());
      exclusions_.insert(other.exclusions_.begin(), other.exclusions_.end());
    }
    FlagCollection operator*(const FlagCollection &other) {
      FlagCollection s(*this);
      s.TakeIntersection(other);
      return s;
    }
    FlagCollection operator&&(const FlagCollection &other) { return (*this) * other; }
    // Set Difference
    template <template <class...> class Container_t, class... extra>
    void Exclude(const Container_t<MetadataFlag, extra...> &&flags) {
      exclusions_.insert(flags.begin(), flags.end());
    }
    template <typename... Args>
    void Exclude(MetadataFlag first, Args... args) {
      Exclude(FlagVec({first, std::forward<Args>(args)...}));
    }
    void Exclude(const FlagCollection &other) {
      exclusions_.insert(other.unions_.begin(), other.unions_.end());
      exclusions_.insert(other.intersections_.begin(), other.intersections_.end());
      exclusions_.insert(other.exclusions_.begin(), other.exclusions_.end());
    }
    FlagCollection operator-(const FlagCollection &other) {
      FlagCollection s(*this);
      s.Exclude(other);
      return s;
    }
    // Accessors
    const std::set<MetadataFlag> &GetUnions() const { return unions_; }
    const std::set<MetadataFlag> &GetIntersections() const { return intersections_; }
    const std::set<MetadataFlag> &GetExclusions() const { return exclusions_; }

   private:
    std::set<MetadataFlag> unions_, intersections_, exclusions_;
  };

  Metadata() = default;
  // Include explicit destructor to get rid of CUDA __host__ __device__ warning
  ~Metadata() {}

  // There are 3 optional arguments: shape, component_labels, and associated, so we'll
  // need 8 constructors to provide all possible variants

  // By default shape is empty; this corresponds to scalar data on the mesh

  // 4 constructors, this is the general constructor called by all other constructors, so
  // we do some sanity checks here
  Metadata(const std::vector<MetadataFlag> &bits, const std::vector<int> &shape = {},
           const std::vector<std::string> &component_labels = {},
           const std::string &associated = "")
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
      refinement_funcs_ = refinement::RefinementFunctions_t::RegisterOps<
          refinement_ops::ProlongateCellMinMod, refinement_ops::RestrictCellAverage>();
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

  // 1 constructor
  Metadata(const std::vector<MetadataFlag> &bits, const std::vector<int> &shape,
           const std::string &associated)
      : Metadata(bits, shape, {}, associated) {}

  // 2 constructors
  Metadata(const std::vector<MetadataFlag> &bits,
           const std::vector<std::string> component_labels,
           const std::string &associated = "")
      : Metadata(bits, {1}, component_labels, associated) {}

  // 1 constructor
  Metadata(const std::vector<MetadataFlag> &bits, const std::string &associated)
      : Metadata(bits, {1}, {}, associated) {}

  // Static routines
  static MetadataFlag AddUserFlag(const std::string &name);
  static bool FlagNameExists(const std::string &flagname);
  static MetadataFlag GetUserFlag(const std::string &flagname);

  // Sparse threshold routines
  void SetSparseThresholds(parthenon::Real alloc, parthenon::Real dealloc,
                           parthenon::Real default_val = 0.0) {
    allocation_threshold_ = alloc;
    deallocation_threshold_ = dealloc;
    default_value_ = default_val;
  }

  parthenon::Real GetDeallocationThreshold() const { return deallocation_threshold_; }
  parthenon::Real GetAllocationThreshold() const { return allocation_threshold_; }
  parthenon::Real GetDefaultValue() const { return default_value_; }

  // Individual flag setters, using these could result in an invalid set of flags, use
  // IsValid to check if the flags are valid
  // TODO(JMM): This is dangerous. See Issue #844.
  void Set(MetadataFlag f) { DoBit(f, true); }    ///< Set specific bit
  void Unset(MetadataFlag f) { DoBit(f, false); } ///< Unset specific bit

  // Return true if the flags constraints are satisfied, false otherwise. If throw_on_fail
  // is true, throw a descriptive exception when invalid
  bool IsValid(bool throw_on_fail = false) const {
    bool valid = true;

    // Topology
    if (CountSet({None, Node, Edge, Face, Cell}) != 1) {
      valid = false;
      if (throw_on_fail) {
        PARTHENON_THROW("Exactly one topology flag must be set");
      }
    }

    // Role
    if (CountSet({Private, Provides, Requires, Overridable}) != 1) {
      valid = false;
      if (throw_on_fail) {
        PARTHENON_THROW("Exactly one role flag must be set");
      }
    }

    // Shape
    if (CountSet({Vector, Tensor}) > 1) {
      valid = false;
      if (throw_on_fail) {
        PARTHENON_THROW("At most one shape flag can be set");
      }
    }

    // Datatype
    if (CountSet({Boolean, Integer, Real}) != 1) {
      valid = false;
      if (throw_on_fail) {
        PARTHENON_THROW("Exactly one data type flag must be set");
      }
    }

    // Independent
    if (CountSet({Independent, Derived}) != 1) {
      valid = false;
      if (throw_on_fail) {
        PARTHENON_THROW("Either the Independent or Derived flag must be set");
      }
    }

    // Prolongation/restriction
    if (IsRefined()) {
      if (refinement_funcs_.label().size() == 0) {
        valid = false;
        if (throw_on_fail) {
          PARTHENON_THROW(
              "Registered for refinment but no prolongation/restriction ops found");
        }
      }
    }

    return valid;
  }

  /*--------------------------------------------------------*/
  // Getters for attributes
  /*--------------------------------------------------------*/
  // returns all set flags
  std::vector<MetadataFlag> Flags() const;

  /// returns the topological location of variable
  MetadataFlag Where() const {
    if (IsSet(Cell)) {
      return Cell;
    } else if (IsSet(Face)) {
      return Face;
    } else if (IsSet(Edge)) {
      return Edge;
    } else if (IsSet(Node)) {
      return Node;
    } else if (IsSet(None)) {
      return None;
    }

    PARTHENON_THROW("No topology flag set");
  }

  bool IsMeshTied() const { return (Where() != None); }

  /// returns the type of the variable
  MetadataFlag Type() const {
    if (IsSet(Boolean)) {
      return Boolean;
    } else if (IsSet(Integer)) {
      return Integer;
    } else if (IsSet(Real)) {
      return Real;
    }

    PARTHENON_THROW("No data type flag set");
  }

  MetadataFlag Role() const {
    if (IsSet(Private)) {
      return Private;
    } else if (IsSet(Provides)) {
      return Provides;
    } else if (IsSet(Requires)) {
      return Requires;
    } else if (IsSet(Overridable)) {
      return Overridable;
    }

    PARTHENON_THROW("No role flag set");
  }

  // Returns true if this variable should do prolongation/restriction
  // and false otherwise.
  bool IsRefined() const {
    return (IsSet(Independent) || IsSet(FillGhost) || IsSet(ForceRemeshComm));
  }

  const std::vector<int> &Shape() const { return shape_; }

  /*--------------------------------------------------------*/
  // Utility functions
  /*--------------------------------------------------------*/

  // get the dims of the 6D array
  std::array<int, 6> GetArrayDims(std::weak_ptr<MeshBlock> wpmb, bool coarse) const;

  /// Returns the attribute flags as a string of 1/0
  std::string MaskAsString() const {
    std::string str;
    for (auto const bit : bits_) {
      str += bit ? '1' : '0';
    }
    return str;
  }
  friend std::ostream &operator<<(std::ostream &os, const parthenon::Metadata &m);

  /**
   * @brief Returns true if any flag is set
   */
  template <template <class...> class Container_t, class... extra>
  bool AnyFlagsSet(const Container_t<MetadataFlag, extra...> &flags) const {
    return std::any_of(flags.begin(), flags.end(),
                       [this](MetadataFlag const &f) { return IsSet(f); });
  }
  template <typename... Args>
  bool AnyFlagsSet(const MetadataFlag &flag, Args... args) const {
    return AnyFlagsSet(FlagVec{flag, std::forward<Args>(args)...});
  }

  template <template <class...> class Container_t, class... extra>
  bool AllFlagsSet(const Container_t<MetadataFlag, extra...> &flags) const {
    return std::all_of(flags.begin(), flags.end(),
                       [this](MetadataFlag const &f) { return IsSet(f); });
  }
  template <typename... Args>
  bool AllFlagsSet(const MetadataFlag &flag, Args... args) const {
    return AllFlagsSet(FlagVec{flag, std::forward<Args>(args)...});
  }

  template <template <class...> class Container_t, class... extra>
  bool FlagsSet(const Container_t<MetadataFlag, extra...> &flags,
                bool matchAny = false) const {
    return ((matchAny && AnyFlagsSet(flags)) || ((!matchAny) && AllFlagsSet(flags)));
  }

  /// returns true if bit is set, false otherwise
  bool IsSet(MetadataFlag bit) const {
    return bit.flag_ < bits_.size() && bits_[bit.flag_];
  }

  // Refinement stuff
  const refinement::RefinementFunctions_t &GetRefinementFunctions() const {
    PARTHENON_REQUIRE_THROWS(IsRefined(), "Variable must be registered for refinement");
    return refinement_funcs_;
  }
  template <class ProlongationOp, class RestrictionOp>
  void RegisterRefinementOps() {
    PARTHENON_REQUIRE_THROWS(
        IsRefined(),
        "Variable must be registered for refinement to accept custom refinement ops");
    refinement_funcs_ =
        refinement::RefinementFunctions_t::RegisterOps<ProlongationOp, RestrictionOp>();
  }

  // Operators
  bool HasSameFlags(const Metadata &b) const {
    auto const &a = *this;

    // Check extra bits are unset
    auto const min_bits = std::min(a.bits_.size(), b.bits_.size());
    auto const &longer = a.bits_.size() > b.bits_.size() ? a.bits_ : b.bits_;
    for (auto i = min_bits; i < longer.size(); i++) {
      if (longer[i]) {
        // Bits are default false, so if any bit in the extraneous portion of the longer
        // bit list is set, then it cannot be equal to a.
        return false;
      }
    }

    for (size_t i = 0; i < min_bits; i++) {
      if (a.bits_[i] != b.bits_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const Metadata &b) const {
    return HasSameFlags(b) && (shape_ == b.shape_);

    // associated_ can be used by downstream codes to associate some variables with
    // others, and component_labels_ are used in output files to label components of
    // vectors/tensors. Both are not true metadata of a variable for the purposes of the
    // infrastructure and hence are not checked here
  }

  bool operator!=(const Metadata &b) const { return !(*this == b); }

  void Associate(const std::string &name) { associated_ = name; }
  const std::string &getAssociated() const { return associated_; }

  const std::vector<std::string> getComponentLabels() const noexcept {
    return component_labels_;
  }

 private:
  /// the attribute flags that are set for the class
  refinement::RefinementFunctions_t refinement_funcs_;
  std::vector<bool> bits_;
  std::vector<int> shape_ = {1};
  std::vector<std::string> component_labels_ = {};
  std::string associated_ = "";

  parthenon::Real allocation_threshold_;
  parthenon::Real deallocation_threshold_;
  parthenon::Real default_value_;

  /// if flag is true set bit, clears otherwise
  void DoBit(MetadataFlag bit, bool flag) {
    if (bit.flag_ >= bits_.size()) {
      bits_.resize(bit.flag_ + 1);
    }
    bits_[bit.flag_] = flag;
  }

  /// count the number of set flags from the given list
  int CountSet(const std::vector<MetadataFlag> &flags) const {
    int num = 0;
    for (const auto f : flags) {
      if (IsSet(f)) {
        ++num;
      }
    }
    return num;
  }
};

namespace MetadataUtils {
// From a given container, extract all variables whose Metadata matchs the all of the
// given flags (if the list of flags is empty, extract all variables), optionally only
// extracting sparse fields with an index from the given list of sparse indices
//
// JMM: This algorithm uses the map from metadata flags to variables
// to accelerate performance.
//
// The cost of this loop scales as O(Nflags * Nvars/flag) In worst
// case, this is linear in number of variables. However, on average,
// the number of vars with a desired flag will be much smaller than
// all vars. So average performance is much better than linear.
template <typename Set_t, typename NameMap_t, typename MetadataMap_t>
Set_t GetByFlag(const Metadata::FlagCollection &flags, NameMap_t &nameMap,
                MetadataMap_t &metadataMap) {
  // JMM: I'd like to make all inputs const, but apparently for (const
  // auto &thing : container) violates const correctness
  Set_t out;
  if (flags.Empty()) {
    for (const auto &p : nameMap) {
      out.insert(p.second);
    }
    return out;
  }
  const auto &intersections = flags.GetIntersections();
  const auto &unions = flags.GetUnions();
  const auto &exclusions = flags.GetExclusions();
  const bool check_excludes = exclusions.size() > 0;

  if (intersections.size() > 0) {
    // Dirty trick to get literally any flag from the intersections set
    MetadataFlag first_required = *(intersections.begin());

    for (const auto &v : metadataMap[first_required]) {
      const auto &m = v->metadata();
      // TODO(JMM): Note that AnyFlagsSet returns FALSE if the set of flags
      // it's asked about is empty.  Not sure that's desired
      // behaviour, but whatever, let's just guard against edge cases
      // here.
      if (m.AllFlagsSet(intersections) &&
          !(check_excludes && m.AnyFlagsSet(exclusions)) &&
          (unions.empty() || m.AnyFlagsSet(unions))) {
        // TODO(JMM): When dense sparse packing is moved to Parthenon
        // develop we need an extra check for IsAllocated here.
        out.insert(v);
      }
    }
  } else { // unions.size() > 0.
    for (const auto &f : unions) {
      for (const auto &v : metadataMap[f]) {
        // we know intersections.size == 0
        if (!(check_excludes && (v->metadata()).AnyFlagsSet(exclusions))) {
          // TODO(JMM): see above regarding IsAllocated()
          out.insert(v);
        }
      }
    }
  }
  return out;
}
} // namespace MetadataUtils

} // namespace parthenon

#endif // INTERFACE_METADATA_HPP_
