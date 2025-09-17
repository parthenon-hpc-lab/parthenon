#ifndef PACK_SCRATCH_VARIABLES_HPP_
#define PACK_SCRATCH_VARIABLES_HPP_

#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "pack/pack_utils.hpp"
#include "utils/type_list.hpp"

namespace parthenon {
KOKKOS_INLINE_FUNCTION constexpr auto TopologicalTypeToMetaData(TopologicalType tt) {
  using TT = TopologicalType;
  if (tt == TT::Face) {
    return Metadata::Face;
  } else if (tt == TT::Edge) {
    return Metadata::Edge;
  } else if (tt == TT::Node) {
    return Metadata::Node;
  }
  return Metadata::Cell;
}

inline std::string TopologicalTypeToString(TopologicalType tt) {
  using TT = TopologicalType;
  if (tt == TT::Face) {
    return "face";
  } else if (tt == TT::Edge) {
    return "edge";
  } else if (tt == TT::Node) {
    return "node";
  }
  return "cell";
}

inline std::string range_regex(unsigned a, unsigned b) {
  std::ostringstream pattern;
  pattern << "((" << std::to_string(a) << ")";
  for (int i = a + 1; i <= b; i++) {
    pattern << "|(" << std::to_string(i) << ")";
  }
  pattern << ")";
  return pattern.str();
}

template <TopologicalType TT, int... NCOMPS>
struct ScratchVariable {
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
};

#define SCRATCH_VARIABLE(var_name, TT, ...)                                              \
  struct var_name##_t : public ScratchVariable<TT, __VA_ARGS__> {                        \
    static std::string name() { return #var_name; }                                      \
  };

template <typename SV, int lower>
struct ScratchVariable_impl : public SV::base_t {
  using type = SV;
  static constexpr int lb = lower;
  static constexpr int ub = lower + SV::size - 1;
  static constexpr auto shape = SV::shape;

  template <class... Ts>
  KOKKOS_INLINE_FUNCTION ScratchVariable_impl(Ts &&...args)
      : SV::base_t(std::forward<Ts>(args)...) {}

  static std::string name() {
#ifdef PARTHENON_DEBUG_SCRATCH
    return "scratch_" + SV::Name();
#else
    return "scratch_" + TopologicalTypeToString(SV::type) + "_" + range_regex(lb, ub);
#endif
  }
};

namespace impl {
template <typename...>
struct SVList_impl {};

template <typename SV>
struct SVList_impl<SV> {
  using type = ScratchVariable_impl<SV, 0>;
  using value = TypeList<type>;
};

template <typename SV, typename... SVs>
struct SVList_impl<SV, SVs...> {
  using list = SVList_impl<SVs...>;
  using type = ScratchVariable_impl<SV, list::type::ub + 1>;
  using value = concatenate_type_lists_t<TypeList<type>, typename list::value>;
};
} // namespace impl

// Gives a tuv index into the common scratch data for a given TopologicalType
// by using an agreed upon pool of scratch_TT_# overrideable var names
// that way the total memory allocated across all packages is the maximum
// size of any single ScratchVariableList for a given TT, but allows
// for each package to index into the common space with their own
// unique types & sizes
template <typename V, typename... SVs>
struct ScratchVariableList {
  static constexpr TopologicalType TT = V::type;
  static constexpr int n_vars = V::size + (SVs::size + ... + 0);
  using TL = TypeList<V, SVs...>;
  using list = impl::SVList_impl<V, SVs...>;

  template <typename SV>
  using type = typename list::value::template type<TL::template GetIdx<SV>()>;

  static const auto GetVarNames() {
    std::array<std::string, n_vars> vars;
    auto base = "scratch_" + TopologicalTypeToString(TT) + "_";
    for (int i = 0; i < n_vars; i++) {
      vars[i] = base + std::to_string(i);
    }
    return vars;
  }
};

namespace impl {
template <typename... Ts>
void AddScratch(ScratchVariableList<Ts...>, StateDescriptor *pkg) {
  using SL = ScratchVariableList<Ts...>;
  (
      [&] {
        auto m = Metadata(
            {TopologicalTypeToMetaData(SL::TT), Metadata::Derived, Metadata::Overridable},
            std::vector<int>(std::begin(Ts::shape), std::end(Ts::shape)));
        pkg->AddField<Ts>(m);
      }(),
      ...);
}
} // namespace impl

template <typename SL>
void AddScratch(StateDescriptor *pkg) {
#ifdef PARTHENON_DEBUG_SCRATCH
  // in debug mode each scratch variable has its own unique name
  impl::AddScratch(SL(), pkg);
#else
  auto m = Metadata(
      {TopologicalTypeToMetaData(SL::TT), Metadata::Derived, Metadata::Overridable});
  for (const auto var : SL::GetVarNames()) {
    pkg->AddField(var, m);
  }
#endif
}

} // namespace parthenon
#endif // PACK_SCRATCH_VARIABLES_HPP_
