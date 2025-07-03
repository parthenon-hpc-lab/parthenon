#include <catch2/catch.hpp>

#include "utils/type_arrays.hpp"

namespace {
struct v1 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v1(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v1"; }
};

struct v2 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v2(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v2"; }
};

struct v3 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v3(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v2"; }
};
} // namespace

TEST_CASE("Test behavior of type based array.") {

  GIVEN("A known typelist of 1-sized variables") {
    using TL = parthenon::TypeList<v1, v2, v3>;
    auto tla = parthenon::TypeListArray(TL());

    // initialize with the variable indices
    tla(v1()) = 1.;
    tla(v2()) = 4.;
    tla(v3()) = 9.;
    for (int idx = 0; idx < TL::n_types; idx++) {
      REQUIRE(tla[idx] == (idx + 1) * (idx + 1));
    }
  }
}
