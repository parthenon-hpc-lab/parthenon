//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_POISSON_GMG_POISSON_PACKAGE_HPP_
#define EXAMPLE_POISSON_GMG_POISSON_PACKAGE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace poisson_package {
using namespace parthenon::package::prelude;

VARIABLE(poisson, D);
VARIABLE(poisson, u);
VARIABLE(poisson, rhs);
VARIABLE(poisson, exact);

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <class... vars>
TaskStatus PrintChosenValues(std::shared_ptr<MeshData<Real>> &md,
                             const std::string &label) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  auto desc = parthenon::MakePackDescriptor<vars...>(md.get());
  auto pack = desc.GetPack(md.get());
  std::array<std::string, sizeof...(vars)> names{vars::name()...};
  printf("%s\n", label.c_str());
  int col_num = 0;
  for (auto &name : names) {
    printf("var %i: %s\n", col_num, name.c_str());
    col_num++;
  }
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Print", DevExecSpace(), 0, pack.GetNBlocks() - 1, 0, 0, 0, 0,
      KOKKOS_LAMBDA(const int b, int, int) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::entire, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::entire, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::entire, te);
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
            for (int i = ib.s; i <= ib.e; ++i) {
              Real x = coords.template X<1, te>(i);
              Real y = coords.template X<2, te>(j);
              Real dx1 = coords.template Dxc<1>(k, j, i);
              Real dx2 = coords.template Dxc<2>(k, j, i);
              std::array<Real, sizeof...(vars)> vals{pack(b, te, vars(), k, j, i)...};
              printf("b = %i i = %2i x = %e dx1 = %e ", b, i, x, dx1);
              for (int v = 0; v < sizeof...(vars); ++v) {
                printf("%e ", vals[v]);
              }
              printf("\n");
            }
          }
        }
      });
  printf("Done with MeshData\n\n");
  return TaskStatus::complete;
}

} // namespace poisson_package

#endif // EXAMPLE_POISSON_GMG_POISSON_PACKAGE_HPP_
