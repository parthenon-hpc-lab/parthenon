//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
VARIABLE(poisson, res_err);
VARIABLE(poisson, rhs);
VARIABLE(poisson, rhs_base);
VARIABLE(poisson, u);
VARIABLE(poisson, solution);
VARIABLE(poisson, temp);

VARIABLE(poisson, Am);
VARIABLE(poisson, Ac);
VARIABLE(poisson, Ap);

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus PrintValues(std::shared_ptr<MeshData<Real>> &md);
TaskStatus CalculateResidual(std::shared_ptr<MeshData<Real>> &md);
template <class x_t>
TaskStatus BlockLocalTriDiagX(std::shared_ptr<MeshData<Real>> &md);
TaskStatus CorrectRHS(std::shared_ptr<MeshData<Real>> &md);
TaskStatus BuildMatrix(std::shared_ptr<MeshData<Real>> &md);

template <class in, class out>
TaskStatus CopyData(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  auto desc = parthenon::MakePackDescriptor<in, out>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out(), k, j, i) = pack(b, te, in(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out>
TaskStatus AddFieldsAndStore(std::shared_ptr<MeshData<Real>> &md, Real wa = 1.0,
                             Real wb = 1.0) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  auto desc = parthenon::MakePackDescriptor<a_t, b_t, out>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out(), k, j, i) =
            wa * pack(b, te, a_t(), k, j, i) + wb * pack(b, te, b_t(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class var>
TaskStatus SetToZero(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<var>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, var(), k, j, i) = 0.0;
        // If fold expressions worked correctly in CUDA, we could make this
        // a variadic template <class... vars>, build the pack with all vars
        // at set everything requested to zero in the same kernel.
        // (pack(b, te, vars(), k, j, i) = 0.0, ...);
      });
  return TaskStatus::complete;
}

template <class in_t, class out_t>
TaskStatus JacobiIteration(std::shared_ptr<MeshData<Real>> &md, double weight,
                           int level) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<Am, Ac, Ap, rhs, in_t, out_t>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JacobiIteration", DevExecSpace(), 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out_t(), k, j, i) =
            weight *
                (pack(b, te, rhs(), k, j, i) -
                 pack(b, te, Am(), k, j, i) * pack(b, te, in_t(), k, j, i - 1) -
                 pack(b, te, Ap(), k, j, i) * pack(b, te, in_t(), k, j, i + 1)) /
                pack(b, te, Ac(), k, j, i) +
            (1.0 - weight) * pack(b, te, in_t(), k, j, i);
        printf("Jacobi: b = %i i = %2i in[i+-1] = (%e, %e, %e) out[i] = %e rhs[i] = %e\n",
               b, i, pack(b, te, in_t(), k, j, i - 1), pack(b, te, in_t(), k, j, i),
               pack(b, te, in_t(), k, j, i + 1), pack(b, te, out_t(), k, j, i),
               pack(b, te, rhs(), k, j, i));
      });
  printf("\n");
  return TaskStatus::complete;
}

template <class... vars>
TaskStatus PrintChosenValues(std::shared_ptr<MeshData<Real>> &md, std::string &label) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<vars...>(md.get());
  auto pack = desc.GetPack(md.get());
  std::array<std::string, sizeof...(vars)> names{vars::name()...};
  printf("%s\n", label.c_str());
  int col_num = 0;
  for (auto &name : names) {
    printf("var %i: %s\n", col_num, name.c_str());
    col_num++;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x = coords.template X<1, te>(i);
        std::array<Real, sizeof...(vars)> vals{pack(b, te, vars(), k, j, i)...};
        printf("b = %i i = %2i x = %e", b, i, x);
        for (int v = 0; v < sizeof...(vars); ++v) {
          printf("%e ", vals[v]);
        }
        printf("\n");
      });
  printf("Done with MeshData\n\n");
  return TaskStatus::complete;
}

} // namespace poisson_package

#endif // EXAMPLE_POISSON_GMG_POISSON_PACKAGE_HPP_
