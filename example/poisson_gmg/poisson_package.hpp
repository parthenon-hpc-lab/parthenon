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
TaskStatus RMSResidual(std::shared_ptr<MeshData<Real>> &md, std::string label);

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
  auto desc = parthenon::MakePackDescriptor<var>(md.get());
  auto pack = desc.GetPack(md.get());
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Print", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        parthenon::par_for_inner(
            parthenon::inner_loop_pattern_simdfor_tag, member, kb.s, kb.e, jb.s, jb.e,
            ib.s, ib.e, [&](int k, int j, int i) { pack(b, te, var(), k, j, i) = 0.0; });
      });
  return TaskStatus::complete;
}

template <class in_t, class out_t>
TaskStatus JacobiIteration(std::shared_ptr<MeshData<Real>> &md, double weight,
                           int level) {
  const int ndim = md->GetMeshPointer()->ndim;
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
        Real val = pack(b, te, rhs(), k, j, i);
        val -= pack(b, te, Am(0), k, j, i) * pack(b, te, in_t(), k, j, i - 1) +
               pack(b, te, Ap(0), k, j, i) * pack(b, te, in_t(), k, j, i + 1);
        if (ndim > 1) {
          val -= pack(b, te, Am(1), k, j, i) * pack(b, te, in_t(), k, j - 1, i) +
                 pack(b, te, Ap(1), k, j, i) * pack(b, te, in_t(), k, j + 1, i);
        }
        if (ndim > 2) {
          val -= pack(b, te, Am(2), k, j, i) * pack(b, te, in_t(), k - 1, j, i) +
                 pack(b, te, Ap(2), k, j, i) * pack(b, te, in_t(), k + 1, j, i);
        }
        pack(b, te, out_t(), k, j, i) = weight * val / pack(b, te, Ac(), k, j, i) +
            (1.0 - weight) * pack(b, te, in_t(), k, j, i);
        printf("b =  %i i = %i stencil = (%e, %e, %e) rhs = %e A = (%e, %e, %e)\n", b, i, 
              pack(b, te, in_t(), k, j, i - 1), pack(b, te, in_t(), k, j, i), pack(b, te, in_t(), k, j, i + 1),
              pack(b, te, rhs(), k, j, i), pack(b, te, Am(), k, j, i), pack(b, te, Ac(), k, j, i), pack(b, te, Ap(), k, j, i));
      });
  printf("\n");
  return TaskStatus::complete;
}

template <class... vars>
TaskStatus PrintChosenValues(std::shared_ptr<MeshData<Real>> &md, const std::string &label) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  auto desc = parthenon::MakePackDescriptor<vars...>(md.get());
  auto pack = desc.GetPack(md.get());
  std::array<std::string, sizeof...(vars)> names{vars::name()...};
  printf("%s\n", label.c_str());
  int col_num = 0;
  for (auto &name : names) {
    printf("var %i: %s\n", col_num, name.c_str());
    col_num++;
  }
  //printf("i=[%i, %i] j=[%i, %i] k=[%i, %i]\n", ib.s, ib.e, jb.s, jb.e, kb.s, kb.e);
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Print", DevExecSpace(), 0, pack.GetNBlocks() - 1, 0, 0, 0, 0,
      KOKKOS_LAMBDA(const int b, int, int) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        //printf("b=%i i=[%i, %i] j=[%i, %i] k=[%i, %i]\n", b, ib.s, ib.e, jb.s, jb.e, kb.s,
        //       kb.e);
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
            for (int i = ib.s; i <= ib.e; ++i) {
              Real x = coords.template X<1, te>(i);
              Real y = coords.template X<2, te>(j);
              Real dx1 = coords.template Dxc<1>(k, j, i);
              Real dx2 = coords.template Dxc<2>(k, j, i);
              std::array<Real, sizeof...(vars)> vals{pack(b, te, vars(), k, j, i)...};
              printf("b = %i i = %2i j = %2i x = %e y = %e dx1 = %e dx2 = %e ", b, i, j,
                     x, y, dx1, dx2);
              for (int v = 0; v < sizeof...(vars); ++v) {
                printf("%e ", vals[v]);
              }
              printf("\n");
            }
          }
        }
      });
  /*
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Print", DevExecSpace(),
      scratch_size_in_bytes, scratch_level,
      0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        printf("b=%i i=[%i, %i] j=[%i, %i] k=[%i, %i]\n", b, ib.s, ib.e, jb.s, jb.e, kb.s,
  kb.e); parthenon::par_for_inner(parthenon::inner_loop_pattern_simdfor_tag, member, kb.s,
  kb.e, jb.s, jb.e, ib.s, ib.e,
                                 [&](int k, int j, int i) {
           // Work here
        });
      });
      */
  /*
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x = coords.template X<1, te>(i);
        Real y = coords.template X<2, te>(j);
        std::array<Real, sizeof...(vars)> vals{pack(b, te, vars(), k, j, i)...};
        printf("b = %i i = %2i j = %2i x = %e y = %e x + 10*y = %e ", b, i, j, x, y, x
  + 10.0*y); for (int v = 0; v < sizeof...(vars); ++v) { printf("%e ", vals[v]);
        }
        printf("\n");
      });
  */
  printf("Done with MeshData\n\n");
  return TaskStatus::complete;
}

} // namespace poisson_package

#endif // EXAMPLE_POISSON_GMG_POISSON_PACKAGE_HPP_
