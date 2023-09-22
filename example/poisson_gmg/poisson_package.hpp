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
VARIABLE(poisson, res_err);
VARIABLE(poisson, rhs);
VARIABLE(poisson, rhs_base);
VARIABLE(poisson, u);
VARIABLE(poisson, u0);
VARIABLE(poisson, uctof);
VARIABLE(poisson, solution);
VARIABLE(poisson, temp);
VARIABLE(poisson, r);
VARIABLE(poisson, p);
VARIABLE(poisson, x);
VARIABLE(poisson, exact);
VARIABLE(poisson, Adotp);
VARIABLE(poisson, D);
VARIABLE(poisson, rhat0);
VARIABLE(poisson, v);
VARIABLE(poisson, s);
VARIABLE(poisson, t);
VARIABLE(poisson, h);

VARIABLE(poisson, Am);
VARIABLE(poisson, Ac);
VARIABLE(poisson, Ap);

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus PrintValues(std::shared_ptr<MeshData<Real>> &md);
TaskStatus CalculateResidual(std::shared_ptr<MeshData<Real>> &md);
// template <class x_t>
// TaskStatus BlockLocalTriDiagX(std::shared_ptr<MeshData<Real>> &md);
// TaskStatus CorrectRHS(std::shared_ptr<MeshData<Real>> &md);
// TaskStatus RMSResidual(std::shared_ptr<MeshData<Real>> &md, std::string label);

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

template <class in, class out>
TaskStatus CopyBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);
  IndexRange ibi = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jbi = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kbi = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<in, out>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i < ibi.s || i > ibi.e || j < jbi.s || j > jbi.e || k < kbi.s || k > kbi.e)
          pack(b, te, out(), k, j, i) = pack(b, te, in(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out>
TaskStatus AddFieldsAndStoreInteriorSelect(std::shared_ptr<MeshData<Real>> &md,
                                           Real wa = 1.0, Real wb = 1.0,
                                           bool only_interior = false) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior) {
    // The neighbors array will only be set for a block if its a leaf block
    for (int b = 0; b < nblocks; ++b)
      include_block[b] = md->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
  }

  auto desc = parthenon::MakePackDescriptor<a_t, b_t, out>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out(), k, j, i) =
            wa * pack(b, te, a_t(), k, j, i) + wb * pack(b, te, b_t(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out>
TaskStatus AddFieldsAndStore(std::shared_ptr<MeshData<Real>> &md, Real wa = 1.0,
                             Real wb = 1.0) {
  return AddFieldsAndStoreInteriorSelect<a_t, b_t, out>(md, wa, wb, false);
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

template <class a_t, class b_t>
TaskStatus DotProductLocal(std::shared_ptr<MeshData<Real>> &md, Real *reduce_sum) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<a_t, b_t>(md.get());
  auto pack = desc.GetPack(md.get());
  Real gsum(0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        lsum += pack(b, te, a_t(), k, j, i) * pack(b, te, b_t(), k, j, i);
      },
      Kokkos::Sum<Real>(gsum));
  *reduce_sum += gsum;
  return TaskStatus::complete;
}

template <class var_t>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  using TE = parthenon::TopologicalElement;

  auto desc = parthenon::MakePackDescriptor<var_t, D>(md.get(), {}, {PDOpt::WithFluxes});
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CaclulateFluxes", DevExecSpace(), 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
        pack.flux(b, X1DIR, var_t(), k, j, i) =
            pack(b, TE::F1, D(), k, j, i) / dx1 *
            (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i));
        if (i == ib.e)
          pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
              pack(b, TE::F1, D(), k, j, i + 1) / dx1 *
              (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1));

        if (ndim > 1) {
          Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          pack.flux(b, X2DIR, var_t(), k, j, i) =
              pack(b, TE::F2, D(), k, j, i) *
              (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) / dx2;
          if (j == jb.e)
            pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                pack(b, TE::F2, D(), k, j + 1, i) *
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) / dx2;
        }

        if (ndim > 2) {
          Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
          pack.flux(b, X3DIR, var_t(), k, j, i) =
              pack(b, TE::F3, D(), k, j, i) *
              (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) / dx3;
          if (k == kb.e)
            pack.flux(b, X2DIR, var_t(), k + 1, j, i) =
                pack(b, TE::F3, D(), k + 1, j, i) *
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) / dx3;
        }
        // printf("b = %i i = %i flux = %e (%e - %e)\n", b, i, pack.flux(b, X1DIR,
        // var_t(), k, j, i), pack(b, te, var_t(), k, j, i - 1), pack(b, te, var_t(), k,
        // j, i)); if (i==ib.e)
        //   printf("b = %i i = %i flux = %e (%e - %e)\n", b, i + 1, pack.flux(b, X1DIR,
        //   var_t(), k, j, i + 1), pack(b, te, var_t(), k, j, i), pack(b, te, var_t(), k,
        //   j, i + 1));
      });
  // printf("\n");
  return TaskStatus::complete;
}

template <class in_t, class out_t>
TaskStatus FluxMultiplyMatrix(std::shared_ptr<MeshData<Real>> &md, bool only_interior) {
  using namespace parthenon;
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto pkg = md->GetMeshPointer()->packages.Get("poisson_package");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior) {
    for (int b = 0; b < nblocks; ++b)
      include_block[b] = md->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
  }

  auto desc =
      parthenon::MakePackDescriptor<in_t, out_t>(md.get(), {}, {PDOpt::WithFluxes});
  auto pack = desc.GetPack(md.get(), include_block);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FluxMultiplyMatrix", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
        pack(b, te, out_t(), k, j, i) = -alpha * pack(b, te, in_t(), k, j, i);
        pack(b, te, out_t(), k, j, i) += (pack.flux(b, X1DIR, in_t(), k, j, i) -
                                          pack.flux(b, X1DIR, in_t(), k, j, i + 1)) /
                                         dx1;

        if (ndim > 1) {
          Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          pack(b, te, out_t(), k, j, i) += (pack.flux(b, X2DIR, in_t(), k, j, i) -
                                            pack.flux(b, X2DIR, in_t(), k, j + 1, i)) /
                                           dx2;
        }
        // if ((b == 2 && (i > ib.e - 5) && (j == jb.s + 64)) ||
        //     (b == 3 && (i < ib.s + 5) && (j == jb.e))) {
        //   printf("b = %i i=%i j=%i FxL * dx = %e FxR * dx = %e in = (%e, %e, %e) out =
        //   %e\n", b, i, j,
        //                                                         dx1 * pack.flux(b,
        //                                                         X1DIR, in_t(), k, j,
        //                                                         i), dx1 * pack.flux(b,
        //                                                         X1DIR, in_t(), k, j, i
        //                                                         + 1), pack(b, te,
        //                                                         in_t(), k, j, i - 1),
        //                                                         pack(b, te, in_t(), k,
        //                                                         j, i), pack(b, te,
        //                                                         in_t(), k, j, i + 1),
        //                                                         pack(b, te, out_t(), k,
        //                                                         j, i));
        // }

        if (ndim > 2) {
          Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
          pack(b, te, out_t(), k, j, i) += (pack.flux(b, X3DIR, in_t(), k, j, i) -
                                            pack.flux(b, X3DIR, in_t(), k + 1, j, i)) /
                                           dx3;
        }
      });
  return TaskStatus::complete;
}

enum class GSType { all, red, black };
template <class div_t, class in_t, class out_t>
TaskStatus FluxJacobi(std::shared_ptr<MeshData<Real>> &md, double weight,
                      GSType gs_type = GSType::all) {
  using namespace parthenon;
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto pkg = md->GetMeshPointer()->packages.Get("poisson_package");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");

  auto desc = parthenon::MakePackDescriptor<in_t, out_t, div_t, rhs, D>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CaclulateFluxes", DevExecSpace(), 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        if ((i + j + k) % 2 == 1 && gs_type == GSType::red) return;
        if ((i + j + k) % 2 == 0 && gs_type == GSType::black) return;
        // Build the unigrid diagonal of the matrix
        Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
        Real diag_elem =
            -(pack(b, TE::F1, D(), k, j, i) + pack(b, TE::F1, D(), k, j, i + 1)) /
                (dx1 * dx1) -
            alpha;
        if (ndim > 1) {
          Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          diag_elem -=
              (pack(b, TE::F2, D(), k, j, i) + pack(b, TE::F2, D(), k, j + 1, i)) /
              (dx2 * dx2);
        }
        if (ndim > 2) {
          Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
          diag_elem -=
              (pack(b, TE::F3, D(), k, j, i) + pack(b, TE::F3, D(), k + 1, j, i)) /
              (dx3 * dx3);
        }

        // Get the off-diagonal contribution to Ax = (D + L + U)x = y
        Real off_diag =
            pack(b, te, div_t(), k, j, i) - diag_elem * pack(b, te, in_t(), k, j, i);

        Real val = pack(b, te, rhs(), k, j, i) - off_diag;
        pack(b, te, out_t(), k, j, i) =
            weight * val / diag_elem + (1.0 - weight) * pack(b, te, in_t(), k, j, i);
      });
  return TaskStatus::complete;
}

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
