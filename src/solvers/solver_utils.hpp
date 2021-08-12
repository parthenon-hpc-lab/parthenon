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
#ifndef SOLVERS_SOLVER_UTILS_HPP_
#define SOLVERS_SOLVER_UTILS_HPP_

#include <vector>

#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace solvers {

template <typename T>
struct Stencil {
  ParArray1D<T> w;
  ParArray1D<int> ioff, joff, koff;
  const int nstencil;

  Stencil(const std::string &label, const int n, std::vector<T> wgt,
    std::vector<std::vector<int>> off)
    : w(label+"_w", n), ioff(label+"_ioff", n), joff(label+"_joff", n),
      koff(label+"_koff", n), nstencil(n) {
    assert(off.size() == 3);
    assert(wgt.size() >= n);
    assert(off[0].size() >= n);
    assert(off[1].size() >= n);
    assert(off[2].size() >= n);
    auto w_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), w);
    auto ioff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff);
    auto joff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff);
    auto koff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff);

    for (int i = 0; i < n; i++) {
      w_h(i) = wgt[i];
      ioff_h(i) = off[0][i];
      joff_h(i) = off[1][i];
      koff_h(i) = off[2][i];
    }

    Kokkos::deep_copy(w, w_h);
    Kokkos::deep_copy(ioff, ioff_h);
    Kokkos::deep_copy(joff, joff_h);
    Kokkos::deep_copy(koff, koff_h);
  }
};

template <typename PackType, typename T>
void StencilMatVec(const PackType &vin, const int ivin,
                   const PackType &vout, const int ivout,
                   const PackType &rhs, const int irhs, const Stencil<T> &s,
                   const IndexRange &ib, const IndexRange &jb, const IndexRange &kb) {
  parthenon::par_for(DEFAULT_LOOP_PATTERN, "StencilMatVec", DevExecSpace(),
    0, vout.GetDim(5)-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      vout(b, ivout, k, j, i) = rhs(b, irhs, k, j, i);
      for (int n = 0; n < s.nstencil; n++) {
        vout(b, ivout, k, j, i) +=
          s.w(n)*vin(b, ivin, k+s.koff(n), j+s.joff(n), i+s.ioff(n));
      }
    });
}


} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_HPP_
