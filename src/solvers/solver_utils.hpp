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

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace solvers {

struct SparseMatrixAccessor {
  ParArray1D<int> ioff, joff, koff;
  ParArray1D<int> ioff_inv, joff_inv, koff_inv;
  ParArray1D<int> inv_entries;

  const int nstencil;
  int ndiag;
  SparseMatrixAccessor() : nstencil(0), ndiag(0) {}
  SparseMatrixAccessor(const SparseMatrixAccessor &sp)
      : ioff(sp.ioff), joff(sp.joff), koff(sp.koff), nstencil(sp.nstencil),
        ndiag(sp.ndiag), inv_entries(sp.inv_entries) {}
  SparseMatrixAccessor(const std::string &label, const int n,
                       std::vector<std::vector<int>> off)
      : ioff(label + "_ioff", n), joff(label + "_joff", n), koff(label + "_koff", n),
        ioff_inv(label + "_ioff_inv", n), joff_inv(label + "_joff_inv", n),
        koff_inv(label + "_koff_inv", n), inv_entries(label + "_inv_ent", n),
        nstencil(n) {
    PARTHENON_REQUIRE_THROWS(off.size() == 3,
                             "Offset array must have dimensions off[3][*]");
    PARTHENON_REQUIRE_THROWS(off[0].size() >= n, "Offset array off[0][*] too small");
    PARTHENON_REQUIRE_THROWS(off[1].size() >= n, "Offset array off[1][*] too small");
    PARTHENON_REQUIRE_THROWS(off[2].size() >= n, "Offset array off[2][*] too small");
    auto ioff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff);
    auto joff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff);
    auto koff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff);

    auto ioff_inv_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff_inv);
    auto joff_inv_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff_inv);
    auto koff_inv_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff_inv);

    auto inv_ent_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), inv_entries);

    for (int i = 0; i < n; i++) {
      ioff_h(i) = off[0][i];
      joff_h(i) = off[1][i];
      koff_h(i) = off[2][i];
      // this is inverse.
      ioff_inv_h(i) = -off[0][i];
      joff_inv_h(i) = -off[1][i];
      koff_inv_h(i) = -off[2][i];

      if (off[0][i] == 0 && off[1][i] == 0 && off[2][i] == 0) {
        ndiag = i;
      }
    }
    for (int i = 0; i < n; i++) {

      for (int j = 0; j < n; j++) {
        if (ioff_h(i) == ioff_inv_h(j) && joff_h(i) == joff_inv_h(j) &&
            koff_h(i) == koff_inv_h(j)) {
          inv_entries(i) = j;
          std::cout << "inv_entries:" << i << " " << j << std::endl;
        }
      } // j
    }   // i

    Kokkos::deep_copy(ioff, ioff_h);
    Kokkos::deep_copy(joff, joff_h);
    Kokkos::deep_copy(koff, koff_h);

    Kokkos::deep_copy(inv_entries, inv_ent_h);
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real MatVec(const PackType &spmat, const int imat_lo,
                                     const int imat_hi, const PackType &v, const int iv,
                                     const int b, const int k, const int j,
                                     const int i) const {
    Real matvec = 0.0;
    for (int n = imat_lo; n <= imat_hi; n++) {
      const int m = n - imat_lo;
      matvec += spmat(b, n, k, j, i) * v(b, iv, k + koff(m), j + joff(m), i + ioff(m));
    }
    return matvec;
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real Jacobi(const PackType &spmat, const int imat_lo,
                                     const int imat_hi, const PackType &v, const int iv,
                                     const int b, const int k, const int j, const int i,
                                     const Real rhs) const {
    const Real matvec = MatVec(spmat, imat_lo, imat_hi, v, iv, b, k, j, i);
    return (rhs - matvec + spmat(b, imat_lo + ndiag, k, j, i) * v(b, iv, k, j, i)) /
           spmat(b, imat_lo + ndiag, k, j, i);
  }
};

template <typename T>
struct Stencil {
  ParArray1D<T> w;
  ParArray1D<int> ioff, joff, koff;
  const int nstencil;
  int ndiag;
  Stencil() : nstencil(0), ndiag(0) {}
  Stencil(const Stencil<T> &st)
      : w(st.w), ioff(st.ioff), joff(st.joff), koff(st.koff), nstencil(st.nstencil),
        ndiag(st.ndiag) {}
  Stencil(const std::string &label, const int n, std::vector<T> wgt,
          std::vector<std::vector<int>> off)
      : w(label + "_w", n), ioff(label + "_ioff", n), joff(label + "_joff", n),
        koff(label + "_koff", n), nstencil(n) {
    PARTHENON_REQUIRE_THROWS(off.size() == 3,
                             "Offset array must have dimensions off[3][*]");
    PARTHENON_REQUIRE_THROWS(wgt.size() >= n, "Weight array too small");
    PARTHENON_REQUIRE_THROWS(off[0].size() >= n, "Offset array off[0][*] too small");
    PARTHENON_REQUIRE_THROWS(off[1].size() >= n, "Offset array off[1][*] too small");
    PARTHENON_REQUIRE_THROWS(off[2].size() >= n, "Offset array off[2][*] too small");
    auto w_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), w);
    auto ioff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff);
    auto joff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff);
    auto koff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff);

    for (int i = 0; i < n; i++) {
      w_h(i) = wgt[i];
      ioff_h(i) = off[0][i];
      joff_h(i) = off[1][i];
      koff_h(i) = off[2][i];
      if (off[0][i] == 0 && off[1][i] == 0 && off[2][i] == 0) {
        ndiag = i;
      }
    }

    Kokkos::deep_copy(w, w_h);
    Kokkos::deep_copy(ioff, ioff_h);
    Kokkos::deep_copy(joff, joff_h);
    Kokkos::deep_copy(koff, koff_h);
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real MatVec(const PackType &v, const int iv, const int b,
                                     const int k, const int j, const int i) const {
    Real matvec = 0.0;
    for (int n = 0; n < nstencil; n++) {
      matvec += w(n) * v(b, iv, k + koff(n), j + joff(n), i + ioff(n));
    }
    return matvec;
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real Jacobi(const PackType &v, const int iv, const int b,
                                     const int k, const int j, const int i,
                                     const Real rhs) const {
    const Real matvec = MatVec(v, iv, b, k, j, i);
    return (rhs - matvec + w(ndiag) * v(b, iv, k, j, i)) / w(ndiag);
  }
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_HPP_
