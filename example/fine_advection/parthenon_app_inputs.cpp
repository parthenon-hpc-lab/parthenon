//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "advection_driver.hpp"
#include "advection_package.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//
namespace {
KOKKOS_INLINE_FUNCTION
Real sign(Real a, Real b) {
  if (a == 0.0) {
    return b > 0.0 ? 1.0 : -1.0;
  }
  return a > 0.0 ? 1.0 : -1.0;
}
} // namespace
namespace advection_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto &data = pmb->meshblock_data.Get();

  auto pkg = pmb->packages.Get("advection_package");
  const auto &profile = pkg->Param<std::string>("profile");
  auto do_regular_advection = pkg->Param<bool>("do_regular_advection");
  auto do_fine_advection = pkg->Param<bool>("do_fine_advection");
  auto do_CT_advection = pkg->Param<bool>("do_CT_advection");

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;

  using namespace advection_package::Conserved;
  if (do_regular_advection) {
    const int sparse_size = pkg->Param<int>("sparse_size");
    for (int s = 0; s < sparse_size; ++s)
      pmb->AllocSparseID(phi::name(), s);
  }
  static auto desc = parthenon::MakePackDescriptor<phi, phi_fine, C, D>(data.get());
  auto pack = desc.GetPack(data.get());

  int profile_type;
  if (profile == "wave") profile_type = 0;
  if (profile == "smooth_gaussian") profile_type = 1;
  if (profile == "hard_sphere") profile_type = 2;
  if (profile == "block") profile_type = 3;
  Real amp = 1.0;
  const int b = 0;
  const int ndim = pmb->pmy_mesh->ndim;
  const int nghost = parthenon::Globals::nghost;

  if (do_regular_advection) {
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, pack.GetLowerBoundHost(b, phi()),
        pack.GetUpperBoundHost(b, phi()), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          const int kf = ndim > 2 ? (k - nghost) * 2 + nghost : k;
          const int jf = ndim > 1 ? (j - nghost) * 2 + nghost : j;
          const int fi = ndim > 0 ? (i - nghost) * 2 + nghost : i;
          if (profile_type == 0) {
            PARTHENON_FAIL("Profile type wave not implemented.");
          } else if (profile_type == 1) {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                       coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            pack(b, phi(l), k, j, i) = 1. + amp * exp(-100.0 * rsq);
          } else if (profile_type == 2) {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                       coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            pack(b, phi(l), k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          } else {
            pack(b, phi(l), k, j, i) = 0.0;
          }
        });
  }

  if (do_fine_advection) {
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int kf = ndim > 2 ? (k - nghost) * 2 + nghost : k;
          const int jf = ndim > 1 ? (j - nghost) * 2 + nghost : j;
          const int fi = ndim > 0 ? (i - nghost) * 2 + nghost : i;
          if (profile_type == 0) {
            PARTHENON_FAIL("Profile type wave not implemented.");
          } else if (profile_type == 1) {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                       coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            for (int koff = 0; koff <= (ndim > 2); ++koff)
              for (int joff = 0; joff <= (ndim > 1); ++joff)
                for (int ioff = 0; ioff <= (ndim > 0); ++ioff) {
                  pack(b, phi_fine(), kf + koff, jf + joff, fi + ioff) =
                      1. + amp * exp(-100.0 * rsq);
                }
          } else if (profile_type == 2) {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                       coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            for (int koff = 0; koff <= (ndim > 2); ++koff)
              for (int joff = 0; joff <= (ndim > 1); ++joff)
                for (int ioff = 0; ioff <= (ndim > 0); ++ioff) {
                  pack(b, phi_fine(), kf + koff, jf + joff, fi + ioff) =
                      (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
                }
          } else {
            for (int koff = 0; koff <= (ndim > 2); ++koff)
              for (int joff = 0; joff <= (ndim > 1); ++joff)
                for (int ioff = 0; ioff <= (ndim > 0); ++ioff) {
                  pack(b, phi_fine(), kf + koff, jf + joff, fi + ioff) = 0.0;
                }
          }
        });
  }

  if (do_CT_advection) {
    ib = cellbounds.GetBoundsI(IndexDomain::interior, TE::F2);
    jb = cellbounds.GetBoundsJ(IndexDomain::interior, TE::F2);
    kb = cellbounds.GetBoundsK(IndexDomain::interior, TE::F2);
    const Real x0 = 0.2;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          auto &coords = pack.GetCoordinates(b);
          Real xlo = coords.X<X1DIR, TE::F1>(k, j, i);
          Real xhi = coords.X<X1DIR, TE::F1>(k, j, i + 1);
          Real Alo = std::abs(xlo) < x0 ? sign(xlo, xhi) * (x0 - std::abs(xlo)) : 0.0;
          Real Ahi = std::abs(xhi) < x0 ? sign(xhi, xlo) * (x0 - std::abs(xhi)) : 0.0;
          pack(b, TE::F2, C(), k, j, i) = (Alo - Ahi) / (xhi - xlo);
        });

    ib = cellbounds.GetBoundsI(IndexDomain::interior, TE::F3);
    jb = cellbounds.GetBoundsJ(IndexDomain::interior, TE::F3);
    kb = cellbounds.GetBoundsK(IndexDomain::interior, TE::F3);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          auto &coords = pack.GetCoordinates(b);
          Real xlo = coords.X<X1DIR, TE::F1>(k, j, i);
          Real xhi = coords.X<X1DIR, TE::F1>(k, j, i + 1);
          Real Alo = std::abs(xlo) < x0 ? sign(xlo, xhi) * (x0 - std::abs(xlo)) : 0.0;
          Real Ahi = std::abs(xhi) < x0 ? sign(xhi, xlo) * (x0 - std::abs(xhi)) : 0.0;
          pack(b, TE::F3, D(), k, j, i) = (Alo - Ahi) / (xhi - xlo);
        });
  }
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = advection_package::Initialize(pin.get());
  packages.Add(pkg);

  auto app = std::make_shared<StateDescriptor>("advection_app");
  packages.Add(app);

  return packages;
}

} // namespace advection_example
