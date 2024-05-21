//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

namespace advection_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto &data = pmb->meshblock_data.Get();

  auto pkg = pmb->packages.Get("advection_package");
  const auto &profile = pkg->Param<std::string>("profile");

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  
  using scalar = advection_package::Conserved::scalar;
  using scalar_fine = advection_package::Conserved::scalar_fine;
  static auto desc = parthenon::MakePackDescriptor<scalar, scalar_fine>(data.get()); 
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
  pmb->par_for(
      PARTHENON_AUTO_LABEL,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const int kf = ndim > 2 ? (k - nghost) * 2 + nghost : k; 
        const int jf = ndim > 1 ? (j - nghost) * 2 + nghost : j; 
        const int fi = ndim > 0 ? (i - nghost) * 2 + nghost : i; 
        if (profile_type == 0) {
          PARTHENON_FAIL("Fuxklg you1"); 
        } else if (profile_type == 1) {
          Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                     coords.Xc<2>(j) * coords.Xc<2>(j) +
                     coords.Xc<3>(k) * coords.Xc<3>(k);
          pack(b, scalar(), k, j, i) = 1. + amp * exp(-100.0 * rsq);
          for (int ioff = 0; ioff <= (ndim > 0); ++ioff)
          for (int joff = 0; joff <= (ndim > 1); ++joff)
          for (int koff = 0; koff <= (ndim > 2); ++koff) {
            pack(b, scalar_fine(), kf + koff, jf + joff, fi + ioff) = 1. + amp * exp(-100.0 * rsq);
          }
        } else if (profile_type == 2) {
          Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) +
                     coords.Xc<2>(j) * coords.Xc<2>(j) +
                     coords.Xc<3>(k) * coords.Xc<3>(k);
          pack(b, scalar(), k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          for (int ioff = 0; ioff <= (ndim > 0); ++ioff)
          for (int joff = 0; joff <= (ndim > 1); ++joff)
          for (int koff = 0; koff <= (ndim > 2); ++koff) {
            pack(b, scalar_fine(), kf + koff, jf + joff, fi + ioff) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          }
        } else {
          pack(b, scalar(), k, j, i) = 0.0;
          for (int ioff = 0; ioff <= (ndim > 0); ++ioff)
          for (int joff = 0; joff <= (ndim > 1); ++joff)
          for (int koff = 0; koff <= (ndim > 2); ++koff) {
            pack(b, scalar_fine(), kf + koff, jf + joff, fi + ioff) = 0.0;
          }
        }
      });

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
