//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include "burgers_driver.hpp"
#include "burgers_package.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/variable_pack.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace burgers_benchmark {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  const Real kx_fact = pin->GetOrAddReal("burgers", "kx_fact", 1.0);
  const Real ky_fact = pin->GetOrAddReal("burgers", "ky_fact", 1.0);
  const Real kz_fact = pin->GetOrAddReal("burgers", "kz_fact", 1.0);

  auto &data = pmb->meshblock_data.Get();

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  PackIndexMap index_map;
  auto q =
      data->PackVariables(std::vector<MetadataFlag>{Metadata::Independent}, index_map);
  const auto num_vars = q.GetDim(4);

  pmb->par_for(
      "Burgers::ProblemGenerator", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.x1v(i);
        const Real y = coords.x2v(j);
        const Real z = coords.x3v(k);

        const Real qx = (std::tanh(-20.0 * x) * std::cos(M_PI * x) + 1.0) *
                        std::exp(-30.0 * y * y) * std::exp(-30.0 * z * z);
        const Real qy = (std::sin(M_PI * y) + 0.2) * std::exp(-30.0 * x * x) *
                        std::exp(-30.0 * z * z);
        const Real qz = (std::tanh(-20. * z) * std::cos(M_PI * z) + 0.5) *
                        std::exp(-30.0 * x * x) * std::exp(-30.0 * y * y);

        q(0, k, j, i) = qx;
        q(1, k, j, i) = qy;
        q(2, k, j, i) = qz;

        for (int n = 3; n < num_vars; n++) {
          q(n, k, j, i) = 1;
          if (std::abs(x) < 0.025 && std::abs(y) < 0.15 && std::abs(z) < 0.025)
            q(n, k, j, i) += 10.0;
        }
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = burgers_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace burgers_benchmark
