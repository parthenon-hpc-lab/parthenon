//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#include <math.h>
#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "diffusion_package.hpp"
#include "pybind/field_init.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace diffusion_example {

void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = md->GetMeshPointer()->ndim;

  Real x0 = pin->GetOrAddReal("diffusion", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("diffusion", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("diffusion", "z0", 0.0);
  const Real t0 = pin->GetOrAddReal("diffusion", "t0", 0.001);
  const Real dt = pin->GetOrAddReal("diffusion", "dt", 1.0);
  const bool constant_coeff =
      pin->GetOrAddBoolean("diffusion", "constant_coefficient", true);

#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS
  // Check if Python initialization is requested
  const bool use_python_init =
      pin->DoesParameterExist("diffusion/python_init", "u_function") &&
      pin->DoesParameterExist("diffusion/python_init", "u_file");
#else
  const bool use_python_init = false;
#endif

  // Initialize field u - either from Python or with C++ default
  if (use_python_init) {
#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS
    // Initialize each block using Python function
    for (int b = 0; b < md->NumBlocks(); ++b) {
      auto pmb_b = md->GetBlockData(b)->GetBlockPointer();
      parthenon::InitializeFieldFromPython(pmb_b, "diffusion.u", pin,
                                           "diffusion/python_init", "u_function",
                                           "u_file", {});
    }
#endif
  } else {
    // Use C++ initialization (original code)
    auto desc_u = parthenon::MakePackDescriptor<diffusion_package::u>(md);
    auto pack_u = desc_u.GetPack(md);
    auto &cellbounds = pmb->cellbounds;
    auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
    auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
    auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for(
        "Diffusion::ProblemGenerator::u", 0, pack_u.GetNBlocks() - 1, kb.s, kb.e, jb.s,
        jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_u.GetCoordinates(b);
          Real x1 = coords.Xc<1>(i);
          Real x2 = coords.Xc<2>(j);
          Real x3 = coords.Xc<3>(k);
          Real rad = (x1 - x0) * (x1 - x0);
          if (ndim > 1) rad += (x2 - y0) * (x2 - y0);
          if (ndim > 2) rad += (x3 - z0) * (x3 - z0);
          rad = std::sqrt(rad);
          Real D = 1.0;
          Real exponent = -rad * rad / (4.0 * D * t0);
          pack_u(b, diffusion_package::u(), k, j, i) = std::exp(exponent);
        });
  }

  // Initialize diffusion coefficient D (always done in C++)
  auto desc_D = parthenon::MakePackDescriptor<diffusion_package::D>(md);
  auto pack_D = desc_D.GetPack(md);
  using TE = parthenon::TopologicalElement;
  auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "Diffusion::ProblemGenerator::D", 0, pack_D.GetNBlocks() - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack_D.GetCoordinates(b);
        Real x1 = coords.Xc<1>(i);
        Real x2 = coords.Xc<2>(j);
        Real x3 = coords.Xc<3>(k);
        Real x1f = coords.X<1, TE::F1>(k, j, i);
        Real x2f = coords.X<2, TE::F2>(k, j, i);
        Real x3f = coords.X<3, TE::F3>(k, j, i);
        if (constant_coeff) {
          pack_D(b, TE::F1, diffusion_package::D(), k, j, i) = 1.0 * dt;
          pack_D(b, TE::F2, diffusion_package::D(), k, j, i) = 1.0 * dt;
          pack_D(b, TE::F3, diffusion_package::D(), k, j, i) = 1.0 * dt;
        } else {
          auto profile = [=](Real x, Real y, Real z) {
            Real r2 = (x - x0) * (x - x0);
            if (ndim > 1) r2 += (y - y0) * (y - y0);
            if (ndim > 2) r2 += (z - z0) * (z - z0);
            Real D = 1.0;
            return std::exp(-r2 / (4.0 * D * t0));
          };
          pack_D(b, TE::F1, diffusion_package::D(), k, j, i) = profile(x1f, x2, x3) * dt;
          pack_D(b, TE::F2, diffusion_package::D(), k, j, i) = profile(x1, x2f, x3) * dt;
          pack_D(b, TE::F3, diffusion_package::D(), k, j, i) = profile(x1, x2, x3f) * dt;
        }
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = diffusion_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace diffusion_example
