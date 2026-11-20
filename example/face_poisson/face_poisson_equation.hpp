//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// [This code was generated with the help of generative AI]
#ifndef EXAMPLE_FACE_POISSON_FACE_POISSON_EQUATION_HPP_
#define EXAMPLE_FACE_POISSON_FACE_POISSON_EQUATION_HPP_

#include <memory>
#include <string>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

namespace face_poisson {
using parthenon::Real;

// Implements A.x = x - alpha Laplacian(x) for a face centered vector field x, where the
// Laplacian is applied to each face component with the standard second order stencil on
// that component's (staggered) grid. On a uniform Cartesian grid this is the same as
// x + alpha (curl curl x - grad div x). Ghost faces must be up to date before calling
// Ax. Fine-coarse boundaries are treated by the ghost face prolongation and restriction
// alone, so the operator is not symmetric there. With dirichlet set, the faces on a
// physical boundary normal to it hold the boundary value, so A is the identity there.
template <class var_t>
class FacePoissonEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  Real alpha;
  bool dirichlet;

  FacePoissonEquation(parthenon::ParameterInput *pin, const std::string &label) {
    alpha = pin->GetOrAddReal(label, "alpha", 1.0);
    dirichlet = pin->GetOrAddString(label, "boundary", "periodic",
                                    {"periodic", "dirichlet"}) == "dirichlet";
  }

  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    return tl.AddTask(depends_on, &FacePoissonEquation::Apply, this, md_in, md_out);
  }

  parthenon::TaskStatus Apply(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                              std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using TE = parthenon::TopologicalElement;
    ApplyElement<TE::F1, false>(md_in, md_out);
    ApplyElement<TE::F2, false>(md_in, md_out);
    ApplyElement<TE::F3, false>(md_in, md_out);
    return parthenon::TaskStatus::complete;
  }

  // The diagonal is exact, since there are no flux corrections
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using TE = parthenon::TopologicalElement;
    ApplyElement<TE::F1, true>(md_diag, md_diag);
    ApplyElement<TE::F2, true>(md_diag, md_diag);
    ApplyElement<TE::F3, true>(md_diag, md_diag);
    return parthenon::TaskStatus::complete;
  }

  // Store A.in in out, or the diagonal of A in out if diagonal is true
  template <parthenon::TopologicalElement te, bool diagonal>
  void ApplyElement(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                    std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    const int ndim = md_in->GetMeshPointer()->ndim;
    IndexRange ib = md_in->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_in->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_in->GetBoundsK(IndexDomain::interior, te);

    static auto desc = parthenon::MakePackDescriptor<var_t>(md_in.get());
    auto pack_in = desc.GetPack(md_in.get());
    auto pack_out = desc.GetPack(md_out.get());
    const Real a = alpha;
    const bool fixed_boundary = dirichlet;
    // Offset of the neighbor across the faces normal to te, and the index of the first
    // and last of those faces in the block
    constexpr int dir = static_cast<int>(te) % 3;
    const IndexRange nb = dir == 0 ? ib : (dir == 1 ? jb : kb);
    parthenon::par_for(
        "FacePoissonAx", 0, pack_in.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          if (fixed_boundary) {
            const int n = dir == 0 ? i : (dir == 1 ? j : k);
            const int off = (n == nb.s) ? -1 : ((n == nb.e) ? 1 : 0);
            if (off != 0 &&
                pack_in.IsPhysicalBoundary(b, (dir == 2) * off, (dir == 1) * off,
                                           (dir == 0) * off)) {
              pack_out(b, te, var_t(), k, j, i) =
                  diagonal ? 1.0 : pack_in(b, te, var_t(), k, j, i);
              return;
            }
          }
          const auto &coords = pack_in.GetCoordinates(b);
          const Real idx1 = 1.0 / coords.template Dxc<X1DIR>(k, j, i);
          const Real idx2 = 1.0 / coords.template Dxc<X2DIR>(k, j, i);
          const Real idx3 = 1.0 / coords.template Dxc<X3DIR>(k, j, i);
          if constexpr (diagonal) {
            Real d = 2.0 * idx1 * idx1;
            if (ndim > 1) d += 2.0 * idx2 * idx2;
            if (ndim > 2) d += 2.0 * idx3 * idx3;
            pack_out(b, te, var_t(), k, j, i) = 1.0 + a * d;
          } else {
            auto u = [&](const int kk, const int jj, const int ii) {
              return pack_in(b, te, var_t(), kk, jj, ii);
            };
            const Real u0 = u(k, j, i);
            Real lap = (u(k, j, i + 1) - 2.0 * u0 + u(k, j, i - 1)) * idx1 * idx1;
            if (ndim > 1)
              lap += (u(k, j + 1, i) - 2.0 * u0 + u(k, j - 1, i)) * idx2 * idx2;
            if (ndim > 2)
              lap += (u(k + 1, j, i) - 2.0 * u0 + u(k - 1, j, i)) * idx3 * idx3;
            pack_out(b, te, var_t(), k, j, i) = u0 - a * lap;
          }
        });
  }
};

} // namespace face_poisson

#endif // EXAMPLE_FACE_POISSON_FACE_POISSON_EQUATION_HPP_
