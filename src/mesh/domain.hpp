//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_DOMAIN_HPP_
#define MESH_DOMAIN_HPP_

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#include "defs.hpp"

namespace parthenon {

struct IndexRange {
  int s = 0; /// Starting Index (inclusive)
  int e = 0; /// Ending Index (inclusive)
};

// Assuming we have a block
//
//  - - - - - - - - - -   ^
//  |  |  ghost    |  |   |
//  - - - - - - - - - -   |
//  |  |     ^     |  |   |
//  |  |     |     |  |
//  |  | interior  |  |   entire
//  |  |     |     |  |
//  |  |     v     |  |   |
//  - - - - - - - - - -   |
//  |  |           |  |   |
//  - - - - - - - - - -   v

// inner/outer_x* are ranges for boundaries
// and cover the ghost zones on each side of the
// meshblock
// as per this diagram
//
//  - - - - - - - - -  --
//  |     outer_x2      |
//  - i- - - - - - - -o -
//  | n |     ^     | u |
//  | n |     |     | t |
//  | e | interior  | e |
//  | r |     |     | r |
//  | x |     v     | x |
//  - 1 - - - - - - - 1 -
//  |    inner_x2       |
//  - - - - - - - - -  --
// with inner and outer x3 the slabs
// containing the ghost zones above
// and below the interior
//
enum class IndexDomain {
  entire,
  interior,
  inner_x1,
  outer_x1,
  inner_x2,
  outer_x2,
  inner_x3,
  outer_x3
};

//! \class IndexVolume
//  \brief Defines the dimensions of a shape of indices
//
//  Defines the range of each dimension of the indices by defining a starting and stopping
//  index also contains a label for defining which region the index shape is assigned too
class IndexShape {
 private:
  std::array<IndexRange, NDIM> x_;
  std::array<int, NDIM> entire_ncells_;

  bool DimensionProvided_(const std::vector<int> &interior_dims, int dim) {
    return dim <= interior_dims.size();
  }

  KOKKOS_INLINE_FUNCTION
  void MakeZeroDimensional_(int const index) {
    x_[index] = IndexRange{0, 0};
    entire_ncells_[index] = 1;
  }

 public:
  KOKKOS_FUNCTION
  IndexShape() {}

  IndexShape(const int &nx3, const int &nx2, const int &nx1, const int &ng)
      : IndexShape(std::vector<int>{nx3, nx2, nx1}, ng) {}

  IndexShape(const int &nx2, const int &nx1, const int &ng)
      : IndexShape(std::vector<int>{nx2, nx1}, ng) {}

  IndexShape(const int &nx1, const int &ng) : IndexShape(std::vector<int>{nx1}, ng) {}

  //----------------------------------------------------------------------------------------
  //! \fn IndexShape::IndexShape(const std::vector<int> & interior_dims, const int &ng)
  //  \brief Builds IndexShape using interior_dims, note that the interior dims must
  //  be specified in the order:
  //
  //  interior_dims.at(0) = nx3
  //  interior_dims.at(1) = nx2
  //  interior_dims.at(2) = nx1
  //
  IndexShape(std::vector<int> interior_dims, const int &ng) {
    std::reverse(interior_dims.begin(), interior_dims.end());
    assert(interior_dims.size() <= NDIM &&
           "IndexShape cannot be initialized, the number of "
           "dimensions exceeds the statically set dimensions, you will need to change "
           "the NDIM "
           "constant.");
    for (int dim = 1, index = 0; dim <= NDIM; ++dim, ++index) {
      if (DimensionProvided_(interior_dims, dim) == false) {
        MakeZeroDimensional_(index);
      } else {
        assert(interior_dims.at(index) > -1 &&
               "IndexShape cannot be initialized with a negative number of "
               "interior cells for any dimension");
        if (interior_dims.at(index) == 0) {
          MakeZeroDimensional_(index);
        } else {
          x_[index] = IndexRange{ng, (ng + interior_dims.at(index) - 1)};
          entire_ncells_[index] = interior_dims.at(index) + 2 * ng;
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  static IndexShape GetOnDevice(int nx3, int nx2, int nx1, int ng) {
    IndexShape out;
    int idx = 0;
    for (auto nx : {nx1, nx2, nx3}) {
      if (nx == 0) {
        out.x_[idx] = IndexRange{0, 0};
        out.entire_ncells_[idx] = 1;
      } else {
        out.x_[idx] = IndexRange{ng, (ng + nx - 1)};
        out.entire_ncells_[idx] = nx + 2 * ng;
      }
      idx++;
    }
    return out;
  }

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsI(const IndexDomain &domain,
                                                     TE el = TE::CC) const noexcept {
    return (domain == IndexDomain::interior && el == TE::CC)
               ? x_[0]
               : IndexRange{is(domain, el), ie(domain, el)};
  }

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsJ(const IndexDomain &domain,
                                                     TE el = TE::CC) const noexcept {
    return (domain == IndexDomain::interior && el == TE::CC)
               ? x_[1]
               : IndexRange{js(domain, el), je(domain, el)};
  }

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsK(const IndexDomain &domain,
                                                     TE el = TE::CC) const noexcept {
    return (domain == IndexDomain::interior && el == TE::CC)
               ? x_[2]
               : IndexRange{ks(domain, el), ke(domain, el)};
  }

  KOKKOS_INLINE_FUNCTION int is(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return x_[0].s;
    case IndexDomain::outer_x1:
      return entire_ncells_[0] == 1 ? 0 : x_[0].e + 1 + TopologicalOffsetI(el);
    default:
      return 0;
    }
  }

  KOKKOS_INLINE_FUNCTION int js(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return x_[1].s;
    case IndexDomain::outer_x2:
      return entire_ncells_[1] == 1 ? 0 : x_[1].e + 1 + TopologicalOffsetJ(el);
    default:
      return 0;
    }
  }

  KOKKOS_INLINE_FUNCTION int ks(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return x_[2].s;
    case IndexDomain::outer_x3:
      return entire_ncells_[2] == 1 ? 0 : x_[2].e + 1 + TopologicalOffsetK(el);
    default:
      return 0;
    }
  }

  KOKKOS_INLINE_FUNCTION int ie(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return entire_ncells_[0] == 1 ? 0 : x_[0].e + TopologicalOffsetI(el);
    case IndexDomain::inner_x1:
      return x_[0].s == 0 ? 0 : x_[0].s - 1;
    default:
      return entire_ncells_[0] == 1 ? 0 : entire_ncells_[0] - 1 + TopologicalOffsetI(el);
    }
  }

  KOKKOS_INLINE_FUNCTION int je(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return entire_ncells_[1] == 1 ? 0 : x_[1].e + TopologicalOffsetJ(el);
    case IndexDomain::inner_x2:
      return x_[1].s == 0 ? 0 : x_[1].s - 1;
    default:
      return entire_ncells_[1] == 1 ? 0 : entire_ncells_[1] - 1 + TopologicalOffsetJ(el);
    }
  }

  KOKKOS_INLINE_FUNCTION int ke(const IndexDomain &domain,
                                TE el = TE::CC) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return entire_ncells_[2] == 1 ? 0 : x_[2].e + TopologicalOffsetK(el);
    case IndexDomain::inner_x3:
      return x_[2].s == 0 ? 0 : x_[2].s - 1;
    default:
      return entire_ncells_[2] == 1 ? 0 : entire_ncells_[2] - 1 + TopologicalOffsetK(el);
    }
  }

  KOKKOS_INLINE_FUNCTION int ncellsi(const IndexDomain &domain,
                                     TE el = TE::CC) const noexcept {
    if (entire_ncells_[0] == 1 &&
        (domain == IndexDomain::inner_x1 || domain == IndexDomain::outer_x1)) {
      return 0; // if x1 is zero-dimensional, there are no ghost zones
    }
    return ie(domain, el) - is(domain, el) + 1;
  }

  KOKKOS_INLINE_FUNCTION int ncellsj(const IndexDomain &domain,
                                     TE el = TE::CC) const noexcept {
    if (entire_ncells_[1] == 1 &&
        (domain == IndexDomain::inner_x2 || domain == IndexDomain::outer_x2)) {
      return 0; // if x2 is zero-dimensional, there are no ghost zones
    }
    return je(domain, el) - js(domain, el) + 1;
  }

  KOKKOS_INLINE_FUNCTION int ncellsk(const IndexDomain &domain,
                                     TE el = TE::CC) const noexcept {
    if (entire_ncells_[2] == 1 &&
        (domain == IndexDomain::inner_x3 || domain == IndexDomain::outer_x3)) {
      return 0; // if x3 is zero-dimensional, there are no ghost zones
    }
    return ke(domain, el) - ks(domain, el) + 1;
  }

  // Kept basic for kokkos
  KOKKOS_INLINE_FUNCTION
  int GetTotal(const IndexDomain &domain, TE el = TE::CC) const noexcept {
    if (NDIM == 0) return 0;
    return ncellsi(domain, el) * ncellsj(domain, el) * ncellsk(domain, el);
  }
};

} // namespace parthenon

#endif // MESH_DOMAIN_HPP_
