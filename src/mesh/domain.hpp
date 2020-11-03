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
//
enum class IndexDomain { entire, interior };

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

  void MakeZeroDimensional_(int index) {
    x_[index] = IndexRange{0, 0};
    entire_ncells_[index] = 1;
  }

 public:
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

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsI(const IndexDomain &domain) const
      noexcept {
    return (domain == IndexDomain::entire) ? IndexRange{0, entire_ncells_[0] - 1} : x_[0];
  }

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsJ(const IndexDomain &domain) const
      noexcept {
    return (domain == IndexDomain::entire) ? IndexRange{0, entire_ncells_[1] - 1} : x_[1];
  }

  KOKKOS_INLINE_FUNCTION const IndexRange GetBoundsK(const IndexDomain &domain) const
      noexcept {
    return (domain == IndexDomain::entire) ? IndexRange{0, entire_ncells_[2] - 1} : x_[2];
  }

  KOKKOS_INLINE_FUNCTION int is(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? 0 : x_[0].s;
  }

  KOKKOS_INLINE_FUNCTION int js(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? 0 : x_[1].s;
  }

  KOKKOS_INLINE_FUNCTION int ks(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? 0 : x_[2].s;
  }

  KOKKOS_INLINE_FUNCTION int ie(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[0] - 1 : x_[0].e;
  }

  KOKKOS_INLINE_FUNCTION int je(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[1] - 1 : x_[1].e;
  }

  KOKKOS_INLINE_FUNCTION int ke(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[2] - 1 : x_[2].e;
  }

  KOKKOS_INLINE_FUNCTION int ncellsi(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[0] : x_[0].e - x_[0].s + 1;
  }

  KOKKOS_INLINE_FUNCTION int ncellsj(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[1] : x_[1].e - x_[1].s + 1;
  }

  KOKKOS_INLINE_FUNCTION int ncellsk(const IndexDomain &domain) const noexcept {
    return (domain == IndexDomain::entire) ? entire_ncells_[2] : x_[2].e - x_[2].s + 1;
  }

  // Kept basic for kokkos
  KOKKOS_INLINE_FUNCTION
  int GetTotal(const IndexDomain &domain) const noexcept {
    if (NDIM == 0) return 0;
    int total = 1;
    if (domain == IndexDomain::entire) {
      for (int i = 0; i < NDIM; ++i)
        total *= entire_ncells_[i];
    } else {
      for (int i = 0; i < NDIM; ++i)
        total *= x_[i].e - x_[i].s + 1;
    }
    return total;
  }
};

} // namespace parthenon

#endif // MESH_DOMAIN_HPP_
