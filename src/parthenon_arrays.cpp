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

#include "parthenon_arrays.hpp"

namespace Kokkos {

template class View<parthenon::Real*>;
template class View<parthenon::Real**>;
template class View<parthenon::Real***>;
template class View<parthenon::Real****>;
template class View<parthenon::Real*****>;
template class View<parthenon::Real******>;
template class View<int******>;

}

namespace parthenon {

template class ParArrayNDGeneric<Kokkos::View<Real******>>;
template class ParArrayNDGeneric<Kokkos::View<int******>>;

template class ParArrayNDGeneric<device_view_t<Real, LayoutWrapper>>;
template class ParArrayNDGeneric<device_view_t<int, LayoutWrapper>>;

template struct FaceArray<Real>;
template struct EdgeArray<Real>;

}
