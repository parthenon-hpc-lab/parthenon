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
//=======================================================================================

#include "variable_pack.hpp"
#include "variable.hpp"

namespace parthenon {

template class VariablePack<Real>;
template class VariableFluxPack<Real>;
template void FillVarView<Real>(const vpack_types::VarList<Real> &vars,
                                PackIndexMap *vmap, ViewOfParArrays<Real> &cv,
                                ParArray1D<int> &sparse_assoc,
                                ParArray1D<int> &vector_component, bool coarse);
template void FillFluxViews(const vpack_types::VarList<Real> &vars, PackIndexMap *vmap,
                            const int ndim, ViewOfParArrays<Real> &f1,
                            ViewOfParArrays<Real> &f2, ViewOfParArrays<Real> &f3);
template VariableFluxPack<Real> MakeFluxPack(const vpack_types::VarList<Real> &vars,
                                             const vpack_types::VarList<Real> &flux_vars,
                                             PackIndexMap *vmap);
template VariablePack<Real> MakePack(const vpack_types::VarList<Real> &vars,
                                     PackIndexMap *vmap, bool coarse);

} // namespace parthenon
