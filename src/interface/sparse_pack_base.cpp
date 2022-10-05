////========================================================================================
//// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
////
//// This program was produced under U.S. Government contract 89233218CNA000001 for Los
//// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
//// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
//// in the program are reserved by Triad National Security, LLC, and the U.S. Department
//// of Energy/National Nuclear Security Administration. The Government is granted for
//// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
//// license in this material to reproduce, prepare derivative works, distribute copies to
//// the public, perform publicly and display publicly, and to permit others to do so.
////========================================================================================
//
//#include <algorithm>
//#include <functional>
//#include <limits>
//#include <map>
//#include <memory>
//#include <regex>
//#include <string>
//#include <type_traits>
//#include <utility>
//#include <vector>
//
//#include "coordinates/coordinates.hpp"
//#include "interface/mesh_data.hpp"
//#include "interface/meshblock_data.hpp"
//#include "interface/sparse_pack_base.hpp"
//#include "interface/variable.hpp"
//#include "utils/utils.hpp"
//
//namespace {
//// SFINAE for block iteration so that sparse packs can work for MeshBlockData and MeshData
//template <class T, class F>
//inline auto ForEachBlock(T *pmd, F func) -> decltype(T().GetBlockData(0), void()) {
//  for (int b = 0; b < pmd->NumBlocks(); ++b) {
//    auto &pmbd = pmd->GetBlockData(b);
//    func(b, pmbd.get());
//  }
//}
//
//template <class T, class F>
//inline auto ForEachBlock(T *pmbd, F func) -> decltype(T().GetBlockPointer(), void()) {
//  func(0, pmbd);
//}
//} // namespace
//
//namespace parthenon {
//
//std::string SparsePackCache::GetIdentifier(const PackDescriptor &desc) const {
//  std::string identifier("");
//  for (const auto &flag : desc.flags)
//    identifier += flag.Name();
//  identifier += "____";
//  for (int i = 0; i < desc.vars.size(); ++i)
//    identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
//  return identifier;
//}
//
//} // namespace parthenon
