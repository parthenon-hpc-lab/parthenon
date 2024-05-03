//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include <string>

#include "utils/error_checking.hpp"

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

#ifdef ENABLE_HDF5
#include "outputs/parthenon_hdf5.hpp"
#endif

#include "params.hpp"

namespace parthenon {

#ifdef ENABLE_HDF5

template <typename T>
void Params::WriteToHDF5AllParamsOfType(const std::string &prefix,
                                        const HDF5::H5G &group) const {
  for (const auto &p : myParams_) {
    const auto &key = p.first;
    const auto type = myTypes_.at(key);
    if (type == std::type_index(typeid(T))) {
      auto typed_ptr = dynamic_cast<Params::object_t<T> *>((p.second).get());
      HDF5::HDF5WriteAttribute(prefix + "/" + key, *typed_ptr->pValue, group);
    }
  }
}

template <typename... Ts>
void Params::WriteToHDF5AllParamsOfMultipleTypes(const std::string &prefix,
                                                 const HDF5::H5G &group) const {
  ([&] { WriteToHDF5AllParamsOfType<Ts>(prefix, group); }(), ...);
}

template <typename T>
void Params::WriteToHDF5AllParamsOfTypeOrVec(const std::string &prefix,
                                             const HDF5::H5G &group) const {
  WriteToHDF5AllParamsOfMultipleTypes<PARTHENON_ATTR_VALID_VEC_TYPES(T)>(prefix, group);
}

template <typename T>
void Params::ReadFromHDF5AllParamsOfType(const std::string &prefix,
                                         const HDF5::H5G &group) {
  for (auto &p : myParams_) {
    auto &key = p.first;
    auto type = myTypes_.at(key);
    auto mutability = myMutable_.at(key);
    if (type == std::type_index(typeid(T)) && mutability == Mutability::Restart) {
      auto typed_ptr = dynamic_cast<Params::object_t<T> *>((p.second).get());
      auto &val = *(typed_ptr->pValue);
      HDF5::HDF5ReadAttribute(group, prefix + "/" + key, val);
      Update(key, val);
    }
  }
}

template <typename... Ts>
void Params::ReadFromHDF5AllParamsOfMultipleTypes(const std::string &prefix,
                                                  const HDF5::H5G &group) {
  ([&] { ReadFromHDF5AllParamsOfType<Ts>(prefix, group); }(), ...);
}

template <typename T>
void Params::ReadFromHDF5AllParamsOfTypeOrVec(const std::string &prefix,
                                              const HDF5::H5G &group) {
  ReadFromHDF5AllParamsOfMultipleTypes<PARTHENON_ATTR_VALID_VEC_TYPES(T)>(prefix, group);
}

void Params::WriteAllToHDF5(const std::string &prefix, const HDF5::H5G &group) const {
  // views and vecs of scalar types
  WriteToHDF5AllParamsOfTypeOrVec<bool>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<int32_t>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<int64_t>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<uint32_t>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<uint64_t>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<float>(prefix, group);
  WriteToHDF5AllParamsOfTypeOrVec<double>(prefix, group);

  // strings
  WriteToHDF5AllParamsOfType<std::string>(prefix, group);
  WriteToHDF5AllParamsOfType<std::vector<std::string>>(prefix, group);
  WriteToHDF5AllParamsOfType<std::vector<char>>(prefix, group);
}

void Params::ReadFromRestart(const std::string &prefix, const HDF5::H5G &group) {
  // views and vecs of scalar types
  ReadFromHDF5AllParamsOfTypeOrVec<bool>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<int32_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<int64_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<uint32_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<uint64_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<float>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<double>(prefix, group);

  // strings
  ReadFromHDF5AllParamsOfType<std::string>(prefix, group);
  ReadFromHDF5AllParamsOfType<std::vector<std::string>>(prefix, group);
  ReadFromHDF5AllParamsOfType<std::vector<char>>(prefix, group);
}

#endif // ifdef ENABLE_HDF5

} // namespace parthenon
