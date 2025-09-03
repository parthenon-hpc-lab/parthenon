//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include "utils/error_checking.hpp"

#include "globals.hpp"
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
  for (const auto &[key, pparam] : myParams_) {
    if (std::type_index(pparam->type()) == std::type_index(typeid(T))) {
      auto typed_ptr = std::any_cast<T>(pparam.get());
      HDF5::HDF5WriteAttribute(prefix + "/" + key, *typed_ptr, group);
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
  for (auto &[key, pparam] : myParams_) {
    if (std::type_index(pparam->type()) == std::type_index(typeid(T)) &&
        GetMutability(key) == Mutability::Restart) {
      auto typed_ptr = std::any_cast<T>(pparam.get());
      auto &val = *typed_ptr;
      std::string fullpath = prefix + "/" + key;
      try {
        HDF5::HDF5ReadAttribute(group, fullpath, val);
        Update(key, val);
      } catch (std::runtime_error e) {
        // TODO(JMM/PG) Add failed load list of "fail/needs fix" list
        if (Globals::my_rank == 0) {
          std::stringstream ss;
          ss << "Failed to load parameter " << fullpath
             << " from the restart file! Using default value." << std::endl;
          PARTHENON_WARN(ss);
        }
      }
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
  WriteToHDF5AllParamsOfTypeOrVec<uint8_t>(prefix, group);

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
  ReadFromHDF5AllParamsOfTypeOrVec<uint8_t>(prefix, group);

  // strings
  ReadFromHDF5AllParamsOfType<std::string>(prefix, group);
  ReadFromHDF5AllParamsOfType<std::vector<std::string>>(prefix, group);
  ReadFromHDF5AllParamsOfType<std::vector<char>>(prefix, group);
}

#endif // ifdef ENABLE_HDF5

} // namespace parthenon
