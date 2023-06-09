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
  WriteToHDF5AllParamsOfMultipleTypes<
      T, std::vector<T>, ParArray0D<T>, ParArray1D<T>, ParArray2D<T>, ParArray3D<T>,
      ParArray4D<T>, ParArray5D<T>, ParArray6D<T>, ParArray7D<T>, ParArray8D<T>,
      HostArray0D<T>, HostArray1D<T>, HostArray2D<T>, HostArray3D<T>, HostArray4D<T>,
      HostArray5D<T>, HostArray6D<T>, HostArray7D<T>, Kokkos::View<T *>,
      Kokkos::View<T **>>(prefix, group);
}

template <typename... Ts>
void Params::WriteToHDF5AllParamsOfMultipleTypesOrVec(const std::string &prefix,
                                                      const HDF5::H5G &group) const {
  ([&] { WriteToHDF5AllParamsOfTypeOrVec<Ts>(prefix, group); }(), ...);
}

void Params::WriteAllToHDF5(const std::string &prefix, const HDF5::H5G &group) const {
  // views and vecs of scalar types
  WriteToHDF5AllParamsOfMultipleTypesOrVec<bool, int32_t, int64_t, uint32_t, uint64_t,
                                           float, double>(prefix, group);
  // strings
  WriteToHDF5AllParamsOfType<std::string>(prefix, group);
  WriteToHDF5AllParamsOfType<std::vector<std::string>>(prefix, group);
}

#endif // ifdef ENABLE_HDF5

} // namespace parthenon
