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
#ifndef PARTHENON_INTERFACE_PropertiesInterface_HPP_
#define PARTHENON_INTERFACE_PropertiesInterface_HPP_

#include <map>
#include <string>

#include "interface/StateDescriptor.hpp"

class PropertiesInterface {
public:
  virtual ~PropertiesInterface() {}

  virtual StateDescriptor &State() = 0;

  static int GetIDFromLabel(std::string &label) {
    return PropertiesInterface::_label_to_id[label];
  }

  static std::string GetLabelFromID(int id) {
    for (auto &x : PropertiesInterface::_label_to_id) {
      if (x.second == id)
        return x.first;
    }
    return "UNKNOWN";
  }

  static void InsertID(const std::string &label, const int &id) {
    PropertiesInterface::_label_to_id[label] = id;
  }

private:
  // _label_to_id is declared here and defined in
  // PropertiesInterface.cpp
  static std::map<std::string, int> _label_to_id;
};

/*template <typename T>
auto ConvertPropertiesToInterface(
    const std::vector<std::shared_ptr<T>> &materials) {
  static_assert(std::is_base_of<PropertiesInterface, T>::value,
                "Type given to ConvertPropertiesToInterface is not "
                "derived from PropertiesInterface");
  std::vector<std::shared_ptr<PropertiesInterface>> res;
  for (auto mat : materials)
    res.push_back(mat);

  return res;
}
*/
#endif // PARTHENON_INTERFACE_PropertiesInterface_HPP_