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
#ifndef INTERFACE_STAGE_HPP_
#define INTERFACE_STAGE_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// forward declaration of needed classes and structs
struct FaceVariable;
struct EdgeVariable;
template <typename T> class Container;
template <typename T> class Variable;
template <typename T> class MaterialVariable;

///
/// The stage class provides a single struct that can be replaced to
/// change all registered variables to new storage.  The Container
/// class will hold a map of stages that can be referred to by name
/// and manipulated as required.
template <typename T>
class Stage {
 public:
  /// A new stage with name
  Stage<T>(std::string name) :
      _name(name), locked(!(name.compare(std::string("base")))) {
    //    std::cout << "_________________CREATE stage: " << _name << std::endl;
  }

  /// A new stage named 'name' initialized from stage src
  Stage<T>(std::string name, Stage<T>& src);

  const bool locked;   // set at initialization.  Only stage named
  // "base" is unlocked
  const std::string name() { return _name;}

  // the variable vectors
  std::vector<std::shared_ptr<Variable<T>>> _varArray = {}; ///< the saved variable array
  ///  std::vector<FaceVariable*> _faceArray = {};  ///< the saved face arrays
  ///  std::vector<EdgeVariable*> _edgeArray = {};  ///< the saved face arrays
  MaterialVariable<T> _matVars;

  // debug destructor
  //  ~Stage() {
  //    std::cout << "_______________________________Stage DESTROY: "<<_name<<std::endl;
  //    _varArray.clear();
  //    std::cout << "varArray destroyed" << std::endl;
  //  }
 private:
  std::string _name; ///< the stage name

  // we want container to be able to access our innards
  friend class Container<T>;
};

#endif // INTERFACE_STAGE_HPP_
