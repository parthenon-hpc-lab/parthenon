//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>

#include "config.hpp"
#include "defs.hpp"
#include "outputs/io_wrapper.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \struct InputLine
//  \brief  node in a singly linked list of parameters contained within 1x input block

struct InputLine {
  std::string param_name;
  std::string param_value; // value of the parameter is stored as a string!
  std::string param_comment;
  InputLine *pnext; // pointer to the next node in this nested singly linked list
};

//----------------------------------------------------------------------------------------
//! \class InputBlock
//  \brief node in a singly linked list of all input blocks contained within input file

class InputBlock {
 public:
  InputBlock() = default;
  ~InputBlock();

  // data
  std::string block_name;
  std::size_t max_len_parname;  // length of longest param_name, for nice-looking output
  std::size_t max_len_parvalue; // length of longest param_value, to format outputs
  InputBlock *pnext; // pointer to the next node in InputBlock singly linked list

  InputLine *pline; // pointer to head node in nested singly linked list (in this block)
  // (not storing a reference to the tail node)

  // functions
  InputLine *GetPtrToLine(std::string name);
  const InputLine *GetPtrToLine(std::string name) const;
};

//----------------------------------------------------------------------------------------
//! \class ParameterInput
//  \brief data and definitions of functions used to store and access input parameters
//  Functions are implemented in parameter_input.cpp

class ParameterInput {
 public:
  // constructor/destructor
  ParameterInput();
  explicit ParameterInput(std::string input_filename);
  ~ParameterInput();

  // data
  InputBlock *pfirst_block; // pointer to head node in singly linked list of InputBlock
  // (not storing a reference to the tail node)

  // functions
  void LoadFromStream(std::istream &is);
  void LoadFromFile(IOWrapper &input);
  void ModifyFromCmdline(int argc, char *argv[]);
  void ParameterDump(std::ostream &os);
  int DoesParameterExist(const std::string &block, const std::string &name);
  int DoesBlockExist(const std::string &block);
  std::string GetComment(const std::string &block, const std::string &name);
//  int GetInteger(const std::string &block, const std::string &name);
  int GetOrAddInteger(const std::string &block, const std::string &name, int value);
  int SetInteger(const std::string &block, const std::string &name, int value);
//  Real GetReal(const std::string &block, const std::string &name);
  Real GetOrAddReal(const std::string &block, const std::string &name, Real value);
  Real GetOrAddPrecise(const std::string &block, const std::string &name, Real value);
  Real SetReal(const std::string &block, const std::string &name, Real value);
  Real SetPrecise(const std::string &block, const std::string &name, Real value);
//  bool GetBoolean(const std::string &block, const std::string &name);
  bool GetOrAddBoolean(const std::string &block, const std::string &name, bool value);
  bool SetBoolean(const std::string &block, const std::string &name, bool value);
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const std::string &value);
  std::string SetString(const std::string &block, const std::string &name,
                        const std::string &value);
  void RollbackNextTime();
  void ForwardNextTime(Real time);
  void CheckRequired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &block, const std::string &name);

  template<class T>
  T Get(const std::string &block, const std::string &name);
  template<class T>
  const T & Get(const std::string &block, const std::string &name) const;

 private:
  std::string last_filename_; // last input file opened, to prevent duplicate reads

  InputBlock *FindOrAddBlock(const std::string &name);
  InputBlock *GetPtrToBlock(const std::string &name);
  const InputBlock *GetPtrToBlock(const std::string &name) const;

  bool ParseLine(InputBlock *pib, std::string line, std::string &name, std::string &value,
                 std::string &comment);
  void AddParameter(InputBlock *pib, const std::string &name, const std::string &value,
                    const std::string &comment);

  std::stringstream GetParameter(const std::string &block, const std::string &name) const;
  // thread safety
#ifdef OPENMP_PARALLEL
  mutable omp_lock_t lock_;
#endif

  void Lock() const;
  void Unlock() const;
};

  inline std::stringstream ParameterInput::GetParameter(const std::string &block, const std::string &name) const {
    const InputBlock *pb;
    const InputLine *pl;
    std::stringstream msg;

    Lock();

    // get pointer to node with same block name in singly linked list of InputBlocks
    pb = GetPtrToBlock(block);
    if (pb == nullptr) {
      msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
      PARTHENON_FAIL(msg);
    }

    // get pointer to node with same parameter name in singly linked list of InputLines
    pl = pb->GetPtrToLine(name);
    if (pl == nullptr) {
      msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
      PARTHENON_FAIL(msg);
    }

    std::stringstream stream(pl->param_value);
    Unlock();

    return stream;
  }

  template<class T>
  inline T ParameterInput::Get(const std::string &block, const std::string &name) {
    std::stringstream stream = GetParameter(block,name);
    T val2;
    stream >> val2;
    /*if(std::is_same<T,bool>::value){
      if (val.compare(0, 1, "0") == 0 || val.compare(0, 1, "1") == 0) {
        return static_cast<T>(stoi(val));
      }

      // convert string to all lower case
      std::transform(val.begin(), val.end(), val.begin(), ::tolower);
      // Convert string to bool and return value
      std::istringstream is(val);
      is >> std::boolalpha >> val2;
     }else if(std::is_same<T,std::string>::value ){
      val2 = static_cast<T>(val);
     } else if(std::is_same<T,Real>::value ) {
      val2 = static_cast<T>(std::atof(val.c_str()));
     } else if(std::is_same<T,int>::value ) {
      val2 = static_cast<T>(stoi(val));
     }*/
    // return value
    return val2;
  }

  template<>
  inline bool ParameterInput::Get<bool>(const std::string &block, const std::string &name) {
    std::stringstream stream = GetParameter(block,name);

    if (stream.str().compare(0, 1, "0") == 0 || stream.str().compare(0, 1, "1") == 0) {
      return static_cast<bool>(std::stoi(stream.str()));
    }

    std::string val = stream.str();
    //   convert string to all lower case
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    // Convert string to bool and return value
    bool b;
    std::istringstream is(val);
    is >> std::boolalpha >> b;
    return (b);
  }

  template<class T>
  inline const T & ParameterInput::Get(const std::string &block, const std::string &name) const {
    return this->Get<T>(block,name);
  }

} // namespace parthenon
#endif // PARAMETER_INPUT_HPP_
