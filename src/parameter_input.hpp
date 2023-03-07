//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "defs.hpp"
#include "outputs/io_wrapper.hpp"
#include "utils/string_utils.hpp"

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
  int GetInteger(const std::string &block, const std::string &name);
  int GetOrAddInteger(const std::string &block, const std::string &name, int value);
  int SetInteger(const std::string &block, const std::string &name, int value);
  Real GetReal(const std::string &block, const std::string &name);
  Real GetOrAddReal(const std::string &block, const std::string &name, Real value);
  Real SetReal(const std::string &block, const std::string &name, Real value);
  bool GetBoolean(const std::string &block, const std::string &name);
  bool GetOrAddBoolean(const std::string &block, const std::string &name, bool value);
  bool SetBoolean(const std::string &block, const std::string &name, bool value);

  std::string GetString(const std::string &block, const std::string &name);
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const std::string &value);
  std::string SetString(const std::string &block, const std::string &name,
                        const std::string &value);
  std::string GetString(const std::string &block, const std::string &name,
                        const std::vector<std::string> &allowed_values);
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const std::string &value,
                             const std::vector<std::string> &allowed_values);
  void RollbackNextTime();
  void ForwardNextTime(Real time);
  void CheckRequired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &block, const std::string &name);

  template <typename T>
  std::vector<T> GetVector(const std::string &block, const std::string &name) {
    std::vector<std::string> fields = GetVector_(block, name);
    if constexpr (std::is_same<T, std::string>::value) return fields;

    std::vector<T> ret;
    for (auto &f : fields) {
      if constexpr (std::is_same<T, int>::value) {
        ret.push_back(stoi(f));
      } else if constexpr (std::is_same<T, Real>::value) {
        ret.push_back(atof(f.c_str()));
      } else if constexpr (std::is_same<T, bool>::value) {
        ret.push_back(stob(f));
      }
    }
    return ret;
  }
  template <typename T>
  std::vector<T> GetOrAddVector(const std::string &block, const std::string &name,
                                std::vector<T> def) {
    if (DoesParameterExist(block, name)) return GetVector<T>(block, name);

    std::string cname = ConcatVector_(def);
    auto *pb = FindOrAddBlock(block);
    AddParameter(pb, name, cname, "# Default value added at run time");
    return def;
  }

 private:
  std::string last_filename_; // last input file opened, to prevent duplicate reads

  InputBlock *FindOrAddBlock(const std::string &name);
  InputBlock *GetPtrToBlock(const std::string &name);
  bool ParseLine(InputBlock *pib, std::string line, std::string &name, std::string &value,
                 std::string &comment);
  void AddParameter(InputBlock *pib, const std::string &name, const std::string &value,
                    const std::string &comment);
  bool stob(std::string val) {
    // check is string contains integers 0 or 1 (instead of true or false) and return
    if (val.compare(0, 1, "0") == 0 || val.compare(0, 1, "1") == 0) {
      return static_cast<bool>(stoi(val));
    }

    // convert string to all lower case
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    // Convert string to bool and return value
    bool b;
    std::istringstream is(val);
    is >> std::boolalpha >> b;
    return b;
  }
  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAllowedValues_(const std::string &block, const std::string &name,
                           const T &val, Container_t<T, extra...> allowed) {
    bool found = std::any_of(allowed.begin(), allowed.end(),
                             [&](const T &t) { return (t == val); });
    if (!found) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ParameterInput::Get*]\n"
          << "Parameter '" << name << "/" << block
          << "' must be one of the following values:\n";
      for (const auto &v : allowed) {
        msg << v << " ";
      }
      msg << std::endl;
      PARTHENON_THROW(msg);
    }
  }
  std::vector<std::string> GetVector_(const std::string &block, const std::string &name) {
    std::string s = GetString(block, name);
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    std::vector<std::string> variables;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      variables.push_back(string_utils::trim(token));
      s.erase(0, pos + delimiter.length());
    }
    variables.push_back(string_utils::trim(s));
    return variables;
  }
  template <typename T>
  std::string ConcatVector_(std::vector<T> &vec) {
    std::stringstream ss;
    const int n = vec.size();
    if (n == 0) return "";

    ss << vec[0];
    for (int i = 1; i < n; i++) {
      ss << "," << vec[i];
    }
    return ss.str();
  }
};
} // namespace parthenon
#endif // PARAMETER_INPUT_HPP_
