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
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

#include <algorithm>
#include <any>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility> // for std::forward, std::pair
#include <vector>

#include "config.hpp"
#include "defs.hpp"
#include "outputs/io_wrapper.hpp"
#include "utils/hash.hpp"
#include "utils/string_utils.hpp"

namespace parthenon {

struct QueryRecord {
  bool has_been_overwritten;
  std::string param_type;
  std::any default_value;               // std::any::has_value to check if default
                                        // val exists
  std::vector<std::any> allowed_values; // size to check if allowed values exist
  std::optional<std::string> docstring; // std::optiona::has_value to check if exists
  // JMM: Surely there's a way of doing this automatically?
  // Unfortunately the value in typeid is implementation defined, so
  // we can't pick if it looks nice...
  template <typename T>
  static std::string GetTypeName() {
    if constexpr (std::is_same_v<T, int>) {
      return "int";
    } else if constexpr (std::is_same_v<T, Real>) {
      return "Real";
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
      return "uint64_t";
    } else if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else if constexpr (std::is_same_v<T, std::string>) {
      return "string";
    } else if constexpr (std::is_arithmetic_v<T>) {
      if (Globals::my_rank == 0) {
        PARTHENON_WARN("Unknown arithmetic type! Attempting to use typeid, which is "
                       "implementation defined.");
      }
      T t;
      return typeid(t).name();
    } else if constexpr (std::is_same_v<T, std::vector<typename T::value_type>>) {
      return "std::vector<" + GetTypeName < GetTypeName() + ">";
    } else {
      if (Globals::my_rank == 0) {
        PARTHENON_WARN("Unknown non-arithmetic type! Attempting to use typeid, which is "
                       "implementation defined.");
      }
      T t;
      return typeid(t).name();
    }
  }
  template <typename T>
  static std::string GetTypeName(const T &t) {
    return GetTypeName<T>();
  }
  template <typename T>
  void SetTypeName() {
    param_type = GetTypeName<T>();
  }
};

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
  friend class std::hash<ParameterInput>;

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
  int GetInteger(const std::string &block, const std::string &name,
                 const std::optional<std::string> &docstring = std::optional<std::string>{});
  int GetOrAddInteger(const std::string &block, const std::string &name, int value,
                      const std::optional<std::string> &docstring = std::optional<std::string>{});
  int SetInteger(const std::string &block, const std::string &name, int value);
  Real GetReal(const std::string &block, const std::string &name,
               const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real GetOrAddReal(const std::string &block, const std::string &name, Real value,
                    const std::optional<std::string> &docstring = std::optional<std::string>{});
  Real SetReal(const std::string &block, const std::string &name, Real value);
  bool GetBoolean(const std::string &block, const std::string &name,
                  const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool GetOrAddBoolean(const std::string &block, const std::string &name, bool value,
                       const std::optional<std::string> &docstring = std::optional<std::string>{});
  bool SetBoolean(const std::string &block, const std::string &name, bool value);

  std::string GetString(const std::string &block, const std::string &name,
                        const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const std::string &value,
                             const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string SetString(const std::string &block, const std::string &name,
                        const std::string &value);
  std::string GetString(const std::string &block, const std::string &name,
                        const std::vector<std::string> &allowed_values,
                        const std::optional<std::string> &docstring = std::optional<std::string>{});
  std::string GetOrAddString(const std::string &block, const std::string &name,
                             const std::string &value,
                             const std::vector<std::string> &allowed_values,
                             const std::optional<std::string> &docstring = std::optional<std::string>{});
  void CheckRequired(const std::string &block, const std::string &name);
  void CheckDesired(const std::string &block, const std::string &name);
  void CheckOrphans();

  template <typename T, typename... Args>
  T GetOrAdd(const std::string &block, const std::string &name, const T &value,
             Args &&...args) {
    // JMM: This is slightly dangerous but helps with the pain point
    // Adam mentioned. Will be resolved with a more flexible parser
    // such as TOML.
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, std::size_t> ||
                  std::is_same_v<T, std::uint64_t>) {
      return GetOrAddInteger(block, name, value, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, Real>) {
      return GetOrAddReal(block, name, value, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, bool>) {
      return GetOrAddBoolean(block, name, value, std::forward<Args>(args)...);
    } else {
      PARTHENON_THROW("Unknown type\n");
    }
  }
  template <typename T, typename... Args>
  T Get(const std::string &block, const std::string &name, Args &&...args) {
    if constexpr (std::is_same_v<T, int>) {
      return GetInteger(block, name, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, Real>) {
      return GetReal(block, name, std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, bool>) {
      return GetOrAddBoolean(block, name, std::forward<Args>(args)...);
    } else {
      PARTHENON_THROW("Unknown type\n");
    }
  }

  template <typename T>
  std::vector<T> GetVector(const std::string &block, const std::string &name,
                           const std::optional<std::string> &docstring = std::optional<std::string>{}) {
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
                                std::vector<T> def,
                                const std::optional<std::string> &docstring = std::optional<std::string>{}) {
    if (DoesParameterExist(block, name)) return GetVector<T>(block, name);

    std::string cname = ConcatVector_(def);
    auto *pb = FindOrAddBlock(block);
    AddParameter(pb, name, cname, "# Default value added at run time");
    return def;
  }

 private:
  std::string last_filename_; // last input file opened, to prevent duplicate reads
  std::unordered_map<std::pair<std::string, std::string>, QueryRecord> queries_;

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

  // JMM: Using std::optional here aggressively to simplify overload
  // and default parameter logic logic
  template <typename T, template <class...> class Container_t, class... extra>
  void CheckAndUpdateQueries_(const std::string &block, const std::string &name,
                              const std::optional<T> &defval,
                              const Container_t<T, extra...> &allowed_vals,
                              const std::optional<std::string> &docstring) {
    auto key = std::make_pair(block, name);
    if (queries_.count(key) > 0) {
      QueryRecord &record = queries_.at(key);
      if (defval.has_value()) {
        // JMM: Forbid setting a default value after requesting but
        // allow requesting without a default if a default has
        // already been set.  I know this is unpleasantly stateful,
        // but we do this in a few places in the code.
        if (!record.default_value.has_value()) {
          std::stringstream msg;
          msg << "Input parameter " << block << "/" << name
              << " called previously without a default value and now called with one."
              << " If a default value is used, the first call must always set one."
              << std::endl;
          PARTHENON_THROW(msg);
        } else if (defval.value() != std::any_cast<T>(record.default_value)) {
          std::stringstream msg;
          msg << "Input parameter " << block << "/" << name
              << " has at least two inconsistent default values. "
              << "The ones I detected are " << defval.value() << " and "
              << std::any_cast<T>(record.default_value) << std::endl;
          PARTHENON_THROW(msg);
        }
      }
      // Only triggers if the container is non-empty
      PARTHENON_REQUIRE_THROWS(allowed_vals.size() == record.allowed_values.size(),
                               "Allowed values must be consistently shaped");
      std::size_t i = 0;
      for (auto &allowed : allowed_vals) {
        PARTHENON_REQUIRE_THROWS(allowed == std::any_cast<T>(record.allowed_values[i]),
                                 "Allowed values must be consistent");
      }
      // if two inconsistent docstrings exist, complain
      if (record.docstring.has_value() && docstring.has_value() &&
          (record.docstring.value() != docstring.value())) {
        std::stringstream msg;
        msg << "Input parameter " << block << "/" << name
            << " has inconsistent docstrings. The strings are:\n"
            << record.docstring.value() << "\nand\n"
            << docstring.value() << std::endl;
        PARTHENON_THROW(msg);
      } else if (docstring.has_value()) {
        // if the new query contains a docstring but the record does
        // not, add the docstring
        record.docstring = docstring; // record is a reference
      }
      // if the record contains a docstring but the new query does
      // not, do nothing
      // if neither contains a docstring, do nothing
    } else {
      QueryRecord record;
      record.SetTypeName<T>();
      record.default_value = defval; // might be empty
      for (const auto &allowed : allowed_vals) {
        record.allowed_values.push_back(std::any(allowed));
      }
      record.docstring = docstring; // might be empty
      queries_[key] = record;
    }
  }
  template <typename T>
  void CheckAndUpdateQueries_(const std::string &block, const std::string &name,
                              std::optional<std::string> &docstring) {
    CheckAndUpdateQueries(block, name, std::optional<T>{}, std::vector<T>{},
                          docstring);
  }
};
} // namespace parthenon

// JMM: Believe it or not, this is the recommended way to overload hash functions
// See: https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <>
struct hash<parthenon::InputLine> {
  std::size_t operator()(const parthenon::InputLine &il) {
    return parthenon::impl::hash_combine(0, il.param_name, il.param_value,
                                         il.param_comment);
  }
};

template <>
struct hash<parthenon::InputBlock> {
  std::size_t operator()(const parthenon::InputBlock &ib) {
    using parthenon::impl::hash_combine;
    std::size_t out =
        hash_combine(0, ib.block_name, ib.max_len_parname, ib.max_len_parvalue);
    for (parthenon::InputLine *pline = ib.pline; pline != nullptr; pline = pline->pnext) {
      out = hash_combine(out, *pline);
    }
    return out;
  }
};

template <>
struct hash<parthenon::ParameterInput> {
  std::size_t operator()(const parthenon::ParameterInput &in) {
    using parthenon::InputBlock;
    using parthenon::impl::hash_combine;
    std::size_t out = 0;
    out = hash_combine(out, in.last_filename_);
    for (InputBlock *pblock = in.pfirst_block; pblock != nullptr;
         pblock = pblock->pnext) {
      out = hash_combine(out, *pblock);
    }
    return out;
  }
};
} // namespace std

#endif // PARAMETER_INPUT_HPP_
